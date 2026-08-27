//! Authoritative Game Server
//!
//! Manages client sessions, input processing, non-Euclidean chart tracking,
//! tick-rate snapshot broadcasts, and automatic timeout disconnections.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::io;
use super::protocol::{ClientMessage, ServerMessage, EntityState, ChannelType, PROTOCOL_VERSION};
use super::transport::UdpTransport;

/// Active client session on the server
#[derive(Debug, Clone)]
pub struct ClientSession {
    pub id: u32,
    pub addr: SocketAddr,
    pub name: String,
    pub last_seen_ms: u64,
    pub last_input_seq: u32,
    pub chart: u32,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

/// Authoritative Multiplayer Game Server
pub struct NetServer {
    pub transport: UdpTransport,
    pub clients: HashMap<SocketAddr, ClientSession>,
    pub next_client_id: u32,
    pub tick_rate: u32,
    pub current_tick: u32,
    pub entities: HashMap<u32, EntityState>,
}

impl NetServer {
    /// Bind server to a local address (e.g. "0.0.0.0:7777")
    pub fn bind(addr: &str, tick_rate: u32) -> io::Result<Self> {
        let transport = UdpTransport::bind(addr)?;
        Ok(Self {
            transport,
            clients: HashMap::new(),
            next_client_id: 1,
            tick_rate,
            current_tick: 0,
            entities: HashMap::new(),
        })
    }

    /// Process incoming client packets (call every frame/tick)
    pub fn poll(&mut self, now_ms: u64) -> Vec<(u32, ClientMessage)> {
        let mut events = Vec::new();

        while let Ok(Some((packet, sender))) = self.transport.recv() {
            if let Ok(msg) = ClientMessage::decode(&packet.payload) {
                match &msg {
                    ClientMessage::Connect { name, protocol_version } => {
                        if *protocol_version == PROTOCOL_VERSION {
                            let client_id = self.next_client_id;
                            self.next_client_id += 1;

                            let session = ClientSession {
                                id: client_id,
                                addr: sender,
                                name: name.clone(),
                                last_seen_ms: now_ms,
                                last_input_seq: 0,
                                chart: 0,
                                pos: [0.0, 0.0, 0.0],
                                yaw: 0.0,
                                pitch: 0.0,
                            };
                            self.clients.insert(sender, session);

                            // Send Welcome packet
                            let welcome = ServerMessage::Welcome {
                                assigned_id: client_id,
                                tick_rate: self.tick_rate,
                            };
                            let _ = self.transport.send(sender, ChannelType::Reliable, &welcome.encode());
                            events.push((client_id, msg));
                        }
                    }
                    ClientMessage::InputState { seq, yaw, pitch, pos, chart, .. } => {
                        if let Some(client) = self.clients.get_mut(&sender) {
                            client.last_seen_ms = now_ms;
                            if *seq > client.last_input_seq {
                                client.last_input_seq = *seq;
                                client.pos = *pos;
                                client.yaw = *yaw;
                                client.pitch = *pitch;
                                client.chart = *chart;
                            }
                            events.push((client.id, msg));
                        }
                    }
                    ClientMessage::Ping { client_time_ms } => {
                        if let Some(client) = self.clients.get_mut(&sender) {
                            client.last_seen_ms = now_ms;
                            let pong = ServerMessage::Pong {
                                client_time_ms: *client_time_ms,
                                server_time_ms: now_ms,
                            };
                            let _ = self.transport.send(sender, ChannelType::Unreliable, &pong.encode());
                        }
                    }
                    ClientMessage::Disconnect => {
                        if let Some(client) = self.clients.remove(&sender) {
                            events.push((client.id, msg));
                        }
                    }
                    ClientMessage::Action { .. } => {
                        if let Some(client) = self.clients.get_mut(&sender) {
                            client.last_seen_ms = now_ms;
                            events.push((client.id, msg));
                        }
                    }
                }
            }
        }

        events
    }

    /// Broadcast the current world snapshot to all connected clients
    pub fn broadcast_snapshot(&mut self, now_ms: u64) {
        self.current_tick = self.current_tick.wrapping_add(1);

        let snapshot = ServerMessage::WorldSnapshot {
            tick: self.current_tick,
            timestamp_ms: now_ms,
            entities: self.entities.values().cloned().collect(),
        };

        let encoded = snapshot.encode();
        for client in self.clients.values() {
            let _ = self.transport.send(client.addr, ChannelType::Unreliable, &encoded);
        }
    }

    /// Broadcast a server event (hitmarker, portal traversal, etc.) to all clients
    pub fn broadcast_event(&mut self, event: ServerMessage) {
        let encoded = event.encode();
        for client in self.clients.values() {
            let _ = self.transport.send(client.addr, ChannelType::Reliable, &encoded);
        }
    }

    /// Remove clients that have timed out
    pub fn cleanup_timeouts(&mut self, now_ms: u64, timeout_ms: u64) {
        self.clients.retain(|_, client| now_ms.saturating_sub(client.last_seen_ms) < timeout_ms);
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_client_handshake() {
        let mut server = NetServer::bind("127.0.0.1:0", 60).unwrap();
        let mut client_transport = UdpTransport::bind("127.0.0.1:0").unwrap();

        let s_addr = server.transport.socket.local_addr().unwrap();

        // Client connects
        let connect_msg = ClientMessage::Connect {
            name: "Player1".into(),
            protocol_version: PROTOCOL_VERSION,
        };
        client_transport.send(s_addr, ChannelType::Reliable, &connect_msg.encode()).unwrap();

        // Server polls
        std::thread::sleep(std::time::Duration::from_millis(5));
        let events = server.poll(1000);

        assert_eq!(events.len(), 1);
        assert_eq!(server.clients.len(), 1);

        // Client receives Welcome
        std::thread::sleep(std::time::Duration::from_millis(5));
        if let Some((pkt, _)) = client_transport.recv().unwrap() {
            let welcome = ServerMessage::decode(&pkt.payload).unwrap();
            match welcome {
                ServerMessage::Welcome { assigned_id, tick_rate } => {
                    assert_eq!(assigned_id, 1);
                    assert_eq!(tick_rate, 60);
                }
                _ => panic!("Expected Welcome message"),
            }
        } else {
            panic!("Expected packet on client");
        }
    }
}
