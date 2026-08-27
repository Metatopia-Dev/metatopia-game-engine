//! Multiplayer Game Client
//!
//! Connects to server, sends sampled player inputs, tracks ping/RTT,
//! buffers incoming world snapshots, and executes client-side prediction.

use std::net::SocketAddr;
use std::io;
use super::protocol::{ClientMessage, ServerMessage, ChannelType, PROTOCOL_VERSION};
use super::transport::UdpTransport;
use super::replication::{Snapshot, SnapshotBuffer, ClientPrediction};

/// State of client connection to server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
}

/// Multiplayer Game Client
pub struct NetClient {
    pub transport: UdpTransport,
    pub server_addr: Option<SocketAddr>,
    pub state: ConnectionState,
    pub assigned_id: Option<u32>,
    pub tick_rate: u32,
    pub rtt_ms: f32,
    pub snapshot_buffer: SnapshotBuffer,
    pub prediction: ClientPrediction,
    pub input_seq: u32,
    last_ping_sent_ms: u64,
}

impl NetClient {
    /// Create a new network client bound to an ephemeral port
    pub fn new() -> io::Result<Self> {
        let transport = UdpTransport::bind("0.0.0.0:0")?;
        Ok(Self {
            transport,
            server_addr: None,
            state: ConnectionState::Disconnected,
            assigned_id: None,
            tick_rate: 60,
            rtt_ms: 0.0,
            snapshot_buffer: SnapshotBuffer::default(),
            prediction: ClientPrediction::new(),
            input_seq: 0,
            last_ping_sent_ms: 0,
        })
    }

    /// Initiate connection to a game server
    pub fn connect(&mut self, server_addr: &str, player_name: &str) -> io::Result<()> {
        let addr: SocketAddr = server_addr.parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        self.server_addr = Some(addr);
        self.state = ConnectionState::Connecting;

        let connect_msg = ClientMessage::Connect {
            name: player_name.into(),
            protocol_version: PROTOCOL_VERSION,
        };

        self.transport.send(addr, ChannelType::Reliable, &connect_msg.encode())?;
        Ok(())
    }

    /// Send current player input state to the server
    pub fn send_input(&mut self, keys_mask: u32, yaw: f32, pitch: f32, pos: [f32; 3], chart: u32) {
        if let Some(server) = self.server_addr {
            self.input_seq = self.input_seq.wrapping_add(1);
            let msg = ClientMessage::InputState {
                seq: self.input_seq,
                keys_mask,
                yaw,
                pitch,
                pos,
                chart,
            };
            let _ = self.transport.send(server, ChannelType::Unreliable, &msg.encode());
        }
    }

    /// Send player action (e.g. fire weapon, interact) to the server
    pub fn send_action(&mut self, action_id: u8, target_id: u32, data: [f32; 3]) {
        if let Some(server) = self.server_addr {
            let msg = ClientMessage::Action { action_id, target_id, data };
            let _ = self.transport.send(server, ChannelType::Reliable, &msg.encode());
        }
    }

    /// Process incoming server messages and update ping/RTT
    pub fn poll(&mut self, now_ms: u64) -> Vec<ServerMessage> {
        let mut messages = Vec::new();

        // Send periodic ping every 1 second
        if self.state == ConnectionState::Connected && now_ms.saturating_sub(self.last_ping_sent_ms) >= 1000 {
            self.last_ping_sent_ms = now_ms;
            if let Some(server) = self.server_addr {
                let ping = ClientMessage::Ping { client_time_ms: now_ms };
                let _ = self.transport.send(server, ChannelType::Unreliable, &ping.encode());
            }
        }

        while let Ok(Some((packet, sender))) = self.transport.recv() {
            if Some(sender) != self.server_addr {
                continue;
            }

            if let Ok(msg) = ServerMessage::decode(&packet.payload) {
                match &msg {
                    ServerMessage::Welcome { assigned_id, tick_rate } => {
                        self.assigned_id = Some(*assigned_id);
                        self.tick_rate = *tick_rate;
                        self.state = ConnectionState::Connected;
                        messages.push(msg);
                    }
                    ServerMessage::WorldSnapshot { tick, timestamp_ms, entities } => {
                        let entity_map = entities.iter().map(|e| (e.id, e.clone())).collect();
                        self.snapshot_buffer.push(Snapshot {
                            tick: *tick,
                            timestamp_ms: *timestamp_ms,
                            entities: entity_map,
                        });
                        messages.push(msg);
                    }
                    ServerMessage::Pong { client_time_ms, .. } => {
                        let sample_rtt = (now_ms.saturating_sub(*client_time_ms)) as f32;
                        if self.rtt_ms <= 0.0 {
                            self.rtt_ms = sample_rtt;
                        } else {
                            // Exponential moving average
                            self.rtt_ms = self.rtt_ms * 0.85 + sample_rtt * 0.15;
                        }
                    }
                    _ => {
                        messages.push(msg);
                    }
                }
            }
        }

        messages
    }

    /// Disconnect from the server
    pub fn disconnect(&mut self) {
        if let Some(server) = self.server_addr {
            let msg = ClientMessage::Disconnect;
            let _ = self.transport.send(server, ChannelType::Reliable, &msg.encode());
        }
        self.state = ConnectionState::Disconnected;
        self.server_addr = None;
        self.assigned_id = None;
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_init_and_disconnect() {
        let mut client = NetClient::new().unwrap();
        assert_eq!(client.state, ConnectionState::Disconnected);

        client.connect("127.0.0.1:7777", "PlayerX").unwrap();
        assert_eq!(client.state, ConnectionState::Connecting);

        client.disconnect();
        assert_eq!(client.state, ConnectionState::Disconnected);
    }
}
