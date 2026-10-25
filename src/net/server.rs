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

    /// Authoritative server-side cross-portal geodesic weapon raycast:
    /// Traces a curved geodesic ray across up to `max_portal_bounces` portals (`ManifoldRaycaster::cast_geodesic_ray`),
    /// checks every segment against all other entities in the matching `ChartId`, applies `damage` to the closest
    /// hit entity, and broadcasts a hit event (`ServerMessage::Event`) to all connected clients.
    pub fn resolve_cross_portal_shot(
        &mut self,
        manifold: &crate::manifold::Manifold,
        shooter_id: u32,
        origin: cgmath::Point3<f32>,
        direction: cgmath::Vector3<f32>,
        start_chart: crate::manifold::ChartId,
        max_distance: f32,
        max_portal_bounces: usize,
        damage: f32,
        hit_radius: f32,
    ) -> Option<CrossPortalHitResult> {
        use cgmath::InnerSpace;
        let trace = crate::manifold::ManifoldRaycaster::cast_geodesic_ray(
            manifold,
            origin,
            direction,
            start_chart,
            max_distance,
            0.20,
            max_portal_bounces,
        );

        let r2_thresh = hit_radius * hit_radius;
        let mut accumulated_dist = 0.0f32;
        let mut portals_crossed = 0usize;
        let mut prev_chart = start_chart;

        for seg in &trace.segments {
            if seg.chart != prev_chart {
                portals_crossed += 1;
                prev_chart = seg.chart;
            }
            let ab = seg.end - seg.start;
            let ab_len2 = ab.magnitude2();

            let mut best_target: Option<(u32, f32, cgmath::Point3<f32>)> = None;
            for (&entity_id, entity) in &self.entities {
                if entity_id == shooter_id || entity.health <= 0.0 || entity.chart != seg.chart.0 {
                    continue;
                }
                let e_pos = cgmath::Point3::new(entity.pos[0], entity.pos[1], entity.pos[2]);
                let ap = e_pos - seg.start;
                let t = if ab_len2 > 1e-8 {
                    (ap.dot(ab) / ab_len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let closest = seg.start + ab * t;
                let d2 = (e_pos - closest).magnitude2();
                if d2 <= r2_thresh {
                    let hit_dist = accumulated_dist + seg.length * t;
                    if best_target.map_or(true, |(_, bd, _)| hit_dist < bd) {
                        best_target = Some((entity_id, hit_dist, closest));
                    }
                }
            }

            if let Some((target_id, total_distance, hit_point)) = best_target {
                let mut remaining_health = 0.0;
                if let Some(target) = self.entities.get_mut(&target_id) {
                    target.health = (target.health - damage).max(0.0);
                    remaining_health = target.health;
                }
                let hit_event = ServerMessage::Event {
                    event_type: 1, // Cross-portal weapon hit
                    source_id: shooter_id,
                    target_id,
                    value: damage,
                };
                self.broadcast_event(hit_event);
                return Some(CrossPortalHitResult {
                    shooter_id,
                    target_id,
                    hit_chart: seg.chart,
                    hit_point,
                    total_distance,
                    portals_crossed: portals_crossed.max(trace.portal_jumps),
                    damage_applied: damage,
                    target_remaining_health: remaining_health,
                });
            }

            accumulated_dist += seg.length;
        }

        None
    }
}

/// Authoritative server hit result for a cross-portal geodesic weapon shot
#[derive(Debug, Clone)]
pub struct CrossPortalHitResult {
    pub shooter_id: u32,
    pub target_id: u32,
    pub hit_chart: crate::manifold::ChartId,
    pub hit_point: cgmath::Point3<f32>,
    pub total_distance: f32,
    pub portals_crossed: usize,
    pub damage_applied: f32,
    pub target_remaining_health: f32,
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{ChartId, GeometryType, Manifold, PortalShape};
    use cgmath::{Matrix4, Point3, SquareMatrix, Vector3};

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

    #[test]
    fn test_authoritative_cross_portal_shot_hits_target_in_destination_chart() {
        let mut server = NetServer::bind("127.0.0.1:0", 60).unwrap();
        let mut manifold = Manifold::new(); // ChartId(0)
        let chart1 = manifold.add_chart(GeometryType::Euclidean); // ChartId(1)

        // Portal in Chart 0 at z = -3.0 transporting to Chart 1 at z = +3.0
        manifold
            .create_portal_with_bounds(
                ChartId(0),
                chart1,
                Point3::new(0.0, 1.5, -3.0),
                Point3::new(0.0, 1.5, 3.0),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.0,
                4.0,
                PortalShape::Rectangular,
            )
            .unwrap();

        // Shooter (Entity 1) in Chart 0 at (0, 1.5, 0)
        server.entities.insert(
            1,
            EntityState {
                id: 1,
                owner_id: 1,
                pos: [0.0, 1.5, 0.0],
                vel: [0.0, 0.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                chart: 0,
                health: 100.0,
            },
        );

        // Target (Entity 2) in Chart 1 at (0, 1.5, 1.0) — directly behind the portal exit!
        server.entities.insert(
            2,
            EntityState {
                id: 2,
                owner_id: 2,
                pos: [0.0, 1.5, 1.0],
                vel: [0.0, 0.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                chart: 1,
                health: 100.0,
            },
        );

        // Shooter fires along -Z into the portal
        let hit = server
            .resolve_cross_portal_shot(
                &manifold,
                1,
                Point3::new(0.0, 1.5, 0.0),
                Vector3::new(0.0, 0.0, -1.0),
                ChartId(0),
                12.0,
                4,
                35.0,
                0.5,
            )
            .expect("Expected cross-portal shot to hit Entity 2 in Chart 1");

        assert_eq!(hit.target_id, 2);
        assert_eq!(hit.hit_chart, ChartId(1));
        assert_eq!(hit.portals_crossed, 1);
        assert!((hit.target_remaining_health - 65.0).abs() < 1e-4);
    }
}
