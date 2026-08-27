//! Network Protocol and Binary Serialization
//!
//! Compact byte-level encoding for multiplayer client-server messages,
//! player inputs, entity states, portal events, and latency pings.

pub const PROTOCOL_MAGIC: u16 = 0x4D54; // "MT"
pub const PROTOCOL_VERSION: u32 = 1;

/// Network transmission channel reliability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Unreliable = 0,
    Reliable = 1,
}

/// Replicated physical state of an entity in the network world
#[derive(Debug, Clone, PartialEq)]
pub struct EntityState {
    pub id: u32,
    pub owner_id: u32,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub chart: u32,
    pub health: f32,
}

impl EntityState {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&self.owner_id.to_le_bytes());
        for p in self.pos { buf.extend_from_slice(&p.to_le_bytes()); }
        for v in self.vel { buf.extend_from_slice(&v.to_le_bytes()); }
        buf.extend_from_slice(&self.yaw.to_le_bytes());
        buf.extend_from_slice(&self.pitch.to_le_bytes());
        buf.extend_from_slice(&self.chart.to_le_bytes());
        buf.extend_from_slice(&self.health.to_le_bytes());
    }

    pub fn decode(slice: &[u8]) -> Result<(Self, &[u8]), String> {
        if slice.len() < 40 {
            return Err("EntityState buffer too short".into());
        }
        let id = u32::from_le_bytes(slice[0..4].try_into().unwrap());
        let owner_id = u32::from_le_bytes(slice[4..8].try_into().unwrap());
        let pos = [
            f32::from_le_bytes(slice[8..12].try_into().unwrap()),
            f32::from_le_bytes(slice[12..16].try_into().unwrap()),
            f32::from_le_bytes(slice[16..20].try_into().unwrap()),
        ];
        let vel = [
            f32::from_le_bytes(slice[20..24].try_into().unwrap()),
            f32::from_le_bytes(slice[24..28].try_into().unwrap()),
            f32::from_le_bytes(slice[28..32].try_into().unwrap()),
        ];
        let yaw = f32::from_le_bytes(slice[32..36].try_into().unwrap());
        let pitch = f32::from_le_bytes(slice[36..40].try_into().unwrap());
        let chart = if slice.len() >= 48 {
            u32::from_le_bytes(slice[40..44].try_into().unwrap())
        } else {
            0
        };
        let health = if slice.len() >= 48 {
            f32::from_le_bytes(slice[44..48].try_into().unwrap())
        } else {
            100.0
        };

        let remaining = if slice.len() >= 48 { &slice[48..] } else { &slice[40..] };

        Ok((Self { id, owner_id, pos, vel, yaw, pitch, chart, health }, remaining))
    }
}

/// Messages sent from a Client to the Server
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    Connect { name: String, protocol_version: u32 },
    InputState { seq: u32, keys_mask: u32, yaw: f32, pitch: f32, pos: [f32; 3], chart: u32 },
    Action { action_id: u8, target_id: u32, data: [f32; 3] },
    Ping { client_time_ms: u64 },
    Disconnect,
}

impl ClientMessage {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        match self {
            ClientMessage::Connect { name, protocol_version } => {
                buf.push(1); // Msg Type 1
                buf.extend_from_slice(&protocol_version.to_le_bytes());
                let bytes = name.as_bytes();
                buf.push(bytes.len() as u8);
                buf.extend_from_slice(bytes);
            }
            ClientMessage::InputState { seq, keys_mask, yaw, pitch, pos, chart } => {
                buf.push(2); // Msg Type 2
                buf.extend_from_slice(&seq.to_le_bytes());
                buf.extend_from_slice(&keys_mask.to_le_bytes());
                buf.extend_from_slice(&yaw.to_le_bytes());
                buf.extend_from_slice(&pitch.to_le_bytes());
                for p in pos { buf.extend_from_slice(&p.to_le_bytes()); }
                buf.extend_from_slice(&chart.to_le_bytes());
            }
            ClientMessage::Action { action_id, target_id, data } => {
                buf.push(3);
                buf.push(*action_id);
                buf.extend_from_slice(&target_id.to_le_bytes());
                for d in data { buf.extend_from_slice(&d.to_le_bytes()); }
            }
            ClientMessage::Ping { client_time_ms } => {
                buf.push(4);
                buf.extend_from_slice(&client_time_ms.to_le_bytes());
            }
            ClientMessage::Disconnect => {
                buf.push(5);
            }
        }
        buf
    }

    pub fn decode(slice: &[u8]) -> Result<Self, String> {
        if slice.is_empty() { return Err("Empty message".into()); }
        match slice[0] {
            1 => {
                if slice.len() < 6 { return Err("Connect message too short".into()); }
                let version = u32::from_le_bytes(slice[1..5].try_into().unwrap());
                let len = slice[5] as usize;
                if slice.len() < 6 + len { return Err("Invalid string length".into()); }
                let name = String::from_utf8_lossy(&slice[6..6+len]).into_owned();
                Ok(ClientMessage::Connect { name, protocol_version: version })
            }
            2 => {
                if slice.len() < 33 { return Err("InputState message too short".into()); }
                let seq = u32::from_le_bytes(slice[1..5].try_into().unwrap());
                let keys_mask = u32::from_le_bytes(slice[5..9].try_into().unwrap());
                let yaw = f32::from_le_bytes(slice[9..13].try_into().unwrap());
                let pitch = f32::from_le_bytes(slice[13..17].try_into().unwrap());
                let pos = [
                    f32::from_le_bytes(slice[17..21].try_into().unwrap()),
                    f32::from_le_bytes(slice[21..25].try_into().unwrap()),
                    f32::from_le_bytes(slice[25..29].try_into().unwrap()),
                ];
                let chart = if slice.len() >= 33 {
                    u32::from_le_bytes(slice[29..33].try_into().unwrap())
                } else {
                    0
                };
                Ok(ClientMessage::InputState { seq, keys_mask, yaw, pitch, pos, chart })
            }
            3 => {
                if slice.len() < 18 { return Err("Action message too short".into()); }
                let action_id = slice[1];
                let target_id = u32::from_le_bytes(slice[2..6].try_into().unwrap());
                let data = [
                    f32::from_le_bytes(slice[6..10].try_into().unwrap()),
                    f32::from_le_bytes(slice[10..14].try_into().unwrap()),
                    f32::from_le_bytes(slice[14..18].try_into().unwrap()),
                ];
                Ok(ClientMessage::Action { action_id, target_id, data })
            }
            4 => {
                if slice.len() < 9 { return Err("Ping message too short".into()); }
                let client_time_ms = u64::from_le_bytes(slice[1..9].try_into().unwrap());
                Ok(ClientMessage::Ping { client_time_ms })
            }
            5 => Ok(ClientMessage::Disconnect),
            tag => Err(format!("Unknown ClientMessage tag {}", tag)),
        }
    }
}

/// Messages sent from the Server to Clients
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMessage {
    Welcome { assigned_id: u32, tick_rate: u32 },
    WorldSnapshot { tick: u32, timestamp_ms: u64, entities: Vec<EntityState> },
    Event { event_type: u8, source_id: u32, target_id: u32, value: f32 },
    PortalTraversed { entity_id: u32, from_chart: u32, to_chart: u32 },
    Pong { client_time_ms: u64, server_time_ms: u64 },
}

impl ServerMessage {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        match self {
            ServerMessage::Welcome { assigned_id, tick_rate } => {
                buf.push(1);
                buf.extend_from_slice(&assigned_id.to_le_bytes());
                buf.extend_from_slice(&tick_rate.to_le_bytes());
            }
            ServerMessage::WorldSnapshot { tick, timestamp_ms, entities } => {
                buf.push(2);
                buf.extend_from_slice(&tick.to_le_bytes());
                buf.extend_from_slice(&timestamp_ms.to_le_bytes());
                buf.extend_from_slice(&(entities.len() as u16).to_le_bytes());
                for e in entities {
                    e.encode(&mut buf);
                }
            }
            ServerMessage::Event { event_type, source_id, target_id, value } => {
                buf.push(3);
                buf.push(*event_type);
                buf.extend_from_slice(&source_id.to_le_bytes());
                buf.extend_from_slice(&target_id.to_le_bytes());
                buf.extend_from_slice(&value.to_le_bytes());
            }
            ServerMessage::PortalTraversed { entity_id, from_chart, to_chart } => {
                buf.push(4);
                buf.extend_from_slice(&entity_id.to_le_bytes());
                buf.extend_from_slice(&from_chart.to_le_bytes());
                buf.extend_from_slice(&to_chart.to_le_bytes());
            }
            ServerMessage::Pong { client_time_ms, server_time_ms } => {
                buf.push(5);
                buf.extend_from_slice(&client_time_ms.to_le_bytes());
                buf.extend_from_slice(&server_time_ms.to_le_bytes());
            }
        }
        buf
    }

    pub fn decode(slice: &[u8]) -> Result<Self, String> {
        if slice.is_empty() { return Err("Empty server message".into()); }
        match slice[0] {
            1 => {
                if slice.len() < 9 { return Err("Welcome message too short".into()); }
                let assigned_id = u32::from_le_bytes(slice[1..5].try_into().unwrap());
                let tick_rate = u32::from_le_bytes(slice[5..9].try_into().unwrap());
                Ok(ServerMessage::Welcome { assigned_id, tick_rate })
            }
            2 => {
                if slice.len() < 15 { return Err("WorldSnapshot message too short".into()); }
                let tick = u32::from_le_bytes(slice[1..5].try_into().unwrap());
                let timestamp_ms = u64::from_le_bytes(slice[5..13].try_into().unwrap());
                let count = u16::from_le_bytes(slice[13..15].try_into().unwrap()) as usize;

                let mut entities = Vec::with_capacity(count);
                let mut ptr = &slice[15..];
                for _ in 0..count {
                    let (entity, rem) = EntityState::decode(ptr)?;
                    entities.push(entity);
                    ptr = rem;
                }
                Ok(ServerMessage::WorldSnapshot { tick, timestamp_ms, entities })
            }
            3 => {
                if slice.len() < 14 { return Err("Event message too short".into()); }
                let event_type = slice[1];
                let source_id = u32::from_le_bytes(slice[2..6].try_into().unwrap());
                let target_id = u32::from_le_bytes(slice[6..10].try_into().unwrap());
                let value = f32::from_le_bytes(slice[10..14].try_into().unwrap());
                Ok(ServerMessage::Event { event_type, source_id, target_id, value })
            }
            4 => {
                if slice.len() < 13 { return Err("PortalTraversed message too short".into()); }
                let entity_id = u32::from_le_bytes(slice[1..5].try_into().unwrap());
                let from_chart = u32::from_le_bytes(slice[5..9].try_into().unwrap());
                let to_chart = u32::from_le_bytes(slice[9..13].try_into().unwrap());
                Ok(ServerMessage::PortalTraversed { entity_id, from_chart, to_chart })
            }
            5 => {
                if slice.len() < 17 { return Err("Pong message too short".into()); }
                let client_time_ms = u64::from_le_bytes(slice[1..9].try_into().unwrap());
                let server_time_ms = u64::from_le_bytes(slice[9..17].try_into().unwrap());
                Ok(ServerMessage::Pong { client_time_ms, server_time_ms })
            }
            tag => Err(format!("Unknown ServerMessage tag {}", tag)),
        }
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_message_roundtrip() {
        let msg = ClientMessage::InputState {
            seq: 142,
            keys_mask: 0b1011,
            yaw: 1.57,
            pitch: -0.2,
            pos: [10.5, 2.0, -4.5],
            chart: 2,
        };

        let encoded = msg.encode();
        let decoded = ClientMessage::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_server_snapshot_roundtrip() {
        let entity = EntityState {
            id: 10,
            owner_id: 1,
            pos: [1.0, 2.0, 3.0],
            vel: [0.5, 0.0, -0.5],
            yaw: 0.8,
            pitch: 0.1,
            chart: 0,
            health: 85.0,
        };

        let msg = ServerMessage::WorldSnapshot {
            tick: 500,
            timestamp_ms: 123456789,
            entities: vec![entity],
        };

        let encoded = msg.encode();
        let decoded = ServerMessage::decode(&encoded).unwrap();
        assert_eq!(msg, decoded);
    }
}
