//! Entity Replication, State Interpolation and Client-Side Prediction
//!
//! Provides snapshot buffering, Hermite/Linear entity interpolation between server ticks,
//! and client-side input prediction with server authoritative reconciliation.

use std::collections::{HashMap, VecDeque};
use super::protocol::EntityState;

/// Timestamped server world snapshot
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub tick: u32,
    pub timestamp_ms: u64,
    pub entities: HashMap<u32, EntityState>,
}

/// Circular buffer storing recent snapshots for smooth rendering interpolation
#[derive(Debug, Clone)]
pub struct SnapshotBuffer {
    snapshots: VecDeque<Snapshot>,
    pub interpolation_delay_ms: u64,
    pub max_history: usize,
}

impl Default for SnapshotBuffer {
    fn default() -> Self {
        Self::new(100, 32)
    }
}

impl SnapshotBuffer {
    pub fn new(interpolation_delay_ms: u64, max_history: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(max_history),
            interpolation_delay_ms,
            max_history,
        }
    }

    /// Add a new server snapshot to the buffer
    pub fn push(&mut self, snapshot: Snapshot) {
        if self.snapshots.len() >= self.max_history {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Interpolate all entities at a given render timestamp (typically `current_time - delay`)
    pub fn sample(&self, render_time_ms: u64) -> Option<HashMap<u32, EntityState>> {
        if self.snapshots.is_empty() { return None; }
        if self.snapshots.len() == 1 {
            return Some(self.snapshots.front().unwrap().entities.clone());
        }

        // Find two enclosing snapshots
        let mut older_idx = None;
        let mut newer_idx = None;

        for i in 0..(self.snapshots.len() - 1) {
            let s0 = &self.snapshots[i];
            let s1 = &self.snapshots[i + 1];

            if render_time_ms >= s0.timestamp_ms && render_time_ms <= s1.timestamp_ms {
                older_idx = Some(i);
                newer_idx = Some(i + 1);
                break;
            }
        }

        if let (Some(i0), Some(i1)) = (older_idx, newer_idx) {
            let s0 = &self.snapshots[i0];
            let s1 = &self.snapshots[i1];

            let duration = (s1.timestamp_ms - s0.timestamp_ms).max(1) as f32;
            let elapsed = (render_time_ms - s0.timestamp_ms) as f32;
            let t = (elapsed / duration).clamp(0.0, 1.0);

            let mut interpolated = HashMap::new();
            for (id, e0) in &s0.entities {
                if let Some(e1) = s1.entities.get(id) {
                    let pos = [
                        e0.pos[0] + (e1.pos[0] - e0.pos[0]) * t,
                        e0.pos[1] + (e1.pos[1] - e0.pos[1]) * t,
                        e0.pos[2] + (e1.pos[2] - e0.pos[2]) * t,
                    ];
                    let vel = [
                        e0.vel[0] + (e1.vel[0] - e0.vel[0]) * t,
                        e0.vel[1] + (e1.vel[1] - e0.vel[1]) * t,
                        e0.vel[2] + (e1.vel[2] - e0.vel[2]) * t,
                    ];
                    let yaw = e0.yaw + (e1.yaw - e0.yaw) * t;
                    let pitch = e0.pitch + (e1.pitch - e0.pitch) * t;
                    let health = e0.health + (e1.health - e0.health) * t;

                    interpolated.insert(*id, EntityState {
                        id: *id,
                        owner_id: e1.owner_id,
                        pos,
                        vel,
                        yaw,
                        pitch,
                        chart: e1.chart,
                        health,
                    });
                } else {
                    interpolated.insert(*id, e0.clone());
                }
            }

            Some(interpolated)
        } else {
            // Extrapolate or clamp to latest
            self.snapshots.back().map(|s| s.entities.clone())
        }
    }
}

/// Client-Side Prediction and Server Reconciliation
#[derive(Debug, Clone)]
pub struct SavedInput {
    pub seq: u32,
    pub keys_mask: u32,
    pub dt: f32,
    pub predicted_pos: [f32; 3],
}

#[derive(Debug, Clone, Default)]
pub struct ClientPrediction {
    pub unacknowledged_inputs: VecDeque<SavedInput>,
    pub authoritative_pos: [f32; 3],
    pub last_acknowledged_seq: u32,
}

impl ClientPrediction {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sent input for later reconciliation
    pub fn record_input(&mut self, seq: u32, keys_mask: u32, dt: f32, predicted_pos: [f32; 3]) {
        self.unacknowledged_inputs.push_back(SavedInput {
            seq,
            keys_mask,
            dt,
            predicted_pos,
        });
    }

    /// Reconcile client prediction against authoritative server state
    pub fn reconcile<F>(&mut self, server_ack_seq: u32, server_pos: [f32; 3], mut replay_step: F) -> [f32; 3]
    where
        F: FnMut([f32; 3], u32, f32) -> [f32; 3],
    {
        self.last_acknowledged_seq = server_ack_seq;
        self.authoritative_pos = server_pos;

        // Discard inputs up to server_ack_seq
        self.unacknowledged_inputs.retain(|inp| inp.seq > server_ack_seq);

        // Replay remaining unacknowledged inputs on top of server authoritative position
        let mut current_pos = server_pos;
        for input in &mut self.unacknowledged_inputs {
            current_pos = replay_step(current_pos, input.keys_mask, input.dt);
            input.predicted_pos = current_pos;
        }

        current_pos
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_linear_interpolation() {
        let mut buffer = SnapshotBuffer::new(50, 10);

        let mut e0_map = HashMap::new();
        e0_map.insert(1, EntityState {
            id: 1,
            owner_id: 0,
            pos: [0.0, 0.0, 0.0],
            vel: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            chart: 0,
            health: 100.0,
        });
        buffer.push(Snapshot { tick: 1, timestamp_ms: 1000, entities: e0_map });

        let mut e1_map = HashMap::new();
        e1_map.insert(1, EntityState {
            id: 1,
            owner_id: 0,
            pos: [10.0, 20.0, 30.0],
            vel: [1.0, 2.0, 3.0],
            yaw: 1.0,
            pitch: 0.5,
            chart: 0,
            health: 50.0,
        });
        buffer.push(Snapshot { tick: 2, timestamp_ms: 2000, entities: e1_map });

        // Interpolate halfway at t = 1500
        let sampled = buffer.sample(1500).unwrap();
        let entity = sampled.get(&1).unwrap();

        assert!((entity.pos[0] - 5.0).abs() < 1e-4);
        assert!((entity.pos[1] - 10.0).abs() < 1e-4);
        assert!((entity.pos[2] - 15.0).abs() < 1e-4);
        assert!((entity.health - 75.0).abs() < 1e-4);
    }

    #[test]
    fn test_client_prediction_reconciliation() {
        let mut pred = ClientPrediction::new();
        pred.record_input(1, 1, 0.1, [1.0, 0.0, 0.0]);
        pred.record_input(2, 1, 0.1, [2.0, 0.0, 0.0]);
        pred.record_input(3, 1, 0.1, [3.0, 0.0, 0.0]);

        // Server says at seq 2, player was actually at (1.9, 0, 0)
        let reconciled = pred.reconcile(2, [1.9, 0.0, 0.0], |pos, _keys, _dt| {
            [pos[0] + 1.0, pos[1], pos[2]]
        });

        assert_eq!(pred.unacknowledged_inputs.len(), 1); // Only seq 3 remains
        assert!((reconciled[0] - 2.9).abs() < 1e-4);
    }
}
