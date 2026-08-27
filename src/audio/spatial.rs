//! 3D Spatial Audio and non-Euclidean portal sound propagation
//!
//! Computes 3D audio attenuation (Linear, InverseSquare, Exponential),
//! ear-azimuth stereo panning, Doppler frequency shift, and portal-aware
//! acoustic propagation across manifold charts.

use cgmath::{Point3, Vector3, InnerSpace};
use crate::manifold::{Manifold, ChartId};

/// Distance attenuation model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttenuationModel {
    Linear { min_dist: f32, max_dist: f32 },
    InverseSquare { ref_dist: f32, roll_off: f32 },
    Exponential { decay_rate: f32 },
}

/// 3D audio listener (typically attached to the camera/player)
#[derive(Debug, Clone)]
pub struct AudioListener {
    pub pos: Point3<f32>,
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub chart: ChartId,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            pos: Point3::new(0.0, 1.7, 0.0),
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            chart: ChartId(0),
        }
    }
}

impl AudioListener {
    pub fn right(&self) -> Vector3<f32> {
        self.forward.cross(self.up).normalize()
    }
}

/// 3D positioned sound emitter source
#[derive(Debug, Clone)]
pub struct SpatialSoundSource {
    pub id: usize,
    pub pos: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub base_volume: f32,
    pub base_pitch: f32,
    pub attenuation: AttenuationModel,
    pub chart: ChartId,
    pub directional: Option<Vector3<f32>>,
}

impl SpatialSoundSource {
    pub fn new(id: usize, pos: Point3<f32>, chart: ChartId) -> Self {
        Self {
            id,
            pos,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            base_volume: 1.0,
            base_pitch: 1.0,
            attenuation: AttenuationModel::InverseSquare { ref_dist: 1.0, roll_off: 1.0 },
            chart,
            directional: None,
        }
    }
}

/// Computed spatialized audio parameters for left/right channels
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatializedAudioSample {
    pub left_volume: f32,
    pub right_volume: f32,
    pub pitch_shift: f32,
    pub effective_distance: f32,
    pub is_audible: bool,
}

/// 3D Spatial Audio Calculator
pub struct SpatialAudioEngine {
    pub listener: AudioListener,
    pub speed_of_sound: f32,
}

impl Default for SpatialAudioEngine {
    fn default() -> Self {
        Self {
            listener: AudioListener::default(),
            speed_of_sound: 343.0, // m/s
        }
    }
}

impl SpatialAudioEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate spatialized stereo volume and Doppler pitch for a sound source
    pub fn spatial_sample(
        &self,
        source: &SpatialSoundSource,
        manifold: Option<&Manifold>,
    ) -> SpatializedAudioSample {
        let (eff_pos, extra_dist) = if source.chart == self.listener.chart || manifold.is_none() {
            (source.pos, 0.0)
        } else {
            // Sound in another chart -> find shortest portal path
            let m = manifold.unwrap();
            let mut best_portal: Option<(Point3<f32>, f32)> = None;

            for portal in m.portals() {
                if portal.target_chart() == source.chart && portal.source_chart() == self.listener.chart && portal.is_active() {
                    let transformed = portal.transform_point(source.pos);
                    let d_portal_to_source = (transformed - Point3::new(0.0, 0.0, 0.0)).magnitude();
                    let d_listener_to_portal = (self.listener.pos - Point3::new(0.0, 0.0, 0.0)).magnitude();
                    let total = d_portal_to_source + d_listener_to_portal;

                    if best_portal.as_ref().map_or(true, |(_, d)| total < *d) {
                        best_portal = Some((transformed, total));
                    }
                }
            }

            if let Some((v_pos, total_d)) = best_portal {
                (v_pos, total_d)
            } else {
                (source.pos, 0.0)
            }
        };

        let delta = eff_pos - self.listener.pos;
        let distance = delta.magnitude() + extra_dist;

        // 1. Distance Attenuation
        let att_gain = match source.attenuation {
            AttenuationModel::Linear { min_dist, max_dist } => {
                if distance <= min_dist {
                    1.0
                } else if distance >= max_dist {
                    0.0
                } else {
                    1.0 - (distance - min_dist) / (max_dist - min_dist)
                }
            }
            AttenuationModel::InverseSquare { ref_dist, roll_off } => {
                ref_dist / (ref_dist + roll_off * (distance.max(ref_dist) - ref_dist))
            }
            AttenuationModel::Exponential { decay_rate } => {
                (-decay_rate * distance).exp()
            }
        };

        let gain = (source.base_volume * att_gain).clamp(0.0, 1.0);
        if gain < 0.001 {
            return SpatializedAudioSample {
                left_volume: 0.0,
                right_volume: 0.0,
                pitch_shift: source.base_pitch,
                effective_distance: distance,
                is_audible: false,
            };
        }

        // 2. Stereo Panning (Equal-Power Pan based on ear azimuth)
        let right_ear = self.listener.right();
        let sound_dir = if distance > 0.0001 { delta / distance } else { self.listener.forward };
        let pan = sound_dir.dot(right_ear); // -1.0 (full left) to +1.0 (full right)

        // Equal power panning curve
        let pan_angle = (pan + 1.0) * (std::f32::consts::PI / 4.0); // 0 to PI/2
        let left_vol = gain * pan_angle.cos();
        let right_vol = gain * pan_angle.sin();

        // 3. Doppler Pitch Shift
        let c = self.speed_of_sound;
        let v_listener = self.listener.velocity.dot(sound_dir);
        let v_source = source.velocity.dot(sound_dir);

        let doppler_ratio = ((c + v_listener) / (c + v_source).max(0.1)).clamp(0.5, 2.0);
        let pitch = source.base_pitch * doppler_ratio;

        SpatializedAudioSample {
            left_volume: left_vol,
            right_volume: right_vol,
            pitch_shift: pitch,
            effective_distance: distance,
            is_audible: true,
        }
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_attenuation() {
        let engine = SpatialAudioEngine::new();
        let source_near = SpatialSoundSource::new(0, Point3::new(0.0, 1.7, -2.0), ChartId(0));
        let source_far = SpatialSoundSource::new(1, Point3::new(0.0, 1.7, -20.0), ChartId(0));

        let sample_near = engine.spatial_sample(&source_near, None);
        let sample_far = engine.spatial_sample(&source_far, None);

        assert!(sample_near.left_volume > sample_far.left_volume);
    }

    #[test]
    fn test_stereo_panning_left_vs_right() {
        let engine = SpatialAudioEngine::new();
        // Listener is at (0, 1.7, 0) looking at (0, 0, -1), right ear is +X

        let source_right = SpatialSoundSource::new(0, Point3::new(5.0, 1.7, 0.0), ChartId(0));
        let source_left = SpatialSoundSource::new(1, Point3::new(-5.0, 1.7, 0.0), ChartId(0));

        let sample_right = engine.spatial_sample(&source_right, None);
        let sample_left = engine.spatial_sample(&source_left, None);

        assert!(sample_right.right_volume > sample_right.left_volume);
        assert!(sample_left.left_volume > sample_left.right_volume);
    }
}
