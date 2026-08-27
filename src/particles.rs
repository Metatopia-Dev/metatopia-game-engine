//! 3D Particle and visual effects simulation system
//!
//! Provides configurable emitters, physical simulations (gravity, wind, drag, bounce),
//! lifetime color/size curves, and dynamic mesh generation for GPU rendering.

use cgmath::{InnerSpace, Vector3};
use crate::quickstart::GameVertex;
use crate::collision::CollisionWorld;

/// 3D particle state
#[derive(Debug, Clone)]
pub struct Particle {
    pub pos: Vector3<f32>,
    pub vel: Vector3<f32>,
    pub acc: Vector3<f32>,
    pub rotation: f32,
    pub angular_vel: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub current_color: [f32; 4],
    pub current_size: f32,
    pub lifetime: f32,
    pub max_life: f32,
    pub bounce: f32,
    pub drag: f32,
    pub alive: bool,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            pos: Vector3::new(0.0, 0.0, 0.0),
            vel: Vector3::new(0.0, 0.0, 0.0),
            acc: Vector3::new(0.0, -9.81, 0.0),
            rotation: 0.0,
            angular_vel: 0.0,
            size_start: 0.2,
            size_end: 0.0,
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
            current_color: [1.0, 1.0, 1.0, 1.0],
            current_size: 0.2,
            lifetime: 0.0,
            max_life: 1.0,
            bounce: 0.4,
            drag: 0.02,
            alive: false,
        }
    }
}

/// Geometric shape defining emitter spawn volume
#[derive(Debug, Clone)]
pub enum EmitterShape {
    Point,
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    Cone { angle_rad: f32, direction: Vector3<f32>, radius: f32 },
    Circle { radius: f32, normal: Vector3<f32> },
}

/// Configurable 3D particle emitter
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    pub pos: Vector3<f32>,
    pub shape: EmitterShape,
    pub spawn_rate: f32, // particles/sec
    pub speed_range: (f32, f32),
    pub life_range: (f32, f32),
    pub size_range: (f32, f32),
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub gravity: Vector3<f32>,
    pub drag: f32,
    pub bounce: f32,
    pub active: bool,
    pub looping: bool,
    pub emission_timer: f32,
    pub duration: f32,
    spawn_accumulator: f32,
    rng_state: u32,
}

impl ParticleEmitter {
    pub fn new(pos: Vector3<f32>) -> Self {
        Self {
            pos,
            shape: EmitterShape::Point,
            spawn_rate: 30.0,
            speed_range: (2.0, 6.0),
            life_range: (0.5, 1.5),
            size_range: (0.25, 0.05),
            color_start: [0.0, 0.9, 1.0, 1.0],
            color_end: [0.8, 0.1, 0.9, 0.0],
            gravity: Vector3::new(0.0, -9.81, 0.0),
            drag: 0.05,
            bounce: 0.4,
            active: true,
            looping: true,
            emission_timer: 0.0,
            duration: 0.0,
            spawn_accumulator: 0.0,
            rng_state: 1234567,
        }
    }

    fn next_rand(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.rng_state >> 16) as f32) / 65536.0
    }

    fn next_rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_rand() * (max - min)
    }

    fn spawn_particle(&mut self) -> Particle {
        let life = self.next_rand_range(self.life_range.0, self.life_range.1);
        let speed = self.next_rand_range(self.speed_range.0, self.speed_range.1);

        // Position & initial direction based on shape
        let (offset, dir) = match self.shape {
            EmitterShape::Point => {
                let u = self.next_rand_range(-1.0, 1.0);
                let theta = self.next_rand_range(0.0, std::f32::consts::PI * 2.0);
                let r = (1.0 - u * u).max(0.0).sqrt();
                let dir = Vector3::new(r * theta.cos(), u, r * theta.sin()).normalize();
                (Vector3::new(0.0, 0.0, 0.0), dir)
            }
            EmitterShape::Sphere { radius } => {
                let u = self.next_rand_range(-1.0, 1.0);
                let theta = self.next_rand_range(0.0, std::f32::consts::PI * 2.0);
                let r = (1.0 - u * u).max(0.0).sqrt();
                let dir = Vector3::new(r * theta.cos(), u, r * theta.sin()).normalize();
                (dir * (radius * self.next_rand()), dir)
            }
            EmitterShape::Box { half_extents } => {
                let offset = Vector3::new(
                    self.next_rand_range(-half_extents[0], half_extents[0]),
                    self.next_rand_range(-half_extents[1], half_extents[1]),
                    self.next_rand_range(-half_extents[2], half_extents[2]),
                );
                let dir = Vector3::new(
                    self.next_rand_range(-1.0, 1.0),
                    self.next_rand_range(0.2, 1.0),
                    self.next_rand_range(-1.0, 1.0),
                ).normalize();
                (offset, dir)
            }
            EmitterShape::Cone { angle_rad, direction, radius } => {
                let base_dir = direction.normalize();
                let right = if base_dir.y.abs() < 0.99 {
                    base_dir.cross(Vector3::new(0.0, 1.0, 0.0)).normalize()
                } else {
                    base_dir.cross(Vector3::new(1.0, 0.0, 0.0)).normalize()
                };
                let up = base_dir.cross(right).normalize();

                let theta = self.next_rand_range(0.0, std::f32::consts::PI * 2.0);
                let phi = self.next_rand_range(0.0, angle_rad);
                let dir = (base_dir * phi.cos() + (right * theta.cos() + up * theta.sin()) * phi.sin()).normalize();
                let offset = (right * theta.cos() + up * theta.sin()) * (radius * self.next_rand());
                (offset, dir)
            }
            EmitterShape::Circle { radius, normal } => {
                let n = normal.normalize();
                let right = if n.y.abs() < 0.99 {
                    n.cross(Vector3::new(0.0, 1.0, 0.0)).normalize()
                } else {
                    n.cross(Vector3::new(1.0, 0.0, 0.0)).normalize()
                };
                let up = n.cross(right).normalize();
                let theta = self.next_rand_range(0.0, std::f32::consts::PI * 2.0);
                let r = radius * self.next_rand().sqrt();
                let offset = (right * theta.cos() + up * theta.sin()) * r;
                let dir = (n + Vector3::new(self.next_rand_range(-0.2, 0.2), 0.0, self.next_rand_range(-0.2, 0.2))).normalize();
                (offset, dir)
            }
        };

        Particle {
            pos: self.pos + offset,
            vel: dir * speed,
            acc: self.gravity,
            rotation: self.next_rand_range(0.0, std::f32::consts::PI * 2.0),
            angular_vel: self.next_rand_range(-3.0, 3.0),
            size_start: self.size_range.0,
            size_end: self.size_range.1,
            color_start: self.color_start,
            color_end: self.color_end,
            current_color: self.color_start,
            current_size: self.size_range.0,
            lifetime: 0.0,
            max_life: life,
            bounce: self.bounce,
            drag: self.drag,
            alive: true,
        }
    }
}

/// Central Particle Simulation System
pub struct ParticleSystem {
    pub emitters: Vec<ParticleEmitter>,
    pub particles: Vec<Particle>,
    pub max_particles: usize,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(2000)
    }
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            emitters: Vec::new(),
            particles: Vec::with_capacity(max_particles),
            max_particles,
        }
    }

    /// Add an emitter to the system
    pub fn add_emitter(&mut self, emitter: ParticleEmitter) -> usize {
        let id = self.emitters.len();
        self.emitters.push(emitter);
        id
    }

    /// Instant burst explosion of particles at a point
    pub fn burst(
        &mut self,
        pos: Vector3<f32>,
        count: usize,
        color_start: [f32; 4],
        color_end: [f32; 4],
        speed: f32,
        lifetime: f32,
    ) {
        let mut rng = (pos.x * 37.0 + pos.z * 13.0).abs() as u32 + 101;
        for _ in 0..count {
            if self.particles.len() >= self.max_particles { break; }
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let rx = ((rng >> 16) as f32 / 32768.0) - 1.0;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let ry = ((rng >> 16) as f32 / 32768.0) - 1.0;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let rz = ((rng >> 16) as f32 / 32768.0) - 1.0;

            let dir = Vector3::new(rx, ry + 0.3, rz).normalize();
            let spd = speed * (0.6 + ((rng % 100) as f32 / 200.0));
            let life = lifetime * (0.7 + ((rng % 100) as f32 / 300.0));

            self.particles.push(Particle {
                pos,
                vel: dir * spd,
                acc: Vector3::new(0.0, -9.81, 0.0),
                rotation: rx * 3.14,
                angular_vel: ry * 4.0,
                size_start: 0.25,
                size_end: 0.02,
                color_start,
                color_end,
                current_color: color_start,
                current_size: 0.25,
                lifetime: 0.0,
                max_life: life,
                bounce: 0.5,
                drag: 0.03,
                alive: true,
            });
        }
    }

    /// Update physics and simulation step
    pub fn update(&mut self, dt: f32, collision: Option<&CollisionWorld>) {
        // 1. Update Emitters & Spawn
        for emitter in &mut self.emitters {
            if !emitter.active { continue; }
            emitter.emission_timer += dt;
            if !emitter.looping && emitter.duration > 0.0 && emitter.emission_timer >= emitter.duration {
                emitter.active = false;
                continue;
            }

            emitter.spawn_accumulator += emitter.spawn_rate * dt;
            let spawn_count = emitter.spawn_accumulator.floor() as usize;
            emitter.spawn_accumulator -= spawn_count as f32;

            for _ in 0..spawn_count {
                if self.particles.len() >= self.max_particles { break; }
                self.particles.push(emitter.spawn_particle());
            }
        }

        // 2. Update Active Particles
        for p in &mut self.particles {
            if !p.alive { continue; }
            p.lifetime += dt;
            if p.lifetime >= p.max_life {
                p.alive = false;
                continue;
            }

            let t = p.lifetime / p.max_life;

            // Interpolate Size
            p.current_size = p.size_start + (p.size_end - p.size_start) * t;

            // Interpolate Color (RGBA)
            for i in 0..4 {
                p.current_color[i] = p.color_start[i] + (p.color_end[i] - p.color_start[i]) * t;
            }

            // Apply Drag & Acceleration
            let drag_force = -p.vel * p.drag;
            p.vel += (p.acc + drag_force) * dt;
            let next_pos = p.pos + p.vel * dt;

            // Floor & Collision Bounce
            if next_pos.y <= 0.02 {
                p.pos.y = 0.02;
                p.vel.y = -p.vel.y * p.bounce;
                p.vel.x *= 0.85;
                p.vel.z *= 0.85;
            } else if let Some(world) = collision {
                // Check sphere query for collisions
                let nearby = world.query_sphere([next_pos.x, next_pos.y, next_pos.z], p.current_size);
                if !nearby.is_empty() {
                    p.vel = -p.vel * p.bounce;
                } else {
                    p.pos = next_pos;
                }
            } else {
                p.pos = next_pos;
            }

            p.rotation += p.angular_vel * dt;
        }

        self.particles.retain(|p| p.alive);
    }

    /// Build a 3D vertex mesh of billboarded diamonds / cubes for GPU rendering
    pub fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.particles.len() * 8);
        let mut indices = Vec::with_capacity(self.particles.len() * 24);

        for p in &self.particles {
            if !p.alive || p.current_color[3] <= 0.01 { continue; }
            let s = p.current_size.max(0.01);
            let rgb = [p.current_color[0], p.current_color[1], p.current_color[2]];
            let alpha = p.current_color[3];

            // 3D Octahedron / Spark
            let top = [p.pos.x, p.pos.y + s, p.pos.z];
            let bot = [p.pos.x, p.pos.y - s, p.pos.z];
            let px = [p.pos.x + s, p.pos.y, p.pos.z];
            let nx = [p.pos.x - s, p.pos.y, p.pos.z];
            let pz = [p.pos.x, p.pos.y, p.pos.z + s];
            let nz = [p.pos.x, p.pos.y, p.pos.z - s];

            let pbr = [0.0, 0.1, alpha * 5.0, 0.0];

            let tris = [
                (top, px, pz), (top, pz, nx), (top, nx, nz), (top, nz, px),
                (bot, pz, px), (bot, nx, pz), (bot, nz, nx), (bot, px, nz),
            ];

            for (a, b, c) in tris {
                let base = verts.len() as u32;
                let norm = [0.0, 1.0, 0.0];
                verts.push(GameVertex::new(a, norm, rgb, pbr));
                verts.push(GameVertex::new(b, norm, rgb, pbr));
                verts.push(GameVertex::new(c, norm, rgb, pbr));
                indices.extend_from_slice(&[base, base + 1, base + 2]);
            }
        }

        (verts, indices)
    }

    /// Number of currently active particles
    pub fn active_count(&self) -> usize {
        self.particles.len()
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_burst_and_update() {
        let mut ps = ParticleSystem::new(500);
        ps.burst(
            Vector3::new(0.0, 5.0, 0.0),
            20,
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            8.0,
            1.0,
        );
        assert_eq!(ps.active_count(), 20);

        ps.update(0.1, None);
        assert_eq!(ps.active_count(), 20);

        // Fast-forward past lifetime
        ps.update(1.5, None);
        assert_eq!(ps.active_count(), 0);
    }

    #[test]
    fn test_emitter_continuous_spawn() {
        let mut ps = ParticleSystem::new(500);
        let mut emitter = ParticleEmitter::new(Vector3::new(0.0, 1.0, 0.0));
        emitter.spawn_rate = 50.0;
        ps.add_emitter(emitter);

        ps.update(0.1, None);
        assert!(ps.active_count() >= 4);

        let (verts, indices) = ps.build_mesh();
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
    }
}
