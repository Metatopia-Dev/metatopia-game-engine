//! Non-Euclidean Interactive Physics Demo
//!
//! Physics change with the geometry:
//!   Euclidean  – normal gravity, standard bounce, straight trajectories
//!   Hyperbolic – weak gravity, high bounce, divergent geodesics push objects apart
//!   Spherical  – strong gravity, low bounce, convergent geodesics pull objects inward
//!
//! Sphere ↔ Wall, Sphere ↔ Floor, Player ↔ Sphere, Sphere ↔ Orb collisions

use metatopia_engine::prelude::*;
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent, ElementState, DeviceEvent},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder as WinitWindowBuilder,
};
use std::sync::{Arc, RwLock};
use cgmath::{InnerSpace, Point3, Vector3, Matrix4, Deg, perspective};
use wgpu::util::DeviceExt;
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use std::time::Duration;

// ─── GPU Uniforms ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_position: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    light0_pos: [f32; 4],
    light0_color: [f32; 4],
    light1_pos: [f32; 4],
    light1_color: [f32; 4],
    light2_pos: [f32; 4],
    light2_color: [f32; 4],
    light3_pos: [f32; 4],
    light3_color: [f32; 4],
    params: [f32; 4],
    sphere_pos: [f32; 4],
    orb0_pos: [f32; 4],
    orb1_pos: [f32; 4],
    orb2_pos: [f32; 4],
    orb3_pos: [f32; 4],
    interaction: [f32; 4],
    hud_info: [f32; 4],
}

// ─── Sound Events & Procedural Audio ───────────────────────────────────────

#[derive(Debug, Clone)]
enum SoundEvent {
    Bounce(f32),        // intensity 0.0–1.0
    OrbCollect(u32),    // count collected so far
    OrbSmash,           // sphere smashed an orb
    Portal,             // geometry transition
    QuantumKick,
}

struct GameAudio {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    ambient_sink: Sink,
    current_drone: u32,
    last_bounce: f32,
}

impl GameAudio {
    fn new() -> Option<Self> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        let ambient = Sink::try_new(&handle).ok()?;
        ambient.set_volume(0.0);
        Some(Self {
            _stream: stream,
            handle,
            ambient_sink: ambient,
            current_drone: 99, // force initial update
            last_bounce: -1.0,
        })
    }

    /// Ambient drone that shifts with geometry type
    fn update_ambient(&mut self, space_type: u32) {
        if space_type == self.current_drone { return; }
        self.current_drone = space_type;
        self.ambient_sink.stop();
        self.ambient_sink = Sink::try_new(&self.handle).unwrap();
        self.ambient_sink.set_volume(0.06);

        let (f1, f2, f3) = match space_type {
            1 => (73.4, 110.0, 146.8),    // Hyperbolic: dark minor cluster
            2 => (146.8, 220.0, 330.0),    // Spherical: warm major triad
            _ => (110.0, 165.0, 220.0),    // Euclidean: clean fifth + octave
        };

        // Layer 3 sine waves for a rich drone
        let s1 = rodio::source::SineWave::new(f1)
            .amplify(0.5)
            .fade_in(Duration::from_secs(2));
        let s2 = rodio::source::SineWave::new(f2)
            .amplify(0.3)
            .fade_in(Duration::from_secs(2));
        let s3 = rodio::source::SineWave::new(f3)
            .amplify(0.15)
            .fade_in(Duration::from_secs(3));

        // Mix by playing on separate sinks (rodio appends sequentially on one sink)
        self.ambient_sink.append(s1);
        // Additional layers on separate sinks
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.04);
            s.append(s2);
            s.detach();
        }
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.025);
            s.append(s3);
            s.detach();
        }
    }

    /// Short low-frequency thud on sphere bounce
    fn play_bounce(&mut self, intensity: f32, time: f32) {
        if time - self.last_bounce < 0.08 { return; }
        self.last_bounce = time;
        let vol = (intensity * 0.6).min(0.5);
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(vol);
            let freq = 55.0 + intensity * 30.0;
            let thud = rodio::source::SineWave::new(freq)
                .take_duration(Duration::from_millis(60));
            let sub = rodio::source::SineWave::new(freq * 0.5)
                .take_duration(Duration::from_millis(90))
                .amplify(0.6);
            s.append(thud);
            s.append(sub);
            s.detach();
        }
    }

    /// Bright ascending chime on orb collection
    fn play_chime(&self, count: u32) {
        let base = 660.0 + count as f32 * 110.0; // pitch rises with each orb
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.25);
            let note1 = rodio::source::SineWave::new(base)
                .take_duration(Duration::from_millis(120))
                .fade_in(Duration::from_millis(5));
            let note2 = rodio::source::SineWave::new(base * 1.25) // major third up
                .take_duration(Duration::from_millis(180))
                .fade_in(Duration::from_millis(5));
            let note3 = rodio::source::SineWave::new(base * 1.5) // perfect fifth
                .take_duration(Duration::from_millis(250))
                .fade_in(Duration::from_millis(10));
            s.append(note1);
            s.append(note2);
            s.append(note3);
            s.detach();
        }
    }

    /// Impact + sparkle on sphere-smash orb collection
    fn play_smash(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.35);
            // Low impact
            let hit = rodio::source::SineWave::new(80.0)
                .take_duration(Duration::from_millis(50));
            // High sparkle
            let spark1 = rodio::source::SineWave::new(1200.0)
                .take_duration(Duration::from_millis(40));
            let spark2 = rodio::source::SineWave::new(1800.0)
                .take_duration(Duration::from_millis(60))
                .amplify(0.6);
            s.append(hit);
            s.append(spark1);
            s.append(spark2);
            s.detach();
        }
    }

    /// Rising frequency sweep on portal transition
    fn play_portal(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.3);
            for i in 0..12 {
                let freq = 150.0 + i as f32 * 80.0;
                let vol = 1.0 - (i as f32 / 12.0) * 0.6;
                let tone = rodio::source::SineWave::new(freq)
                    .take_duration(Duration::from_millis(40))
                    .amplify(vol);
                s.append(tone);
            }
            s.detach();
        }
    }

    /// Brief electronic chirp on quantum kick
    fn play_zap(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.12);
            let z1 = rodio::source::SineWave::new(2200.0)
                .take_duration(Duration::from_millis(15));
            let z2 = rodio::source::SineWave::new(1400.0)
                .take_duration(Duration::from_millis(25))
                .amplify(0.7);
            s.append(z1);
            s.append(z2);
            s.detach();
        }
    }

    /// Process all queued sound events
    fn process(&mut self, events: &[SoundEvent], space_type: u32, time: f32) {
        self.update_ambient(space_type);
        for ev in events {
            match ev {
                SoundEvent::Bounce(intensity) => self.play_bounce(*intensity, time),
                SoundEvent::OrbCollect(count) => self.play_chime(*count),
                SoundEvent::OrbSmash => self.play_smash(),
                SoundEvent::Portal => self.play_portal(),
                SoundEvent::QuantumKick => self.play_zap(),
            }
        }
    }
}

// ─── Physics Objects ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SphereState {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    radius: f32,
}

#[derive(Debug, Clone)]
struct OrbState {
    base_position: Vector3<f32>,
    display_position: Vector3<f32>,
    phase: f32,
    active: bool,
}

/// Physics constants that change per geometry
struct SpacePhysics {
    gravity: f32,
    restitution: f32,
    drag: f32,
    friction: f32,
    push_force: f32,
    geodesic_strength: f32,
    move_speed: f32,
    label: &'static str,
}

impl SpacePhysics {
    fn euclidean() -> Self {
        Self { gravity: -12.0, restitution: 0.55, drag: 0.985, friction: 0.92,
               push_force: 18.0, geodesic_strength: 0.0, move_speed: 1.0, label: "Flat" }
    }
    fn hyperbolic() -> Self {
        Self { gravity: -4.0, restitution: 0.88, drag: 0.998, friction: 0.98,
               push_force: 25.0, geodesic_strength: 0.8, move_speed: 1.4, label: "K<0 Divergent" }
    }
    fn spherical() -> Self {
        Self { gravity: -22.0, restitution: 0.30, drag: 0.96, friction: 0.80,
               push_force: 12.0, geodesic_strength: 2.0, move_speed: 0.7, label: "K>0 Convergent" }
    }
    fn for_type(t: u32) -> Self {
        match t { 1 => Self::hyperbolic(), 2 => Self::spherical(), _ => Self::euclidean() }
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

fn default_orb_positions() -> [OrbState; 4] {
    [
        OrbState { base_position: Vector3::new( 5.0, 1.0, -6.0), display_position: Vector3::new(5.0, 1.0, -6.0), phase: 0.0, active: true },
        OrbState { base_position: Vector3::new(-5.0, 1.5,  6.0), display_position: Vector3::new(-5.0, 1.5, 6.0), phase: 1.2, active: true },
        OrbState { base_position: Vector3::new( 7.0, 2.0,  4.0), display_position: Vector3::new(7.0, 2.0, 4.0), phase: 2.5, active: true },
        OrbState { base_position: Vector3::new(-7.0, 0.5, -4.0), display_position: Vector3::new(-7.0, 0.5, -4.0), phase: 3.8, active: true },
    ]
}

// ─── Demo ──────────────────────────────────────────────────────────────────

struct NonEuclideanDemo {
    manifold: Arc<RwLock<Manifold>>,
    camera_position: Point3<f32>,
    camera_rotation: (f32, f32),
    current_chart: ChartId,
    movement_speed: f32,
    camera_uniform: CameraUniform,
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    scene_uniform: SceneUniform,
    scene_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    mouse_sensitivity: f32,
    sphere: SphereState,
    orbs: [OrbState; 4],
    collected: u32,
    space_type: u32,
    frame_count: u32,
    noise_seed: f32,
    sound_events: Vec<SoundEvent>,
}

impl NonEuclideanDemo {
    fn new() -> Self {
        let mut manifold = Manifold::new();
        let hyp = manifold.add_chart(GeometryType::Hyperbolic);
        let sph = manifold.add_chart(GeometryType::Spherical);
        manifold.create_portal(ChartId(0), hyp,
            Point3::new(5.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0), Mat4::from_scale(1.0)).unwrap();
        manifold.create_portal(hyp, sph,
            Point3::new(0.5, 0.5, 0.0), Point3::new(0.0, 0.0, 1.0), Mat4::from_scale(1.0)).unwrap();
        manifold.create_portal(sph, ChartId(0),
            Point3::new(0.0, 1.0, 0.0), Point3::new(-5.0, 0.0, 0.0), Mat4::from_scale(1.0)).unwrap();

        Self {
            manifold: Arc::new(RwLock::new(manifold)),
            camera_position: Point3::new(0.0, 1.0, -5.0),
            camera_rotation: (0.0, 0.0),
            current_chart: ChartId(0),
            movement_speed: 0.1,
            camera_uniform: CameraUniform { view_proj: [[0.0; 4]; 4], view_position: [0.0, 1.0, -5.0, 0.0] },
            camera_buffer: None,
            camera_bind_group: None,
            scene_uniform: SceneUniform {
                sun_direction: [0.0;4], sun_color: [0.0;4],
                light0_pos: [0.0;4], light0_color: [0.0;4],
                light1_pos: [0.0;4], light1_color: [0.0;4],
                light2_pos: [0.0;4], light2_color: [0.0;4],
                light3_pos: [0.0;4], light3_color: [0.0;4],
                params: [0.0;4], sphere_pos: [0.0;4],
                orb0_pos: [0.0;4], orb1_pos: [0.0;4], orb2_pos: [0.0;4], orb3_pos: [0.0;4],
                interaction: [0.0;4],
                hud_info: [0.0;4],
            },
            scene_buffer: None, scene_bind_group: None,
            mouse_sensitivity: 0.002,
            sphere: SphereState { position: Vector3::new(0.0, 0.5, 0.0), velocity: Vector3::new(0.0, 0.0, 0.0), radius: 1.2 },
            orbs: default_orb_positions(),
            collected: 0,
            space_type: 0,
            frame_count: 0,
            noise_seed: 0.0,
            sound_events: Vec::new(),
        }
    }

    // ── Main Update ────────────────────────────────────────────────

    fn update(&mut self, dt: f32, time: f32) {
        self.check_portal_transitions();

        // Per-frame non-deterministic seed (LCG-style)
        self.frame_count = self.frame_count.wrapping_add(1);
        self.noise_seed = (self.frame_count.wrapping_mul(1103515245).wrapping_add(12345) >> 16) as f32
            / 65536.0;

        let phys = SpacePhysics::for_type(self.space_type);
        self.update_sphere_physics(dt, &phys);
        self.apply_quantum_kick(time);
        self.update_orbs(time, &phys);
        self.check_sphere_orb_collisions();
    }

    /// Non-deterministic random impulse on the sphere ("quantum kick")
    fn apply_quantum_kick(&mut self, time: f32) {
        // Probability of kick varies by geometry
        let (kick_chance, kick_strength) = match self.space_type {
            1 => (0.02_f32, 3.5_f32),   // Hyperbolic: frequent strong kicks
            2 => (0.005, 1.0),           // Spherical: rare weak kicks
            _ => (0.008, 2.0),           // Euclidean: moderate
        };

        // Pseudo-random test using hash of frame + time
        let h = ((self.frame_count as f32 * 0.1031).fract() * 33.33).fract();
        if h < kick_chance {
            let angle = time * 17.3 + self.noise_seed * 100.0;
            let kick = Vector3::new(
                angle.cos() * kick_strength,
                kick_strength * 0.7,
                angle.sin() * kick_strength,
            );
            self.sphere.velocity += kick;
            self.sound_events.push(SoundEvent::QuantumKick);
        }
    }

    // ── Geometry-Dependent Sphere Physics ──────────────────────────

    fn update_sphere_physics(&mut self, dt: f32, phys: &SpacePhysics) {
        let room = 10.0;
        let floor_y = -2.0 + self.sphere.radius;
        let ceil_y  =  5.0 - self.sphere.radius;

        // ── 1. Gravity (space-dependent) ──────────────────────────
        self.sphere.velocity.y += phys.gravity * dt;

        // ── 2. Geodesic curvature force ───────────────────────────
        let r_xz = (self.sphere.position.x.powi(2) + self.sphere.position.z.powi(2)).sqrt();
        if r_xz > 0.5 && phys.geodesic_strength > 0.0 {
            let dir_xz = Vector3::new(self.sphere.position.x, 0.0, self.sphere.position.z) / r_xz;
            match self.space_type {
                1 => {
                    // Hyperbolic: DIVERGENT geodesics – objects pushed outward
                    self.sphere.velocity += dir_xz * r_xz * phys.geodesic_strength * dt;
                }
                2 => {
                    // Spherical: CONVERGENT geodesics – objects pulled inward
                    self.sphere.velocity -= dir_xz * r_xz * phys.geodesic_strength * dt;
                }
                _ => {}
            }
        }

        // ── 3. Integrate position ─────────────────────────────────
        self.sphere.position += self.sphere.velocity * dt;

        // ── 4. Boundary collisions (space-dependent restitution) ──
        if self.sphere.position.y < floor_y {
            let impact = self.sphere.velocity.y.abs();
            self.sphere.position.y = floor_y;
            self.sphere.velocity.y = -self.sphere.velocity.y * phys.restitution;
            self.sphere.velocity.x *= phys.friction;
            self.sphere.velocity.z *= phys.friction;
            if impact > 1.5 { self.sound_events.push(SoundEvent::Bounce((impact / 8.0).min(1.0))); }
        }
        if self.sphere.position.y > ceil_y {
            let impact = self.sphere.velocity.y.abs();
            self.sphere.position.y = ceil_y;
            self.sphere.velocity.y = -self.sphere.velocity.y * phys.restitution;
            if impact > 2.0 { self.sound_events.push(SoundEvent::Bounce((impact / 10.0).min(1.0))); }
        }

        let wall = room - self.sphere.radius;
        if self.sphere.position.x >  wall {
            let impact = self.sphere.velocity.x.abs();
            self.sphere.position.x =  wall; self.sphere.velocity.x = -self.sphere.velocity.x * phys.restitution;
            if impact > 2.0 { self.sound_events.push(SoundEvent::Bounce((impact / 10.0).min(1.0))); }
        }
        if self.sphere.position.x < -wall {
            let impact = self.sphere.velocity.x.abs();
            self.sphere.position.x = -wall; self.sphere.velocity.x = -self.sphere.velocity.x * phys.restitution;
            if impact > 2.0 { self.sound_events.push(SoundEvent::Bounce((impact / 10.0).min(1.0))); }
        }
        if self.sphere.position.z >  wall {
            let impact = self.sphere.velocity.z.abs();
            self.sphere.position.z =  wall; self.sphere.velocity.z = -self.sphere.velocity.z * phys.restitution;
            if impact > 2.0 { self.sound_events.push(SoundEvent::Bounce((impact / 10.0).min(1.0))); }
        }
        if self.sphere.position.z < -wall {
            let impact = self.sphere.velocity.z.abs();
            self.sphere.position.z = -wall; self.sphere.velocity.z = -self.sphere.velocity.z * phys.restitution;
            if impact > 2.0 { self.sound_events.push(SoundEvent::Bounce((impact / 10.0).min(1.0))); }
        }

        // ── 5. Drag (space-dependent) ─────────────────────────────
        self.sphere.velocity *= phys.drag;

        // ── 6. Player → Sphere push ───────────────────────────────
        let cam = Vector3::new(self.camera_position.x, self.camera_position.y, self.camera_position.z);
        let to_sphere = self.sphere.position - cam;
        let dist = to_sphere.magnitude();
        let push_range = self.sphere.radius + 0.6;

        if dist < push_range && dist > 0.01 {
            let push_dir = to_sphere / dist;
            let overlap = push_range - dist;
            self.sphere.velocity += push_dir * overlap * phys.push_force;
            // Hyperbolic: pop up more (floaty), Spherical: barely pops
            let pop = match self.space_type {
                1 => overlap * phys.push_force * 0.6,  // floaty pop
                2 => overlap * phys.push_force * 0.1,  // heavy, stays low
                _ => overlap * phys.push_force * 0.35,
            };
            self.sphere.velocity.y += pop;
        }
    }

    // ── Geometry-Dependent Orb Behavior ────────────────────────────

    fn update_orbs(&mut self, time: f32, _phys: &SpacePhysics) {
        let collect_radius = 1.8;

        for (i, orb) in self.orbs.iter_mut().enumerate() {
            if !orb.active { continue; }

            // ── Orb motion depends on geometry ────────────────────
            match self.space_type {
                1 => {
                    // Hyperbolic: orbs orbit outward, expanding orbits
                    let angle = time * 0.35 + orb.phase;
                    let r = 5.5 + (time * 0.08 + orb.phase).sin() * 2.5;
                    orb.display_position = Vector3::new(
                        angle.cos() * r,
                        orb.base_position.y + (time * 2.0 + orb.phase).sin() * 0.5,
                        angle.sin() * r,
                    );
                }
                2 => {
                    // Spherical: orbs cluster inward, tight slow orbit
                    let angle = time * 0.15 + orb.phase;
                    let r = 2.5 + (time * 0.2 + orb.phase).sin() * 0.8;
                    orb.display_position = Vector3::new(
                        angle.cos() * r,
                        0.5 + (time * 1.0 + orb.phase).sin() * 0.15 + i as f32 * 0.3,
                        angle.sin() * r,
                    );
                }
                _ => {
                    // Euclidean: stationary with gentle bob
                    orb.display_position = Vector3::new(
                        orb.base_position.x,
                        orb.base_position.y + (time * 1.5 + orb.phase).sin() * 0.3,
                        orb.base_position.z,
                    );
                }
            }

            // ── Player → Orb collection ──────────────────────────
            let dx = self.camera_position.x - orb.display_position.x;
            let dy = self.camera_position.y - orb.display_position.y;
            let dz = self.camera_position.z - orb.display_position.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < collect_radius {
                orb.active = false;
                self.collected += 1;
                self.sound_events.push(SoundEvent::OrbCollect(self.collected));
                let phys_label = SpacePhysics::for_type(self.space_type).label;
                println!("✨ Orb collected! ({}/4) [{}]", self.collected, phys_label);
                if self.collected == 4 { println!("🎉 All orbs collected in {} space!", phys_label); }
            }
        }
    }

    // ── Sphere ↔ Orb Collision ─────────────────────────────────────

    fn check_sphere_orb_collisions(&mut self) {
        for orb in &mut self.orbs {
            if !orb.active { continue; }
            let to_orb = orb.display_position - self.sphere.position;
            let dist = to_orb.magnitude();
            let collision_r = self.sphere.radius + 0.35;

            if dist < collision_r && dist > 0.01 {
                orb.active = false;
                self.collected += 1;
                // Sphere bounces off the orb
                let bounce = -to_orb / dist;
                self.sphere.velocity += bounce * 4.0;
                self.sound_events.push(SoundEvent::OrbSmash);
                println!("💥 Sphere smashed orb! ({}/4)", self.collected);
                if self.collected == 4 { println!("🎉 All orbs collected!"); }
            }
        }
    }

    // ── Portal Transitions ─────────────────────────────────────────

    fn check_portal_transitions(&mut self) {
        let forward = self.get_forward_vector();
        if let Ok(manifold) = self.manifold.read() {
            if let Some((_id, intersection, new_chart)) =
                manifold.ray_portal_intersection(self.camera_position, forward, self.current_chart)
            {
                self.camera_position = intersection;
                self.current_chart = new_chart;
                drop(manifold);
                if let Ok(mut m) = self.manifold.write() { m.set_active_chart(new_chart); }

                // Update cached space type
                self.space_type = self.resolve_space_type();
                self.sound_events.push(SoundEvent::Portal);
                let phys = SpacePhysics::for_type(self.space_type);
                println!("🌀 Portal → Chart {:?} | Physics: {} | G={:.0} bounce={:.0}%",
                    new_chart, phys.label, phys.gravity, phys.restitution * 100.0);

                // Reset objects with new physics
                self.sphere.position = Vector3::new(0.0, 3.0, 0.0); // drop from height
                self.sphere.velocity = Vector3::new(0.0, 0.0, 0.0);
                self.orbs = default_orb_positions();
                self.collected = 0;
            }
        }
    }

    fn resolve_space_type(&self) -> u32 {
        if let Ok(m) = self.manifold.read() {
            match m.chart(self.current_chart).unwrap().geometry() {
                GeometryType::Euclidean  => 0,
                GeometryType::Hyperbolic => 1,
                GeometryType::Spherical  => 2,
                _                        => 0,
            }
        } else { 0 }
    }

    // ── Camera ─────────────────────────────────────────────────────

    fn get_forward_vector(&self) -> Vector3<f32> {
        let (yaw, pitch) = self.camera_rotation;
        Vector3::new(yaw.cos() * pitch.cos(), pitch.sin(), yaw.sin() * pitch.cos())
    }

    fn update_camera_uniform(&mut self, aspect: f32) {
        let fwd = self.get_forward_vector();
        let tgt = self.camera_position + fwd;
        let view = Matrix4::<f32>::look_at_rh(
            self.camera_position, Point3::new(tgt.x, tgt.y, tgt.z), Vector3::unit_y(),
        );
        let proj = perspective(Deg(60.0), aspect, 0.1, 150.0);
        self.camera_uniform.view_proj = (proj * view).into();

        self.space_type = self.resolve_space_type();
        self.camera_uniform.view_position = [
            self.camera_position.x, self.camera_position.y, self.camera_position.z,
            self.space_type as f32,
        ];
    }

    fn update_scene_uniform(&mut self, time: f32) {
        let st = self.space_type;

        // ── Stochastic light flicker ──────────────────────────────
        // Non-deterministic intensity modulation using hash-like trig
        let flicker = |freq: f32, phase: f32| -> f32 {
            let a = (time * freq + phase).sin();
            let b = (time * freq * 1.618 + phase * 2.3).cos();
            1.0 + a * b * 0.12  // ±12% random-looking flicker
        };

        let sd = Vector3::new(0.3_f32, 0.8, 0.4).normalize();
        self.scene_uniform.sun_direction = [sd.x, sd.y, sd.z, 3.0 * flicker(7.3, 0.0)];
        self.scene_uniform.sun_color     = [1.0, 0.95, 0.85, 0.0];
        self.scene_uniform.light0_pos    = [0.0, 4.5, 0.0, 15.0];
        self.scene_uniform.light0_color  = [1.0, 0.92, 0.80, 40.0 * flicker(11.7, 1.0)];

        let (ar, ag, ab) = match st { 1 => (0.6,0.3,1.0), 2 => (1.0,0.5,0.15), _ => (0.3,0.6,1.0) };
        self.scene_uniform.light1_pos   = [6.0*(time*0.4).cos(), 2.0, 6.0*(time*0.4).sin(), 10.0];
        self.scene_uniform.light1_color = [ar, ag, ab, 25.0 * flicker(13.1, 2.5)];

        let pulse = (time * 1.2).sin() * 0.3 + 0.7;
        self.scene_uniform.light2_pos   = [-4.0, 1.0, -4.0, 8.0];
        self.scene_uniform.light2_color = [0.9, 0.85, 0.7, 15.0 * pulse * flicker(9.3, 4.0)];

        let (pr,pg,pb) = match st { 1=>(0.8,0.3,1.0), 2=>(1.0,0.6,0.2), _=>(0.4,0.6,1.0) };
        let pp = (time * 2.0).sin() * 0.25 + 0.75;
        self.scene_uniform.light3_pos   = [9.5, 1.5, 0.0, 6.0];
        self.scene_uniform.light3_color = [pr, pg, pb, 20.0 * pp * flicker(17.7, 6.0)];

        self.scene_uniform.params = [time, 1.8, 3.5, 4.0];

        // ── Dynamic sphere ────────────────────────────────────────
        let cam = Vector3::new(self.camera_position.x, self.camera_position.y, self.camera_position.z);
        let sd2 = (self.sphere.position - cam).magnitude();
        let glow = (1.0 - (sd2 / 4.0).min(1.0)).max(0.0);
        self.scene_uniform.sphere_pos = [self.sphere.position.x, self.sphere.position.y, self.sphere.position.z, glow];

        // ── Orbs ──────────────────────────────────────────────────
        let orb_fields: [&mut [f32; 4]; 4] = [
            &mut self.scene_uniform.orb0_pos, &mut self.scene_uniform.orb1_pos,
            &mut self.scene_uniform.orb2_pos, &mut self.scene_uniform.orb3_pos,
        ];
        for (i, u) in orb_fields.into_iter().enumerate() {
            let o = &self.orbs[i];
            *u = [o.display_position.x, o.display_position.y, o.display_position.z,
                  if o.active { 1.0 } else { 0.0 }];
        }

        self.scene_uniform.interaction = [self.collected as f32, 4.0, sd2, self.noise_seed];
    }

    // ── Input (speed varies by geometry) ───────────────────────────

    fn handle_keyboard(&mut self, key: KeyCode, pressed: bool) {
        if !pressed { return; }
        let phys = SpacePhysics::for_type(self.space_type);
        let speed = self.movement_speed * phys.move_speed;
        let fwd   = self.get_forward_vector();
        let right = Vector3::new(-fwd.z, 0.0, fwd.x).normalize();

        match key {
            KeyCode::KeyW       => self.camera_position += fwd * speed,
            KeyCode::KeyS       => self.camera_position -= fwd * speed,
            KeyCode::KeyA       => self.camera_position -= right * speed,
            KeyCode::KeyD       => self.camera_position += right * speed,
            KeyCode::Space      => self.camera_position.y += speed,
            KeyCode::ShiftLeft  => self.camera_position.y -= speed,
            KeyCode::ArrowLeft  => self.camera_rotation.0 -= 0.05,
            KeyCode::ArrowRight => self.camera_rotation.0 += 0.05,
            KeyCode::ArrowUp    => self.camera_rotation.1 = (self.camera_rotation.1 - 0.05).clamp(-1.5, 1.5),
            KeyCode::ArrowDown  => self.camera_rotation.1 = (self.camera_rotation.1 + 0.05).clamp(-1.5, 1.5),
            KeyCode::KeyR => {
                self.camera_position = Point3::new(0.0, 1.0, -5.0);
                self.camera_rotation = (0.0, 0.0);
                self.current_chart = ChartId(0);
                self.space_type = 0;
                self.sphere = SphereState { position: Vector3::new(0.0, 3.0, 0.0),
                    velocity: Vector3::new(0.0, 0.0, 0.0), radius: 1.2 };
                self.orbs = default_orb_positions();
                self.collected = 0;
                println!("↻ Reset to Euclidean origin");
            }
            _ => {}
        }
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.camera_rotation.0 += dx as f32 * self.mouse_sensitivity;
        self.camera_rotation.1 = (self.camera_rotation.1 - dy as f32 * self.mouse_sensitivity).clamp(-1.5, 1.5);
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

async fn run() {
    env_logger::init();

    println!("🌐 Non-Euclidean Physics Demo");
    println!("══════════════════════════════");
    println!("Controls: WASD Move · Space/Shift Up/Down · Mouse Look · R Reset · ESC Quit");
    println!();
    println!("Physics change with geometry:");
    println!("  🔵 Euclidean  – Normal gravity, standard bounce");
    println!("  🟣 Hyperbolic – Weak gravity, super bouncy, objects fly apart (K<0)");
    println!("  🟠 Spherical  – Strong gravity, sticky, objects pulled to center (K>0)");
    println!();
    println!("Interactions:");
    println!("  ⚽ Push the sphere (it bounces off walls with space-dependent physics)");
    println!("  ✨ Collect orbs by walking near them OR by smashing the sphere into them!");
    println!("  🌀 Walk through portals to change geometry (physics + visuals change)");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("Metatopia – Non-Euclidean Physics")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop).unwrap();
    let window = Arc::new(window);
    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked)).ok();
    window.set_cursor_visible(false);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface), force_fallback_adapter: false,
    }).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Metatopia"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
    }, None).await.unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    let mut demo = NonEuclideanDemo::new();
    let mut audio = GameAudio::new();
    if audio.is_some() { println!("🔊 Audio system initialized"); }
    else { println!("⚠️  No audio device found – continuing without sound"); }

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("NE Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/non_euclidean.wgsl").into()),
    });

    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cam"), contents: bytemuck::cast_slice(&[demo.camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cam_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry { binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None }],
    });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cam_bg"), layout: &cam_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }],
    });
    demo.camera_buffer = Some(cam_buf); demo.camera_bind_group = Some(cam_bg);

    let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene"), contents: bytemuck::cast_slice(&[demo.scene_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry { binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None }],
    });
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_bg"), layout: &scene_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: scene_buf.as_entire_binding() }],
    });
    demo.scene_buffer = Some(scene_buf); demo.scene_bind_group = Some(scene_bg);

    let mut depth_view = create_depth_texture(&device, config.width, config.height);

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&cam_bgl, &scene_bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, layout: Some(&pl),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
    });

    let mut frame_count = 0u32;
    let start_time = std::time::Instant::now();
    let mut last_time = start_time;
    const VERTS: u32 = 228;

    let _ = event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => match event {
                WinitWindowEvent::CloseRequested => target.exit(),
                WinitWindowEvent::Resized(s) => {
                    if s.width > 0 && s.height > 0 {
                        config.width = s.width; config.height = s.height;
                        surface.configure(&device, &config);
                        depth_view = create_depth_texture(&device, config.width, config.height);
                    }
                }
                WinitWindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(kc) = event.physical_key {
                        if kc == KeyCode::Escape { target.exit(); }
                        else { demo.handle_keyboard(kc, event.state == ElementState::Pressed); }
                    }
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let now = std::time::Instant::now();
                    let dt = (now - last_time).as_secs_f32().min(0.05);
                    last_time = now;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    demo.update(dt, elapsed);

                    // Process audio events
                    if let Some(ref mut a) = audio {
                        a.process(&demo.sound_events, demo.space_type, elapsed);
                    }
                    demo.sound_events.clear();

                    demo.update_camera_uniform(config.width as f32 / config.height as f32);
                    demo.update_scene_uniform(elapsed);

                    // Pass resolution to shader for HUD overlay
                    demo.scene_uniform.hud_info = [
                        config.width as f32, config.height as f32, 0.0, 0.0,
                    ];

                    if let Some(ref b) = demo.camera_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(&[demo.camera_uniform])); }
                    if let Some(ref b) = demo.scene_buffer  { queue.write_buffer(b, 0, bytemuck::cast_slice(&[demo.scene_uniform])); }

                    let output = match surface.get_current_texture() {
                        Ok(t) => t,
                        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                            surface.configure(&device, &config);
                            return;
                        }
                        Err(e) => { eprintln!("Surface error: {e:?}"); return; }
                    };
                    let cv = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let bg_col = match demo.space_type {
                            1 => wgpu::Color { r: 0.03, g: 0.015, b: 0.06, a: 1.0 },
                            2 => wgpu::Color { r: 0.06, g: 0.03, b: 0.015, a: 1.0 },
                            _ => wgpu::Color { r: 0.015, g: 0.02, b: 0.05, a: 1.0 },
                        };
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &cv, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(bg_col), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_view,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            occlusion_query_set: None, timestamp_writes: None,
                        });
                        rp.set_pipeline(&pipeline);
                        if let Some(ref bg) = demo.camera_bind_group { rp.set_bind_group(0, bg, &[]); }
                        if let Some(ref bg) = demo.scene_bind_group  { rp.set_bind_group(1, bg, &[]); }
                        rp.draw(0..VERTS, 0..1);
                    }
                    queue.submit(std::iter::once(enc.finish()));
                    output.present();

                    if frame_count % 60 == 0 {
                        let fps = frame_count as f32 / elapsed;
                        let phys = SpacePhysics::for_type(demo.space_type);
                        let dir = match ((demo.camera_rotation.0.to_degrees() + 360.0) % 360.0) as i32 {
                            0..=45 | 316..=360 => "N", 46..=135 => "E", 136..=225 => "S", 226..=315 => "W", _ => "?",
                        };
                        let ind = match demo.space_type { 1 => "🟣Hyp", 2 => "🟠Sph", _ => "🔵Euc" };
                        let cam = Vector3::new(demo.camera_position.x, demo.camera_position.y, demo.camera_position.z);
                        let sd = (demo.sphere.position - cam).magnitude();
                        let sv = demo.sphere.velocity.magnitude();
                        println!("{} {} | ✨{}/4 | ⚽{:.1}m v={:.1} | G={:.0} b={:.0}% | 🧭{} | {:.0}fps",
                            ind, phys.label, demo.collected, sd, sv, phys.gravity, phys.restitution*100.0, dir, fps);
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => { demo.handle_mouse_motion(delta.0, delta.1); }
            Event::AboutToWait => { window.request_redraw(); }
            _ => {}
        }
    });
}

fn main() { pollster::block_on(run()); }