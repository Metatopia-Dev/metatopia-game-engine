//! ⚡ CYBERSHOCK: Neon Grid Arena ⚡
//!
//! A fast-paced, retro-futuristic 3D First-Person Shooter built on Metatopia Engine.
//!
//! Features:
//!   - 3 Sci-Fi Weapons: Rapid Plasma Rifle, Scatter Shotgun, Piercing Railgun
//!   - Tactical Player Movement: WASD, Jump, Boost Dash (<kbd>Shift</kbd>)
//!   - Diverse Enemy AI: Neon Drones (swarm), Cyber Tanks (armored chasers), Phantom Snipers, Hex-Core Boss
//!   - Dynamic Combat: Hitmarkers, damage vignette, muzzle flash point lights, particle explosions
//!   - Cyberpunk PBR Shader: Reflective glowing grid floor, metallic cover pillars, sci-fi HUD
//!   - Procedural Audio: 10 synthetic sound effects synthesized at runtime via Rodio
//!   - Wave Survival: 10 escalating waves + Boss battle + Score combo multipliers
//!
//! Controls:
//!   - WASD            : Move
//!   - Mouse           : Look / Aim
//!   - Left Click      : Fire active weapon
//!   - 1 / 2 / 3 or Q  : Switch weapon (1: Plasma, 2: Shotgun, 3: Railgun)
//!   - R               : Reload weapon
//!   - Space           : Jump
//!   - Left Shift      : Cyber Dash / Boost
//!   - ESC             : Pause / Quit
//!   - Enter           : Start game / Restart after Game Over

use std::sync::Arc;
use std::time::Instant;
use std::io::Write;
use std::fs::File;
use std::path::PathBuf;
use cgmath::{InnerSpace, Point3, Vector3, Matrix4, Deg, perspective};
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent, ElementState, DeviceEvent, MouseButton as WinitMouseButton},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder as WinitWindowBuilder,
};
use wgpu::util::DeviceExt;
use metatopia_engine::audio::AudioEngine;

// ─── GPU Uniforms & Vertex ─────────────────────────────────────────────────

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
    params: [f32; 4],
    game_data: [f32; 4],
    extra0: [f32; 4],
    extra1: [f32; 4],
    extra2: [f32; 4],
    extra3: [f32; 4],
    hud_info: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GameVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 3],
    pbr: [f32; 4], // x=metallic, y=roughness, z=emission, w=type
}

impl GameVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2,
            3 => Float32x3,
            4 => Float32x4,
        ],
    };

    fn new(pos: [f32; 3], norm: [f32; 3], color: [f32; 3], pbr: [f32; 4]) -> Self {
        Self { position: pos, normal: norm, uv: [0.0, 0.0], color, pbr }
    }
}

// ─── Procedural Audio Synthesis ────────────────────────────────────────────

fn generate_sound_effects() -> PathBuf {
    let sound_dir = std::env::temp_dir().join("metatopia_cybershock_sounds");
    let _ = std::fs::create_dir_all(&sound_dir);

    fn write_wav(path: &PathBuf, samples: &[i16], sample_rate: u32) {
        if path.exists() { return; }
        if let Ok(mut file) = File::create(path) {
            let data_len = (samples.len() * 2) as u32;
            let file_len = 36 + data_len;
            let _ = file.write_all(b"RIFF");
            let _ = file.write_all(&file_len.to_le_bytes());
            let _ = file.write_all(b"WAVE");
            let _ = file.write_all(b"fmt ");
            let _ = file.write_all(&16u32.to_le_bytes()); // subchunk size
            let _ = file.write_all(&1u16.to_le_bytes());  // PCM
            let _ = file.write_all(&1u16.to_le_bytes());  // Mono
            let _ = file.write_all(&sample_rate.to_le_bytes());
            let _ = file.write_all(&(sample_rate * 2).to_le_bytes()); // Byte rate
            let _ = file.write_all(&2u16.to_le_bytes());  // Block align
            let _ = file.write_all(&16u16.to_le_bytes()); // Bits per sample
            let _ = file.write_all(b"data");
            let _ = file.write_all(&data_len.to_le_bytes());
            for s in samples {
                let _ = file.write_all(&s.to_le_bytes());
            }
        }
    }

    let sr = 44100u32;

    // 1. Plasma Rifle: Fast laser chirp (880Hz -> 220Hz)
    {
        let dur = 0.09;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let freq = 880.0 * (1.0 - t / dur).powi(2) + 200.0;
            let env = (1.0 - t / dur).powf(1.5);
            let s = (t * freq * 2.0 * std::f32::consts::PI).sin() * env;
            samples.push((s * 24000.0) as i16);
        }
        write_wav(&sound_dir.join("laser.wav"), &samples, sr);
    }

    // 2. Shotgun: Low punch + white noise blast
    {
        let dur = 0.18;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        let mut rng = 12345u32;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((rng >> 16) as f32 / 32768.0) - 1.0;
            let punch = (t * 80.0 * 2.0 * std::f32::consts::PI).sin();
            let env = (1.0 - t / dur).powf(2.0);
            let s = (noise * 0.7 + punch * 0.5) * env;
            samples.push((s * 28000.0) as i16);
        }
        write_wav(&sound_dir.join("shotgun.wav"), &samples, sr);
    }

    // 3. Railgun: High charge + supersonic beam snap
    {
        let dur = 0.35;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let freq = 1600.0 * (1.0 - t / dur).powf(3.0) + 120.0;
            let env = (1.0 - t / dur).powf(1.2);
            let s = ((t * freq * 2.0 * std::f32::consts::PI).sin()
                   + (t * (freq * 1.5) * 2.0 * std::f32::consts::PI).sin() * 0.5) * env;
            samples.push((s * 26000.0) as i16);
        }
        write_wav(&sound_dir.join("railgun.wav"), &samples, sr);
    }

    // 4. Hitmarker: Short crisp blip (1200Hz)
    {
        let dur = 0.04;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let env = 1.0 - t / dur;
            let s = (t * 1200.0 * 2.0 * std::f32::consts::PI).sin() * env;
            samples.push((s * 18000.0) as i16);
        }
        write_wav(&sound_dir.join("hit.wav"), &samples, sr);
    }

    // 5. Kill: Heavy bass drop + arpeggio
    {
        let dur = 0.25;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let freq = if t < 0.1 { 587.33 } else { 880.0 };
            let sub = (t * 60.0 * 2.0 * std::f32::consts::PI).sin();
            let tone = (t * freq * 2.0 * std::f32::consts::PI).sin();
            let env = (1.0 - t / dur).powf(1.8);
            let s = (tone * 0.6 + sub * 0.5) * env;
            samples.push((s * 26000.0) as i16);
        }
        write_wav(&sound_dir.join("kill.wav"), &samples, sr);
    }

    // 6. Player Hurt: Low crunch
    {
        let dur = 0.15;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let env = (1.0 - t / dur).powf(2.0);
            let s = (t * 90.0 * 2.0 * std::f32::consts::PI).sin() * env;
            samples.push((s * 25000.0) as i16);
        }
        write_wav(&sound_dir.join("hurt.wav"), &samples, sr);
    }

    // 7. Dash Boost: Air whoosh
    {
        let dur = 0.2;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        let mut rng = 98765u32;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((rng >> 16) as f32 / 32768.0) - 1.0;
            let env = (t / dur * 2.0).min(1.0) * (1.0 - t / dur);
            let s = noise * env;
            samples.push((s * 22000.0) as i16);
        }
        write_wav(&sound_dir.join("dash.wav"), &samples, sr);
    }

    // 8. Powerup: Ascending major triad
    {
        let dur = 0.3;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let freq = if t < 0.08 { 523.25 } else if t < 0.16 { 659.25 } else { 783.99 };
            let env = (1.0 - (t / dur)).powf(1.2);
            let s = (t * freq * 2.0 * std::f32::consts::PI).sin() * env;
            samples.push((s * 20000.0) as i16);
        }
        write_wav(&sound_dir.join("powerup.wav"), &samples, sr);
    }

    // 9. Wave Horn: Retro alarm
    {
        let dur = 0.45;
        let n = (sr as f32 * dur) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let freq = if t < 0.2 { 330.0 } else { 493.88 };
            let env = (1.0 - t / dur).powf(1.1);
            let s = (t * freq * 2.0 * std::f32::consts::PI).sin() * env;
            samples.push((s * 25000.0) as i16);
        }
        write_wav(&sound_dir.join("wave.wav"), &samples, sr);
    }

    sound_dir
}

// ─── Game Types & Structures ───────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GameState {
    Title,
    Playing,
    WaveClear,
    GameOver,
    Victory,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum WeaponType {
    PlasmaRifle = 0,
    ScatterShot = 1,
    Railgun = 2,
}

#[derive(Clone)]
struct Weapon {
    name: &'static str,
    weapon_type: WeaponType,
    damage: f32,
    fire_cooldown: f32,
    mag_size: u32,
    current_ammo: u32,
    reload_time: f32,
    last_fired: f32,
    pellets: u32,
    spread: f32,
    projectile_speed: f32,
    color: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EnemyType {
    Drone,   // Fast flying prism, swarms player
    Tank,    // Heavy armored cube, charges and soaks hits
    Sniper,  // Stationary teleporting turret, laser beam
    Boss,    // Quantum Hex-Core
}

struct Enemy {
    pos: Vector3<f32>,
    #[allow(dead_code)]
    vel: Vector3<f32>,
    enemy_type: EnemyType,
    hp: f32,
    max_hp: f32,
    speed: f32,
    attack_cooldown: f32,
    last_attack: f32,
    hit_flash: f32,
    orbit_angle: f32,
    active: bool,
}

struct Projectile {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    damage: f32,
    is_enemy: bool,
    lifetime: f32,
    color: [f32; 3],
    radius: f32,
}

struct Particle {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    lifetime: f32,
    max_life: f32,
    color: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PowerupType {
    Medkit,
    ShieldRecharge,
    AmmoRefill,
    Overcharge,
}

struct Powerup {
    pos: Vector3<f32>,
    powerup_type: PowerupType,
    #[allow(dead_code)]
    spawn_time: f32,
    active: bool,
}

struct Pillar {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

impl Pillar {
    fn contains(&self, p: Vector3<f32>, radius: f32) -> bool {
        p.x >= self.min.x - radius && p.x <= self.max.x + radius &&
        p.z >= self.min.z - radius && p.z <= self.max.z + radius
    }
}

// ─── Mesh Builder Helper ───────────────────────────────────────────────────

struct CyberMeshBuilder {
    verts: Vec<GameVertex>,
    indices: Vec<u32>,
}

impl CyberMeshBuilder {
    fn new() -> Self {
        Self { verts: Vec::new(), indices: Vec::new() }
    }

    fn add_quad(&mut self, p0: [f32;3], p1: [f32;3], p2: [f32;3], p3: [f32;3], norm: [f32;3], col: [f32;3], pbr: [f32;4]) {
        let base = self.verts.len() as u32;
        self.verts.push(GameVertex::new(p0, norm, col, pbr));
        self.verts.push(GameVertex::new(p1, norm, col, pbr));
        self.verts.push(GameVertex::new(p2, norm, col, pbr));
        self.verts.push(GameVertex::new(p3, norm, col, pbr));
        self.indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    fn add_box(&mut self, min: [f32;3], max: [f32;3], col: [f32;3], pbr: [f32;4]) {
        let (x0, y0, z0) = (min[0], min[1], min[2]);
        let (x1, y1, z1) = (max[0], max[1], max[2]);
        self.add_quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0], [0.0,0.0,-1.0], col, pbr); // Front
        self.add_quad([x1,y0,z1],[x0,y0,z1],[x0,y1,z1],[x1,y1,z1], [0.0,0.0, 1.0], col, pbr); // Back
        self.add_quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1], [0.0,1.0, 0.0], col, pbr); // Top
        self.add_quad([x0,y0,z1],[x1,y0,z1],[x1,y0,z0],[x0,y0,z0], [0.0,-1.0,0.0], col, pbr); // Bottom
        self.add_quad([x0,y0,z1],[x0,y0,z0],[x0,y1,z0],[x0,y1,z1], [-1.0,0.0,0.0], col, pbr); // Left
        self.add_quad([x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0], [ 1.0,0.0,0.0], col, pbr); // Right
    }

    fn add_octahedron(&mut self, center: Vector3<f32>, size: f32, col: [f32; 3], pbr: [f32; 4]) {
        let top = [center.x, center.y + size, center.z];
        let bot = [center.x, center.y - size, center.z];
        let px = [center.x + size, center.y, center.z];
        let nx = [center.x - size, center.y, center.z];
        let pz = [center.x, center.y, center.z + size];
        let nz = [center.x, center.y, center.z - size];

        let tris = [
            (top, px, pz), (top, pz, nx), (top, nx, nz), (top, nz, px),
            (bot, pz, px), (bot, nx, pz), (bot, nz, nx), (bot, px, nz),
        ];
        for (a, b, c) in tris {
            let base = self.verts.len() as u32;
            let norm = [0.0, 1.0, 0.0];
            self.verts.push(GameVertex::new(a, norm, col, pbr));
            self.verts.push(GameVertex::new(b, norm, col, pbr));
            self.verts.push(GameVertex::new(c, norm, col, pbr));
            self.indices.extend_from_slice(&[base, base+1, base+2]);
        }
    }

    fn finish(self) -> (Vec<GameVertex>, Vec<u32>) {
        (self.verts, self.indices)
    }
}

// ─── FPS Player State ──────────────────────────────────────────────────────

struct Player {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    yaw: f32,
    pitch: f32,
    hp: f32,
    max_hp: f32,
    shield: f32,
    max_shield: f32,
    shield_regen_delay: f32,
    speed: f32,
    is_grounded: bool,
    dash_cooldown_timer: f32,
    dash_active_timer: f32,
    weapons: Vec<Weapon>,
    current_weapon: usize,
    reloading_timer: f32,
    hitmarker_timer: f32,
    dmg_flash_timer: f32,
    score: u32,
    combo: u32,
    combo_timer: f32,
    overcharge_timer: f32,
}

impl Player {
    fn new() -> Self {
        Self {
            pos: Vector3::new(0.0, 1.7, 16.0),
            vel: Vector3::new(0.0, 0.0, 0.0),
            yaw: 3.14159,
            pitch: 0.0,
            hp: 100.0,
            max_hp: 100.0,
            shield: 50.0,
            max_shield: 50.0,
            shield_regen_delay: 0.0,
            speed: 12.0,
            is_grounded: true,
            dash_cooldown_timer: 0.0,
            dash_active_timer: 0.0,
            weapons: vec![
                Weapon {
                    name: "PLASMA RIFLE",
                    weapon_type: WeaponType::PlasmaRifle,
                    damage: 22.0,
                    fire_cooldown: 0.12,
                    mag_size: 30,
                    current_ammo: 30,
                    reload_time: 1.2,
                    last_fired: 0.0,
                    pellets: 1,
                    spread: 0.015,
                    projectile_speed: 65.0,
                    color: [0.0, 0.9, 1.0],
                },
                Weapon {
                    name: "SCATTER CANNON",
                    weapon_type: WeaponType::ScatterShot,
                    damage: 16.0,
                    fire_cooldown: 0.65,
                    mag_size: 8,
                    current_ammo: 8,
                    reload_time: 1.8,
                    last_fired: 0.0,
                    pellets: 8,
                    spread: 0.09,
                    projectile_speed: 55.0,
                    color: [1.0, 0.1, 0.7],
                },
                Weapon {
                    name: "HEAVY RAILGUN",
                    weapon_type: WeaponType::Railgun,
                    damage: 150.0,
                    fire_cooldown: 1.1,
                    mag_size: 4,
                    current_ammo: 4,
                    reload_time: 2.2,
                    last_fired: 0.0,
                    pellets: 1,
                    spread: 0.001,
                    projectile_speed: 180.0,
                    color: [1.0, 0.85, 0.1],
                },
            ],
            current_weapon: 0,
            reloading_timer: 0.0,
            hitmarker_timer: 0.0,
            dmg_flash_timer: 0.0,
            score: 0,
            combo: 1,
            combo_timer: 0.0,
            overcharge_timer: 0.0,
        }
    }

    fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    fn right(&self) -> Vector3<f32> {
        let fwd = self.forward();
        Vector3::new(-fwd.z, 0.0, fwd.x).normalize()
    }

    fn take_damage(&mut self, dmg: f32) {
        self.dmg_flash_timer = 1.0;
        self.shield_regen_delay = 3.5;
        if self.shield > 0.0 {
            if self.shield >= dmg {
                self.shield -= dmg;
            } else {
                let rem = dmg - self.shield;
                self.shield = 0.0;
                self.hp = (self.hp - rem).max(0.0);
            }
        } else {
            self.hp = (self.hp - dmg).max(0.0);
        }
    }
}

// ─── Main Game Manager ─────────────────────────────────────────────────────

struct CyberShockGame {
    state: GameState,
    player: Player,
    enemies: Vec<Enemy>,
    projectiles: Vec<Projectile>,
    particles: Vec<Particle>,
    powerups: Vec<Powerup>,
    pillars: Vec<Pillar>,
    current_wave: u32,
    wave_state_timer: f32,
    muzzle_flash_timer: f32,
    active_light_pos: Vector3<f32>,
    active_light_color: [f32; 3],
    active_light_intensity: f32,
    audio: AudioEngine,
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_delta: (f32, f32),
    screen_shake: f32,
}

impl CyberShockGame {
    fn new() -> Self {
        let mut audio = AudioEngine::new();
        let sound_dir = generate_sound_effects();
        audio.preload("laser", sound_dir.join("laser.wav").to_str().unwrap());
        audio.preload("shotgun", sound_dir.join("shotgun.wav").to_str().unwrap());
        audio.preload("railgun", sound_dir.join("railgun.wav").to_str().unwrap());
        audio.preload("hit", sound_dir.join("hit.wav").to_str().unwrap());
        audio.preload("kill", sound_dir.join("kill.wav").to_str().unwrap());
        audio.preload("hurt", sound_dir.join("hurt.wav").to_str().unwrap());
        audio.preload("dash", sound_dir.join("dash.wav").to_str().unwrap());
        audio.preload("powerup", sound_dir.join("powerup.wav").to_str().unwrap());
        audio.preload("wave", sound_dir.join("wave.wav").to_str().unwrap());

        // Cover Pillars
        let pillars = vec![
            Pillar { min: Vector3::new(-9.0, 0.0, -9.0), max: Vector3::new(-6.0, 5.0, -6.0) },
            Pillar { min: Vector3::new(6.0, 0.0, -9.0), max: Vector3::new(9.0, 5.0, -6.0) },
            Pillar { min: Vector3::new(-9.0, 0.0, 6.0), max: Vector3::new(-6.0, 5.0, 9.0) },
            Pillar { min: Vector3::new(6.0, 0.0, 6.0), max: Vector3::new(9.0, 5.0, 9.0) },
            // Central low barricades
            Pillar { min: Vector3::new(-2.0, 0.0, -2.0), max: Vector3::new(2.0, 1.8, 2.0) },
        ];

        Self {
            state: GameState::Playing,
            player: Player::new(),
            enemies: Vec::new(),
            projectiles: Vec::new(),
            particles: Vec::new(),
            powerups: Vec::new(),
            pillars,
            current_wave: 1,
            wave_state_timer: 1.5,
            muzzle_flash_timer: 0.0,
            active_light_pos: Vector3::new(0.0, 2.0, 0.0),
            active_light_color: [0.0, 0.8, 1.0],
            active_light_intensity: 0.0,
            audio,
            keys_held: std::collections::HashSet::new(),
            mouse_delta: (0.0, 0.0),
            screen_shake: 0.0,
        }
    }

    fn spawn_wave(&mut self, wave: u32) {
        self.enemies.clear();
        self.audio.play("wave");
        println!("\n⚡ ═══ WAVE {} INITIATED ═══ ⚡", wave);

        let drone_count = (3 + wave * 2).min(14);
        let tank_count = if wave >= 2 { (wave / 2).min(5) } else { 0 };
        let sniper_count = if wave >= 3 { (wave / 3).min(4) } else { 0 };
        let is_boss_wave = wave % 5 == 0;

        let arena_radius = 18.0;

        // Spawn Drones
        for i in 0..drone_count {
            let angle = (i as f32 / drone_count as f32) * std::f32::consts::PI * 2.0;
            let pos = Vector3::new(angle.cos() * arena_radius, 2.2 + (i % 3) as f32 * 0.8, angle.sin() * arena_radius);
            self.enemies.push(Enemy {
                pos,
                vel: Vector3::new(0.0, 0.0, 0.0),
                enemy_type: EnemyType::Drone,
                hp: 40.0 + wave as f32 * 10.0,
                max_hp: 40.0 + wave as f32 * 10.0,
                speed: 6.5 + (wave as f32 * 0.4).min(4.0),
                attack_cooldown: 1.8,
                last_attack: i as f32 * 0.3,
                hit_flash: 0.0,
                orbit_angle: angle,
                active: true,
            });
        }

        // Spawn Tanks
        for i in 0..tank_count {
            let angle = ((i as f32 + 0.5) / tank_count as f32) * std::f32::consts::PI * 2.0;
            let pos = Vector3::new(angle.cos() * arena_radius, 1.2, angle.sin() * arena_radius);
            self.enemies.push(Enemy {
                pos,
                vel: Vector3::new(0.0, 0.0, 0.0),
                enemy_type: EnemyType::Tank,
                hp: 150.0 + wave as f32 * 40.0,
                max_hp: 150.0 + wave as f32 * 40.0,
                speed: 3.5 + (wave as f32 * 0.2).min(2.5),
                attack_cooldown: 1.2,
                last_attack: 0.0,
                hit_flash: 0.0,
                orbit_angle: angle,
                active: true,
            });
        }

        // Spawn Snipers
        for i in 0..sniper_count {
            let corner = match i % 4 {
                0 => Vector3::new(-16.0, 1.5, -16.0),
                1 => Vector3::new(16.0, 1.5, -16.0),
                2 => Vector3::new(-16.0, 1.5, 16.0),
                _ => Vector3::new(16.0, 1.5, 16.0),
            };
            self.enemies.push(Enemy {
                pos: corner,
                vel: Vector3::new(0.0, 0.0, 0.0),
                enemy_type: EnemyType::Sniper,
                hp: 80.0 + wave as f32 * 15.0,
                max_hp: 80.0 + wave as f32 * 15.0,
                speed: 0.0,
                attack_cooldown: 3.0,
                last_attack: i as f32 * 0.5,
                hit_flash: 0.0,
                orbit_angle: 0.0,
                active: true,
            });
        }

        // Spawn Boss on wave 5 & 10
        if is_boss_wave {
            self.enemies.push(Enemy {
                pos: Vector3::new(0.0, 4.0, -12.0),
                vel: Vector3::new(0.0, 0.0, 0.0),
                enemy_type: EnemyType::Boss,
                hp: 600.0 + wave as f32 * 150.0,
                max_hp: 600.0 + wave as f32 * 150.0,
                speed: 3.0,
                attack_cooldown: 0.8,
                last_attack: 0.0,
                hit_flash: 0.0,
                orbit_angle: 0.0,
                active: true,
            });
            println!("🚨 ═══ QUANTUM HEX-CORE BOSS DETECTED! ═══ 🚨");
        }
    }

    fn spawn_particles(&mut self, pos: Vector3<f32>, color: [f32; 3], count: usize, speed: f32) {
        let mut rng = (pos.x * 100.0 + pos.z * 50.0).abs() as u32;
        for _ in 0..count {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let rx = ((rng >> 16) as f32 / 32768.0) - 1.0;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let ry = (rng >> 16) as f32 / 65536.0;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let rz = ((rng >> 16) as f32 / 32768.0) - 1.0;

            let vel = Vector3::new(rx, ry + 0.2, rz).normalize() * (speed * (0.5 + ry * 0.5));
            self.particles.push(Particle {
                pos,
                vel,
                lifetime: 0.0,
                max_life: 0.35 + ry * 0.3,
                color,
            });
        }
    }

    fn try_fire(&mut self, time: f32) {
        if self.state != GameState::Playing || self.player.reloading_timer > 0.0 {
            return;
        }

        let fwd = self.player.forward();
        let right = self.player.right();
        let up = right.cross(fwd).normalize();
        let spawn_pos = self.player.pos + fwd * 0.8 + right * 0.25 - up * 0.2;
        let is_overcharge = self.player.overcharge_timer > 0.0;
        let dmg_mult = if is_overcharge { 2.0 } else { 1.0 };

        let weapon_idx = self.player.current_weapon;
        let weapon = &mut self.player.weapons[weapon_idx];

        if time - weapon.last_fired < weapon.fire_cooldown {
            return;
        }

        if weapon.current_ammo == 0 {
            self.start_reload();
            return;
        }

        weapon.current_ammo -= 1;
        weapon.last_fired = time;
        self.muzzle_flash_timer = 0.12;

        let sound_name = match weapon.weapon_type {
            WeaponType::PlasmaRifle => "laser",
            WeaponType::ScatterShot => "shotgun",
            WeaponType::Railgun => "railgun",
        };
        let weapon_type = weapon.weapon_type;
        let pellets = weapon.pellets;
        let spread = weapon.spread;
        let projectile_speed = weapon.projectile_speed;
        let damage = weapon.damage;
        let color = weapon.color;
        let is_empty = weapon.current_ammo == 0;

        self.audio.play(sound_name);

        let mut rng = (time * 1000.0) as u32;

        for _ in 0..pellets {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let spread_x = (((rng >> 16) as f32 / 32768.0) - 1.0) * spread;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let spread_y = (((rng >> 16) as f32 / 32768.0) - 1.0) * spread;

            let dir = (fwd + right * spread_x + up * spread_y).normalize();

            self.projectiles.push(Projectile {
                pos: spawn_pos,
                vel: dir * projectile_speed,
                damage: damage * dmg_mult,
                is_enemy: false,
                lifetime: 2.0,
                color,
                radius: match weapon_type {
                    WeaponType::PlasmaRifle => 0.18,
                    WeaponType::ScatterShot => 0.12,
                    WeaponType::Railgun => 0.35,
                },
            });
        }

        self.screen_shake = match weapon_type {
            WeaponType::PlasmaRifle => 0.05,
            WeaponType::ScatterShot => 0.15,
            WeaponType::Railgun => 0.3,
        };

        if is_empty {
            self.start_reload();
        }
    }

    fn start_reload(&mut self) {
        let weapon = &self.player.weapons[self.player.current_weapon];
        if weapon.current_ammo < weapon.mag_size && self.player.reloading_timer <= 0.0 {
            self.player.reloading_timer = weapon.reload_time;
        }
    }

    fn dash(&mut self) {
        if self.player.dash_cooldown_timer <= 0.0 && self.player.dash_active_timer <= 0.0 {
            self.player.dash_active_timer = 0.18;
            self.player.dash_cooldown_timer = 1.2;
            self.audio.play("dash");

            let fwd = self.player.forward();
            let flat_fwd = Vector3::new(fwd.x, 0.0, fwd.z).normalize();
            self.player.vel += flat_fwd * 35.0;
        }
    }

    fn update(&mut self, dt: f32, time: f32) {
        if self.state == GameState::WaveClear {
            self.wave_state_timer -= dt;
            if self.wave_state_timer <= 0.0 {
                self.current_wave += 1;
                if self.current_wave > 10 {
                    self.state = GameState::Victory;
                    println!("🎉 CONGRATULATIONS! You cleared all 10 waves!");
                } else {
                    self.state = GameState::Playing;
                    self.spawn_wave(self.current_wave);
                }
            }
            return;
        }

        if self.state != GameState::Playing {
            return;
        }

        // ── 1. Update Player ──────────────────────────────────────────────
        // Mouse Look
        let sensitivity = 0.0022;
        self.player.yaw += self.mouse_delta.0 * sensitivity;
        self.player.pitch -= self.mouse_delta.1 * sensitivity;
        self.player.pitch = self.player.pitch.clamp(-1.45, 1.45);
        self.mouse_delta = (0.0, 0.0);

        // Movement input
        let fwd = self.player.forward();
        let flat_fwd = Vector3::new(fwd.x, 0.0, fwd.z).normalize();
        let right = self.player.right();

        let mut move_dir = Vector3::new(0.0, 0.0, 0.0);
        if self.keys_held.contains(&KeyCode::KeyW) { move_dir += flat_fwd; }
        if self.keys_held.contains(&KeyCode::KeyS) { move_dir -= flat_fwd; }
        if self.keys_held.contains(&KeyCode::KeyD) { move_dir += right; }
        if self.keys_held.contains(&KeyCode::KeyA) { move_dir -= right; }

        if move_dir.magnitude2() > 0.001 {
            move_dir = move_dir.normalize();
        }

        let speed = self.player.speed;
        self.player.vel.x = move_dir.x * speed;
        self.player.vel.z = move_dir.z * speed;

        // Jump & Gravity
        if self.keys_held.contains(&KeyCode::Space) && self.player.is_grounded {
            self.player.vel.y = 8.5;
            self.player.is_grounded = false;
        }

        self.player.vel.y -= 22.0 * dt; // Gravity
        self.player.pos += self.player.vel * dt;

        // Floor collision
        if self.player.pos.y <= 1.7 {
            self.player.pos.y = 1.7;
            self.player.vel.y = 0.0;
            self.player.is_grounded = true;
        }

        // Arena Boundaries (-20 to 20)
        let bound = 19.0;
        self.player.pos.x = self.player.pos.x.clamp(-bound, bound);
        self.player.pos.z = self.player.pos.z.clamp(-bound, bound);

        // Pillar Collisions
        for p in &self.pillars {
            if p.contains(self.player.pos, 0.6) {
                // Push player out
                let center_x = (p.min.x + p.max.x) * 0.5;
                let center_z = (p.min.z + p.max.z) * 0.5;
                let diff_x = self.player.pos.x - center_x;
                let diff_z = self.player.pos.z - center_z;
                if diff_x.abs() > diff_z.abs() {
                    self.player.pos.x = center_x + diff_x.signum() * ((p.max.x - p.min.x) * 0.5 + 0.65);
                } else {
                    self.player.pos.z = center_z + diff_z.signum() * ((p.max.z - p.min.z) * 0.5 + 0.65);
                }
            }
        }

        // Shield Regen
        if self.player.shield_regen_delay > 0.0 {
            self.player.shield_regen_delay -= dt;
        } else if self.player.shield < self.player.max_shield {
            self.player.shield = (self.player.shield + 20.0 * dt).min(self.player.max_shield);
        }

        // Timers
        if self.player.reloading_timer > 0.0 {
            self.player.reloading_timer -= dt;
            if self.player.reloading_timer <= 0.0 {
                let idx = self.player.current_weapon;
                self.player.weapons[idx].current_ammo = self.player.weapons[idx].mag_size;
            }
        }
        if self.player.dash_cooldown_timer > 0.0 { self.player.dash_cooldown_timer -= dt; }
        if self.player.dash_active_timer > 0.0 { self.player.dash_active_timer -= dt; }
        if self.player.hitmarker_timer > 0.0 { self.player.hitmarker_timer -= dt * 4.0; }
        if self.player.dmg_flash_timer > 0.0 { self.player.dmg_flash_timer -= dt * 3.0; }
        if self.player.overcharge_timer > 0.0 { self.player.overcharge_timer -= dt; }
        if self.muzzle_flash_timer > 0.0 { self.muzzle_flash_timer -= dt * 8.0; }
        if self.screen_shake > 0.0 { self.screen_shake -= dt * 4.0; }

        // Combo decay
        if self.player.combo_timer > 0.0 {
            self.player.combo_timer -= dt;
            if self.player.combo_timer <= 0.0 {
                self.player.combo = 1;
            }
        }

        // ── 2. Update Enemies ─────────────────────────────────────────────
        let player_pos = self.player.pos;
        let mut enemy_projectiles = Vec::new();

        for e in &mut self.enemies {
            if !e.active { continue; }
            if e.hit_flash > 0.0 { e.hit_flash -= dt * 4.0; }

            let to_player = player_pos - e.pos;
            let dist = to_player.magnitude();

            match e.enemy_type {
                EnemyType::Drone => {
                    // Circle and swoop towards player
                    e.orbit_angle += dt * 1.5;
                    let target_pos = player_pos + Vector3::new(
                        e.orbit_angle.cos() * 7.0,
                        1.5 + (e.orbit_angle * 2.0).sin() * 0.8,
                        e.orbit_angle.sin() * 7.0
                    );
                    let move_vec = target_pos - e.pos;
                    e.pos += move_vec * (dt * 3.0).min(1.0);

                    // Shoot laser bolt
                    if time - e.last_attack > e.attack_cooldown {
                        e.last_attack = time;
                        let dir = (player_pos - e.pos).normalize();
                        enemy_projectiles.push(Projectile {
                            pos: e.pos,
                            vel: dir * 16.0,
                            damage: 12.0,
                            is_enemy: true,
                            lifetime: 4.0,
                            color: [1.0, 0.2, 0.3],
                            radius: 0.2,
                        });
                    }
                }
                EnemyType::Tank => {
                    // Charge directly on ground
                    if dist > 1.2 {
                        let dir = Vector3::new(to_player.x, 0.0, to_player.z).normalize();
                        e.pos += dir * e.speed * dt;
                    } else if time - e.last_attack > e.attack_cooldown {
                        // Melee slam
                        e.last_attack = time;
                        self.player.take_damage(25.0);
                        self.audio.play("hurt");
                    }
                }
                EnemyType::Sniper => {
                    // Stationary / aim laser then blast
                    if time - e.last_attack > e.attack_cooldown {
                        e.last_attack = time;
                        let dir = (player_pos - e.pos).normalize();
                        enemy_projectiles.push(Projectile {
                            pos: e.pos,
                            vel: dir * 40.0,
                            damage: 30.0,
                            is_enemy: true,
                            lifetime: 3.0,
                            color: [1.0, 0.8, 0.1],
                            radius: 0.25,
                        });
                    }
                }
                EnemyType::Boss => {
                    // Boss moves smoothly and shoots spiral barrages
                    e.orbit_angle += dt * 0.8;
                    let target_pos = Vector3::new(e.orbit_angle.cos() * 12.0, 4.0, e.orbit_angle.sin() * 12.0);
                    e.pos += (target_pos - e.pos) * dt * 1.2;

                    if time - e.last_attack > e.attack_cooldown {
                        e.last_attack = time;
                        // 3-way spread attack
                        let fwd = (player_pos - e.pos).normalize();
                        let r = Vector3::new(-fwd.z, 0.0, fwd.x).normalize();
                        for angle_offset in [-0.25, 0.0, 0.25] {
                            let dir = (fwd + r * angle_offset).normalize();
                            enemy_projectiles.push(Projectile {
                                pos: e.pos,
                                vel: dir * 18.0,
                                damage: 18.0,
                                is_enemy: true,
                                lifetime: 4.0,
                                color: [0.9, 0.1, 0.9],
                                radius: 0.35,
                            });
                        }
                    }
                }
            }
        }
        self.projectiles.extend(enemy_projectiles);

        // ── 3. Update Projectiles ─────────────────────────────────────────
        let mut new_particles = Vec::new();
        let mut spawn_powerups = Vec::new();

        for p in &mut self.projectiles {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 { continue; }

            let next_pos = p.pos + p.vel * dt;

            // Pillar hit
            let mut hit_pillar = false;
            for pil in &self.pillars {
                if pil.contains(next_pos, p.radius) {
                    hit_pillar = true;
                    break;
                }
            }
            if hit_pillar || next_pos.y <= 0.0 || next_pos.x.abs() > 20.0 || next_pos.z.abs() > 20.0 {
                p.lifetime = 0.0;
                new_particles.push((p.pos, p.color, 6, 4.0));
                continue;
            }

            if p.is_enemy {
                // Hit Player
                let dist_to_player = (player_pos - next_pos).magnitude();
                if dist_to_player < 0.9 {
                    p.lifetime = 0.0;
                    self.player.take_damage(p.damage);
                    self.audio.play("hurt");
                    new_particles.push((p.pos, [1.0, 0.1, 0.2], 8, 5.0));
                }
            } else {
                // Hit Enemies
                for e in &mut self.enemies {
                    if !e.active { continue; }
                    let enemy_hit_radius = match e.enemy_type {
                        EnemyType::Drone => 0.7,
                        EnemyType::Tank => 1.2,
                        EnemyType::Sniper => 0.8,
                        EnemyType::Boss => 2.2,
                    };
                    let dist = (e.pos - next_pos).magnitude();
                    if dist < enemy_hit_radius {
                        p.lifetime = 0.0;
                        e.hp -= p.damage;
                        e.hit_flash = 1.0;
                        self.player.hitmarker_timer = 1.0;
                        self.audio.play("hit");
                        new_particles.push((p.pos, p.color, 7, 6.0));

                        if e.hp <= 0.0 {
                            e.active = false;
                            self.audio.play("kill");
                            self.player.combo = (self.player.combo + 1).min(8);
                            self.player.combo_timer = 4.0;
                            let points = match e.enemy_type {
                                EnemyType::Drone => 150,
                                EnemyType::Tank => 350,
                                EnemyType::Sniper => 250,
                                EnemyType::Boss => 2000,
                            };
                            self.player.score += points * self.player.combo;
                            new_particles.push((e.pos, [0.0, 1.0, 1.0], 20, 9.0));

                            // Random powerup drop (35% chance)
                            let mut rng = (time * 5000.0) as u32;
                            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                            if (rng % 100) < 35 {
                                let ptype = match (rng / 100) % 4 {
                                    0 => PowerupType::Medkit,
                                    1 => PowerupType::ShieldRecharge,
                                    2 => PowerupType::AmmoRefill,
                                    _ => PowerupType::Overcharge,
                                };
                                spawn_powerups.push(Powerup {
                                    pos: Vector3::new(e.pos.x, 0.6, e.pos.z),
                                    powerup_type: ptype,
                                    spawn_time: time,
                                    active: true,
                                });
                            }
                        }
                        break;
                    }
                }
            }

            p.pos = next_pos;
        }

        self.projectiles.retain(|p| p.lifetime > 0.0);
        for (pos, col, count, spd) in new_particles {
            self.spawn_particles(pos, col, count, spd);
        }
        self.powerups.extend(spawn_powerups);

        // ── 4. Update Powerups ────────────────────────────────────────────
        let mut pw_particles = Vec::new();
        for pw in &mut self.powerups {
            if !pw.active { continue; }
            let dist = (self.player.pos - pw.pos).magnitude();
            if dist < 1.4 {
                pw.active = false;
                self.audio.play("powerup");
                match pw.powerup_type {
                    PowerupType::Medkit => {
                        self.player.hp = (self.player.hp + 40.0).min(self.player.max_hp);
                        println!("💚 MEDKIT: +40 HP");
                    }
                    PowerupType::ShieldRecharge => {
                        self.player.shield = self.player.max_shield;
                        println!("🛡️ SHIELD: Restored to 100%");
                    }
                    PowerupType::AmmoRefill => {
                        for w in &mut self.player.weapons {
                            w.current_ammo = w.mag_size;
                        }
                        println!("⚡ AMMO REFILL: All weapons maxed!");
                    }
                    PowerupType::Overcharge => {
                        self.player.overcharge_timer = 8.0;
                        println!("🔥 OVERCHARGE: 2x Damage for 8s!");
                    }
                }
                pw_particles.push((pw.pos, [1.0, 1.0, 0.0], 15, 6.0));
            }
        }
        for (pos, col, cnt, spd) in pw_particles {
            self.spawn_particles(pos, col, cnt, spd);
        }
        self.powerups.retain(|p| p.active);

        // ── 5. Update Particles ───────────────────────────────────────────
        for part in &mut self.particles {
            part.lifetime += dt;
            part.vel.y -= 15.0 * dt;
            part.pos += part.vel * dt;
            if part.pos.y < 0.05 {
                part.pos.y = 0.05;
                part.vel.y = -part.vel.y * 0.4;
            }
        }
        self.particles.retain(|p| p.lifetime < p.max_life);

        // ── 6. Check Wave Completion / Player Death ───────────────────────
        if self.player.hp <= 0.0 {
            self.state = GameState::GameOver;
            println!("\n💀 ═══ CRITICAL FAILURE: SYSTEM COMPROMISED ═══ 💀");
            println!("Final Score: {} | Reached Wave: {}", self.player.score, self.current_wave);
            println!("Press ENTER to Reboot, ESC to Quit.");
        } else if self.enemies.iter().all(|e| !e.active) && self.state == GameState::Playing {
            self.state = GameState::WaveClear;
            self.wave_state_timer = 2.0;
            println!("✨ WAVE {} CLEARED! +500 Bonus", self.current_wave);
            self.player.score += 500 * self.current_wave;
        }
    }

    // ── 7. Build Dynamic Vertex Mesh ──────────────────────────────────────
    fn build_scene_mesh(&self, time: f32) -> (Vec<GameVertex>, Vec<u32>) {
        let mut builder = CyberMeshBuilder::new();

        // ── A. Arena Floor (40x40) ────────────────────────────────────────
        builder.add_quad(
            [-22.0, 0.0, -22.0], [ 22.0, 0.0, -22.0],
            [ 22.0, 0.0,  22.0], [-22.0, 0.0,  22.0],
            [0.0, 1.0, 0.0], [0.02, 0.05, 0.08], [0.9, 0.2, 0.0, 1.0] // pbr.w=1 (floor)
        );

        // ── B. Boundary Walls & Glowing Trims ──────────────────────────────
        let wall_h = 6.0;
        let w_bound = 21.0;
        // North Wall
        builder.add_quad([-w_bound, 0.0, -w_bound], [w_bound, 0.0, -w_bound], [w_bound, wall_h, -w_bound], [-w_bound, wall_h, -w_bound], [0.0,0.0,1.0], [0.1,0.1,0.2], [0.8,0.3,0.0,2.0]);
        // South Wall
        builder.add_quad([w_bound, 0.0, w_bound], [-w_bound, 0.0, w_bound], [-w_bound, wall_h, w_bound], [w_bound, wall_h, w_bound], [0.0,0.0,-1.0], [0.1,0.1,0.2], [0.8,0.3,0.0,2.0]);
        // West Wall
        builder.add_quad([-w_bound, 0.0, w_bound], [-w_bound, 0.0, -w_bound], [-w_bound, wall_h, -w_bound], [-w_bound, wall_h, w_bound], [1.0,0.0,0.0], [0.1,0.1,0.2], [0.8,0.3,0.0,2.0]);
        // East Wall
        builder.add_quad([w_bound, 0.0, -w_bound], [w_bound, 0.0, w_bound], [w_bound, wall_h, w_bound], [w_bound, wall_h, -w_bound], [-1.0,0.0,0.0], [0.1,0.1,0.2], [0.8,0.3,0.0,2.0]);

        // ── C. Cover Pillars ──────────────────────────────────────────────
        for p in &self.pillars {
            builder.add_box([p.min.x, p.min.y, p.min.z], [p.max.x, p.max.y, p.max.z], [0.15, 0.2, 0.3], [0.7, 0.3, 0.0, 2.0]);
        }

        // ── D. Enemies ────────────────────────────────────────────────────
        for e in &self.enemies {
            if !e.active { continue; }
            let flash = e.hit_flash;
            let base_col = match e.enemy_type {
                EnemyType::Drone => [1.0, 0.2, 0.3],
                EnemyType::Tank => [0.8, 0.4, 0.1],
                EnemyType::Sniper => [0.9, 0.8, 0.1],
                EnemyType::Boss => [0.8, 0.1, 0.9],
            };
            let col = [
                base_col[0] * (1.0 - flash) + flash,
                base_col[1] * (1.0 - flash) + flash,
                base_col[2] * (1.0 - flash) + flash,
            ];

            match e.enemy_type {
                EnemyType::Drone => {
                    builder.add_octahedron(e.pos, 0.65, col, [0.8, 0.2, 1.5, 3.0]);
                }
                EnemyType::Tank => {
                    let s = 1.0;
                    builder.add_box(
                        [e.pos.x - s, e.pos.y - s, e.pos.z - s],
                        [e.pos.x + s, e.pos.y + s, e.pos.z + s],
                        col, [0.9, 0.1, 0.8, 3.0]
                    );
                }
                EnemyType::Sniper => {
                    builder.add_octahedron(e.pos + Vector3::new(0.0, 0.5, 0.0), 0.7, col, [0.9, 0.1, 2.5, 3.0]);
                }
                EnemyType::Boss => {
                    // Massive central octahedron + orbiting shield blocks
                    builder.add_octahedron(e.pos, 2.0, col, [0.9, 0.1, 3.0, 3.0]);
                    let rot = time * 2.0;
                    for i in 0..4 {
                        let a = rot + (i as f32) * std::f32::consts::PI * 0.5;
                        let sat_pos = e.pos + Vector3::new(a.cos() * 3.5, (a * 3.0).sin() * 0.8, a.sin() * 3.5);
                        builder.add_box(
                            [sat_pos.x - 0.4, sat_pos.y - 0.4, sat_pos.z - 0.4],
                            [sat_pos.x + 0.4, sat_pos.y + 0.4, sat_pos.z + 0.4],
                            [0.0, 0.9, 1.0], [0.9, 0.1, 4.0, 3.0]
                        );
                    }
                }
            }
        }

        // ── E. Projectiles & Laser Tracers ────────────────────────────────
        for p in &self.projectiles {
            let fwd = p.vel.normalize();
            let p_start = p.pos - fwd * (p.radius * 2.0);
            let p_end = p.pos + fwd * (p.radius * 2.0);
            builder.add_box(
                [p_start.x - p.radius, p_start.y - p.radius, p_start.z - p.radius],
                [p_end.x + p.radius, p_end.y + p.radius, p_end.z + p.radius],
                p.color, [0.1, 0.1, 5.0, 0.0]
            );
        }

        // ── F. Powerup Crystals ───────────────────────────────────────────
        for pw in &self.powerups {
            if !pw.active { continue; }
            let hover_y = 0.6 + (time * 4.0).sin() * 0.2;
            let pos = Vector3::new(pw.pos.x, hover_y, pw.pos.z);
            let col = match pw.powerup_type {
                PowerupType::Medkit => [0.1, 1.0, 0.4],
                PowerupType::ShieldRecharge => [0.2, 0.7, 1.0],
                PowerupType::AmmoRefill => [1.0, 0.8, 0.1],
                PowerupType::Overcharge => [1.0, 0.1, 0.7],
            };
            builder.add_octahedron(pos, 0.45, col, [0.2, 0.1, 4.5, 4.0]);
        }

        // ── G. Particle Sparks ────────────────────────────────────────────
        for part in &self.particles {
            let s = (1.0 - part.lifetime / part.max_life) * 0.12;
            builder.add_box(
                [part.pos.x - s, part.pos.y - s, part.pos.z - s],
                [part.pos.x + s, part.pos.y + s, part.pos.z + s],
                part.color, [0.0, 0.1, 6.0, 0.0]
            );
        }

        builder.finish()
    }
}

// ─── Entry Point & Event Loop ──────────────────────────────────────────────

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WinitWindowBuilder::new()
            .with_title("CYBERSHOCK: Neon Grid Arena — Metatopia FPS")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)
            .unwrap()
    );

    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked));
    window.set_cursor_visible(false);

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
    ).expect("No GPU adapter found");

    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("CyberShock Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None)
    ).unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    let mut depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default());

    // Uniform Buffers
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Camera Buffer"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scene Buffer"),
        size: std::mem::size_of::<SceneUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Uniform BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Uniform BG"), layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scene_buffer.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cyber Arena Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cyber_arena.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"), bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[GameVertex::LAYOUT] },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut game = CyberShockGame::new();
    game.spawn_wave(1);

    let start_time = Instant::now();
    let mut last_frame = Instant::now();

    println!("\n🎮 ══════════════════════════════════════════════════════");
    println!("   CYBERSHOCK: Neon Grid Arena is LIVE!");
    println!("   Controls: WASD=Move | Mouse=Aim | Click=Fire");
    println!("   1/2/3/Q=Switch Weapons | R=Reload | Shift=Dash | ESC=Quit");
    println!("══════════════════════════════════════════════════════\n");

    let _ = event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                game.mouse_delta.0 += delta.0 as f32;
                game.mouse_delta.1 += delta.1 as f32;
            }
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => match event {
                WinitWindowEvent::CloseRequested => target.exit(),
                WinitWindowEvent::Resized(s) => {
                    if s.width > 0 && s.height > 0 {
                        config.width = s.width; config.height = s.height;
                        surface.configure(&device, &config);
                        depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("Depth"),
                            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
                            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                            format: DEPTH_FORMAT,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        }).create_view(&wgpu::TextureViewDescriptor::default());
                    }
                }
                WinitWindowEvent::KeyboardInput { event: ke, .. } => {
                    if let PhysicalKey::Code(code) = ke.physical_key {
                        match ke.state {
                            ElementState::Pressed => {
                                game.keys_held.insert(code);
                                match code {
                                    KeyCode::Escape => target.exit(),
                                    KeyCode::Digit1 => {
                                        game.player.current_weapon = 0;
                                        println!("🔫 Weapon: {}", game.player.weapons[0].name);
                                    }
                                    KeyCode::Digit2 => {
                                        game.player.current_weapon = 1;
                                        println!("🔫 Weapon: {}", game.player.weapons[1].name);
                                    }
                                    KeyCode::Digit3 => {
                                        game.player.current_weapon = 2;
                                        println!("🔫 Weapon: {}", game.player.weapons[2].name);
                                    }
                                    KeyCode::KeyQ => {
                                        game.player.current_weapon = (game.player.current_weapon + 1) % 3;
                                        println!("🔫 Weapon: {}", game.player.weapons[game.player.current_weapon].name);
                                    }
                                    KeyCode::KeyR => game.start_reload(),
                                    KeyCode::ShiftLeft | KeyCode::ShiftRight => game.dash(),
                                    KeyCode::Enter => {
                                        if game.state == GameState::GameOver || game.state == GameState::Victory {
                                            game = CyberShockGame::new();
                                            game.spawn_wave(1);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            ElementState::Released => {
                                game.keys_held.remove(&code);
                            }
                        }
                    }
                }
                WinitWindowEvent::MouseInput { state: ElementState::Pressed, button: WinitMouseButton::Left, .. } => {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    game.try_fire(elapsed);
                }
                WinitWindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32().min(0.05);
                    last_frame = now;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    game.update(dt, elapsed);

                    // Rebuild Scene Mesh
                    let (verts, idxs) = game.build_scene_mesh(elapsed);
                    let num_indices = idxs.len() as u32;

                    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Dynamic Vertices"), contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Dynamic Indices"), contents: bytemuck::cast_slice(&idxs),
                        usage: wgpu::BufferUsages::INDEX,
                    });

                    // Update Camera Uniform with screen shake
                    let aspect = config.width as f32 / config.height as f32;
                    let proj = perspective(Deg(75.0), aspect, 0.1, 500.0);

                    let shake_x = if game.screen_shake > 0.0 { (elapsed * 45.0).sin() * game.screen_shake * 0.3 } else { 0.0 };
                    let shake_y = if game.screen_shake > 0.0 { (elapsed * 35.0).cos() * game.screen_shake * 0.3 } else { 0.0 };

                    let cam_pos = game.player.pos + Vector3::new(shake_x, shake_y, 0.0);
                    let fwd = game.player.forward();
                    let target_pt = Point3::new(cam_pos.x + fwd.x, cam_pos.y + fwd.y, cam_pos.z + fwd.z);
                    let view = Matrix4::look_at_rh(
                        Point3::new(cam_pos.x, cam_pos.y, cam_pos.z),
                        target_pt,
                        Vector3::new(0.0, 1.0, 0.0)
                    );
                    let vp: [[f32; 4]; 4] = (proj * view).into();

                    let cam_uniform = CameraUniform {
                        view_proj: vp,
                        view_position: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
                    };
                    queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[cam_uniform]));

                    // Find Boss HP percentage if boss is present
                    let boss_hp_pct = game.enemies.iter()
                        .find(|e| e.enemy_type == EnemyType::Boss && e.active)
                        .map(|b| b.hp / b.max_hp)
                        .unwrap_or(0.0);

                    let active_weapon = &game.player.weapons[game.player.current_weapon];

                    let scene_uniform = SceneUniform {
                        sun_direction: [0.3, 0.8, 0.5, 1.8],
                        sun_color: [0.8, 0.85, 1.0, 0.0],
                        light0_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 12.0],
                        light0_color: [active_weapon.color[0], active_weapon.color[1], active_weapon.color[2], game.muzzle_flash_timer * 15.0],
                        params: [elapsed, game.player.dmg_flash_timer, 0.15, game.muzzle_flash_timer],
                        game_data: [
                            game.player.hp,
                            game.player.shield,
                            game.player.current_weapon as f32,
                            active_weapon.current_ammo as f32,
                        ],
                        extra0: [
                            game.player.score as f32,
                            game.player.combo as f32,
                            game.current_wave as f32,
                            game.enemies.iter().filter(|e| e.active).count() as f32,
                        ],
                        extra1: [
                            game.player.hitmarker_timer,
                            game.player.dash_active_timer * 5.0,
                            boss_hp_pct,
                            match game.state {
                                GameState::Title => 0.0,
                                GameState::Playing => 1.0,
                                GameState::WaveClear => 2.0,
                                GameState::Victory => 3.0,
                                GameState::GameOver => 4.0,
                            },
                        ],
                        extra2: [game.active_light_pos.x, game.active_light_pos.y, game.active_light_pos.z, 15.0],
                        extra3: [game.active_light_color[0], game.active_light_color[1], game.active_light_color[2], game.active_light_intensity],
                        hud_info: [config.width as f32, config.height as f32, game.player.max_hp, active_weapon.mag_size as f32],
                    };
                    queue.write_buffer(&scene_buffer, 0, bytemuck::cast_slice(&[scene_uniform]));

                    // Render Pass
                    let frame = surface.get_current_texture().unwrap();
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

                    {
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Scene Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.01, g: 0.01, b: 0.03, a: 1.0 }), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_texture,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                        rp.set_pipeline(&pipeline);
                        rp.set_bind_group(0, &bind_group, &[]);
                        rp.set_vertex_buffer(0, vb.slice(..));
                        rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        rp.draw_indexed(0..num_indices, 0, 0..1);
                    }

                    queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }
                _ => {}
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
