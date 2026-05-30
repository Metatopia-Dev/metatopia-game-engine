//! Pest Control Simulator — Playable FPS
//!
//! Hunt pests in a kitchen! Each level spawns different bugs.
//! Left-click to spray, 1/2 to switch tools, R to restart level.

use metatopia_engine::prelude::*;
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent, ElementState, DeviceEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder as WinitWindowBuilder,
};
use std::sync::{Arc, RwLock};
use cgmath::{InnerSpace, Point3, Vector3, Matrix4, Deg, perspective};
use wgpu::util::DeviceExt;

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
    params: [f32; 4],
    game_info: [f32; 4],
    pest0: [f32; 4],
    pest1: [f32; 4],
    pest2: [f32; 4],
    pest3: [f32; 4],
    pest4: [f32; 4],
    pest5: [f32; 4],
    pest6: [f32; 4],
    pest7: [f32; 4],
    pest_flash: [f32; 4],
    pest_flash2: [f32; 4],
}

// ─── Game Objects ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum PestAI { Wandering, Fleeing, Hiding, Dead }

#[derive(Debug, Clone)]
struct PestState {
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    pest_type: u32,         // 1=cockroach, 2=ant, 3=spider, 4=rat, 5=wasp
    health: f32,
    max_health: f32,
    speed: f32,
    detection_radius: f32,
    ai_state: PestAI,
    hit_flash: f32,
    wander_timer: f32,
    rng: u32,
}

struct ToolState {
    name: &'static str,
    range: f32,
    damage: f32,
    cooldown: f32,
    ammo: u32,
    max_ammo: u32,
    last_fire: f32,
}

const LOCATION_NAMES: [&str; 6] = ["Kitchen", "Bathroom", "Basement", "Attic", "Garden", "Restaurant"];

// ─── Helpers ───────────────────────────────────────────────────────────────

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"), size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

fn next_rand(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    ((*state >> 16) & 0xFFFF) as f32 / 65536.0
}

fn pest_radius(t: u32) -> f32 {
    match t { 1 => 0.18, 2 => 0.10, 3 => 0.15, 4 => 0.35, 5 => 0.20, _ => 0.0 }
}

fn pest_score(t: u32) -> u32 {
    match t { 1 => 10, 2 => 5, 3 => 15, 4 => 25, 5 => 20, _ => 0 }
}

fn pest_name(t: u32) -> &'static str {
    match t { 1 => "Cockroach", 2 => "Ant", 3 => "Spider", 4 => "Rat", 5 => "Wasp", _ => "?" }
}

fn pest_stats(t: u32) -> (f32, f32, f32) {
    // (health, speed, detection_radius)
    match t {
        1 => (50.0, 3.0, 4.5),   // cockroach: fast, medium HP
        2 => (15.0, 1.5, 3.0),   // ant: fragile, slow
        3 => (35.0, 2.5, 5.0),   // spider: mid, lurks
        4 => (100.0, 2.0, 6.0),  // rat: tough, medium speed
        5 => (40.0, 4.0, 8.0),   // wasp: medium, fast, detects far
        _ => (10.0, 1.0, 3.0),
    }
}

// ─── Game State ────────────────────────────────────────────────────────────

struct PestControlGame {
    _manifold: Arc<RwLock<Manifold>>,
    camera_position: Point3<f32>,
    camera_rotation: (f32, f32),
    camera_uniform: CameraUniform,
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    scene_uniform: SceneUniform,
    scene_buffer: Option<wgpu::Buffer>,
    scene_bind_group: Option<wgpu::BindGroup>,
    mouse_sensitivity: f32,
    pests: Vec<PestState>,
    tools: Vec<ToolState>,
    current_tool: usize,
    score: u32,
    level: u32,
    time_remaining: f32,
    firing_flash: f32,
    frame_count: u32,
    rng_state: u32,
}

impl PestControlGame {
    fn new() -> Self {
        let manifold = Manifold::new();
        let mut game = Self {
            _manifold: Arc::new(RwLock::new(manifold)),
            camera_position: Point3::new(0.0, 1.7, -4.0),
            camera_rotation: (0.0, 0.0),
            camera_uniform: CameraUniform { view_proj: [[0.0; 4]; 4], view_position: [0.0; 4] },
            camera_buffer: None, camera_bind_group: None,
            scene_uniform: SceneUniform {
                sun_direction: [0.0;4], sun_color: [0.0;4],
                light0_pos: [0.0;4], light0_color: [0.0;4],
                light1_pos: [0.0;4], light1_color: [0.0;4],
                params: [0.0;4], game_info: [0.0;4],
                pest0: [0.0;4], pest1: [0.0;4], pest2: [0.0;4], pest3: [0.0;4],
                pest4: [0.0;4], pest5: [0.0;4], pest6: [0.0;4], pest7: [0.0;4],
                pest_flash: [0.0;4], pest_flash2: [0.0;4],
            },
            scene_buffer: None, scene_bind_group: None,
            mouse_sensitivity: 0.002,
            pests: Vec::new(),
            tools: vec![
                ToolState { name: "Spray", range: 4.0, damage: 30.0, cooldown: 0.15,
                    ammo: 100, max_ammo: 100, last_fire: -10.0 },
                ToolState { name: "Vacuum", range: 8.0, damage: 100.0, cooldown: 0.6,
                    ammo: 20, max_ammo: 20, last_fire: -10.0 },
            ],
            current_tool: 0,
            score: 0,
            level: 1,
            time_remaining: 120.0,
            firing_flash: 0.0,
            frame_count: 0,
            rng_state: 42,
        };
        game.spawn_level();
        game
    }

    // ── Level Spawning ─────────────────────────────────────────────

    fn spawn_level(&mut self) {
        self.pests.clear();
        let count = (5 + self.level * 2).min(8) as usize;
        let location = ((self.level - 1) % 6) as usize;

        println!("\n🏠 Level {} — {} | {} pests incoming!",
            self.level, LOCATION_NAMES[location], count);

        for i in 0..count {
            let mut seed = self.level * 1000 + i as u32 * 137 + 7;
            let pest_type = self.pick_pest(location, &mut seed);
            let (health, speed, detect) = pest_stats(pest_type);
            let x = next_rand(&mut seed) * 10.0 - 5.0;
            let z = next_rand(&mut seed) * 10.0 - 5.0;
            let y = if pest_type == 5 {
                1.0 + next_rand(&mut seed) * 1.5
            } else {
                pest_radius(pest_type)
            };

            self.pests.push(PestState {
                position: Vector3::new(x, y, z),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                pest_type,
                health, max_health: health,
                speed, detection_radius: detect,
                ai_state: PestAI::Wandering,
                hit_flash: 0.0,
                wander_timer: next_rand(&mut seed) * 2.0 + 1.0,
                rng: seed,
            });
        }

        // Refill tools
        for t in &mut self.tools { t.ammo = t.max_ammo; }
        self.time_remaining = 120.0;
    }

    fn pick_pest(&mut self, location: usize, seed: &mut u32) -> u32 {
        let r = next_rand(seed);
        match location {
            0 => if r < 0.4 { 1 } else if r < 0.7 { 2 } else { 4 },  // Kitchen
            1 => if r < 0.5 { 1 } else { 3 },                          // Bathroom
            2 => 4,                                                      // Basement: rats
            3 => if r < 0.5 { 3 } else { 5 },                          // Attic
            4 => 5,                                                      // Garden: wasps
            5 => if r < 0.6 { 1 } else { 2 },                          // Restaurant
            _ => 1,
        }
    }

    // ── Main Update ────────────────────────────────────────────────

    fn update(&mut self, dt: f32, _time: f32) {
        self.frame_count = self.frame_count.wrapping_add(1);
        self.firing_flash = (self.firing_flash - dt * 6.0).max(0.0);

        // Decay hit flash
        for pest in &mut self.pests {
            pest.hit_flash = (pest.hit_flash - dt * 4.0).max(0.0);
        }

        self.update_pest_ai(dt);
        self.time_remaining -= dt;
    }

    // ── Pest AI ────────────────────────────────────────────────────

    fn update_pest_ai(&mut self, dt: f32) {
        let cam = Vector3::new(self.camera_position.x, self.camera_position.y, self.camera_position.z);

        for pest in &mut self.pests {
            if pest.ai_state == PestAI::Dead { continue; }

            pest.wander_timer -= dt;
            let to_player = cam - pest.position;
            let dist = to_player.magnitude();

            match pest.ai_state {
                PestAI::Wandering => {
                    if dist < pest.detection_radius {
                        pest.ai_state = PestAI::Fleeing;
                    }
                    if pest.wander_timer <= 0.0 {
                        pest.wander_timer = next_rand(&mut pest.rng) * 2.0 + 1.0;
                        let angle = next_rand(&mut pest.rng) * std::f32::consts::TAU;
                        pest.velocity = Vector3::new(angle.cos(), 0.0, angle.sin()) * pest.speed;
                    }
                }
                PestAI::Fleeing => {
                    if dist > 0.1 {
                        let flee = Vector3::new(-to_player.x, 0.0, -to_player.z);
                        if flee.magnitude() > 0.01 {
                            pest.velocity = flee.normalize() * pest.speed * 1.8;
                        }
                    }
                    if dist > pest.detection_radius * 2.5 {
                        pest.ai_state = PestAI::Hiding;
                        pest.wander_timer = 2.0 + next_rand(&mut pest.rng) * 2.0;
                    }
                }
                PestAI::Hiding => {
                    pest.velocity *= 0.92;
                    if pest.wander_timer <= 0.0 && dist > pest.detection_radius * 1.5 {
                        pest.ai_state = PestAI::Wandering;
                        pest.wander_timer = 1.5;
                    }
                }
                PestAI::Dead => {}
            }

            pest.position += pest.velocity * dt;

            // Keep ground pests on floor
            if pest.pest_type != 5 {
                pest.position.y = pest_radius(pest.pest_type);
            } else {
                // Wasp bobs
                pest.position.y = pest.position.y.clamp(0.8, 2.5);
                pest.velocity.y = (next_rand(&mut pest.rng) - 0.5) * 1.5;
            }

            // Wall bounds
            let wall = 5.5;
            if pest.position.x > wall { pest.position.x = wall; pest.velocity.x = -pest.velocity.x; }
            if pest.position.x < -wall { pest.position.x = -wall; pest.velocity.x = -pest.velocity.x; }
            if pest.position.z > wall { pest.position.z = wall; pest.velocity.z = -pest.velocity.z; }
            if pest.position.z < -wall { pest.position.z = -wall; pest.velocity.z = -pest.velocity.z; }
        }
    }

    // ── Firing ─────────────────────────────────────────────────────

    fn try_fire(&mut self, elapsed: f32) {
        let tool = &self.tools[self.current_tool];
        if tool.ammo == 0 {
            println!("🔴 Out of ammo! Press R to reload");
            return;
        }
        if elapsed - tool.last_fire < tool.cooldown { return; }

        let tool_range = tool.range;
        let tool_damage = tool.damage;
        let tool_name = tool.name;
        self.tools[self.current_tool].ammo -= 1;
        self.tools[self.current_tool].last_fire = elapsed;
        self.firing_flash = 1.0;

        // Raycast from camera
        let fwd = self.get_forward_vector();
        let origin = Vector3::new(self.camera_position.x, self.camera_position.y, self.camera_position.z);

        let mut best_idx: Option<usize> = None;
        let mut best_t = f32::MAX;

        for (i, pest) in self.pests.iter().enumerate() {
            if pest.ai_state == PestAI::Dead { continue; }

            let to_pest = pest.position - origin;
            let t = to_pest.dot(fwd);
            if t < 0.0 || t > tool_range { continue; }

            let closest = origin + fwd * t;
            let miss_dist = (closest - pest.position).magnitude();
            let hit_radius = pest_radius(pest.pest_type) * 2.5; // generous hitbox

            if miss_dist < hit_radius && t < best_t {
                best_t = t;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            let pest = &mut self.pests[idx];
            pest.health -= tool_damage;
            pest.hit_flash = 1.0;
            pest.ai_state = PestAI::Fleeing;

            if pest.health <= 0.0 {
                pest.ai_state = PestAI::Dead;
                let pts = pest_score(pest.pest_type);
                self.score += pts;
                println!("💀 {} eliminated! +{} pts (Score: {})",
                    pest_name(pest.pest_type), pts, self.score);

                // Check level complete
                let alive = self.pests.iter().filter(|p| p.ai_state != PestAI::Dead).count();
                if alive == 0 {
                    let bonus = (self.time_remaining as u32) * 2;
                    self.score += bonus;
                    println!("🎉 Level {} Complete! Time bonus: +{} | Total: {}",
                        self.level, bonus, self.score);
                    self.level += 1;
                    self.spawn_level();
                }
            } else {
                println!("🎯 Hit {} with {}! ({:.0}/{:.0} HP)",
                    pest_name(pest.pest_type), tool_name, pest.health, pest.max_health);
            }
        }
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
        let proj = perspective(Deg(70.0), aspect, 0.05, 100.0);
        self.camera_uniform.view_proj = (proj * view).into();
        self.camera_uniform.view_position = [
            self.camera_position.x, self.camera_position.y, self.camera_position.z, 0.0,
        ];
    }

    fn update_scene_uniform(&mut self, time: f32) {
        // ── Kitchen lighting ──────────────────────────────────────
        let sd = Vector3::new(0.2_f32, 0.9, 0.3).normalize();
        self.scene_uniform.sun_direction = [sd.x, sd.y, sd.z, 2.0];
        self.scene_uniform.sun_color     = [1.0, 0.95, 0.85, 0.0];
        self.scene_uniform.light0_pos    = [0.0, 2.8, 0.0, 10.0];
        self.scene_uniform.light0_color  = [1.0, 0.92, 0.75, 30.0]; // warm overhead
        self.scene_uniform.light1_pos    = [0.0, 2.0, -5.5, 8.0];
        self.scene_uniform.light1_color  = [0.7, 0.8, 1.0, 15.0]; // cool window

        self.scene_uniform.params = [time, 2.2, 4.0, self.firing_flash];

        let alive = self.pests.iter().filter(|p| p.ai_state != PestAI::Dead).count();
        self.scene_uniform.game_info = [
            self.score as f32, self.level as f32, alive as f32, self.time_remaining,
        ];

        // ── Pest uniforms ─────────────────────────────────────────
        let pest_slots: [&mut [f32; 4]; 8] = [
            &mut self.scene_uniform.pest0, &mut self.scene_uniform.pest1,
            &mut self.scene_uniform.pest2, &mut self.scene_uniform.pest3,
            &mut self.scene_uniform.pest4, &mut self.scene_uniform.pest5,
            &mut self.scene_uniform.pest6, &mut self.scene_uniform.pest7,
        ];
        for (i, slot) in pest_slots.into_iter().enumerate() {
            if i < self.pests.len() && self.pests[i].ai_state != PestAI::Dead {
                let p = &self.pests[i];
                *slot = [p.position.x, p.position.y, p.position.z, p.pest_type as f32];
            } else {
                *slot = [0.0, -10.0, 0.0, 0.0]; // hidden below floor
            }
        }

        // Hit flash values
        let mut flash = [0.0_f32; 8];
        for (i, p) in self.pests.iter().enumerate().take(8) {
            flash[i] = p.hit_flash;
        }
        self.scene_uniform.pest_flash  = [flash[0], flash[1], flash[2], flash[3]];
        self.scene_uniform.pest_flash2 = [flash[4], flash[5], flash[6], flash[7]];
    }

    // ── Input ──────────────────────────────────────────────────────

    fn handle_keyboard(&mut self, key: KeyCode, pressed: bool) {
        if !pressed { return; }
        let speed = 0.12;
        let fwd = self.get_forward_vector();
        let right = Vector3::new(-fwd.z, 0.0, fwd.x).normalize();

        match key {
            KeyCode::KeyW       => self.camera_position += fwd * speed,
            KeyCode::KeyS       => self.camera_position -= fwd * speed,
            KeyCode::KeyA       => self.camera_position -= right * speed,
            KeyCode::KeyD       => self.camera_position += right * speed,
            KeyCode::Space      => self.camera_position.y = (self.camera_position.y + speed).min(2.5),
            KeyCode::ShiftLeft  => self.camera_position.y = (self.camera_position.y - speed).max(0.8),
            KeyCode::Digit1     => { self.current_tool = 0; println!("🔫 Spray Bottle selected"); }
            KeyCode::Digit2     => { self.current_tool = 1; println!("🔫 Vacuum Gun selected"); }
            KeyCode::KeyR       => {
                for t in &mut self.tools { t.ammo = t.max_ammo; }
                println!("🔄 Tools reloaded!");
            }
            _ => {}
        }

        // Clamp to room bounds
        self.camera_position.x = self.camera_position.x.clamp(-5.5, 5.5);
        self.camera_position.z = self.camera_position.z.clamp(-5.5, 5.5);
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.camera_rotation.0 += dx as f32 * self.mouse_sensitivity;
        self.camera_rotation.1 = (self.camera_rotation.1 - dy as f32 * self.mouse_sensitivity).clamp(-1.5, 1.5);
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

async fn run() {
    env_logger::init();

    println!("🪳 PEST CONTROL SIMULATOR 🪳");
    println!("═══════════════════════════════");
    println!("Controls:");
    println!("  WASD         – Move");
    println!("  Mouse        – Look");
    println!("  Left Click   – Fire tool");
    println!("  1            – Spray Bottle (fast, close range)");
    println!("  2            – Vacuum Gun (powerful, long range, limited ammo)");
    println!("  R            – Reload all tools");
    println!("  Space/Shift  – Up / Down");
    println!("  ESC          – Quit");
    println!();
    println!("Pests: 🪳Cockroach 🐜Ant 🕷Spider 🐀Rat 🐝Wasp");
    println!("Aim at pests and click to eliminate them!");

    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("Pest Control Simulator")
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
        label: Some("PestControl"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
    }, None).await.unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    let mut game = PestControlGame::new();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Pest Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pest_control.wgsl").into()),
    });

    // Camera bind group
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cam"), contents: bytemuck::cast_slice(&[game.camera_uniform]),
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
    game.camera_buffer = Some(cam_buf); game.camera_bind_group = Some(cam_bg);

    // Scene bind group
    let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene"), contents: bytemuck::cast_slice(&[game.scene_uniform]),
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
    game.scene_buffer = Some(scene_buf); game.scene_bind_group = Some(scene_bg);

    let mut depth_view = create_depth_texture(&device, config.width, config.height);

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&cam_bgl, &scene_bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, layout: Some(&pl),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState { format: config.format,
                blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }),
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
    });

    let start_time = std::time::Instant::now();
    let mut last_time = start_time;
    let mut frame_count = 0u32;
    let mut game_over = false;
    const VERTS: u32 = 240; // 36 room + 192 pests + 12 crosshair

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
                WinitWindowEvent::KeyboardInput { event: key_event, .. } => {
                    if let PhysicalKey::Code(kc) = key_event.physical_key {
                        if kc == KeyCode::Escape { target.exit(); }
                        else { game.handle_keyboard(kc, key_event.state == ElementState::Pressed); }
                    }
                }
                WinitWindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    if !game_over {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        game.try_fire(elapsed);
                    }
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let now = std::time::Instant::now();
                    let dt = (now - last_time).as_secs_f32().min(0.05);
                    last_time = now;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    if !game_over {
                        game.update(dt, elapsed);

                        if game.time_remaining <= 0.0 {
                            game_over = true;
                            println!("\n⏰ TIME'S UP! Game Over!");
                            println!("══════════════════════════");
                            println!("Final Score: {} | Reached Level: {}", game.score, game.level);
                            println!("Press ESC to exit, R to restart");
                        }
                    }

                    game.update_camera_uniform(config.width as f32 / config.height as f32);
                    game.update_scene_uniform(elapsed);

                    if let Some(ref b) = game.camera_buffer { queue.write_buffer(b, 0, bytemuck::cast_slice(&[game.camera_uniform])); }
                    if let Some(ref b) = game.scene_buffer  { queue.write_buffer(b, 0, bytemuck::cast_slice(&[game.scene_uniform])); }

                    let output = surface.get_current_texture().unwrap();
                    let cv = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &cv, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(
                                    wgpu::Color { r: 0.08, g: 0.07, b: 0.06, a: 1.0 }),
                                    store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_view,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            occlusion_query_set: None, timestamp_writes: None,
                        });
                        rp.set_pipeline(&pipeline);
                        if let Some(ref bg) = game.camera_bind_group { rp.set_bind_group(0, bg, &[]); }
                        if let Some(ref bg) = game.scene_bind_group  { rp.set_bind_group(1, bg, &[]); }
                        rp.draw(0..VERTS, 0..1);
                    }
                    queue.submit(std::iter::once(enc.finish()));
                    output.present();

                    // HUD
                    if frame_count % 60 == 0 && !game_over {
                        let alive = game.pests.iter().filter(|p| p.ai_state != PestAI::Dead).count();
                        let t = &game.tools[game.current_tool];
                        let fps = frame_count as f32 / elapsed;
                        println!("Lv{} | 🏆{} | 🪳{} left | ⏱{:.0}s | 🔫{}({}) | {:.0}fps",
                            game.level, game.score, alive,
                            game.time_remaining, t.name, t.ammo, fps);
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                game.handle_mouse_motion(delta.0, delta.1);
            }
            Event::AboutToWait => { window.request_redraw(); }
            _ => {}
        }
    });
}

fn main() { pollster::block_on(run()); }