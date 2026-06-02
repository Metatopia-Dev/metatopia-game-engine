//! Quickstart module — Zero-boilerplate game launcher
//!
//! Handles all wgpu setup, window creation, event loop, depth buffers,
//! camera uniforms, and rendering pipeline. New developers only need to
//! implement the [`GameApp`] trait and call [`run_game`].
//!
//! # Example
//! ```no_run
//! use metatopia_engine::quickstart::*;
//!
//! struct MyGame { score: u32 }
//!
//! impl GameApp for MyGame {
//!     fn title(&self) -> &str { "My Game" }
//!     fn init(&mut self) { println!("Game started!"); }
//!     fn update(&mut self, ctx: &mut UpdateCtx) {
//!         if ctx.key_pressed(VirtualKey::Escape) { ctx.quit(); }
//!     }
//! }
//!
//! fn main() { run_game(MyGame { score: 0 }); }
//! ```

use cgmath::{InnerSpace, Matrix4, Deg, perspective, Point3, Vector3};
use winit::{
    event::{Event, WindowEvent, ElementState, DeviceEvent, MouseButton as WinitMouseButton},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use wgpu::util::DeviceExt;
use std::collections::HashSet;
use std::time::Instant;
use crate::collision::{CollisionWorld, Ray};
use crate::audio::AudioEngine;

// ─── Public Key Enum (simplified) ──────────────────────────────────────────

/// Virtual key codes for input handling.
/// Re-exports winit's KeyCode for convenience.
pub use winit::keyboard::KeyCode as VirtualKey;

// ─── GPU Uniform Types ─────────────────────────────────────────────────────

/// Camera data uploaded to the GPU each frame.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_position: [f32; 4],
}

/// Scene/game data uploaded to the GPU each frame.
/// All fields are `vec4<f32>` in WGSL. Use them however you like.
///
/// Default mapping:
///  - `params`:    x=time, y=exposure, z=ambient, w=custom
///  - `game_data`: 4 floats for game-specific values (score, level, etc.)
///  - `extra0–3`:  16 more floats for entities, lights, anything
///  - `hud_info`:  x=resolution_x, y=resolution_y, z=custom, w=custom
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub light0_pos: [f32; 4],
    pub light0_color: [f32; 4],
    pub params: [f32; 4],
    pub game_data: [f32; 4],
    pub extra0: [f32; 4],
    pub extra1: [f32; 4],
    pub extra2: [f32; 4],
    pub extra3: [f32; 4],
    pub hud_info: [f32; 4],
}

impl Default for SceneUniform {
    fn default() -> Self {
        Self {
            sun_direction: [0.3, 0.8, 0.5, 2.0],
            sun_color: [1.0, 0.95, 0.85, 0.0],
            light0_pos: [0.0, 3.0, 0.0, 10.0],
            light0_color: [1.0, 0.9, 0.8, 20.0],
            params: [0.0; 4],
            game_data: [0.0; 4],
            extra0: [0.0; 4], extra1: [0.0; 4],
            extra2: [0.0; 4], extra3: [0.0; 4],
            hud_info: [0.0; 4],
        }
    }
}

// ─── FPS Camera ────────────────────────────────────────────────────────────

/// Simple first-person camera with WASD + mouse look.
pub struct FpsCamera {
    pub position: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,
}

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 1.7, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 5.0,
            look_sensitivity: 0.002,
        }
    }
}

impl FpsCamera {
    pub fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> Vector3<f32> {
        let fwd = self.forward();
        Vector3::new(-fwd.z, 0.0, fwd.x).normalize()
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let p = Point3::new(self.position.x, self.position.y, self.position.z);
        let fwd = self.forward();
        let target = p + fwd;
        Matrix4::look_at_rh(p, target, Vector3::new(0.0, 1.0, 0.0))
    }
}

// ─── Vertex Type ───────────────────────────────────────────────────────────

/// Vertex with position, normal, UV, color, and a material/PBR param field.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GameVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
    pub pbr: [f32; 4], // metallic, roughness, emission, custom
}

impl GameVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,  // position
            1 => Float32x3,  // normal
            2 => Float32x2,  // uv
            3 => Float32x3,  // color
            4 => Float32x4,  // pbr
        ],
    };

    /// Quick constructor for a colored vertex.
    pub fn colored(pos: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self { position: pos, normal, uv: [0.0, 0.0], color, pbr: [0.0, 0.5, 0.0, 0.0] }
    }
}

// ─── Mesh Builder ──────────────────────────────────────────────────────────

/// Helper to build common meshes.
pub struct MeshBuilder;

impl MeshBuilder {
    /// Generate a flat grid floor centered at origin.
    pub fn floor(size: f32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let h = size / 2.0;
        let verts = vec![
            GameVertex::colored([-h, 0.0, -h], [0.0,1.0,0.0], color),
            GameVertex::colored([ h, 0.0, -h], [0.0,1.0,0.0], color),
            GameVertex::colored([ h, 0.0,  h], [0.0,1.0,0.0], color),
            GameVertex::colored([-h, 0.0,  h], [0.0,1.0,0.0], color),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (verts, indices)
    }

    /// Generate a simple cube centered at origin.
    pub fn cube(size: f32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let h = size / 2.0;
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        let faces: [([f32;3], [[f32;3];4]); 6] = [
            ([0.0,0.0,-1.0], [[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h]]),  // front
            ([0.0,0.0, 1.0], [[h,-h,h],[-h,-h,h],[-h,h,h],[h,h,h]]),      // back
            ([0.0,1.0, 0.0], [[-h,h,-h],[h,h,-h],[h,h,h],[-h,h,h]]),      // top
            ([0.0,-1.0,0.0], [[-h,-h,h],[h,-h,h],[h,-h,-h],[-h,-h,-h]]),   // bottom
            ([-1.0,0.0,0.0], [[-h,-h,h],[-h,-h,-h],[-h,h,-h],[-h,h,h]]),  // left
            ([1.0,0.0, 0.0], [[h,-h,-h],[h,-h,h],[h,h,h],[h,h,-h]]),      // right
        ];
        for (n, corners) in &faces {
            let base = verts.len() as u32;
            for c in corners {
                verts.push(GameVertex::colored(*c, *n, color));
            }
            indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }
        (verts, indices)
    }

    /// Generate a sphere (UV sphere).
    pub fn sphere(radius: f32, segments: u32, rings: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=rings {
            let theta = std::f32::consts::PI * j as f32 / rings as f32;
            for i in 0..=segments {
                let phi = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
                let x = theta.sin() * phi.cos();
                let y = theta.cos();
                let z = theta.sin() * phi.sin();
                verts.push(GameVertex::colored(
                    [x * radius, y * radius, z * radius],
                    [x, y, z],
                    color,
                ));
            }
        }
        for j in 0..rings {
            for i in 0..segments {
                let a = j * (segments + 1) + i;
                let b = a + segments + 1;
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            }
        }
        (verts, indices)
    }
}

// ─── Update Context ────────────────────────────────────────────────────────

/// Context passed to [`GameApp::update`] each frame.
/// Provides input state, camera access, scene uniform, timing, and quit control.
pub struct UpdateCtx<'a> {
    /// The first-person camera. Move it, rotate it, read its position.
    pub camera: &'a mut FpsCamera,
    /// Scene uniform data sent to the GPU. Set lighting, game_data, etc.
    pub scene: &'a mut SceneUniform,
    /// Collision world. Add colliders, query rays and overlaps.
    pub collision: &'a mut CollisionWorld,
    /// Audio engine. Play SFX and music.
    pub audio: &'a mut AudioEngine,
    /// Elapsed time since game start (seconds).
    pub time: f32,
    /// Delta time since last frame (seconds).
    pub dt: f32,
    /// Window resolution in pixels `(width, height)`.
    pub resolution: (u32, u32),
    /// Frame counter.
    pub frame: u64,

    keys_held: &'a HashSet<KeyCode>,
    keys_just_pressed: &'a HashSet<KeyCode>,
    mouse_just_pressed: &'a HashSet<WinitMouseButton>,
    mouse_delta_val: (f32, f32),
    quit_flag: &'a mut bool,
}

impl<'a> UpdateCtx<'a> {
    /// Returns `true` while the key is held down.
    pub fn key_held(&self, key: VirtualKey) -> bool { self.keys_held.contains(&key) }
    /// Returns `true` only on the frame the key was first pressed.
    pub fn key_pressed(&self, key: VirtualKey) -> bool { self.keys_just_pressed.contains(&key) }
    /// Returns `true` only on the frame a mouse button was clicked.
    pub fn mouse_pressed(&self, btn: WinitMouseButton) -> bool { self.mouse_just_pressed.contains(&btn) }
    /// Signal the game to exit.
    pub fn quit(&mut self) { *self.quit_flag = true; }

    /// Raw mouse delta this frame (pixels).
    pub fn mouse_delta(&self) -> (f32, f32) { self.mouse_delta_val }

    /// World-space ray from camera through the crosshair (screen center).
    pub fn aim_direction(&self) -> Vector3<f32> {
        self.camera.forward()
    }

    /// Cast a ray from the camera along the aim direction through the collision world.
    /// Returns the nearest hit within `max_dist`.
    pub fn raycast(&self, max_dist: f32) -> Option<crate::collision::QueryHit> {
        let ray = Ray::from_vectors(self.camera.position, self.camera.forward());
        self.collision.raycast(&ray, max_dist)
    }

    /// Move the camera using WASD + Space/Shift. Call this if you want built-in movement.
    pub fn default_camera_movement(&mut self) {
        let speed = self.camera.move_speed * self.dt;
        let fwd = self.camera.forward();
        let right = self.camera.right();
        let flat_fwd = Vector3::new(fwd.x, 0.0, fwd.z).normalize() * speed;
        let flat_right = right * speed;

        if self.key_held(VirtualKey::KeyW) { self.camera.position += flat_fwd; }
        if self.key_held(VirtualKey::KeyS) { self.camera.position -= flat_fwd; }
        if self.key_held(VirtualKey::KeyA) { self.camera.position -= flat_right; }
        if self.key_held(VirtualKey::KeyD) { self.camera.position += flat_right; }
        if self.key_held(VirtualKey::Space) { self.camera.position.y += speed; }
        if self.key_held(VirtualKey::ShiftLeft) { self.camera.position.y -= speed; }
    }
}

// ─── GameApp Trait ─────────────────────────────────────────────────────────

/// The main trait a new developer implements to create a game.
///
/// Only [`update`](GameApp::update) is required. Everything else has defaults.
pub trait GameApp {
    /// Window title.
    fn title(&self) -> &str { "Metatopia Game" }
    /// Window dimensions. Default: 1280×720.
    fn window_size(&self) -> (u32, u32) { (1280, 720) }

    /// Called once at startup. Load resources, set initial state.
    fn init(&mut self) {}

    /// Called every frame. Handle input, update game logic, write to `ctx.scene`.
    fn update(&mut self, ctx: &mut UpdateCtx);

    /// Return the WGSL shader source. Default loads `shaders/template_game.wgsl`.
    fn shader_source(&self) -> String {
        std::fs::read_to_string("shaders/template_game.wgsl")
            .unwrap_or_else(|_| include_str!("../shaders/template_game.wgsl").to_string())
    }

    /// Return mesh data (vertices, indices). Default creates a floor + cube scene.
    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let (mut verts, mut idxs) = MeshBuilder::floor(20.0, [0.3, 0.3, 0.35]);
        let (cv, ci) = MeshBuilder::cube(1.0, [0.8, 0.2, 0.3]);
        let offset = verts.len() as u32;
        for mut v in cv { v.position[1] += 0.5; verts.push(v); }
        for i in ci { idxs.push(i + offset); }
        (verts, idxs)
    }
}

// ─── Depth Texture Helper ──────────────────────────────────────────────────

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

// ─── run_game() — the single entry point ───────────────────────────────────

/// Launch a game. Handles ALL wgpu, window, and event loop boilerplate.
///
/// ```no_run
/// use metatopia_engine::quickstart::*;
/// struct MyGame;
/// impl GameApp for MyGame {
///     fn update(&mut self, ctx: &mut UpdateCtx) {
///         ctx.default_camera_movement();
///         if ctx.key_pressed(VirtualKey::Escape) { ctx.quit(); }
///     }
/// }
/// fn main() { run_game(MyGame); }
/// ```
pub fn run_game(mut app: impl GameApp + 'static) {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let (w, h) = app.window_size();
    let window = std::sync::Arc::new(
        WindowBuilder::new()
            .with_title(app.title())
            .with_inner_size(winit::dpi::LogicalSize::new(w, h))
            .build(&event_loop)
            .unwrap()
    );

    // ── wgpu setup ─────────────────────────────────────────────────
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
            label: Some("Game Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None)
    ).unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    // ── Depth texture ──────────────────────────────────────────────
    let mut depth_view = create_depth_texture(&device, config.width, config.height);

    // ── Mesh ───────────────────────────────────────────────────────
    let (vertices, indices) = app.build_mesh();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertices"), contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices"), contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let num_indices = indices.len() as u32;

    // ── Uniforms ───────────────────────────────────────────────────
    let mut camera_uniform = CameraUniform { view_proj: [[0.0;4];4], view_position: [0.0;4] };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera"), contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let mut scene_uniform = SceneUniform::default();
    let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene"), contents: bytemuck::cast_slice(&[scene_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // ── Bind group layout + bind group ─────────────────────────────
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Uniforms"),
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
        label: Some("Uniforms"), layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scene_buffer.as_entire_binding() },
        ],
    });

    // ── Pipeline ───────────────────────────────────────────────────
    let shader_src = app.shader_source();
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Game Shader"), source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bind_layout], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Game Pipeline"), layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module, entry_point: "vs_main",
            buffers: &[GameVertex::LAYOUT],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format, blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back), ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // ── State ──────────────────────────────────────────────────────
    let mut camera = FpsCamera::default();
    let mut collision_world = CollisionWorld::new();
    let mut audio_engine = AudioEngine::new();
    let mut keys_held: HashSet<KeyCode> = HashSet::new();
    let mut keys_just_pressed: HashSet<KeyCode> = HashSet::new();
    let mut mouse_just_pressed: HashSet<WinitMouseButton> = HashSet::new();
    let mut mouse_delta: (f32, f32) = (0.0, 0.0);
    let mut quit = false;
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut cursor_grabbed = false;
    let mut frame_count: u64 = 0;

    app.init();

    println!("🎮 {} — Running!", app.title());
    println!("   WASD=Move  Mouse=Look  Space/Shift=Up/Down  ESC=Quit");

    // ── Event Loop ─────────────────────────────────────────────────
    let _ = event_loop.run(move |event, target| {
        if quit { target.exit(); return; }

        match event {
            Event::WindowEvent { event: ref we, .. } => match we {
                WindowEvent::CloseRequested => { quit = true; }
                WindowEvent::Resized(s) => {
                    if s.width > 0 && s.height > 0 {
                        config.width = s.width; config.height = s.height;
                        surface.configure(&device, &config);
                        depth_view = create_depth_texture(&device, s.width, s.height);
                    }
                }
                WindowEvent::KeyboardInput { event: ke, .. } => {
                    if let PhysicalKey::Code(code) = ke.physical_key {
                        match ke.state {
                            ElementState::Pressed => {
                                if !keys_held.contains(&code) {
                                    keys_just_pressed.insert(code);
                                }
                                keys_held.insert(code);
                            }
                            ElementState::Released => { keys_held.remove(&code); }
                        }
                    }
                }
                WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                    mouse_just_pressed.insert(*button);
                    if !cursor_grabbed {
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                        window.set_cursor_visible(false);
                        cursor_grabbed = true;
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32().min(0.1);
                    last_frame = now;
                    let elapsed = (now - start).as_secs_f32();

                    // Auto-fill params.x with time
                    scene_uniform.params[0] = elapsed;
                    // Auto-fill hud_info with resolution
                    scene_uniform.hud_info[0] = config.width as f32;
                    scene_uniform.hud_info[1] = config.height as f32;

                    // Call the user's update
                    {
                        let mut ctx = UpdateCtx {
                            camera: &mut camera, scene: &mut scene_uniform,
                            collision: &mut collision_world,
                            audio: &mut audio_engine,
                            time: elapsed, dt,
                            resolution: (config.width, config.height),
                            frame: frame_count,
                            keys_held: &keys_held,
                            keys_just_pressed: &keys_just_pressed,
                            mouse_just_pressed: &mouse_just_pressed,
                            mouse_delta_val: mouse_delta,
                            quit_flag: &mut quit,
                        };
                        app.update(&mut ctx);
                    }

                    frame_count += 1;
                    mouse_delta = (0.0, 0.0);
                    keys_just_pressed.clear();
                    mouse_just_pressed.clear();

                    // Update camera uniform
                    let aspect = config.width as f32 / config.height as f32;
                    let proj = perspective(Deg(70.0), aspect, 0.1, 500.0);
                    let view = camera.view_matrix();
                    let vp: [[f32;4];4] = (proj * view).into();
                    camera_uniform.view_proj = vp;
                    camera_uniform.view_position = [
                        camera.position.x, camera.position.y, camera.position.z, 1.0,
                    ];

                    queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
                    queue.write_buffer(&scene_buffer, 0, bytemuck::cast_slice(&[scene_uniform]));

                    // Render
                    let output = match surface.get_current_texture() {
                        Ok(t) => t,
                        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                            surface.configure(&device, &config);
                            return;
                        }
                        Err(_) => return,
                    };
                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view, resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.08, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_view,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None, occlusion_query_set: None,
                        });
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..num_indices, 0, 0..1);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if cursor_grabbed {
                    camera.yaw += delta.0 as f32 * camera.look_sensitivity;
                    camera.pitch = (camera.pitch - delta.1 as f32 * camera.look_sensitivity)
                        .clamp(-1.5, 1.5);
                    mouse_delta.0 += delta.0 as f32;
                    mouse_delta.1 += delta.1 as f32;
                }
            }
            Event::AboutToWait => { window.request_redraw(); }
            _ => {}
        }
    });
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MeshBuilder Tests ──────────────────────────────────────────

    #[test]
    fn floor_has_correct_geometry() {
        let (verts, idxs) = MeshBuilder::floor(10.0, [1.0, 0.0, 0.0]);
        assert_eq!(verts.len(), 4, "floor should have 4 vertices");
        assert_eq!(idxs.len(), 6, "floor should have 6 indices (2 triangles)");
        // All normals point up
        for v in &verts {
            assert!((v.normal[1] - 1.0).abs() < 0.001, "floor normal should be (0,1,0)");
        }
    }

    #[test]
    fn cube_has_correct_geometry() {
        let (verts, idxs) = MeshBuilder::cube(2.0, [1.0, 1.0, 1.0]);
        assert_eq!(verts.len(), 24, "cube should have 24 vertices (4 per face × 6 faces)");
        assert_eq!(idxs.len(), 36, "cube should have 36 indices (6 per face × 6 faces)");
        // All indices should be in range
        for &i in &idxs {
            assert!(i < verts.len() as u32, "index out of range");
        }
    }

    #[test]
    fn cube_normals_point_outward() {
        let (verts, _) = MeshBuilder::cube(1.0, [1.0, 1.0, 1.0]);
        for v in &verts {
            let n_len = (v.normal[0]*v.normal[0] + v.normal[1]*v.normal[1] + v.normal[2]*v.normal[2]).sqrt();
            assert!((n_len - 1.0).abs() < 0.01, "normals should be unit length, got {n_len}");
        }
    }

    #[test]
    fn sphere_vertex_count() {
        let segments = 16u32;
        let rings = 12u32;
        let (verts, idxs) = MeshBuilder::sphere(1.0, segments, rings, [1.0, 1.0, 1.0]);
        let expected_verts = (segments + 1) * (rings + 1);
        assert_eq!(verts.len(), expected_verts as usize);
        let expected_idxs = segments * rings * 6;
        assert_eq!(idxs.len(), expected_idxs as usize);
    }

    #[test]
    fn sphere_normals_are_unit_length() {
        let (verts, _) = MeshBuilder::sphere(2.0, 8, 6, [1.0, 1.0, 1.0]);
        for v in &verts {
            let n_len = (v.normal[0]*v.normal[0] + v.normal[1]*v.normal[1] + v.normal[2]*v.normal[2]).sqrt();
            assert!((n_len - 1.0).abs() < 0.02, "sphere normal should be unit length, got {n_len}");
        }
    }

    // ── FpsCamera Tests ────────────────────────────────────────────

    #[test]
    fn camera_default_faces_negative_z() {
        // At yaw=0, pitch=0 the camera should face along +X (cos(0)=1, sin(0)=0)
        let cam = FpsCamera::default();
        let fwd = cam.forward();
        assert!((fwd.x - 1.0).abs() < 0.01, "forward.x should be ~1 at yaw=0");
        assert!(fwd.y.abs() < 0.01, "forward.y should be ~0 at pitch=0");
        assert!(fwd.z.abs() < 0.01, "forward.z should be ~0 at yaw=0");
    }

    #[test]
    fn camera_right_is_perpendicular() {
        let cam = FpsCamera::default();
        let fwd = cam.forward();
        let right = cam.right();
        let dot = fwd.x * right.x + fwd.y * right.y + fwd.z * right.z;
        assert!(dot.abs() < 0.01, "right should be perpendicular to forward, dot={dot}");
    }

    #[test]
    fn camera_pitch_clamp() {
        let mut cam = FpsCamera::default();
        cam.pitch = 2.0; // beyond 1.5
        let fwd = cam.forward();
        // Even with extreme pitch, forward should still be valid
        let len = (fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z).sqrt();
        assert!((len - 1.0).abs() < 0.01, "forward should stay unit length");
    }

    // ── Struct Size Tests ──────────────────────────────────────────

    #[test]
    fn scene_uniform_gpu_size() {
        // 11 × vec4 = 11 × 16 = 176 bytes
        assert_eq!(std::mem::size_of::<SceneUniform>(), 176);
    }

    #[test]
    fn camera_uniform_gpu_size() {
        // mat4x4 (64) + vec4 (16) = 80 bytes
        assert_eq!(std::mem::size_of::<CameraUniform>(), 80);
    }

    #[test]
    fn game_vertex_pod() {
        // Ensure GameVertex is Pod (required for bytemuck)
        let v = GameVertex::colored([0.0; 3], [0.0; 3], [0.0; 3]);
        let _bytes: &[u8] = bytemuck::bytes_of(&v);
    }

    // ── GameVertex Tests ───────────────────────────────────────────

    #[test]
    fn colored_vertex_defaults() {
        let v = GameVertex::colored([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        assert_eq!(v.color, [1.0, 0.0, 0.0]);
        assert_eq!(v.uv, [0.0, 0.0]);
        assert_eq!(v.pbr[1], 0.5, "default roughness should be 0.5");
    }
}
