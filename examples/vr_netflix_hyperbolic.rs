//! VR Netflix — Hyperbolic Cinema
//!
//! GPU-rendered non-Euclidean movie theater with 5 viewing spaces,
//! procedural audio drones, PBR screens, and Poincaré disk floor.
//!
//! Controls:
//!   WASD         – Move camera
//!   Mouse        – Look around
//!   1–5          – Switch theater space
//!   Left Click   – Play/pause selected screen
//!   Tab          – Cycle selected screen
//!   R            – Reset view
//!   +/-          – Ambient brightness
//!   Space/Shift  – Up / Down
//!   ESC          – Quit

// Engine crate (available but this example uses wgpu directly)
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent, ElementState, DeviceEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder as WinitWindowBuilder,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use std::time::Duration;

// ─── GPU Uniform ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TheaterUniform {
    camera_pos: [f32; 4],       // xyz=pos, w=fov
    camera_target: [f32; 4],    // xyz=target, w=time
    resolution: [f32; 4],       // x=w, y=h, z=space_type, w=selected_screen
    params: [f32; 4],           // x=screen_count, y=ambient, z=0, w=transition
    screen0: [f32; 4],          // xyz=position, w=playing
    screen1: [f32; 4],
    screen2: [f32; 4],
    screen3: [f32; 4],
    screen4: [f32; 4],
    screen5: [f32; 4],
    screen6: [f32; 4],
    screen7: [f32; 4],
}

// ─── Movie & Screen ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Movie {
    title: &'static str,
    genre: &'static str,
    rating: f32,
    year: u32,
}

const MOVIES: &[Movie] = &[
    Movie { title: "Inception",       genre: "Sci-Fi",     rating: 8.8, year: 2010 },
    Movie { title: "Interstellar",    genre: "Sci-Fi",     rating: 8.6, year: 2014 },
    Movie { title: "The Matrix",      genre: "Action",     rating: 8.7, year: 1999 },
    Movie { title: "Spirited Away",   genre: "Animation",  rating: 8.6, year: 2001 },
    Movie { title: "2001: A Space Odyssey", genre: "Sci-Fi", rating: 8.3, year: 1968 },
    Movie { title: "Blade Runner",    genre: "Sci-Fi",     rating: 8.1, year: 1982 },
    Movie { title: "Arrival",         genre: "Sci-Fi",     rating: 7.9, year: 2016 },
    Movie { title: "TENET",           genre: "Action",     rating: 7.3, year: 2020 },
];

struct ScreenState {
    position: [f32; 3],
    movie_idx: usize,
    playing: bool,
}

// ─── Theater Spaces ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum TheaterSpace {
    HyperbolicLobby = 0,
    SphericalDome   = 1,
    EscherTheater   = 2,
    PersonalPocket  = 3,
    SocialHub       = 4,
}

impl TheaterSpace {
    fn name(self) -> &'static str {
        match self {
            Self::HyperbolicLobby => "Hyperbolic Lobby",
            Self::SphericalDome   => "Spherical Dome",
            Self::EscherTheater   => "Escher Theater",
            Self::PersonalPocket  => "Personal Pocket",
            Self::SocialHub       => "Social Hub",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Self::HyperbolicLobby => "🟣",
            Self::SphericalDome   => "⭐",
            Self::EscherTheater   => "🔄",
            Self::PersonalPocket  => "🏠",
            Self::SocialHub       => "👥",
        }
    }
}

// ─── Audio ─────────────────────────────────────────────────────────────────

struct TheaterAudio {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    ambient_sink: Sink,
    current_space: i32,
}

impl TheaterAudio {
    fn new() -> Option<Self> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        let ambient = Sink::try_new(&handle).ok()?;
        ambient.set_volume(0.0);
        Some(Self { _stream: stream, handle, ambient_sink: ambient, current_space: -1 })
    }

    fn update_space(&mut self, space: TheaterSpace) {
        let idx = space as i32;
        if idx == self.current_space { return; }
        self.current_space = idx;
        self.ambient_sink.stop();
        self.ambient_sink = Sink::try_new(&self.handle).unwrap();
        self.ambient_sink.set_volume(0.05);

        let (f1, f2) = match space {
            TheaterSpace::HyperbolicLobby => (65.4, 98.0),    // Dark C2 + G2
            TheaterSpace::SphericalDome   => (130.8, 196.0),   // Celestial C3 + G3
            TheaterSpace::EscherTheater   => (87.3, 116.5),    // Mysterious F2 + Bb2
            TheaterSpace::PersonalPocket  => (110.0, 165.0),   // Warm A2 + E3
            TheaterSpace::SocialHub       => (98.0, 146.8),    // Social G2 + D3
        };

        let s1 = rodio::source::SineWave::new(f1)
            .amplify(0.4).fade_in(Duration::from_secs(3));
        self.ambient_sink.append(s1);

        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.035);
            let s2 = rodio::source::SineWave::new(f2)
                .amplify(0.25).fade_in(Duration::from_secs(4));
            s.append(s2);
            s.detach();
        }
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.02);
            let s3 = rodio::source::SineWave::new(f1 * 0.5)
                .amplify(0.3).fade_in(Duration::from_secs(5));
            s.append(s3);
            s.detach();
        }
    }

    fn play_transition(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.25);
            for i in 0..8 {
                let freq = 200.0 + i as f32 * 100.0;
                let tone = rodio::source::SineWave::new(freq)
                    .take_duration(Duration::from_millis(35))
                    .amplify(1.0 - i as f32 / 8.0);
                s.append(tone);
            }
            s.detach();
        }
    }

    fn play_select(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.2);
            let t = rodio::source::SineWave::new(660.0)
                .take_duration(Duration::from_millis(50));
            s.append(t);
            s.detach();
        }
    }

    fn play_toggle(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.2);
            let t1 = rodio::source::SineWave::new(440.0)
                .take_duration(Duration::from_millis(60));
            let t2 = rodio::source::SineWave::new(660.0)
                .take_duration(Duration::from_millis(80));
            s.append(t1);
            s.append(t2);
            s.detach();
        }
    }
}

// ─── Main State ────────────────────────────────────────────────────────────

struct VRTheater {
    camera_pos: cgmath::Point3<f32>,
    camera_rot: (f32, f32),
    mouse_sensitivity: f32,
    movement_speed: f32,
    space: TheaterSpace,
    screens: Vec<ScreenState>,
    selected: usize,
    ambient: f32,
    transition: f32,
    uniform: TheaterUniform,
    buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl VRTheater {
    fn new() -> Self {
        let mut t = Self {
            camera_pos: cgmath::Point3::new(0.0, 1.7, -6.0),
            camera_rot: (0.0, 0.0),
            mouse_sensitivity: 0.002,
            movement_speed: 0.12,
            space: TheaterSpace::HyperbolicLobby,
            screens: Vec::new(),
            selected: 0,
            ambient: 0.6,
            transition: 0.0,
            uniform: TheaterUniform {
                camera_pos: [0.0; 4],
                camera_target: [0.0; 4],
                resolution: [0.0; 4],
                params: [0.0; 4],
                screen0: [0.0, -100.0, 0.0, 0.0],
                screen1: [0.0, -100.0, 0.0, 0.0],
                screen2: [0.0, -100.0, 0.0, 0.0],
                screen3: [0.0, -100.0, 0.0, 0.0],
                screen4: [0.0, -100.0, 0.0, 0.0],
                screen5: [0.0, -100.0, 0.0, 0.0],
                screen6: [0.0, -100.0, 0.0, 0.0],
                screen7: [0.0, -100.0, 0.0, 0.0],
            },
            buffer: None,
            bind_group: None,
        };
        t.setup_space(TheaterSpace::HyperbolicLobby);
        t
    }

    fn setup_space(&mut self, space: TheaterSpace) {
        self.space = space;
        self.screens.clear();
        self.selected = 0;
        self.transition = 1.0;

        match space {
            TheaterSpace::HyperbolicLobby => {
                self.camera_pos = cgmath::Point3::new(0.0, 1.7, -6.0);
                self.camera_rot = (0.0, 0.0);
                // 7 screens in hyperbolic ring + 1 mega featured
                for i in 0..7 {
                    let angle = i as f32 * std::f32::consts::TAU / 7.0;
                    let r = 8.0;
                    self.screens.push(ScreenState {
                        position: [angle.cos() * r, 2.5, angle.sin() * r],
                        movie_idx: i % MOVIES.len(),
                        playing: false,
                    });
                }
                // Center featured screen
                self.screens.push(ScreenState {
                    position: [0.0, 3.5, 8.0],
                    movie_idx: 0,
                    playing: true,
                });
            }
            TheaterSpace::SphericalDome => {
                self.camera_pos = cgmath::Point3::new(0.0, 0.0, 0.0);
                self.camera_rot = (0.0, 0.0);
                for i in 0..8 {
                    let angle = i as f32 * std::f32::consts::TAU / 8.0;
                    let phi: f32 = 0.5;
                    let r = 10.0;
                    self.screens.push(ScreenState {
                        position: [
                            r * phi.sin() * angle.cos(),
                            r * phi.cos(),
                            r * phi.sin() * angle.sin(),
                        ],
                        movie_idx: i % MOVIES.len(),
                        playing: false,
                    });
                }
            }
            TheaterSpace::EscherTheater => {
                self.camera_pos = cgmath::Point3::new(0.0, 1.7, -5.0);
                self.camera_rot = (0.0, 0.0);
                // Staircase-like screen arrangement
                for i in 0..6 {
                    let angle = i as f32 * std::f32::consts::TAU / 6.0;
                    let height = 1.5 + (i as f32) * 0.8;
                    let r = 7.0;
                    self.screens.push(ScreenState {
                        position: [angle.cos() * r, height, angle.sin() * r],
                        movie_idx: i % MOVIES.len(),
                        playing: false,
                    });
                }
            }
            TheaterSpace::PersonalPocket => {
                self.camera_pos = cgmath::Point3::new(0.0, 1.5, -3.0);
                self.camera_rot = (0.0, 0.0);
                // Single large screen
                self.screens.push(ScreenState {
                    position: [0.0, 2.0, 3.0],
                    movie_idx: 2, // The Matrix
                    playing: true,
                });
                // Small side screen
                self.screens.push(ScreenState {
                    position: [-3.0, 1.8, 1.0],
                    movie_idx: 3,
                    playing: false,
                });
            }
            TheaterSpace::SocialHub => {
                self.camera_pos = cgmath::Point3::new(0.0, 1.7, -5.0);
                self.camera_rot = (0.0, 0.0);
                // Main shared screen
                self.screens.push(ScreenState {
                    position: [0.0, 3.0, 7.0],
                    movie_idx: 0,
                    playing: true,
                });
                // Side screens for friends
                for i in 0..4 {
                    let x = (i as f32 - 1.5) * 4.0;
                    self.screens.push(ScreenState {
                        position: [x, 2.0, 6.0],
                        movie_idx: (i + 1) % MOVIES.len(),
                        playing: false,
                    });
                }
            }
        }
    }

    fn update_uniform(&mut self, width: f32, height: f32, time: f32) {
        let (yaw, pitch) = self.camera_rot;
        let forward = cgmath::Vector3::new(
            yaw.sin() * pitch.cos(),
            -pitch.sin(),
            yaw.cos() * pitch.cos(),
        );
        let target = self.camera_pos + forward * 5.0;

        self.uniform.camera_pos = [self.camera_pos.x, self.camera_pos.y, self.camera_pos.z, 1.5];
        self.uniform.camera_target = [target.x, target.y, target.z, time];
        self.uniform.resolution = [width, height, self.space as u32 as f32, self.selected as f32];
        self.uniform.params = [self.screens.len().min(8) as f32, self.ambient, 0.0, self.transition];

        // Decay transition flash
        self.transition = (self.transition - 0.03).max(0.0);

        // Pack screen data
        let screen_data = [
            &mut self.uniform.screen0,
            &mut self.uniform.screen1,
            &mut self.uniform.screen2,
            &mut self.uniform.screen3,
            &mut self.uniform.screen4,
            &mut self.uniform.screen5,
            &mut self.uniform.screen6,
            &mut self.uniform.screen7,
        ];
        for (i, slot) in screen_data.into_iter().enumerate() {
            if i < self.screens.len() {
                let s = &self.screens[i];
                *slot = [s.position[0], s.position[1], s.position[2],
                         if s.playing { 1.0 } else { 0.0 }];
            } else {
                *slot = [0.0, -100.0, 0.0, 0.0]; // hidden
            }
        }
    }

    fn handle_keyboard(&mut self, key: KeyCode, pressed: bool) {
        if !pressed { return; }
        let (yaw, _) = self.camera_rot;
        let forward = cgmath::Vector3::new(yaw.sin(), 0.0, yaw.cos());
        let right = cgmath::Vector3::new(yaw.cos(), 0.0, -yaw.sin());
        let speed = self.movement_speed;

        match key {
            KeyCode::KeyW => self.camera_pos += forward * speed,
            KeyCode::KeyS => self.camera_pos -= forward * speed,
            KeyCode::KeyA => self.camera_pos -= right * speed,
            KeyCode::KeyD => self.camera_pos += right * speed,
            KeyCode::Space => self.camera_pos.y += speed,
            KeyCode::ShiftLeft => self.camera_pos.y -= speed,
            KeyCode::ArrowLeft => self.camera_rot.0 -= 0.05,
            KeyCode::ArrowRight => self.camera_rot.0 += 0.05,
            KeyCode::ArrowUp => self.camera_rot.1 = (self.camera_rot.1 - 0.05).clamp(-1.5, 1.5),
            KeyCode::ArrowDown => self.camera_rot.1 = (self.camera_rot.1 + 0.05).clamp(-1.5, 1.5),
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.ambient = (self.ambient + 0.1).min(1.5);
                println!("💡 Ambient: {:.1}", self.ambient);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.ambient = (self.ambient - 0.1).max(0.1);
                println!("💡 Ambient: {:.1}", self.ambient);
            }
            KeyCode::KeyR => {
                self.setup_space(self.space);
                println!("↻ View reset");
            }
            _ => {}
        }
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.camera_rot.0 += dx as f32 * self.mouse_sensitivity;
        self.camera_rot.1 = (self.camera_rot.1 - dy as f32 * self.mouse_sensitivity).clamp(-1.5, 1.5);
    }

    fn cycle_selected(&mut self) -> bool {
        if self.screens.is_empty() { return false; }
        self.selected = (self.selected + 1) % self.screens.len();
        true
    }

    fn toggle_playing(&mut self) -> bool {
        if self.selected < self.screens.len() {
            self.screens[self.selected].playing = !self.screens[self.selected].playing;
            true
        } else { false }
    }

    fn current_movie_title(&self) -> &'static str {
        if self.selected < self.screens.len() {
            MOVIES[self.screens[self.selected].movie_idx].title
        } else { "?" }
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

async fn run() {
    env_logger::init();

    println!("🎬 VR NETFLIX — HYPERBOLIC CINEMA 🎬");
    println!("═════════════════════════════════════");
    println!("Controls:");
    println!("  WASD / Arrows    – Move / Look");
    println!("  Mouse            – Look around");
    println!("  1–5              – Switch theater:");
    println!("    1 🟣 Hyperbolic Lobby  – Infinite screens on Poincaré disk");
    println!("    2 ⭐ Spherical Dome    – Starfield cinema");
    println!("    3 🔄 Escher Theater    – Impossible geometry");
    println!("    4 🏠 Personal Pocket   – Cozy curved room");
    println!("    5 👥 Social Hub        – Watch with friends");
    println!("  Tab              – Cycle screen selection");
    println!("  Left Click       – Play / Pause selected screen");
    println!("  +/-              – Ambient brightness");
    println!("  R                – Reset view");
    println!("  ESC              – Quit");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("VR Netflix — Hyperbolic Cinema")
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
        label: Some("Theater"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
    }, None).await.unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    let mut theater = VRTheater::new();
    let mut audio = TheaterAudio::new();
    if audio.is_some() { println!("🔊 Audio initialized"); }

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Theater Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vr_theater.wgsl").into()),
    });

    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Theater Uniform"),
        contents: bytemuck::cast_slice(&[theater.uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        }],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });
    theater.buffer = Some(buf);
    theater.bind_group = Some(bg);

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None, layout: Some(&pl),
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[] },
        fragment: Some(wgpu::FragmentState {
            module: &shader, entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
    });

    let start_time = std::time::Instant::now();
    let mut frame_count = 0u32;

    // Print initial state
    println!("{} {} | {} screens | 🎬 {}", theater.space.icon(), theater.space.name(),
        theater.screens.len(), theater.current_movie_title());

    let _ = event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => match event {
                WinitWindowEvent::CloseRequested => target.exit(),
                WinitWindowEvent::Resized(s) => {
                    if s.width > 0 && s.height > 0 {
                        config.width = s.width; config.height = s.height;
                        surface.configure(&device, &config);
                    }
                }
                WinitWindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(kc) = event.physical_key {
                        if event.state == ElementState::Pressed {
                            if kc == KeyCode::Escape { target.exit(); return; }

                            // Space switching
                            let new_space = match kc {
                                KeyCode::Digit1 => Some(TheaterSpace::HyperbolicLobby),
                                KeyCode::Digit2 => Some(TheaterSpace::SphericalDome),
                                KeyCode::Digit3 => Some(TheaterSpace::EscherTheater),
                                KeyCode::Digit4 => Some(TheaterSpace::PersonalPocket),
                                KeyCode::Digit5 => Some(TheaterSpace::SocialHub),
                                _ => None,
                            };
                            if let Some(sp) = new_space {
                                if sp != theater.space {
                                    theater.setup_space(sp);
                                    if let Some(ref a) = audio { a.play_transition(); }
                                    println!("\n{} Entering {} | {} screens",
                                        sp.icon(), sp.name(), theater.screens.len());
                                }
                            }

                            // Tab = cycle selection
                            if kc == KeyCode::Tab {
                                if theater.cycle_selected() {
                                    if let Some(ref a) = audio { a.play_select(); }
                                    let m = &MOVIES[theater.screens[theater.selected].movie_idx];
                                    let status = if theater.screens[theater.selected].playing { "▶" } else { "⏸" };
                                    println!("  → [{}] {} {} ({}) ★{:.1}", status,
                                        m.title, m.genre, m.year, m.rating);
                                }
                            }

                            theater.handle_keyboard(kc, true);
                        }
                    }
                }
                WinitWindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    if theater.toggle_playing() {
                        let playing = theater.screens[theater.selected].playing;
                        let m = &MOVIES[theater.screens[theater.selected].movie_idx];
                        if playing {
                            println!("  ▶ Playing: {}", m.title);
                        } else {
                            println!("  ⏸ Paused: {}", m.title);
                        }
                        if let Some(ref a) = audio { a.play_toggle(); }
                    }
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    theater.update_uniform(config.width as f32, config.height as f32, elapsed);

                    if let Some(ref mut a) = audio { a.update_space(theater.space); }

                    if let Some(ref b) = theater.buffer {
                        queue.write_buffer(b, 0, bytemuck::cast_slice(&[theater.uniform]));
                    }

                    let output = surface.get_current_texture().unwrap();
                    let cv = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &cv, resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        rp.set_pipeline(&pipeline);
                        if let Some(ref bg) = theater.bind_group { rp.set_bind_group(0, bg, &[]); }
                        rp.draw(0..6, 0..1);
                    }
                    queue.submit(std::iter::once(enc.finish()));
                    output.present();

                    if frame_count % 120 == 0 {
                        let fps = frame_count as f32 / elapsed;
                        let playing_count = theater.screens.iter().filter(|s| s.playing).count();
                        println!("{} {} | 📺{}+{}▶ | 🎬{} | 💡{:.1} | {:.0}fps",
                            theater.space.icon(), theater.space.name(),
                            theater.screens.len(), playing_count,
                            theater.current_movie_title(), theater.ambient, fps);
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                theater.handle_mouse_motion(delta.0, delta.1);
            }
            Event::AboutToWait => { window.request_redraw(); }
            _ => {}
        }
    });
}

fn main() { pollster::block_on(run()); }