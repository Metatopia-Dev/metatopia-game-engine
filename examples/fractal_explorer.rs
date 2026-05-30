//! Mandelbulb Fractal Explorer
//!
//! GPU ray-marched 3D fractal with orbit camera, 5 color palettes,
//! adjustable power, and audio-reactive pulsation.
//!
//! Controls:
//!   Mouse drag – Orbit camera
//!   Scroll/+/- – Zoom in/out
//!   1–5        – Color palette (Fire, Ocean, Nebula, Earth, Mono)
//!   P / O      – Increase / Decrease fractal power
//!   A          – Toggle audio-reactive mode
//!   R          – Reset view
//!   ESC        – Quit

use metatopia_engine::prelude::*;
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent, ElementState, DeviceEvent, MouseButton, MouseScrollDelta},
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
struct FractalUniform {
    camera_pos: [f32; 4],       // xyz=position, w=fov
    camera_target: [f32; 4],    // xyz=look-at, w=zoom
    resolution: [f32; 4],       // x=width, y=height, z=time, w=power
    params: [f32; 4],           // x=palette, y=audio, z=max_iter, w=warp
}

// ─── Audio ─────────────────────────────────────────────────────────────────

struct FractalAudio {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    ambient: Sink,
    current_palette: u32,
}

impl FractalAudio {
    fn new() -> Option<Self> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        let ambient = Sink::try_new(&handle).ok()?;
        ambient.set_volume(0.0);
        Some(Self { _stream: stream, handle, ambient, current_palette: 99 })
    }

    fn update_drone(&mut self, palette: u32) {
        if palette == self.current_palette { return; }
        self.current_palette = palette;
        self.ambient.stop();
        self.ambient = Sink::try_new(&self.handle).unwrap();
        self.ambient.set_volume(0.05);

        // Each palette gets a unique harmonic drone
        let (f1, f2) = match palette {
            0 => (82.4, 123.5),     // Fire: E2 + B2 (power fifth)
            1 => (130.8, 196.0),    // Ocean: C3 + G3 (perfect fifth)
            2 => (92.5, 138.6),     // Nebula: F#2 + C#3 (tritone – eerie)
            3 => (110.0, 164.8),    // Earth: A2 + E3 (perfect fifth)
            _ => (98.0, 146.8),     // Mono: G2 + D3
        };

        let s1 = rodio::source::SineWave::new(f1)
            .amplify(0.4)
            .fade_in(Duration::from_secs(3));
        self.ambient.append(s1);

        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.03);
            let s2 = rodio::source::SineWave::new(f2)
                .amplify(0.25)
                .fade_in(Duration::from_secs(4));
            s.append(s2);
            s.detach();
        }
        // Sub-harmonic
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.02);
            let s3 = rodio::source::SineWave::new(f1 * 0.5)
                .amplify(0.3)
                .fade_in(Duration::from_secs(5));
            s.append(s3);
            s.detach();
        }
    }

    fn play_palette_shift(&self) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.2);
            let tone = rodio::source::SineWave::new(440.0)
                .take_duration(Duration::from_millis(80));
            let tone2 = rodio::source::SineWave::new(660.0)
                .take_duration(Duration::from_millis(120));
            s.append(tone);
            s.append(tone2);
            s.detach();
        }
    }

    fn play_power_change(&self, power: f32) {
        if let Ok(s) = Sink::try_new(&self.handle) {
            s.set_volume(0.15);
            let freq = 200.0 + power * 40.0;
            let tone = rodio::source::SineWave::new(freq)
                .take_duration(Duration::from_millis(60));
            s.append(tone);
            s.detach();
        }
    }
}

// ─── Explorer State ────────────────────────────────────────────────────────

struct FractalExplorer {
    theta: f32,         // horizontal orbit angle
    phi: f32,           // vertical orbit angle
    radius: f32,        // orbit distance
    power: f32,         // Mandelbulb power (2–12)
    palette: u32,       // 0–4
    max_iter: f32,      // fractal iterations (8–20)
    audio_reactive: bool,
    audio_level: f32,
    mouse_down: bool,
    mouse_sensitivity: f32,
    uniform: FractalUniform,
    buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
}

impl FractalExplorer {
    fn new() -> Self {
        Self {
            theta: 0.3,
            phi: 0.4,
            radius: 3.5,
            power: 8.0,
            palette: 0,
            max_iter: 12.0,
            audio_reactive: false,
            audio_level: 0.0,
            mouse_down: false,
            mouse_sensitivity: 0.005,
            uniform: FractalUniform {
                camera_pos: [0.0; 4],
                camera_target: [0.0; 4],
                resolution: [0.0; 4],
                params: [0.0; 4],
            },
            buffer: None,
            bind_group: None,
        }
    }

    fn update_uniform(&mut self, width: f32, height: f32, time: f32) {
        // Orbit camera
        let cam_x = self.radius * self.theta.cos() * self.phi.cos();
        let cam_y = self.radius * self.phi.sin();
        let cam_z = self.radius * self.theta.sin() * self.phi.cos();

        self.uniform.camera_pos = [cam_x, cam_y, cam_z, 1.8]; // fov
        self.uniform.camera_target = [0.0, 0.0, 0.0, self.radius];
        self.uniform.resolution = [width, height, time, self.power];

        // Audio reactive level with smooth decay
        if self.audio_reactive {
            self.audio_level = (self.audio_level + 0.02).min(1.0);
        } else {
            self.audio_level = (self.audio_level - 0.05).max(0.0);
        }

        self.uniform.params = [
            self.palette as f32,
            self.audio_level,
            self.max_iter,
            0.0,
        ];
    }

    fn handle_keyboard(&mut self, key: KeyCode) -> Option<&'static str> {
        match key {
            KeyCode::Digit1 => { self.palette = 0; return Some("🔥 Fire palette"); }
            KeyCode::Digit2 => { self.palette = 1; return Some("🌊 Ocean palette"); }
            KeyCode::Digit3 => { self.palette = 2; return Some("🌌 Nebula palette"); }
            KeyCode::Digit4 => { self.palette = 3; return Some("🌍 Earth palette"); }
            KeyCode::Digit5 => { self.palette = 4; return Some("⬛ Monochrome palette"); }
            KeyCode::KeyP => {
                self.power = (self.power + 0.5).min(12.0);
                println!("Power: {:.1}", self.power);
                return Some("power_up");
            }
            KeyCode::KeyO => {
                self.power = (self.power - 0.5).max(2.0);
                println!("Power: {:.1}", self.power);
                return Some("power_down");
            }
            KeyCode::KeyA => {
                self.audio_reactive = !self.audio_reactive;
                if self.audio_reactive {
                    return Some("🎵 Audio-reactive ON");
                } else {
                    return Some("🔇 Audio-reactive OFF");
                }
            }
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.max_iter = (self.max_iter + 1.0).min(20.0);
                println!("Iterations: {}", self.max_iter as u32);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.max_iter = (self.max_iter - 1.0).max(6.0);
                println!("Iterations: {}", self.max_iter as u32);
            }
            KeyCode::KeyR => {
                self.theta = 0.3; self.phi = 0.4; self.radius = 3.5;
                self.power = 8.0; self.palette = 0; self.max_iter = 12.0;
                self.audio_reactive = false;
                return Some("↻ View reset");
            }
            _ => {}
        }
        None
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.mouse_down {
            self.theta += dx as f32 * self.mouse_sensitivity;
            self.phi = (self.phi + dy as f32 * self.mouse_sensitivity).clamp(-1.5, 1.5);
        }
    }

    fn handle_scroll(&mut self, delta: f32) {
        self.radius = (self.radius - delta * 0.2).clamp(1.5, 8.0);
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

async fn run() {
    env_logger::init();

    println!("🌀 MANDELBULB FRACTAL EXPLORER 🌀");
    println!("═══════════════════════════════════");
    println!("Controls:");
    println!("  Mouse Drag     – Orbit camera");
    println!("  Scroll / +/-   – Zoom / Change iterations");
    println!("  1–5            – Color palette (Fire, Ocean, Nebula, Earth, Mono)");
    println!("  P / O          – Increase / Decrease fractal power");
    println!("  A              – Toggle audio-reactive mode");
    println!("  R              – Reset view");
    println!("  ESC            – Quit");
    println!();
    println!("Tip: Change power (P/O) to morph the fractal shape!");
    println!("     Low power (2-4) = smooth blobs, High (8-12) = intricate details");

    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("Mandelbulb Fractal Explorer")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop).unwrap();
    let window = Arc::new(window);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface), force_fallback_adapter: false,
    }).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Fractal"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
    }, None).await.unwrap();

    let size = window.inner_size();
    let mut config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);

    let mut explorer = FractalExplorer::new();
    let mut audio = FractalAudio::new();
    if audio.is_some() { println!("🔊 Audio initialized"); }

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Fractal Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fractal_explorer.wgsl").into()),
    });

    // Uniform
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fractal Uniform"),
        contents: bytemuck::cast_slice(&[explorer.uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });
    explorer.buffer = Some(buf);
    explorer.bind_group = Some(bg);

    // Pipeline (no depth needed for fullscreen ray march)
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
    });

    let start_time = std::time::Instant::now();
    let mut frame_count = 0u32;

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
                    if event.state != ElementState::Pressed { return; }
                    if let PhysicalKey::Code(kc) = event.physical_key {
                        if kc == KeyCode::Escape { target.exit(); return; }
                        if let Some(msg) = explorer.handle_keyboard(kc) {
                            if msg.starts_with("power") {
                                if let Some(ref a) = audio { a.play_power_change(explorer.power); }
                            } else {
                                println!("{}", msg);
                                if msg.contains("palette") {
                                    if let Some(ref a) = audio { a.play_palette_shift(); }
                                }
                            }
                        }
                    }
                }
                WinitWindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    explorer.mouse_down = *state == ElementState::Pressed;
                }
                WinitWindowEvent::MouseWheel { delta, .. } => {
                    let d = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                    };
                    explorer.handle_scroll(d);
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let elapsed = start_time.elapsed().as_secs_f32();

                    explorer.update_uniform(config.width as f32, config.height as f32, elapsed);

                    // Update audio drone
                    if let Some(ref mut a) = audio {
                        a.update_drone(explorer.palette);
                    }

                    if let Some(ref b) = explorer.buffer {
                        queue.write_buffer(b, 0, bytemuck::cast_slice(&[explorer.uniform]));
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
                        if let Some(ref bg) = explorer.bind_group { rp.set_bind_group(0, bg, &[]); }
                        rp.draw(0..6, 0..1); // fullscreen quad
                    }
                    queue.submit(std::iter::once(enc.finish()));
                    output.present();

                    if frame_count % 60 == 0 {
                        let fps = frame_count as f32 / elapsed;
                        let pal_name = match explorer.palette {
                            0 => "🔥Fire", 1 => "🌊Ocean", 2 => "🌌Nebula",
                            3 => "🌍Earth", _ => "⬛Mono",
                        };
                        let ar = if explorer.audio_reactive { "🎵" } else { "" };
                        println!("{} | P={:.1} | Z={:.1} | I={} {} | {:.0}fps",
                            pal_name, explorer.power, explorer.radius,
                            explorer.max_iter as u32, ar, fps);
                    }
                    window.request_redraw();
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                explorer.handle_mouse_motion(delta.0, delta.1);
            }
            Event::AboutToWait => { window.request_redraw(); }
            _ => {}
        }
    });
}

fn main() { pollster::block_on(run()); }
