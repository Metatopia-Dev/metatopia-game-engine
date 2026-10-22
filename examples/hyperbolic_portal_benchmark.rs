//! Metatopia Engine — Non-Euclidean WGSL Shader & Hyperbolic Portal Transition Benchmark
//!
//! Interactive GPU ray-marched non-Euclidean manifold explorer and portal transition
//! benchmark suite supporting:
//!   - Recursive GPU portal ray traversal (0 to 16 bounces across E³, H³, S³ charts)
//!   - Analytical 3D conformal Christoffel symbol RK2 geodesic ray integration
//!   - Möbius gyrovector domain folding for H³ {4,3,5} hyperbolic honeycombs
//!   - Zero-seam camera portal crossing (`Manifold::check_portal_crossing`)
//!   - Headless/CLI GPU + CPU benchmark runner (`--bench`)
//!
//! Usage:
//!   Interactive Window:
//!     cargo run --release --example hyperbolic_portal_benchmark
//!   Automated CLI + Offscreen GPU Benchmark Suite:
//!     cargo run --release --example hyperbolic_portal_benchmark -- --bench

use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};
use metatopia_engine::manifold::{
    ChartId, ChristoffelSymbols, GeometryType, Manifold, ManifoldRaycaster, Metric, PortalShape,
};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent as WinitWindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder as WinitWindowBuilder,
};

const SHADER_SRC: &str = include_str!("../shaders/hyperbolic_portal_benchmark.wgsl");

// ─── GPU Uniform Buffer (16-byte aligned) ──────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PortalBenchmarkUniform {
    pub camera_pos: [f32; 4],    // xyz = pos, w = fov
    pub camera_dir: [f32; 4],    // xyz = forward, w = active_chart_id
    pub camera_up: [f32; 4],     // xyz = up, w = curvature_k
    pub resolution: [f32; 4],    // x = width, y = height, z = time, w = scene_mode
    pub params: [f32; 4],        // x = max_portal_bounces, y = geodesic_on, z = heatmap_mode, w = max_steps
    pub portal0_src: [f32; 4],   // xyz = src_center, w = src_chart
    pub portal0_dst: [f32; 4],   // xyz = dst_center, w = dst_chart
    pub portal0_norm: [f32; 4],  // xyz = normal, w = yaw_rotation_rad
    pub portal1_src: [f32; 4],
    pub portal1_dst: [f32; 4],
    pub portal1_norm: [f32; 4],
    pub portal2_src: [f32; 4],
    pub portal2_dst: [f32; 4],
    pub portal2_norm: [f32; 4],
    pub portal3_src: [f32; 4],
    pub portal3_dst: [f32; 4],
    pub portal3_norm: [f32; 4],
    pub telemetry: [f32; 4],     // x = fps_norm, y = bounce_ratio, z = cpu_throughput, w = bench_active
}

// ─── Benchmark Scene Presets ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkScene {
    ImpossibleCorridorLoop = 0,
    HyperbolicHoneycomb = 1,
    MultiChartCurvatureBridge = 2,
    RecursiveMirrorTunnel = 3,
}

impl BenchmarkScene {
    pub fn name(self) -> &'static str {
        match self {
            Self::ImpossibleCorridorLoop => "Impossible Non-Euclidean Corridor & 4-Chart Loop",
            Self::HyperbolicHoneycomb => "Hyperbolic H3 {4,3,5} Conformal Honeycomb Lattice",
            Self::MultiChartCurvatureBridge => "Multi-Chart Curvature Bridge (E3 -> H3 -> S3)",
            Self::RecursiveMirrorTunnel => "Recursive Portal Mirror Tunnel Stress Test",
        }
    }

    pub fn from_index(idx: u32) -> Self {
        match idx % 4 {
            0 => Self::ImpossibleCorridorLoop,
            1 => Self::HyperbolicHoneycomb,
            2 => Self::MultiChartCurvatureBridge,
            _ => Self::RecursiveMirrorTunnel,
        }
    }
}

/// Build the multi-chart CPU `Manifold` matching the GPU WGSL portal configuration
pub fn build_benchmark_manifold(scene: BenchmarkScene) -> (Manifold, [[f32; 4]; 12]) {
    let mut manifold = Manifold::new(); // ChartId(0) = Euclidean Sanctuary
    let chart_hyp = manifold.add_chart(GeometryType::Hyperbolic); // ChartId(1) = Hyperbolic H³
    let chart_sph = manifold.add_chart(GeometryType::Spherical);  // ChartId(2) = Spherical S³
    let chart_esc = manifold.add_chart(GeometryType::Custom);     // ChartId(3) = Escher Twisted

    let portal_uniforms = match scene {
        BenchmarkScene::RecursiveMirrorTunnel => {
            // Coaxial 4-chart infinity portal loop at z = -3.0 -> z = +2.8 (5.8m spacing for 10-16 visible nested bounces!)
            let _ = manifold.create_portal_with_bounds(
                ChartId(0),
                chart_hyp,
                Point3::new(0.0, 1.5, -3.0),
                Point3::new(0.0, 1.5, 2.8),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_hyp,
                chart_sph,
                Point3::new(0.0, 1.5, -3.0),
                Point3::new(0.0, 1.5, 2.8),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_sph,
                chart_esc,
                Point3::new(0.0, 1.5, -3.0),
                Point3::new(0.0, 1.5, 2.8),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_esc,
                ChartId(0),
                Point3::new(0.0, 1.5, -3.0),
                Point3::new(0.0, 1.5, 2.8),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            [
                // Portal 0: Chart 0 -> Chart 1 (coaxial along Z; shader applies helical roll per bounce)
                [0.0, 1.5, -3.0, 0.0],
                [0.0, 1.5, 2.8, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 1: Chart 1 -> Chart 2
                [0.0, 1.5, -3.0, 1.0],
                [0.0, 1.5, 2.8, 2.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 2: Chart 2 -> Chart 3
                [0.0, 1.5, -3.0, 2.0],
                [0.0, 1.5, 2.8, 3.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 3: Chart 3 -> Chart 0
                [0.0, 1.5, -3.0, 3.0],
                [0.0, 1.5, 2.8, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        }
        _ => {
            // 4-Chart Non-Euclidean Loop: Chart 0 -> Chart 1 -> Chart 2 -> Chart 3 -> Chart 0
            let _ = manifold.create_portal_with_bounds(
                ChartId(0),
                chart_hyp,
                Point3::new(0.0, 1.5, -4.8),
                Point3::new(0.0, 1.5, 4.5),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_hyp,
                chart_sph,
                Point3::new(0.0, 1.5, -4.8),
                Point3::new(0.0, 1.5, 4.5),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_sph,
                chart_esc,
                Point3::new(0.0, 1.5, -4.8),
                Point3::new(0.0, 1.5, 4.5),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            let _ = manifold.create_portal_with_bounds(
                chart_esc,
                ChartId(0),
                Point3::new(0.0, 1.5, -4.8),
                Point3::new(0.0, 1.5, 4.5),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                3.3,
                4.7,
                PortalShape::Rectangular,
            );
            [
                // Portal 0: Chart 0 -> Chart 1 (Hyperbolic H³)
                [0.0, 1.5, -4.8, 0.0],
                [0.0, 1.5, 4.5, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 1: Chart 1 -> Chart 2 (Spherical S³)
                [0.0, 1.5, -4.8, 1.0],
                [0.0, 1.5, 4.5, 2.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 2: Chart 2 -> Chart 3 (Escher Manifold)
                [0.0, 1.5, -4.8, 2.0],
                [0.0, 1.5, 4.5, 3.0],
                [0.0, 0.0, 1.0, 0.0],
                // Portal 3: Chart 3 -> Chart 0 (Euclidean Sanctuary)
                [0.0, 1.5, -4.8, 3.0],
                [0.0, 1.5, 4.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        }
    };

    (manifold, portal_uniforms)
}

// ─── Interactive Explorer & Benchmark State ────────────────────────────────

struct HyperbolicPortalApp {
    scene: BenchmarkScene,
    manifold: Manifold,
    portal_data: [[f32; 4]; 12],
    camera_pos: Point3<f32>,
    yaw: f32,
    pitch: f32,
    active_chart: ChartId,
    curvature_k: f32,
    max_portal_bounces: u32,
    geodesic_enabled: bool,
    heatmap_mode: u32,
    max_steps: f32,
    auto_tour: bool,
    mouse_captured: bool,
    keys_held: [bool; 6], // W, A, S, D, Space, Shift
    total_portal_crossings: usize,
    fps_smoothed: f32,
    last_cpu_krays_sec: f32,
    uniform: PortalBenchmarkUniform,
}

impl HyperbolicPortalApp {
    fn new(scene: BenchmarkScene) -> Self {
        let (manifold, portal_data) = build_benchmark_manifold(scene);
        let mut app = Self {
            scene,
            manifold,
            portal_data,
            camera_pos: Point3::new(0.0, 1.6, 3.8),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            active_chart: ChartId(0),
            curvature_k: -0.85,
            max_portal_bounces: 6,
            geodesic_enabled: true,
            heatmap_mode: 0,
            max_steps: 96.0,
            auto_tour: false,
            mouse_captured: false,
            keys_held: [false; 6],
            total_portal_crossings: 0,
            fps_smoothed: 60.0,
            last_cpu_krays_sec: 0.0,
            uniform: bytemuck::Zeroable::zeroed(),
        };
        app.sync_portal_uniforms();
        app
    }

    fn set_scene(&mut self, scene: BenchmarkScene) {
        self.scene = scene;
        let (manifold, portal_data) = build_benchmark_manifold(scene);
        self.manifold = manifold;
        self.portal_data = portal_data;
        self.active_chart = ChartId(0);
        self.camera_pos = Point3::new(0.0, 1.6, 2.4);
        self.yaw = -std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
        self.sync_portal_uniforms();
        println!(
            "[ManifoldBench] Switched to Scene {}: {}",
            scene as u32 + 1,
            scene.name()
        );
    }

    fn sync_portal_uniforms(&mut self) {
        self.uniform.portal0_src = self.portal_data[0];
        self.uniform.portal0_dst = self.portal_data[1];
        self.uniform.portal0_norm = self.portal_data[2];
        self.uniform.portal1_src = self.portal_data[3];
        self.uniform.portal1_dst = self.portal_data[4];
        self.uniform.portal1_norm = self.portal_data[5];
        self.uniform.portal2_src = self.portal_data[6];
        self.uniform.portal2_dst = self.portal_data[7];
        self.uniform.portal2_norm = self.portal_data[8];
        self.uniform.portal3_src = self.portal_data[9];
        self.uniform.portal3_dst = self.portal_data[10];
        self.uniform.portal3_norm = self.portal_data[11];
    }

    fn forward_dir(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn update(&mut self, dt: f32, elapsed: f32, width: u32, height: u32) {
        let prev_pos = self.camera_pos;
        let fwd = self.forward_dir();
        let right = fwd.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();

        if self.auto_tour {
            // Smooth forward walk through the non-Euclidean portal loop with centered sightline
            self.camera_pos += Vector3::new(0.0, 0.0, -3.5 * dt);
            self.yaw = -std::f32::consts::FRAC_PI_2 + 0.042 * (elapsed * 0.85).sin();
            self.pitch = 0.022 * (elapsed * 1.1).cos();
        } else {
            let speed = 4.2 * dt;
            if self.keys_held[0] {
                self.camera_pos += fwd * speed;
            }
            if self.keys_held[2] {
                self.camera_pos -= fwd * speed;
            }
            if self.keys_held[1] {
                self.camera_pos -= right * speed;
            }
            if self.keys_held[3] {
                self.camera_pos += right * speed;
            }
            if self.keys_held[4] {
                self.camera_pos.y += speed;
            }
            if self.keys_held[5] {
                self.camera_pos.y -= speed;
            }
        }

        // Check zero-seam portal crossing in the CPU Manifold
        if let Some((new_pos, new_dir, target_chart, portal_id)) = self
            .manifold
            .check_portal_crossing(self.active_chart, prev_pos, self.camera_pos, fwd)
        {
            self.camera_pos = new_pos;
            self.yaw = new_dir.z.atan2(new_dir.x);
            self.pitch = new_dir.y.clamp(-1.4, 1.4).asin();
            self.active_chart = target_chart;
            self.total_portal_crossings += 1;
            println!(
                "[PortalCrossing #{:02}] Traversed {:?} -> Active Chart {:?} at ({:.2}, {:.2}, {:.2})",
                self.total_portal_crossings,
                portal_id,
                self.active_chart,
                self.camera_pos.x,
                self.camera_pos.y,
                self.camera_pos.z
            );
        }

        let inst_fps = if dt > 1e-5 { 1.0 / dt } else { 60.0 };
        self.fps_smoothed = self.fps_smoothed * 0.92 + inst_fps * 0.08;

        let dir = self.forward_dir();
        self.uniform.camera_pos = [self.camera_pos.x, self.camera_pos.y, self.camera_pos.z, 1.25];
        self.uniform.camera_dir = [dir.x, dir.y, dir.z, self.active_chart.0 as f32];
        self.uniform.camera_up = [0.0, 1.0, 0.0, self.curvature_k];
        self.uniform.resolution = [
            width as f32,
            height as f32,
            elapsed,
            self.scene as u32 as f32,
        ];
        self.uniform.params = [
            self.max_portal_bounces as f32,
            if self.geodesic_enabled { 1.0 } else { 0.0 },
            self.heatmap_mode as f32,
            self.max_steps,
        ];
        self.uniform.telemetry = [
            (self.fps_smoothed / 120.0).clamp(0.0, 1.0),
            (self.max_portal_bounces as f32 / 16.0).clamp(0.0, 1.0),
            (self.last_cpu_krays_sec / 500.0).clamp(0.0, 1.0),
            0.0,
        ];
    }
}

// ─── Headless GPU + CPU Benchmark Runner (`--bench`) ───────────────────────

fn run_headless_benchmark_suite() {
    println!("================================================================================");
    println!("  METATOPIA ENGINE — NON-EUCLIDEAN WGSL & HYPERBOLIC PORTAL BENCHMARK SUITE");
    println!("================================================================================");

    // 1. Initialize headless WGPU adapter & compile `hyperbolic_portal_benchmark.wgsl`
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to acquire GPU adapter for benchmark");

    let adapter_info = adapter.get_info();
    println!(
        "GPU Adapter     : {} ({:?}, {:?})",
        adapter_info.name, adapter_info.backend, adapter_info.device_type
    );

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("PortalBenchmarkDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create WGPU device");

    let shader_t0 = Instant::now();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hyperbolic_portal_benchmark.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let shader_compile_ms = shader_t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "WGSL Shader     : shaders/hyperbolic_portal_benchmark.wgsl compiled in {:.2} ms",
        shader_compile_ms
    );

    // Create offscreen 1280x720 render target for real GPU ray-march timing
    let width = 1280u32;
    let height = 720u32;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("OffscreenPortalBenchTarget"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut app = HyperbolicPortalApp::new(BenchmarkScene::RecursiveMirrorTunnel);
    app.update(0.016, 1.25, width, height);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PortalBenchUniformBuffer"),
        contents: bytemuck::cast_slice(&[app.uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("PortalBenchBGL"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PortalBenchBG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PortalBenchPipelineLayout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PortalBenchPipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    println!("\n--- 1. Offscreen GPU WGSL Ray-Marching + Portal Recursion Sweep (1280x720) ---");
    println!(
        "{:<34} | {:>8} | {:>10} | {:>10} | {:>12}",
        "Scene / Mode", "Bounces", "Avg (ms)", "p95 (ms)", "Est. FPS"
    );
    println!("{}", "-".repeat(86));

    let bounce_levels = [0u32, 2, 4, 8, 12, 16];
    let frames_per_level = 20usize;

    for &bounces in &bounce_levels {
        app.max_portal_bounces = bounces;
        let mut frame_times_ms = Vec::with_capacity(frames_per_level);

        for f in 0..frames_per_level {
            app.update(0.016, f as f32 * 0.05, width, height);
            queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[app.uniform]));

            let t0 = Instant::now();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BenchEncoder"),
            });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("BenchPass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass.draw(0..6, 0..1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            frame_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        frame_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg_ms: f64 = frame_times_ms.iter().sum::<f64>() / (frames_per_level as f64);
        let p95_ms = frame_times_ms[(frames_per_level as f64 * 0.95) as usize - 1];
        let est_fps = 1000.0 / avg_ms.max(0.001);

        println!(
            "{:<34} | {:>8} | {:>8.3}ms | {:>8.3}ms | {:>10.1} FPS",
            "Recursive Mirror Tunnel (RK2)", bounces, avg_ms, p95_ms, est_fps
        );
    }

    println!("\n--- 2. CPU ManifoldRaycaster & RK2 Christoffel Geodesic Benchmark (4096 rays) ---");
    println!(
        "{:<34} | {:>8} | {:>12} | {:>12} | {:>12}",
        "Manifold Topology", "Bounces", "PortalJumps", "KRays/sec", "MSegments/s"
    );
    println!("{}", "-".repeat(86));

    for scene_idx in 0..4 {
        let scene = BenchmarkScene::from_index(scene_idx);
        let (manifold, _) = build_benchmark_manifold(scene);
        let metrics = ManifoldRaycaster::benchmark_portal_transitions(
            &manifold,
            ChartId(0),
            4096,
            8,
            0.25,
        );
        println!(
            "{:<34} | {:>8} | {:>12} | {:>10.2} K | {:>10.2} M",
            &scene.name()[..scene.name().len().min(34)],
            metrics.max_portal_jumps,
            metrics.total_portal_jumps,
            metrics.rays_per_sec / 1_000.0,
            metrics.segments_per_sec / 1_000_000.0
        );
    }

    // 3. Verify Christoffel Symbol & Möbius Gyrovector Invariants
    let p_test = Point3::new(0.3, 0.4, 0.0);
    let cs = ChristoffelSymbols::from_conformal_point(p_test, -1.0);
    let u_vec = Vector3::new(0.25, -0.15, 0.10);
    let mobius_inv_err = Metric::mobius_add(u_vec, -u_vec, -1.0).magnitude();
    println!("\n--- 3. Differential Geometry & Non-Euclidean Invariant Checks ---");
    println!(
        "Poincaré Christoffel Γ⁰₀₀(0.3,0.4,0) : {:.6} (exact analytical = 0.800000)",
        cs.gamma[0][0][0]
    );
    println!(
        "Möbius Gyrovector Inverse ||u ⊕ (-u)||: {:.2e} (exact identity = 0.00e0)",
        mobius_inv_err
    );
    println!("================================================================================");
}

// ─── 5x7 Bitmap Font for Recorded Gameplay Video HUD ───────────────────────

fn glyph_5x7(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '(' | '[' => [0b00110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00110],
        ')' | ']' => [0b01100, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01100],
        '{' => [0b00011, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00011],
        '}' => [0b11000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b11000],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        _ => [0; 7],
    }
}

#[allow(dead_code)]
fn draw_rect_rgba(buf: &mut [u8], width: u32, height: u32, x0: u32, y0: u32, w: u32, h: u32, rgba: [u8; 4]) {
    let a = rgba[3] as u32;
    let inv_a = 255 - a;
    for y in y0..(y0 + h).min(height) {
        for x in x0..(x0 + w).min(width) {
            let idx = ((y * width + x) * 4) as usize;
            buf[idx] = ((rgba[0] as u32 * a + buf[idx] as u32 * inv_a) / 255) as u8;
            buf[idx + 1] = ((rgba[1] as u32 * a + buf[idx + 1] as u32 * inv_a) / 255) as u8;
            buf[idx + 2] = ((rgba[2] as u32 * a + buf[idx + 2] as u32 * inv_a) / 255) as u8;
            buf[idx + 3] = 255;
        }
    }
}

fn draw_text_rgba(buf: &mut [u8], width: u32, height: u32, x0: u32, y0: u32, scale: u32, text: &str, rgb: [u8; 3]) {
    let mut cx = x0;
    for ch in text.chars() {
        let bits = glyph_5x7(ch);
        for (row, &b) in bits.iter().enumerate() {
            for col in 0..5u32 {
                if (b & (1 << (4 - col))) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cx + col * scale + sx;
                            let py = y0 + (row as u32) * scale + sy;
                            if px < width && py < height {
                                let idx = ((py * width + px) * 4) as usize;
                                buf[idx] = rgb[0];
                                buf[idx + 1] = rgb[1];
                                buf[idx + 2] = rgb[2];
                                buf[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Synthesizes a full 3-minute (180.0-second, 96 bars = 384 beats @ 128 BPM) 48 kHz 16-bit stereo
/// futuristic melodic techno / progressive house song ("GEODESIC HORIZON (EXTENDED MIX)" in F Minor)
/// across 9 distinct musical movements and writes it to `wav_path`.
fn synthesize_geodesic_horizon_wav(wav_path: &str, duration_secs: f32) {
    use std::io::Write;
    let sample_rate = 48_000u32;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let bpm = 128.0f32;
    let beat_sec = 60.0 / bpm; // 0.46875s per beat
    let step16_sec = beat_sec * 0.25; // 0.1171875s per 16th note
    let tau = std::f32::consts::TAU;

    // Primary 16-step F-minor futuristic melodic techno arpeggio (frequencies in Hz)
    // F4=349.23, G4=392.00, Ab4=415.30, Bb4=466.16, C5=523.25, Db5=554.37, Eb5=622.25, F5=698.46
    let arp_notes_a: [f32; 16] = [
        349.23, 523.25, 415.30, 392.00, 349.23, 523.25, 622.25, 523.25, 277.18, 415.30, 698.46,
        622.25, 261.63, 392.00, 466.16, 523.25,
    ];
    // Secondary variation arpeggio for Breakdown & Climax Drop II (soaring Abmaj7 / Fm9 intervals)
    let arp_notes_b: [f32; 16] = [
        415.30, 523.25, 698.46, 622.25, 415.30, 622.25, 783.99, 698.46, 349.23, 523.25, 830.61,
        698.46, 392.00, 523.25, 622.25, 698.46,
    ];

    // 16-step Lead Synth Anthem Melody (each step = 1/8th note = 2 sixteenth steps; 0.0 = rest)
    let lead_melody: [f32; 16] = [
        698.46, // F5
        698.46, // F5
        622.25, // Eb5
        523.25, // C5
        415.30, // Ab4
        466.16, // Bb4
        523.25, // C5
        0.0,    // breath
        554.37, // Db5
        698.46, // F5
        622.25, // Eb5
        523.25, // C5
        392.00, // G4
        415.30, // Ab4
        523.25, // C5
        392.00, // G4
    ];

    // High-register counter-melody harmony for Climax Drop II (Bars 72..88)
    let counter_melody: [f32; 16] = [
        830.61,  // Ab5
        1046.50, // C6
        932.33,  // Bb5
        783.99,  // G5
        698.46,  // F5
        783.99,  // G5
        830.61,  // Ab5
        698.46,  // F5
        830.61,  // Ab5
        1108.73, // Db6
        1046.50, // C6
        830.61,  // Ab5
        783.99,  // G5
        830.61,  // Ab5
        932.33,  // Bb5
        1046.50, // C6
    ];

    // Dotted-eighth stereo ping-pong delay buffer (3/4 of a beat = 0.3515625s)
    let delay_samples = ((beat_sec * 0.75) * sample_rate as f32) as usize;
    let mut delay_l = vec![0.0f32; delay_samples];
    let mut delay_r = vec![0.0f32; delay_samples];
    let mut delay_idx = 0usize;

    let mut kick_phase = 0.0f32;
    let mut pcm_bytes = Vec::with_capacity(44 + num_samples * 4);

    // Write standard 44-byte RIFF WAVE header (48kHz, 16-bit stereo)
    let data_bytes = (num_samples * 4) as u32;
    let file_size = 36 + data_bytes;
    pcm_bytes.extend_from_slice(b"RIFF");
    pcm_bytes.extend_from_slice(&file_size.to_le_bytes());
    pcm_bytes.extend_from_slice(b"WAVEfmt ");
    pcm_bytes.extend_from_slice(&16u32.to_le_bytes());
    pcm_bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    pcm_bytes.extend_from_slice(&2u16.to_le_bytes()); // 2 channels (Stereo)
    pcm_bytes.extend_from_slice(&sample_rate.to_le_bytes());
    pcm_bytes.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    pcm_bytes.extend_from_slice(&4u16.to_le_bytes()); // block align
    pcm_bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    pcm_bytes.extend_from_slice(b"data");
    pcm_bytes.extend_from_slice(&data_bytes.to_le_bytes());

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let beat_f = t / beat_sec;
        let beat_idx = beat_f.floor() as usize;
        let beat_pos = beat_f.fract(); // 0..1 within beat
        let bar_idx = beat_idx / 4; // 0..96 across 180s

        let step16_f = t / step16_sec;
        let step16_idx = step16_f.floor() as usize;
        let step16_pos = step16_f.fract(); // 0..1 within 16th note
        let step_in_beat = step16_idx % 4;

        // 9-Movement Song Structure Flags:
        // M1 Intro: Bars 0..8 | M2 Acid Groove: Bars 8..16 | M3 Verse+Lead: Bars 16..32
        // M4 Breakdown: Bars 32..40 | M5 Buildup I: Bars 40..48 | M6 Drop I: Bars 48..64
        // M7 S3 Bridge + Riser II: Bars 64..72 | M8 Climax Drop II: Bars 72..88 | M9 Outro: Bars 88..96
        let in_intro = bar_idx < 8;
        let in_breakdown = (32..40).contains(&bar_idx);
        let in_buildup_1 = (40..48).contains(&bar_idx);
        let in_drop_1 = (48..64).contains(&bar_idx);
        let in_bridge = (64..68).contains(&bar_idx);
        let in_buildup_2 = (68..72).contains(&bar_idx);
        let in_drop_2 = (72..88).contains(&bar_idx);
        let in_outro = bar_idx >= 88;

        // Pre-drop tension pauses & drumless breakdown sections
        let pre_drop_pause = (188..192).contains(&beat_idx)
            || (284..288).contains(&beat_idx)
            || (beat_idx == 62 || beat_idx == 63)
            || (beat_idx == 126 || beat_idx == 127);
        let kick_active = !in_breakdown && !in_buildup_1 && !in_buildup_2 && !pre_drop_pause
            && !(in_bridge && (beat_idx % 2 == 1));

        // Deterministic white/metallic noise source
        let n_seed = (i as u32).wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let white_noise = ((n_seed >> 9) as f32 / 4_194_304.0) - 1.0;

        // 1. 909 Four-on-the-Floor Techno Kick
        let kick_sample = if kick_active {
            let k_freq = 43.65 + 135.0 * (-beat_pos * 36.0).exp();
            kick_phase = (kick_phase + k_freq / sample_rate as f32).fract();
            let k_env = (-beat_pos * 5.8).exp();
            let click = (-beat_pos * 95.0).exp() * (tau * 1800.0 * t).sin() * 0.35;
            let punch_boost = if in_drop_1 || in_drop_2 { 0.84 } else { 0.76 };
            (((tau * kick_phase).sin() * 1.45 + click).tanh()) * k_env * punch_boost
        } else {
            kick_phase = 0.0;
            0.0
        };

        // Sidechain compression envelope (ducks bass, pads & leads on every kick hit)
        let sidechain = if kick_active {
            (1.0 - 0.86 * (-beat_pos * 7.2).exp()).clamp(0.14, 1.0)
        } else {
            1.0
        };

        // 2. Stereo Reese + Acid 303 Bassline in F Minor
        let bass_root = match bar_idx % 8 {
            0 | 2 | 4 | 6 => 43.65, // F1
            1 => 34.65,             // Db1
            3 => 38.89,             // Eb1
            5 => 32.70,             // C1
            _ => {
                if (beat_idx % 4) < 2 {
                    34.65
                } else {
                    38.89
                }
            }
        };
        let acid_mult = if step_in_beat == 2 || ((in_drop_1 || in_drop_2) && step_in_beat == 3) {
            2.0
        } else {
            1.0
        };
        let b_freq = bass_root * acid_mult;
        let sub_sine = (tau * bass_root * t).sin() * 0.65;
        let saw_l = 2.0 * ((b_freq * 0.995 * t).fract()) - 1.0;
        let saw_r = 2.0 * ((b_freq * 1.005 * t).fract()) - 1.0;

        let section_acid_boost = if in_drop_1 || in_drop_2 {
            0.32
        } else if in_buildup_1 {
            0.28 * ((bar_idx - 40) as f32 / 8.0)
        } else if in_intro || in_outro {
            -0.12
        } else {
            0.05
        };
        let acid_cutoff = (0.35 + 0.40 * ((t * 0.38).sin() * 0.5 + 0.5) + section_acid_boost)
            .clamp(0.15, 0.98);
        let acid_env = (-step16_pos * 6.5).exp() * acid_cutoff;
        let harm_3 = (tau * b_freq * 3.0 * t).sin() * acid_env * 0.45;
        let bass_gain = if in_breakdown {
            0.22 // Warm deep sub-drone during breakdown
        } else if pre_drop_pause {
            0.0
        } else {
            0.46
        };
        let bass_l = ((sub_sine + saw_l * 0.35 * acid_env + harm_3).tanh()) * sidechain * bass_gain;
        let bass_r = ((sub_sine + saw_r * 0.35 * acid_env + harm_3).tanh()) * sidechain * bass_gain;

        // 3. Crisp 909 Off-Beat Open Hi-Hat, 16th Shaker, Ride Cymbal & Snare/Clap
        let hats_enabled = !in_intro && !in_breakdown && !pre_drop_pause;
        let ohat_env = if !hats_enabled {
            0.0
        } else if step_in_beat == 2 {
            (-step16_pos * 13.5).exp() * if in_drop_1 || in_drop_2 { 0.28 } else { 0.22 }
        } else {
            (-step16_pos * 40.0).exp() * 0.07
        };
        let metallic = ((tau * 6340.0 * t).sin() * (tau * 9810.0 * t).cos() + white_noise) * 0.5;
        let ride_env = if (in_drop_1 || in_drop_2) && step_in_beat == 0 {
            (-step16_pos * 8.0).exp() * 0.11
        } else {
            0.0
        };
        let hat_sample = metallic * (ohat_env + ride_env);

        let clap_env = if beat_idx % 2 == 1 && kick_active && !in_intro && !in_outro {
            (-beat_pos * 22.0).exp() * 0.26
        } else {
            0.0
        };
        let clap_sample = (white_noise * 0.8 + (tau * 220.0 * t).sin() * 0.2) * clap_env;

        // 4. Snare Rolls, Pitch Risers & Drop Crash Impacts
        let (riser_l, riser_r) = if in_buildup_1 && !pre_drop_pause {
            // 8-Bar Progressive Riser I (Bars 40..48, Beats 160..188)
            let p = ((beat_f - 160.0) / 28.0).clamp(0.0, 1.0);
            let subdiv = if p < 0.5 {
                2.0
            } else if p < 0.8 {
                4.0
            } else {
                8.0
            };
            let roll_pos = (beat_f * subdiv).fract();
            let roll_hit = (-roll_pos * 16.0).exp() * (0.10 + 0.34 * p);
            let sweep_freq = 130.81 * (2.0f32).powf(p * 3.5);
            let sweep = (tau * sweep_freq * t).sin() * p * 0.22;
            (white_noise * roll_hit + sweep, white_noise * roll_hit - sweep)
        } else if in_buildup_2 && !pre_drop_pause {
            // 4-Bar Climax Riser II (Bars 68..72, Beats 272..284)
            let p = ((beat_f - 272.0) / 12.0).clamp(0.0, 1.0);
            let subdiv = if p < 0.5 { 4.0 } else { 8.0 };
            let roll_pos = (beat_f * subdiv).fract();
            let roll_hit = (-roll_pos * 18.0).exp() * (0.14 + 0.36 * p);
            let sweep_freq = 174.61 * (2.0f32).powf(p * 3.5);
            let sweep = (tau * sweep_freq * t).sin() * p * 0.24;
            (white_noise * roll_hit + sweep, white_noise * roll_hit - sweep)
        } else if matches!(beat_idx, 32 | 64 | 128 | 192 | 256 | 288 | 352) && beat_pos < 0.98 {
            // Crash cymbal impact on major structural transitions & Drops
            let crash = white_noise * (-beat_pos * 3.0).exp() * 0.32;
            (crash, crash * 0.94)
        } else {
            (0.0, 0.0)
        };

        // 5. Futuristic 2-Operator FM Pluck Arpeggiator with Stereo Ping-Pong Delay
        let arp_table = if in_breakdown || in_drop_2 {
            &arp_notes_b
        } else {
            &arp_notes_a
        };
        let base_arp = arp_table[step16_idx % 16];
        let arp_freq = if (in_drop_1 || in_drop_2 || in_breakdown) && (step16_idx % 4 == 2) {
            base_arp * 2.0
        } else {
            base_arp
        };
        let fm_mod_env =
            (-step16_pos * 18.0).exp() * (1.5 + 1.5 * ((t * 0.72).sin() * 0.5 + 0.5));
        let fm_mod = (tau * arp_freq * 2.0 * t).sin() * fm_mod_env;
        let arp_env = (-step16_pos * 9.5).exp();
        let arp_gain = if in_intro {
            0.12 + 0.12 * (t / 15.0).clamp(0.0, 1.0)
        } else {
            0.23
        };
        let arp_dry = (tau * arp_freq * t + fm_mod).sin() * arp_env * arp_gain;
        let pan = 0.5 + 0.38 * (step16_idx as f32 * 1.3).sin();
        let arp_in_l = arp_dry * (1.0 - pan).sqrt();
        let arp_in_r = arp_dry * pan.sqrt();

        // 6. Cyberpunk Lead Synth Hook + Climax Counter-Lead Harmony
        let lead_active = (16..32).contains(&bar_idx)
            || in_breakdown
            || in_drop_1
            || in_drop_2;
        let (lead_l, lead_r) = if lead_active {
            let step8_idx = step16_idx / 2;
            let step8_pos = (beat_f * 2.0).fract();
            let l_freq = lead_melody[step8_idx % 16];
            let mut ll = 0.0f32;
            let mut lr = 0.0f32;
            if l_freq > 0.0 {
                let vib = 1.0 + 0.0035 * (tau * 5.5 * t).sin();
                let lf = l_freq * vib;
                let l_env = (1.0 - (-step8_pos * 28.0).exp()) * (-step8_pos * 2.6).exp();
                let osc_l = 0.65 * (tau * lf * 0.998 * t).sin()
                    + 0.35 * (2.0 * ((lf * 1.002 * t).fract()) - 1.0);
                let osc_r = 0.65 * (tau * lf * 1.002 * t).sin()
                    + 0.35 * (2.0 * ((lf * 0.998 * t).fract()) - 1.0);
                let l_gain = if in_breakdown { 0.14 } else { 0.17 } * sidechain;
                ll += osc_l.tanh() * l_env * l_gain;
                lr += osc_r.tanh() * l_env * l_gain;
            }
            if in_drop_2 {
                // Add shimmering high-register counter-melody harmony in Climax Drop II
                let c_freq = counter_melody[step16_idx % 16];
                let c_env = (-step16_pos * 7.5).exp();
                let c_sig = (tau * c_freq * t).sin() * c_env * 0.095 * sidechain;
                ll += c_sig * 0.85;
                lr += c_sig * 1.15;
            }
            (ll, lr)
        } else {
            (0.0, 0.0)
        };

        let dl = delay_l[delay_idx];
        let dr = delay_r[delay_idx];
        // Cross-feedback ping-pong stereo delay fed by both Arpeggiator & Lead Synth
        delay_l[delay_idx] = arp_in_l + lead_l * 0.45 + dr * 0.45;
        delay_r[delay_idx] = arp_in_r + lead_r * 0.45 + dl * 0.45;
        delay_idx = (delay_idx + 1) % delay_samples;

        // 7. Lush Detuned Stereo Supersaw Chord Pad (Fm9 -> Dbmaj7 -> Eb6 -> C7sus4)
        let (pad_l, pad_r) = if bar_idx >= 4 {
            let chord: [f32; 4] = match bar_idx % 4 {
                0 => [174.61, 207.65, 261.63, 392.00], // Fm9
                1 => [138.59, 174.61, 207.65, 261.63], // Dbmaj7
                2 => [155.56, 196.00, 233.08, 293.66], // Eb6
                _ => [130.81, 174.61, 196.00, 233.08], // C7sus4
            };
            let mut pl = 0.0f32;
            let mut pr = 0.0f32;
            for &cf in &chord {
                pl += (tau * cf * 0.997 * t).sin() + 0.5 * (tau * cf * 1.994 * t).sin();
                pr += (tau * cf * 1.003 * t).sin() + 0.5 * (tau * cf * 2.006 * t).sin();
            }
            let pad_gain = if in_breakdown || in_bridge {
                0.068
            } else {
                0.048 * sidechain
            };
            (pl * pad_gain, pr * pad_gain)
        } else {
            (0.0, 0.0)
        };

        // Master Stereo Bus with soft-knee mastering limiter & smooth 4.0s outro fade
        let fade = if t < 1.2 {
            (t / 1.2).clamp(0.0, 1.0)
        } else if t > duration_secs - 4.0 {
            ((duration_secs - t) / 4.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mix_l = (kick_sample
            + bass_l
            + arp_in_l
            + lead_l
            + dl * 0.36
            + pad_l
            + hat_sample * 0.85
            + clap_sample
            + riser_l)
            * fade;
        let mix_r = (kick_sample
            + bass_r
            + arp_in_r
            + lead_r
            + dr * 0.36
            + pad_r
            + hat_sample * 1.15
            + clap_sample
            + riser_r)
            * fade;

        let out_l = ((mix_l * 1.12).tanh() * 32_000.0).clamp(-32_767.0, 32_767.0) as i16;
        let out_r = ((mix_r * 1.12).tanh() * 32_000.0).clamp(-32_767.0, 32_767.0) as i16;
        pcm_bytes.extend_from_slice(&out_l.to_le_bytes());
        pcm_bytes.extend_from_slice(&out_r.to_le_bytes());
    }

    let mut file = std::fs::File::create(wav_path).expect("Failed to create WAV file");
    file.write_all(&pcm_bytes).expect("Failed to write WAV file");
}

/// Computes instantaneous (kick_env, bass_env, arp_env) in [0.0, 1.0] at timestamp `t`
/// for 128 BPM audio-reactive shader modulation across all 96 bars (180.0s).
fn compute_audio_envelopes(t: f32) -> (f32, f32, f32) {
    let bpm = 128.0f32;
    let beat_sec = 60.0 / bpm;
    let step16_sec = beat_sec * 0.25;
    let beat_f = t / beat_sec;
    let beat_idx = beat_f.floor() as usize;
    let bar_idx = beat_idx / 4;
    let beat_pos = beat_f.fract();
    let step16_pos = (t / step16_sec).fract();

    let in_breakdown = (32..40).contains(&bar_idx);
    let in_buildup_1 = (40..48).contains(&bar_idx);
    let in_buildup_2 = (68..72).contains(&bar_idx);
    let pre_drop_pause = (188..192).contains(&beat_idx)
        || (284..288).contains(&beat_idx)
        || (beat_idx == 62 || beat_idx == 63)
        || (beat_idx == 126 || beat_idx == 127);
    let kick_active = !in_breakdown && !in_buildup_1 && !in_buildup_2 && !pre_drop_pause;

    let kick_env = if kick_active {
        (-beat_pos * 6.2).exp().clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bass_env = ((1.0 - kick_env * 0.75) * (0.45 + 0.55 * (-step16_pos * 4.5).exp())).clamp(0.0, 1.0);
    let arp_env = (-step16_pos * 7.5).exp().clamp(0.0, 1.0);
    (kick_env, bass_env, arp_env)
}

fn run_video_recording(output_path: &str) {
    use std::io::Write;
    use std::path::Path;
    use std::process::{Command, Stdio};

    let width = 1280u32;
    let height = 720u32;
    let fps = 30u32;
    let duration_secs = 180.0f32; // Full 3-minute song (96 bars = 384 beats @ 128 BPM)
    let total_frames = (fps as f32 * duration_secs) as u32; // 5,400 frames @ 30 FPS

    // Determine paths for keeping the high-bitrate Original Master AND the <20 MB compressed version
    let out_dir = Path::new(output_path)
        .parent()
        .unwrap_or_else(|| Path::new("/tmp"));
    let original_3min_path = out_dir
        .join("hyperbolic_portal_3min_original.mp4")
        .to_string_lossy()
        .to_string();
    let compressed_3min_path = out_dir
        .join("hyperbolic_portal_3min_under20mb.mp4")
        .to_string_lossy()
        .to_string();

    println!("================================================================================");
    println!(
        "  RECORDING FULL 3:00 (96-BAR @ 128 BPM) TECHNO/HOUSE VIDEOCLIP\n  - Original Master : {}\n  - <20 MB Compressed: {}\n  - Target Output   : {}",
        original_3min_path, compressed_3min_path, output_path
    );
    println!("================================================================================");

    let wav_path = "/tmp/geodesic_horizon_3min_128bpm.wav";
    println!(
        "[AudioSynth] Synthesizing 3:00 (180.0s) 128 BPM stereo melodic techno track -> {}",
        wav_path
    );
    synthesize_geodesic_horizon_wav(wav_path, duration_secs);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to acquire GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("VideoRecordDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create WGPU device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hyperbolic_portal_benchmark.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("VideoCaptureTarget"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (width * 4 + 255) & !255;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VideoReadbackBuffer"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut app = HyperbolicPortalApp::new(BenchmarkScene::ImpossibleCorridorLoop);
    app.auto_tour = true;
    app.max_portal_bounces = 8;

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VideoUniformBuffer"),
        contents: bytemuck::cast_slice(&[app.uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VideoBGL"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("VideoBG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("VideoPipelineLayout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("VideoPipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // Step 1: Encode the High-Bitrate Original 3-Minute Master (CRF 18 + 320k AAC)
    let mut ffmpeg = Command::new("/opt/homebrew/bin/ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", width, height),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-i",
            wav_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-shortest",
            "-movflags",
            "+faststart",
            &original_3min_path,
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn ffmpeg for original master");

    let ffmpeg_stdin = ffmpeg.stdin.as_mut().expect("Failed to open ffmpeg stdin");
    let dt = 1.0 / (fps as f32);
    let mut frame_rgba = vec![0u8; (width * height * 4) as usize];
    let mut prev_bar = usize::MAX;

    for frame_idx in 0..total_frames {
        let elapsed = frame_idx as f32 * dt;
        let (kick_env, bass_env, arp_env) = compute_audio_envelopes(elapsed);
        let bar_idx = ((elapsed / (60.0 / 128.0)) / 4.0).floor() as usize;

        // Beat-synchronized 96-bar (180.0s @ 128 BPM) 9-Movement Choreography:
        if bar_idx != prev_bar {
            prev_bar = bar_idx;
            match bar_idx {
                0 => {
                    // Movement I (0..15s): INTRO // POINCARE HORIZON
                    app.set_scene(BenchmarkScene::ImpossibleCorridorLoop);
                    app.auto_tour = true;
                    app.max_portal_bounces = 8;
                    app.max_steps = 104.0;
                    app.curvature_k = -0.82;
                    app.heatmap_mode = 0;
                }
                8 => {
                    // Movement II (15..30s): GROOVE // ACID REESE PROPULSION
                    app.set_scene(BenchmarkScene::HyperbolicHoneycomb);
                    app.auto_tour = true;
                    app.max_portal_bounces = 8;
                    app.max_steps = 104.0;
                    app.curvature_k = -1.22;
                    app.heatmap_mode = 0;
                }
                16 => {
                    // Movement III-A (30..45s): VERSE // MULTI-CHART CURVATURE BRIDGE
                    app.set_scene(BenchmarkScene::MultiChartCurvatureBridge);
                    app.auto_tour = true;
                    app.max_portal_bounces = 10;
                    app.max_steps = 104.0;
                    app.curvature_k = 0.92;
                    app.heatmap_mode = 0;
                }
                24 => {
                    // Movement III-B (45..60s): VERSE // 12-BOUNCE CORRIDOR ACCELERATION
                    app.set_scene(BenchmarkScene::ImpossibleCorridorLoop);
                    app.auto_tour = true;
                    app.max_portal_bounces = 12;
                    app.max_steps = 108.0;
                    app.curvature_k = -0.95;
                    app.heatmap_mode = 0;
                }
                32 => {
                    // Movement IV (60..75s): BREAKDOWN // ETHEREAL H3 CATHEDRAL
                    app.set_scene(BenchmarkScene::HyperbolicHoneycomb);
                    app.auto_tour = true;
                    app.max_portal_bounces = 10;
                    app.max_steps = 108.0;
                    app.curvature_k = -1.30;
                    app.heatmap_mode = 0;
                }
                40 => {
                    // Movement V (75..90s): BUILDUP I // GEODESIC SNARE RISER
                    app.set_scene(BenchmarkScene::MultiChartCurvatureBridge);
                    app.auto_tour = true;
                    app.max_portal_bounces = 12;
                    app.max_steps = 108.0;
                    app.curvature_k = -0.90;
                    app.heatmap_mode = 0;
                }
                48 => {
                    // Movement VI (90..120s): THE DROP I // 16-BOUNCE INFINITY TUNNEL
                    app.set_scene(BenchmarkScene::RecursiveMirrorTunnel);
                    app.auto_tour = true;
                    app.max_portal_bounces = 16;
                    app.max_steps = 120.0;
                    app.curvature_k = -1.08;
                    app.heatmap_mode = 0;
                }
                56 => {
                    // Bars 56..60 (105..112.5s): Full-3D Neon Spectrum Overdrive
                    app.heatmap_mode = 2;
                }
                60 => {
                    // Bars 60..64 (112.5..120s): Return to PBR 16-Bounce Infinity Tunnel
                    app.heatmap_mode = 0;
                }
                64 => {
                    // Movement VII (120..135s): BRIDGE // SPHERICAL S3 STAR-VAULT + RISER II
                    app.set_scene(BenchmarkScene::MultiChartCurvatureBridge);
                    app.auto_tour = true;
                    app.max_portal_bounces = 10;
                    app.max_steps = 104.0;
                    app.curvature_k = 1.05;
                    app.heatmap_mode = 0;
                }
                72 => {
                    // Movement VIII-A (135..150s): CLIMAX DROP II // 16-BOUNCE INFINITY OVERDRIVE
                    app.set_scene(BenchmarkScene::RecursiveMirrorTunnel);
                    app.auto_tour = true;
                    app.max_portal_bounces = 16;
                    app.max_steps = 120.0;
                    app.curvature_k = -1.15;
                    app.heatmap_mode = 0;
                }
                80 => {
                    // Movement VIII-B (150..157.5s): CLIMAX // HYPERBOLIC HONEYCOMB OVERDRIVE
                    app.set_scene(BenchmarkScene::HyperbolicHoneycomb);
                    app.auto_tour = true;
                    app.max_portal_bounces = 12;
                    app.max_steps = 112.0;
                    app.curvature_k = -1.35;
                    app.heatmap_mode = 2;
                }
                82 => {
                    app.heatmap_mode = 0;
                }
                84 => {
                    // Movement VIII-C (157.5..165s): CLIMAX // 16-BOUNCE CORRIDOR FINALE
                    app.set_scene(BenchmarkScene::ImpossibleCorridorLoop);
                    app.auto_tour = true;
                    app.max_portal_bounces = 16;
                    app.max_steps = 116.0;
                    app.curvature_k = -1.05;
                    app.heatmap_mode = 0;
                }
                88 => {
                    // Movement IX (165..180s): OUTRO // CONFORMAL HORIZON FADE
                    app.set_scene(BenchmarkScene::ImpossibleCorridorLoop);
                    app.auto_tour = true;
                    app.max_portal_bounces = 8;
                    app.max_steps = 100.0;
                    app.curvature_k = -0.78;
                    app.heatmap_mode = 0;
                }
                _ => {}
            }
        }

        // Audio-reactive curvature modulation & hypnotic orbital camera glide
        match app.scene {
            BenchmarkScene::ImpossibleCorridorLoop => {
                app.curvature_k = -0.82 - 0.28 * kick_env;
            }
            BenchmarkScene::HyperbolicHoneycomb => {
                app.curvature_k = -1.18 + 0.35 * (elapsed * 1.67).sin() - 0.22 * kick_env;
            }
            BenchmarkScene::MultiChartCurvatureBridge => {
                let s3_bias = if (64..72).contains(&bar_idx) { 0.85 } else { 0.0 };
                app.curvature_k = (0.90 * (elapsed * 0.85).cos() + s3_bias + 0.25 * bass_env)
                    .clamp(-1.35, 1.35);
            }
            BenchmarkScene::RecursiveMirrorTunnel => {
                app.curvature_k = -1.05 - 0.35 * kick_env;
            }
        }

        app.update(dt, elapsed, width, height);

        // Add subtle cinema camera sway & Dutch-roll banking synced to the 128 BPM groove
        let sway_x = 0.14 * (elapsed * 1.675).sin();
        let sway_y = 1.62 + 0.10 * (elapsed * 3.35).cos();
        app.camera_pos.x = sway_x;
        app.camera_pos.y = sway_y;
        app.uniform.camera_pos = [app.camera_pos.x, app.camera_pos.y, app.camera_pos.z, 1.25];

        let roll = 0.085 * (elapsed * 1.675).sin() + 0.035 * kick_env * (elapsed * 3.35).cos();
        app.uniform.camera_up = [roll.sin(), roll.cos(), 0.0, app.curvature_k];

        // Pass audio envelopes into shader telemetry (telemetry.w = 1.0 enables cinema letterbox mode)
        app.uniform.telemetry = [kick_env, bass_env, arp_env, 1.0];

        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[app.uniform]));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VideoFrameEncoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VideoFramePass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        {
            let data = slice.get_mapped_range();
            for y in 0..height {
                let src_start = (y * bytes_per_row) as usize;
                let dst_start = (y * width * 4) as usize;
                frame_rgba[dst_start..dst_start + (width * 4) as usize]
                    .copy_from_slice(&data[src_start..src_start + (width * 4) as usize]);
            }
        }
        readback_buffer.unmap();

        // Minimalist, unobtrusive cinema letterbox credits tucked strictly inside the bottom matte bar (y = 684)
        let movement_label = match bar_idx {
            0..=7 => "I. INTRO // POINCARE HORIZON [BARS 1-8]",
            8..=15 => "II. GROOVE // ACID REESE PROPULSION [BARS 9-16]",
            16..=31 => "III. VERSE // CYBERPUNK LEAD HOOK [BARS 17-32]",
            32..=39 => "IV. BREAKDOWN // ETHEREAL H3 CATHEDRAL [BARS 33-40]",
            40..=47 => "V. BUILDUP // GEODESIC SNARE RISER [BARS 41-48]",
            48..=63 => "VI. DROP I // 16-BOUNCE INFINITY TUNNEL [BARS 49-64]",
            64..=71 => "VII. BRIDGE // SPHERICAL S3 STAR-VAULT [BARS 65-72]",
            72..=87 => "VIII. CLIMAX DROP II // HYPERBOLIC OVERDRIVE [BARS 73-88]",
            _ => "IX. OUTRO // CONFORMAL HORIZON FADE [BARS 89-96]",
        };
        draw_text_rgba(
            &mut frame_rgba,
            width,
            height,
            32,
            684,
            2,
            movement_label,
            [70, 225, 255],
        );
        draw_text_rgba(
            &mut frame_rgba,
            width,
            height,
            920,
            684,
            2,
            "GEODESIC HORIZON // 128 BPM",
            [255, 90, 210],
        );

        ffmpeg_stdin.write_all(&frame_rgba).unwrap();

        if (frame_idx + 1) % 600 == 0 || frame_idx + 1 == total_frames {
            println!(
                "  [Render] Frame {}/{} ({:.1}s / {:.1}s) — {}",
                frame_idx + 1,
                total_frames,
                elapsed + dt,
                duration_secs,
                movement_label
            );
        }
    }

    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait().expect("Failed waiting on ffmpeg");
    println!(
        "[Master] Original 3:00 Master Recorded ({} frames @ {} FPS, status={}) -> {}",
        total_frames, fps, status, original_3min_path
    );

    // Step 2: Compress the 3:00 Master to strictly < 20 MB (~16.8 MB total via 620k H.264 + 160k AAC)
    println!(
        "[Compress] Encoding < 20 MB delivery version -> {}",
        compressed_3min_path
    );
    let compress_status = Command::new("/opt/homebrew/bin/ffmpeg")
        .args([
            "-y",
            "-i",
            &original_3min_path,
            "-c:v",
            "libx264",
            "-b:v",
            "620k",
            "-maxrate",
            "680k",
            "-bufsize",
            "2000k",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            &compressed_3min_path,
        ])
        .stderr(Stdio::null())
        .status()
        .expect("Failed to run ffmpeg compression pass");

    if output_path != compressed_3min_path && output_path != original_3min_path {
        std::fs::copy(&compressed_3min_path, output_path)
            .expect("Failed to copy compressed <20MB video to output_path");
    }
    println!(
        "[Compress] Finished < 20 MB 3:00 video (status={}) -> {} & {}",
        compress_status, compressed_3min_path, output_path
    );
}


// ─── Main Entrypoint (Windowed Explorer, `--bench`, or `--record`) ─────────

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if let Some(idx) = args.iter().position(|a| a == "--record") {
        let out_path = args
            .get(idx + 1)
            .map(|s| s.as_str())
            .unwrap_or("hyperbolic_portal_gameplay.mp4");
        run_video_recording(out_path);
        return;
    }
    if args.iter().any(|a| a == "--bench" || a == "--benchmark") {
        run_headless_benchmark_suite();
        return;
    }

    println!("================================================================================");
    println!("  Metatopia Non-Euclidean WGSL Shader & Hyperbolic Portal Prototype");
    println!("  Controls:");
    println!("    WASD + Space/Shift : Walk & fly through non-Euclidean portals");
    println!("    Mouse Left Drag    : Look around");
    println!("    1 - 4              : Switch Non-Euclidean Scene Preset");
    println!("    P / O              : Increase / Decrease Max Portal Recursion Bounces (0..16)");
    println!("    K / J              : Increase / Decrease Gaussian Curvature κ (-2.0..+2.0)");
    println!("    G                  : Toggle Analytical RK2 Christoffel Geodesic Bending");
    println!("    H                  : Cycle GPU Heatmap (0=PBR, 1=Step Cost, 2=Portal Depth)");
    println!("    Tab                : Toggle Automated Portal Loop Tour");
    println!("    B                  : Run CPU ManifoldRaycaster Benchmark in Console");
    println!("    ESC                : Quit");
    println!("================================================================================");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WinitWindowBuilder::new()
            .with_title("Metatopia Engine — Non-Euclidean WGSL & Hyperbolic Portal Benchmark")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("HyperbolicPortalDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .unwrap();

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hyperbolic_portal_benchmark.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let mut app = HyperbolicPortalApp::new(BenchmarkScene::ImpossibleCorridorLoop);
    app.update(0.016, 0.0, config.width, config.height);

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PortalUniformBuffer"),
        contents: bytemuck::cast_slice(&[app.uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("PortalBGL"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PortalBG"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PortalPipelineLayout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PortalRenderPipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let start_time = Instant::now();
    let mut last_frame = Instant::now();
    event_loop.set_control_flow(ControlFlow::Poll);

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WinitWindowEvent::CloseRequested => elwt.exit(),
            WinitWindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(&device, &config);
                }
            }
            WinitWindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    app.mouse_captured = state == ElementState::Pressed;
                }
            }
            WinitWindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape => elwt.exit(),
                        KeyCode::KeyW => app.keys_held[0] = pressed,
                        KeyCode::KeyA => app.keys_held[1] = pressed,
                        KeyCode::KeyS => app.keys_held[2] = pressed,
                        KeyCode::KeyD => app.keys_held[3] = pressed,
                        KeyCode::Space => app.keys_held[4] = pressed,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => app.keys_held[5] = pressed,
                        KeyCode::Digit1 if pressed => {
                            app.set_scene(BenchmarkScene::ImpossibleCorridorLoop)
                        }
                        KeyCode::Digit2 if pressed => {
                            app.set_scene(BenchmarkScene::HyperbolicHoneycomb)
                        }
                        KeyCode::Digit3 if pressed => {
                            app.set_scene(BenchmarkScene::MultiChartCurvatureBridge)
                        }
                        KeyCode::Digit4 if pressed => {
                            app.set_scene(BenchmarkScene::RecursiveMirrorTunnel)
                        }
                        KeyCode::KeyP | KeyCode::ArrowUp if pressed => {
                            app.max_portal_bounces = (app.max_portal_bounces + 1).min(16);
                            println!(
                                "[PortalBench] Max Portal Bounces = {}",
                                app.max_portal_bounces
                            );
                        }
                        KeyCode::KeyO | KeyCode::ArrowDown if pressed => {
                            app.max_portal_bounces = app.max_portal_bounces.saturating_sub(1);
                            println!(
                                "[PortalBench] Max Portal Bounces = {}",
                                app.max_portal_bounces
                            );
                        }
                        KeyCode::KeyK | KeyCode::ArrowRight if pressed => {
                            app.curvature_k = (app.curvature_k + 0.15).min(2.0);
                            println!("[PortalBench] Gaussian Curvature κ = {:.2}", app.curvature_k);
                        }
                        KeyCode::KeyJ | KeyCode::ArrowLeft if pressed => {
                            app.curvature_k = (app.curvature_k - 0.15).max(-2.0);
                            println!("[PortalBench] Gaussian Curvature κ = {:.2}", app.curvature_k);
                        }
                        KeyCode::KeyG if pressed => {
                            app.geodesic_enabled = !app.geodesic_enabled;
                            println!(
                                "[PortalBench] Analytical RK2 Geodesic Ray Bending = {}",
                                app.geodesic_enabled
                            );
                        }
                        KeyCode::KeyH if pressed => {
                            app.heatmap_mode = (app.heatmap_mode + 1) % 3;
                            let mode_name = match app.heatmap_mode {
                                0 => "Cook-Torrance PBR Shaded",
                                1 => "GPU Ray-March Step Cost Heatmap",
                                _ => "Recursive Portal Bounce Depth Heatmap",
                            };
                            println!("[PortalBench] Display Mode = {}", mode_name);
                        }
                        KeyCode::Tab if pressed => {
                            app.auto_tour = !app.auto_tour;
                            println!("[PortalBench] Automated Portal Loop Tour = {}", app.auto_tour);
                        }
                        KeyCode::KeyB if pressed => {
                            let m = ManifoldRaycaster::benchmark_portal_transitions(
                                &app.manifold,
                                app.active_chart,
                                2048,
                                app.max_portal_bounces as usize,
                                0.25,
                            );
                            app.last_cpu_krays_sec = (m.rays_per_sec / 1000.0) as f32;
                            println!(
                                "[PortalBench] CPU Raycast: {:.1} KRays/s | {:.2} MSegments/s | {} portal jumps | Render FPS: {:.1}",
                                m.rays_per_sec / 1000.0,
                                m.segments_per_sec / 1_000_000.0,
                                m.total_portal_jumps,
                                app.fps_smoothed
                            );
                        }
                        _ => {}
                    }
                }
            }
            WinitWindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - last_frame).as_secs_f32().min(0.1);
                last_frame = now;
                let elapsed = start_time.elapsed().as_secs_f32();

                app.update(dt, elapsed, config.width, config.height);
                queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[app.uniform]));

                if let Ok(output) = surface.get_current_texture() {
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("PortalFrameEncoder"),
                        });
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("PortalFramePass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        rpass.set_pipeline(&pipeline);
                        rpass.set_bind_group(0, &bind_group, &[]);
                        rpass.draw(0..6, 0..1);
                    }
                    queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            if app.mouse_captured {
                app.yaw += delta.0 as f32 * 0.004;
                app.pitch = (app.pitch - delta.1 as f32 * 0.004).clamp(-1.45, 1.45);
            }
        }
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => {}
    });
}
