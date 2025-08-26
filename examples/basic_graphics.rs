//! Basic graphics example that actually renders to screen
//! This demonstrates the non-Euclidean engine with GPU rendering

use metatopia_engine::prelude::*;
use winit::{
    event::{Event, WindowEvent as WinitWindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder as WinitWindowBuilder,
};
use std::sync::Arc;

async fn run() {
    env_logger::init();
    
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("Metatopia Non-Euclidean Engine - Graphics Demo")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();
    
    let window = Arc::new(window);
    
    // Create WGPU instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    // Create surface
    let surface = instance.create_surface(window.clone()).unwrap();
    
    // Get adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    
    // Create device and queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Metatopia Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();
    
    // Configure surface
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.present_mode = wgpu::PresentMode::Fifo; // VSync
    surface.configure(&device, &config);
    
    // Create manifold for non-Euclidean world
    let mut manifold = Manifold::new();
    let hyperbolic_chart = manifold.add_chart(GeometryType::Hyperbolic);
    let spherical_chart = manifold.add_chart(GeometryType::Spherical);
    
    // Create portals
    manifold.create_portal(
        ChartId(0),
        hyperbolic_chart,
        cgmath::Point3::new(5.0, 0.0, 0.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        Mat4::from_scale(1.0),
    ).unwrap();
    
    manifold.create_portal(
        hyperbolic_chart,
        spherical_chart,
        cgmath::Point3::new(0.5, 0.5, 0.0),
        cgmath::Point3::new(0.0, 0.0, 1.0),
        Mat4::from_scale(1.0),
    ).unwrap();
    
    // Simple render pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Basic Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/basic.wgsl").into()),
    });
    
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });
    
    let mut frame_count = 0u32;
    let start_time = std::time::Instant::now();
    
    // Run event loop
    let _ = event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WinitWindowEvent::CloseRequested => {
                    target.exit();
                }
                WinitWindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&device, &config);
                        window.request_redraw();
                    }
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let elapsed = start_time.elapsed().as_secs_f32();
                    
                    // Get next frame
                    let output = surface.get_current_texture().unwrap();
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Render Encoder"),
                    });
                    
                    {
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    // Animate background color based on time
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: (elapsed.sin() * 0.5 + 0.5) as f64 * 0.1,
                                        g: (elapsed.cos() * 0.5 + 0.5) as f64 * 0.1,
                                        b: 0.2,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        
                        render_pass.set_pipeline(&render_pipeline);
                        render_pass.draw(0..3, 0..1);
                    }
                    
                    queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                    
                    // Print FPS every 60 frames
                    if frame_count % 60 == 0 {
                        let fps = frame_count as f32 / elapsed;
                        println!("FPS: {:.1} | Frame: {} | Time: {:.1}s", fps, frame_count, elapsed);
                        
                        // Show manifold info
                        println!("  Active Chart: {:?}", manifold.active_chart().geometry());
                        println!("  Total Charts: {}", manifold.charts().len());
                        println!("  Total Portals: {}", manifold.portals_from_chart(ChartId(0)).len());
                    }
                    
                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn main() {
    pollster::block_on(run());
}