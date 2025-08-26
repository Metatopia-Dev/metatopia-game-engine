//! Basic demo showcasing non-Euclidean spaces with GPU graphics
//! 
//! This example creates a world with:
//! - A Euclidean room
//! - A hyperbolic space (Poincaré disk)
//! - A spherical space
//! - Portals connecting them seamlessly

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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_position: [f32; 4],
}

struct NonEuclideanDemo {
    manifold: Arc<RwLock<Manifold>>,
    camera_position: Point3<f32>,
    camera_rotation: (f32, f32), // yaw, pitch
    current_chart: ChartId,
    movement_speed: f32,
    camera_uniform: CameraUniform,
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    mouse_sensitivity: f32,
}

impl NonEuclideanDemo {
    fn new() -> Self {
        // Create the manifold with different geometries
        let mut manifold = Manifold::new();
        
        // Add a hyperbolic chart (Poincaré disk)
        let hyperbolic_chart = manifold.add_chart(GeometryType::Hyperbolic);
        
        // Add a spherical chart
        let spherical_chart = manifold.add_chart(GeometryType::Spherical);
        
        // Create portals between spaces
        manifold.create_portal(
            ChartId(0), // Euclidean
            hyperbolic_chart,
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        manifold.create_portal(
            hyperbolic_chart,
            spherical_chart,
            Point3::new(0.5, 0.5, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        manifold.create_portal(
            spherical_chart,
            ChartId(0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-5.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        Self {
            manifold: Arc::new(RwLock::new(manifold)),
            camera_position: Point3::new(0.0, 1.0, -5.0),
            camera_rotation: (0.0, 0.0),
            current_chart: ChartId(0),
            movement_speed: 0.1,
            camera_uniform: CameraUniform {
                view_proj: [[0.0; 4]; 4],
                view_position: [0.0, 1.0, -5.0, 1.0],
            },
            camera_buffer: None,
            camera_bind_group: None,
            mouse_sensitivity: 0.002,
        }
    }
    
    fn update(&mut self, _dt: f32) {
        // Simple physics update
        // In a real game, this would handle more complex movement
        
        // Check for portal transitions
        self.check_portal_transitions();
    }
    
    fn check_portal_transitions(&mut self) {
        let forward = self.get_forward_vector();
        
        if let Ok(manifold) = self.manifold.read() {
            if let Some((_portal_id, intersection, new_chart)) = 
                manifold.ray_portal_intersection(self.camera_position, forward, self.current_chart) {
                
                println!("Transitioning through portal to chart {:?}", new_chart);
                
                // Update position to new chart
                self.camera_position = intersection;
                self.current_chart = new_chart;
                
                // Update manifold active chart
                drop(manifold); // Release read lock
                if let Ok(mut manifold) = self.manifold.write() {
                    manifold.set_active_chart(new_chart);
                }
            }
        }
    }
    
    fn get_forward_vector(&self) -> Vector3<f32> {
        let (yaw, pitch) = self.camera_rotation;
        Vector3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
    }
    
    fn update_camera_uniform(&mut self, aspect_ratio: f32) {
        // Create view matrix
        let (yaw, pitch) = self.camera_rotation;
        let forward = self.get_forward_vector();
        let target = self.camera_position + forward;
        let view = Matrix4::<f32>::look_at_rh(
            self.camera_position,
            Point3::new(target.x, target.y, target.z),
            Vector3::unit_y(),
        );
        
        // Create projection matrix
        let proj = perspective(Deg(45.0), aspect_ratio, 0.1, 100.0);
        
        // Combine into view-projection matrix
        let view_proj = proj * view;
        
        self.camera_uniform.view_proj = view_proj.into();
        
        // Encode space type in w component (0=Euclidean, 1=Hyperbolic, 2=Spherical)
        let space_type = if let Ok(manifold) = self.manifold.read() {
            match manifold.chart(self.current_chart).unwrap().geometry() {
                GeometryType::Euclidean => 0.0,
                GeometryType::Hyperbolic => 1.0,
                GeometryType::Spherical => 2.0,
                GeometryType::Custom => 0.0,
            }
        } else {
            0.0
        };
        
        self.camera_uniform.view_position = [
            self.camera_position.x,
            self.camera_position.y,
            self.camera_position.z,
            space_type,
        ];
    }
    
    fn handle_keyboard(&mut self, key: KeyCode, pressed: bool) {
        if !pressed {
            return;
        }
        
        let forward = self.get_forward_vector();
        let right = Vector3::new(-forward.z, 0.0, forward.x).normalize();
        
        match key {
            KeyCode::KeyW => self.camera_position += forward * self.movement_speed,
            KeyCode::KeyS => self.camera_position -= forward * self.movement_speed,
            KeyCode::KeyA => self.camera_position -= right * self.movement_speed,
            KeyCode::KeyD => self.camera_position += right * self.movement_speed,
            KeyCode::Space => self.camera_position.y += self.movement_speed,
            KeyCode::ShiftLeft => self.camera_position.y -= self.movement_speed,
            KeyCode::ArrowLeft => self.camera_rotation.0 -= 0.05,
            KeyCode::ArrowRight => self.camera_rotation.0 += 0.05,
            KeyCode::ArrowUp => self.camera_rotation.1 = (self.camera_rotation.1 - 0.05).max(-1.5).min(1.5),
            KeyCode::ArrowDown => self.camera_rotation.1 = (self.camera_rotation.1 + 0.05).max(-1.5).min(1.5),
            KeyCode::KeyR => {
                // Reset position to start
                self.camera_position = Point3::new(0.0, 1.0, -5.0);
                self.camera_rotation = (0.0, 0.0);
                self.current_chart = ChartId(0);
                println!("Reset to starting position");
            }
            _ => {}
        }
    }
    
    fn handle_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        self.camera_rotation.0 += delta_x as f32 * self.mouse_sensitivity;
        self.camera_rotation.1 -= delta_y as f32 * self.mouse_sensitivity;
        
        // Clamp pitch to prevent flipping
        self.camera_rotation.1 = self.camera_rotation.1.clamp(-1.5, 1.5);
    }
}

async fn run() {
    env_logger::init();
    
    println!("🌐 Non-Euclidean Game Engine Demo");
    println!("==================================");
    println!("Controls:");
    println!("  WASD - Move horizontally");
    println!("  Space/Shift - Move up/down");
    println!("  Mouse - Look around");
    println!("  Arrow Keys - Manual camera rotation");
    println!("  R - Reset to start position");
    println!("  ESC - Exit");
    println!("\nSpaces:");
    println!("  Blue room - Euclidean space (starting area)");
    println!("  Purple room - Hyperbolic space");
    println!("  Orange room - Spherical space");
    println!("\nPortals are on the walls - walk through to transition!");
    println!();
    
    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WinitWindowBuilder::new()
        .with_title("Metatopia - Non-Euclidean Spaces Demo")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();
    
    let window = Arc::new(window);
    
    // Set cursor grab for mouse look
    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
        .ok();
    window.set_cursor_visible(false);
    
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
    
    // Create demo
    let mut demo = NonEuclideanDemo::new();
    
    // Create shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Non-Euclidean Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/non_euclidean.wgsl").into()),
    });
    
    // Create camera buffer and bind group
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[demo.camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    
    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        label: Some("camera_bind_group_layout"),
    });
    
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
        label: Some("camera_bind_group"),
    });
    
    demo.camera_buffer = Some(camera_buffer);
    demo.camera_bind_group = Some(camera_bind_group);
    
    // Create render pipeline layout
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&camera_bind_group_layout],
        push_constant_ranges: &[],
    });
    
    // Create render pipeline
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
    let mut last_time = start_time;
    
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
                WinitWindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        if key_code == KeyCode::Escape {
                            target.exit();
                        } else {
                            demo.handle_keyboard(key_code, event.state == ElementState::Pressed);
                        }
                    }
                }
                WinitWindowEvent::RedrawRequested => {
                    frame_count += 1;
                    let current_time = std::time::Instant::now();
                    let dt = (current_time - last_time).as_secs_f32();
                    last_time = current_time;
                    
                    // Update demo
                    demo.update(dt);
                    
                    // Update camera matrices
                    let aspect_ratio = config.width as f32 / config.height as f32;
                    demo.update_camera_uniform(aspect_ratio);
                    
                    // Write camera uniform to buffer
                    if let Some(ref camera_buffer) = demo.camera_buffer {
                        queue.write_buffer(camera_buffer, 0, bytemuck::cast_slice(&[demo.camera_uniform]));
                    }
                    
                    // Get next frame
                    let output = surface.get_current_texture().unwrap();
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Render Encoder"),
                    });
                    
                    {
                        // Determine background color based on current space
                        let color = if let Ok(manifold) = demo.manifold.read() {
                            match manifold.chart(demo.current_chart).unwrap().geometry() {
                                GeometryType::Euclidean => wgpu::Color {
                                    r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                                },
                                GeometryType::Hyperbolic => wgpu::Color {
                                    r: 0.1, g: 0.05, b: 0.15, a: 1.0,
                                },
                                GeometryType::Spherical => wgpu::Color {
                                    r: 0.15, g: 0.1, b: 0.05, a: 1.0,
                                },
                                GeometryType::Custom => wgpu::Color {
                                    r: 0.1, g: 0.1, b: 0.1, a: 1.0,
                                },
                            }
                        } else {
                            wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }
                        };
                        
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(color),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        
                        render_pass.set_pipeline(&render_pipeline);
                        if let Some(ref bind_group) = demo.camera_bind_group {
                            render_pass.set_bind_group(0, bind_group, &[]);
                        }
                        // Draw multiple quads to form a room
                        render_pass.draw(0..36, 0..1); // Draw a cube (6 faces * 6 vertices)
                    }
                    
                    queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                    
                    // Print status every 60 frames
                    if frame_count % 60 == 0 {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let fps = frame_count as f32 / elapsed;
                        
                        if let Ok(manifold) = demo.manifold.read() {
                            let geometry = manifold.chart(demo.current_chart).unwrap().geometry();
                            println!("FPS: {:.1} | Position: ({:.2}, {:.2}, {:.2}) | Space: {:?}", 
                                fps, demo.camera_position.x, demo.camera_position.y, demo.camera_position.z, geometry);
                        }
                    }
                    
                    window.request_redraw();
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                demo.handle_mouse_motion(delta.0, delta.1);
            }
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