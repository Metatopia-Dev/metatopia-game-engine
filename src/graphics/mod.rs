//! Graphics rendering module using wgpu

use wgpu::{
    Surface, Device, Queue, SurfaceConfiguration,
    TextureUsages, PresentMode, CompositeAlphaMode,
    CommandEncoder, TextureView, RenderPass,
};
use cgmath::{Matrix4, Vector3, Vector4, Rad, perspective};
use std::sync::Arc;

pub mod mesh;
pub mod shader;
pub mod texture;
pub mod camera;

pub use mesh::{Mesh, Vertex};
pub use shader::{Shader, ShaderProgram};
pub use texture::Texture;
pub use camera::Camera;

/// Render context passed to rendering functions
pub struct RenderContext<'a> {
    pub encoder: &'a mut CommandEncoder,
    pub view: &'a TextureView,
    pub device: &'a Device,
    pub queue: &'a Queue,
}

/// Main renderer struct
pub struct Renderer {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: (u32, u32),
    current_frame: Option<CurrentFrame>,
    shader: Shader,
}

struct CurrentFrame {
    output: wgpu::SurfaceTexture,
    view: TextureView,
    encoder: CommandEncoder,
}

impl Renderer {
    /// Create a new renderer for the given window
    pub async fn new(window: &crate::window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.dimensions();
        
        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // Create surface
        let surface = instance.create_surface(window.window_arc())?;
        
        // Request adapter
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("Failed to find suitable adapter")?;
        
        // Create device and queue
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Metatopia Renderer Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ).await?;
        
        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.0,
            height: size.1,
            present_mode: PresentMode::Fifo, // VSync
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);
        
        let shader = Shader::new(device.clone());
        
        Ok(Self {
            surface,
            device: device.clone(),
            queue,
            config,
            size,
            current_frame: None,
            shader,
        })
    }
    
    /// Begin a new frame
    pub fn begin_frame(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.size.0, self.size.1);
                return self.begin_frame();
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                panic!("Out of memory!");
            }
            Err(e) => {
                eprintln!("Surface error: {:?}", e);
                return;
            }
        };
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        self.current_frame = Some(CurrentFrame {
            output,
            view,
            encoder,
        });
    }
    
    /// End the current frame and present it
    pub fn end_frame(&mut self) {
        if let Some(frame) = self.current_frame.take() {
            self.queue.submit(std::iter::once(frame.encoder.finish()));
            frame.output.present();
        }
    }
    
    /// Get a render pass for the current frame
    pub fn begin_render_pass(&mut self) -> Option<RenderPass> {
        self.current_frame.as_mut().map(|frame| {
            frame.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
        })
    }
    
    /// Clear the screen with a color
    pub fn clear(&mut self, r: f32, g: f32, b: f32, a: f32) {
        if let Some(frame) = &mut self.current_frame {
            frame.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
    }
    
    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get the queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    
    /// Get current size
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    
    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }
    
    /// Get mutable reference to shader
    pub fn shader_mut(&mut self) -> &mut Shader {
        &mut self.shader
    }
}

/// Color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }
    
    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}