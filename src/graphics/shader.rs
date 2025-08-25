//! Shader management for non-Euclidean rendering

use wgpu::{
    Device, ShaderModule, PipelineLayout, RenderPipeline,
    VertexBufferLayout, ShaderModuleDescriptor, ShaderSource,
};
use std::collections::HashMap;

/// Shader program for metric-aware rendering
pub struct ShaderProgram {
    pub vertex_module: ShaderModule,
    pub fragment_module: ShaderModule,
    pub pipeline: Option<RenderPipeline>,
    pub geometry_type: GeometryType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryType {
    Euclidean,
    Spherical,
    Hyperbolic,
    Custom,
}

impl ShaderProgram {
    /// Create a new shader program for non-Euclidean rendering
    pub fn from_wgsl(
        device: &Device, 
        vertex_src: &str, 
        fragment_src: &str,
        geometry_type: GeometryType,
    ) -> Self {
        let vertex_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Non-Euclidean Vertex Shader"),
            source: ShaderSource::Wgsl(vertex_src.into()),
        });
        
        let fragment_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Non-Euclidean Fragment Shader"),
            source: ShaderSource::Wgsl(fragment_src.into()),
        });
        
        Self {
            vertex_module,
            fragment_module,
            pipeline: None,
            geometry_type,
        }
    }
    
    /// Create a render pipeline with metric-aware transformations
    pub fn create_pipeline(
        &mut self,
        device: &Device,
        layout: &PipelineLayout,
        vertex_layout: VertexBufferLayout,
        format: wgpu::TextureFormat,
    ) {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Non-Euclidean Render Pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &self.vertex_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.fragment_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for non-Euclidean spaces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        self.pipeline = Some(pipeline);
    }
}

/// Shader manager for non-Euclidean spaces
pub struct Shader {
    programs: HashMap<String, ShaderProgram>,
    device: Device,
}

impl Shader {
    pub fn new(device: Device) -> Self {
        Self {
            programs: HashMap::new(),
            device,
        }
    }
    
    pub fn load_program(
        &mut self, 
        name: &str, 
        vertex_src: &str, 
        fragment_src: &str,
        geometry_type: GeometryType,
    ) {
        let program = ShaderProgram::from_wgsl(&self.device, vertex_src, fragment_src, geometry_type);
        self.programs.insert(name.to_string(), program);
    }
    
    pub fn get_program(&self, name: &str) -> Option<&ShaderProgram> {
        self.programs.get(name)
    }
    
    pub fn get_program_mut(&mut self, name: &str) -> Option<&mut ShaderProgram> {
        self.programs.get_mut(name)
    }
    
    /// Create shaders for different geometries
    pub fn create_geometry_shaders(&mut self) {
        // Euclidean shader (standard)
        let euclidean_vertex = r#"
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
            }

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) world_pos: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
                @location(4) chart_id: f32,
            }

            struct Uniforms {
                view_proj: mat4x4<f32>,
                model: mat4x4<f32>,
                chart_id: f32,
                metric_params: vec4<f32>,
            }

            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;

            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                let world_pos = (uniforms.model * vec4<f32>(input.position, 1.0)).xyz;
                out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
                out.world_pos = world_pos;
                out.tex_coords = input.tex_coords;
                out.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
                out.color = input.color;
                out.chart_id = uniforms.chart_id;
                return out;
            }
        "#;
        
        // Hyperbolic shader (Poincaré disk model)
        let hyperbolic_vertex = r#"
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
            }

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) hyperbolic_pos: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
                @location(4) chart_id: f32,
            }

            struct Uniforms {
                view_proj: mat4x4<f32>,
                model: mat4x4<f32>,
                chart_id: f32,
                metric_params: vec4<f32>, // x: curvature, y: scale
            }

            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;

            // Hyperbolic transformation (Poincaré disk)
            fn hyperbolic_transform(p: vec2<f32>) -> vec2<f32> {
                let r = length(p);
                if (r >= 0.99) {
                    return p * 0.99 / r;
                }
                return p;
            }

            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                
                // Apply hyperbolic transformation in 2D
                let hyperbolic_xy = hyperbolic_transform(input.position.xy * uniforms.metric_params.y);
                let hyperbolic_pos = vec3<f32>(hyperbolic_xy, input.position.z);
                
                out.hyperbolic_pos = hyperbolic_pos;
                out.clip_position = uniforms.view_proj * uniforms.model * vec4<f32>(hyperbolic_pos, 1.0);
                out.tex_coords = input.tex_coords;
                out.normal = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
                out.color = input.color;
                out.chart_id = uniforms.chart_id;
                return out;
            }
        "#;
        
        // Spherical shader
        let spherical_vertex = r#"
            struct VertexInput {
                @location(0) position: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
            }

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) spherical_pos: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
                @location(4) chart_id: f32,
            }

            struct Uniforms {
                view_proj: mat4x4<f32>,
                model: mat4x4<f32>,
                chart_id: f32,
                metric_params: vec4<f32>, // x: radius, y: scale
            }

            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;

            // Spherical projection
            fn spherical_transform(p: vec3<f32>) -> vec3<f32> {
                let radius = uniforms.metric_params.x;
                let normalized = normalize(p);
                return normalized * radius;
            }

            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                
                let spherical_pos = spherical_transform(input.position * uniforms.metric_params.y);
                
                out.spherical_pos = spherical_pos;
                out.clip_position = uniforms.view_proj * uniforms.model * vec4<f32>(spherical_pos, 1.0);
                out.tex_coords = input.tex_coords;
                out.normal = normalize(spherical_pos);
                out.color = input.color;
                out.chart_id = uniforms.chart_id;
                return out;
            }
        "#;
        
        // Common fragment shader with portal support
        let fragment_shader = r#"
            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) world_pos: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) normal: vec3<f32>,
                @location(3) color: vec4<f32>,
                @location(4) chart_id: f32,
            }

            struct PortalData {
                active: f32,
                target_chart: f32,
                transform: mat4x4<f32>,
            }

            @group(0) @binding(1)
            var<uniform> portal: PortalData;

            @group(0) @binding(2)
            var t_diffuse: texture_2d<f32>;
            @group(0) @binding(3)
            var s_diffuse: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                var base_color = in.color;
                
                // Simple lighting
                let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
                let diffuse = max(dot(in.normal, light_dir), 0.2);
                
                // Portal edge visualization
                if (portal.active > 0.5) {
                    let portal_glow = 0.1 * sin(in.world_pos.x * 10.0) * sin(in.world_pos.y * 10.0);
                    base_color = base_color + vec4<f32>(portal_glow, portal_glow, portal_glow * 0.5, 0.0);
                }
                
                return vec4<f32>(base_color.rgb * diffuse, base_color.a);
            }
        "#;
        
        self.load_program("euclidean", euclidean_vertex, fragment_shader, GeometryType::Euclidean);
        self.load_program("hyperbolic", hyperbolic_vertex, fragment_shader, GeometryType::Hyperbolic);
        self.load_program("spherical", spherical_vertex, fragment_shader, GeometryType::Spherical);
    }
}