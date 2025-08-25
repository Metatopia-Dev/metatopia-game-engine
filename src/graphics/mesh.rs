//! Mesh and vertex data structures

use wgpu::{Buffer, Device, BufferUsages, util::DeviceExt};
use bytemuck::{Pod, Zeroable};

/// Vertex data structure
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn new(position: [f32; 3], tex_coords: [f32; 2], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            tex_coords,
            normal,
            color,
        }
    }
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinates
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Mesh structure containing vertex and index data
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub num_indices: u32,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    /// Create a new mesh from vertices and indices
    pub fn new(device: &Device, vertices: Vec<Vertex>, indices: Vec<u16>) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });
        
        let num_indices = indices.len() as u32;
        
        Self {
            vertex_buffer,
            index_buffer,
            num_indices,
            vertices,
            indices,
        }
    }
    
    /// Create a quad mesh
    pub fn create_quad(device: &Device, size: f32) -> Self {
        let half_size = size / 2.0;
        
        let vertices = vec![
            Vertex::new(
                [-half_size, -half_size, 0.0],
                [0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [half_size, -half_size, 0.0],
                [1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [half_size, half_size, 0.0],
                [1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [-half_size, half_size, 0.0],
                [0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ),
        ];
        
        let indices = vec![0, 1, 2, 0, 2, 3];
        
        Self::new(device, vertices, indices)
    }
    
    /// Create a cube mesh
    pub fn create_cube(device: &Device, size: f32) -> Self {
        let half_size = size / 2.0;
        
        let vertices = vec![
            // Front face
            Vertex::new([-half_size, -half_size, half_size], [0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
            Vertex::new([half_size, -half_size, half_size], [1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
            Vertex::new([half_size, half_size, half_size], [1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
            Vertex::new([-half_size, half_size, half_size], [0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]),
            
            // Back face
            Vertex::new([half_size, -half_size, -half_size], [0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0, 1.0]),
            Vertex::new([-half_size, -half_size, -half_size], [1.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 1.0, 1.0]),
            Vertex::new([-half_size, half_size, -half_size], [1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0, 1.0]),
            Vertex::new([half_size, half_size, -half_size], [0.0, 0.0], [0.0, 0.0, -1.0], [0.5, 0.5, 0.5, 1.0]),
        ];
        
        let indices = vec![
            // Front face
            0, 1, 2, 0, 2, 3,
            // Back face
            4, 5, 6, 4, 6, 7,
            // Top face
            3, 2, 7, 3, 7, 6,
            // Bottom face
            5, 4, 1, 5, 1, 0,
            // Right face
            1, 4, 7, 1, 7, 2,
            // Left face
            5, 0, 3, 5, 3, 6,
        ];
        
        Self::new(device, vertices, indices)
    }
}