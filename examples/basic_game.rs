//! Basic demo showcasing non-Euclidean spaces
//! 
//! This example creates a world with:
//! - A Euclidean room
//! - A hyperbolic space (Poincaré disk)
//! - A spherical space
//! - Portals connecting them seamlessly

use metatopia_engine::prelude::*;
use std::sync::{Arc, RwLock};

/// Demo game state
struct NonEuclideanDemo {
    manifold: Arc<RwLock<Manifold>>,
    camera: Camera,
    camera_controller: FPSCameraController,
    world: World,
    player_entity: Entity,
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
        // Portal from Euclidean to Hyperbolic
        manifold.create_portal(
            ChartId(0), // Euclidean
            hyperbolic_chart,
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        // Portal from Hyperbolic to Spherical
        manifold.create_portal(
            hyperbolic_chart,
            spherical_chart,
            Point3::new(0.5, 0.5, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        // Portal from Spherical back to Euclidean
        manifold.create_portal(
            spherical_chart,
            ChartId(0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-5.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        let manifold = Arc::new(RwLock::new(manifold));
        
        // Create camera in Euclidean space
        let camera = Camera::new(
            ChartId(0),
            Point3::new(0.0, 1.0, -5.0),
            Point3::new(0.0, 0.0, 0.0),
            1280.0 / 720.0,
        );
        
        let camera_controller = FPSCameraController::new();
        
        // Create ECS world
        let mut world = World::new();
        
        // Add systems
        world.add_system(Box::new(TransformSystem));
        world.add_system(Box::new(PortalTransitionSystem::new(manifold.clone())));
        
        // Create player entity
        let player_entity = world.create_entity();
        world.add_component(
            player_entity,
            EcsTransform::new(ChartId(0), Point3::new(0.0, 1.0, -5.0)),
        );
        world.add_component(
            player_entity,
            Velocity {
                linear: Vector3::new(0.0, 0.0, 0.0),
                angular: Vector3::new(0.0, 0.0, 0.0),
            },
        );
        
        Self {
            manifold,
            camera,
            camera_controller,
            world,
            player_entity,
        }
    }
    
    fn create_world_geometry(&mut self, engine: &mut Engine) {
        // Create floor meshes for each space
        let device = engine.renderer.device();
        
        // Store meshes directly in the demo struct
        // (Resource manager requires Clone which wgpu::Buffer doesn't implement)
        
        // Euclidean room floor
        let _euclidean_floor = Mesh::create_quad(device, 20.0);
        
        // Hyperbolic space floor (Poincaré disk)
        let _hyperbolic_floor = create_poincare_disk_mesh(device, 0.99, 32);
        
        // Spherical space surface
        let _spherical_surface = create_sphere_mesh(device, 10.0, 32, 16);
        
        // Portal frames
        let _portal_frame = create_portal_frame_mesh(device);
    }
}

impl GameState for NonEuclideanDemo {
    fn on_init(&mut self, engine: &mut Engine) {
        println!("Non-Euclidean Demo Starting!");
        println!("Controls:");
        println!("  WASD - Move");
        println!("  Mouse - Look around");
        println!("  Space - Move up");
        println!("  Shift - Move down");
        println!("  Walk through portals to transition between spaces!");
        
        // Create world geometry
        self.create_world_geometry(engine);
        
        // Initialize shaders for different geometries
        engine.renderer.shader_mut().create_geometry_shaders();
    }
    
    fn on_update(&mut self, engine: &mut Engine, dt: f32) {
        // Update camera based on input
        self.camera_controller.update(&mut self.camera, &engine.input, dt);
        
        // Update camera matrices based on current manifold chart
        if let Ok(manifold) = self.manifold.read() {
            self.camera.update(&manifold);
        }
        
        // Update player entity position to match camera
        if let Some(transform) = self.world.get_component_mut::<EcsTransform>(self.player_entity) {
            transform.position = self.camera.position;
        }
        
        // Update ECS systems
        self.world.update(dt);
        
        // Check for portal transitions
        self.check_portal_transitions(engine);
        
        // Handle escape key
        if engine.input.is_key_pressed(KeyCode::Escape) {
            engine.quit();
        }
    }
    
    fn on_render(&mut self, renderer: &mut Renderer) {
        // Clear screen
        renderer.clear(0.05, 0.05, 0.1, 1.0);
        
        // Render based on current chart geometry
        if let Ok(manifold) = self.manifold.read() {
            let chart = manifold.active_chart();
            
            match chart.geometry() {
                GeometryType::Euclidean => {
                    self.render_euclidean_space(renderer);
                }
                GeometryType::Hyperbolic => {
                    self.render_hyperbolic_space(renderer);
                }
                GeometryType::Spherical => {
                    self.render_spherical_space(renderer);
                }
                GeometryType::Custom => {}
            }
            
            // Render portals
            self.render_portals(renderer, &manifold);
        }
    }
    
    fn on_cleanup(&mut self, _engine: &mut Engine) {
        println!("Non-Euclidean Demo Ending!");
    }
}

impl NonEuclideanDemo {
    fn check_portal_transitions(&mut self, _engine: &mut Engine) {
        let position = self.camera.position.local.to_point();
        let forward = self.camera.forward();
        
        if let Ok(manifold) = self.manifold.read() {
            if let Some((_portal_id, intersection, new_chart)) = 
                manifold.ray_portal_intersection(position, forward, self.camera.position.chart_id) {
                
                println!("Transitioning through portal to chart {:?}", new_chart);
                
                // Update camera position to new chart
                self.camera.set_position(new_chart, intersection);
                
                // Update manifold active chart
                drop(manifold); // Release read lock
                if let Ok(mut manifold) = self.manifold.write() {
                    manifold.set_active_chart(new_chart);
                }
            }
        }
    }
    
    fn render_euclidean_space(&self, _renderer: &mut Renderer) {
        // Render Euclidean room with grid pattern
        // This would use the euclidean shader program
    }
    
    fn render_hyperbolic_space(&self, _renderer: &mut Renderer) {
        // Render Poincaré disk with hyperbolic tiling
        // This would use the hyperbolic shader program
    }
    
    fn render_spherical_space(&self, _renderer: &mut Renderer) {
        // Render spherical space
        // This would use the spherical shader program
    }
    
    fn render_portals(&self, _renderer: &mut Renderer, manifold: &Manifold) {
        // Render portal edges and effects
        for _portal in manifold.portals_from_chart(self.camera.position.chart_id) {
            // Render portal frame with ripple effect
        }
    }
}

// Helper functions to create specialized meshes

fn create_poincare_disk_mesh(device: &wgpu::Device, radius: f32, segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Center vertex
    vertices.push(Vertex::new(
        [0.0, 0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 1.0, 1.0],
    ));
    
    // Create disk vertices
    for i in 0..segments {
        let angle = (i as f32) * 2.0 * std::f32::consts::PI / segments as f32;
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        
        vertices.push(Vertex::new(
            [x, y, 0.0],
            [(x + 1.0) / 2.0, (y + 1.0) / 2.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.3, 0.8, 1.0],
        ));
    }
    
    // Create triangles
    for i in 0..segments {
        indices.push(0);
        indices.push((i + 1) as u16);
        indices.push(((i + 1) % segments + 1) as u16);
    }
    
    Mesh::new(device, vertices, indices)
}

fn create_sphere_mesh(device: &wgpu::Device, radius: f32, slices: u32, stacks: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Generate sphere vertices
    for stack in 0..=stacks {
        let phi = std::f32::consts::PI * (stack as f32) / (stacks as f32);
        
        for slice in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * (slice as f32) / (slices as f32);
            
            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();
            
            let normal = [x / radius, y / radius, z / radius];
            
            vertices.push(Vertex::new(
                [x, y, z],
                [slice as f32 / slices as f32, stack as f32 / stacks as f32],
                normal,
                [0.7, 0.7, 0.3, 1.0],
            ));
        }
    }
    
    // Generate indices
    for stack in 0..stacks {
        for slice in 0..slices {
            let first = stack * (slices + 1) + slice;
            let second = first + slices + 1;
            
            indices.push(first as u16);
            indices.push(second as u16);
            indices.push((first + 1) as u16);
            
            indices.push(second as u16);
            indices.push((second + 1) as u16);
            indices.push((first + 1) as u16);
        }
    }
    
    Mesh::new(device, vertices, indices)
}

fn create_portal_frame_mesh(device: &wgpu::Device) -> Mesh {
    // Create a rectangular frame for portal visualization
    let vertices = vec![
        // Outer rectangle
        Vertex::new([-1.2, -1.8, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.8]),
        Vertex::new([1.2, -1.8, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.8]),
        Vertex::new([1.2, 1.8, 0.0], [1.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.8]),
        Vertex::new([-1.2, 1.8, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.8]),
        // Inner rectangle (cutout)
        Vertex::new([-1.0, -1.5, 0.0], [0.1, 0.1], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.3]),
        Vertex::new([1.0, -1.5, 0.0], [0.9, 0.1], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.3]),
        Vertex::new([1.0, 1.5, 0.0], [0.9, 0.9], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.3]),
        Vertex::new([-1.0, 1.5, 0.0], [0.1, 0.9], [0.0, 0.0, 1.0], [0.5, 0.8, 1.0, 0.3]),
    ];
    
    let indices = vec![
        // Top bar
        0, 1, 5, 0, 5, 4,
        // Right bar
        1, 2, 6, 1, 6, 5,
        // Bottom bar
        2, 3, 7, 2, 7, 6,
        // Left bar
        3, 0, 4, 3, 4, 7,
    ];
    
    Mesh::new(device, vertices, indices)
}

fn main() {
    // Run the demo
    pollster::block_on(async {
        let config = EngineConfig {
            title: "Non-Euclidean Game Engine Demo".to_string(),
            width: 1280,
            height: 720,
            vsync: true,
            target_fps: Some(60),
            resizable: true,
        };
        
        let engine = Engine::new(config).await.expect("Failed to create engine");
        let demo = NonEuclideanDemo::new();
        
        engine.run(demo).expect("Failed to run demo");
    });
}