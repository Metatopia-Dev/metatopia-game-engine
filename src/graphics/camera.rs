//! Camera system for non-Euclidean rendering

use cgmath::{Point3, Vector3, Matrix4, Rad, perspective, InnerSpace};
use crate::manifold::{ManifoldPosition, ChartId, GeometryType};

/// Camera for viewing non-Euclidean spaces
pub struct Camera {
    pub position: ManifoldPosition,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fovy: Rad<f32>,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
    pub view_matrix: Matrix4<f32>,
    pub projection_matrix: Matrix4<f32>,
    pub geometry_type: GeometryType,
}

impl Camera {
    /// Create a new camera
    pub fn new(
        chart_id: ChartId,
        position: Point3<f32>,
        target: Point3<f32>,
        aspect: f32,
    ) -> Self {
        let up = Vector3::new(0.0, 1.0, 0.0);
        let fovy = Rad(45.0_f32.to_radians());
        let znear = 0.1;
        let zfar = 1000.0;
        
        let view_matrix = Matrix4::look_at_rh(position, target, up);
        let projection_matrix = perspective(fovy, aspect, znear, zfar);
        
        Self {
            position: ManifoldPosition::new(chart_id, position),
            target,
            up,
            fovy,
            aspect,
            znear,
            zfar,
            view_matrix,
            projection_matrix,
            geometry_type: GeometryType::Euclidean,
        }
    }
    
    /// Update camera matrices based on geometry
    pub fn update(&mut self, manifold: &crate::manifold::Manifold) {
        if let Some(chart) = manifold.chart(self.position.chart_id) {
            self.geometry_type = chart.geometry();
            
            let eye = self.position.local.to_point();
            
            // Adjust view matrix based on geometry
            match self.geometry_type {
                GeometryType::Euclidean => {
                    self.view_matrix = Matrix4::look_at_rh(eye, self.target, self.up);
                }
                GeometryType::Spherical => {
                    // Adjust for spherical viewing
                    let adjusted_up = self.parallel_transport_up(manifold);
                    self.view_matrix = Matrix4::look_at_rh(eye, self.target, adjusted_up);
                }
                GeometryType::Hyperbolic => {
                    // Poincaré disk viewing adjustments
                    let adjusted_view = self.hyperbolic_view_matrix(eye);
                    self.view_matrix = adjusted_view;
                }
                GeometryType::Custom => {
                    self.view_matrix = Matrix4::look_at_rh(eye, self.target, self.up);
                }
            }
            
            // Update projection based on geometry
            self.update_projection();
        }
    }
    
    /// Update projection matrix based on geometry
    fn update_projection(&mut self) {
        match self.geometry_type {
            GeometryType::Hyperbolic => {
                // Wider FOV for hyperbolic space
                let hyperbolic_fov = Rad((self.fovy.0 * 1.5).min(170.0_f32.to_radians()));
                self.projection_matrix = perspective(hyperbolic_fov, self.aspect, self.znear, self.zfar);
            }
            GeometryType::Spherical => {
                // Adjust for spherical distortion
                let spherical_fov = Rad((self.fovy.0 * 0.9).max(30.0_f32.to_radians()));
                self.projection_matrix = perspective(spherical_fov, self.aspect, self.znear, self.zfar);
            }
            _ => {
                self.projection_matrix = perspective(self.fovy, self.aspect, self.znear, self.zfar);
            }
        }
    }
    
    /// Compute hyperbolic view matrix for Poincaré disk
    fn hyperbolic_view_matrix(&self, eye: Point3<f32>) -> Matrix4<f32> {
        // Apply Möbius transformation for hyperbolic view
        let r = (eye.x * eye.x + eye.y * eye.y).sqrt();
        
        if r < 0.99 {
            // Standard look-at with hyperbolic adjustment
            let factor = 2.0 / (1.0 + r * r);
            let adjusted_eye = Point3::new(
                eye.x * factor,
                eye.y * factor,
                eye.z,
            );
            Matrix4::look_at_rh(adjusted_eye, self.target, self.up)
        } else {
            // Near boundary, use special handling
            Matrix4::look_at_rh(eye, self.target, self.up)
        }
    }
    
    /// Parallel transport the up vector in curved space
    fn parallel_transport_up(&self, manifold: &crate::manifold::Manifold) -> Vector3<f32> {
        // Simplified parallel transport for camera up vector
        let position = self.position.local.to_point();
        
        // Create a small path for parallel transport
        let path_end = position + self.up * 0.1;
        if let Some(path) = manifold.compute_geodesic(
            position,
            path_end,
            self.position.chart_id,
            5,
        ) {
            if let Some(transported) = manifold.parallel_transport(
                self.up,
                &path,
                self.position.chart_id,
            ) {
                return transported.normalize();
            }
        }
        
        self.up
    }
    
    /// Move camera in local chart coordinates
    pub fn move_local(&mut self, delta: Vector3<f32>) {
        let new_pos = self.position.local.to_point() + delta;
        self.position.local = crate::manifold::LocalCoordinate::from_point(new_pos);
    }
    
    /// Rotate camera
    pub fn rotate(&mut self, yaw: Rad<f32>, pitch: Rad<f32>) {
        let position = self.position.local.to_point();
        let forward = (self.target - position).normalize();
        let right = forward.cross(self.up).normalize();
        
        // Apply yaw rotation
        let yaw_rotation = Matrix4::from_axis_angle(self.up, yaw);
        let forward = (yaw_rotation * forward.extend(0.0)).truncate();
        
        // Apply pitch rotation
        let pitch_rotation = Matrix4::from_axis_angle(right, pitch);
        let forward = (pitch_rotation * forward.extend(0.0)).truncate();
        
        self.target = position + forward;
    }
    
    /// Set camera position in manifold
    pub fn set_position(&mut self, chart_id: ChartId, position: Point3<f32>) {
        self.position = ManifoldPosition::new(chart_id, position);
    }
    
    /// Get view-projection matrix
    pub fn view_projection(&self) -> Matrix4<f32> {
        self.projection_matrix * self.view_matrix
    }
    
    /// Get camera forward direction
    pub fn forward(&self) -> Vector3<f32> {
        let position = self.position.local.to_point();
        (self.target - position).normalize()
    }
    
    /// Get camera right direction
    pub fn right(&self) -> Vector3<f32> {
        self.forward().cross(self.up).normalize()
    }
    
    /// Resize camera viewport
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
        self.update_projection();
    }
}

/// First-person camera controller
pub struct FPSCameraController {
    pub move_speed: f32,
    pub rotate_speed: f32,
    pub sensitivity: f32,
}

impl FPSCameraController {
    pub fn new() -> Self {
        Self {
            move_speed: 5.0,
            rotate_speed: 1.0,
            sensitivity: 0.002,
        }
    }
    
    pub fn update(
        &self,
        camera: &mut Camera,
        input: &crate::input::InputManager,
        dt: f32,
    ) {
        use crate::input::KeyCode;
        
        // Movement
        let mut movement = Vector3::new(0.0, 0.0, 0.0);
        
        if input.is_key_pressed(KeyCode::W) {
            movement += camera.forward();
        }
        if input.is_key_pressed(KeyCode::S) {
            movement -= camera.forward();
        }
        if input.is_key_pressed(KeyCode::A) {
            movement -= camera.right();
        }
        if input.is_key_pressed(KeyCode::D) {
            movement += camera.right();
        }
        if input.is_key_pressed(KeyCode::Space) {
            movement += Vector3::new(0.0, 1.0, 0.0);
        }
        if input.is_key_pressed(KeyCode::LeftShift) {
            movement -= Vector3::new(0.0, 1.0, 0.0);
        }
        
        if movement.magnitude() > 0.0 {
            movement = movement.normalize() * self.move_speed * dt;
            camera.move_local(movement);
        }
        
        // Rotation (mouse look)
        let mouse_delta = input.mouse_delta();
        if mouse_delta.x != 0.0 || mouse_delta.y != 0.0 {
            let yaw = Rad(-mouse_delta.x * self.sensitivity);
            let pitch = Rad(-mouse_delta.y * self.sensitivity);
            camera.rotate(yaw, pitch);
        }
    }
}