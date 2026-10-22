//! Portal system for connecting non-Euclidean spaces

use cgmath::{Point3, Vector3, Matrix4, InnerSpace, Transform, SquareMatrix};
use super::ChartId;

/// Unique identifier for a portal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortalId(pub u32);

/// Portal connection between two charts
#[derive(Debug, Clone)]
pub struct PortalConnection {
    pub portal_id: PortalId,
    pub from_chart: ChartId,
    pub to_chart: ChartId,
}

/// A portal that connects two charts in the manifold
#[derive(Clone)]
pub struct Portal {
    id: PortalId,
    from_chart: ChartId,
    to_chart: ChartId,
    from_position: Point3<f32>,
    to_position: Point3<f32>,
    transform: Matrix4<f32>,
    bounds: PortalBounds,
    active: bool,
    bidirectional: bool,
}

/// Portal boundary for intersection testing
#[derive(Debug, Clone)]
pub struct PortalBounds {
    pub center: Point3<f32>,
    pub normal: Vector3<f32>,
    pub width: f32,
    pub height: f32,
    pub shape: PortalShape,
}

#[derive(Debug, Clone, Copy)]
pub enum PortalShape {
    Rectangular,
    Circular,
    Custom,
}

impl Portal {
    /// Create a new portal
    pub fn new(
        id: PortalId,
        from_chart: ChartId,
        to_chart: ChartId,
        from_position: Point3<f32>,
        to_position: Point3<f32>,
        transform: Matrix4<f32>,
    ) -> Self {
        let bounds = PortalBounds {
            center: from_position,
            normal: Vector3::new(0.0, 0.0, 1.0),
            width: 2.0,
            height: 3.0,
            shape: PortalShape::Rectangular,
        };
        
        Self {
            id,
            from_chart,
            to_chart,
            from_position,
            to_position,
            transform,
            bounds,
            active: true,
            bidirectional: true,
        }
    }
    
    /// Configure custom portal bounds (normal, width, height, shape)
    pub fn with_bounds(
        mut self,
        normal: Vector3<f32>,
        width: f32,
        height: f32,
        shape: PortalShape,
    ) -> Self {
        self.bounds.normal = normal.normalize();
        self.bounds.width = width;
        self.bounds.height = height;
        self.bounds.shape = shape;
        self
    }

    /// Get portal bounds
    pub fn bounds(&self) -> &PortalBounds {
        &self.bounds
    }

    /// Get portal entry center position in source chart
    pub fn from_position(&self) -> Point3<f32> {
        self.from_position
    }

    /// Get portal exit center position in target chart
    pub fn to_position(&self) -> Point3<f32> {
        self.to_position
    }

    /// Get portal transformation matrix
    pub fn transform_matrix(&self) -> Matrix4<f32> {
        self.transform
    }

    /// Get portal ID
    pub fn id(&self) -> PortalId {
        self.id
    }
    
    /// Get source chart
    pub fn source_chart(&self) -> ChartId {
        self.from_chart
    }
    
    /// Get target chart
    pub fn target_chart(&self) -> ChartId {
        self.to_chart
    }
    
    /// Transform a point through the portal (pivoting around `from_position` to `to_position`)
    pub fn transform_point(&self, point: Point3<f32>) -> Point3<f32> {
        let local_offset = point - self.from_position;
        let transformed_offset = self.transform.transform_vector(local_offset);
        self.to_position + transformed_offset
    }
    
    /// Transform a vector (direction) through the portal
    pub fn transform_vector(&self, vector: Vector3<f32>) -> Vector3<f32> {
        self.transform.transform_vector(vector)
    }

    /// Check if a moving point segment `prev_pos -> curr_pos` crosses the active portal aperture
    pub fn check_crossing(
        &self,
        prev_pos: Point3<f32>,
        curr_pos: Point3<f32>,
        forward_dir: Vector3<f32>,
    ) -> Option<(Point3<f32>, Vector3<f32>, ChartId)> {
        if !self.active {
            return None;
        }
        let d_prev = (prev_pos - self.bounds.center).dot(self.bounds.normal);
        let d_curr = (curr_pos - self.bounds.center).dot(self.bounds.normal);

        // Must cross the portal plane between prev_pos and curr_pos
        if (d_prev > 0.0 && d_curr <= 0.0) || (self.bidirectional && d_prev < 0.0 && d_curr >= 0.0) {
            let denom = (d_prev - d_curr).abs().max(1e-6);
            let t = (d_prev.abs() / denom).clamp(0.0, 1.0);
            let crossing_pt = prev_pos + (curr_pos - prev_pos) * t;
            if self.contains_point(crossing_pt) {
                let new_pos = self.transform_point(curr_pos);
                let new_dir = self.transform_vector(forward_dir).normalize();
                return Some((new_pos, new_dir, self.to_chart));
            }
        }
        None
    }
    
    /// Check if a ray intersects the portal
    pub fn ray_intersection(&self, origin: Point3<f32>, direction: Vector3<f32>) -> Option<Point3<f32>> {
        // Plane equation: (p - center) · normal = 0
        let denominator = direction.dot(self.bounds.normal);
        
        // Ray is parallel to portal plane
        if denominator.abs() < 1e-6 {
            return None;
        }
        
        let t = (self.bounds.center - origin).dot(self.bounds.normal) / denominator;
        
        // Intersection is behind the ray origin
        if t < 0.0 {
            return None;
        }
        
        let intersection = origin + direction * t;
        
        // Check if intersection is within portal bounds
        if self.contains_point(intersection) {
            Some(intersection)
        } else {
            None
        }
    }
    
    /// Check if a point is within the portal bounds
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        let local = point - self.bounds.center;
        
        match self.bounds.shape {
            PortalShape::Rectangular => {
                // Project onto portal plane using singularity-free reference up vector
                let n = self.bounds.normal.normalize();
                let ref_up = if n.y.abs() > 0.99 {
                    Vector3::new(1.0, 0.0, 0.0)
                } else {
                    Vector3::new(0.0, 1.0, 0.0)
                };
                let right = n.cross(ref_up).normalize();
                let up = right.cross(n).normalize();
                
                let x = local.dot(right);
                let y = local.dot(up);
                
                x.abs() <= self.bounds.width / 2.0 && y.abs() <= self.bounds.height / 2.0
            }
            PortalShape::Circular => {
                let distance = (local - local.dot(self.bounds.normal) * self.bounds.normal).magnitude();
                distance <= self.bounds.width / 2.0
            }
            PortalShape::Custom => {
                // Custom shape logic would go here
                true
            }
        }
    }
    
    /// Get the view matrix looking through the portal
    pub fn get_view_matrix(&self, camera_position: Point3<f32>) -> Matrix4<f32> {
        // Transform camera position through portal
        let transformed_position = self.transform_point(camera_position);
        
        // Look towards the portal exit
        let look_at = self.to_position + self.transform_vector(self.bounds.normal);
        let up = self.transform_vector(Vector3::new(0.0, 1.0, 0.0));
        
        Matrix4::look_at_rh(transformed_position, look_at, up)
    }
    
    /// Check if portal is active
    pub fn is_active(&self) -> bool {
        self.active
    }
    
    /// Activate/deactivate portal
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
    
    /// Check if portal is bidirectional
    pub fn is_bidirectional(&self) -> bool {
        self.bidirectional
    }
    
    /// Create the reverse portal (for bidirectional connections)
    pub fn create_reverse(&self, id: PortalId) -> Portal {
        let inverse_transform = self.transform.invert()
            .unwrap_or(Matrix4::from_scale(1.0));
        
        Portal {
            id,
            from_chart: self.to_chart,
            to_chart: self.from_chart,
            from_position: self.to_position,
            to_position: self.from_position,
            transform: inverse_transform,
            bounds: PortalBounds {
                center: self.to_position,
                normal: self.transform_vector(self.bounds.normal),
                width: self.bounds.width,
                height: self.bounds.height,
                shape: self.bounds.shape,
            },
            active: self.active,
            bidirectional: self.bidirectional,
        }
    }
}

/// Portal renderer for visualizing portal edges and transitions
#[allow(dead_code)]
pub struct PortalRenderer {
    edge_color: [f32; 4],
    ripple_effect: bool,
    depth_fade: f32,
}

impl PortalRenderer {
    pub fn new() -> Self {
        Self {
            edge_color: [0.5, 0.8, 1.0, 0.8],
            ripple_effect: true,
            depth_fade: 0.1,
        }
    }
    
    /// Generate portal edge geometry for rendering
    pub fn generate_edge_mesh(&self, portal: &Portal) -> Vec<Point3<f32>> {
        let mut vertices = Vec::new();
        
        match portal.bounds.shape {
            PortalShape::Rectangular => {
                let half_width = portal.bounds.width / 2.0;
                let half_height = portal.bounds.height / 2.0;
                let center = portal.bounds.center;
                
                // Generate rectangle corners
                vertices.push(Point3::new(center.x - half_width, center.y - half_height, center.z));
                vertices.push(Point3::new(center.x + half_width, center.y - half_height, center.z));
                vertices.push(Point3::new(center.x + half_width, center.y + half_height, center.z));
                vertices.push(Point3::new(center.x - half_width, center.y + half_height, center.z));
            }
            PortalShape::Circular => {
                let center = portal.bounds.center;
                let radius = portal.bounds.width / 2.0;
                let segments = 32;
                
                for i in 0..segments {
                    let angle = (i as f32) * 2.0 * std::f32::consts::PI / segments as f32;
                    vertices.push(Point3::new(
                        center.x + radius * angle.cos(),
                        center.y + radius * angle.sin(),
                        center.z,
                    ));
                }
            }
            PortalShape::Custom => {
                // Custom shape vertices
                vertices.push(portal.bounds.center);
            }
        }
        
        vertices
    }
}