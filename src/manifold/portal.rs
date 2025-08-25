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
    
    /// Transform a point through the portal
    pub fn transform_point(&self, point: Point3<f32>) -> Point3<f32> {
        // Apply portal transformation matrix
        let transformed = self.transform.transform_point(point);
        
        // Translate to target position
        let offset = self.to_position - self.from_position;
        Point3::new(
            transformed.x + offset.x,
            transformed.y + offset.y,
            transformed.z + offset.z,
        )
    }
    
    /// Transform a vector (direction) through the portal
    pub fn transform_vector(&self, vector: Vector3<f32>) -> Vector3<f32> {
        self.transform.transform_vector(vector)
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
                // Project onto portal plane
                let right = self.bounds.normal.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
                let up = self.bounds.normal.cross(right);
                
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