//! Local coordinate charts for manifold patches

use cgmath::{Point3, Vector3, Matrix4, InnerSpace, EuclideanSpace, SquareMatrix};
use super::{GeodesicPath, Metric, GeometryType};

/// Unique identifier for a chart
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChartId(pub u32);

/// Local coordinates within a chart
#[derive(Debug, Clone, Copy)]
pub struct LocalCoordinate(pub Point3<f32>);

impl LocalCoordinate {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Point3::new(x, y, z))
    }
    
    pub fn from_point(point: Point3<f32>) -> Self {
        Self(point)
    }
    
    pub fn to_point(&self) -> Point3<f32> {
        self.0
    }
}

/// A chart representing a local coordinate patch in the manifold
#[derive(Clone)]
pub struct Chart {
    id: ChartId,
    geometry: GeometryType,
    metric: Metric,
    bounds: ChartBounds,
    transform: Matrix4<f32>,
}

/// Bounds of a chart in local coordinates
#[derive(Debug, Clone)]
pub struct ChartBounds {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
    pub wrap_mode: WrapMode,
}

#[derive(Debug, Clone, Copy)]
pub enum WrapMode {
    None,           // No wrapping
    Periodic,       // Wrap like a torus
    Spherical,      // Wrap on a sphere
    Hyperbolic,     // Poincaré disk boundary
}

impl Chart {
    /// Create a new chart with specified geometry
    pub fn new(id: ChartId, geometry: GeometryType) -> Self {
        let metric = Metric::from_geometry(geometry);
        let bounds = match geometry {
            GeometryType::Euclidean => ChartBounds {
                min: Point3::new(-1000.0, -1000.0, -1000.0),
                max: Point3::new(1000.0, 1000.0, 1000.0),
                wrap_mode: WrapMode::None,
            },
            GeometryType::Spherical => ChartBounds {
                min: Point3::new(-1.0, -1.0, -1.0),
                max: Point3::new(1.0, 1.0, 1.0),
                wrap_mode: WrapMode::Spherical,
            },
            GeometryType::Hyperbolic => ChartBounds {
                min: Point3::new(-1.0, -1.0, -10.0),
                max: Point3::new(1.0, 1.0, 10.0),
                wrap_mode: WrapMode::Hyperbolic,
            },
            GeometryType::Custom => ChartBounds {
                min: Point3::new(-100.0, -100.0, -100.0),
                max: Point3::new(100.0, 100.0, 100.0),
                wrap_mode: WrapMode::None,
            },
        };
        
        Self {
            id,
            geometry,
            metric,
            bounds,
            transform: Matrix4::from_scale(1.0),
        }
    }
    
    /// Get chart ID
    pub fn id(&self) -> ChartId {
        self.id
    }
    
    /// Get geometry type
    pub fn geometry(&self) -> GeometryType {
        self.geometry
    }
    
    /// Get metric
    pub fn metric(&self) -> &Metric {
        &self.metric
    }
    
    /// Convert local coordinates to world coordinates
    pub fn to_world(&self, local: LocalCoordinate) -> Point3<f32> {
        let point = local.to_point();
        
        match self.geometry {
            GeometryType::Euclidean => {
                // Simple transformation
                Point3::from_homogeneous(self.transform * point.to_homogeneous())
            }
            GeometryType::Spherical => {
                // Project onto sphere
                let normalized = Vector3::new(point.x, point.y, point.z).normalize();
                let radius = 10.0; // Default sphere radius
                Point3::from_vec(normalized * radius)
            }
            GeometryType::Hyperbolic => {
                // Poincaré disk model
                let r = (point.x * point.x + point.y * point.y).sqrt();
                if r >= 0.99 {
                    let scale = 0.99 / r;
                    Point3::new(point.x * scale, point.y * scale, point.z)
                } else {
                    point
                }
            }
            GeometryType::Custom => {
                Point3::from_homogeneous(self.transform * point.to_homogeneous())
            }
        }
    }
    
    /// Convert world coordinates to local coordinates
    pub fn to_local(&self, world: Point3<f32>) -> LocalCoordinate {
        let inverse = self.transform.invert().unwrap_or(Matrix4::from_scale(1.0));
        let local = Point3::from_homogeneous(inverse * world.to_homogeneous());
        
        // Apply geometry-specific inverse transformations
        let adjusted = match self.geometry {
            GeometryType::Spherical => {
                // Inverse spherical projection
                let radius = (world.x * world.x + world.y * world.y + world.z * world.z).sqrt();
                if radius > 0.0 {
                    Point3::new(world.x / radius, world.y / radius, world.z / radius)
                } else {
                    local
                }
            }
            GeometryType::Hyperbolic => {
                // Inverse Poincaré disk
                local // Simplified - proper inverse Klein model would go here
            }
            _ => local,
        };
        
        LocalCoordinate::from_point(adjusted)
    }
    
    /// Check if a point is within chart bounds
    pub fn contains(&self, local: LocalCoordinate) -> bool {
        let point = local.to_point();
        
        match self.bounds.wrap_mode {
            WrapMode::None => {
                point.x >= self.bounds.min.x && point.x <= self.bounds.max.x &&
                point.y >= self.bounds.min.y && point.y <= self.bounds.max.y &&
                point.z >= self.bounds.min.z && point.z <= self.bounds.max.z
            }
            WrapMode::Spherical => {
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
                r <= 1.0
            }
            WrapMode::Hyperbolic => {
                let r = (point.x * point.x + point.y * point.y).sqrt();
                r < 1.0
            }
            WrapMode::Periodic => true, // Always contains (wraps around)
        }
    }
    
    /// Apply boundary wrapping to coordinates
    pub fn wrap_coordinates(&self, mut local: LocalCoordinate) -> LocalCoordinate {
        let point = local.to_point();
        
        match self.bounds.wrap_mode {
            WrapMode::Periodic => {
                let width = self.bounds.max.x - self.bounds.min.x;
                let height = self.bounds.max.y - self.bounds.min.y;
                
                let mut wrapped = point;
                if wrapped.x < self.bounds.min.x {
                    wrapped.x += width;
                } else if wrapped.x > self.bounds.max.x {
                    wrapped.x -= width;
                }
                
                if wrapped.y < self.bounds.min.y {
                    wrapped.y += height;
                } else if wrapped.y > self.bounds.max.y {
                    wrapped.y -= height;
                }
                
                LocalCoordinate::from_point(wrapped)
            }
            WrapMode::Hyperbolic => {
                // Keep within Poincaré disk
                let r = (point.x * point.x + point.y * point.y).sqrt();
                if r >= 0.99 {
                    let scale = 0.98 / r;
                    LocalCoordinate::new(point.x * scale, point.y * scale, point.z)
                } else {
                    local
                }
            }
            _ => local,
        }
    }
    
    /// Parallel transport a vector along a geodesic path
    pub fn parallel_transport(&self, vector: Vector3<f32>, path: &GeodesicPath) -> Vector3<f32> {
        self.metric.parallel_transport(vector, path)
    }
    
    /// Compute transport matrix for orientation preservation
    pub fn compute_transport_matrix(&self, path: &GeodesicPath) -> Matrix4<f32> {
        self.metric.compute_transport_matrix(path)
    }
    
    /// Get distance between two points using the metric
    pub fn distance(&self, a: LocalCoordinate, b: LocalCoordinate) -> f32 {
        self.metric.distance(a.to_point(), b.to_point())
    }
}