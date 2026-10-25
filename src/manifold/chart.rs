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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    None,           // No wrapping
    Periodic,       // Periodic 3-Torus wrapping (T³ = R³ / Z³)
    KleinBottle,    // Non-orientable Klein Bottle wrapping (crossing Z flips X -> -X and chirality)
    Spherical,      // Wrap on a 3-sphere S³
    Hyperbolic,     // 3D Poincaré ball boundary (||x|| < 1)
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
                min: Point3::new(-1.0, -1.0, -1.0),
                max: Point3::new(1.0, 1.0, 1.0),
                wrap_mode: WrapMode::Hyperbolic,
            },
            GeometryType::Torus => ChartBounds {
                min: Point3::new(-10.0, -10.0, -10.0),
                max: Point3::new(10.0, 10.0, 10.0),
                wrap_mode: WrapMode::Periodic,
            },
            GeometryType::KleinBottle => ChartBounds {
                min: Point3::new(-10.0, -10.0, -10.0),
                max: Point3::new(10.0, 10.0, 10.0),
                wrap_mode: WrapMode::KleinBottle,
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

    /// Get chart bounds
    pub fn bounds(&self) -> &ChartBounds {
        &self.bounds
    }

    /// Map a point in the 3D Poincaré ball (||p|| < 1) to the Beltrami-Klein ball model:
    /// x_K = 2 x_P / (1 + ||x_P||²)
    pub fn poincare_to_klein(p: Point3<f32>) -> Point3<f32> {
        let r2 = p.x * p.x + p.y * p.y + p.z * p.z;
        let clamped_p = if r2 >= 0.998 {
            let s = (0.998 / r2).sqrt();
            Point3::new(p.x * s, p.y * s, p.z * s)
        } else {
            p
        };
        let cr2 = clamped_p.x * clamped_p.x + clamped_p.y * clamped_p.y + clamped_p.z * clamped_p.z;
        let factor = 2.0 / (1.0 + cr2);
        Point3::new(clamped_p.x * factor, clamped_p.y * factor, clamped_p.z * factor)
    }

    /// Inverse map from the 3D Beltrami-Klein ball (||k|| < 1) to the 3D Poincaré ball:
    /// x_P = x_K / (1 + sqrt(1 - ||x_K||²))
    pub fn klein_to_poincare(k: Point3<f32>) -> Point3<f32> {
        let r2 = k.x * k.x + k.y * k.y + k.z * k.z;
        let clamped_k = if r2 >= 0.998 {
            let s = (0.998 / r2).sqrt();
            Point3::new(k.x * s, k.y * s, k.z * s)
        } else {
            k
        };
        let cr2 = clamped_k.x * clamped_k.x + clamped_k.y * clamped_k.y + clamped_k.z * clamped_k.z;
        let denom = 1.0 + (1.0 - cr2).max(0.0).sqrt();
        Point3::new(clamped_k.x / denom, clamped_k.y / denom, clamped_k.z / denom)
    }
    
    /// Convert local coordinates to world coordinates
    pub fn to_world(&self, local: LocalCoordinate) -> Point3<f32> {
        let point = local.to_point();
        
        match self.geometry {
            GeometryType::Euclidean | GeometryType::Custom => {
                Point3::from_homogeneous(self.transform * point.to_homogeneous())
            }
            GeometryType::Torus | GeometryType::KleinBottle => {
                let wrapped = self.wrap_coordinates(local).to_point();
                Point3::from_homogeneous(self.transform * wrapped.to_homogeneous())
            }
            GeometryType::Spherical => {
                let mag = Vector3::new(point.x, point.y, point.z).magnitude();
                if mag > 1e-6 {
                    let normalized = Vector3::new(point.x, point.y, point.z) / mag;
                    let radius = self.metric.parameters.radius;
                    Point3::from_vec(normalized * radius)
                } else {
                    Point3::new(0.0, 0.0, self.metric.parameters.radius)
                }
            }
            GeometryType::Hyperbolic => {
                // Full 3D Poincaré ball model (||x|| < 1)
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
                if r >= 0.99 {
                    let scale = 0.99 / r;
                    Point3::new(point.x * scale, point.y * scale, point.z * scale)
                } else {
                    point
                }
            }
        }
    }
    
    /// Convert world coordinates to local coordinates
    pub fn to_local(&self, world: Point3<f32>) -> LocalCoordinate {
        let inverse = self.transform.invert().unwrap_or(Matrix4::from_scale(1.0));
        let local = Point3::from_homogeneous(inverse * world.to_homogeneous());
        
        let adjusted = match self.geometry {
            GeometryType::Spherical => {
                let radius = (world.x * world.x + world.y * world.y + world.z * world.z).sqrt();
                if radius > 1e-6 {
                    Point3::new(world.x / radius, world.y / radius, world.z / radius)
                } else {
                    local
                }
            }
            GeometryType::Hyperbolic => {
                // Clamp into the open 3D Poincaré unit ball
                let r = (local.x * local.x + local.y * local.y + local.z * local.z).sqrt();
                if r >= 0.99 {
                    let s = 0.99 / r;
                    Point3::new(local.x * s, local.y * s, local.z * s)
                } else {
                    local
                }
            }
            GeometryType::Torus | GeometryType::KleinBottle => {
                self.wrap_coordinates(LocalCoordinate::from_point(local)).to_point()
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
                // Full 3D Poincaré ball check
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
                r < 1.0
            }
            WrapMode::Periodic | WrapMode::KleinBottle => true, // Quotient manifolds wrap seamlessly
        }
    }
    
    /// Apply boundary wrapping to coordinates
    pub fn wrap_coordinates(&self, local: LocalCoordinate) -> LocalCoordinate {
        let (wrapped, _, _) = self.wrap_coordinates_with_parity(local, Vector3::new(0.0, 0.0, 1.0));
        wrapped
    }

    /// Wrap coordinates and tangent vector across quotient manifold boundaries.
    /// Returns `(wrapped_coord, wrapped_tangent, orientation_parity)` where `orientation_parity`
    /// is `+1` in orientable manifolds (`Torus`) and flips by `-1` each time a trajectory crosses
    /// the non-orientable $Z$-identification boundary of a `KleinBottle`.
    pub fn wrap_coordinates_with_parity(
        &self,
        local: LocalCoordinate,
        mut tangent: Vector3<f32>,
    ) -> (LocalCoordinate, Vector3<f32>, i32) {
        let point = local.to_point();
        let mut parity = 1i32;

        match self.bounds.wrap_mode {
            WrapMode::Periodic => {
                let width = (self.bounds.max.x - self.bounds.min.x).max(1e-4);
                let height = (self.bounds.max.y - self.bounds.min.y).max(1e-4);
                let depth = (self.bounds.max.z - self.bounds.min.z).max(1e-4);

                let wx = point.x - width * ((point.x - self.bounds.min.x) / width).floor();
                let wy = point.y - height * ((point.y - self.bounds.min.y) / height).floor();
                let wz = point.z - depth * ((point.z - self.bounds.min.z) / depth).floor();

                (LocalCoordinate::new(wx, wy, wz), tangent, 1)
            }
            WrapMode::KleinBottle => {
                let width = (self.bounds.max.x - self.bounds.min.x).max(1e-4);
                let height = (self.bounds.max.y - self.bounds.min.y).max(1e-4);
                let depth = (self.bounds.max.z - self.bounds.min.z).max(1e-4);

                let kz = ((point.z - self.bounds.min.z) / depth).floor() as i32;
                let wz = point.z - depth * (kz as f32);
                let wy = point.y - height * ((point.y - self.bounds.min.y) / height).floor();

                let mut x_in = point.x;
                if kz.rem_euclid(2) != 0 {
                    // Non-orientable Klein identification: crossing Z boundary reflects X -> -X
                    x_in = -x_in;
                    tangent.x = -tangent.x;
                    parity = -1;
                }
                let wx = x_in - width * ((x_in - self.bounds.min.x) / width).floor();
                (LocalCoordinate::new(wx, wy, wz), tangent, parity)
            }
            WrapMode::Hyperbolic => {
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
                if r >= 0.99 {
                    let scale = 0.98 / r;
                    (
                        LocalCoordinate::new(point.x * scale, point.y * scale, point.z * scale),
                        tangent,
                        1,
                    )
                } else {
                    (local, tangent, 1)
                }
            }
            _ => (local, tangent, 1),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_klein_roundtrip_3d() {
        let p = Point3::new(0.35, -0.42, 0.28);
        let k = Chart::poincare_to_klein(p);
        let p_back = Chart::klein_to_poincare(k);
        assert!((p_back - p).magnitude() < 1e-5);
    }

    #[test]
    fn test_klein_bottle_parity_flip_on_z_wrap() {
        let chart = Chart::new(ChartId(10), GeometryType::KleinBottle);
        // Bounds are [-10, 10]^3. A point at z = 12.0 wraps to z = -8.0 with kz = 1 (odd), flipping x -> -x!
        let pos = LocalCoordinate::new(3.5, 2.0, 12.0);
        let vel = Vector3::new(1.0, 0.0, 2.0);
        let (wrapped, wrapped_vel, parity) = chart.wrap_coordinates_with_parity(pos, vel);
        assert_eq!(parity, -1);
        assert!((wrapped.0.x - (-3.5)).abs() < 1e-4);
        assert!((wrapped.0.z - (-8.0)).abs() < 1e-4);
        assert!((wrapped_vel.x - (-1.0)).abs() < 1e-4);
    }
}