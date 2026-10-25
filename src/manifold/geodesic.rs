//! Geodesic computation for paths in curved spaces

use cgmath::{Point3, Vector3, InnerSpace, EuclideanSpace};
use super::{Metric, GeometryType};

/// A geodesic path through curved space
#[derive(Debug, Clone)]
pub struct GeodesicPath {
    pub points: Vec<Point3<f32>>,
    pub tangents: Vec<Vector3<f32>>,
    pub arc_length: f32,
    pub geometry: GeometryType,
}

impl GeodesicPath {
    /// Create a new geodesic path
    pub fn new(geometry: GeometryType) -> Self {
        Self {
            points: Vec::new(),
            tangents: Vec::new(),
            arc_length: 0.0,
            geometry,
        }
    }
    
    /// Add a point to the path
    pub fn add_point(&mut self, point: Point3<f32>, tangent: Vector3<f32>) {
        if let Some(last_point) = self.points.last() {
            self.arc_length += (point - *last_point).magnitude();
        }
        self.points.push(point);
        self.tangents.push(tangent.normalize());
    }
    
    /// Get interpolated position along the affinely parameterized geodesic path (`t` in `[0.0, 1.0]`)
    pub fn interpolate(&self, t: f32) -> Option<Point3<f32>> {
        let n = self.points.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(self.points[0]);
        }

        let t_clamped = t.clamp(0.0, 1.0);
        let scaled = t_clamped * ((n - 1) as f32);
        let i0 = (scaled.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let local_t = scaled - (i0 as f32);

        let p0 = self.points[i0];
        let p1 = self.points[i1];
        if i0 == i1 || local_t.abs() < 1e-7 {
            return Some(p0);
        }

        match self.geometry {
            GeometryType::Hyperbolic => {
                Some(Metric::gyrometric_geodesic_point_3d(p0, p1, local_t, -1.0))
            }
            GeometryType::Torus | GeometryType::KleinBottle => {
                let d = p1 - p0;
                // Guard against periodic boundary wrap jumps between adjacent samples
                if d.magnitude2() > 25.0 {
                    if local_t < 0.5 {
                        Some(p0)
                    } else {
                        Some(p1)
                    }
                } else {
                    Some(p0 + d * local_t)
                }
            }
            _ => Some(p0 + (p1 - p0) * local_t),
        }
    }
    
    /// Get tangent at parameter t
    pub fn tangent_at(&self, t: f32) -> Option<Vector3<f32>> {
        if self.tangents.is_empty() {
            return None;
        }
        
        let index = ((t * (self.tangents.len() - 1) as f32) as usize).min(self.tangents.len() - 1);
        Some(self.tangents[index])
    }
}

/// Geodesic solver for different geometries
pub struct Geodesic;

impl Geodesic {
    /// Compute geodesic between two points
    pub fn compute(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let steps = steps.max(1);
        match metric.geometry {
            GeometryType::Euclidean => Self::euclidean_geodesic(start, end, steps),
            GeometryType::Spherical => Self::spherical_geodesic(start, end, metric, steps),
            GeometryType::Hyperbolic => Self::hyperbolic_geodesic(start, end, metric, steps),
            GeometryType::Torus => Self::torus_geodesic(start, end, metric, steps),
            GeometryType::KleinBottle => Self::klein_bottle_geodesic(start, end, metric, steps),
            GeometryType::Custom => Self::numerical_geodesic(start, end, metric, steps),
        }
    }

    /// Integrate the initial-value geodesic ODE (ẍᵏ + Γᵏᵢⱼ ẋⁱ ẋʲ = 0) via 4th-Order Runge-Kutta (RK4)
    pub fn solve_rk4_ivp(
        origin: Point3<f32>,
        initial_velocity: Vector3<f32>,
        metric: &Metric,
        dt: f32,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(metric.geometry);
        let mut pos = Vector3::new(origin.x, origin.y, origin.z);
        let mut vel = initial_velocity;

        let init_dir = if vel.magnitude2() > 1e-12 {
            vel.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };
        path.add_point(origin, init_dir);

        let accel = |x: Vector3<f32>, v: Vector3<f32>| -> Vector3<f32> {
            let pt = Point3::new(x.x, x.y, x.z);
            let tensor = metric.tensor_at(pt);
            let cs = tensor.christoffel_symbols_at(pt);
            cs.geodesic_acceleration(x, v)
        };

        for _ in 0..steps {
            let k1_x = vel;
            let k1_v = accel(pos, vel);

            let x2 = pos + k1_x * (0.5 * dt);
            let v2 = vel + k1_v * (0.5 * dt);
            let k2_x = v2;
            let k2_v = accel(x2, v2);

            let x3 = pos + k2_x * (0.5 * dt);
            let v3 = vel + k2_v * (0.5 * dt);
            let k3_x = v3;
            let k3_v = accel(x3, v3);

            let x4 = pos + k3_x * dt;
            let v4 = vel + k3_v * dt;
            let k4_x = v4;
            let k4_v = accel(x4, v4);

            pos += (k1_x + k2_x * 2.0 + k3_x * 2.0 + k4_x) * (dt / 6.0);
            vel += (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);

            let dir = if vel.magnitude2() > 1e-12 {
                vel.normalize()
            } else {
                init_dir
            };
            path.add_point(Point3::new(pos.x, pos.y, pos.z), dir);
        }

        path
    }
    
    /// Straight line in Euclidean space
    fn euclidean_geodesic(start: Point3<f32>, end: Point3<f32>, steps: usize) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Euclidean);
        let diff = end - start;
        let direction = if diff.magnitude2() > 1e-12 {
            diff.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };
        
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let point = Point3::new(
                start.x + diff.x * t,
                start.y + diff.y * t,
                start.z + diff.z * t,
            );
            path.add_point(point, direction);
        }
        
        path
    }
    
    /// Great circle on a sphere
    fn spherical_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Spherical);
        let radius = metric.parameters.radius.max(1e-4);
        
        let start_vec = Vector3::new(start.x, start.y, start.z);
        let end_vec = Vector3::new(end.x, end.y, end.z);
        if start_vec.magnitude2() < 1e-10 || end_vec.magnitude2() < 1e-10 {
            return Self::euclidean_geodesic(start, end, steps);
        }

        let start_norm = start_vec.normalize() * radius;
        let end_norm = end_vec.normalize() * radius;
        
        let cos_angle = (start_norm.dot(end_norm) / (radius * radius)).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();
        let raw_axis = start_norm.cross(end_norm);
        let axis = if raw_axis.magnitude2() > 1e-10 {
            raw_axis.normalize()
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let a = (1.0 - t) * angle;
            let b = t * angle;
            
            let point = if angle.abs() > 0.001 {
                let sin_angle = angle.sin();
                let p = (start_norm * a.sin() + end_norm * b.sin()) / sin_angle;
                Point3::from_vec(p)
            } else {
                Point3::from_vec(start_norm + (end_norm - start_norm) * t)
            };
            
            let tangent = if i < steps {
                let next_t = (i + 1) as f32 / steps as f32;
                let next = if angle.abs() > 0.001 {
                    let sin_angle = angle.sin();
                    (start_norm * ((1.0 - next_t) * angle).sin() + end_norm * (next_t * angle).sin()) / sin_angle
                } else {
                    start_norm + (end_norm - start_norm) * next_t
                };
                let d = next - point.to_vec();
                if d.magnitude2() > 1e-12 {
                    d.normalize()
                } else {
                    axis.cross(point.to_vec()).normalize()
                }
            } else {
                let d = axis.cross(point.to_vec());
                if d.magnitude2() > 1e-12 {
                    d.normalize()
                } else {
                    Vector3::new(0.0, 0.0, 1.0)
                }
            };
            
            path.add_point(point, tangent);
        }
        
        path
    }
    
    /// Exact 3D Geodesic in the Hyperbolic Poincaré Ball Model (H³):
    /// Uses closed-form Ungar gyrometric geodesic interpolation
    /// γ(t) = start ⊕_K (t ⊗_K ((-start) ⊕_K end)) across all 3 dimensions (x, y, z).
    fn hyperbolic_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Hyperbolic);
        let k = if metric.parameters.curvature < 0.0 {
            metric.parameters.curvature
        } else {
            -1.0
        };

        // Clamp endpoints strictly inside the 3D Poincaré horizon
        let clamp_ball = |p: Point3<f32>| -> Point3<f32> {
            let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if r >= 0.99 {
                let s = 0.99 / r;
                Point3::new(p.x * s, p.y * s, p.z * s)
            } else {
                p
            }
        };
        let a = clamp_ball(start);
        let b = clamp_ball(end);

        let mut sampled = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            sampled.push(Metric::gyrometric_geodesic_point_3d(a, b, t, k));
        }

        for i in 0..=steps {
            let tangent = if i < steps {
                let d = sampled[i + 1] - sampled[i];
                if d.magnitude2() > 1e-12 {
                    d.normalize()
                } else {
                    Vector3::new(0.0, 0.0, 1.0)
                }
            } else if i > 0 {
                let d = sampled[i] - sampled[i - 1];
                if d.magnitude2() > 1e-12 {
                    d.normalize()
                } else {
                    Vector3::new(0.0, 0.0, 1.0)
                }
            } else {
                Vector3::new(0.0, 0.0, 1.0)
            };
            path.add_point(sampled[i], tangent);
        }

        // Store the exact Riemannian hyperbolic arc length in `path.arc_length`
        let riemannian_len = Metric::poincare_distance_3d(a, b);
        if riemannian_len.is_finite() && riemannian_len > 0.0 {
            path.arc_length = riemannian_len;
        }

        path
    }

    /// Shortest periodic geodesic on a flat 3-Torus T³ = R³ / (2R * Z)³
    fn torus_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Torus);
        let period = (metric.parameters.radius * 2.0).max(1e-4);
        let half = period * 0.5;

        // Minimum-image displacement vector across periodic boundaries
        let dx = (end.x - start.x) - period * ((end.x - start.x) / period).round();
        let dy = (end.y - start.y) - period * ((end.y - start.y) / period).round();
        let dz = (end.z - start.z) - period * ((end.z - start.z) / period).round();
        let delta = Vector3::new(dx, dy, dz);
        let dir = if delta.magnitude2() > 1e-12 {
            delta.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        let wrap_coord = |v: f32| -> f32 { v - period * ((v + half) / period).floor() };

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let raw = start + delta * t;
            let wrapped = Point3::new(wrap_coord(raw.x), wrap_coord(raw.y), wrap_coord(raw.z));
            path.add_point(wrapped, dir);
        }
        path.arc_length = delta.magnitude();
        path
    }

    /// Shortest geodesic on a non-orientable 3D Klein Bottle K³ (evaluating direct and z-parity-flipped branches)
    fn klein_bottle_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::KleinBottle);
        let period = (metric.parameters.radius * 2.0).max(1e-4);
        let half = period * 0.5;

        let mut best_delta = Vector3::new(0.0, 0.0, 0.0);
        let mut best_d2 = f32::INFINITY;

        for kz in -1..=1 {
            let parity = if kz != 0 { -1.0 } else { 1.0 };
            let bx_img = parity * end.x;
            let bz_img = end.z + (kz as f32) * period;
            let dx = (bx_img - start.x) - period * ((bx_img - start.x) / period).round();
            let dy = (end.y - start.y) - period * ((end.y - start.y) / period).round();
            let dz = bz_img - start.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_delta = Vector3::new(dx, dy, dz);
            }
        }

        let base_dir = if best_delta.magnitude2() > 1e-12 {
            best_delta.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let raw = start + best_delta * t;
            let kz = ((raw.z + half) / period).floor() as i32;
            let wz = raw.z - period * (kz as f32);
            let wy = raw.y - period * ((raw.y + half) / period).floor();
            let parity = if kz.rem_euclid(2) != 0 { -1.0 } else { 1.0 };
            let rx = raw.x * parity;
            let wx = rx - period * ((rx + half) / period).floor();
            let tangent = Vector3::new(base_dir.x * parity, base_dir.y, base_dir.z).normalize();
            path.add_point(Point3::new(wx, wy, wz), tangent);
        }
        path.arc_length = best_d2.sqrt();
        path
    }
    
    /// Numerical BVP geodesic solver using 4th-Order Runge-Kutta (RK4) Christoffel shooting refinement
    fn numerical_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let dt = 1.0 / (steps as f32);
        let mut v0 = end - start;
        if v0.magnitude2() < 1e-12 {
            return Self::euclidean_geodesic(start, end, steps);
        }

        // Iterative shooting refinement using RK4 integration of ẍᵏ = -Γᵏᵢⱼ ẋⁱ ẋʲ
        for _ in 0..6 {
            let trial = Self::solve_rk4_ivp(start, v0, metric, dt, steps);
            if let Some(&reached) = trial.points.last() {
                let err = end - reached;
                if err.magnitude2() < 1e-8 {
                    return trial;
                }
                v0 += err * 0.65;
            }
        }

        Self::solve_rk4_ivp(start, v0, metric, dt, steps)
    }
}

/// Ray casting in curved spaces
pub struct GeodesicRay {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub path: GeodesicPath,
    pub max_distance: f32,
}

impl GeodesicRay {
    /// Cast a ray through curved space using RK4 Christoffel geodesic integration
    pub fn cast(
        origin: Point3<f32>,
        direction: Vector3<f32>,
        metric: &Metric,
        max_distance: f32,
        steps: usize,
    ) -> Self {
        let steps = steps.max(1);
        let dir = if direction.magnitude2() > 1e-12 {
            direction.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };
        let dt = max_distance / (steps as f32);
        let path = Geodesic::solve_rk4_ivp(origin, dir, metric, dt, steps);
        
        Self {
            origin,
            direction: dir,
            path,
            max_distance,
        }
    }
    
    /// Get point along ray at distance t
    pub fn point_at(&self, t: f32) -> Option<Point3<f32>> {
        self.path.interpolate(t / self.max_distance)
    }
    
    /// Get tangent direction at distance t
    pub fn direction_at(&self, t: f32) -> Option<Vector3<f32>> {
        self.path.tangent_at(t / self.max_distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_3d_geodesic_preserves_z_axis() {
        let metric = Metric::from_geometry(GeometryType::Hyperbolic);
        let start = Point3::new(0.15, -0.20, 0.30);
        let end = Point3::new(-0.25, 0.35, 0.40);
        let path = Geodesic::compute(start, end, &metric, 32);
        assert_eq!(path.points.len(), 33);
        assert!((path.points[0] - start).magnitude() < 1e-5);
        assert!((path.points[32] - end).magnitude() < 1e-5);
        // Midpoint must match exact 3D gyrometric midpoint (non-zero z!)
        let mid_expected = Metric::gyrometric_geodesic_point_3d(start, end, 0.5, -1.0);
        assert!((path.points[16] - mid_expected).magnitude() < 1e-5);
        assert!(path.points[16].z > 0.20);
    }

    #[test]
    fn test_rk4_christoffel_geodesic_solver_converges() {
        let mut metric = Metric::from_geometry(GeometryType::Custom);
        metric.parameters.curvature = -0.8;
        let start = Point3::new(0.1, 0.1, -0.2);
        let end = Point3::new(-0.2, 0.25, 0.15);
        let path = Geodesic::compute(start, end, &metric, 24);
        let reached = *path.points.last().unwrap();
        assert!((reached - end).magnitude() < 1e-2);
    }

    #[test]
    fn test_klein_bottle_geodesic_wraps_with_parity_flip() {
        let metric = Metric::from_geometry(GeometryType::KleinBottle);
        let a = Point3::new(4.0, 0.0, -9.0);
        let b = Point3::new(-4.0, 0.0, 9.0);
        let path = Geodesic::compute(a, b, &metric, 20);
        // Shortest path across z = ±10 boundary in K³ has length 2.0 (since x=4 flips to -4)
        assert!((path.arc_length - 2.0).abs() < 1e-4);
    }
}