//! Metric tensor and geometry definitions for curved spaces

use cgmath::{Point3, Vector3, Matrix3, Matrix4, InnerSpace, SquareMatrix};
use super::GeodesicPath;

/// Type of geometry for a space region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeometryType {
    Euclidean,      // Flat space (zero curvature, R^3)
    Spherical,      // Positive curvature (S^3)
    Hyperbolic,     // Negative curvature (H^3 Poincaré ball)
    Torus,          // Flat periodic 3-Torus quotient manifold (T^3 = R^3 / Z^3)
    KleinBottle,    // Non-orientable parity-flipping quotient manifold (K^3)
    Custom,         // User-defined metric
}

/// Metric tensor at a point in space
#[derive(Debug, Clone, Copy)]
pub struct MetricTensor {
    pub g: Matrix3<f32>,  // Metric tensor components
    pub curvature: f32,   // Scalar curvature at this point
}

impl MetricTensor {
    /// Create identity metric (Euclidean)
    pub fn identity() -> Self {
        Self {
            g: Matrix3::from_diagonal(Vector3::new(1.0, 1.0, 1.0)),
            curvature: 0.0,
        }
    }
    
    /// Create spherical metric
    pub fn spherical(radius: f32, theta: f32, _phi: f32) -> Self {
        let r2 = radius * radius;
        let sin_theta = theta.sin();
        let sin2_theta = (sin_theta * sin_theta).max(1e-4);
        
        Self {
            g: Matrix3::new(
                1.0, 0.0, 0.0,
                0.0, r2, 0.0,
                0.0, 0.0, r2 * sin2_theta,
            ),
            curvature: 2.0 / r2,
        }
    }
    
    /// Create 2D/cylindrical hyperbolic metric (Poincaré disk model)
    pub fn hyperbolic_poincare(x: f32, y: f32) -> Self {
        let r2 = x * x + y * y;
        let lambda = 2.0 / (1.0 - r2).max(0.01);
        let lambda2 = lambda * lambda;
        
        Self {
            g: Matrix3::new(
                lambda2, 0.0, 0.0,
                0.0, lambda2, 0.0,
                0.0, 0.0, 1.0,
            ),
            curvature: -1.0,
        }
    }

    /// Create full 3D hyperbolic Poincaré ball metric: ds² = (2 / (1 - ||x||²))² (dx² + dy² + dz²)
    pub fn hyperbolic_poincare_3d(p: Point3<f32>) -> Self {
        let r2 = p.x * p.x + p.y * p.y + p.z * p.z;
        let lambda = 2.0 / (1.0 - r2).max(0.01);
        let lambda2 = lambda * lambda;

        Self {
            g: Matrix3::from_diagonal(Vector3::new(lambda2, lambda2, lambda2)),
            curvature: -1.0,
        }
    }
    
    /// Create 3D conformal metric (Poincaré ball for k < 0, stereographic sphere for k > 0)
    pub fn conformal_3d(p: Point3<f32>, curvature: f32) -> Self {
        let r2 = p.x * p.x + p.y * p.y + p.z * p.z;
        let denom = (1.0 + curvature * 0.25 * r2).max(0.02);
        let lambda = 2.0 / denom;
        let lambda2 = lambda * lambda;

        Self {
            g: Matrix3::from_diagonal(Vector3::new(lambda2, lambda2, lambda2)),
            curvature,
        }
    }

    /// Compute the norm of a vector using this metric
    pub fn norm(&self, v: Vector3<f32>) -> f32 {
        let gv = self.g * v;
        v.dot(gv).max(0.0).sqrt()
    }
    
    /// Compute inner product of two vectors
    pub fn inner_product(&self, v1: Vector3<f32>, v2: Vector3<f32>) -> f32 {
        let gv2 = self.g * v2;
        v1.dot(gv2)
    }
    
    /// Get Christoffel symbols for parallel transport
    pub fn christoffel_symbols(&self) -> ChristoffelSymbols {
        ChristoffelSymbols::from_metric(self)
    }

    /// Get analytical position-dependent Christoffel symbols at `point`
    pub fn christoffel_symbols_at(&self, point: Point3<f32>) -> ChristoffelSymbols {
        ChristoffelSymbols::from_conformal_point(point, self.curvature)
    }
}

/// Christoffel symbols for computing geodesics and parallel transport
pub struct ChristoffelSymbols {
    pub gamma: [[[f32; 3]; 3]; 3],  // Γⁱⱼₖ
}

impl ChristoffelSymbols {
    pub fn from_metric(metric: &MetricTensor) -> Self {
        // Default origin / reference point evaluation when point is not supplied
        Self::from_conformal_point(Point3::new(0.25, 0.25, 0.0), metric.curvature)
    }

    /// Exact analytical Christoffel symbols for 3D conformally flat space
    /// ds² = λ(x)² (dx² + dy² + dz²), where λ(x) = 2 / (1 + K ||x||²)
    /// Γⁱⱼₗ = ∂ₗ(ln λ) δⁱⱼ + ∂ⱼ(ln λ) δⁱₗ - ∂ᵢ(ln λ) δⱼₗ
    pub fn from_conformal_point(point: Point3<f32>, curvature: f32) -> Self {
        let mut gamma = [[[0.0; 3]; 3]; 3];
        if curvature.abs() < 1e-6 {
            return Self { gamma };
        }

        let mut coords = [point.x, point.y, point.z];
        let raw_r2 = coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
        // Project radially inside the conformal horizon (1 + K r² >= 0.04) so d_sigma remains a true gradient
        let k = curvature;
        let r2 = if k < 0.0 {
            let max_r2 = 0.96 / (-k);
            if raw_r2 > max_r2 {
                let scale = (max_r2 / raw_r2).sqrt();
                coords[0] *= scale;
                coords[1] *= scale;
                coords[2] *= scale;
                max_r2
            } else {
                raw_r2
            }
        } else {
            raw_r2
        };
        let denom = (1.0 + k * r2).max(0.04);
        let factor = (-2.0 * k) / denom;
        let d_sigma = [
            factor * coords[0],
            factor * coords[1],
            factor * coords[2],
        ];

        for i in 0..3 {
            for j in 0..3 {
                for l in 0..3 {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    let delta_il = if i == l { 1.0 } else { 0.0 };
                    let delta_jl = if j == l { 1.0 } else { 0.0 };
                    gamma[i][j][l] = delta_ij * d_sigma[l] + delta_il * d_sigma[j] - delta_jl * d_sigma[i];
                }
            }
        }

        Self { gamma }
    }
    
    /// Apply Christoffel symbols to compute geodesic acceleration: aⁱ = -Γⁱⱼₖ vʲ vᵏ
    pub fn geodesic_acceleration(&self, _position: Vector3<f32>, velocity: Vector3<f32>) -> Vector3<f32> {
        let mut accel = Vector3::new(0.0, 0.0, 0.0);
        
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    accel[i] -= self.gamma[i][j][k] * velocity[j] * velocity[k];
                }
            }
        }
        
        accel
    }
    /// Compute parallel transport differential: dVⁱ = -Γⁱⱼₖ Vʲ dxᵏ
    pub fn parallel_transport_delta(&self, vector: Vector3<f32>, dx: Vector3<f32>) -> Vector3<f32> {
        let mut dv = Vector3::new(0.0, 0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    dv[i] -= self.gamma[i][j][k] * vector[j] * dx[k];
                }
            }
        }
        dv
    }
}

/// Metric for a region of space
#[derive(Clone)]
pub struct Metric {
    pub geometry: GeometryType,
    pub scale: f32,
    pub parameters: MetricParameters,
}

/// Parameters defining the metric
#[derive(Debug, Clone)]
pub struct MetricParameters {
    pub curvature: f32,
    pub radius: f32,
    pub custom_fn: Option<fn(Point3<f32>) -> MetricTensor>,
}

impl Metric {
    /// Create metric from geometry type
    pub fn from_geometry(geometry: GeometryType) -> Self {
        let parameters = match geometry {
            GeometryType::Euclidean => MetricParameters {
                curvature: 0.0,
                radius: 1.0,
                custom_fn: None,
            },
            GeometryType::Spherical => MetricParameters {
                curvature: 1.0,
                radius: 10.0,
                custom_fn: None,
            },
            GeometryType::Hyperbolic => MetricParameters {
                curvature: -1.0,
                radius: 1.0,
                custom_fn: None,
            },
            GeometryType::Torus => MetricParameters {
                curvature: 0.0,
                radius: 10.0, // Default fundamental domain half-extent L = 10.0 (period = 20.0)
                custom_fn: None,
            },
            GeometryType::KleinBottle => MetricParameters {
                curvature: 0.0,
                radius: 10.0, // Default fundamental domain half-extent L = 10.0 with z-parity flip
                custom_fn: None,
            },
            GeometryType::Custom => MetricParameters {
                curvature: 0.0,
                radius: 1.0,
                custom_fn: None,
            },
        };
        
        Self {
            geometry,
            scale: 1.0,
            parameters,
        }
    }
    
    /// Get metric tensor at a point
    pub fn tensor_at(&self, point: Point3<f32>) -> MetricTensor {
        match self.geometry {
            GeometryType::Euclidean | GeometryType::Torus | GeometryType::KleinBottle => {
                MetricTensor::identity()
            }
            GeometryType::Spherical => {
                // Convert to spherical coordinates (safe against r = 0)
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt().max(1e-5);
                let theta = (point.z / r).clamp(-1.0, 1.0).acos();
                let phi = point.y.atan2(point.x);
                MetricTensor::spherical(self.parameters.radius, theta, phi)
            }
            GeometryType::Hyperbolic => {
                MetricTensor::hyperbolic_poincare_3d(point)
            }
            GeometryType::Custom => {
                if let Some(custom_fn) = self.parameters.custom_fn {
                    custom_fn(point)
                } else {
                    MetricTensor::conformal_3d(point, self.parameters.curvature)
                }
            }
        }
    }
    
    /// Compute distance between two points
    pub fn distance(&self, a: Point3<f32>, b: Point3<f32>) -> f32 {
        match self.geometry {
            GeometryType::Euclidean => {
                (b - a).magnitude()
            }
            GeometryType::Spherical => {
                // Great circle distance
                let r = self.parameters.radius;
                let a_norm = Vector3::new(a.x, a.y, a.z).normalize();
                let b_norm = Vector3::new(b.x, b.y, b.z).normalize();
                let cos_angle = a_norm.dot(b_norm).min(1.0).max(-1.0);
                r * cos_angle.acos()
            }
            GeometryType::Hyperbolic => {
                Self::poincare_distance_3d(a, b)
            }
            GeometryType::Torus => {
                Self::torus_distance_3d(a, b, self.parameters.radius * 2.0)
            }
            GeometryType::KleinBottle => {
                Self::klein_bottle_distance_3d(a, b, self.parameters.radius * 2.0)
            }
            GeometryType::Custom => {
                (b - a).magnitude()
            }
        }
    }

    /// Minimum-image distance on a flat 3-Torus T³ = R³ / (period * Z)³
    pub fn torus_distance_3d(a: Point3<f32>, b: Point3<f32>, period: f32) -> f32 {
        let p = period.max(1e-4);
        let dx = (b.x - a.x) - p * ((b.x - a.x) / p).round();
        let dy = (b.y - a.y) - p * ((b.y - a.y) / p).round();
        let dz = (b.z - a.z) - p * ((b.z - a.z) / p).round();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Geodesic distance on a non-orientable 3D Klein Bottle K³ with period `p`:
    /// Crossing z by ±k*p applies the parity involution x -> (-1)^k * x.
    pub fn klein_bottle_distance_3d(a: Point3<f32>, b: Point3<f32>, period: f32) -> f32 {
        let p = period.max(1e-4);
        let mut best_d2 = f32::INFINITY;
        for kz in -1..=1 {
            let parity = if kz != 0 { -1.0 } else { 1.0 };
            let bx_img = parity * b.x;
            let bz_img = b.z + (kz as f32) * p;
            let dx = (bx_img - a.x) - p * ((bx_img - a.x) / p).round();
            let dy = (b.y - a.y) - p * ((b.y - a.y) / p).round();
            let dz = bz_img - a.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
            }
        }
        best_d2.sqrt()
    }

    /// Exact 3D Poincaré ball geodesic distance:
    /// d(a, b) = arcosh(1 + 2 ||a - b||² / ((1 - ||a||²)(1 - ||b||²)))
    pub fn poincare_distance_3d(a: Point3<f32>, b: Point3<f32>) -> f32 {
        let a_r2 = a.x * a.x + a.y * a.y + a.z * a.z;
        let b_r2 = b.x * b.x + b.y * b.y + b.z * b.z;
        if a_r2 >= 0.999 || b_r2 >= 0.999 {
            return f32::INFINITY;
        }
        let diff = b - a;
        let delta2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        let arg = 1.0 + (2.0 * delta2) / ((1.0 - a_r2) * (1.0 - b_r2)).max(1e-6);
        arg.max(1.0).acosh()
    }

    /// Möbius gyrovector addition u ⊕_K v in the 3D Poincaré ball / stereographic sphere
    /// For Hyperbolic space use curvature_k < 0 (e.g. -1.0); for Euclidean 0.0; for Spherical > 0.
    pub fn mobius_add(u: Vector3<f32>, v: Vector3<f32>, curvature_k: f32) -> Vector3<f32> {
        if curvature_k.abs() < 1e-6 {
            return u + v;
        }
        let k = -curvature_k; // c = -K in Ungar's convention (+1 for unit Poincaré ball when K = -1)
        let u2 = u.dot(u);
        let v2 = v.dot(v);
        let uv = u.dot(v);
        let denom = (1.0 + 2.0 * k * uv + k * k * u2 * v2).max(1e-6);
        let coef_u = 1.0 + 2.0 * k * uv + k * v2;
        let coef_v = 1.0 - k * u2;
        (u * coef_u + v * coef_v) / denom
    }

    /// Ungar gyroscalar multiplication r ⊗_K u in the 3D Poincaré ball (K < 0) or sphere (K > 0)
    pub fn mobius_scalar_mul(r: f32, u: Vector3<f32>, curvature_k: f32) -> Vector3<f32> {
        let norm = u.magnitude();
        if norm < 1e-7 || r.abs() < 1e-7 {
            return Vector3::new(0.0, 0.0, 0.0);
        }
        if curvature_k.abs() < 1e-6 {
            return u * r;
        }
        let c_sqrt = curvature_k.abs().sqrt();
        let unit = u / norm;
        if curvature_k < 0.0 {
            let scaled = (c_sqrt * norm).min(0.999_9).atanh();
            unit * ((r * scaled).tanh() / c_sqrt)
        } else {
            let scaled = (c_sqrt * norm).atan();
            unit * ((r * scaled).tan() / c_sqrt)
        }
    }

    /// Closed-form 3D gyrometric geodesic point γ(t) = a ⊕_K (t ⊗_K ((-a) ⊕_K b)) for t ∈ [0, 1]
    pub fn gyrometric_geodesic_point_3d(
        a: Point3<f32>,
        b: Point3<f32>,
        t: f32,
        curvature_k: f32,
    ) -> Point3<f32> {
        let ua = Vector3::new(a.x, a.y, a.z);
        let ub = Vector3::new(b.x, b.y, b.z);
        let delta = Self::mobius_add(-ua, ub, curvature_k);
        let step = Self::mobius_scalar_mul(t, delta, curvature_k);
        let res = Self::mobius_add(ua, step, curvature_k);
        Point3::new(res.x, res.y, res.z)
    }
    
    /// Parallel transport a vector along a path using analytical Christoffel symbols
    pub fn parallel_transport(&self, vector: Vector3<f32>, path: &GeodesicPath) -> Vector3<f32> {
        let orig_mag = vector.magnitude();
        if orig_mag < 1e-6 {
            return vector;
        }
        let mut transported = vector;
        
        // Integrate parallel transport equation dVⁱ = -Γⁱⱼₖ(x) Vʲ dxᵏ along the path
        for i in 1..path.points.len() {
            let p0 = path.points[i - 1];
            let p1 = path.points[i];
            let dx = p1 - p0;
            
            let metric = self.tensor_at(p0);
            let symbols = metric.christoffel_symbols_at(p0);
            let dv = symbols.parallel_transport_delta(transported, dx);
            transported = transported + dv;
        }
        
        let new_mag = transported.magnitude();
        if new_mag > 1e-6 {
            (transported / new_mag) * orig_mag
        } else {
            vector
        }
    }
    
    /// Compute transport matrix for orientation
    pub fn compute_transport_matrix(&self, path: &GeodesicPath) -> Matrix4<f32> {
        // Build orthonormal frame and transport it
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 0.0);
        let z = Vector3::new(0.0, 0.0, 1.0);
        
        let tx = self.parallel_transport(x, path);
        let ty = self.parallel_transport(y, path);
        let tz = self.parallel_transport(z, path);
        
        Matrix4::from_cols(
            tx.extend(0.0),
            ty.extend(0.0),
            tz.extend(0.0),
            Vector3::new(0.0, 0.0, 0.0).extend(1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_christoffel_symbols_poincare_nonzero() {
        let p = Point3::new(0.3, 0.4, 0.0);
        let cs = ChristoffelSymbols::from_conformal_point(p, -1.0);
        // At (0.3, 0.4, 0.0), r² = 0.25, 1 - r² = 0.75, factor = 2 / 0.75 = 2.6667
        // Γ⁰₀₀ = d_sigma[0] = 2 * 0.3 / 0.75 = 0.8
        assert!((cs.gamma[0][0][0] - 0.8).abs() < 1e-4);
        // Γ⁰₁₁ = -d_sigma[0] = -0.8
        assert!((cs.gamma[0][1][1] + 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_mobius_addition_identity_and_inverse() {
        let u = Vector3::new(0.25, -0.15, 0.1);
        let zero = Vector3::new(0.0, 0.0, 0.0);
        let added_zero = Metric::mobius_add(u, zero, -1.0);
        assert!((added_zero - u).magnitude() < 1e-6);

        let inv = Metric::mobius_add(u, -u, -1.0);
        assert!(inv.magnitude() < 1e-6);
    }

    #[test]
    fn test_poincare_3d_distance_symmetry() {
        let a = Point3::new(0.1, 0.2, -0.1);
        let b = Point3::new(-0.3, 0.1, 0.2);
        let d_ab = Metric::poincare_distance_3d(a, b);
        let d_ba = Metric::poincare_distance_3d(b, a);
        assert!((d_ab - d_ba).abs() < 1e-5);
        assert!(d_ab > 0.0);
    }

    #[test]
    fn test_gyrometric_geodesic_midpoint_bisects_poincare_distance() {
        let a = Point3::new(0.15, -0.25, 0.20);
        let b = Point3::new(-0.30, 0.20, -0.15);
        let mid = Metric::gyrometric_geodesic_point_3d(a, b, 0.5, -1.0);
        let d_total = Metric::poincare_distance_3d(a, b);
        let d_a_mid = Metric::poincare_distance_3d(a, mid);
        let d_mid_b = Metric::poincare_distance_3d(mid, b);
        assert!((d_a_mid - d_mid_b).abs() < 1e-4);
        assert!(((d_a_mid + d_mid_b) - d_total).abs() < 1e-4);
    }

    #[test]
    fn test_torus_and_klein_bottle_distances() {
        let a = Point3::new(4.0, 0.0, -9.0);
        let b = Point3::new(-4.0, 0.0, 9.0);
        // On a period-20 Torus, dz wraps to -2.0, but dx stays -8.0 -> sqrt(64 + 4) = sqrt(68) = 8.246
        let d_torus = Metric::torus_distance_3d(a, b, 20.0);
        assert!((d_torus - 68.0_f32.sqrt()).abs() < 1e-4);

        // On a period-20 Klein Bottle, crossing z-boundary flips x -> -x, so b.x=-4 becomes +4 (= a.x)!
        // Thus dx = 0, dy = 0, dz = -2 -> distance is 2.0!
        let d_klein = Metric::klein_bottle_distance_3d(a, b, 20.0);
        assert!((d_klein - 2.0).abs() < 1e-4);
    }
}