//! Metric tensor and geometry definitions for curved spaces

use cgmath::{Point3, Vector3, Matrix3, Matrix4, InnerSpace, SquareMatrix};
use super::GeodesicPath;

/// Type of geometry for a space region
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryType {
    Euclidean,      // Flat space (zero curvature)
    Spherical,      // Positive curvature
    Hyperbolic,     // Negative curvature
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
            g: Matrix3::from_diagonal(1.0),
            curvature: 0.0,
        }
    }
    
    /// Create spherical metric
    pub fn spherical(radius: f32, theta: f32, phi: f32) -> Self {
        let r2 = radius * radius;
        let sin_theta = theta.sin();
        let sin2_theta = sin_theta * sin_theta;
        
        Self {
            g: Matrix3::new(
                1.0, 0.0, 0.0,
                0.0, r2, 0.0,
                0.0, 0.0, r2 * sin2_theta,
            ),
            curvature: 2.0 / r2,
        }
    }
    
    /// Create hyperbolic metric (Poincaré disk model)
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
    
    /// Compute the norm of a vector using this metric
    pub fn norm(&self, v: Vector3<f32>) -> f32 {
        let gv = self.g * v;
        v.dot(gv).sqrt()
    }
    
    /// Compute inner product of two vectors
    pub fn inner_product(&self, v1: Vector3<f32>, v2: Vector3<f32>) -> f32 {
        let gv2 = self.g * v2;
        v1.dot(gv2)
    }
    
    /// Get Christoffel symbols for parallel transport
    pub fn christoffel_symbols(&self) -> ChristoffelSymbols {
        // Simplified computation for common geometries
        ChristoffelSymbols::from_metric(self)
    }
}

/// Christoffel symbols for computing geodesics and parallel transport
pub struct ChristoffelSymbols {
    pub gamma: [[[f32; 3]; 3]; 3],  // Γⁱⱼₖ
}

impl ChristoffelSymbols {
    pub fn from_metric(metric: &MetricTensor) -> Self {
        // Simplified - full computation would involve metric derivatives
        let mut gamma = [[[0.0; 3]; 3]; 3];
        
        // For hyperbolic geometry in Poincaré disk
        if metric.curvature < 0.0 {
            // Non-zero Christoffel symbols for Poincaré metric
            // These would be computed from metric derivatives
        }
        
        Self { gamma }
    }
    
    /// Apply Christoffel symbols to compute geodesic acceleration
    pub fn geodesic_acceleration(&self, position: Vector3<f32>, velocity: Vector3<f32>) -> Vector3<f32> {
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
            GeometryType::Euclidean => MetricTensor::identity(),
            GeometryType::Spherical => {
                // Convert to spherical coordinates
                let r = (point.x * point.x + point.y * point.y + point.z * point.z).sqrt();
                let theta = (point.z / r).acos();
                let phi = point.y.atan2(point.x);
                MetricTensor::spherical(self.parameters.radius, theta, phi)
            }
            GeometryType::Hyperbolic => {
                MetricTensor::hyperbolic_poincare(point.x, point.y)
            }
            GeometryType::Custom => {
                if let Some(custom_fn) = self.parameters.custom_fn {
                    custom_fn(point)
                } else {
                    MetricTensor::identity()
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
                // Poincaré disk distance
                let a_r = (a.x * a.x + a.y * a.y).sqrt();
                let b_r = (b.x * b.x + b.y * b.y).sqrt();
                
                if a_r >= 0.99 || b_r >= 0.99 {
                    return f32::INFINITY;
                }
                
                let delta = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
                let numerator = 2.0 * delta;
                let denominator = (1.0 - a_r * a_r) * (1.0 - b_r * b_r);
                
                (1.0 + numerator / denominator.sqrt()).ln()
            }
            GeometryType::Custom => {
                // Fallback to Euclidean
                (b - a).magnitude()
            }
        }
    }
    
    /// Parallel transport a vector along a path
    pub fn parallel_transport(&self, vector: Vector3<f32>, path: &GeodesicPath) -> Vector3<f32> {
        let mut transported = vector;
        
        // Integrate parallel transport equation along the path
        for i in 1..path.points.len() {
            let p0 = path.points[i - 1];
            let p1 = path.points[i];
            let tangent = (p1 - p0).normalize();
            
            let metric = self.tensor_at(p0);
            let symbols = metric.christoffel_symbols();
            
            // Update vector using parallel transport equation
            let correction = symbols.geodesic_acceleration(
                Vector3::new(p0.x, p0.y, p0.z),
                tangent,
            );
            
            transported = transported - correction * 0.01; // Small step
        }
        
        transported.normalize() * vector.magnitude()
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