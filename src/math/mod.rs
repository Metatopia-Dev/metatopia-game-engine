//! Math utilities for the non-Euclidean engine

pub use cgmath::{
    Vector2 as Vec2,
    Vector3 as Vec3,
    Vector4 as Vec4,
    Matrix3 as Mat3,
    Matrix4 as Mat4,
    Point2,
    Point3,
    Quaternion,
    Euler,
    Rad,
    Deg,
    InnerSpace,
    MetricSpace,
    SquareMatrix,
    EuclideanSpace,
    Transform as CgTransform,
};

use cgmath::{Matrix4, Vector3};

/// Transform wrapper for non-Euclidean spaces
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub matrix: Matrix4<f32>,
}

impl Transform {
    /// Create identity transform
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::from_scale(1.0),
        }
    }
    
    /// Create from position, rotation, and scale
    pub fn from_trs(position: Point3<f32>, rotation: Quaternion<f32>, scale: f32) -> Self {
        let translation = Matrix4::from_translation(position.to_vec());
        let rotation = Matrix4::from(rotation);
        let scale = Matrix4::from_scale(scale);
        
        Self {
            matrix: translation * rotation * scale,
        }
    }
    
    /// Get position from transform
    pub fn position(&self) -> Point3<f32> {
        Point3::new(self.matrix.w.x, self.matrix.w.y, self.matrix.w.z)
    }
    
    /// Apply transform to a point
    pub fn transform_point(&self, point: Point3<f32>) -> Point3<f32> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }
    
    /// Apply transform to a vector
    pub fn transform_vector(&self, vector: Vector3<f32>) -> Vector3<f32> {
        (self.matrix * vector.extend(0.0)).truncate()
    }
    
    /// Compose with another transform
    pub fn compose(&self, other: &Transform) -> Transform {
        Transform {
            matrix: self.matrix * other.matrix,
        }
    }
    
    /// Get inverse transform
    pub fn inverse(&self) -> Option<Transform> {
        self.matrix.invert().map(|matrix| Transform { matrix })
    }
}

/// Interpolation utilities for smooth transitions
pub struct Interpolation;

impl Interpolation {
    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    
    /// Spherical linear interpolation for quaternions
    pub fn slerp(a: Quaternion<f32>, b: Quaternion<f32>, t: f32) -> Quaternion<f32> {
        let dot = a.s * b.s + a.v.x * b.v.x + a.v.y * b.v.y + a.v.z * b.v.z;
        
        let (b, dot) = if dot < 0.0 {
            (Quaternion::new(-b.s, -b.v.x, -b.v.y, -b.v.z), -dot)
        } else {
            (b, dot)
        };
        
        if dot > 0.9995 {
            // Linear interpolation for very close quaternions
            let result = Quaternion::new(
                Self::lerp(a.s, b.s, t),
                Self::lerp(a.v.x, b.v.x, t),
                Self::lerp(a.v.y, b.v.y, t),
                Self::lerp(a.v.z, b.v.z, t),
            );
            return result.normalize();
        }
        
        let theta = dot.acos();
        let sin_theta = theta.sin();
        
        let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
        let scale_b = (t * theta).sin() / sin_theta;
        
        Quaternion::new(
            scale_a * a.s + scale_b * b.s,
            scale_a * a.v.x + scale_b * b.v.x,
            scale_a * a.v.y + scale_b * b.v.y,
            scale_a * a.v.z + scale_b * b.v.z,
        )
    }
    
    /// Smooth step interpolation
    pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Ray for ray casting
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }
    
    pub fn point_at(&self, t: f32) -> Point3<f32> {
        self.origin + self.direction * t
    }
}

/// Bounding box for spatial queries
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl BoundingBox {
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }
    
    pub fn from_points(points: &[Point3<f32>]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        
        let mut min = points[0];
        let mut max = points[0];
        
        for point in points.iter().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);
            
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }
        
        Some(Self { min, max })
    }
    
    pub fn contains(&self, point: Point3<f32>) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }
    
    pub fn intersects_ray(&self, ray: &Ray) -> Option<f32> {
        let inv_dir = Vector3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        
        let t1 = (self.min.x - ray.origin.x) * inv_dir.x;
        let t2 = (self.max.x - ray.origin.x) * inv_dir.x;
        let t3 = (self.min.y - ray.origin.y) * inv_dir.y;
        let t4 = (self.max.y - ray.origin.y) * inv_dir.y;
        let t5 = (self.min.z - ray.origin.z) * inv_dir.z;
        let t6 = (self.max.z - ray.origin.z) * inv_dir.z;
        
        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));
        
        if tmax < 0.0 || tmin > tmax {
            None
        } else {
            Some(if tmin < 0.0 { tmax } else { tmin })
        }
    }
}