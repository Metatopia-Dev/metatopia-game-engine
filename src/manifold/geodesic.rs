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
    
    /// Get interpolated position along the path
    pub fn interpolate(&self, t: f32) -> Option<Point3<f32>> {
        if self.points.is_empty() {
            return None;
        }
        
        let t_clamped = t.max(0.0).min(1.0);
        let target_length = t_clamped * self.arc_length;
        
        let mut accumulated_length = 0.0;
        for i in 1..self.points.len() {
            let segment_length = (self.points[i] - self.points[i-1]).magnitude();
            if accumulated_length + segment_length >= target_length {
                let local_t = (target_length - accumulated_length) / segment_length;
                return Some(Point3::new(
                    self.points[i-1].x + (self.points[i].x - self.points[i-1].x) * local_t,
                    self.points[i-1].y + (self.points[i].y - self.points[i-1].y) * local_t,
                    self.points[i-1].z + (self.points[i].z - self.points[i-1].z) * local_t,
                ));
            }
            accumulated_length += segment_length;
        }
        
        self.points.last().copied()
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
        match metric.geometry {
            GeometryType::Euclidean => Self::euclidean_geodesic(start, end, steps),
            GeometryType::Spherical => Self::spherical_geodesic(start, end, metric, steps),
            GeometryType::Hyperbolic => Self::hyperbolic_geodesic(start, end, metric, steps),
            GeometryType::Custom => Self::numerical_geodesic(start, end, metric, steps),
        }
    }
    
    /// Straight line in Euclidean space
    fn euclidean_geodesic(start: Point3<f32>, end: Point3<f32>, steps: usize) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Euclidean);
        let direction = (end - start).normalize();
        
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let point = Point3::new(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.z + (end.z - start.z) * t,
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
        let radius = metric.parameters.radius;
        
        // Normalize to sphere surface
        let start_norm = Vector3::new(start.x, start.y, start.z).normalize() * radius;
        let end_norm = Vector3::new(end.x, end.y, end.z).normalize() * radius;
        
        // Compute rotation axis and angle
        let axis = start_norm.cross(end_norm).normalize();
        let angle = (start_norm.dot(end_norm) / (radius * radius)).acos();
        
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let theta = angle * t;
            
            // Slerp on sphere
            let a = (1.0 - t) * angle;
            let b = t * angle;
            
            let point = if angle.abs() > 0.001 {
                let sin_angle = angle.sin();
                let p = (start_norm * a.sin() + end_norm * b.sin()) / sin_angle;
                Point3::from_vec(p)
            } else {
                Point3::from_vec(start_norm + (end_norm - start_norm) * t)
            };
            
            // Tangent is perpendicular to radius
            let tangent = if i < steps {
                let next_t = (i + 1) as f32 / steps as f32;
                let next_theta = angle * next_t;
                let next = if angle.abs() > 0.001 {
                    let sin_angle = angle.sin();
                    (start_norm * (1.0 - next_t) * angle.sin() + end_norm * next_t * angle.sin()) / sin_angle
                } else {
                    start_norm + (end_norm - start_norm) * next_t
                };
                (next - point.to_vec()).normalize()
            } else {
                axis.cross(point.to_vec()).normalize()
            };
            
            path.add_point(point, tangent);
        }
        
        path
    }
    
    /// Geodesic in hyperbolic space (Poincaré disk)
    fn hyperbolic_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Hyperbolic);
        
        // Project to Poincaré disk (z=0 plane)
        let start_2d = Vector3::new(start.x, start.y, 0.0);
        let end_2d = Vector3::new(end.x, end.y, 0.0);
        
        let start_r = (start.x * start.x + start.y * start.y).sqrt();
        let end_r = (end.x * end.x + end.y * end.y).sqrt();
        
        // Check if points are in the disk
        if start_r >= 0.99 || end_r >= 0.99 {
            // Fallback to boundary
            return Self::euclidean_geodesic(start, end, steps);
        }
        
        // Geodesics in Poincaré disk are circular arcs
        // perpendicular to the boundary circle
        
        // Special case: geodesic through origin is a straight line
        if start_r < 0.01 || end_r < 0.01 {
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let point = Point3::new(
                    start.x + (end.x - start.x) * t,
                    start.y + (end.y - start.y) * t,
                    0.0,
                );
                let tangent = (end_2d - start_2d).normalize();
                path.add_point(point, tangent);
            }
        } else {
            // General case: find the circle through both points
            // perpendicular to unit circle
            let midpoint = (start_2d + end_2d) / 2.0;
            let direction = (end_2d - start_2d).normalize();
            let perpendicular = Vector3::new(-direction.y, direction.x, 0.0);
            
            // Find center of the geodesic circle
            let t_center = -midpoint.dot(perpendicular) / perpendicular.dot(perpendicular);
            let center = midpoint + perpendicular * t_center;
            
            // Compute arc
            let radius = (start_2d - center).magnitude();
            let angle_start = (start.y - center.y).atan2(start.x - center.x);
            let angle_end = (end.y - center.y).atan2(end.x - center.x);
            
            let mut angle_diff = angle_end - angle_start;
            if angle_diff > std::f32::consts::PI {
                angle_diff -= 2.0 * std::f32::consts::PI;
            } else if angle_diff < -std::f32::consts::PI {
                angle_diff += 2.0 * std::f32::consts::PI;
            }
            
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let angle = angle_start + angle_diff * t;
                
                let point = Point3::new(
                    center.x + radius * angle.cos(),
                    center.y + radius * angle.sin(),
                    0.0,
                );
                
                // Tangent to the arc
                let tangent = Vector3::new(
                    -radius * angle.sin(),
                    radius * angle.cos(),
                    0.0,
                ).normalize();
                
                path.add_point(point, tangent);
            }
        }
        
        path
    }
    
    /// Numerical geodesic solver using gradient descent
    fn numerical_geodesic(
        start: Point3<f32>,
        end: Point3<f32>,
        metric: &Metric,
        steps: usize,
    ) -> GeodesicPath {
        let mut path = GeodesicPath::new(GeometryType::Custom);
        
        // Initialize with straight line
        let mut points = Vec::new();
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            points.push(Point3::new(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.z + (end.z - start.z) * t,
            ));
        }
        
        // Optimize path to minimize length
        let iterations = 20;
        let learning_rate = 0.1;
        
        for _ in 0..iterations {
            // Keep endpoints fixed
            for i in 1..points.len() - 1 {
                let prev = points[i - 1];
                let curr = points[i];
                let next = points[i + 1];
                
                // Compute gradient of arc length
                let metric_curr = metric.tensor_at(curr);
                let to_prev = prev - curr;
                let to_next = next - curr;
                
                let grad = Vector3::new(
                    metric_curr.norm(to_prev) + metric_curr.norm(to_next),
                    metric_curr.norm(to_prev) + metric_curr.norm(to_next),
                    metric_curr.norm(to_prev) + metric_curr.norm(to_next),
                );
                
                // Update point
                points[i] = Point3::from_vec(curr.to_vec() - grad * learning_rate);
            }
        }
        
        // Build final path
        for i in 0..points.len() {
            let tangent = if i < points.len() - 1 {
                (points[i + 1] - points[i]).normalize()
            } else if i > 0 {
                (points[i] - points[i - 1]).normalize()
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            };
            
            path.add_point(points[i], tangent);
        }
        
        path
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
    /// Cast a ray through curved space
    pub fn cast(
        origin: Point3<f32>,
        direction: Vector3<f32>,
        metric: &Metric,
        max_distance: f32,
        steps: usize,
    ) -> Self {
        let end = origin + direction.normalize() * max_distance;
        let path = Geodesic::compute(origin, end, metric, steps);
        
        Self {
            origin,
            direction: direction.normalize(),
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