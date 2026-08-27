//! Portal-aware and non-Euclidean geodesic raycasting system
//!
//! Enables raycasting through curved spaces (Hyperbolic, Spherical, Euclidean)
//! and seamless ray traversal across portal connections between manifold charts.

use cgmath::{Point3, Vector3, InnerSpace, EuclideanSpace};
use super::{Manifold, ChartId, PortalId, GeometryType, Metric};
use crate::collision::{CollisionWorld, Ray};

/// A ray defined within a specific manifold chart
#[derive(Debug, Clone)]
pub struct ManifoldRay {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub chart: ChartId,
    pub max_distance: f32,
}

impl ManifoldRay {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>, chart: ChartId, max_distance: f32) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            chart,
            max_distance,
        }
    }
}

/// A linear or curved segment of a ray traced within a single chart
#[derive(Debug, Clone)]
pub struct RaySegment {
    pub chart: ChartId,
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub direction: Vector3<f32>,
    pub length: f32,
}

/// Result of a manifold raycast hit
#[derive(Debug, Clone)]
pub struct ManifoldRayHit {
    /// World position of the hit in the final chart's local coordinates
    pub point: Point3<f32>,
    /// Chart in which the hit occurred
    pub chart: ChartId,
    /// Surface normal at hit point
    pub normal: Vector3<f32>,
    /// Total ray distance traveled across all charts
    pub total_distance: f32,
    /// Number of portals traversed
    pub portal_traversals: usize,
    /// Traced path segments
    pub segments: Vec<RaySegment>,
    /// Optional tag from hit collider
    pub tag: Option<String>,
}

/// Full ray trace history through a manifold
#[derive(Debug, Clone)]
pub struct ManifoldRayTrace {
    pub segments: Vec<RaySegment>,
    pub total_length: f32,
    pub portal_jumps: usize,
    pub hit: Option<ManifoldRayHit>,
}

/// Manifold raycaster engine
pub struct ManifoldRaycaster;

impl ManifoldRaycaster {
    /// Cast a straight/portal ray through the manifold testing against portals and optional collision worlds
    pub fn cast_ray(
        manifold: &Manifold,
        ray: &ManifoldRay,
        max_portal_jumps: usize,
        collision_worlds: Option<&std::collections::HashMap<ChartId, &CollisionWorld>>,
    ) -> Option<ManifoldRayHit> {
        let mut current_origin = ray.origin;
        let mut current_dir = ray.direction;
        let mut current_chart = ray.chart;
        let mut remaining_distance = ray.max_distance;
        let mut total_traveled = 0.0;
        let mut portal_traversals = 0;
        let mut segments = Vec::new();

        while remaining_distance > 0.001 && portal_traversals <= max_portal_jumps {
            // 1. Check collider hits in current chart if provided
            let mut nearest_collider_hit: Option<(f32, Vector3<f32>, String)> = None;
            if let Some(worlds) = collision_worlds {
                if let Some(world) = worlds.get(&current_chart) {
                    let col_ray = Ray::new(
                        [current_origin.x, current_origin.y, current_origin.z],
                        [current_dir.x, current_dir.y, current_dir.z],
                    );
                    if let Some(hit) = world.raycast(&col_ray, remaining_distance) {
                        if let Some(rh) = hit.ray_hit {
                            nearest_collider_hit = Some((rh.t, Vector3::new(rh.normal[0], rh.normal[1], rh.normal[2]), hit.tag.to_string()));
                        }
                    }
                }
            }

            // 2. Find nearest portal in current chart
            let mut nearest_portal_hit: Option<(f32, Point3<f32>, PortalId)> = None;
            for portal in manifold.portals() {
                if portal.source_chart() == current_chart && portal.is_active() {
                    if let Some(intersection) = portal.ray_intersection(current_origin, current_dir) {
                        let dist = (intersection - current_origin).magnitude();
                        if dist > 0.0001 && dist <= remaining_distance {
                            if nearest_portal_hit.as_ref().map_or(true, |(d, _, _)| dist < *d) {
                                nearest_portal_hit = Some((dist, intersection, portal.id()));
                            }
                        }
                    }
                }
            }

            // 3. Resolve nearest event (collider vs portal)
            match (nearest_collider_hit, nearest_portal_hit) {
                (Some((col_dist, col_norm, col_tag)), Some((port_dist, _, _))) if col_dist < port_dist => {
                    // Hit collider before portal
                    let hit_point = current_origin + current_dir * col_dist;
                    segments.push(RaySegment {
                        chart: current_chart,
                        start: current_origin,
                        end: hit_point,
                        direction: current_dir,
                        length: col_dist,
                    });
                    return Some(ManifoldRayHit {
                        point: hit_point,
                        chart: current_chart,
                        normal: col_norm,
                        total_distance: total_traveled + col_dist,
                        portal_traversals,
                        segments,
                        tag: Some(col_tag),
                    });
                }
                (Some((col_dist, col_norm, col_tag)), None) => {
                    // Hit collider (no portal in way)
                    let hit_point = current_origin + current_dir * col_dist;
                    segments.push(RaySegment {
                        chart: current_chart,
                        start: current_origin,
                        end: hit_point,
                        direction: current_dir,
                        length: col_dist,
                    });
                    return Some(ManifoldRayHit {
                        point: hit_point,
                        chart: current_chart,
                        normal: col_norm,
                        total_distance: total_traveled + col_dist,
                        portal_traversals,
                        segments,
                        tag: Some(col_tag),
                    });
                }
                (_, Some((port_dist, port_intersection, portal_id))) => {
                    // Hit portal first -> traverse through portal
                    segments.push(RaySegment {
                        chart: current_chart,
                        start: current_origin,
                        end: port_intersection,
                        direction: current_dir,
                        length: port_dist,
                    });
                    total_traveled += port_dist;
                    remaining_distance -= port_dist;
                    portal_traversals += 1;

                    let portal = manifold.get_portal(portal_id).unwrap();
                    current_origin = portal.transform_point(port_intersection);
                    current_dir = portal.transform_vector(current_dir).normalize();
                    current_chart = portal.target_chart();
                }
                (None, None) => {
                    // Ray travels full remaining distance into empty space
                    let end_point = current_origin + current_dir * remaining_distance;
                    segments.push(RaySegment {
                        chart: current_chart,
                        start: current_origin,
                        end: end_point,
                        direction: current_dir,
                        length: remaining_distance,
                    });
                    break;
                }
            }
        }

        None
    }

    /// Cast a geodesic ray that bends according to the Riemannian metric tensor in curved manifolds
    pub fn cast_geodesic_ray(
        manifold: &Manifold,
        start: Point3<f32>,
        direction: Vector3<f32>,
        chart: ChartId,
        max_distance: f32,
        step_size: f32,
        max_portal_jumps: usize,
    ) -> ManifoldRayTrace {
        let mut segments = Vec::new();
        let mut current_pos = start;
        let mut current_dir = direction.normalize();
        let mut current_chart = chart;
        let mut total_length = 0.0;
        let mut portal_jumps = 0;

        let mut remaining = max_distance;
        let step = step_size.max(0.01);

        while remaining > 0.0 && portal_jumps <= max_portal_jumps {
            let chart_geom = manifold.get_chart(current_chart)
                .map(|c| c.geometry())
                .unwrap_or(GeometryType::Euclidean);

            // Compute curvature deflection vector
            let metric = Metric::from_geometry(chart_geom);
            let deflection = match chart_geom {
                GeometryType::Euclidean => Vector3::new(0.0, 0.0, 0.0),
                GeometryType::Hyperbolic => {
                    // Hyperbolic geodesic acceleration points outward from origin
                    let r = current_pos.to_vec().magnitude().max(0.1);
                    (current_pos.to_vec() / (1.0 - (r * 0.1).min(0.9))) * 0.05
                }
                GeometryType::Spherical => {
                    // Spherical geodesic bends toward origin/great circle
                    -current_pos.to_vec() * 0.08
                }
                GeometryType::Custom => {
                    let tensor = metric.tensor_at(current_pos);
                    Vector3::new(tensor.curvature * 0.02, 0.0, tensor.curvature * 0.02)
                }
            };

            let next_dir = (current_dir + deflection * step).normalize();
            let next_pos = current_pos + next_dir * step.min(remaining);
            let seg_len = (next_pos - current_pos).magnitude();

            // Check portal intersection along this step
            let mut hit_portal: Option<(Point3<f32>, PortalId, f32)> = None;
            for portal in manifold.portals() {
                if portal.source_chart() == current_chart && portal.is_active() {
                    if let Some(inter) = portal.ray_intersection(current_pos, next_dir) {
                        let d: f32 = (inter - current_pos).magnitude();
                        if d <= seg_len {
                            hit_portal = Some((inter, portal.id(), d));
                            break;
                        }
                    }
                }
            }

            if let Some((inter, p_id, d)) = hit_portal {
                segments.push(RaySegment {
                    chart: current_chart,
                    start: current_pos,
                    end: inter,
                    direction: current_dir,
                    length: d,
                });
                total_length += d;
                remaining -= d;
                portal_jumps += 1;

                let portal = manifold.get_portal(p_id).unwrap();
                current_pos = portal.transform_point(inter);
                current_dir = portal.transform_vector(next_dir).normalize();
                current_chart = portal.target_chart();
            } else {
                segments.push(RaySegment {
                    chart: current_chart,
                    start: current_pos,
                    end: next_pos,
                    direction: current_dir,
                    length: seg_len,
                });
                total_length += seg_len;
                remaining -= seg_len;
                current_pos = next_pos;
                current_dir = next_dir;
            }
        }

        ManifoldRayTrace {
            segments,
            total_length,
            portal_jumps,
            hit: None,
        }
    }

    /// Test line-of-sight visibility between two points across charts through portals
    pub fn is_visible(
        manifold: &Manifold,
        from: Point3<f32>,
        from_chart: ChartId,
        to: Point3<f32>,
        to_chart: ChartId,
        collision_worlds: Option<&std::collections::HashMap<ChartId, &CollisionWorld>>,
    ) -> bool {
        if from_chart == to_chart {
            let dir = to - from;
            let dist = dir.magnitude();
            if dist < 0.0001 { return true; }
            let ray = ManifoldRay::new(from, dir.normalize(), from_chart, dist);
            if let Some(hit) = Self::cast_ray(manifold, &ray, 0, collision_worlds) {
                return hit.total_distance >= dist - 0.01;
            }
            return true;
        }

        // Multi-chart line of sight through portals
        let dir = (to - from).normalize();
        let ray = ManifoldRay::new(from, dir, from_chart, 100.0);
        if let Some(hit) = Self::cast_ray(manifold, &ray, 8, collision_worlds) {
            if hit.chart == to_chart {
                return (hit.point - to).magnitude() < 0.2;
            }
        }
        false
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, SquareMatrix};

    #[test]
    fn test_euclidean_straight_ray() {
        let manifold = Manifold::new();
        let ray = ManifoldRay::new(
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            ChartId(0),
            20.0,
        );
        let hit = ManifoldRaycaster::cast_ray(&manifold, &ray, 3, None);
        assert!(hit.is_none());
    }

    #[test]
    fn test_ray_portal_traversal() {
        let mut manifold = Manifold::new();
        let chart1 = manifold.add_chart(GeometryType::Euclidean);

        // Create portal at z = -5.0 in chart 0 -> target at z = 10.0 in chart 1
        let _ = manifold.create_portal(
            ChartId(0),
            chart1,
            Point3::new(0.0, 1.0, -5.0),
            Point3::new(0.0, 1.0, 10.0),
            Matrix4::identity(),
        );

        let ray = ManifoldRay::new(
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            ChartId(0),
            30.0,
        );

        let trace = ManifoldRaycaster::cast_geodesic_ray(
            &manifold,
            ray.origin,
            ray.direction,
            ray.chart,
            20.0,
            0.5,
            4,
        );

        assert!(trace.portal_jumps >= 1);
    }
}
