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
        let max_iters = ((max_distance / step) as usize + max_portal_jumps * 4 + 32).min(4096);
        let mut iter_count = 0usize;

        // Cache chart geometry and active portals; refresh only when chart changes on a portal jump
        let mut chart_geom = manifold
            .get_chart(current_chart)
            .map(|c| c.geometry())
            .unwrap_or(GeometryType::Euclidean);
        let mut active_portals: Vec<_> = manifold
            .portals_from_chart(current_chart)
            .into_iter()
            .filter(|p| p.is_active())
            .collect();

        while remaining > 1e-4 && portal_jumps <= max_portal_jumps && iter_count < max_iters {
            iter_count += 1;
            let actual_step = step.min(remaining);
            let half_step = actual_step * 0.5;
            let coord_scale = 0.08f32;

            // Compute midpoint RK2 geodesic tangent & position update
            let (mid_dir, next_dir) = match chart_geom {
                GeometryType::Euclidean => (current_dir, current_dir),
                GeometryType::Hyperbolic | GeometryType::Spherical => {
                    let k = if chart_geom == GeometryType::Hyperbolic { -1.0 } else { 1.0 };
                    let scaled_pos = Point3::from_vec(current_pos.to_vec() * coord_scale);
                    let cs1 = super::metric::ChristoffelSymbols::from_conformal_point(scaled_pos, k);
                    let a1 = cs1.geodesic_acceleration(scaled_pos.to_vec(), current_dir) * coord_scale;
                    let m_dir = {
                        let v = current_dir + a1 * half_step;
                        if v.magnitude2() > 1e-8 { v.normalize() } else { current_dir }
                    };
                    let mid_pos = Point3::from_vec((current_pos + current_dir * half_step).to_vec() * coord_scale);
                    let cs2 = super::metric::ChristoffelSymbols::from_conformal_point(mid_pos, k);
                    let a2 = cs2.geodesic_acceleration(mid_pos.to_vec(), m_dir) * coord_scale;
                    let n_dir = {
                        let v = current_dir + a2 * actual_step;
                        if v.magnitude2() > 1e-8 { v.normalize() } else { m_dir }
                    };
                    (m_dir, n_dir)
                }
                GeometryType::Custom => {
                    let metric = Metric::from_geometry(chart_geom);
                    let tensor = metric.tensor_at(current_pos);
                    let cs = super::metric::ChristoffelSymbols::from_conformal_point(current_pos, tensor.curvature);
                    let a = cs.geodesic_acceleration(current_pos.to_vec(), current_dir) * coord_scale;
                    let n_dir = {
                        let v = current_dir + a * actual_step;
                        if v.magnitude2() > 1e-8 { v.normalize() } else { current_dir }
                    };
                    (n_dir, n_dir)
                }
            };

            let next_pos = current_pos + mid_dir * actual_step;

            // Check portal intersection against cached active portals for current chart
            let mut hit_portal: Option<(Point3<f32>, PortalId, f32)> = None;
            for portal in &active_portals {
                if let Some(inter) = portal.ray_intersection(current_pos, mid_dir) {
                    let d: f32 = (inter - current_pos).magnitude();
                    if d > 1e-4 && d <= actual_step {
                        if hit_portal.as_ref().map_or(true, |(_, _, best_d)| d < *best_d) {
                            hit_portal = Some((inter, portal.id(), d));
                        }
                    }
                }
            }

            if let Some((inter, p_id, d)) = hit_portal {
                segments.push(RaySegment {
                    chart: current_chart,
                    start: current_pos,
                    end: inter,
                    direction: mid_dir,
                    length: d,
                });
                total_length += d;
                remaining -= d.max(1e-4);
                portal_jumps += 1;

                if let Some(portal) = manifold.get_portal(p_id) {
                    current_dir = portal.transform_vector(next_dir).normalize();
                    let nudge = (step * 0.02).clamp(0.002, 0.02);
                    current_pos = portal.transform_point(inter) + current_dir * nudge;
                    current_chart = portal.target_chart();
                    chart_geom = manifold
                        .get_chart(current_chart)
                        .map(|c| c.geometry())
                        .unwrap_or(GeometryType::Euclidean);
                    active_portals = manifold
                        .portals_from_chart(current_chart)
                        .into_iter()
                        .filter(|p| p.is_active())
                        .collect();
                } else {
                    break;
                }
            } else {
                segments.push(RaySegment {
                    chart: current_chart,
                    start: current_pos,
                    end: next_pos,
                    direction: next_dir,
                    length: actual_step,
                });
                total_length += actual_step;
                remaining -= actual_step;
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

    /// Benchmark CPU geodesic & portal ray traversal throughput across a manifold
    pub fn benchmark_portal_transitions(
        manifold: &Manifold,
        start_chart: ChartId,
        ray_count: usize,
        max_portal_jumps: usize,
        step_size: f32,
    ) -> PortalBenchmarkMetrics {
        let t0 = std::time::Instant::now();
        let mut total_portal_jumps = 0usize;
        let mut total_segments = 0usize;
        let mut total_distance = 0.0f32;

        let n = ray_count.max(1);
        for i in 0..n {
            let angle = (i as f32 / n as f32) * 0.4 - 0.2;
            let pitch = ((i % 16) as f32 / 16.0) * 0.2 - 0.1;
            let dir = Vector3::new(angle.sin(), pitch, -angle.cos()).normalize();
            let trace = Self::cast_geodesic_ray(
                manifold,
                Point3::new(0.0, 1.0, 2.0),
                dir,
                start_chart,
                40.0,
                step_size,
                max_portal_jumps,
            );
            total_portal_jumps += trace.portal_jumps;
            total_segments += trace.segments.len();
            total_distance += trace.total_length;
        }

        let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-9);
        let rays_per_sec = (n as f64) / elapsed_secs;
        let segments_per_sec = (total_segments as f64) / elapsed_secs;

        PortalBenchmarkMetrics {
            ray_count: n,
            max_portal_jumps,
            total_portal_jumps,
            total_segments,
            avg_distance: total_distance / (n as f32),
            elapsed_ms: elapsed_secs * 1000.0,
            rays_per_sec,
            segments_per_sec,
        }
    }
}

/// Telemetry summary for a portal + geodesic raycast benchmark run
#[derive(Debug, Clone, Copy)]
pub struct PortalBenchmarkMetrics {
    pub ray_count: usize,
    pub max_portal_jumps: usize,
    pub total_portal_jumps: usize,
    pub total_segments: usize,
    pub avg_distance: f32,
    pub elapsed_ms: f64,
    pub rays_per_sec: f64,
    pub segments_per_sec: f64,
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

    #[test]
    fn test_recursive_portal_loop_and_benchmark() {
        let mut manifold = Manifold::new();
        let chart_hyp = manifold.add_chart(GeometryType::Hyperbolic);
        let chart_sph = manifold.add_chart(GeometryType::Spherical);

        // 3-chart cycle: Chart 0 (Euclidean) -> Chart 1 (Hyperbolic) -> Chart 2 (Spherical) -> Chart 0
        let _ = manifold.create_portal(
            ChartId(0),
            chart_hyp,
            Point3::new(0.0, 1.0, -3.0),
            Point3::new(0.0, 1.0, 3.0),
            Matrix4::identity(),
        );
        let _ = manifold.create_portal(
            chart_hyp,
            chart_sph,
            Point3::new(0.0, 1.0, -3.0),
            Point3::new(0.0, 1.0, 3.0),
            Matrix4::identity(),
        );
        let _ = manifold.create_portal(
            chart_sph,
            ChartId(0),
            Point3::new(0.0, 1.0, -3.0),
            Point3::new(0.0, 1.0, 3.0),
            Matrix4::identity(),
        );

        let metrics = ManifoldRaycaster::benchmark_portal_transitions(
            &manifold,
            ChartId(0),
            64,
            6,
            0.25,
        );
        assert_eq!(metrics.ray_count, 64);
        assert!(metrics.total_portal_jumps > 0);
        assert!(metrics.rays_per_sec > 100.0);
    }
}
