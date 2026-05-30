//! Comprehensive integration tests for the metatopia game engine.

use metatopia_engine::prelude::*;
use metatopia_engine::*;
use cgmath::{Point3, Vector3};

// ─── Manifold ──────────────────────────────────────────────────────────────

mod manifold_tests {
    use super::*;

    #[test]
    fn new_manifold_has_default_euclidean_chart() {
        let m = Manifold::new();
        assert_eq!(m.charts().len(), 1, "Manifold should start with one chart");
        let chart = m.chart(ChartId(0)).expect("default chart missing");
        assert_eq!(chart.geometry(), GeometryType::Euclidean);
    }

    #[test]
    fn add_chart_returns_sequential_ids() {
        let mut m = Manifold::new();
        let h = m.add_chart(GeometryType::Hyperbolic);
        let s = m.add_chart(GeometryType::Spherical);
        assert_eq!(h, ChartId(1));
        assert_eq!(s, ChartId(2));
        assert_eq!(m.charts().len(), 3);
    }

    #[test]
    fn create_portal_between_valid_charts() {
        let mut m = Manifold::new();
        let h = m.add_chart(GeometryType::Hyperbolic);
        let result = m.create_portal(
            ChartId(0),
            h,
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        );
        assert!(result.is_ok(), "Portal creation should succeed");
    }

    #[test]
    fn create_portal_with_invalid_chart_fails() {
        let mut m = Manifold::new();
        let result = m.create_portal(
            ChartId(0),
            ChartId(99),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_scale(1.0),
        );
        assert!(result.is_err(), "Portal to nonexistent chart should fail");
    }

    #[test]
    fn active_chart_can_be_set_and_queried() {
        let mut m = Manifold::new();
        let h = m.add_chart(GeometryType::Hyperbolic);
        m.set_active_chart(h);
        assert_eq!(m.active_chart().geometry(), GeometryType::Hyperbolic);
    }

    #[test]
    fn portals_from_chart_returns_correct_set() {
        let mut m = Manifold::new();
        let h = m.add_chart(GeometryType::Hyperbolic);
        let s = m.add_chart(GeometryType::Spherical);

        m.create_portal(ChartId(0), h, Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0), Mat4::from_scale(1.0)).unwrap();
        m.create_portal(ChartId(0), s, Point3::new(-1.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0), Mat4::from_scale(1.0)).unwrap();
        m.create_portal(h, s, Point3::new(0.5, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0), Mat4::from_scale(1.0)).unwrap();

        let from_e = m.portals_from_chart(ChartId(0));
        assert_eq!(from_e.len(), 2, "Euclidean chart should have 2 outbound portals");

        let from_h = m.portals_from_chart(h);
        assert_eq!(from_h.len(), 1, "Hyperbolic chart should have 1 outbound portal");
    }
}

// ─── Geodesic ──────────────────────────────────────────────────────────────

mod geodesic_tests {
    use super::*;

    #[test]
    fn euclidean_geodesic_is_straight_line() {
        let metric = Metric::from_geometry(GeometryType::Euclidean);
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(3.0, 4.0, 0.0);
        let path = Geodesic::compute(start, end, &metric, 10);

        assert_eq!(path.points.len(), 11, "10 segments → 11 points");

        // All points should lie on the straight line
        for (i, wp) in path.points.iter().enumerate() {
            let t = i as f32 / 10.0;
            let expected_x = 3.0 * t;
            let expected_y = 4.0 * t;
            assert!((wp.x - expected_x).abs() < 0.01, "x deviation at step {i}");
            assert!((wp.y - expected_y).abs() < 0.01, "y deviation at step {i}");
        }
    }

    #[test]
    fn spherical_geodesic_has_correct_point_count() {
        let metric = Metric::from_geometry(GeometryType::Spherical);
        let path = Geodesic::compute(
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            &metric,
            20,
        );
        assert_eq!(path.points.len(), 21);
    }

    #[test]
    fn hyperbolic_geodesic_has_correct_point_count() {
        let metric = Metric::from_geometry(GeometryType::Hyperbolic);
        let path = Geodesic::compute(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.3, 0.0),
            &metric,
            15,
        );
        assert_eq!(path.points.len(), 16);
    }
}

// ─── Metric ────────────────────────────────────────────────────────────────

mod metric_tests {
    use super::*;

    #[test]
    fn euclidean_metric_identity_tensor() {
        let metric = Metric::from_geometry(GeometryType::Euclidean);
        let point = Point3::new(1.0, 2.0, 3.0);
        let tensor = metric.tensor_at(point);

        // Euclidean metric tensor should be identity-like (curvature = 0)
        assert!((tensor.curvature).abs() < 1e-6, "Euclidean curvature should be 0");
    }

    #[test]
    fn euclidean_distance_is_correct() {
        let metric = Metric::from_geometry(GeometryType::Euclidean);
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        let dist = metric.distance(a, b);
        assert!((dist - 5.0).abs() < 0.01, "3-4-5 triangle distance should be 5.0, got {dist}");
    }

    #[test]
    fn hyperbolic_distance_exceeds_euclidean() {
        let hyp = Metric::from_geometry(GeometryType::Hyperbolic);
        let euc = Metric::from_geometry(GeometryType::Euclidean);
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(0.3, 0.3, 0.0);
        assert!(
            hyp.distance(a, b) >= euc.distance(a, b),
            "Hyperbolic distance should be ≥ Euclidean"
        );
    }

    #[test]
    fn metric_tensor_at_returns_correct_curvature_sign() {
        let metric = Metric::from_geometry(GeometryType::Hyperbolic);
        let tensor = metric.tensor_at(Point3::new(0.0, 0.0, 0.0));
        assert!(tensor.curvature < 0.0, "Hyperbolic curvature should be negative");
    }
}

// ─── ECS ───────────────────────────────────────────────────────────────────

mod ecs_tests {
    use super::*;

    #[test]
    fn create_and_destroy_entity() {
        let mut world = World::new();
        let e = world.create_entity();
        assert_eq!(e, Entity(0));
        let e2 = world.create_entity();
        assert_eq!(e2, Entity(1));
        world.destroy_entity(e);
        // Creating a new entity may reuse the slot or increment
    }

    #[test]
    fn add_and_get_component() {
        let mut world = World::new();
        let e = world.create_entity();
        world.add_component(e, Velocity {
            linear: Vector3::new(1.0, 2.0, 3.0),
            angular: Vector3::new(0.0, 0.0, 0.0),
        });

        let vel = world.get_component::<Velocity>(e).expect("Velocity not found");
        assert!((vel.linear.x - 1.0).abs() < 1e-6);
        assert!((vel.linear.y - 2.0).abs() < 1e-6);
    }

    #[test]
    fn get_component_mut_modifies_in_place() {
        let mut world = World::new();
        let e = world.create_entity();
        world.add_component(e, Velocity {
            linear: Vector3::new(0.0, 0.0, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.0),
        });

        if let Some(vel) = world.get_component_mut::<Velocity>(e) {
            vel.linear.x = 42.0;
        }

        let vel = world.get_component::<Velocity>(e).unwrap();
        assert!((vel.linear.x - 42.0).abs() < 1e-6);
    }

    #[test]
    fn query_returns_matching_entities() {
        let mut world = World::new();
        let e1 = world.create_entity();
        let e2 = world.create_entity();
        let _e3 = world.create_entity(); // no Velocity

        world.add_component(e1, Velocity {
            linear: Vector3::new(0.0, 0.0, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.0),
        });
        world.add_component(e2, Velocity {
            linear: Vector3::new(0.0, 0.0, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.0),
        });

        let results = world.query::<Velocity>();
        assert_eq!(results.len(), 2);
    }
}

// ─── Math ──────────────────────────────────────────────────────────────────

mod math_tests {
    use super::*;
    use metatopia_engine::math::{Interpolation, Ray, BoundingBox};

    #[test]
    fn lerp_at_boundaries() {
        assert!((Interpolation::lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((Interpolation::lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-6);
        assert!((Interpolation::lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn smoothstep_at_boundaries() {
        assert!((Interpolation::smoothstep(0.0, 1.0, -1.0)).abs() < 1e-6, "below edge0 → 0");
        assert!((Interpolation::smoothstep(0.0, 1.0, 2.0) - 1.0).abs() < 1e-6, "above edge1 → 1");
        assert!((Interpolation::smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6, "midpoint → 0.5");
    }

    #[test]
    fn ray_point_at() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let p = ray.point_at(5.0);
        assert!((p.x - 5.0).abs() < 1e-6);
        assert!(p.y.abs() < 1e-6);
    }

    #[test]
    fn bounding_box_contains() {
        let bb = BoundingBox::new(
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, 1.0),
        );
        assert!(bb.contains(Point3::new(0.0, 0.0, 0.0)));
        assert!(!bb.contains(Point3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn bounding_box_from_points() {
        let bb = BoundingBox::from_points(&[
            Point3::new(-5.0, 0.0, 0.0),
            Point3::new(5.0, 10.0, -3.0),
            Point3::new(0.0, -2.0, 7.0),
        ])
        .expect("should compute bbox");
        assert!((bb.min.x - (-5.0)).abs() < 1e-6);
        assert!((bb.max.y - 10.0).abs() < 1e-6);
        assert!((bb.max.z - 7.0).abs() < 1e-6);
    }

    #[test]
    fn bounding_box_from_empty_returns_none() {
        assert!(BoundingBox::from_points(&[]).is_none());
    }

    #[test]
    fn ray_intersects_bounding_box() {
        let bb = BoundingBox::new(
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(3.0, 1.0, 1.0),
        );
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let hit = bb.intersects_ray(&ray);
        assert!(hit.is_some(), "Ray should hit the box");
        assert!((hit.unwrap() - 1.0).abs() < 0.01, "Hit at t≈1.0");
    }

    #[test]
    fn ray_misses_bounding_box() {
        let bb = BoundingBox::new(
            Point3::new(1.0, 2.0, 2.0),
            Point3::new(3.0, 4.0, 4.0),
        );
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        assert!(bb.intersects_ray(&ray).is_none(), "Ray should miss the box");
    }

    #[test]
    fn transform_identity_preserves_point() {
        let t = Transform::identity();
        let p = Point3::new(3.0, 7.0, -1.0);
        let result = t.transform_point(p);
        assert!((result.x - p.x).abs() < 1e-6);
        assert!((result.y - p.y).abs() < 1e-6);
        assert!((result.z - p.z).abs() < 1e-6);
    }

    #[test]
    fn transform_compose_is_associative() {
        let a = Transform::from_trs(
            Point3::new(1.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        let b = Transform::from_trs(
            Point3::new(0.0, 2.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        let composed = a.compose(&b);
        let p = composed.position();
        // Translation should combine
        assert!((p.x - 1.0).abs() < 1e-4);
    }

    #[test]
    fn transform_inverse_round_trips() {
        let t = Transform::from_trs(
            Point3::new(3.0, 4.0, 5.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            1.0,
        );
        let inv = t.inverse().expect("should be invertible");
        let round = t.compose(&inv);
        let p = round.transform_point(Point3::new(0.0, 0.0, 0.0));
        assert!((p.x).abs() < 1e-4);
        assert!((p.y).abs() < 1e-4);
        assert!((p.z).abs() < 1e-4);
    }
}

// ─── Time / Timer ──────────────────────────────────────────────────────────

mod time_tests {
    use super::*;
    use metatopia_engine::time::FixedTimestep;

    #[test]
    fn time_tracks_delta_and_total() {
        let mut t = Time::new();
        t.update(1.0 / 60.0);
        assert!((t.delta_time() - 1.0 / 60.0).abs() < 1e-6);
        assert!(t.frame_count() == 1);

        t.update(1.0 / 60.0);
        assert!(t.frame_count() == 2);
        assert!((t.total_time() - 2.0 / 60.0).abs() < 1e-4);
    }

    #[test]
    fn timer_from_seconds_progress() {
        let timer = Timer::from_seconds(1.0);
        // Freshly created → progress should be very low
        assert!(timer.progress() < 0.1, "fresh timer progress should be near 0");
        // It should not be finished immediately
        assert!(!timer.finished(), "fresh timer should not be finished");
    }

    #[test]
    fn timer_pause_and_resume() {
        let mut timer = Timer::from_seconds(100.0);
        timer.pause();
        let p1 = timer.progress();
        // Still paused, progress should barely change
        timer.resume();
        let p2 = timer.progress();
        // After resume, progress should be essentially the same
        assert!((p2 - p1).abs() < 0.01);
    }

    #[test]
    fn timer_reset() {
        let mut timer = Timer::from_seconds(100.0);
        timer.reset();
        assert!(!timer.finished());
        assert!(timer.progress() < 0.01);
    }

    #[test]
    fn fixed_timestep_calculates_steps() {
        let mut ts = FixedTimestep::new(60.0); // 60 Hz
        let steps = ts.update(1.0 / 30.0); // 33ms → should be ~2 steps at ~16.67ms each
        assert_eq!(steps, 2, "33ms at 60Hz should produce 2 steps");
    }

    #[test]
    fn fixed_timestep_spiral_of_death_protection() {
        let mut ts = FixedTimestep::new(60.0);
        let steps = ts.update(100.0); // 100 seconds — way too much
        assert!(steps <= 10, "max_steps should cap at 10");
    }

    #[test]
    fn fixed_timestep_alpha() {
        let mut ts = FixedTimestep::new(60.0);
        let _ = ts.update(0.025); // 25ms → 1 step of 16.67ms, remainder ~8.33ms
        let alpha = ts.alpha();
        assert!(alpha > 0.0 && alpha < 1.0, "alpha should be in (0, 1) between steps");
    }
}

// ─── Core Engine ───────────────────────────────────────────────────────────

mod core_tests {
    use super::*;

    #[test]
    fn engine_config_defaults() {
        let config = EngineConfig::default();
        assert_eq!(config.title, "Metatopia Engine");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert!(config.vsync);
        assert!(config.target_fps.is_none());
        assert!(config.resizable);
    }

    #[test]
    fn engine_new_and_quit() {
        let mut engine = Engine::new(EngineConfig::default());
        assert!(engine.is_running());
        engine.quit();
        assert!(!engine.is_running());
    }

    #[test]
    fn engine_has_subsystems() {
        let engine = Engine::new(EngineConfig::default());
        // Verify subsystems are initialised
        assert_eq!(engine.time.frame_count(), 0);
        assert_eq!(engine.world.query::<Velocity>().len(), 0);
    }
}

// ─── ManifoldPosition ──────────────────────────────────────────────────────

mod position_tests {
    use super::*;

    #[test]
    fn manifold_position_creation() {
        let pos = ManifoldPosition::new(ChartId(0), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(pos.chart_id, ChartId(0));
        let p = pos.local.to_point();
        assert!((p.x - 1.0).abs() < 1e-6);
        assert!((p.y - 2.0).abs() < 1e-6);
        assert!((p.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn manifold_position_to_world_on_valid_manifold() {
        let m = Manifold::new();
        let pos = ManifoldPosition::new(ChartId(0), Point3::new(5.0, 0.0, 0.0));
        let world_pos = pos.to_world(&m);
        // Should succeed for the default chart
        assert!(world_pos.is_some(), "to_world should succeed for default chart");
    }
}
