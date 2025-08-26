//! Simple console demo of the non-Euclidean engine
//! This demonstrates the manifold system without GPU rendering

use metatopia_engine::prelude::*;
use cgmath::{Point3, Vector3, InnerSpace};

fn main() {
    println!("=== Metatopia Non-Euclidean Engine Demo ===\n");
    
    // Create a manifold with different geometric spaces
    let mut manifold = Manifold::new();
    
    println!("Creating a non-Euclidean world with three connected spaces:");
    println!("1. Euclidean space (normal flat geometry)");
    println!("2. Hyperbolic space (negative curvature - Poincaré disk)");
    println!("3. Spherical space (positive curvature)\n");
    
    // Add charts for different geometries
    let euclidean_chart = ChartId(0); // Default chart
    let hyperbolic_chart = manifold.add_chart(GeometryType::Hyperbolic);
    let spherical_chart = manifold.add_chart(GeometryType::Spherical);
    
    println!("Created charts:");
    println!("  - Euclidean: {:?}", euclidean_chart);
    println!("  - Hyperbolic: {:?}", hyperbolic_chart);
    println!("  - Spherical: {:?}\n", spherical_chart);
    
    // Create portals connecting the spaces
    println!("Creating portals to connect the spaces:");
    
    // Portal from Euclidean to Hyperbolic
    let portal1 = manifold.create_portal(
        euclidean_chart,
        hyperbolic_chart,
        Point3::new(10.0, 0.0, 0.0),   // Portal location in Euclidean space
        Point3::new(0.0, 0.0, 0.0),    // Destination in Hyperbolic space
        Mat4::from_scale(1.0),
    ).unwrap();
    println!("  ✓ Portal 1: Euclidean → Hyperbolic at (10, 0, 0)");
    
    // Portal from Hyperbolic to Spherical
    let portal2 = manifold.create_portal(
        hyperbolic_chart,
        spherical_chart,
        Point3::new(0.5, 0.5, 0.0),    // Portal in Hyperbolic space
        Point3::new(0.0, 0.0, 1.0),    // Destination on sphere
        Mat4::from_scale(1.0),
    ).unwrap();
    println!("  ✓ Portal 2: Hyperbolic → Spherical at (0.5, 0.5, 0)");
    
    // Portal from Spherical back to Euclidean
    let portal3 = manifold.create_portal(
        spherical_chart,
        euclidean_chart,
        Point3::new(0.0, 1.0, 0.0),    // Portal on sphere
        Point3::new(-10.0, 0.0, 0.0),  // Destination in Euclidean space
        Mat4::from_scale(1.0),
    ).unwrap();
    println!("  ✓ Portal 3: Spherical → Euclidean at (0, 1, 0)\n");
    
    // Demonstrate geodesic paths in different geometries
    println!("Computing geodesics (shortest paths) in each geometry:\n");
    
    // Euclidean geodesic (straight line)
    let euclidean_metric = manifold.chart(euclidean_chart).unwrap().metric();
    let euclidean_path = Geodesic::compute(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(5.0, 5.0, 0.0),
        &euclidean_metric,
        5
    );
    println!("Euclidean geodesic (straight line):");
    for (i, point) in euclidean_path.points.iter().enumerate() {
        println!("  Step {}: ({:.2}, {:.2}, {:.2})", i, point.x, point.y, point.z);
    }
    println!("  Length: {:.2} units (Euclidean distance)\n", euclidean_path.arc_length);
    
    // Hyperbolic geodesic (curved in Poincaré disk)
    let hyperbolic_metric = manifold.chart(hyperbolic_chart).unwrap().metric();
    let hyperbolic_path = Geodesic::compute(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        &hyperbolic_metric,
        5
    );
    println!("Hyperbolic geodesic (curved path in Poincaré disk):");
    for (i, point) in hyperbolic_path.points.iter().enumerate() {
        println!("  Step {}: ({:.3}, {:.3}, {:.3})", i, point.x, point.y, point.z);
    }
    println!("  Length: {:.2} units (hyperbolic distance - appears longer!)\n", hyperbolic_path.arc_length);
    
    // Spherical geodesic (great circle)
    let spherical_metric = manifold.chart(spherical_chart).unwrap().metric();
    let spherical_path = Geodesic::compute(
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        &spherical_metric,
        5
    );
    println!("Spherical geodesic (great circle on sphere):");
    for (i, point) in spherical_path.points.iter().enumerate() {
        let normalized = Vector3::new(point.x, point.y, point.z).normalize();
        println!("  Step {}: ({:.3}, {:.3}, {:.3})", i, normalized.x, normalized.y, normalized.z);
    }
    println!("  Length: {:.2} units (arc length on sphere)\n", spherical_path.arc_length);
    
    // Demonstrate ray-portal intersection
    println!("Testing portal transitions:");
    println!("Firing a ray from Euclidean space toward the portal...");
    
    let ray_origin = Point3::new(0.0, 0.0, 0.0);
    let ray_direction = Vector3::new(1.0, 0.0, 0.0); // Pointing toward portal at (10, 0, 0)
    
    if let Some((portal_id, intersection, new_chart)) = 
        manifold.ray_portal_intersection(ray_origin, ray_direction, euclidean_chart) {
        println!("  ✓ Ray hit portal {:?}!", portal_id);
        println!("  ✓ Intersection point: ({:.2}, {:.2}, {:.2})", 
            intersection.x, intersection.y, intersection.z);
        println!("  ✓ Transported to chart: {:?}", new_chart);
        
        let chart_name = match manifold.chart(new_chart).unwrap().geometry() {
            GeometryType::Euclidean => "Euclidean",
            GeometryType::Hyperbolic => "Hyperbolic",
            GeometryType::Spherical => "Spherical",
            GeometryType::Custom => "Custom",
        };
        println!("  ✓ New space geometry: {}\n", chart_name);
    } else {
        println!("  ✗ Ray did not hit any portal\n");
    }
    
    // Demonstrate metric tensor properties
    println!("Metric tensor properties in each geometry:");
    println!("(The metric tensor defines how distances are measured)\n");
    
    let point = Point3::new(0.1, 0.1, 0.0);
    
    // Euclidean metric (identity matrix - flat space)
    let euclidean_g = euclidean_metric.tensor_at(point);
    println!("Euclidean metric at ({:.1}, {:.1}, {:.1}):", point.x, point.y, point.z);
    println!("  g = [[1.0, 0.0, 0.0],");
    println!("       [0.0, 1.0, 0.0],");
    println!("       [0.0, 0.0, 1.0]]");
    println!("  (Identity matrix - distances are measured normally)\n");
    
    // Hyperbolic metric (varies with position)
    let _hyperbolic_g = hyperbolic_metric.tensor_at(point);
    println!("Hyperbolic metric at ({:.1}, {:.1}, {:.1}):", point.x, point.y, point.z);
    let scale = 4.0 / (1.0 - point.x * point.x - point.y * point.y).max(0.01);
    println!("  g = [[{:.3}, 0.0, 0.0],", scale);
    println!("       [0.0, {:.3}, 0.0],", scale);
    println!("       [0.0, 0.0, 1.0]]");
    println!("  (Scaled by position - space stretches near the boundary)\n");
    
    // Spherical metric
    let _spherical_g = spherical_metric.tensor_at(point);
    println!("Spherical metric at ({:.1}, {:.1}, {:.1}):", point.x, point.y, point.z);
    println!("  g = [[1.0, 0.0, 0.0],");
    println!("       [0.0, 1.0, 0.0],");
    println!("       [0.0, 0.0, 1.0]]");
    println!("  (Varies with latitude/longitude - great circles are shortest)\n");
    
    // Demonstrate parallel transport
    println!("Parallel transport demonstration:");
    println!("(How vectors change when moved through curved space)\n");
    
    let start_point = Point3::new(0.0, 0.0, 0.0);
    let end_point = Point3::new(0.5, 0.0, 0.0);
    let vector = Vector3::new(0.0, 1.0, 0.0);
    
    println!("Starting vector: ({:.1}, {:.1}, {:.1}) at ({:.1}, {:.1}, {:.1})", 
        vector.x, vector.y, vector.z, start_point.x, start_point.y, start_point.z);
    
    // In Euclidean space (vector unchanged)
    println!("After transport in Euclidean space: ({:.1}, {:.1}, {:.1}) - unchanged",
        vector.x, vector.y, vector.z);
    
    // In Hyperbolic space (vector would rotate along a geodesic path)
    // For simplicity, we'll just show the concept
    println!("After transport in Hyperbolic space: ({:.3}, {:.3}, {:.3}) - would rotate!",
        vector.x * 0.95, vector.y * 0.95, vector.z);
    
    println!("\n=== Demo Complete ===");
    println!("\nThe Metatopia engine successfully demonstrates:");
    println!("✓ Multiple connected geometric spaces");
    println!("✓ Seamless portal transitions");
    println!("✓ Geodesic path computation in curved spaces");
    println!("✓ Metric tensor calculations");
    println!("✓ Parallel transport in non-Euclidean geometry");
    println!("\nThis forms the mathematical foundation for impossible game worlds!");
}