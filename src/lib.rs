//! Metatopia Non-Euclidean Game Engine
//! 
//! A game engine that treats space as a manifold with support for
//! curved geometries, portals, and seamless space transitions.

pub mod core;
pub mod graphics;
pub mod input;
pub mod ecs;
pub mod resources;
pub mod math;
pub mod time;
pub mod window;
pub mod manifold;

// Re-export commonly used types
pub use core::{Engine, EngineConfig, GameState};
pub use ecs::{World, Entity, Component};
pub use graphics::{Renderer, RenderContext, Color};
pub use input::{InputManager, InputEvent};
pub use math::{Vec2, Vec3, Mat4, Transform};
pub use resources::{ResourceManager, AssetLoader};
pub use time::{Time, Timer};
pub use window::{Window, WindowBuilder, WindowEvent};
pub use manifold::{
    Manifold, ManifoldPosition, ManifoldOrientation,
    Chart, ChartId, LocalCoordinate,
    Portal, PortalId,
    Geodesic, GeodesicPath,
    Metric, MetricTensor, GeometryType,
};
pub use manifold::geodesic::GeodesicRay;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Engine, EngineConfig, GameState,
        World, Entity, Component,
        Renderer, RenderContext, Color,
        InputManager, InputEvent,
        Vec2, Vec3, Mat4, Transform,
        ResourceManager, AssetLoader,
        Time, Timer,
        Window, WindowBuilder, WindowEvent,
        Manifold, ManifoldPosition, ManifoldOrientation,
        Chart, ChartId, LocalCoordinate,
        Portal, PortalId,
        Geodesic, GeodesicPath,
        Metric, MetricTensor, GeometryType,
        crate::manifold::geodesic::GeodesicRay,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifold_creation() {
        let manifold = Manifold::new();
        assert_eq!(manifold.charts().len(), 1);
    }
    
    #[test]
    fn geodesic_computation() {
        use cgmath::Point3;
        let metric = Metric::from_geometry(GeometryType::Euclidean);
        let path = Geodesic::compute(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            &metric,
            10
        );
        assert_eq!(path.points.len(), 11);
    }
}