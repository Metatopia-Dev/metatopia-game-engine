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
pub use ecs::{World, Entity, Component, Velocity, Renderable, Transform as EcsTransform, TransformSystem, PortalTransitionSystem};
pub use graphics::{Renderer, RenderContext, Color, Mesh, Vertex, Camera, camera::FPSCameraController};
pub use input::{InputManager, InputEvent, KeyCode, MouseButton};
pub use math::{Vec2, Vec3, Mat4, Transform};
pub use resources::{ResourceManager, AssetLoader};
pub use time::{Time, Timer};
pub use window::{Window, WindowBuilder, WindowEvent};

// Prelude module for easy imports
pub mod prelude {
    pub use crate::core::{Engine, EngineConfig, GameState};
    pub use crate::ecs::{World, Entity, Component, Velocity, Renderable,
                         Transform as EcsTransform, TransformSystem, PortalTransitionSystem};
    pub use crate::graphics::{Renderer, RenderContext, Color, Mesh, Vertex,
                              Camera, camera::FPSCameraController};
    pub use crate::input::{InputManager, InputEvent, KeyCode, MouseButton};
    pub use crate::math::{Vec2, Vec3, Mat4, Transform};
    pub use crate::resources::{ResourceManager, AssetLoader};
    pub use crate::time::{Time, Timer};
    pub use crate::window::{Window, WindowBuilder, WindowEvent};
    pub use crate::manifold::{Manifold, Chart, ChartId, Portal, PortalId,
                              GeometryType, MetricTensor, Geodesic, ManifoldPosition};
    pub use cgmath::{Point3, Vector3, Quaternion};
}
pub use manifold::{
    Manifold, ManifoldPosition, ManifoldOrientation,
    Chart, ChartId, LocalCoordinate,
    Portal, PortalId,
    Geodesic, GeodesicPath,
    Metric, MetricTensor, GeometryType,
};
pub use manifold::geodesic::GeodesicRay;


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