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
pub mod scoring;
pub mod collision;
pub mod audio;
pub mod quickstart;
pub mod particles;
pub mod physics;
pub mod geometry;
pub mod animation;
pub mod scene;
pub mod ai;
pub mod lighting;
pub mod ui;
pub mod decals;
pub mod net;
pub mod scripting;
pub mod vr;

// Re-export commonly used types
pub use core::{Engine, EngineConfig, GameState};
pub use ecs::{World, Entity, Component, Velocity, Renderable, Transform as EcsTransform, TransformSystem, PortalTransitionSystem};
pub use graphics::{Renderer, RenderContext, Color, Mesh, Vertex, Camera, camera::FPSCameraController};
pub use input::{InputManager, InputEvent, KeyCode, MouseButton, GamepadButton, GamepadAxis};
pub use math::{Vec2, Vec3, Mat4, Transform};
pub use resources::{ResourceManager, AssetLoader};
pub use time::{Time, Timer};
pub use window::{Window, WindowBuilder, WindowEvent};
pub use scoring::{ScoreTracker, ScoreEvent, HudData};
pub use collision::{AABB, SphereCollider, Ray, RayHit, Collider, CollisionWorld};
pub use audio::{AudioEngine, AudioListener, SpatialSoundSource, AttenuationModel, SpatialAudioEngine};
pub use particles::{Particle, ParticleEmitter, EmitterShape, ParticleSystem};
pub use physics::{RigidBody, BodyType, PhysicsCollider, GravityField, PhysicsWorld};
pub use geometry::ProceduralMesh;
pub use animation::{EaseFunction, Tween, Keyframe, KeyframeTrack, LoopMode, Interpolate};
pub use scene::{SceneGraph, SceneNode, NodeId};
pub use ai::{BehaviorNode, SequenceNode, SelectorNode, InverterNode, ActionNode, ConditionNode, NodeStatus, Blackboard, BlackboardValue, NavGraph, NavNode};
pub use lighting::{LightManager, PointLight, SpotLight, DirectionalLight, GpuLight};
pub use ui::{UiCanvas, Rect, Anchor, DamagePopup, UiQuad};
pub use decals::{Decal, DecalSystem, DecalType};
pub use net::{NetServer, NetClient, ClientMessage, ServerMessage, EntityState, ChannelType, Snapshot, SnapshotBuffer, ClientPrediction};
pub use scripting::{ScriptEngine, ScriptEntityState};
pub use vr::{StereoCameraRig, StereoMode, Eye, VrHeadPose, VrController, VrHand, VrTrackingContext};

// Prelude module for easy imports
pub mod prelude {
    pub use crate::core::{Engine, EngineConfig, GameState};
    pub use crate::ecs::{World, Entity, Component, Velocity, Renderable,
                         Transform as EcsTransform, TransformSystem, PortalTransitionSystem};
    pub use crate::graphics::{Renderer, RenderContext, Color, Mesh, Vertex,
                              Camera, camera::FPSCameraController};
    pub use crate::input::{InputManager, InputEvent, KeyCode, MouseButton, GamepadButton, GamepadAxis};
    pub use crate::math::{Vec2, Vec3, Mat4, Transform};
    pub use crate::resources::{ResourceManager, AssetLoader};
    pub use crate::time::{Time, Timer};
    pub use crate::window::{Window, WindowBuilder, WindowEvent};
    pub use crate::scoring::{ScoreTracker, ScoreEvent, HudData};
    pub use crate::collision::{AABB, SphereCollider, Ray, RayHit, Collider, CollisionWorld};
    pub use crate::audio::{AudioEngine, AudioListener, SpatialSoundSource, AttenuationModel, SpatialAudioEngine};
    pub use crate::manifold::{Manifold, Chart, ChartId, Portal, PortalId,
                              GeometryType, MetricTensor, Geodesic, ManifoldPosition,
                              ManifoldRay, ManifoldRayHit, ManifoldRayTrace, ManifoldRaycaster};
    pub use crate::particles::{Particle, ParticleEmitter, EmitterShape, ParticleSystem};
    pub use crate::physics::{RigidBody, BodyType, PhysicsCollider, GravityField, PhysicsWorld};
    pub use crate::geometry::ProceduralMesh;
    pub use crate::animation::{EaseFunction, Tween, Keyframe, KeyframeTrack, LoopMode, Interpolate};
    pub use crate::scene::{SceneGraph, SceneNode, NodeId};
    pub use crate::ai::{BehaviorNode, SequenceNode, SelectorNode, InverterNode, ActionNode, ConditionNode, NodeStatus, Blackboard, BlackboardValue, NavGraph, NavNode};
    pub use crate::lighting::{LightManager, PointLight, SpotLight, DirectionalLight, GpuLight};
    pub use crate::ui::{UiCanvas, Rect, Anchor, DamagePopup, UiQuad};
    pub use crate::decals::{Decal, DecalSystem, DecalType};
    pub use crate::net::{NetServer, NetClient, ClientMessage, ServerMessage, EntityState, ChannelType, Snapshot, SnapshotBuffer, ClientPrediction};
    pub use crate::scripting::{ScriptEngine, ScriptEntityState};
    pub use crate::vr::{StereoCameraRig, StereoMode, Eye, VrHeadPose, VrController, VrHand, VrTrackingContext};
    pub use cgmath::{Point3, Vector3, Quaternion};
}
pub use manifold::{
    Manifold, ManifoldPosition, ManifoldOrientation,
    Chart, ChartId, LocalCoordinate,
    Portal, PortalId,
    Geodesic, GeodesicPath,
    Metric, MetricTensor, GeometryType,
    ManifoldRay, ManifoldRayHit, ManifoldRayTrace, ManifoldRaycaster,
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