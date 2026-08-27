//! Asset Pipeline & Scene Persistence Subsystem
//!
//! Provides JSON/binary scene loading and 3D mesh format parsing.

pub mod scene_io;
pub mod gltf_loader;

pub use scene_io::{SceneDocument, SceneEntityData, SceneEnvironmentData};
pub use gltf_loader::{LoadedMesh, ModelLoader};
