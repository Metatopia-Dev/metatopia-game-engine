//! Scene Persistence & IO Serialization System
//!
//! Provides JSON and binary scene serialization and loading for Metatopia scenes.

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};

/// Environment settings stored in a scene
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneEnvironmentData {
    pub ambient_light: [f32; 3],
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub fog_density: f32,
    pub fog_color: [f32; 3],
}

impl Default for SceneEnvironmentData {
    fn default() -> Self {
        Self {
            ambient_light: [0.15, 0.18, 0.22],
            sun_direction: [-0.5, -1.0, -0.6],
            sun_color: [1.0, 0.98, 0.92],
            fog_density: 0.005,
            fog_color: [0.08, 0.10, 0.15],
        }
    }
}

/// Serialized entity node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneEntityData {
    pub id: u32,
    pub name: String,
    pub mesh_type: String, // "Cube", "Sphere", "Cylinder", "Torus", "Capsule", "Custom"
    pub pos: [f32; 3],
    pub rot: [f32; 3], // Yaw, Pitch, Roll
    pub scale: [f32; 3],
    pub color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: f32,
    pub is_physics: bool,
    pub mass: f32,
    pub script_source: String,
    pub script_preset: String,
}

impl SceneEntityData {
    pub fn new(id: u32, name: impl Into<String>, mesh_type: impl Into<String>, pos: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            mesh_type: mesh_type.into(),
            pos,
            rot: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            color,
            metallic: 0.2,
            roughness: 0.5,
            emissive: 0.0,
            is_physics: false,
            mass: 1.0,
            script_source: String::new(),
            script_preset: "None".into(),
        }
    }
}

/// Complete Scene Document
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneDocument {
    pub version: u32,
    pub scene_name: String,
    pub author: String,
    pub environment: SceneEnvironmentData,
    pub entities: Vec<SceneEntityData>,
}

impl Default for SceneDocument {
    fn default() -> Self {
        Self {
            version: 1,
            scene_name: "Untitled Metatopia Scene".into(),
            author: "Metatopia Developer".into(),
            environment: SceneEnvironmentData::default(),
            entities: Vec::new(),
        }
    }
}

impl SceneDocument {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            scene_name: name.into(),
            ..Default::default()
        }
    }

    /// Add an entity to the scene
    pub fn add_entity(&mut self, entity: SceneEntityData) {
        self.entities.push(entity);
    }

    /// Serialize scene to JSON string
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Deserialize scene from JSON string
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization error: {}", e))
    }

    /// Save scene to a `.json` or `.metatopia` file on disk
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let json = self.to_json()?;
        let mut file = File::create(path).map_err(|e| format!("Failed to create file {}: {}", path, e))?;
        file.write_all(json.as_bytes()).map_err(|e| format!("Failed to write file {}: {}", path, e))?;
        Ok(())
    }

    /// Load scene from file on disk
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open file {}: {}", path, e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).map_err(|e| format!("Failed to read file {}: {}", path, e))?;
        Self::from_json(&contents)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_serialization_roundtrip() {
        let mut doc = SceneDocument::new("Cyber Arena Level 1");
        let mut ent1 = SceneEntityData::new(1, "NeonCore", "Torus", [0.0, 2.5, 0.0], [0.0, 0.9, 1.0]);
        ent1.emissive = 2.0;
        ent1.script_source = "fn update(e, dt) { e.rotate(1.0*dt, 0.0, 0.0); return e; }".into();
        doc.add_entity(ent1);

        let ent2 = SceneEntityData::new(2, "HoverPillar", "Cylinder", [4.0, 1.0, -2.0], [1.0, 0.8, 0.2]);
        doc.add_entity(ent2);

        let json = doc.to_json().expect("Serialization failed");
        assert!(json.contains("Cyber Arena Level 1"));
        assert!(json.contains("NeonCore"));

        let loaded = SceneDocument::from_json(&json).expect("Deserialization failed");
        assert_eq!(doc, loaded);
        assert_eq!(loaded.entities.len(), 2);
        assert_eq!(loaded.entities[0].emissive, 2.0);
    }
}
