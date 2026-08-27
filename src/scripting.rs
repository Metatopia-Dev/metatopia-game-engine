//! Rhai Scripting Engine for Game Entities and Live Editor
//!
//! Provides embedded, memory-safe scripting in pure Rust (Rhai),
//! binding entity transforms, colors, physics impulses, math utilities,
//! and live editor console logging.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rhai::{Engine, AST, Scope, Dynamic, Array};

/// Replicated entity state accessible and modifiable by Rhai scripts
#[derive(Debug, Clone)]
pub struct ScriptEntityState {
    pub id: u32,
    pub name: String,
    pub pos: [f32; 3],
    pub rot: [f32; 3], // Yaw, Pitch, Roll in radians
    pub scale: [f32; 3],
    pub color: [f32; 3],
    pub emissive: f32,
    pub vel: [f32; 3],
    pub custom_vars: HashMap<String, Dynamic>,
}

impl Default for ScriptEntityState {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Entity".into(),
            pos: [0.0, 0.0, 0.0],
            rot: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0],
            emissive: 0.0,
            vel: [0.0, 0.0, 0.0],
            custom_vars: HashMap::new(),
        }
    }
}

impl ScriptEntityState {
    pub fn new(id: u32, name: impl Into<String>, pos: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            pos,
            rot: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            color,
            emissive: 0.0,
            vel: [0.0, 0.0, 0.0],
            custom_vars: HashMap::new(),
        }
    }
}

/// Script execution and management engine
pub struct ScriptEngine {
    pub engine: Engine,
    pub console_logs: Arc<Mutex<Vec<String>>>,
    pub compiled_cache: HashMap<String, AST>,
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptEngine {
    /// Create a new Rhai ScriptEngine and register game engine bindings
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let console_logs = Arc::new(Mutex::new(Vec::new()));
        let logs_clone = console_logs.clone();

        // 1. Console Logging
        engine.register_fn("log", move |msg: &str| {
            if let Ok(mut logs) = logs_clone.lock() {
                logs.push(msg.to_string());
            }
            println!("[Script]: {}", msg);
        });

        // 2. Register ScriptEntityState Type and Methods
        engine.register_type_with_name::<ScriptEntityState>("Entity")
            .register_fn("get_x", |e: &mut ScriptEntityState| -> f64 { e.pos[0] as f64 })
            .register_fn("get_y", |e: &mut ScriptEntityState| -> f64 { e.pos[1] as f64 })
            .register_fn("get_z", |e: &mut ScriptEntityState| -> f64 { e.pos[2] as f64 })
            .register_fn("set_x", |e: &mut ScriptEntityState, val: f64| { e.pos[0] = val as f32; })
            .register_fn("set_y", |e: &mut ScriptEntityState, val: f64| { e.pos[1] = val as f32; })
            .register_fn("set_z", |e: &mut ScriptEntityState, val: f64| { e.pos[2] = val as f32; })
            .register_fn("set_pos", |e: &mut ScriptEntityState, x: f64, y: f64, z: f64| {
                e.pos = [x as f32, y as f32, z as f32];
            })
            .register_fn("get_pos", |e: &mut ScriptEntityState| -> Array {
                vec![Dynamic::from(e.pos[0] as f64), Dynamic::from(e.pos[1] as f64), Dynamic::from(e.pos[2] as f64)]
            })
            .register_fn("move", |e: &mut ScriptEntityState, dx: f64, dy: f64, dz: f64| {
                e.pos[0] += dx as f32;
                e.pos[1] += dy as f32;
                e.pos[2] += dz as f32;
            })
            .register_fn("rotate", |e: &mut ScriptEntityState, dyaw: f64, dpitch: f64, droll: f64| {
                e.rot[0] += dyaw as f32;
                e.rot[1] += dpitch as f32;
                e.rot[2] += droll as f32;
            })
            .register_fn("set_color", |e: &mut ScriptEntityState, r: f64, g: f64, b: f64| {
                e.color = [r as f32, g as f32, b as f32];
            })
            .register_fn("set_emissive", |e: &mut ScriptEntityState, val: f64| {
                e.emissive = val as f32;
            })
            .register_fn("set_scale", |e: &mut ScriptEntityState, sx: f64, sy: f64, sz: f64| {
                e.scale = [sx as f32, sy as f32, sz as f32];
            })
            .register_fn("set_var", |e: &mut ScriptEntityState, key: &str, val: Dynamic| {
                e.custom_vars.insert(key.to_string(), val);
            })
            .register_fn("get_var", |e: &mut ScriptEntityState, key: &str| -> Dynamic {
                e.custom_vars.get(key).cloned().unwrap_or(Dynamic::UNIT)
            });

        // 3. Math Helpers
        engine.register_fn("sin", |x: f64| -> f64 { x.sin() });
        engine.register_fn("cos", |x: f64| -> f64 { x.cos() });
        engine.register_fn("abs", |x: f64| -> f64 { x.abs() });
        engine.register_fn("sqrt", |x: f64| -> f64 { x.sqrt() });
        engine.register_fn("clamp", |x: f64, min: f64, max: f64| -> f64 { x.clamp(min, max) });

        Self {
            engine,
            console_logs,
            compiled_cache: HashMap::new(),
        }
    }

    /// Compile a Rhai script string into an Abstract Syntax Tree (AST)
    pub fn compile(&mut self, script_source: &str) -> Result<AST, String> {
        self.engine.compile(script_source)
            .map_err(|e| format!("Script Compilation Error: {}", e))
    }

    /// Execute `init(entity)` function in the script
    pub fn execute_init(&self, ast: &AST, entity: &mut ScriptEntityState) -> Result<(), String> {
        let mut scope = Scope::new();
        if self.engine.call_fn::<()>(&mut scope, ast, "init", (entity.clone(),)).is_ok() {
            // If fn takes by value or modifies scope
        }
        // Also support calling with mutable reference syntax:
        let _ = self.engine.call_fn::<()>(&mut scope, ast, "on_init", ());
        Ok(())
    }

    /// Execute `update(entity, dt)` or `update(dt)` function in the script
    pub fn execute_update(&self, ast: &AST, entity: &mut ScriptEntityState, dt: f32) -> Result<(), String> {
        let mut scope = Scope::new();
        scope.push("entity", entity.clone());
        scope.push("dt", dt as f64);

        // Try calling update(entity, dt)
        if let Ok(modified_entity) = self.engine.call_fn::<ScriptEntityState>(&mut scope, ast, "update", (entity.clone(), dt as f64)) {
            *entity = modified_entity;
            return Ok(());
        }

        // Try evaluating script directly in scope if no function defined
        if let Err(e) = self.engine.run_ast_with_scope(&mut scope, ast) {
            return Err(format!("Script Runtime Error: {}", e));
        }

        if let Some(modified) = scope.get_value::<ScriptEntityState>("entity") {
            *entity = modified;
        }

        Ok(())
    }

    /// Clear console logs
    pub fn clear_console(&self) {
        if let Ok(mut logs) = self.console_logs.lock() {
            logs.clear();
        }
    }

    /// Get all recent console logs
    pub fn get_console_logs(&self) -> Vec<String> {
        self.console_logs.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_rotation_and_movement() {
        let mut script_engine = ScriptEngine::new();
        let mut entity = ScriptEntityState::new(1, "Spinner", [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);

        let script = r#"
            fn update(e, dt) {
                e.rotate(2.0 * dt, 0.0, 0.0);
                e.move(0.0, 1.0 * dt, 0.0);
                return e;
            }
        "#;

        let ast = script_engine.compile(script).unwrap();
        script_engine.execute_update(&ast, &mut entity, 0.5).unwrap();

        assert!((entity.rot[0] - 1.0).abs() < 1e-4);
        assert!((entity.pos[1] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_script_color_and_variables() {
        let mut script_engine = ScriptEngine::new();
        let mut entity = ScriptEntityState::new(2, "GlowingCube", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        let script = r#"
            fn update(e, dt) {
                e.set_color(0.0, 1.0, 0.5);
                e.set_emissive(2.5);
                e.set_var("health", 100);
                return e;
            }
        "#;

        let ast = script_engine.compile(script).unwrap();
        script_engine.execute_update(&ast, &mut entity, 0.1).unwrap();

        assert_eq!(entity.color, [0.0, 1.0, 0.5]);
        assert_eq!(entity.emissive, 2.5);
        assert_eq!(entity.custom_vars.get("health").unwrap().clone().cast::<i64>(), 100);
    }
}
