//! Metatopia Studio — 3D Game Developer Editor & Rhai Scripting Suite
//!
//! Interactive 3D visual editor with:
//! - 3D Scene Viewport with infinite grid & transform gizmo
//! - Entity Outliner & Inspector (transforms, PBR materials, physics)
//! - Embedded Rhai Scripting Engine with live reload & console
//! - Play / Edit simulation modes
//! - Procedural mesh generators (Cube, Sphere, Cylinder, Torus, Capsule, Particles)

use metatopia_engine::quickstart::*;
use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::particles::ParticleSystem;
use metatopia_engine::physics::PhysicsWorld;
use metatopia_engine::scripting::{ScriptEngine, ScriptEntityState};
use cgmath::Vector3;

const SHADER_SRC: &str = include_str!("../shaders/editor.wgsl");

/// Types of 3D meshes that can be placed in the scene
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshType {
    Cube,
    Sphere,
    Cylinder,
    Torus,
    Capsule,
    ParticleEmitter,
    PortalFrame,
}

/// An editable 3D entity in the editor
#[derive(Debug, Clone)]
pub struct EditorEntity {
    pub id: u32,
    pub name: String,
    pub mesh_type: MeshType,
    pub pos: [f32; 3],
    pub rot: [f32; 3], // Yaw, Pitch, Roll
    pub scale: [f32; 3],
    pub color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: f32,
    pub is_dynamic_physics: bool,
    pub script_source: String,
    pub script_ast: Option<rhai::AST>,
    pub script_state: ScriptEntityState,
    pub script_preset_name: String,
}

impl EditorEntity {
    pub fn new(id: u32, name: impl Into<String>, mesh_type: MeshType, pos: [f32; 3], color: [f32; 3]) -> Self {
        let name_str = name.into();
        Self {
            id,
            name: name_str.clone(),
            mesh_type,
            pos,
            rot: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            color,
            metallic: 0.2,
            roughness: 0.4,
            emissive: 0.0,
            is_dynamic_physics: false,
            script_source: String::new(),
            script_ast: None,
            script_state: ScriptEntityState::new(id, name_str, pos, color),
            script_preset_name: "None".into(),
        }
    }
}

/// Simulation Mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    Edit,
    Play,
}

/// Metatopia Studio Game Developer Editor
pub struct MetatopiaStudio {
    pub entities: Vec<EditorEntity>,
    pub next_entity_id: u32,
    pub selected_index: usize,
    pub mode: EditorMode,
    pub script_engine: ScriptEngine,
    pub physics_world: PhysicsWorld,
    pub particle_system: ParticleSystem,
    pub status_message: String,
    pub status_timer: f32,
    pub current_script_preset: usize,
    pub camera_initialized: bool,
}

impl MetatopiaStudio {
    pub fn new() -> Self {
        let mut script_engine = ScriptEngine::new();
        let mut entities = Vec::new();

        // 1. Central Sci-Fi Power Core (Torus with Pulsing Glow)
        let mut core = EditorEntity::new(1, "NeonCore", MeshType::Torus, [0.0, 2.5, 0.0], [0.0, 0.9, 1.0]);
        core.metallic = 0.8;
        core.roughness = 0.1;
        core.emissive = 2.0;
        let core_script = r#"
            fn update(e, dt) {
                e.rotate(1.5 * dt, 0.8 * dt, 0.0);
                let t = get_var(e, "t") + dt * 3.0;
                set_var(e, "t", t);
                e.set_emissive(sin(t) * 2.0 + 2.5);
                return e;
            }
        "#;
        core.script_source = core_script.to_string();
        core.script_ast = script_engine.compile(core_script).ok();
        core.script_preset_name = "Pulsing Glow".into();
        entities.push(core);

        // 2. Floating Hover Cube
        let mut cube = EditorEntity::new(2, "HoverCube", MeshType::Cube, [-4.0, 1.5, 2.0], [1.0, 0.3, 0.7]);
        cube.emissive = 0.5;
        let cube_script = r#"
            fn update(e, dt) {
                e.rotate(0.0, 2.0 * dt, 0.0);
                let t = get_var(e, "t") + dt * 2.0;
                set_var(e, "t", t);
                e.set_y(1.5 + sin(t) * 0.8);
                return e;
            }
        "#;
        cube.script_source = cube_script.to_string();
        cube.script_ast = script_engine.compile(cube_script).ok();
        cube.script_preset_name = "Hover & Bob".into();
        entities.push(cube);

        // 3. Golden Cyber Pillar (Cylinder)
        let mut pillar = EditorEntity::new(3, "Pillar", MeshType::Cylinder, [4.0, 2.0, -2.0], [1.0, 0.8, 0.1]);
        pillar.scale = [0.8, 2.0, 0.8];
        pillar.metallic = 0.9;
        pillar.roughness = 0.2;
        pillar.script_preset_name = "Static".into();
        entities.push(pillar);

        // 4. Emerald Quantum Sphere
        let mut sphere = EditorEntity::new(4, "QuantumOrb", MeshType::Sphere, [3.0, 1.2, 3.0], [0.1, 1.0, 0.5]);
        sphere.scale = [0.7, 0.7, 0.7];
        sphere.emissive = 1.0;
        let sphere_script = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt * 1.5;
                set_var(e, "t", t);
                e.set_color(sin(t) * 0.5 + 0.5, sin(t + 2.0) * 0.5 + 0.5, sin(t + 4.0) * 0.5 + 0.5);
                return e;
            }
        "#;
        sphere.script_source = sphere_script.to_string();
        sphere.script_ast = script_engine.compile(sphere_script).ok();
        sphere.script_preset_name = "Color Shift".into();
        entities.push(sphere);

        // 5. Non-Euclidean Portal Gate
        let mut portal = EditorEntity::new(5, "PortalGate", MeshType::Capsule, [0.0, 2.0, -6.0], [0.7, 0.1, 1.0]);
        portal.scale = [0.2, 3.0, 2.0];
        portal.emissive = 3.0;
        portal.script_preset_name = "Portal Gate".into();
        entities.push(portal);

        Self {
            entities,
            next_entity_id: 6,
            selected_index: 0,
            mode: EditorMode::Edit,
            script_engine,
            physics_world: PhysicsWorld::new(),
            particle_system: ParticleSystem::new(1000),
            status_message: "METATOPIA STUDIO READY. Press [SPACE] to Play/Edit, [1-5] to Spawn Objects, [G] for Scripts.".into(),
            status_timer: 5.0,
            current_script_preset: 0,
            camera_initialized: false,
        }
    }

    /// Add a new 3D entity to the scene
    pub fn spawn_entity(&mut self, mesh_type: MeshType, name: &str, color: [f32; 3]) {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        let spawn_pos = [0.0, 1.5, 0.0];
        let entity = EditorEntity::new(id, name, mesh_type, spawn_pos, color);
        self.entities.push(entity);
        self.selected_index = self.entities.len() - 1;

        self.set_status(&format!("Spawned {} (ID: {})", name, id));
    }

    /// Cycle script presets on selected entity
    pub fn cycle_script_preset(&mut self) {
        if self.entities.is_empty() { return; }
        let presets = [
            ("Pulsing Glow", r#"
                fn update(e, dt) {
                    e.rotate(1.5 * dt, 0.8 * dt, 0.0);
                    let t = get_var(e, "t") + dt * 4.0;
                    set_var(e, "t", t);
                    e.set_emissive(sin(t) * 3.0 + 3.0);
                    return e;
                }
            "#),
            ("Hover & Bob", r#"
                fn update(e, dt) {
                    e.rotate(0.0, 2.5 * dt, 0.0);
                    let t = get_var(e, "t") + dt * 2.5;
                    set_var(e, "t", t);
                    e.set_y(1.5 + sin(t) * 1.2);
                    return e;
                }
            "#),
            ("Color Shift", r#"
                fn update(e, dt) {
                    let t = get_var(e, "t") + dt * 2.0;
                    set_var(e, "t", t);
                    e.set_color(sin(t) * 0.5 + 0.5, sin(t + 2.0) * 0.5 + 0.5, sin(t + 4.0) * 0.5 + 0.5);
                    return e;
                }
            "#),
            ("Orbit Center", r#"
                fn update(e, dt) {
                    let t = get_var(e, "t") + dt * 1.5;
                    set_var(e, "t", t);
                    e.set_x(sin(t) * 5.0);
                    e.set_z(cos(t) * 5.0);
                    e.rotate(3.0 * dt, 0.0, 0.0);
                    return e;
                }
            "#),
            ("None", ""),
        ];

        self.current_script_preset = (self.current_script_preset + 1) % presets.len();
        let (preset_name, src) = presets[self.current_script_preset];

        let entity_name = self.entities[self.selected_index].name.clone();
        let entity = &mut self.entities[self.selected_index];
        entity.script_preset_name = preset_name.to_string();
        entity.script_source = src.to_string();

        if !src.is_empty() {
            entity.script_ast = self.script_engine.compile(src).ok();
            self.set_status(&format!("Attached Script Preset '{}' to {}", preset_name, entity_name));
        } else {
            entity.script_ast = None;
            self.set_status(&format!("Cleared Script on {}", entity_name));
        }
    }

    pub fn set_status(&mut self, msg: &str) {
        self.status_message = msg.to_string();
        self.status_timer = 4.0;
        println!("[Studio]: {}", msg);
    }
}

impl GameApp for MetatopiaStudio {
    fn title(&self) -> &str {
        "Metatopia Studio — 3D Game Editor & Rhai Scripting Suite"
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        if !self.camera_initialized {
            ctx.camera.position = Vector3::new(0.0, 6.0, 14.0);
            ctx.camera.pitch = -0.35;
            ctx.camera.move_speed = 12.0;
            self.camera_initialized = true;
        }

        ctx.scene.sun_direction = [-0.5, -1.0, -0.6, 0.0];
        ctx.scene.sun_color = [1.0, 0.98, 0.92, 1.0];

        // Status timer decay
        if self.status_timer > 0.0 {
            self.status_timer -= ctx.dt;
        }

        // 1. Camera Fly Movement (WASD + Right-click / Free Fly)
        ctx.default_camera_movement();

        // 2. Play / Edit Mode Toggle (Spacebar)
        if ctx.key_pressed(VirtualKey::Space) {
            self.mode = match self.mode {
                EditorMode::Edit => {
                    self.set_status("▶ PLAY MODE ACTIVATED (Scripts & Physics Live)");
                    EditorMode::Play
                }
                EditorMode::Play => {
                    self.set_status("⏸ EDIT MODE ACTIVATED (Inspector & Gizmos Active)");
                    EditorMode::Edit
                }
            };
        }

        // 3. Selection Navigation (Tab to cycle entities)
        if ctx.key_pressed(VirtualKey::Tab) && !self.entities.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.entities.len();
            let name = self.entities[self.selected_index].name.clone();
            self.set_status(&format!("Selected Entity: '{}' (Index: {})", name, self.selected_index));
        }

        // 4. Object Spawning Hotkeys (1..5)
        if ctx.key_pressed(VirtualKey::Digit1) {
            self.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.8, 1.0]);
        }
        if ctx.key_pressed(VirtualKey::Digit2) {
            self.spawn_entity(MeshType::Sphere, "NeonSphere", [1.0, 0.2, 0.5]);
        }
        if ctx.key_pressed(VirtualKey::Digit3) {
            self.spawn_entity(MeshType::Cylinder, "PowerPillar", [1.0, 0.9, 0.1]);
        }
        if ctx.key_pressed(VirtualKey::Digit4) {
            self.spawn_entity(MeshType::Torus, "GravityRing", [0.8, 0.2, 1.0]);
        }
        if ctx.key_pressed(VirtualKey::Digit5) {
            self.spawn_entity(MeshType::Capsule, "BioCapsule", [0.1, 1.0, 0.7]);
        }

        // 5. Script Preset Cycler (G)
        if ctx.key_pressed(VirtualKey::KeyG) {
            self.cycle_script_preset();
        }

        // 6. Delete Entity (X or Delete)
        if ctx.key_pressed(VirtualKey::KeyX) && !self.entities.is_empty() {
            let removed = self.entities.remove(self.selected_index);
            self.set_status(&format!("Deleted Entity: '{}'", removed.name));
            if self.selected_index >= self.entities.len() && !self.entities.is_empty() {
                self.selected_index = self.entities.len() - 1;
            }
        }

        // 7. Entity Manipulation with Arrow Keys & Q/E in Edit Mode
        if !self.entities.is_empty() {
            let entity = &mut self.entities[self.selected_index];

            let move_spd = 6.0 * ctx.dt;
            if ctx.key_held(VirtualKey::ArrowLeft) { entity.pos[0] -= move_spd; }
            if ctx.key_held(VirtualKey::ArrowRight) { entity.pos[0] += move_spd; }
            if ctx.key_held(VirtualKey::ArrowUp) { entity.pos[2] -= move_spd; }
            if ctx.key_held(VirtualKey::ArrowDown) { entity.pos[2] += move_spd; }
            if ctx.key_held(VirtualKey::PageUp) { entity.pos[1] += move_spd; }
            if ctx.key_held(VirtualKey::PageDown) { entity.pos[1] -= move_spd; }

            if ctx.key_held(VirtualKey::KeyQ) { entity.rot[0] += 2.0 * ctx.dt; }
            if ctx.key_held(VirtualKey::KeyE) { entity.rot[0] -= 2.0 * ctx.dt; }

            // Sync script state
            entity.script_state.pos = entity.pos;
            entity.script_state.rot = entity.rot;
            entity.script_state.scale = entity.scale;
            entity.script_state.color = entity.color;
            entity.script_state.emissive = entity.emissive;
        }

        // 8. Re-compile All Scripts (R)
        if ctx.key_pressed(VirtualKey::KeyR) {
            for entity in &mut self.entities {
                if !entity.script_source.is_empty() {
                    match self.script_engine.compile(&entity.script_source) {
                        Ok(ast) => {
                            entity.script_ast = Some(ast);
                        }
                        Err(e) => {
                            self.script_engine.console_logs.lock().unwrap().push(e);
                        }
                    }
                }
            }
            self.set_status("All Scripts Re-Compiled & Reloaded.");
        }

        // 9. Particle Simulation
        if self.mode == EditorMode::Play {
            self.particle_system.burst(
                Vector3::new(0.0, 2.5, 0.0),
                2,
                [0.0, 0.9, 1.0, 1.0],
                [0.8, 0.1, 0.9, 0.0],
                3.0,
                0.8,
            );
        }
        self.particle_system.update(ctx.dt, None);

        // 10. Execute Scripts in Play Mode
        if self.mode == EditorMode::Play {
            for entity in &mut self.entities {
                if let Some(ast) = &entity.script_ast {
                    let _ = self.script_engine.execute_update(ast, &mut entity.script_state, ctx.dt);
                    // Write back modified properties
                    entity.pos = entity.script_state.pos;
                    entity.rot = entity.script_state.rot;
                    entity.color = entity.script_state.color;
                    entity.emissive = entity.script_state.emissive;
                    entity.scale = entity.script_state.scale;
                }
            }
        }

        // Pass selection ID & mode to Shader Uniforms
        let selected_id = if !self.entities.is_empty() {
            self.entities[self.selected_index].id as f32
        } else {
            0.0
        };

        let is_play = if self.mode == EditorMode::Play { 1.0 } else { 0.0 };
        ctx.scene.game_data = [ctx.time, is_play, selected_id, 0.0];
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(20000);
        let mut indices = Vec::with_capacity(40000);

        // 1. Grid Floor Plane (100x100)
        let (plane_v, plane_i) = ProceduralMesh::plane(100.0, 100.0, 20, 20, [0.15, 0.18, 0.22]);
        let base = verts.len() as u32;
        for v in plane_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.1, 0.8, 0.0, 0.0]));
        }
        for idx in plane_i { indices.push(base + idx); }

        // 2. Render Scene Entities
        for entity in &self.entities {
            let (shape_v, shape_i) = match entity.mesh_type {
                MeshType::Cube => {
                    let hx = entity.scale[0] * 0.5;
                    let hy = entity.scale[1] * 0.5;
                    let hz = entity.scale[2] * 0.5;
                    // Cube faces
                    let mut c_v = Vec::new();
                    let mut c_i = Vec::new();
                    let corners = [
                        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
                    ];
                    let faces = [
                        (0, 1, 2, 3, [0.0, 0.0, -1.0]), (5, 4, 7, 6, [0.0, 0.0, 1.0]),
                        (4, 0, 3, 7, [-1.0, 0.0, 0.0]), (1, 5, 6, 2, [1.0, 0.0, 0.0]),
                        (3, 2, 6, 7, [0.0, 1.0, 0.0]),  (4, 5, 1, 0, [0.0, -1.0, 0.0]),
                    ];
                    for &(a, b, c, d, norm) in &faces {
                        let f_base = c_v.len() as u32;
                        c_v.push(GameVertex::colored(corners[a], norm, entity.color));
                        c_v.push(GameVertex::colored(corners[b], norm, entity.color));
                        c_v.push(GameVertex::colored(corners[c], norm, entity.color));
                        c_v.push(GameVertex::colored(corners[d], norm, entity.color));
                        c_i.extend_from_slice(&[f_base, f_base + 1, f_base + 2, f_base, f_base + 2, f_base + 3]);
                    }
                    (c_v, c_i)
                }
                MeshType::Sphere => {
                    ProceduralMesh::capsule(entity.scale[0] * 0.5, 0.01, 8, 16, entity.color)
                }
                MeshType::Cylinder => {
                    ProceduralMesh::cylinder(entity.scale[0] * 0.5, entity.scale[1], 16, entity.color)
                }
                MeshType::Torus => {
                    ProceduralMesh::torus(entity.scale[0] * 0.8, entity.scale[0] * 0.25, 24, 12, entity.color)
                }
                MeshType::Capsule | MeshType::PortalFrame => {
                    ProceduralMesh::capsule(entity.scale[0] * 0.4, entity.scale[1], 8, 16, entity.color)
                }
                MeshType::ParticleEmitter => {
                    ProceduralMesh::torus(0.6, 0.1, 16, 8, entity.color)
                }
            };

            let base_offset = verts.len() as u32;
            let pbr = [entity.metallic, entity.roughness, entity.emissive, entity.id as f32];

            // Apply Entity Transform (Translation & Yaw Rotation)
            let cos_yaw = entity.rot[0].cos();
            let sin_yaw = entity.rot[0].sin();

            for v in shape_v {
                let rx = v.position[0] * cos_yaw - v.position[2] * sin_yaw;
                let rz = v.position[0] * sin_yaw + v.position[2] * cos_yaw;
                let world_pos = [
                    rx + entity.pos[0],
                    v.position[1] + entity.pos[1],
                    rz + entity.pos[2],
                ];

                let r_norm_x = v.normal[0] * cos_yaw - v.normal[2] * sin_yaw;
                let r_norm_z = v.normal[0] * sin_yaw + v.normal[2] * cos_yaw;
                let world_norm = [r_norm_x, v.normal[1], r_norm_z];

                verts.push(GameVertex::new(world_pos, world_norm, v.color, pbr));
            }

            for idx in shape_i {
                indices.push(base_offset + idx);
            }
        }

        // 3. Selection Transform Gizmo (Red X, Green Y, Blue Z axes)
        if !self.entities.is_empty() && self.mode == EditorMode::Edit {
            let entity = &self.entities[self.selected_index];
            let ep = entity.pos;
            let g_len = 1.5;
            let g_rad = 0.04;

            // X Axis (Red)
            let (x_v, x_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [1.0, 0.1, 0.1]);
            let x_base = verts.len() as u32;
            for v in x_v {
                let pos = [ep[0] + v.position[1] + g_len * 0.5, ep[1] + v.position[0], ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [1.0, 0.0, 0.0], [1.0, 0.1, 0.1], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in x_i { indices.push(x_base + idx); }

            // Y Axis (Green)
            let (y_v, y_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [0.1, 1.0, 0.2]);
            let y_base = verts.len() as u32;
            for v in y_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[1] + g_len * 0.5, ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [0.0, 1.0, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in y_i { indices.push(y_base + idx); }

            // Z Axis (Blue)
            let (z_v, z_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [0.2, 0.4, 1.0]);
            let z_base = verts.len() as u32;
            for v in z_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[2], ep[2] + v.position[1] + g_len * 0.5];
                verts.push(GameVertex::new(pos, [0.0, 0.0, 1.0], [0.2, 0.4, 1.0], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in z_i { indices.push(z_base + idx); }
        }

        // 4. Particle Mesh
        let (p_v, p_i) = self.particle_system.build_mesh();
        let p_base = verts.len() as u32;
        verts.extend(p_v);
        for idx in p_i { indices.push(p_base + idx); }

        (verts, indices)
    }
}

fn main() {
    println!("============================================================");
    println!("  METATOPIA STUDIO — 3D Game Editor & Rhai Scripting Suite");
    println!("============================================================");
    println!(" Controls:");
    println!("   WASD + Mouse   : Fly / Orbit 3D Camera");
    println!("   SPACE          : Toggle [PLAY] / [EDIT] simulation modes");
    println!("   TAB            : Cycle selection through scene entities");
    println!("   1 / 2 / 3 / 4 / 5 : Spawn Cube / Sphere / Cylinder / Torus / Capsule");
    println!("   G              : Cycle Rhai Script Presets on selected object");
    println!("   R              : Recompile and Hot-Reload all Rhai scripts");
    println!("   Arrow Keys     : Translate selected object (X / Z)");
    println!("   PgUp / PgDown  : Translate selected object vertically (Y)");
    println!("   Q / E          : Rotate selected object");
    println!("   X              : Delete selected object");
    println!("============================================================");

    run_game(MetatopiaStudio::new());
}
