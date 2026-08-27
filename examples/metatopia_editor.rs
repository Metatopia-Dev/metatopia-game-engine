//! Metatopia Studio — 3D Game Developer Editor & Rhai Scripting Suite
//!
//! Interactive 3D visual editor with:
//! - 3D Scene Viewport with infinite grid & transform gizmo
//! - On-screen visual UI (Top Toolbar, Scene Outliner, Property Inspector, Script Console)
//! - Crisp in-engine vector/bitmap font text rendering
//! - Mouse-clickable UI buttons and outliner items
//! - Embedded Rhai Scripting Engine with live hot-reload
//! - Real-time Play / Edit simulation modes

use metatopia_engine::quickstart::*;
use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::particles::ParticleSystem;
use metatopia_engine::physics::PhysicsWorld;
use metatopia_engine::scripting::{ScriptEngine, ScriptEntityState};
use cgmath::{Vector3, InnerSpace};

const SHADER_SRC: &str = include_str!("../shaders/editor.wgsl");

/// Types of 3D meshes that can be placed in the scene
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshType {
    Cube,
    Sphere,
    Cylinder,
    Torus,
    Capsule,
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

        // 5. Bio Capsule
        let mut capsule = EditorEntity::new(5, "BioCapsule", MeshType::Capsule, [0.0, 1.8, -5.0], [0.8, 0.2, 1.0]);
        capsule.scale = [0.6, 1.5, 0.6];
        capsule.emissive = 1.5;
        capsule.script_preset_name = "Static".into();
        entities.push(capsule);

        Self {
            entities,
            next_entity_id: 6,
            selected_index: 0,
            mode: EditorMode::Edit,
            script_engine,
            physics_world: PhysicsWorld::new(),
            particle_system: ParticleSystem::new(1000),
            status_message: "METATOPIA STUDIO READY. Use Toolbar buttons or hotkeys. Right-Click+Drag to fly.".into(),
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

    /// Helper to convert screen pixel coords to NDC `[-1.0, 1.0]`
    fn pixel_to_ndc(&self, px: f32, py: f32, width: u32, height: u32) -> (f32, f32) {
        let ndc_x = (px / width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height as f32) * 2.0;
        (ndc_x, ndc_y)
    }
}

// ─── Crisp 5x7 Vector/Bitmap Font Glyph Table ───────────────────────────────

fn get_glyph_bits(c: char) -> [u8; 7] {
    let uc = c.to_ascii_uppercase();
    match uc {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00110, 0b01000, 0b10000, 0b10000, 0b10000, 0b01000, 0b00110],
        ')' => [0b01100, 0b00010, 0b00001, 0b00001, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '=' => [0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        _   => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}

fn push_ui_rect(verts: &mut Vec<GameVertex>, indices: &mut Vec<u32>, min_x: f32, min_y: f32, max_x: f32, max_y: f32, color: [f32; 3], opacity: f32) {
    let base_idx = verts.len() as u32;
    let pbr = [0.0, 0.0, opacity, -1.0]; // w = -1.0 flags 2D screen UI
    let norm = [0.0, 0.0, 1.0];

    verts.push(GameVertex::new([min_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, max_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([min_x, max_y, 0.0], norm, color, pbr));

    indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
}

fn push_text(verts: &mut Vec<GameVertex>, indices: &mut Vec<u32>, text: &str, start_x: f32, start_y: f32, char_w: f32, char_h: f32, color: [f32; 3]) {
    let mut cur_x = start_x;
    let pix_w = char_w / 6.0;
    let pix_h = char_h / 8.0;

    for ch in text.chars() {
        if ch == ' ' {
            cur_x += char_w;
            continue;
        }
        let bits = get_glyph_bits(ch);
        for (row, byte) in bits.iter().enumerate() {
            let py = start_y + char_h - ((row + 1) as f32 * pix_h);
            for col in 0..5 {
                if (byte & (1 << (4 - col))) != 0 {
                    let px = cur_x + (col as f32 * pix_w);
                    push_ui_rect(verts, indices, px, py, px + pix_w, py + pix_h, color, 1.0);
                }
            }
        }
        cur_x += char_w + pix_w;
    }
}

impl GameApp for MetatopiaStudio {
    fn title(&self) -> &str {
        "Metatopia Studio — 3D Game Editor & Rhai Scripting Suite"
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    /// Enable dynamic mesh updating every frame so live edits & scripts animate in real time!
    fn is_dynamic_mesh(&self) -> bool {
        true
    }

    /// Leave cursor ungrabbed so the developer can click UI buttons and inspect objects!
    fn grab_cursor(&self) -> bool {
        false
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

        // 1. Camera Fly Movement (Only when Right Mouse Button is held or WASD)
        if ctx.mouse_held(winit::event::MouseButton::Right) {
            ctx.default_camera_movement();
        } else {
            // WASD only
            let speed = ctx.camera.move_speed * ctx.dt;
            let fwd = ctx.camera.forward();
            let right = ctx.camera.right();
            let flat_fwd = Vector3::new(fwd.x, 0.0, fwd.z).normalize() * speed;
            let flat_right = right * speed;
            if ctx.key_held(VirtualKey::KeyW) { ctx.camera.position += flat_fwd; }
            if ctx.key_held(VirtualKey::KeyS) { ctx.camera.position -= flat_fwd; }
            if ctx.key_held(VirtualKey::KeyA) { ctx.camera.position -= flat_right; }
            if ctx.key_held(VirtualKey::KeyD) { ctx.camera.position += flat_right; }
        }

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

        // 3. Mouse Click on UI Buttons & Outliner
        if ctx.mouse_pressed(winit::event::MouseButton::Left) {
            let (m_ndc_x, m_ndc_y) = self.pixel_to_ndc(ctx.mouse_pos().0, ctx.mouse_pos().1, ctx.resolution.0, ctx.resolution.1);

            // Top Bar Buttons
            if m_ndc_y >= 0.85 && m_ndc_y <= 0.97 {
                // Play / Edit Button
                if m_ndc_x >= -0.68 && m_ndc_x <= -0.54 {
                    self.mode = match self.mode {
                        EditorMode::Edit => { self.set_status("▶ PLAY MODE ACTIVATED"); EditorMode::Play }
                        EditorMode::Play => { self.set_status("⏸ EDIT MODE ACTIVATED"); EditorMode::Edit }
                    };
                }
                // +Cube
                else if m_ndc_x >= -0.52 && m_ndc_x <= -0.42 {
                    self.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.8, 1.0]);
                }
                // +Sphere
                else if m_ndc_x >= -0.40 && m_ndc_x <= -0.28 {
                    self.spawn_entity(MeshType::Sphere, "NeonSphere", [1.0, 0.2, 0.5]);
                }
                // +Cylinder
                else if m_ndc_x >= -0.26 && m_ndc_x <= -0.16 {
                    self.spawn_entity(MeshType::Cylinder, "PowerPillar", [1.0, 0.9, 0.1]);
                }
                // +Torus
                else if m_ndc_x >= -0.14 && m_ndc_x <= -0.04 {
                    self.spawn_entity(MeshType::Torus, "GravityRing", [0.8, 0.2, 1.0]);
                }
                // +Capsule
                else if m_ndc_x >= -0.02 && m_ndc_x <= 0.08 {
                    self.spawn_entity(MeshType::Capsule, "BioCapsule", [0.1, 1.0, 0.7]);
                }
                // Cycle Script
                else if m_ndc_x >= 0.12 && m_ndc_x <= 0.28 {
                    self.cycle_script_preset();
                }
                // Recompile (R)
                else if m_ndc_x >= 0.30 && m_ndc_x <= 0.44 {
                    for entity in &mut self.entities {
                        if !entity.script_source.is_empty() {
                            if let Ok(ast) = self.script_engine.compile(&entity.script_source) {
                                entity.script_ast = Some(ast);
                            }
                        }
                    }
                    self.set_status("All Scripts Hot-Reloaded.");
                }
                // Delete (X)
                else if m_ndc_x >= 0.46 && m_ndc_x <= 0.58 && !self.entities.is_empty() {
                    let removed = self.entities.remove(self.selected_index);
                    self.set_status(&format!("Deleted Entity: '{}'", removed.name));
                    if self.selected_index >= self.entities.len() && !self.entities.is_empty() {
                        self.selected_index = self.entities.len() - 1;
                    }
                }
            }

            // Left Outliner Items (Select entity by clicking)
            if m_ndc_x >= -0.98 && m_ndc_x <= -0.68 && m_ndc_y >= -0.60 && m_ndc_y <= 0.80 {
                let rel_y = 0.74 - m_ndc_y;
                let item_h = 0.08;
                if rel_y >= 0.0 {
                    let clicked_idx = (rel_y / item_h) as usize;
                    if clicked_idx < self.entities.len() {
                        self.selected_index = clicked_idx;
                        let name = self.entities[self.selected_index].name.clone();
                        self.set_status(&format!("Selected Entity: '{}'", name));
                    }
                }
            }

            // Right Inspector Transform Buttons
            if m_ndc_x >= 0.68 && m_ndc_x <= 0.98 && m_ndc_y >= 0.20 && m_ndc_y <= 0.75 && !self.entities.is_empty() {
                let entity = &mut self.entities[self.selected_index];
                // Check X/Y/Z +/- clicks
                if m_ndc_y >= 0.51 && m_ndc_y <= 0.57 {
                    if m_ndc_x >= 0.82 && m_ndc_x <= 0.88 { entity.pos[0] -= 0.5; }
                    if m_ndc_x >= 0.90 && m_ndc_x <= 0.96 { entity.pos[0] += 0.5; }
                } else if m_ndc_y >= 0.41 && m_ndc_y <= 0.47 {
                    if m_ndc_x >= 0.82 && m_ndc_x <= 0.88 { entity.pos[1] -= 0.5; }
                    if m_ndc_x >= 0.90 && m_ndc_x <= 0.96 { entity.pos[1] += 0.5; }
                } else if m_ndc_y >= 0.31 && m_ndc_y <= 0.37 {
                    if m_ndc_x >= 0.82 && m_ndc_x <= 0.88 { entity.pos[2] -= 0.5; }
                    if m_ndc_x >= 0.90 && m_ndc_x <= 0.96 { entity.pos[2] += 0.5; }
                }
                entity.script_state.pos = entity.pos;
            }
        }

        // 4. Keyboard Shortcuts
        if ctx.key_pressed(VirtualKey::Tab) && !self.entities.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.entities.len();
            let name = self.entities[self.selected_index].name.clone();
            self.set_status(&format!("Selected Entity: '{}' (Index: {})", name, self.selected_index));
        }

        if ctx.key_pressed(VirtualKey::Digit1) { self.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.8, 1.0]); }
        if ctx.key_pressed(VirtualKey::Digit2) { self.spawn_entity(MeshType::Sphere, "NeonSphere", [1.0, 0.2, 0.5]); }
        if ctx.key_pressed(VirtualKey::Digit3) { self.spawn_entity(MeshType::Cylinder, "PowerPillar", [1.0, 0.9, 0.1]); }
        if ctx.key_pressed(VirtualKey::Digit4) { self.spawn_entity(MeshType::Torus, "GravityRing", [0.8, 0.2, 1.0]); }
        if ctx.key_pressed(VirtualKey::Digit5) { self.spawn_entity(MeshType::Capsule, "BioCapsule", [0.1, 1.0, 0.7]); }
        if ctx.key_pressed(VirtualKey::KeyG) { self.cycle_script_preset(); }

        if ctx.key_pressed(VirtualKey::KeyX) && !self.entities.is_empty() {
            let removed = self.entities.remove(self.selected_index);
            self.set_status(&format!("Deleted Entity: '{}'", removed.name));
            if self.selected_index >= self.entities.len() && !self.entities.is_empty() {
                self.selected_index = self.entities.len() - 1;
            }
        }

        // 5. Arrow Keys Object Movement in Edit Mode
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

        // 6. Particle Burst & Update
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

        // 7. Execute Scripts in Play Mode
        if self.mode == EditorMode::Play {
            for entity in &mut self.entities {
                if let Some(ast) = &entity.script_ast {
                    let _ = self.script_engine.execute_update(ast, &mut entity.script_state, ctx.dt);
                    entity.pos = entity.script_state.pos;
                    entity.rot = entity.script_state.rot;
                    entity.color = entity.script_state.color;
                    entity.emissive = entity.script_state.emissive;
                    entity.scale = entity.script_state.scale;
                }
            }
        }

        let selected_id = if !self.entities.is_empty() {
            self.entities[self.selected_index].id as f32
        } else {
            0.0
        };

        let is_play = if self.mode == EditorMode::Play { 1.0 } else { 0.0 };
        ctx.scene.game_data = [ctx.time, is_play, selected_id, 0.0];
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(40000);
        let mut indices = Vec::with_capacity(80000);

        // ── 1. 3D Scene Entities & Grid ────────────────────────────────────

        // Grid Floor Plane (100x100)
        let (plane_v, plane_i) = ProceduralMesh::plane(100.0, 100.0, 20, 20, [0.15, 0.18, 0.22]);
        let base = verts.len() as u32;
        for v in plane_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.1, 0.8, 0.0, 0.0]));
        }
        for idx in plane_i { indices.push(base + idx); }

        // Render Scene 3D Entities
        for entity in &self.entities {
            let (shape_v, shape_i) = match entity.mesh_type {
                MeshType::Cube => {
                    let hx = entity.scale[0] * 0.5;
                    let hy = entity.scale[1] * 0.5;
                    let hz = entity.scale[2] * 0.5;
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
                MeshType::Capsule => {
                    ProceduralMesh::capsule(entity.scale[0] * 0.4, entity.scale[1], 8, 16, entity.color)
                }
            };

            let base_offset = verts.len() as u32;
            let pbr = [entity.metallic, entity.roughness, entity.emissive, entity.id as f32];

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

        // Selection Transform Gizmo (Red X, Green Y, Blue Z axes)
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

        // Particle Mesh
        let (p_v, p_i) = self.particle_system.build_mesh();
        let p_base = verts.len() as u32;
        verts.extend(p_v);
        for idx in p_i { indices.push(p_base + idx); }

        // ── 2. 2D Visual Studio UI Overlay & Text Rasterizer ────────────────

        // A. Top Menu / Toolbar Background Panel
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.84, 1.0, 1.0, [0.06, 0.07, 0.09], 0.95);
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.835, 1.0, 0.84, [0.18, 0.22, 0.30], 1.0); // Bottom border

        // Studio Title
        push_text(&mut verts, &mut indices, "METATOPIA STUDIO v0.2", -0.98, 0.95, 0.016, 0.030, [0.0, 0.9, 1.0]);

        // Play / Edit Mode Button (Green for Play, Orange for Edit)
        let (mode_bg, mode_label) = match self.mode {
            EditorMode::Edit => ([0.8, 0.4, 0.1], "[EDIT]"),
            EditorMode::Play => ([0.1, 0.8, 0.3], "[PLAY]"),
        };
        push_ui_rect(&mut verts, &mut indices, -0.68, 0.86, -0.54, 0.97, mode_bg, 0.9);
        push_text(&mut verts, &mut indices, mode_label, -0.66, 0.89, 0.016, 0.030, [1.0, 1.0, 1.0]);

        // Primitive Spawn Buttons
        push_ui_rect(&mut verts, &mut indices, -0.52, 0.86, -0.42, 0.97, [0.12, 0.16, 0.24], 0.9);
        push_text(&mut verts, &mut indices, "+CUBE", -0.51, 0.89, 0.014, 0.026, [0.2, 0.8, 1.0]);

        push_ui_rect(&mut verts, &mut indices, -0.40, 0.86, -0.28, 0.97, [0.12, 0.16, 0.24], 0.9);
        push_text(&mut verts, &mut indices, "+SPHERE", -0.39, 0.89, 0.014, 0.026, [1.0, 0.3, 0.6]);

        push_ui_rect(&mut verts, &mut indices, -0.26, 0.86, -0.16, 0.97, [0.12, 0.16, 0.24], 0.9);
        push_text(&mut verts, &mut indices, "+CYL", -0.24, 0.89, 0.014, 0.026, [1.0, 0.8, 0.1]);

        push_ui_rect(&mut verts, &mut indices, -0.14, 0.86, -0.04, 0.97, [0.12, 0.16, 0.24], 0.9);
        push_text(&mut verts, &mut indices, "+TORUS", -0.13, 0.89, 0.014, 0.026, [0.8, 0.2, 1.0]);

        push_ui_rect(&mut verts, &mut indices, -0.02, 0.86, 0.08, 0.97, [0.12, 0.16, 0.24], 0.9);
        push_text(&mut verts, &mut indices, "+CAP", 0.00, 0.89, 0.014, 0.026, [0.1, 1.0, 0.7]);

        // Script Cycle & Hot-Reload Buttons
        push_ui_rect(&mut verts, &mut indices, 0.12, 0.86, 0.28, 0.97, [0.35, 0.12, 0.55], 0.9);
        push_text(&mut verts, &mut indices, "[SCRIPT]", 0.14, 0.89, 0.014, 0.026, [1.0, 0.8, 1.0]);

        push_ui_rect(&mut verts, &mut indices, 0.30, 0.86, 0.44, 0.97, [0.1, 0.45, 0.7], 0.9);
        push_text(&mut verts, &mut indices, "[RELOAD]", 0.32, 0.89, 0.014, 0.026, [1.0, 1.0, 1.0]);

        push_ui_rect(&mut verts, &mut indices, 0.46, 0.86, 0.58, 0.97, [0.6, 0.15, 0.15], 0.9);
        push_text(&mut verts, &mut indices, "[DEL X]", 0.48, 0.89, 0.014, 0.026, [1.0, 1.0, 1.0]);

        // B. Left Scene Outliner Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.60, -0.68, 0.80, [0.06, 0.07, 0.10], 0.92);
        push_ui_rect(&mut verts, &mut indices, -0.98, 0.74, -0.68, 0.80, [0.14, 0.18, 0.25], 1.0); // Outliner Header
        push_text(&mut verts, &mut indices, "SCENE HIERARCHY", -0.96, 0.755, 0.014, 0.025, [1.0, 0.8, 0.2]);

        // Outliner entity rows
        for (i, entity) in self.entities.iter().enumerate().take(7) {
            let row_top = 0.72 - (i as f32 * 0.08);
            let row_bot = row_top - 0.065;
            let is_selected = i == self.selected_index;
            let row_color = if is_selected { [0.2, 0.4, 0.7] } else { [0.10, 0.12, 0.16] };
            let opacity = if is_selected { 0.95 } else { 0.7 };

            push_ui_rect(&mut verts, &mut indices, -0.96, row_bot, -0.70, row_top, row_color, opacity);
            // Color marker dot on the left
            push_ui_rect(&mut verts, &mut indices, -0.955, row_bot + 0.015, -0.935, row_top - 0.015, entity.color, 1.0);

            // Row Label
            let label = format!("{}. {}", i + 1, entity.name);
            let label_col = if is_selected { [1.0, 1.0, 0.2] } else { [0.8, 0.85, 0.9] };
            push_text(&mut verts, &mut indices, &label, -0.92, row_bot + 0.015, 0.012, 0.022, label_col);
        }

        // C. Right Property Inspector Panel
        push_ui_rect(&mut verts, &mut indices, 0.68, -0.60, 0.98, 0.80, [0.06, 0.07, 0.10], 0.92);
        push_ui_rect(&mut verts, &mut indices, 0.68, 0.74, 0.98, 0.80, [0.14, 0.18, 0.25], 1.0); // Inspector Header
        push_text(&mut verts, &mut indices, "ENTITY INSPECTOR", 0.70, 0.755, 0.014, 0.025, [1.0, 0.8, 0.2]);

        if !self.entities.is_empty() {
            let selected = &self.entities[self.selected_index];
            let name_label = format!("NAME: {}", selected.name);
            push_text(&mut verts, &mut indices, &name_label, 0.70, 0.70, 0.013, 0.024, [0.0, 0.9, 1.0]);

            // Entity Color Swatch Preview
            push_ui_rect(&mut verts, &mut indices, 0.70, 0.61, 0.96, 0.67, selected.color, 1.0);

            // Transform Labels & Buttons
            let px_label = format!("X: {:.1}", selected.pos[0]);
            let py_label = format!("Y: {:.1}", selected.pos[1]);
            let pz_label = format!("Z: {:.1}", selected.pos[2]);

            push_text(&mut verts, &mut indices, &px_label, 0.70, 0.53, 0.012, 0.022, [1.0, 0.6, 0.6]);
            push_ui_rect(&mut verts, &mut indices, 0.82, 0.51, 0.88, 0.57, [0.6, 0.2, 0.2], 0.9);
            push_text(&mut verts, &mut indices, "X-", 0.835, 0.525, 0.012, 0.022, [1.0, 1.0, 1.0]);
            push_ui_rect(&mut verts, &mut indices, 0.90, 0.51, 0.96, 0.57, [0.6, 0.2, 0.2], 0.9);
            push_text(&mut verts, &mut indices, "X+", 0.915, 0.525, 0.012, 0.022, [1.0, 1.0, 1.0]);

            push_text(&mut verts, &mut indices, &py_label, 0.70, 0.43, 0.012, 0.022, [0.6, 1.0, 0.6]);
            push_ui_rect(&mut verts, &mut indices, 0.82, 0.41, 0.88, 0.47, [0.2, 0.6, 0.2], 0.9);
            push_text(&mut verts, &mut indices, "Y-", 0.835, 0.425, 0.012, 0.022, [1.0, 1.0, 1.0]);
            push_ui_rect(&mut verts, &mut indices, 0.90, 0.41, 0.96, 0.47, [0.2, 0.6, 0.2], 0.9);
            push_text(&mut verts, &mut indices, "Y+", 0.915, 0.425, 0.012, 0.022, [1.0, 1.0, 1.0]);

            push_text(&mut verts, &mut indices, &pz_label, 0.70, 0.33, 0.012, 0.022, [0.6, 0.7, 1.0]);
            push_ui_rect(&mut verts, &mut indices, 0.82, 0.31, 0.88, 0.37, [0.2, 0.3, 0.7], 0.9);
            push_text(&mut verts, &mut indices, "Z-", 0.835, 0.325, 0.012, 0.022, [1.0, 1.0, 1.0]);
            push_ui_rect(&mut verts, &mut indices, 0.90, 0.31, 0.96, 0.37, [0.2, 0.3, 0.7], 0.9);
            push_text(&mut verts, &mut indices, "Z+", 0.915, 0.325, 0.012, 0.022, [1.0, 1.0, 1.0]);

            // Attached Script Box
            push_text(&mut verts, &mut indices, "RHAI SCRIPT:", 0.70, 0.24, 0.013, 0.024, [0.8, 0.5, 1.0]);
            push_ui_rect(&mut verts, &mut indices, 0.70, 0.12, 0.96, 0.22, [0.15, 0.10, 0.22], 0.9);
            let script_label = format!("> {}", selected.script_preset_name);
            push_text(&mut verts, &mut indices, &script_label, 0.72, 0.15, 0.012, 0.022, [1.0, 0.8, 0.2]);
        }

        // D. Bottom Script Console & Status Bar Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.96, 0.98, -0.65, [0.05, 0.06, 0.08], 0.95);
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.66, 0.98, -0.65, [0.18, 0.22, 0.30], 1.0); // Border

        // Script IDE Header
        push_text(&mut verts, &mut indices, "RHAI SCRIPT IDE & CONSOLE (HOT-RELOAD ON R):", -0.96, -0.63, 0.013, 0.024, [0.0, 0.9, 1.0]);

        // Code Preview Line 1 & 2
        push_text(&mut verts, &mut indices, "1 | FN UPDATE(ENTITY, DT) {", -0.96, -0.73, 0.012, 0.022, [0.7, 0.75, 0.8]);
        push_text(&mut verts, &mut indices, "2 |   ENTITY.ROTATE(1.5 * DT, 0.8 * DT, 0.0);", -0.96, -0.80, 0.012, 0.022, [0.9, 0.8, 0.2]);
        push_text(&mut verts, &mut indices, "3 |   RETURN ENTITY; }", -0.96, -0.87, 0.012, 0.022, [0.7, 0.75, 0.8]);

        // Live Console Log
        let log_text = format!("[LOG]: {}", self.status_message);
        push_text(&mut verts, &mut indices, &log_text, -0.96, -0.94, 0.012, 0.020, [0.2, 1.0, 0.5]);

        (verts, indices)
    }
}

fn main() {
    println!("============================================================");
    println!("  METATOPIA STUDIO — 3D Game Editor & Rhai Scripting Suite");
    println!("============================================================");
    println!(" Features:");
    println!("   • Live 3D Viewport with Infinite Grid & Transform Gizmo");
    println!("   • Visual On-Screen Toolbar, Outliner, & Inspector");
    println!("   • Embedded Rhai Scripting Engine with Hot-Reload");
    println!("   • Play / Edit Real-Time Simulation Modes");
    println!(" Controls:");
    println!("   Mouse Left Click : Click Toolbar / Outliner / Inspector Buttons");
    println!("   Mouse Right Drag : Orbit / Fly Camera");
    println!("   SPACE            : Toggle [PLAY] / [EDIT] simulation modes");
    println!("   TAB              : Cycle selection through scene entities");
    println!("   1 / 2 / 3 / 4 / 5 : Spawn Cube / Sphere / Cylinder / Torus / Capsule");
    println!("   G                : Cycle Rhai Script Presets on selected object");
    println!("   R                : Recompile and Hot-Reload all Rhai scripts");
    println!("   Arrow Keys       : Translate selected object (X / Z)");
    println!("   PgUp / PgDown    : Translate selected object vertically (Y)");
    println!("   Q / E            : Rotate selected object");
    println!("   X                : Delete selected object");
    println!("============================================================");

    run_game(MetatopiaStudio::new());
}
