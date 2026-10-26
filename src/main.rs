//! Metatopia Game Engine v0.2.0 — Native macOS / Cross-Platform Executable
//!
//! Unified 3D Non-Euclidean & 4D Relativistic Game Engine Studio:
//! - Interactive 3D Scene Viewport with infinite reference grid & XYZ transform gizmo
//! - On-screen visual Studio UI (Top Toolbar, Scene Outliner, Property Inspector, Rhai Script IDE)
//! - Non-Euclidean Riemannian Manifolds (H^3, E^3, S^3, K^3), Portal Pairs & Geodesic Raycasting
//! - 4D Lorentzian Spacetime: Schwarzschild Black Hole (photon sphere 1.5 r_s) & Morris-Thorne Wormhole
//! - Authoritative Cross-Portal Geodesic Weapon Raycast & Multiplayer Snapshot Replication
//! - Embedded Rhai Scripting Engine with live hot-reload (`KeyR`)

use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::manifold::{ChartId, GeometryType, Manifold, ManifoldRaycaster, PortalShape};
use metatopia_engine::net::protocol::EntityState;
use metatopia_engine::net::server::NetServer;
use metatopia_engine::particles::ParticleSystem;
use metatopia_engine::physics::PhysicsWorld;
use metatopia_engine::quickstart::*;
use metatopia_engine::relativity::{MorrisThorneWormhole, SchwarzschildBlackHole, WormholeState};
use metatopia_engine::scripting::{ScriptEngine, ScriptEntityState};
use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

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

/// An editable 3D entity in the engine
#[derive(Debug, Clone)]
pub struct EditorEntity {
    pub id: u32,
    pub name: String,
    pub mesh_type: MeshType,
    pub pos: [f32; 3],
    pub rot: [f32; 3],
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
    pub fn new(
        id: u32,
        name: impl Into<String>,
        mesh_type: MeshType,
        pos: [f32; 3],
        color: [f32; 3],
    ) -> Self {
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

/// Unified Metatopia Game Engine Application
pub struct MetatopiaEngineApp {
    pub entities: Vec<EditorEntity>,
    pub next_entity_id: u32,
    pub selected_index: usize,
    pub mode: EditorMode,
    pub manifold: Manifold,
    pub black_hole: SchwarzschildBlackHole,
    pub wormhole: MorrisThorneWormhole,
    pub net_server: Option<NetServer>,
    pub sentinel_hp: f32,
    pub last_laser_segments: Vec<([f32; 3], [f32; 3])>,
    pub script_engine: ScriptEngine,
    pub physics_world: PhysicsWorld,
    pub particle_system: ParticleSystem,
    pub status_message: String,
    pub status_timer: f32,
    pub current_script_preset: usize,
    pub camera_initialized: bool,
}

impl MetatopiaEngineApp {
    pub fn new(start_in_play_mode: bool) -> Self {
        let mut script_engine = ScriptEngine::new();
        let mut entities = Vec::new();

        // 1. Central Rotating Cyber Core
        let mut core = EditorEntity::new(1, "HyperCore", MeshType::Torus, [0.0, 2.5, 0.0], [0.0, 0.9, 1.0]);
        core.scale = [1.4, 1.4, 1.4];
        core.emissive = 2.0;
        core.script_preset_name = "Orbit & Pulse".into();
        core.script_source = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt;
                set_var(e, "t", t);
                e.rotate(1.5 * dt, 0.8 * dt, 0.0);
                e.set_pos(0.0, 2.5 + sin(t * 2.5) * 0.5, 0.0);
                e.set_emissive(1.8 + sin(t * 4.0) * 1.2);
                return e;
            }
        "#
        .into();
        core.script_ast = script_engine.compile(&core.script_source).ok();
        entities.push(core);

        // 2. Orbiting Satellite Sphere
        let mut sat = EditorEntity::new(2, "ChronoOrb", MeshType::Sphere, [4.0, 2.0, 0.0], [1.0, 0.2, 0.6]);
        sat.scale = [0.9, 0.9, 0.9];
        sat.emissive = 1.6;
        sat.script_preset_name = "Circular Orbit".into();
        sat.script_source = r#"
            fn update(e, dt) {
                let angle = get_var(e, "angle") + 1.8 * dt;
                set_var(e, "angle", angle);
                e.set_pos(cos(angle) * 4.5, 2.0 + sin(angle * 2.0) * 0.6, sin(angle) * 4.5);
                return e;
            }
        "#
        .into();
        sat.script_ast = script_engine.compile(&sat.script_source).ok();
        entities.push(sat);

        // 3. Guardian Pillar Left
        let mut pillar_l = EditorEntity::new(3, "PillarLeft", MeshType::Cylinder, [-5.0, 2.0, -4.0], [0.2, 0.85, 0.5]);
        pillar_l.scale = [0.8, 2.5, 0.8];
        entities.push(pillar_l);

        // 4. Guardian Pillar Right
        let mut pillar_r = EditorEntity::new(4, "PillarRight", MeshType::Cylinder, [5.0, 2.0, -4.0], [0.2, 0.85, 0.5]);
        pillar_r.scale = [0.8, 2.5, 0.8];
        entities.push(pillar_r);

        // 5. Schwarzschild Black Hole Core Entity
        let mut bh_entity = EditorEntity::new(5, "SchwarzschildBH", MeshType::Sphere, [0.0, 3.2, -7.5], [0.95, 0.45, 0.08]);
        bh_entity.scale = [1.5, 1.5, 1.5];
        bh_entity.emissive = 3.2;
        bh_entity.script_preset_name = "Relativistic Lens".into();
        bh_entity.script_source = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt;
                set_var(e, "t", t);
                e.rotate(2.4 * dt, 0.0, 0.0);
                e.set_emissive(2.8 + sin(t * 5.0) * 1.0);
                return e;
            }
        "#
        .into();
        bh_entity.script_ast = script_engine.compile(&bh_entity.script_source).ok();
        entities.push(bh_entity);

        // Build non-Euclidean manifold with Charts 0 (E^3) and 1 (H^3, K=-1.0)
        let mut manifold = Manifold::new();
        let _chart_1 = manifold.add_chart(GeometryType::Hyperbolic);

        let black_hole = SchwarzschildBlackHole::new(Point3::new(0.0, 3.2, -7.5), 1.6);
        let wormhole = MorrisThorneWormhole::new(
            ChartId(0),
            ChartId(1),
            Point3::new(-4.0, 2.2, -2.5),
            Point3::new(4.0, 2.2, 2.5),
            1.2,
        );

        let mut net_server = NetServer::bind("127.0.0.1:0", 60).ok();
        if let Some(ref mut srv) = net_server {
            srv.entities.insert(
                99,
                EntityState {
                    id: 99,
                    owner_id: 99,
                    pos: [0.0, 2.0, -9.0],
                    vel: [0.0, 0.0, 0.0],
                    yaw: 0.0,
                    pitch: 0.0,
                    health: 100.0,
                    chart: 1,
                },
            );
        }

        let mut app = Self {
            entities,
            next_entity_id: 6,
            selected_index: 0,
            mode: if start_in_play_mode { EditorMode::Play } else { EditorMode::Edit },
            manifold,
            black_hole,
            wormhole,
            net_server,
            sentinel_hp: 100.0,
            last_laser_segments: Vec::new(),
            script_engine,
            physics_world: PhysicsWorld::new(),
            particle_system: ParticleSystem::new(400),
            status_message: "Metatopia Engine v0.2.0 Ready. [SPACE] Play/Edit | [P] Spawn Portal | [K] Curvature | [F] Fire Geodesic Laser".into(),
            status_timer: 10.0,
            current_script_preset: 0,
            camera_initialized: false,
        };

        // Pre-spawn a linked Non-Euclidean Portal Pair connecting Chart 0 <-> Chart 1
        app.spawn_portal_pair();
        app
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        println!("[Metatopia Engine]: {}", s);
        self.status_message = s;
        self.status_timer = 6.0;
    }

    pub fn spawn_entity(&mut self, mesh_type: MeshType, prefix: &str, color: [f32; 3]) {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        let offset_x = ((id as f32 * 2.3).sin()) * 4.0;
        let offset_z = ((id as f32 * 1.7).cos()) * 4.0;
        let name = format!("{}_{}", prefix, id);
        let entity = EditorEntity::new(id, name.clone(), mesh_type, [offset_x, 1.5, offset_z], color);
        self.entities.push(entity);
        self.selected_index = self.entities.len() - 1;
        self.set_status(format!("Spawned '{}' (ID #{}) at [{:.1}, 1.5, {:.1}]", name, id, offset_x, offset_z));
    }

    pub fn spawn_portal_pair(&mut self) {
        let id_a = self.next_entity_id;
        let id_b = self.next_entity_id + 1;
        self.next_entity_id += 2;

        let pos_a = [0.0, 2.0, -3.0];
        let pos_b = [0.0, 2.0, -3.0];

        let portal_script = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt * 2.0;
                set_var(e, "t", t);
                e.set_curvature(-1.0 + sin(t) * 0.25);
                e.set_emissive(2.5 + sin(t * 2.0) * 1.0);
                e.rotate(0.0, 0.6 * dt, 0.0);
                return e;
            }
        "#;

        let mut portal_a = EditorEntity::new(id_a, format!("Portal_A_{}", id_a), MeshType::Torus, [-3.5, 2.0, -3.0], [0.0, 0.95, 1.0]);
        portal_a.scale = [1.5, 1.5, 0.35];
        portal_a.emissive = 3.0;
        portal_a.script_state.chart_id = 0;
        portal_a.script_state.curvature_k = 0.0;
        portal_a.script_state.portal_target_chart = Some(1);
        portal_a.script_preset_name = "Non-Euclidean Portal".into();
        portal_a.script_source = portal_script.to_string();
        portal_a.script_ast = self.script_engine.compile(portal_script).ok();

        let mut portal_b = EditorEntity::new(id_b, format!("Portal_B_{}", id_b), MeshType::Torus, [3.5, 2.0, 3.0], [1.0, 0.2, 0.85]);
        portal_b.scale = [1.5, 1.5, 0.35];
        portal_b.emissive = 3.0;
        portal_b.script_state.chart_id = 1;
        portal_b.script_state.curvature_k = -1.0;
        portal_b.script_state.portal_target_chart = Some(0);
        portal_b.script_preset_name = "Non-Euclidean Portal".into();
        portal_b.script_source = portal_script.to_string();
        portal_b.script_ast = self.script_engine.compile(portal_script).ok();

        let _ = self.manifold.create_portal_with_bounds(
            ChartId(0),
            ChartId(1),
            Point3::new(pos_a[0], pos_a[1], pos_a[2]),
            Point3::new(pos_b[0], pos_b[1], pos_b[2]),
            Matrix4::identity(),
            Vector3::new(0.0, 0.0, 1.0),
            4.5,
            4.5,
            PortalShape::Circular,
        );

        self.entities.push(portal_a);
        self.entities.push(portal_b);
        self.selected_index = self.entities.len() - 1;
        self.set_status(format!(
            "Spawned Linked Portal Pair #{} <-> #{} (Charts 0 <-> 1, K=-1.0)",
            id_a, id_b
        ));
    }

    pub fn fire_cross_portal_laser(&mut self) {
        let origin = Point3::new(0.0, 2.0, 1.0);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let trace = ManifoldRaycaster::cast_geodesic_ray(
            &self.manifold,
            origin,
            dir,
            ChartId(0),
            20.0,
            0.20,
            4,
        );
        self.last_laser_segments.clear();
        for seg in &trace.segments {
            self.last_laser_segments.push((
                [seg.start.x, seg.start.y, seg.start.z],
                [seg.end.x, seg.end.y, seg.end.z],
            ));
        }

        if let Some(ref mut srv) = self.net_server {
            if let Some(hit) = srv.resolve_cross_portal_shot(
                &self.manifold,
                1,
                origin,
                dir,
                ChartId(0),
                20.0,
                4,
                35.0,
                2.5,
            ) {
                self.sentinel_hp = hit.target_remaining_health;
                self.particle_system.burst(
                    Vector3::new(hit.hit_point.x, hit.hit_point.y, hit.hit_point.z),
                    24,
                    [0.0, 0.98, 1.0, 1.0],
                    [1.0, 0.2, 0.85, 0.0],
                    5.5,
                    0.8,
                );
                self.set_status(format!(
                    "CROSS-PORTAL HIT! Chart {:?} | Portals Crossed: {} | Sentinel HP: {:.0}",
                    hit.hit_chart, hit.portals_crossed, self.sentinel_hp
                ));
                return;
            }
        }
        self.set_status(format!(
            "Geodesic Laser Fired: {} segments, {} portal jumps",
            trace.segments.len(),
            trace.portal_jumps
        ));
    }

    pub fn cycle_selected_curvature(&mut self) {
        if self.entities.is_empty() {
            return;
        }
        let entity = &mut self.entities[self.selected_index];
        let next_k = if entity.script_state.curvature_k < -0.5 {
            0.0
        } else if entity.script_state.curvature_k < 0.5 {
            1.0
        } else {
            -1.0
        };
        entity.script_state.curvature_k = next_k;
        let geom_label = if next_k < -0.5 {
            "Hyperbolic H^3 (K=-1)"
        } else if next_k > 0.5 {
            "Spherical S^3 (K=+1)"
        } else {
            "Euclidean E^3 (K=0)"
        };
        let ename = entity.name.clone();
        self.set_status(format!("Entity '{}' Curvature -> {}", ename, geom_label));
    }

    pub fn cycle_script_preset(&mut self) {
        if self.entities.is_empty() {
            return;
        }
        self.current_script_preset = (self.current_script_preset + 1) % 5;
        let entity = &mut self.entities[self.selected_index];

        let (name, src) = match self.current_script_preset {
            0 => (
                "Pulsing Glow",
                r#"
                fn update(e, dt) {
                    e.rotate(1.5 * dt, 0.8 * dt, 0.0);
                    let t = get_var(e, "t") + dt * 4.0;
                    set_var(e, "t", t);
                    e.set_emissive(sin(t) * 3.0 + 3.0);
                    return e;
                }
            "#,
            ),
            1 => (
                "Hover & Bob",
                r#"
                fn update(e, dt) {
                    e.rotate(0.0, 2.5 * dt, 0.0);
                    let t = get_var(e, "t") + dt * 2.5;
                    set_var(e, "t", t);
                    e.move(0.0, cos(t) * 2.0 * dt, 0.0);
                    return e;
                }
            "#,
            ),
            2 => (
                "Color Cycle",
                r#"
                fn update(e, dt) {
                    let t = get_var(e, "t") + dt * 2.0;
                    set_var(e, "t", t);
                    e.set_color(sin(t) * 0.5 + 0.5, cos(t) * 0.5 + 0.5, sin(t * 0.5) * 0.5 + 0.5);
                    e.rotate(1.0 * dt, 1.0 * dt, 0.0);
                    return e;
                }
            "#,
            ),
            3 => (
                "Non-Euclidean Portal",
                r#"
                fn update(e, dt) {
                    let t = get_var(e, "t") + dt * 2.0;
                    set_var(e, "t", t);
                    e.set_curvature(-1.0 + sin(t) * 0.35);
                    e.mobius_step(0.05 * dt * cos(t), 0.0, 0.05 * dt * sin(t));
                    e.set_emissive(2.2 + sin(t * 3.0) * 1.0);
                    return e;
                }
            "#,
            ),
            _ => ("None", "fn update(e, dt) { return e; }"),
        };

        entity.script_preset_name = name.into();
        entity.script_source = src.into();
        entity.script_ast = self.script_engine.compile(src).ok();
        let msg = format!(
            "Assigned Script Preset '{}' to '{}'",
            entity.script_preset_name, entity.name
        );
        self.set_status(msg);
    }

    fn pixel_to_ndc(&self, px: f32, py: f32, width: u32, height: u32) -> (f32, f32) {
        let ndc_x = (px / width.max(1) as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height.max(1) as f32) * 2.0;
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
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        _ => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}

fn push_ui_rect(
    verts: &mut Vec<GameVertex>,
    indices: &mut Vec<u32>,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    color: [f32; 3],
    opacity: f32,
) {
    let base_idx = verts.len() as u32;
    let pbr = [0.0, 0.0, opacity, -1.0]; // w = -1.0 flags 2D screen UI
    let norm = [0.0, 0.0, 1.0];

    verts.push(GameVertex::new([min_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, max_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([min_x, max_y, 0.0], norm, color, pbr));

    indices.extend_from_slice(&[
        base_idx,
        base_idx + 1,
        base_idx + 2,
        base_idx,
        base_idx + 2,
        base_idx + 3,
    ]);
}

fn push_text(
    verts: &mut Vec<GameVertex>,
    indices: &mut Vec<u32>,
    text: &str,
    start_x: f32,
    start_y: f32,
    char_w: f32,
    char_h: f32,
    color: [f32; 3],
) {
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

fn push_3d_line(
    verts: &mut Vec<GameVertex>,
    indices: &mut Vec<u32>,
    p0: [f32; 3],
    p1: [f32; 3],
    thickness: f32,
    color: [f32; 3],
    emissive: f32,
) {
    let base = verts.len() as u32;
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dz = p1[2] - p0[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt().max(0.0001);

    let (ox, oy, oz) = if (dy / len).abs() > 0.9 {
        (thickness, 0.0, 0.0)
    } else {
        (0.0, thickness, 0.0)
    };

    let n = [0.0, 1.0, 0.0];
    let pbr = [0.0, 0.2, emissive, 0.0];
    verts.push(GameVertex::new([p0[0] - ox, p0[1] - oy, p0[2] - oz], n, color, pbr));
    verts.push(GameVertex::new([p0[0] + ox, p0[1] + oy, p0[2] + oz], n, color, pbr));
    verts.push(GameVertex::new([p1[0] + ox, p1[1] + oy, p1[2] + oz], n, color, pbr));
    verts.push(GameVertex::new([p1[0] - ox, p1[1] - oy, p1[2] - oz], n, color, pbr));
    indices.extend_from_slice(&[
        base, base + 1, base + 2, base, base + 2, base + 3,
        base + 2, base + 1, base, base + 3, base + 2, base,
    ]);
}

impl GameApp for MetatopiaEngineApp {
    fn title(&self) -> &str {
        "Metatopia Engine v0.2.0 — macOS Non-Euclidean & 4D Relativistic Studio"
    }

    fn window_size(&self) -> (u32, u32) {
        (1440, 900)
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    fn is_dynamic_mesh(&self) -> bool {
        true
    }

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

        if self.status_timer > 0.0 {
            self.status_timer -= ctx.dt;
        }

        // 1. Camera Fly Movement
        if ctx.mouse_held(winit::event::MouseButton::Right) {
            ctx.default_camera_movement();
        } else {
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
                    self.set_status("PLAY MODE ACTIVATED (Scripts, Relativity & Cross-Portal Combat Live)");
                    EditorMode::Play
                }
                EditorMode::Play => {
                    self.set_status("EDIT MODE ACTIVATED (Inspector & Transform Gizmos Active)");
                    EditorMode::Edit
                }
            };
        }

        // 3. Mouse Click on UI Buttons & Outliner
        if ctx.mouse_pressed(winit::event::MouseButton::Left) {
            let (m_ndc_x, m_ndc_y) = self.pixel_to_ndc(
                ctx.mouse_pos().0,
                ctx.mouse_pos().1,
                ctx.resolution.0,
                ctx.resolution.1,
            );

            // Top Bar Buttons
            if m_ndc_y >= 0.85 && m_ndc_y <= 0.97 {
                if m_ndc_x >= -0.68 && m_ndc_x <= -0.54 {
                    self.mode = match self.mode {
                        EditorMode::Edit => {
                            self.set_status("PLAY MODE ACTIVATED");
                            EditorMode::Play
                        }
                        EditorMode::Play => {
                            self.set_status("EDIT MODE ACTIVATED");
                            EditorMode::Edit
                        }
                    };
                } else if m_ndc_x >= -0.52 && m_ndc_x <= -0.42 {
                    self.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.8, 1.0]);
                } else if m_ndc_x >= -0.40 && m_ndc_x <= -0.28 {
                    self.spawn_entity(MeshType::Sphere, "NeonSphere", [1.0, 0.2, 0.5]);
                } else if m_ndc_x >= -0.26 && m_ndc_x <= -0.16 {
                    self.spawn_entity(MeshType::Cylinder, "PowerPillar", [1.0, 0.9, 0.1]);
                } else if m_ndc_x >= -0.14 && m_ndc_x <= -0.04 {
                    self.spawn_entity(MeshType::Torus, "GravityRing", [0.8, 0.2, 1.0]);
                } else if m_ndc_x >= -0.02 && m_ndc_x <= 0.08 {
                    self.spawn_entity(MeshType::Capsule, "BioCapsule", [0.1, 1.0, 0.7]);
                } else if m_ndc_x >= 0.12 && m_ndc_x <= 0.28 {
                    self.cycle_script_preset();
                } else if m_ndc_x >= 0.30 && m_ndc_x <= 0.44 {
                    for entity in &mut self.entities {
                        if !entity.script_source.is_empty() {
                            if let Ok(ast) = self.script_engine.compile(&entity.script_source) {
                                entity.script_ast = Some(ast);
                            }
                        }
                    }
                    self.set_status("All Rhai Scripts Hot-Reloaded.");
                } else if m_ndc_x >= 0.46 && m_ndc_x <= 0.58 && !self.entities.is_empty() {
                    let removed = self.entities.remove(self.selected_index);
                    self.set_status(format!("Deleted Entity: '{}'", removed.name));
                    if self.selected_index >= self.entities.len() && !self.entities.is_empty() {
                        self.selected_index = self.entities.len() - 1;
                    }
                } else if m_ndc_x >= 0.60 && m_ndc_x <= 0.76 {
                    self.spawn_portal_pair();
                } else if m_ndc_x >= 0.78 && m_ndc_x <= 0.96 {
                    self.fire_cross_portal_laser();
                }
            }

            // Left Panel Scene Outliner Selection
            if m_ndc_x >= -0.98 && m_ndc_x <= -0.66 && m_ndc_y <= 0.72 && m_ndc_y >= -0.60 {
                let row_height = 0.085;
                let top_y = 0.70;
                let clicked_idx = ((top_y - m_ndc_y) / row_height).floor() as isize;
                if clicked_idx >= 0 && (clicked_idx as usize) < self.entities.len() {
                    self.selected_index = clicked_idx as usize;
                    let name = self.entities[self.selected_index].name.clone();
                    let id = self.entities[self.selected_index].id;
                    self.set_status(format!("Selected Entity #{} '{}'", id, name));
                }
            }
        }

        // 4. Keyboard Shortcuts
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
        if ctx.key_pressed(VirtualKey::KeyP) {
            self.spawn_portal_pair();
        }
        if ctx.key_pressed(VirtualKey::KeyK) {
            self.cycle_selected_curvature();
        }
        if ctx.key_pressed(VirtualKey::KeyF) {
            self.fire_cross_portal_laser();
        }
        if ctx.key_pressed(VirtualKey::Tab) && !self.entities.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.entities.len();
        }
        if ctx.key_pressed(VirtualKey::KeyC) {
            self.cycle_script_preset();
        }
        if ctx.key_pressed(VirtualKey::KeyR) {
            for entity in &mut self.entities {
                if !entity.script_source.is_empty() {
                    if let Ok(ast) = self.script_engine.compile(&entity.script_source) {
                        entity.script_ast = Some(ast);
                    }
                }
            }
            self.set_status("Hot-Reloaded All Rhai Scripts.");
        }
        if ctx.key_pressed(VirtualKey::Backspace) && !self.entities.is_empty() {
            let removed = self.entities.remove(self.selected_index);
            self.set_status(format!("Deleted Entity '{}'", removed.name));
            if self.selected_index >= self.entities.len() && !self.entities.is_empty() {
                self.selected_index = self.entities.len() - 1;
            }
        }

        // 5. Arrow Keys Object Movement
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

        // 1. Grid Floor Plane (100x100)
        let (plane_v, plane_i) = ProceduralMesh::plane(100.0, 100.0, 20, 20, [0.15, 0.18, 0.22]);
        let base = verts.len() as u32;
        for v in plane_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.1, 0.8, 0.0, 0.0]));
        }
        for idx in plane_i {
            indices.push(base + idx);
        }

        // Render Geodesic Laser Segments if fired
        for &(p0, p1) in &self.last_laser_segments {
            push_3d_line(&mut verts, &mut indices, p0, p1, 0.08, [0.0, 0.98, 1.0], 3.5);
        }

        // 2. Render All Scene Entities
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
                MeshType::Sphere => ProceduralMesh::capsule(entity.scale[0] * 0.5, 0.01, 8, 16, entity.color),
                MeshType::Cylinder => ProceduralMesh::cylinder(entity.scale[0] * 0.5, entity.scale[1], 16, entity.color),
                MeshType::Torus => ProceduralMesh::torus(entity.scale[0] * 0.8, entity.scale[0] * 0.25, 24, 12, entity.color),
                MeshType::Capsule => ProceduralMesh::capsule(entity.scale[0] * 0.4, entity.scale[1], 8, 16, entity.color),
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

            let (x_v, x_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [1.0, 0.1, 0.1]);
            let x_base = verts.len() as u32;
            for v in x_v {
                let pos = [ep[0] + v.position[1] + g_len * 0.5, ep[1] + v.position[0], ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [1.0, 0.0, 0.0], [1.0, 0.1, 0.1], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in x_i { indices.push(x_base + idx); }

            let (y_v, y_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [0.1, 1.0, 0.2]);
            let y_base = verts.len() as u32;
            for v in y_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[1] + g_len * 0.5, ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [0.0, 1.0, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in y_i { indices.push(y_base + idx); }

            let (z_v, z_i) = ProceduralMesh::cylinder(g_rad, g_len, 8, [0.1, 0.4, 1.0]);
            let z_base = verts.len() as u32;
            for v in z_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[2], ep[2] + v.position[1] + g_len * 0.5];
                verts.push(GameVertex::new(pos, [0.0, 0.0, 1.0], [0.1, 0.4, 1.0], [0.0, 0.2, 2.0, 0.0]));
            }
            for idx in z_i { indices.push(z_base + idx); }
        }

        // 3. Render Particle System in Play Mode
        if self.mode == EditorMode::Play {
            let (p_verts, p_indices) = self.particle_system.build_mesh();
            let base = verts.len() as u32;
            for pv in p_verts {
                verts.push(pv);
            }
            for pi in p_indices {
                indices.push(base + pi);
            }
        }

        // 4. 2D On-Screen Studio UI Overlay
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.84, 1.0, 1.0, [0.06, 0.07, 0.10], 0.94);
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.835, 1.0, 0.845, [0.0, 0.85, 1.0], 1.0);

        push_text(&mut verts, &mut indices, "METATOPIA ENGINE", -0.97, 0.91, 0.015, 0.032, [0.0, 0.95, 1.0]);

        let (mode_label, mode_col) = match self.mode {
            EditorMode::Edit => ("PLAY", [0.15, 0.75, 0.35]),
            EditorMode::Play => ("EDIT", [0.90, 0.35, 0.15]),
        };
        push_ui_rect(&mut verts, &mut indices, -0.68, 0.86, -0.54, 0.97, mode_col, 0.95);
        push_text(&mut verts, &mut indices, mode_label, -0.65, 0.90, 0.014, 0.028, [1.0, 1.0, 1.0]);

        let buttons = [
            (-0.52, -0.42, "+CUBE", [0.16, 0.22, 0.32]),
            (-0.40, -0.28, "+SPHERE", [0.16, 0.22, 0.32]),
            (-0.26, -0.16, "+CYL", [0.16, 0.22, 0.32]),
            (-0.14, -0.04, "+TORUS", [0.16, 0.22, 0.32]),
            (-0.02, 0.08, "+CAPS", [0.16, 0.22, 0.32]),
            (0.12, 0.28, "SCRIPT(C)", [0.28, 0.18, 0.42]),
            (0.30, 0.44, "RELOAD(R)", [0.18, 0.36, 0.28]),
            (0.46, 0.58, "DEL(BS)", [0.45, 0.16, 0.18]),
            (0.60, 0.76, "+PORTAL(P)", [0.05, 0.42, 0.52]),
            (0.78, 0.96, "LASER(F)", [0.55, 0.12, 0.42]),
        ];
        for (bx0, bx1, label, bcol) in buttons {
            push_ui_rect(&mut verts, &mut indices, bx0, 0.86, bx1, 0.97, bcol, 0.92);
            push_text(&mut verts, &mut indices, label, bx0 + 0.012, 0.90, 0.011, 0.024, [0.92, 0.96, 1.0]);
        }

        // Left Scene Outliner Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.62, -0.65, 0.81, [0.07, 0.08, 0.12], 0.88);
        push_ui_rect(&mut verts, &mut indices, -0.655, -0.62, -0.65, 0.81, [0.2, 0.25, 0.35], 1.0);
        push_text(&mut verts, &mut indices, "SCENE OUTLINER", -0.95, 0.75, 0.014, 0.026, [0.0, 0.9, 1.0]);

        for (i, entity) in self.entities.iter().enumerate().take(14) {
            let row_y = 0.66 - (i as f32) * 0.085;
            let is_sel = i == self.selected_index;
            if is_sel {
                push_ui_rect(&mut verts, &mut indices, -0.96, row_y - 0.02, -0.67, row_y + 0.04, [0.15, 0.38, 0.55], 0.92);
            }
            let txt_col = if is_sel { [1.0, 1.0, 1.0] } else { [0.75, 0.80, 0.88] };
            let item_label = format!("{}:{}", entity.id, entity.name);
            push_text(&mut verts, &mut indices, &item_label, -0.94, row_y, 0.012, 0.022, txt_col);
        }

        // Right Property Inspector + 4D Relativity Panel
        push_ui_rect(&mut verts, &mut indices, 0.65, -0.62, 0.98, 0.81, [0.07, 0.08, 0.12], 0.88);
        push_ui_rect(&mut verts, &mut indices, 0.65, -0.62, 0.655, 0.81, [0.2, 0.25, 0.35], 1.0);
        push_text(&mut verts, &mut indices, "INSPECTOR", 0.68, 0.75, 0.014, 0.026, [1.0, 0.75, 0.2]);

        if !self.entities.is_empty() {
            let selected = &self.entities[self.selected_index];
            push_text(&mut verts, &mut indices, &format!("NAME: {}", selected.name), 0.68, 0.65, 0.012, 0.022, [0.95, 0.95, 1.0]);
            push_text(&mut verts, &mut indices, "TRANSFORM (POS):", 0.68, 0.55, 0.012, 0.022, [0.5, 0.8, 1.0]);
            push_text(&mut verts, &mut indices, &format!("X: {:.2}", selected.pos[0]), 0.70, 0.48, 0.012, 0.022, [1.0, 0.4, 0.4]);
            push_text(&mut verts, &mut indices, &format!("Y: {:.2}", selected.pos[1]), 0.70, 0.41, 0.012, 0.022, [0.4, 1.0, 0.5]);
            push_text(&mut verts, &mut indices, &format!("Z: {:.2}", selected.pos[2]), 0.70, 0.34, 0.012, 0.022, [0.4, 0.7, 1.0]);

            push_text(&mut verts, &mut indices, &format!("SCALE: {:.2}", selected.scale[0]), 0.68, 0.24, 0.012, 0.022, [0.85, 0.85, 0.9]);
            push_text(&mut verts, &mut indices, &format!("YAW:   {:.2}", selected.rot[0]), 0.68, 0.17, 0.012, 0.022, [0.85, 0.85, 0.9]);

            push_text(&mut verts, &mut indices, "SCRIPT PRESET:", 0.68, 0.07, 0.012, 0.022, [0.8, 0.5, 1.0]);
            push_text(&mut verts, &mut indices, &selected.script_preset_name, 0.68, 0.00, 0.012, 0.022, [0.3, 1.0, 0.7]);

            push_text(&mut verts, &mut indices, "MANIFOLD & 4D GR:", 0.68, -0.10, 0.012, 0.022, [0.0, 0.95, 1.0]);
            let manifold_label = format!(
                "CHART: {} | K: {:.1}",
                selected.script_state.chart_id, selected.script_state.curvature_k
            );
            push_text(&mut verts, &mut indices, &manifold_label, 0.68, -0.17, 0.011, 0.020, [0.85, 0.95, 1.0]);
            let gr_label = format!(
                "BH R_PH: {:.1} | BOSS HP: {:.0}",
                self.black_hole.photon_sphere_radius(),
                self.sentinel_hp
            );
            push_text(&mut verts, &mut indices, &gr_label, 0.68, -0.24, 0.011, 0.020, [1.0, 0.45, 0.85]);
        }

        // Bottom Script Console & Status Bar Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.96, 0.98, -0.65, [0.05, 0.06, 0.08], 0.95);
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.66, 0.98, -0.65, [0.18, 0.22, 0.30], 1.0);

        push_text(&mut verts, &mut indices, "RHAI SCRIPT IDE & CONSOLE (HOT-RELOAD ON R | LASER ON F):", -0.96, -0.71, 0.013, 0.024, [0.0, 0.9, 1.0]);
        push_text(&mut verts, &mut indices, "1 | FN UPDATE(E, DT) { E.SET_CURVATURE(-1.0); E.MOBIUS_STEP(0.05*DT, 0.0, 0.0); RETURN E; }", -0.96, -0.79, 0.011, 0.021, [0.9, 0.8, 0.2]);
        push_text(&mut verts, &mut indices, "2 | CONTROLS: [SPACE] PLAY/EDIT | [WASD+RMB] FLY | [P] +PORTAL | [K] CURVATURE | [F] FIRE LASER", -0.96, -0.87, 0.011, 0.021, [0.7, 0.78, 0.88]);

        let log_text = format!("[LOG]: {}", self.status_message);
        push_text(&mut verts, &mut indices, &log_text, -0.96, -0.94, 0.012, 0.020, [0.2, 1.0, 0.5]);

        (verts, indices)
    }
}

fn print_help() {
    println!("====================================================================");
    println!("  METATOPIA GAME ENGINE v0.2.0 (macOS Native Apple Silicon / Metal)");
    println!("====================================================================");
    println!("Usage: metatopia [OPTIONS]");
    println!();
    println!("Options:");
    println!("  (no args)    Launch Metatopia Studio 2.0 & 3D Game Engine GUI");
    println!("  --play       Launch directly in live Play Mode (Scripts & Physics active)");
    println!("  --verify     Run full headless verification suite (Studio, 4D GR, Portals, Net)");
    println!("  --version    Print engine version");
    println!("  --help, -h   Print this help summary");
}

fn run_headless_verification() {
    let mut app = MetatopiaEngineApp::new(true);

    // Verify Schwarzschild photon sphere orbit stability
    let r_ph = app.black_hole.photon_sphere_radius();
    assert!((r_ph - 2.4).abs() < 1e-4);

    // Verify Morris-Thorne wormhole throat traversal
    let mut wh_state = WormholeState {
        l: 3.0,
        theta: std::f32::consts::FRAC_PI_2,
        phi: 0.0,
        dl: -2.5,
        dtheta: 0.0,
        dphi: 0.05,
    };
    let mut chart = ChartId(0);
    for _ in 0..80 {
        let prev_l = wh_state.l;
        wh_state = app.wormhole.step_geodesic_rk4(&wh_state, 0.04);
        if prev_l >= 0.0 && wh_state.l < 0.0 {
            chart = app.wormhole.chart_b;
        }
    }
    assert_eq!(chart, ChartId(1), "Wormhole throat traversal failed");

    // Verify Rhai scripts on all entities
    for entity in &mut app.entities {
        if let Some(ast) = &entity.script_ast {
            app.script_engine
                .execute_update(ast, &mut entity.script_state, 0.016)
                .expect("Rhai script execution failed");
        }
    }

    // Verify Cross-Portal Geodesic Weapon Shot (3 hits defeats Sentinel Boss)
    for _ in 0..3 {
        app.fire_cross_portal_laser();
    }
    assert!(app.sentinel_hp <= 0.0, "Cross-portal weapon shot did not defeat boss");

    let (verts, indices) = app.build_mesh();
    assert!(!verts.is_empty() && !indices.is_empty());

    println!("====================================================================");
    println!("  METATOPIA ENGINE v0.2.0 — MACOS EXECUTABLE VERIFIED");
    println!("====================================================================");
    println!("  • Entities & Linked Portals : {} entities / {} portal pair", app.entities.len(), app.manifold.portals().count());
    println!("  • Schwarzschild Black Hole  : r_s = {:.2}, r_ph = {:.2}, ISCO = {:.2}", app.black_hole.schwarzschild_radius, r_ph, app.black_hole.isco_radius());
    println!("  • Morris-Thorne Wormhole    : Traversed Chart 0 -> {:?} (l = {:.2})", chart, wh_state.l);
    println!("  • Cross-Portal Weapon Shot  : Sentinel Boss HP = {:.1} (SOLVED)", app.sentinel_hp);
    println!("  • Procedural Studio Mesh    : {} verts / {} indices", verts.len(), indices.len());
    println!("====================================================================");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--version" || a == "-v") {
        println!("metatopia_engine 0.2.0 (aarch64-apple-darwin / Metal WGPU)");
        return;
    }
    if args.iter().any(|a| a == "--verify" || a == "--bench") {
        run_headless_verification();
        return;
    }

    let start_in_play = args.iter().any(|a| a == "--play");
    println!("====================================================================");
    println!("  METATOPIA GAME ENGINE v0.2.0 — NATIVE MACOS STUDIO & RUNTIME");
    println!("  • Spacebar       : Toggle EDIT / PLAY Mode");
    println!("  • Keys 1..5      : Spawn Cube, Sphere, Cylinder, Torus, Capsule");
    println!("  • Key P / Key K  : Spawn Linked Non-Euclidean Portal / Cycle Curvature");
    println!("  • Key F          : Fire Cross-Portal Geodesic Weapon Shot");
    println!("  • Key C / Key R  : Cycle Rhai Script Preset / Hot-Reload Scripts");
    println!("====================================================================");
    run_game(MetatopiaEngineApp::new(start_in_play));
}
