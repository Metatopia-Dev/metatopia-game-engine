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
    pub rot: [f32; 3], // [yaw, pitch, roll]
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
        let mut state = ScriptEntityState::new(id, name_str.clone(), pos, color);
        state.emissive = 0.25;
        Self {
            id,
            name: name_str,
            mesh_type,
            pos,
            rot: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            color,
            metallic: 0.25,
            roughness: 0.38,
            emissive: 0.25,
            is_dynamic_physics: false,
            script_source: String::new(),
            script_ast: None,
            script_state: state,
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
    pub portal_pair_count: u32,
    pub selected_index: usize,
    pub mode: EditorMode,
    pub manifold: Manifold,
    pub black_hole: SchwarzschildBlackHole,
    pub wormhole: MorrisThorneWormhole,
    pub net_server: Option<NetServer>,
    pub sentinel_hp: f32,
    pub last_laser_segments: Vec<([f32; 3], [f32; 3])>,
    pub laser_flash_timer: f32,
    pub screen_fade_alpha: f32,
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

        // 1. Central Rotating Cyber Core (tilted so 3D rotation is visibly dynamic)
        let mut core = EditorEntity::new(1, "HyperCore", MeshType::Torus, [0.0, 2.4, 0.2], [0.0, 0.88, 1.0]);
        core.scale = [1.35, 1.35, 1.35];
        core.rot = [0.3, 0.65, 0.0];
        core.script_state.rot = core.rot;
        core.emissive = 0.55;
        core.script_state.emissive = 0.55;
        core.script_preset_name = "Orbit & Pulse".into();
        core.script_source = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt;
                set_var(e, "t", t);
                e.rotate(1.8 * dt, 1.1 * dt, 0.0);
                e.set_pos(0.0, 2.4 + sin(t * 2.5) * 0.45, 0.2);
                e.set_emissive(0.45 + sin(t * 4.0) * 0.25);
                return e;
            }
        "#
        .into();
        core.script_ast = script_engine.compile(&core.script_source).ok();
        entities.push(core);

        // 2. Orbiting Satellite Sphere
        let mut sat = EditorEntity::new(2, "ChronoOrb", MeshType::Sphere, [3.6, 2.1, 0.0], [1.0, 0.18, 0.65]);
        sat.scale = [1.0, 1.0, 1.0];
        sat.emissive = 0.50;
        sat.script_state.emissive = 0.50;
        sat.script_preset_name = "Circular Orbit".into();
        sat.script_source = r#"
            fn update(e, dt) {
                let angle = get_var(e, "angle") + 1.8 * dt;
                set_var(e, "angle", angle);
                e.set_pos(cos(angle) * 3.8, 2.1 + sin(angle * 2.0) * 0.55, sin(angle) * 3.8);
                return e;
            }
        "#
        .into();
        sat.script_ast = script_engine.compile(&sat.script_source).ok();
        entities.push(sat);

        // 3. Guardian Pillar Left
        let mut pillar_l = EditorEntity::new(3, "PillarLeft", MeshType::Cylinder, [-5.2, 1.5, -3.5], [0.18, 0.82, 0.52]);
        pillar_l.scale = [0.75, 1.8, 0.75];
        pillar_l.emissive = 0.18;
        entities.push(pillar_l);

        // 4. Guardian Pillar Right
        let mut pillar_r = EditorEntity::new(4, "PillarRight", MeshType::Cylinder, [5.2, 1.5, -3.5], [0.18, 0.82, 0.52]);
        pillar_r.scale = [0.75, 1.8, 0.75];
        pillar_r.emissive = 0.18;
        entities.push(pillar_r);

        // 5. Schwarzschild Black Hole Core Entity (event horizon + relativistic Doppler disk rendered in build_mesh)
        let mut bh_entity = EditorEntity::new(5, "SchwarzschildBH", MeshType::Sphere, [-3.2, 3.1, -5.2], [0.02, 0.02, 0.04]);
        bh_entity.scale = [1.4, 1.4, 1.4];
        bh_entity.emissive = 0.05;
        bh_entity.script_state.emissive = 0.05;
        bh_entity.script_preset_name = "Relativistic Lens".into();
        bh_entity.script_source = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt;
                set_var(e, "t", t);
                e.rotate(1.6 * dt, 0.0, 0.0);
                return e;
            }
        "#
        .into();
        bh_entity.script_ast = script_engine.compile(&bh_entity.script_source).ok();
        entities.push(bh_entity);

        // Build non-Euclidean manifold with Charts 0 (E^3) and 1 (H^3, K=-1.0)
        let mut manifold = Manifold::new();
        let _chart_1 = manifold.add_chart(GeometryType::Hyperbolic);

        let black_hole = SchwarzschildBlackHole::new(Point3::new(-3.2, 3.1, -5.2), 1.6);
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
                    pos: [0.0, 2.2, -7.8],
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
            portal_pair_count: 0,
            selected_index: 0,
            mode: if start_in_play_mode { EditorMode::Play } else { EditorMode::Edit },
            manifold,
            black_hole,
            wormhole,
            net_server,
            sentinel_hp: 100.0,
            last_laser_segments: Vec::new(),
            laser_flash_timer: 0.0,
            screen_fade_alpha: 0.0,
            script_engine,
            physics_world: PhysicsWorld::new(),
            particle_system: ParticleSystem::new(450),
            status_message: "Metatopia Engine v0.2.0 Ready - [SPACE] Play/Edit | [P] +Portal | [K] Curvature | [F] Fire Laser".into(),
            status_timer: 10.0,
            current_script_preset: 0,
            camera_initialized: false,
        };

        // Pre-spawn a linked Non-Euclidean Portal Pair connecting Chart 0 <-> Chart 1
        app.spawn_portal_pair();
        app
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        let s = msg.into().replace('—', "-");
        println!("[Metatopia Engine]: {}", s);
        self.status_message = s;
        self.status_timer = 6.0;
    }

    pub fn spawn_entity(&mut self, mesh_type: MeshType, prefix: &str, color: [f32; 3]) {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        let offset_x = ((id as f32 * 2.1).sin()) * 3.4;
        let offset_z = ((id as f32 * 1.6).cos()) * 2.6 - 0.5;
        let name = format!("{}_{}", prefix, id);
        let mut entity = EditorEntity::new(id, name.clone(), mesh_type, [offset_x, 1.6, offset_z], color);
        if mesh_type == MeshType::Torus {
            entity.rot = [0.2, 0.55, 0.0];
            entity.script_state.rot = entity.rot;
        }
        self.entities.push(entity);
        self.selected_index = self.entities.len() - 1;
        self.set_status(format!("Spawned '{}' (ID #{}) at [{:.1}, 1.6, {:.1}]", name, id, offset_x, offset_z));
    }

    pub fn spawn_portal_pair(&mut self) {
        let id_a = self.next_entity_id;
        let id_b = self.next_entity_id + 1;
        self.next_entity_id += 2;

        let pair_idx = self.portal_pair_count;
        self.portal_pair_count += 1;

        // First pair sits at x=0.0 along the central laser axis; subsequent pairs offset cleanly in X
        let x_off = if pair_idx == 0 {
            0.0
        } else if pair_idx % 2 == 1 {
            3.2 * ((pair_idx + 1) / 2) as f32
        } else {
            -3.2 * (pair_idx / 2) as f32
        };

        let pos_a = [x_off, 2.2, -2.8];
        let pos_b = [x_off, 2.2, -5.4];

        let portal_script = r#"
            fn update(e, dt) {
                let t = get_var(e, "t") + dt * 2.5;
                set_var(e, "t", t);
                e.set_curvature(-1.0 + sin(t) * 0.25);
                e.set_emissive(0.68 + sin(t * 2.5) * 0.25);
                return e;
            }
        "#;

        let mut portal_a = EditorEntity::new(id_a, format!("Portal_A_{}", id_a), MeshType::Torus, pos_a, [0.0, 0.92, 1.0]);
        portal_a.scale = [1.45, 1.45, 1.0];
        portal_a.rot = [0.0, std::f32::consts::FRAC_PI_2, 0.0]; // Vertical ring facing ±Z
        portal_a.script_state.rot = portal_a.rot;
        portal_a.emissive = 0.72;
        portal_a.script_state.emissive = 0.72;
        portal_a.script_state.chart_id = 0;
        portal_a.script_state.curvature_k = 0.0;
        portal_a.script_state.portal_target_chart = Some(1);
        portal_a.script_preset_name = "Non-Euclidean Portal".into();
        portal_a.script_source = portal_script.to_string();
        portal_a.script_ast = self.script_engine.compile(portal_script).ok();

        let mut portal_b = EditorEntity::new(id_b, format!("Portal_B_{}", id_b), MeshType::Torus, pos_b, [1.0, 0.22, 0.85]);
        portal_b.scale = [1.45, 1.45, 1.0];
        portal_b.rot = [0.0, std::f32::consts::FRAC_PI_2, 0.0]; // Vertical ring facing ±Z
        portal_b.script_state.rot = portal_b.rot;
        portal_b.emissive = 0.72;
        portal_b.script_state.emissive = 0.72;
        portal_b.script_state.chart_id = 1;
        portal_b.script_state.curvature_k = -1.0;
        portal_b.script_state.portal_target_chart = Some(0);
        portal_b.script_preset_name = "Non-Euclidean Portal".into();
        portal_b.script_source = portal_script.to_string();
        portal_b.script_ast = self.script_engine.compile(portal_script).ok();

        if pair_idx == 0 {
            let _ = self.manifold.create_portal_with_bounds(
                ChartId(0),
                ChartId(1),
                Point3::new(pos_a[0], pos_a[1], pos_a[2]),
                Point3::new(pos_a[0], pos_a[1], pos_a[2]),
                Matrix4::identity(),
                Vector3::new(0.0, 0.0, 1.0),
                4.5,
                4.5,
                PortalShape::Circular,
            );
        }

        self.entities.push(portal_a);
        self.entities.push(portal_b);
        self.selected_index = self.entities.len() - 1;
        self.set_status(format!(
            "Spawned Linked Portal Pair #{} <-> #{} (Charts 0 <-> 1, K=-1.0)",
            id_a, id_b
        ));
    }

    pub fn fire_cross_portal_laser(&mut self) {
        let origin = Point3::new(0.0, 2.2, 2.2);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let trace = ManifoldRaycaster::cast_geodesic_ray(
            &self.manifold,
            origin,
            dir,
            ChartId(0),
            12.0,
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
        self.laser_flash_timer = 1.15;

        if let Some(ref mut srv) = self.net_server {
            if let Some(hit) = srv.resolve_cross_portal_shot(
                &self.manifold,
                1,
                origin,
                dir,
                ChartId(0),
                14.0,
                4,
                35.0,
                2.5,
            ) {
                self.sentinel_hp = hit.target_remaining_health;
                self.particle_system.burst(
                    Vector3::new(hit.hit_point.x, hit.hit_point.y, hit.hit_point.z),
                    28,
                    [0.0, 0.95, 1.0, 1.0],
                    [1.0, 0.2, 0.85, 0.0],
                    5.5,
                    0.85,
                );
                self.set_status(format!(
                    "CROSS-PORTAL HIT! Chart {} | Portals Crossed: {} | Sentinel HP: {:.0}",
                    hit.hit_chart.0, hit.portals_crossed, self.sentinel_hp
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
                    e.set_emissive(0.45 + sin(t) * 0.35);
                    return e;
                }
            "#,
            ),
            1 => (
                "Hover & Bob",
                r#"
                fn update(e, dt) {
                    e.rotate(0.8 * dt, 1.4 * dt, 0.0);
                    let t = get_var(e, "t") + dt * 2.5;
                    set_var(e, "t", t);
                    e.move(0.0, cos(t) * 1.5 * dt, 0.0);
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
                    e.set_color(sin(t) * 0.4 + 0.55, cos(t) * 0.4 + 0.55, sin(t * 0.5) * 0.4 + 0.55);
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
                    e.set_emissive(0.55 + sin(t * 3.0) * 0.25);
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

// ─── Complete 5x7 Vector/Bitmap Font Glyph Table ───────────────────────────

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
        '{' => [0b00011, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00011],
        '}' => [0b11000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b11000],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        ';' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00010, 0b00100],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '=' => [0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        '\\' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '*' => [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        '\'' => [0b00100, 0b00100, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000],
        '"' => [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '?' => [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
        '^' => [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
        '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b10110, 0b00110, 0b00000],
        '@' => [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
        '~' => [0b00000, 0b00000, 0b01101, 0b10010, 0b00000, 0b00000, 0b00000],
        '$' => [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
        _ => [0b00000, 0b01110, 0b01010, 0b01110, 0b00000, 0b00000, 0b00000],
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
    push_ui_rect_layer(verts, indices, min_x, min_y, max_x, max_y, color, opacity, -1.0);
}

fn push_ui_rect_layer(
    verts: &mut Vec<GameVertex>,
    indices: &mut Vec<u32>,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    color: [f32; 3],
    opacity: f32,
    layer_w: f32,
) {
    let base_idx = verts.len() as u32;
    let pbr = [0.0, 0.0, opacity, layer_w];
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

/// Pixel-snapped 2D text renderer: snaps every glyph cell to integer device pixels (at 1280x720)
/// and guarantees >= 2 device pixels per font row so underscores and strokes never drop out.
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
    let half_w = 640.0f32;
    let half_h = 360.0f32;

    let px_cell_w = ((char_w / 6.0) * half_w).round().max(1.0);
    let px_cell_h = ((char_h / 8.0) * half_h).round().max(1.5);
    let pix_w = px_cell_w / half_w;
    let pix_h = px_cell_h / half_h;
    let glyph_advance = 6.0 * pix_w;

    let mut cur_x = ((start_x * half_w).round()) / half_w;
    let snapped_y = ((start_y * half_h).round()) / half_h;
    let top_y = snapped_y + 7.0 * pix_h;

    for ch in text.chars() {
        if cur_x + 5.0 * pix_w > 0.992 {
            break; // Never overflow right screen edge
        }
        if ch == ' ' {
            cur_x += glyph_advance;
            continue;
        }
        let bits = get_glyph_bits(ch);
        for (row, byte) in bits.iter().enumerate() {
            let py = top_y - ((row + 1) as f32 * pix_h);
            for col in 0..5 {
                if (byte & (1 << (4 - col))) != 0 {
                    let px = cur_x + (col as f32 * pix_w);
                    push_ui_rect_layer(verts, indices, px, py, px + pix_w, py + pix_h, color, 1.0, -2.0);
                }
            }
        }
        cur_x += glyph_advance;
    }
}

/// Cross-quad 3D beam renderer (emits both horizontal and vertical ribbons so laser beams are visible from any angle)
fn push_3d_line(
    verts: &mut Vec<GameVertex>,
    indices: &mut Vec<u32>,
    p0: [f32; 3],
    p1: [f32; 3],
    thickness: f32,
    color: [f32; 3],
    emissive: f32,
) {
    let n = [0.0, 1.0, 0.0];
    let pbr = [0.0, 0.2, emissive, 0.0];

    for &(ox, oy, oz) in &[
        (0.0, thickness, 0.0),
        (thickness, 0.0, 0.0),
    ] {
        let base = verts.len() as u32;
        verts.push(GameVertex::new([p0[0] - ox, p0[1] - oy, p0[2] - oz], n, color, pbr));
        verts.push(GameVertex::new([p0[0] + ox, p0[1] + oy, p0[2] + oz], n, color, pbr));
        verts.push(GameVertex::new([p1[0] + ox, p1[1] + oy, p1[2] + oz], n, color, pbr));
        verts.push(GameVertex::new([p1[0] - ox, p1[1] - oy, p1[2] - oz], n, color, pbr));
        indices.extend_from_slice(&[
            base, base + 1, base + 2, base, base + 2, base + 3,
            base + 2, base + 1, base, base + 3, base + 2, base,
        ]);
    }
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
            ctx.camera.position = Vector3::new(0.0, 5.2, 10.5);
            ctx.camera.pitch = -0.32;
            ctx.camera.move_speed = 12.0;
            self.camera_initialized = true;
        }

        ctx.scene.sun_direction = [-0.5, -1.0, -0.6, 0.0];
        ctx.scene.sun_color = [1.0, 0.98, 0.92, 1.0];

        if self.status_timer > 0.0 {
            self.status_timer -= ctx.dt;
        }
        if self.laser_flash_timer > 0.0 {
            self.laser_flash_timer = (self.laser_flash_timer - ctx.dt).max(0.0);
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
                    self.set_status("PLAY MODE ACTIVATED - Scripts, Relativity & Cross-Portal Combat Live");
                    EditorMode::Play
                }
                EditorMode::Play => {
                    self.set_status("EDIT MODE ACTIVATED - Inspector & Transform Gizmos Active");
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
                if m_ndc_x >= -0.68 && m_ndc_x <= -0.52 {
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
                } else if m_ndc_x >= -0.50 && m_ndc_x <= -0.40 {
                    self.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.8, 1.0]);
                } else if m_ndc_x >= -0.38 && m_ndc_x <= -0.26 {
                    self.spawn_entity(MeshType::Sphere, "NeonSphere", [1.0, 0.2, 0.5]);
                } else if m_ndc_x >= -0.24 && m_ndc_x <= -0.14 {
                    self.spawn_entity(MeshType::Cylinder, "PowerPillar", [1.0, 0.9, 0.1]);
                } else if m_ndc_x >= -0.12 && m_ndc_x <= -0.02 {
                    self.spawn_entity(MeshType::Torus, "GravityRing", [0.8, 0.2, 1.0]);
                } else if m_ndc_x >= 0.00 && m_ndc_x <= 0.10 {
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
            if m_ndc_x >= -0.98 && m_ndc_x <= -0.65 && m_ndc_y <= 0.72 && m_ndc_y >= -0.60 {
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
                Vector3::new(0.0, 2.4, 0.2),
                2,
                [0.0, 0.9, 1.0, 1.0],
                [0.85, 0.15, 0.9, 0.0],
                2.8,
                0.75,
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
        let mut verts = Vec::with_capacity(48000);
        let mut indices = Vec::with_capacity(96000);

        // 1. Grid Floor Plane (140x140)
        let (plane_v, plane_i) = ProceduralMesh::plane(140.0, 140.0, 28, 28, [0.06, 0.08, 0.12]);
        let base = verts.len() as u32;
        for v in plane_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.1, 0.8, 0.0, 0.0]));
        }
        for idx in plane_i {
            indices.push(base + idx);
        }

        // 2. Render Schwarzschild Black Hole Relativistic Accretion Disk & Photon Sphere Ring
        if let Some(bh_ent) = self.entities.iter().find(|e| e.name == "SchwarzschildBH") {
            let bh_c = bh_ent.pos;
            let spin_yaw = bh_ent.rot[0];

            // Relativistic Doppler-Beamed Accretion Disk (approaching wing bright gold/cyan, receding wing crimson)
            let rings = 6usize;
            let segs = 32usize;
            for r_idx in 0..rings {
                let r_frac = r_idx as f32 / (rings - 1).max(1) as f32;
                let radius = 1.15 + r_frac * 1.65;
                let doppler_r = 2.2 + r_frac * 3.0;
                for s_idx in 0..segs {
                    let a0 = (s_idx as f32 / segs as f32) * std::f32::consts::TAU + spin_yaw;
                    let a1 = ((s_idx + 1) as f32 / segs as f32) * std::f32::consts::TAU + spin_yaw;
                    let p0 = [
                        bh_c[0] + a0.cos() * radius,
                        bh_c[1] + a0.sin() * radius * 0.26,
                        bh_c[2] + a0.sin() * radius,
                    ];
                    let p1 = [
                        bh_c[0] + a1.cos() * radius,
                        bh_c[1] + a1.sin() * radius * 0.26,
                        bh_c[2] + a1.sin() * radius,
                    ];
                    let (delta, _boost4) = self.black_hole.accretion_disk_doppler(
                        cgmath::Point3::new(
                            self.black_hole.center.x + a0.cos() * doppler_r,
                            self.black_hole.center.y,
                            self.black_hole.center.z + a0.sin() * doppler_r,
                        ),
                        cgmath::Vector3::new(0.0, -0.25, -1.0),
                    );
                    let boost = delta.clamp(0.40, 2.1);
                    let disk_col = [
                        (0.95_f32 * boost).min(1.0),
                        (0.42_f32 * boost * boost).min(1.0),
                        (0.12_f32 * boost * boost * boost).min(1.0),
                    ];
                    push_3d_line(&mut verts, &mut indices, p0, p1, 0.085, disk_col, 0.85);
                }
            }

            // Photon Sphere Ring at r_ph
            let r_ph_vis = 1.05;
            for s_idx in 0..28 {
                let a0 = (s_idx as f32 / 28.0) * std::f32::consts::TAU;
                let a1 = ((s_idx + 1) as f32 / 28.0) * std::f32::consts::TAU;
                let p0 = [bh_c[0] + a0.cos() * r_ph_vis, bh_c[1] + a0.sin() * r_ph_vis, bh_c[2]];
                let p1 = [bh_c[0] + a1.cos() * r_ph_vis, bh_c[1] + a1.sin() * r_ph_vis, bh_c[2]];
                push_3d_line(&mut verts, &mut indices, p0, p1, 0.035, [1.0, 0.78, 0.25], 1.1);
            }
        }

        // 3. Render Sentinel Boss Target in Chart 1 (visible behind Portal_B)
        if self.sentinel_hp > 0.0 {
            let hp_ratio = (self.sentinel_hp / 100.0).clamp(0.0, 1.0);
            let boss_col = [1.0, 0.15 + 0.55 * hp_ratio, 0.35];
            let (bv, bi) = ProceduralMesh::capsule(0.65, 0.5, 10, 18, boss_col);
            let b_base = verts.len() as u32;
            for v in bv {
                verts.push(GameVertex::new(
                    [v.position[0], v.position[1] + 2.2, v.position[2] - 7.8],
                    v.normal,
                    boss_col,
                    [0.4, 0.25, 0.75, 99.0],
                ));
            }
            for idx in bi {
                indices.push(b_base + idx);
            }
        }

        // 4. Render Cross-Portal Geodesic Laser Beam (when active)
        if self.laser_flash_timer > 0.0 {
            let intensity = (self.laser_flash_timer / 1.15).clamp(0.2, 1.0);
            for (seg_idx, &(p0, p1)) in self.last_laser_segments.iter().enumerate() {
                let beam_col = if seg_idx == 0 {
                    [0.0, 0.96, 1.0]
                } else {
                    [1.0, 0.22, 0.88]
                };
                push_3d_line(&mut verts, &mut indices, p0, p1, 0.095 * intensity, beam_col, 1.35 * intensity);
                push_3d_line(&mut verts, &mut indices, p0, p1, 0.038 * intensity, [1.0, 1.0, 1.0], 1.6 * intensity);
            }
        }

        // 5. Render All Scene Entities (with full yaw + pitch 3D rotation)
        for entity in &self.entities {
            let (shape_v, shape_i) = match entity.mesh_type {
                MeshType::Cube => {
                    let hx = entity.scale[0] * 0.55;
                    let hy = entity.scale[1] * 0.55;
                    let hz = entity.scale[2] * 0.55;
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
                MeshType::Sphere => ProceduralMesh::capsule(entity.scale[0] * 0.5, 0.01, 10, 18, entity.color),
                MeshType::Cylinder => ProceduralMesh::cylinder(entity.scale[0] * 0.5, entity.scale[1], 18, entity.color),
                MeshType::Torus => ProceduralMesh::torus(entity.scale[0] * 0.85, entity.scale[0] * 0.22, 28, 14, entity.color),
                MeshType::Capsule => ProceduralMesh::capsule(entity.scale[0] * 0.4, entity.scale[1], 10, 16, entity.color),
            };

            let base_offset = verts.len() as u32;
            let pbr = [entity.metallic, entity.roughness, entity.emissive.clamp(0.0, 1.25), entity.id as f32];
            let (sy, cy) = entity.rot[0].sin_cos();
            let (sp, cp) = entity.rot[1].sin_cos();

            for v in shape_v {
                // Pitch around X, then Yaw around Y
                let x0 = v.position[0];
                let y0 = v.position[1] * cp - v.position[2] * sp;
                let z0 = v.position[1] * sp + v.position[2] * cp;

                let rx = x0 * cy - z0 * sy;
                let rz = x0 * sy + z0 * cy;
                let world_pos = [
                    rx + entity.pos[0],
                    y0 + entity.pos[1],
                    rz + entity.pos[2],
                ];

                let nx0 = v.normal[0];
                let ny0 = v.normal[1] * cp - v.normal[2] * sp;
                let nz0 = v.normal[1] * sp + v.normal[2] * cp;
                let r_norm_x = nx0 * cy - nz0 * sy;
                let r_norm_z = nx0 * sy + nz0 * cy;
                let world_norm = [r_norm_x, ny0, r_norm_z];

                verts.push(GameVertex::new(world_pos, world_norm, v.color, pbr));
            }

            for idx in shape_i {
                indices.push(base_offset + idx);
            }
        }

        // 6. Selection Transform Gizmo (Red X, Green Y, Blue Z axes with pbr.w = -3.0 depth priority)
        if !self.entities.is_empty() && self.mode == EditorMode::Edit {
            let entity = &self.entities[self.selected_index];
            let ep = entity.pos;
            let g_len = 2.1;
            let g_rad = 0.065;

            let (x_v, x_i) = ProceduralMesh::cylinder(g_rad, g_len, 10, [1.0, 0.18, 0.18]);
            let x_base = verts.len() as u32;
            for v in x_v {
                let pos = [ep[0] + v.position[1] + g_len * 0.5, ep[1] + v.position[0], ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [1.0, 0.0, 0.0], [1.0, 0.18, 0.18], [0.0, 0.2, 1.0, -3.0]));
            }
            for idx in x_i { indices.push(x_base + idx); }

            let (y_v, y_i) = ProceduralMesh::cylinder(g_rad, g_len, 10, [0.18, 1.0, 0.28]);
            let y_base = verts.len() as u32;
            for v in y_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[1] + g_len * 0.5, ep[2] + v.position[2]];
                verts.push(GameVertex::new(pos, [0.0, 1.0, 0.0], [0.18, 1.0, 0.28], [0.0, 0.2, 1.0, -3.0]));
            }
            for idx in y_i { indices.push(y_base + idx); }

            let (z_v, z_i) = ProceduralMesh::cylinder(g_rad, g_len, 10, [0.22, 0.58, 1.0]);
            let z_base = verts.len() as u32;
            for v in z_v {
                let pos = [ep[0] + v.position[0], ep[1] + v.position[2], ep[2] + v.position[1] + g_len * 0.5];
                verts.push(GameVertex::new(pos, [0.0, 0.0, 1.0], [0.22, 0.58, 1.0], [0.0, 0.2, 1.0, -3.0]));
            }
            for idx in z_i { indices.push(z_base + idx); }
        }

        // 7. Render Particle System in Play Mode (with clamped emissive so cyan/magenta colors stay vivid)
        if self.mode == EditorMode::Play {
            let (p_verts, p_indices) = self.particle_system.build_mesh();
            let base = verts.len() as u32;
            for mut pv in p_verts {
                pv.pbr[2] = pv.pbr[2].clamp(0.15, 0.95);
                verts.push(pv);
            }
            for pi in p_indices {
                indices.push(base + pi);
            }
        }

        // 8. 2D On-Screen Studio UI Overlay (sRGB-correct deep obsidian panels)
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.84, 1.0, 1.0, [0.05, 0.06, 0.09], 0.97);
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.835, 1.0, 0.845, [0.0, 0.85, 1.0], 1.0);

        push_text(&mut verts, &mut indices, "METATOPIA ENGINE", -0.97, 0.89, 0.015, 0.032, [0.0, 0.95, 1.0]);

        let (mode_label, mode_col) = match self.mode {
            EditorMode::Edit => ("MODE:EDIT", [0.16, 0.52, 0.88]),
            EditorMode::Play => ("MODE:PLAY", [0.15, 0.78, 0.38]),
        };
        push_ui_rect(&mut verts, &mut indices, -0.68, 0.86, -0.52, 0.97, mode_col, 0.98);
        push_text(&mut verts, &mut indices, mode_label, -0.665, 0.89, 0.012, 0.026, [1.0, 1.0, 1.0]);

        let buttons = [
            (-0.50, -0.40, "+CUBE", [0.14, 0.20, 0.30]),
            (-0.38, -0.26, "+SPHERE", [0.14, 0.20, 0.30]),
            (-0.24, -0.14, "+CYL", [0.14, 0.20, 0.30]),
            (-0.12, -0.02, "+TORUS", [0.14, 0.20, 0.30]),
            (0.00, 0.10, "+CAPS", [0.14, 0.20, 0.30]),
            (0.12, 0.28, "SCRIPT(C)", [0.26, 0.16, 0.40]),
            (0.30, 0.44, "RELOAD(R)", [0.16, 0.34, 0.26]),
            (0.46, 0.58, "DEL(BS)", [0.42, 0.15, 0.18]),
            (0.60, 0.76, "+PORTAL(P)", [0.06, 0.40, 0.50]),
            (0.78, 0.96, "LASER(F)", [0.55, 0.14, 0.42]),
        ];
        for (bx0, bx1, label, bcol) in buttons {
            push_ui_rect(&mut verts, &mut indices, bx0, 0.86, bx1, 0.97, bcol, 0.96);
            push_text(&mut verts, &mut indices, label, bx0 + 0.012, 0.89, 0.011, 0.024, [0.94, 0.97, 1.0]);
        }

        // Left Scene Outliner Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.62, -0.65, 0.81, [0.055, 0.065, 0.10], 0.96);
        push_ui_rect(&mut verts, &mut indices, -0.655, -0.62, -0.65, 0.81, [0.0, 0.65, 0.85], 1.0);
        push_text(&mut verts, &mut indices, "SCENE OUTLINER", -0.95, 0.74, 0.014, 0.026, [0.0, 0.92, 1.0]);

        for (i, entity) in self.entities.iter().enumerate().take(14) {
            let row_y = 0.64 - (i as f32) * 0.085;
            let is_sel = i == self.selected_index;
            if is_sel {
                push_ui_rect(&mut verts, &mut indices, -0.96, row_y - 0.015, -0.67, row_y + 0.045, [0.14, 0.40, 0.58], 0.98);
            }
            let txt_col = if is_sel { [1.0, 1.0, 1.0] } else { [0.78, 0.84, 0.92] };
            let item_label = format!("{}:{}", entity.id, entity.name);
            push_text(&mut verts, &mut indices, &item_label, -0.94, row_y, 0.012, 0.024, txt_col);
        }

        // Right Property Inspector + 4D Relativity Panel (widened to x=0.63..0.985 so zero rows clip)
        push_ui_rect(&mut verts, &mut indices, 0.63, -0.62, 0.985, 0.81, [0.055, 0.065, 0.10], 0.96);
        push_ui_rect(&mut verts, &mut indices, 0.63, -0.62, 0.635, 0.81, [0.0, 0.65, 0.85], 1.0);
        push_text(&mut verts, &mut indices, "INSPECTOR", 0.66, 0.74, 0.014, 0.026, [1.0, 0.78, 0.22]);

        if !self.entities.is_empty() {
            let selected = &self.entities[self.selected_index];
            push_text(&mut verts, &mut indices, &format!("NAME: {}", selected.name), 0.66, 0.64, 0.011, 0.024, [0.96, 0.97, 1.0]);
            push_text(&mut verts, &mut indices, "TRANSFORM (POS):", 0.66, 0.54, 0.011, 0.024, [0.55, 0.82, 1.0]);
            push_text(&mut verts, &mut indices, &format!("X: {:.2}", selected.pos[0]), 0.68, 0.46, 0.011, 0.024, [1.0, 0.45, 0.45]);
            push_text(&mut verts, &mut indices, &format!("Y: {:.2}", selected.pos[1]), 0.68, 0.38, 0.011, 0.024, [0.45, 1.0, 0.55]);
            push_text(&mut verts, &mut indices, &format!("Z: {:.2}", selected.pos[2]), 0.68, 0.30, 0.011, 0.024, [0.45, 0.75, 1.0]);

            push_text(&mut verts, &mut indices, &format!("SCALE: {:.2} | YAW: {:.2}", selected.scale[0], selected.rot[0]), 0.66, 0.20, 0.011, 0.024, [0.88, 0.90, 0.95]);

            push_text(&mut verts, &mut indices, "SCRIPT PRESET:", 0.66, 0.10, 0.011, 0.024, [0.85, 0.55, 1.0]);
            push_text(&mut verts, &mut indices, &selected.script_preset_name, 0.66, 0.02, 0.011, 0.024, [0.35, 1.0, 0.72]);

            push_text(&mut verts, &mut indices, "MANIFOLD & 4D GR:", 0.66, -0.08, 0.011, 0.024, [0.0, 0.95, 1.0]);
            let manifold_label = format!(
                "CHART: {} | CURV K: {:.1}",
                selected.script_state.chart_id, selected.script_state.curvature_k
            );
            push_text(&mut verts, &mut indices, &manifold_label, 0.66, -0.16, 0.011, 0.024, [0.88, 0.96, 1.0]);

            let bh_label = format!(
                "BH R_PH: {:.2} | ISCO: {:.1}",
                self.black_hole.photon_sphere_radius(),
                self.black_hole.isco_radius()
            );
            push_text(&mut verts, &mut indices, &bh_label, 0.66, -0.24, 0.011, 0.024, [1.0, 0.78, 0.28]);

            let boss_label = format!("SENTINEL HP: {:.0} / 100", self.sentinel_hp);
            let boss_col = if self.sentinel_hp > 0.0 { [1.0, 0.42, 0.78] } else { [0.25, 1.0, 0.55] };
            push_text(&mut verts, &mut indices, &boss_label, 0.66, -0.32, 0.011, 0.024, boss_col);
        }

        // Bottom Script Console & Status Bar Panel
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.97, 0.985, -0.64, [0.045, 0.055, 0.085], 0.97);
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.65, 0.985, -0.64, [0.0, 0.65, 0.85], 1.0);

        push_text(&mut verts, &mut indices, "RHAI SCRIPT IDE & CONSOLE (HOT-RELOAD ON R | LASER ON F):", -0.95, -0.71, 0.012, 0.024, [0.0, 0.92, 1.0]);
        push_text(&mut verts, &mut indices, "1 | FN UPDATE(E, DT) { E.SET_CURVATURE(-1.0); E.MOBIUS_STEP(0.05*DT, 0.0, 0.0); RETURN E; }", -0.95, -0.79, 0.011, 0.024, [0.95, 0.82, 0.25]);
        push_text(&mut verts, &mut indices, "2 | CONTROLS: [SPACE] PLAY/EDIT | [WASD+RMB] FLY | [P] +PORTAL | [K] CURVATURE | [F] FIRE LASER", -0.95, -0.87, 0.011, 0.024, [0.75, 0.82, 0.92]);

        let log_text = format!("[LOG]: {}", self.status_message);
        push_text(&mut verts, &mut indices, &log_text, -0.95, -0.95, 0.011, 0.024, [0.25, 1.0, 0.55]);

        // Smooth cinema fade-in / fade-out overlay when recording video
        if self.screen_fade_alpha > 0.001 {
            push_ui_rect_layer(&mut verts, &mut indices, -1.0, -1.0, 1.0, 1.0, [0.0, 0.0, 0.0], self.screen_fade_alpha, -2.2);
        }

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
    println!("  (no args)              Launch Metatopia Studio 2.0 & 3D Game Engine GUI");
    println!("  --play                 Launch directly in live Play Mode (Scripts & Physics active)");
    println!("  --record-video <PATH>  Render 60 FPS macOS Engine & Studio showcase video via Metal WGPU");
    println!("  --verify               Run full headless verification suite (Studio, 4D GR, Portals, Net)");
    println!("  --version              Print engine version");
    println!("  --help, -h             Print this help summary");
}

/// Synthesizes a 16.0-second 128 BPM stereo soundtrack with:
/// - Wide stereo detuned FM arpeggiator & progression
/// - Mode-switch synth riser into Bar 5 (7.50s = Play Mode activation)
/// - Synchronized sci-fi laser zap + sub-impact SFX at 10.31s, 12.19s, and 14.06s
/// - True-peak limiting below -1.8 dBTP (zero clipping)
fn synthesize_engine_demo_wav(wav_path: &str, duration_secs: f32) {
    use std::io::Write;
    let sample_rate = 48_000u32;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let bpm = 128.0f32;
    let beat_sec = 60.0 / bpm; // 0.46875s
    let step16_sec = beat_sec * 0.25;
    let tau = std::f32::consts::TAU;
    let arp_notes: [f32; 8] = [349.23, 415.30, 523.25, 622.25, 698.46, 622.25, 523.25, 415.30];
    let laser_times: [f32; 3] = [619.0 / 60.0, 731.0 / 60.0, 844.0 / 60.0];
    let play_switch_time = 450.0 / 60.0; // 7.50s (Bar 5 downbeat)

    let data_bytes = (num_samples * 4) as u32;
    let mut pcm = Vec::with_capacity(44 + data_bytes as usize);
    pcm.extend_from_slice(b"RIFF");
    pcm.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    pcm.extend_from_slice(b"WAVEfmt ");
    pcm.extend_from_slice(&16u32.to_le_bytes());
    pcm.extend_from_slice(&1u16.to_le_bytes());
    pcm.extend_from_slice(&2u16.to_le_bytes());
    pcm.extend_from_slice(&sample_rate.to_le_bytes());
    pcm.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    pcm.extend_from_slice(&4u16.to_le_bytes());
    pcm.extend_from_slice(&16u16.to_le_bytes());
    pcm.extend_from_slice(b"data");
    pcm.extend_from_slice(&data_bytes.to_le_bytes());

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let beat_pos = (t / beat_sec).fract();
        let step16_idx = (t / step16_sec).floor() as usize;
        let step16_pos = (t / step16_sec).fract();
        let in_play_mode = t >= play_switch_time;

        // 1. Punchy 909 Kick
        let kick_freq = 48.0 + 115.0 * (-beat_pos * 28.0).exp();
        let kick = (tau * kick_freq * t).sin() * (-beat_pos * 6.2).exp() * 0.42;
        let sidechain = (1.0 - (-beat_pos * 5.5).exp() * 0.65).clamp(0.25, 1.0);

        // 2. Stereo Reese Bassline (shifts up in energy once Play Mode activates at 7.50s)
        let bass_freq = match (step16_idx / 16) % 4 {
            0 => 43.65,
            1 => 51.91,
            2 => 58.27,
            _ => 65.41,
        };
        let bass_l = ((tau * bass_freq * 0.998 * t).sin() + 0.45 * (tau * bass_freq * 2.003 * t).sin()).tanh()
            * sidechain
            * if in_play_mode { 0.30 } else { 0.24 };
        let bass_r = ((tau * bass_freq * 1.002 * t).sin() + 0.45 * (tau * bass_freq * 1.997 * t).sin()).tanh()
            * sidechain
            * if in_play_mode { 0.30 } else { 0.24 };

        // 3. Wide Stereo Ping-Pong FM Arpeggiator
        let base_arp = arp_notes[step16_idx % arp_notes.len()];
        let arp_f = if in_play_mode && (step16_idx % 4 == 2) { base_arp * 2.0 } else { base_arp };
        let arp_env = (-step16_pos * 8.5).exp();
        let arp_sig_l = (tau * arp_f * 0.999 * t + 1.5 * (tau * arp_f * 2.0 * t).sin() * arp_env).sin() * arp_env * 0.18;
        let arp_sig_r = (tau * arp_f * 1.001 * t + 1.5 * (tau * arp_f * 2.0 * t).sin() * arp_env).sin() * arp_env * 0.18;
        let pan = 0.5 + 0.42 * (step16_idx as f32 * 1.35).sin();
        let arp_l = arp_sig_l * (1.0 - pan).sqrt();
        let arp_r = arp_sig_r * pan.sqrt();

        // 4. Play Mode Activation Riser & Chime at 7.50s
        let mut sfx_l = 0.0f32;
        let mut sfx_r = 0.0f32;
        if t >= play_switch_time - 0.9 && t < play_switch_time {
            let p = (t - (play_switch_time - 0.9)) / 0.9;
            let sweep_f = 220.0 * (2.0f32).powf(p * 2.2);
            sfx_l += (tau * sweep_f * t).sin() * p * 0.16;
            sfx_r += (tau * sweep_f * 1.01 * t).sin() * p * 0.16;
        } else if t >= play_switch_time && t < play_switch_time + 0.65 {
            let dt_chime = t - play_switch_time;
            let env = (-dt_chime * 5.0).exp();
            sfx_l += ((tau * 698.46 * t).sin() + (tau * 1046.50 * t).sin()) * env * 0.14;
            sfx_r += ((tau * 880.00 * t).sin() + (tau * 1396.91 * t).sin()) * env * 0.14;
        }

        // 5. Diegetic Cross-Portal Geodesic Laser Zap + Sub-Impact SFX
        for &lt in &laser_times {
            if t >= lt && t < lt + 0.48 {
                let dt_l = t - lt;
                let zap_f = 160.0 + 1850.0 * (-dt_l * 18.0).exp();
                let zap_env = (-dt_l * 7.5).exp();
                let zap = (tau * zap_f * t + 2.5 * (tau * 92.0 * t).sin()).sin() * zap_env * 0.24;
                sfx_l += zap * 1.1;
                sfx_r += zap * 0.9;
            }
        }

        let fade = if t < 0.4 {
            t / 0.4
        } else if t > duration_secs - 1.0 {
            ((duration_secs - t) / 1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Soft-knee limiter with generous headroom (~ -2.0 dBTP)
        let mix_l = (kick + bass_l + arp_l + sfx_l).tanh() * fade;
        let mix_r = (kick + bass_r + arp_r + sfx_r).tanh() * fade;
        let l = (mix_l * 23_500.0).clamp(-32_767.0, 32_767.0) as i16;
        let r = (mix_r * 23_500.0).clamp(-32_767.0, 32_767.0) as i16;
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }

    let mut f = std::fs::File::create(wav_path).expect("Failed to create WAV file");
    f.write_all(&pcm).expect("Failed to write WAV file");
}

fn run_engine_video_recording(output_path: &str) {
    use cgmath::{Deg, perspective};
    use std::io::Write;
    use std::process::{Command, Stdio};
    use wgpu::util::DeviceExt;

    let width = 1280u32;
    let height = 720u32;
    let fps = 60u32;
    let duration_secs = 16.0f32;
    let total_frames = (fps as f32 * duration_secs) as u32; // 960 frames @ 60 FPS

    let wav_path = "/tmp/metatopia_macos_engine_demo.wav";
    synthesize_engine_demo_wav(wav_path, duration_secs);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to acquire Metal GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("MetatopiaVideoDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create WGPU device");

    let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("EngineVideoTarget"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("EngineVideoDepth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (width * 4 + 255) & !255;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("EngineVideoReadback"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut camera_uniform = CameraUniform {
        view_proj: [[0.0; 4]; 4],
        view_position: [0.0; 4],
    };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CameraUniform"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut scene_uniform = SceneUniform::default();
    scene_uniform.sun_direction = [-0.5, -1.0, -0.6, 0.0];
    scene_uniform.sun_color = [1.0, 0.98, 0.92, 1.0];
    let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SceneUniform"),
        contents: bytemuck::cast_slice(&[scene_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Uniforms"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("UniformsBG"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scene_buffer.as_entire_binding(),
            },
        ],
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("EditorShader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("EngineVideoPipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &[GameVertex::LAYOUT],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let ffmpeg_bin = if std::path::Path::new("/opt/homebrew/bin/ffmpeg").exists() {
        "/opt/homebrew/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let mut ffmpeg = Command::new(ffmpeg_bin)
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", width, height),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-i",
            wav_path,
            "-vf",
            "scale=out_color_matrix=bt709",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-g",
            "120",
            "-keyint_min",
            "60",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn ffmpeg");

    let ffmpeg_stdin = ffmpeg.stdin.as_mut().expect("Failed to open ffmpeg stdin");
    let mut app = MetatopiaEngineApp::new(false);
    let mut camera = FpsCamera::default();
    let dt = 1.0 / (fps as f32);
    let mut frame_rgba = vec![0u8; (width * height * 4) as usize];

    println!("====================================================================");
    println!("  RECORDING MACOS ENGINE SHOWCASE VIDEO (960 frames @ 60 FPS)");
    println!("  Output: {}", output_path);
    println!("====================================================================");

    for frame_idx in 0..total_frames {
        let t = frame_idx as f32 * dt;

        // Smooth cinema fade-in (0.0s..0.4s) & fade-out (15.0s..16.0s) matching audio
        app.screen_fade_alpha = if t < 0.4 {
            1.0 - (t / 0.4)
        } else if t > duration_secs - 1.0 {
            ((t - (duration_secs - 1.0)) / 1.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        if app.laser_flash_timer > 0.0 {
            app.laser_flash_timer = (app.laser_flash_timer - dt).max(0.0);
        }

        // Beat-Quantized 128 BPM Choreography (1 bar = 1.875s = 112.5 frames):
        match frame_idx {
            0 => {
                app.selected_index = 0;
                app.set_status("PHASE 1: EDIT MODE - Inspecting HyperCore & 3D Transform Gizmo");
            }
            112 => {
                app.selected_index = 1;
                app.set_status("SELECTED: ChronoOrb (ID #2) - Circular Orbit Preset");
            }
            225 => {
                app.selected_index = 4;
                app.set_status("SELECTED: SchwarzschildBH (ID #5) - Doppler Accretion Disk & Photon Sphere");
            }
            309 => {
                app.spawn_entity(MeshType::Cube, "CyberCube", [0.2, 0.85, 1.0]);
            }
            365 => {
                app.spawn_entity(MeshType::Torus, "GravityRing", [0.85, 0.25, 1.0]);
                app.cycle_selected_curvature();
                app.cycle_script_preset();
            }
            415 => {
                app.spawn_portal_pair();
            }
            450 => {
                // Bar 5 Downbeat (7.50s): Enter Play Mode!
                app.mode = EditorMode::Play;
                app.set_status("PHASE 3: PLAY MODE ACTIVATED - Live Rhai Scripts & Particles Running");
            }
            619 => {
                // Bar 6 Beat 3 (10.31s): Fire Cross-Portal Shot #1
                app.fire_cross_portal_laser();
            }
            731 => {
                // Bar 7 Beat 3 (12.19s): Fire Cross-Portal Shot #2
                app.fire_cross_portal_laser();
            }
            844 => {
                // Bar 9 Downbeat (14.06s): Fire Cross-Portal Shot #3 (Boss Defeated!)
                app.fire_cross_portal_laser();
            }
            _ => {}
        }

        // Update particles & Rhai scripts in Play Mode
        if app.mode == EditorMode::Play {
            app.particle_system.burst(
                Vector3::new(0.0, 2.4, 0.2),
                2,
                [0.0, 0.92, 1.0, 1.0],
                [0.9, 0.18, 0.85, 0.0],
                3.0,
                0.75,
            );
            for entity in &mut app.entities {
                if let Some(ast) = &entity.script_ast {
                    let _ = app
                        .script_engine
                        .execute_update(ast, &mut entity.script_state, dt);
                    entity.pos = entity.script_state.pos;
                    entity.rot = entity.script_state.rot;
                    entity.color = entity.script_state.color;
                    entity.emissive = entity.script_state.emissive;
                    entity.scale = entity.script_state.scale;
                }
            }
        }
        app.particle_system.update(dt, None);

        // Event-aware camera framing: wide front-arc dolly at positive Z so the entire 3D studio & gizmos are clearly framed
        let (cam_pos, look_target) = if t < 7.5 {
            // Edit & Spawn phase (0.0s..7.5s): smooth front-arc dolly around HyperCore (0,2.2,0), SchwarzschildBH (-3.2,3.1,-5.2), and Portals
            let phase_p = t / 7.5;
            let side_x = -2.6 * (phase_p * std::f32::consts::PI * 1.35 - 0.55).sin();
            let pos = Vector3::new(side_x, 4.1 + 0.35 * (t * 0.9).sin(), 5.3 - 0.5 * phase_p);
            let target = if (3.5..5.2).contains(&t) {
                Vector3::new(-1.5, 2.5, -2.6) // Frame SchwarzschildBH & Doppler Accretion Disk
            } else {
                Vector3::new(-0.2, 2.1, -1.6)
            };
            (pos, target)
        } else {
            // Play & Cross-Portal Laser Combat phase (7.5s..16.0s): frame emitter (0,2.2,2.2) -> Portal_A (0,2.2,-2.8) -> Portal_B (0,2.2,-5.4) -> Sentinel Boss (0,2.2,-7.8)
            let play_p = (t - 7.5) / 8.5;
            let side_x = 3.2 * (play_p * std::f32::consts::PI * 1.4 - 0.45).sin();
            let pos = Vector3::new(side_x, 3.8 + 0.35 * (play_p * 4.0).cos(), 4.8 - 1.0 * play_p);
            let target = Vector3::new(0.0, 2.1, -3.5);
            (pos, target)
        };

        camera.position = cam_pos;
        let to_center = (look_target - camera.position).normalize();
        camera.pitch = to_center.y.asin();
        camera.yaw = to_center.z.atan2(to_center.x);

        let aspect = width as f32 / height as f32;
        let proj = perspective(Deg(56.0), aspect, 0.1, 500.0);
        let view = camera.view_matrix();
        camera_uniform.view_proj = (proj * view).into();
        camera_uniform.view_position = [camera.position.x, camera.position.y, camera.position.z, 1.0];

        let selected_id = if !app.entities.is_empty() {
            app.entities[app.selected_index].id as f32
        } else {
            0.0
        };
        let is_play = if app.mode == EditorMode::Play { 1.0 } else { 0.0 };
        scene_uniform.params = [t, 1.2, 0.35, 0.0];
        scene_uniform.game_data = [t, is_play, selected_id, 0.0];
        scene_uniform.hud_info = [width as f32, height as f32, 0.0, 0.0];

        queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
        queue.write_buffer(&scene_buffer, 0, bytemuck::cast_slice(&[scene_uniform]));

        let (verts, indices) = app.build_mesh();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FrameVertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FrameIndices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("VideoEncoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VideoPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.004,
                            g: 0.005,
                            b: 0.010,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        {
            let data = slice.get_mapped_range();
            for y in 0..height {
                let src_start = (y * bytes_per_row) as usize;
                let dst_start = (y * width * 4) as usize;
                frame_rgba[dst_start..dst_start + (width * 4) as usize]
                    .copy_from_slice(&data[src_start..src_start + (width * 4) as usize]);
            }
        }
        readback_buffer.unmap();
        ffmpeg_stdin.write_all(&frame_rgba).unwrap();

        if (frame_idx + 1) % 240 == 0 || frame_idx + 1 == total_frames {
            println!(
                "  • Rendered {}/{} frames ({:.1}s / {:.1}s)",
                frame_idx + 1,
                total_frames,
                (frame_idx + 1) as f32 / fps as f32,
                duration_secs
            );
        }
    }

    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait().expect("Failed to wait on ffmpeg");
    assert!(status.success(), "ffmpeg encoding failed");
    println!("====================================================================");
    println!("  MACOS ENGINE VIDEO RECORDED SUCCESSFULLY -> {}", output_path);
    println!("====================================================================");
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
    if let Some(idx) = args.iter().position(|a| a == "--record-video") {
        let out_path = args
            .get(idx + 1)
            .map(|s| s.as_str())
            .unwrap_or("metatopia_macos_engine_demo.mp4");
        run_engine_video_recording(out_path);
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
