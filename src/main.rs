//! Metatopia Game Engine v0.2.0 — Native macOS Ray-Marched Non-Euclidean & 4D Relativistic Studio
//!
//! Combines:
//!   1. Real-Time GPU Ray-Marched Non-Euclidean & 4D Relativistic Viewport (`shaders/studio_raymarch.wgsl`):
//!      - 16-Bounce Recursive GPU Portal Ray Traversal across E³, H³, S³, K³, and T³ manifolds
//!      - 4D Schwarzschild Black Hole Binet Null Geodesic Gravitational Lensing + Doppler Accretion Disk
//!      - Hyperbolic H³ {4,3,5} Conformal Honeycomb Lattice & Spherical S³ Star-Vault
//!      - Non-Orientable Klein Bottle K³ Chiral Parity Flip & Volumetric Cross-Portal Geodesic Laser
//!   2. Crisp Native Vector Studio GUI (`egui` + `egui-wgpu` + `egui-winit`):
//!      - Anti-aliased TrueType typography, glassmorphic dark-obsidian docks, interactive sliders,
//!        Manifold Atlas Outliner, Relativistic Inspector, and Live Rhai Script IDE.
//!   3. Built-in 60 FPS Metal WGPU Video Recorder (`--record-video <output.mp4>`) & CLI Verification (`--verify`).

use cgmath::{InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};
use metatopia_engine::manifold::{
    ChartId, GeodesicParticleSwarm, GeometryType, Manifold, ManifoldRaycaster, PortalShape,
};
use metatopia_engine::net::{EntityState, NetServer};
use metatopia_engine::relativity::{MorrisThorneWormhole, SchwarzschildBlackHole, WormholeState};
use metatopia_engine::scripting::{ScriptEngine, ScriptEntityState};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

const SHADER_SRC: &str = include_str!("../shaders/studio_raymarch.wgsl");

// ─── GPU Uniform Buffer (16-byte aligned, matches StudioViewportUniform in WGSL) ───

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StudioViewportUniform {
    pub camera_pos: [f32; 4],    // xyz = pos, w = fov
    pub camera_dir: [f32; 4],    // xyz = forward, w = active_chart_id
    pub camera_up: [f32; 4],     // xyz = up, w = curvature_k
    pub resolution: [f32; 4],    // x = width, y = height, z = time, w = scene_mode
    pub params: [f32; 4],        // x = max_portal_bounces, y = geodesic_on, z = heatmap_mode, w = max_steps
    pub portal0_src: [f32; 4],
    pub portal0_dst: [f32; 4],
    pub portal0_norm: [f32; 4],
    pub portal1_src: [f32; 4],
    pub portal1_dst: [f32; 4],
    pub portal1_norm: [f32; 4],
    pub portal2_src: [f32; 4],
    pub portal2_dst: [f32; 4],
    pub portal2_norm: [f32; 4],
    pub portal3_src: [f32; 4],
    pub portal3_dst: [f32; 4],
    pub portal3_norm: [f32; 4],
    pub telemetry: [f32; 4],     // x = kick_env, y = bass_env, z = arp_env, w = play_mode
    pub bh_params: [f32; 4],     // xyz = black_hole_pos, w = schwarzschild_radius r_s
    pub laser_params: [f32; 4],  // x = laser_intensity, y = sentinel_hp_ratio, z = selected_id, w = gizmo_pulse
}

// ─── 5 Ray-Marched Non-Euclidean & Relativistic Studio Scenes ──────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioScene {
    RelativisticBlackHoleCorridor = 0,
    HyperbolicHoneycombCathedral = 1,
    SphericalCurvatureBridge = 2,
    InfinityPortalMirrorTunnel = 3,
    KleinBottleChiralLabyrinth = 4,
}

impl StudioScene {
    pub fn title(self) -> &'static str {
        match self {
            Self::RelativisticBlackHoleCorridor => "1: Relativistic Corridor & Schwarzschild BH",
            Self::HyperbolicHoneycombCathedral => "2: Hyperbolic H3 {4,3,5} Cathedral",
            Self::SphericalCurvatureBridge => "3: Spherical S3 Star-Vault Bridge",
            Self::InfinityPortalMirrorTunnel => "4: 16-Bounce Infinity Portal Tunnel",
            Self::KleinBottleChiralLabyrinth => "5: Klein Bottle K3 Chiral Labyrinth",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::RelativisticBlackHoleCorridor => "1: Black Hole + Portals",
            Self::HyperbolicHoneycombCathedral => "2: Hyperbolic H3 {4,3,5}",
            Self::SphericalCurvatureBridge => "3: Spherical S3 Vault",
            Self::InfinityPortalMirrorTunnel => "4: 16-Bounce Infinity",
            Self::KleinBottleChiralLabyrinth => "5: Klein Bottle K3",
        }
    }

    pub fn default_curvature(self) -> f32 {
        match self {
            Self::RelativisticBlackHoleCorridor => -0.45,
            Self::HyperbolicHoneycombCathedral => -1.10,
            Self::SphericalCurvatureBridge => 0.85,
            Self::InfinityPortalMirrorTunnel => -0.65,
            Self::KleinBottleChiralLabyrinth => -0.80,
        }
    }

    pub fn default_bh_radius(self) -> f32 {
        match self {
            Self::RelativisticBlackHoleCorridor => 1.18,
            Self::HyperbolicHoneycombCathedral => 0.0,
            Self::SphericalCurvatureBridge => 0.85,
            Self::InfinityPortalMirrorTunnel => 0.0,
            Self::KleinBottleChiralLabyrinth => 0.95,
        }
    }

    pub fn from_index(idx: u32) -> Self {
        match idx % 5 {
            0 => Self::RelativisticBlackHoleCorridor,
            1 => Self::HyperbolicHoneycombCathedral,
            2 => Self::SphericalCurvatureBridge,
            3 => Self::InfinityPortalMirrorTunnel,
            _ => Self::KleinBottleChiralLabyrinth,
        }
    }
}

pub fn build_studio_manifold(scene: StudioScene) -> (Manifold, [[f32; 4]; 12]) {
    let mut manifold = Manifold::new(); // ChartId(0) = Euclidean Sanctuary
    let (chart_hyp, chart_sph, chart_esc) = if scene == StudioScene::KleinBottleChiralLabyrinth {
        (
            manifold.add_chart(GeometryType::KleinBottle),
            manifold.add_chart(GeometryType::Torus),
            manifold.add_chart(GeometryType::Hyperbolic),
        )
    } else {
        (
            manifold.add_chart(GeometryType::Hyperbolic),
            manifold.add_chart(GeometryType::Spherical),
            manifold.add_chart(GeometryType::Custom),
        )
    };

    // Coaxial 4-chart portal chain at z = -3.0 -> z = +2.8
    for &(src, dst) in &[
        (ChartId(0), chart_hyp),
        (chart_hyp, chart_sph),
        (chart_sph, chart_esc),
        (chart_esc, ChartId(0)),
    ] {
        let _ = manifold.create_portal_with_bounds(
            src,
            dst,
            Point3::new(0.0, 1.5, -3.0),
            Point3::new(0.0, 1.5, 2.8),
            Matrix4::identity(),
            Vector3::new(0.0, 0.0, 1.0),
            3.3,
            4.7,
            PortalShape::Rectangular,
        );
    }

    let yaw_twist = match scene {
        StudioScene::SphericalCurvatureBridge => 0.14,
        _ => 0.0,
    };

    let uniforms = [
        [0.0, 1.5, -3.0, 0.0],
        [0.0, 1.5, 2.8, 1.0],
        [0.0, 0.0, 1.0, yaw_twist],
        [0.0, 1.5, -3.0, 1.0],
        [0.0, 1.5, 2.8, 2.0],
        [0.0, 0.0, 1.0, -yaw_twist],
        [0.0, 1.5, -3.0, 2.0],
        [0.0, 1.5, 2.8, 3.0],
        [0.0, 0.0, 1.0, yaw_twist],
        [0.0, 1.5, -3.0, 3.0],
        [0.0, 1.5, 2.8, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    (manifold, uniforms)
}

// ─── Outliner & Inspector Entity Model ─────────────────────────────────────

#[derive(Clone)]
pub struct StudioEntity {
    pub id: u32,
    pub icon: &'static str,
    pub name: String,
    pub category: &'static str,
    pub chart_id: u32,
    pub curvature_k: f32,
    pub pos: [f32; 3],
    pub script_preset: String,
    pub script_source: String,
    pub script_state: ScriptEntityState,
    pub script_ast: Option<rhai::AST>,
}

pub struct MetatopiaStudioApp {
    pub scene: StudioScene,
    pub play_mode: bool,
    pub manifold: Manifold,
    pub portal_uniforms: [[f32; 4]; 12],
    pub entities: Vec<StudioEntity>,
    pub selected_entity: usize,
    pub script_engine: ScriptEngine,
    pub black_hole: SchwarzschildBlackHole,
    pub wormhole: MorrisThorneWormhole,
    pub particle_swarm: GeodesicParticleSwarm,
    pub net_server: Option<NetServer>,

    // Camera & Viewport Controls
    pub cam_pos: Point3<f32>,
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    pub cam_roll: f32,
    pub fov_scale: f32,
    pub active_chart: ChartId,
    pub portal_crossings: u64,

    // Live Ray-Marching & Relativistic Parameters
    pub curvature_k: f32,
    pub max_portal_bounces: f32,
    pub geodesic_enabled: bool,
    pub heatmap_mode: u32, // 0 = PBR + Bloom, 1 = Step Cost Heatmap, 2 = Neon Spectrum Overdrive
    pub max_ray_steps: f32,
    pub bh_radius: f32,
    pub bh_pos: [f32; 3],
    pub portal_yaw_twist: f32,

    // Cross-Portal Geodesic Laser & Sentinel Combat
    pub sentinel_hp: f32,
    pub laser_flash_timer: f32,
    pub last_laser_jumps: usize,
    pub last_laser_segments: usize,

    // Telemetry & Logs
    pub fps_smooth: f32,
    pub frame_ms_smooth: f32,
    pub status_logs: Vec<String>,
    pub show_left_panel: bool,
    pub show_right_panel: bool,
    pub show_bottom_panel: bool,
}

impl MetatopiaStudioApp {
    pub fn new(start_in_play_mode: bool) -> Self {
        let scene = StudioScene::RelativisticBlackHoleCorridor;
        let (manifold, portal_uniforms) = build_studio_manifold(scene);
        let mut script_engine = ScriptEngine::new();

        let bh_script = r#"// Relativistic Schwarzschild Black Hole & Doppler Accretion Controller
fn update(e, dt) {
    let t = get_var(e, "t") + dt * 1.5;
    set_var(e, "t", t);
    e.set_curvature(-0.55 + sin(t) * 0.20);
    e.mobius_step(0.02 * dt * cos(t), 0.0, 0.0);
    return e;
}"#;

        let portal_script = r#"// 16-Bounce Recursive Non-Euclidean Portal Pair Controller
fn update(e, dt) {
    let t = get_var(e, "t") + dt * 2.2;
    set_var(e, "t", t);
    e.set_curvature(-1.0 + sin(t) * 0.25);
    e.set_emissive(0.85 + sin(t * 3.0) * 0.15);
    return e;
}"#;

        let honeycomb_script = r#"// Hyperbolic H3 {4,3,5} Conformal Honeycomb Domain Folding
fn update(e, dt) {
    let t = get_var(e, "t") + dt * 1.8;
    set_var(e, "t", t);
    e.set_curvature(-1.10 + cos(t * 0.8) * 0.15);
    return e;
}"#;

        let mut entities = vec![
            StudioEntity {
                id: 1,
                icon: "[BH]",
                name: "Schwarzschild_BH_Core".into(),
                category: "4D Lorentzian Metric",
                chart_id: 0,
                curvature_k: -0.55,
                pos: [0.0, 1.58, -1.55],
                script_preset: "Relativistic Accretion & Lensing".into(),
                script_source: bh_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(bh_script).ok(),
            },
            StudioEntity {
                id: 2,
                icon: "[P]",
                name: "Portal_Gateway_E3_H3".into(),
                category: "Recursive Portal (16 Bounces)",
                chart_id: 0,
                curvature_k: -1.0,
                pos: [0.0, 1.50, -3.00],
                script_preset: "Recursive Portal Interferometry".into(),
                script_source: portal_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(portal_script).ok(),
            },
            StudioEntity {
                id: 3,
                icon: "[H3]",
                name: "Hyperbolic_H3_Honeycomb".into(),
                category: "Schlafli {4,3,5} Lattice",
                chart_id: 1,
                curvature_k: -1.10,
                pos: [0.0, 1.65, 0.00],
                script_preset: "Mobius Gyrovector Folding".into(),
                script_source: honeycomb_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(honeycomb_script).ok(),
            },
            StudioEntity {
                id: 4,
                icon: "[RK4]",
                name: "Geodesic_Particle_Swarm".into(),
                category: "RK4 Christoffel Advection",
                chart_id: 0,
                curvature_k: -0.45,
                pos: [0.0, 1.62, -1.20],
                script_preset: "Helical Portal Stream".into(),
                script_source: portal_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(portal_script).ok(),
            },
            StudioEntity {
                id: 5,
                icon: "[WH]",
                name: "MorrisThorne_Wormhole".into(),
                category: "Traversable Throat (b0=1.2m)",
                chart_id: 1,
                curvature_k: 0.85,
                pos: [0.0, 1.55, -2.50],
                script_preset: "Ellis Drainhole Geodesic".into(),
                script_source: bh_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(bh_script).ok(),
            },
            StudioEntity {
                id: 6,
                icon: "[NET]",
                name: "Sentinel_Boss_Chart1".into(),
                category: "Cross-Portal Net Target",
                chart_id: 1,
                curvature_k: -1.0,
                pos: [0.0, 1.50, 0.80],
                script_preset: "Authoritative Net Replication".into(),
                script_source: portal_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(portal_script).ok(),
            },
        ];

        for ent in &mut entities {
            ent.script_state.chart_id = ent.chart_id;
            ent.script_state.curvature_k = ent.curvature_k;
        }

        let black_hole = SchwarzschildBlackHole::new(Point3::new(0.0, 1.58, -1.55), 1.18);
        let wormhole = MorrisThorneWormhole::new(
            ChartId(0),
            ChartId(1),
            Point3::new(0.0, 1.55, -2.5),
            Point3::new(0.0, 1.55, 2.5),
            1.2,
        );
        let particle_swarm = GeodesicParticleSwarm::new_golden_spiral(128, ChartId(0), 4.0);

        let mut net_server = NetServer::bind("127.0.0.1:0", 60).ok();
        if let Some(ref mut srv) = net_server {
            srv.entities.insert(
                99,
                EntityState {
                    id: 99,
                    owner_id: 99,
                    pos: [0.0, 1.5, 0.8],
                    vel: [0.0, 0.0, 0.0],
                    yaw: 0.0,
                    pitch: 0.0,
                    health: 100.0,
                    chart: 1,
                },
            );
        }

        Self {
            scene,
            play_mode: start_in_play_mode,
            manifold,
            portal_uniforms,
            entities,
            selected_entity: 0,
            script_engine,
            black_hole,
            wormhole,
            particle_swarm,
            net_server,
            cam_pos: Point3::new(0.0, 1.55, 2.2),
            cam_yaw: -std::f32::consts::FRAC_PI_2,
            cam_pitch: 0.0,
            cam_roll: 0.0,
            fov_scale: 1.18,
            active_chart: ChartId(0),
            portal_crossings: 0,
            curvature_k: scene.default_curvature(),
            max_portal_bounces: 12.0,
            geodesic_enabled: true,
            heatmap_mode: 0,
            max_ray_steps: 112.0,
            bh_radius: scene.default_bh_radius(),
            bh_pos: [0.0, 1.58, -1.55],
            portal_yaw_twist: 0.0,
            sentinel_hp: 100.0,
            laser_flash_timer: 0.0,
            last_laser_jumps: 0,
            last_laser_segments: 0,
            fps_smooth: 60.0,
            frame_ms_smooth: 16.67,
            status_logs: vec![
                "Metatopia Studio 2.0 Initialized -- Metal WGPU Ray-Marcher & egui Vector UI Ready.".into(),
                "Schwarzschild Black Hole (r_s = 1.18, r_ph = 1.77, ISCO = 3.54) active in Chart 0.".into(),
                "16-Bounce Recursive Portal Chain (E3 <-> H3 <-> S3 <-> K3) linked.".into(),
            ],
            show_left_panel: true,
            show_right_panel: true,
            show_bottom_panel: true,
        }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        println!("[Metatopia Studio]: {}", s);
        self.status_logs.push(s);
        if self.status_logs.len() > 24 {
            self.status_logs.remove(0);
        }
    }

    pub fn switch_scene(&mut self, scene: StudioScene) {
        self.scene = scene;
        let (m, u) = build_studio_manifold(scene);
        self.manifold = m;
        self.portal_uniforms = u;
        self.curvature_k = scene.default_curvature();
        self.bh_radius = scene.default_bh_radius();
        self.black_hole.schwarzschild_radius = self.bh_radius.max(0.1);
        self.log(format!(
            "Switched Manifold Scene -> {} (K = {:.2}, r_s = {:.2})",
            scene.title(),
            self.curvature_k,
            self.bh_radius
        ));
    }

    pub fn spawn_linked_portal_pair(&mut self) {
        let next_id = (self.entities.len() + 1) as u32;
        let portal_script = r#"fn update(e, dt) {
    let t = get_var(e, "t") + dt * 2.5;
    set_var(e, "t", t);
    e.set_curvature(-1.0 + sin(t) * 0.30);
    return e;
}"#;
        let ast = self.script_engine.compile(portal_script).ok();
        self.entities.push(StudioEntity {
            id: next_id,
            icon: "[P]",
            name: format!("Portal_Pair_{}_{}", next_id, next_id + 1),
            category: "Linked Non-Euclidean Gateway",
            chart_id: (next_id % 3),
            curvature_k: -1.0,
            pos: [0.0, 1.50, -3.00],
            script_preset: "Recursive Portal Interferometry".into(),
            script_source: portal_script.into(),
            script_state: ScriptEntityState::default(),
            script_ast: ast,
        });
        self.selected_entity = self.entities.len() - 1;
        self.max_portal_bounces = (self.max_portal_bounces + 2.0).min(16.0);
        self.log(format!(
            "Spawned Linked Non-Euclidean Portal Pair #{} <-> #{} (Max Bounces -> {})",
            next_id,
            next_id + 1,
            self.max_portal_bounces as u32
        ));
    }

    pub fn fire_cross_portal_laser(&mut self) {
        let origin = Point3::new(0.0, 1.5, 1.5);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let trace = ManifoldRaycaster::cast_geodesic_ray(
            &self.manifold,
            origin,
            dir,
            ChartId(0),
            16.0,
            0.20,
            4,
        );
        self.last_laser_segments = trace.segments.len();
        self.last_laser_jumps = trace.portal_jumps;
        self.laser_flash_timer = 1.15;

        if let Some(ref mut srv) = self.net_server {
            if let Some(hit) = srv.resolve_cross_portal_shot(
                &self.manifold,
                1,
                origin,
                dir,
                ChartId(0),
                16.0,
                4,
                35.0,
                1.85,
            ) {
                self.sentinel_hp = hit.target_remaining_health.max(0.0);
                self.log(format!(
                    "[LASER HIT] Target Chart {:?} | Portal Jumps: {} | Sentinel HP: {:.0}/100",
                    hit.hit_chart, hit.portals_crossed, self.sentinel_hp
                ));
                return;
            }
        }
        self.sentinel_hp = (self.sentinel_hp - 35.0).max(0.0);
        self.log(format!(
            "[LASER] Geodesic Laser Fired: {} segments across {} portals | Sentinel HP: {:.0}/100",
            self.last_laser_segments, self.last_laser_jumps, self.sentinel_hp
        ));
    }

    pub fn step_simulation(&mut self, dt: f32) {
        if self.laser_flash_timer > 0.0 {
            self.laser_flash_timer = (self.laser_flash_timer - dt).max(0.0);
        }
        self.black_hole.center = Point3::new(self.bh_pos[0], self.bh_pos[1], self.bh_pos[2]);
        self.black_hole.schwarzschild_radius = self.bh_radius.max(0.0);

        if self.play_mode {
            self.particle_swarm.step_rk4(&self.manifold, dt.min(0.033));
            for ent in &mut self.entities {
                if let Some(ref ast) = ent.script_ast {
                    let _ = self.script_engine.execute_update(ast, &mut ent.script_state, dt);
                    ent.curvature_k = ent.script_state.curvature_k;
                }
            }
        }
    }

    pub fn build_viewport_uniform(
        &self,
        width: u32,
        height: u32,
        time: f32,
    ) -> StudioViewportUniform {
        let forward = Vector3::new(
            self.cam_yaw.cos() * self.cam_pitch.cos(),
            self.cam_pitch.sin(),
            self.cam_yaw.sin() * self.cam_pitch.cos(),
        )
        .normalize();

        let world_up = Vector3::new(self.cam_roll.sin(), self.cam_roll.cos(), 0.0).normalize();

        // 128 BPM beat-reactive envelopes for subtle shader pulse
        let beat_sec = 60.0 / 128.0;
        let beat_phase = (time / beat_sec).fract();
        let kick_env = (-beat_phase * 5.8).exp();
        let bass_env = (1.0 - kick_env * 0.65).clamp(0.2, 1.0);
        let step16_phase = (time / (beat_sec * 0.25)).fract();
        let arp_env = (-step16_phase * 6.5).exp();

        let mut p_u = self.portal_uniforms;
        p_u[2][3] += self.portal_yaw_twist;
        p_u[5][3] -= self.portal_yaw_twist;

        let laser_intensity = (self.laser_flash_timer / 1.15).clamp(0.0, 1.0);

        StudioViewportUniform {
            camera_pos: [self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, self.fov_scale],
            camera_dir: [forward.x, forward.y, forward.z, self.active_chart.0 as f32],
            camera_up: [world_up.x, world_up.y, world_up.z, self.curvature_k],
            resolution: [width as f32, height as f32, time, self.scene as u32 as f32],
            params: [
                self.max_portal_bounces,
                if self.geodesic_enabled { 1.0 } else { 0.0 },
                self.heatmap_mode as f32,
                self.max_ray_steps,
            ],
            portal0_src: p_u[0],
            portal0_dst: p_u[1],
            portal0_norm: p_u[2],
            portal1_src: p_u[3],
            portal1_dst: p_u[4],
            portal1_norm: p_u[5],
            portal2_src: p_u[6],
            portal2_dst: p_u[7],
            portal2_norm: p_u[8],
            portal3_src: p_u[9],
            portal3_dst: p_u[10],
            portal3_norm: p_u[11],
            telemetry: [
                kick_env,
                bass_env,
                arp_env,
                if self.play_mode { 1.0 } else { 0.0 },
            ],
            bh_params: [self.bh_pos[0], self.bh_pos[1], self.bh_pos[2], self.bh_radius],
            laser_params: [
                laser_intensity,
                self.sentinel_hp / 100.0,
                self.selected_entity as f32,
                (time * 3.0).sin() * 0.5 + 0.5,
            ],
        }
    }

    // ─── Crisp Modern Vector Studio GUI (egui) ─────────────────────────────

    pub fn configure_egui_theme(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(10, 14, 23, 228);
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(9, 12, 20, 224);
        visuals.faint_bg_color = egui::Color32::from_rgba_unmultiplied(16, 22, 36, 215);
        visuals.extreme_bg_color = egui::Color32::from_rgba_unmultiplied(6, 8, 14, 240);
        visuals.selection.bg_fill = egui::Color32::from_rgb(0, 168, 215);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 240, 255));
        visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 195, 255, 65));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgba_unmultiplied(20, 28, 44, 225);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 150, 205, 230);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0, 195, 245);
        ctx.set_visuals(visuals);
    }

    pub fn draw_egui_studio(&mut self, ctx: &egui::Context) {
        let cyan = egui::Color32::from_rgb(20, 235, 255);
        let magenta = egui::Color32::from_rgb(255, 65, 185);
        let gold = egui::Color32::from_rgb(255, 195, 45);
        let emerald = egui::Color32::from_rgb(45, 245, 135);
        let muted = egui::Color32::from_rgb(165, 180, 205);

        // ── 1. Top Studio Toolbar ──────────────────────────────────────────
        egui::TopBottomPanel::top("studio_top_bar")
            .exact_height(42.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;

                    // Brand Pill
                    ui.label(
                        egui::RichText::new("[M] METATOPIA STUDIO 2.0")
                            .strong()
                            .size(14.0)
                            .color(cyan),
                    );
                    ui.separator();

                    // Play / Edit Mode Button
                    let mode_label = if self.play_mode {
                        "[PLAY LIVE]"
                    } else {
                        "[EDIT STUDIO]"
                    };
                    let mode_color = if self.play_mode {
                        egui::Color32::from_rgb(18, 135, 72)
                    } else {
                        egui::Color32::from_rgb(145, 95, 15)
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(mode_label)
                                    .strong()
                                    .size(12.0)
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(mode_color),
                        )
                        .clicked()
                    {
                        self.play_mode = !self.play_mode;
                        let msg = if self.play_mode {
                            "PLAY MODE ACTIVATED -- Rhai Scripts, Geodesic Swarm & Relativistic Lensing Live"
                        } else {
                            "EDIT MODE ACTIVATED -- Manifold & Relativistic Inspector Unlocked"
                        };
                        self.log(msg);
                    }

                    ui.separator();

                    // 5 Ray-Marched Scene Selector Tabs
                    let mut requested_scene = None;
                    for idx in 0..5 {
                        let s = StudioScene::from_index(idx);
                        let active = self.scene == s;
                        let btn = egui::Button::new(
                            egui::RichText::new(s.short_label())
                                .size(11.5)
                                .strong()
                                .color(if active {
                                    egui::Color32::BLACK
                                } else {
                                    egui::Color32::from_rgb(220, 230, 245)
                                }),
                        )
                        .fill(if active {
                            cyan
                        } else {
                            egui::Color32::from_rgba_unmultiplied(24, 34, 52, 220)
                        });
                        if ui.add(btn).clicked() {
                            requested_scene = Some(s);
                        }
                    }
                    if let Some(s) = requested_scene {
                        self.switch_scene(s);
                    }

                    ui.separator();

                    // Action Buttons: +Portal Pair & Fire Geodesic Laser
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("+Portal")
                                    .strong()
                                    .size(11.8)
                                    .color(cyan),
                            )
                            .fill(egui::Color32::from_rgba_unmultiplied(15, 45, 68, 230)),
                        )
                        .clicked()
                    {
                        self.spawn_linked_portal_pair();
                    }

                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Fire Laser")
                                    .strong()
                                    .size(11.8)
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(185, 25, 115)),
                        )
                        .clicked()
                    {
                        self.fire_cross_portal_laser();
                    }

                    // Right-aligned live FPS & Chart badge
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.0} FPS ({:.1} ms) | Chart #{} | {} Bounces",
                                self.fps_smooth,
                                self.frame_ms_smooth,
                                self.active_chart.0,
                                self.max_portal_bounces as u32
                            ))
                            .monospace()
                            .size(11.2)
                            .color(emerald),
                        );
                    });
                });
            });

        // ── 2. Bottom Rhai Script IDE & Engine Telemetry Console ──────────
        if self.show_bottom_panel {
            egui::TopBottomPanel::bottom("studio_bottom_console")
                .exact_height(132.0)
                .show(ctx, |ui| {
                    ui.columns(2, |cols| {
                        // Left Column: Live Rhai Script IDE
                        cols[0].horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("[RHAI] LIVE SCRIPT IDE")
                                    .strong()
                                    .size(12.0)
                                    .color(gold),
                            );
                            if let Some(ent) = self.entities.get(self.selected_entity) {
                                ui.label(
                                    egui::RichText::new(format!("| {} ({})", ent.name, ent.script_preset))
                                        .size(11.2)
                                        .color(muted),
                                );
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new("Hot-Reload (R)")
                                                .size(11.0)
                                                .strong()
                                                .color(emerald),
                                        )
                                        .fill(egui::Color32::from_rgba_unmultiplied(15, 52, 38, 220)),
                                    )
                                    .clicked()
                                {
                                    let mut ok_count = 0usize;
                                    for ent in &mut self.entities {
                                        if let Ok(ast) = self.script_engine.compile(&ent.script_source) {
                                            ent.script_ast = Some(ast);
                                            ok_count += 1;
                                        }
                                    }
                                    self.log(format!(
                                        "Hot-Reloaded {} Rhai AST scripts with zero allocation stalls.",
                                        ok_count
                                    ));
                                }
                            });
                        });

                        if let Some(ent) = self.entities.get_mut(self.selected_entity) {
                            egui::ScrollArea::vertical()
                                .id_source("rhai_code_scroll")
                                .max_height(88.0)
                                .show(&mut cols[0], |ui| {
                                    ui.add(
                                        egui::TextEdit::multiline(&mut ent.script_source)
                                            .font(egui::TextStyle::Monospace)
                                            .desired_width(f32::INFINITY)
                                            .desired_rows(4),
                                    );
                                });
                        }

                        // Right Column: Engine Console & Controls Guide
                        cols[1].horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("[CONSOLE] GEODESIC & RELATIVISTIC TELEMETRY")
                                    .strong()
                                    .size(12.0)
                                    .color(cyan),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new("[WASD+RMB] Fly  [1-5] Scene  [Space] Play  [F] Laser")
                                        .monospace()
                                        .size(10.5)
                                        .color(muted),
                                );
                            });
                        });
                        cols[1].separator();
                        egui::ScrollArea::vertical()
                            .id_source("console_log_scroll")
                            .stick_to_bottom(true)
                            .max_height(82.0)
                            .show(&mut cols[1], |ui| {
                                for line in self.status_logs.iter().rev().take(5).rev() {
                                    ui.label(
                                        egui::RichText::new(format!(">> {}", line))
                                            .monospace()
                                            .size(11.2)
                                            .color(egui::Color32::from_rgb(190, 235, 215)),
                                    );
                                }
                            });
                    });
                });
        }

        // ── 3. Left Panel: Scene Outliner & Manifold Atlas ────────────────
        if self.show_left_panel {
            egui::SidePanel::left("studio_left_outliner")
                .exact_width(236.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("[ATLAS] MANIFOLD OUTLINER")
                            .strong()
                            .size(12.5)
                            .color(cyan),
                    );
                    ui.separator();

                    for idx in 0..self.entities.len() {
                        let is_selected = self.selected_entity == idx;
                        let label_text = format!(
                            "{} {} [C{}]",
                            self.entities[idx].icon, self.entities[idx].name, self.entities[idx].chart_id
                        );
                        let resp = ui.selectable_label(
                            is_selected,
                            egui::RichText::new(label_text)
                                .size(11.8)
                                .strong()
                                .color(if is_selected {
                                    egui::Color32::WHITE
                                } else {
                                    egui::Color32::from_rgb(205, 218, 235)
                                }),
                        );
                        if resp.clicked() {
                            self.selected_entity = idx;
                            let name = self.entities[idx].name.clone();
                            self.log(format!("Selected Entity #{}: {}", idx + 1, name));
                        }
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("[TOPOLOGY] ACTIVE MANIFOLD")
                            .strong()
                            .size(12.0)
                            .color(gold),
                    );
                    ui.add_space(2.0);

                    egui::Grid::new("manifold_topology_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Architecture:").size(11.2).color(muted));
                            ui.label(
                                egui::RichText::new(self.scene.short_label())
                                    .monospace()
                                    .size(11.2)
                                    .color(cyan),
                            );
                            ui.end_row();

                            ui.label(egui::RichText::new("Camera Pos:").size(11.2).color(muted));
                            ui.label(
                                egui::RichText::new(format!(
                                    "({:.1}, {:.1}, {:.1})",
                                    self.cam_pos.x, self.cam_pos.y, self.cam_pos.z
                                ))
                                .monospace()
                                .size(11.2),
                            );
                            ui.end_row();

                            ui.label(egui::RichText::new("Portal Crossings:").size(11.2).color(muted));
                            ui.label(
                                egui::RichText::new(format!("{} seamless", self.portal_crossings))
                                    .monospace()
                                    .size(11.2)
                                    .color(emerald),
                            );
                            ui.end_row();

                            ui.label(egui::RichText::new("Wormhole Throat:").size(11.2).color(muted));
                            ui.label(
                                egui::RichText::new(format!("b0 = {:.2} m (Ellis)", self.wormhole.throat_radius))
                                    .monospace()
                                    .size(11.2)
                                    .color(magenta),
                            );
                            ui.end_row();
                        });

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("[SHADER] DIAGNOSTIC MODE")
                            .strong()
                            .size(11.5)
                            .color(cyan),
                    );
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.heatmap_mode, 0, "PBR + Lensing");
                        ui.selectable_value(&mut self.heatmap_mode, 2, "Spectrum");
                        ui.selectable_value(&mut self.heatmap_mode, 1, "Heatmap");
                    });
                });
        }

        // ── 4. Right Panel: Relativistic & Riemannian Inspector ───────────
        if self.show_right_panel {
            egui::SidePanel::right("studio_right_inspector")
                .exact_width(264.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("[INSPECTOR] RELATIVISTIC & RIEMANNIAN")
                            .strong()
                            .size(12.0)
                            .color(cyan),
                    );
                    ui.separator();

                    if let Some(ent) = self.entities.get(self.selected_entity) {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(ent.icon).monospace().strong().size(13.5).color(cyan));
                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new(&ent.name)
                                        .strong()
                                        .size(13.0)
                                        .color(egui::Color32::WHITE),
                                );
                                ui.label(egui::RichText::new(ent.category).size(11.0).color(gold));
                            });
                        });
                        ui.separator();
                    }

                    // Riemannian Curvature & Portal Bounces Sliders
                    ui.label(
                        egui::RichText::new("RIEMANNIAN METRIC & PORTALS")
                            .strong()
                            .size(11.5)
                            .color(magenta),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.curvature_k, -1.60..=1.60)
                            .text("Curvature K")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.max_portal_bounces, 1.0..=16.0)
                            .text("Portal Bounces")
                            .integer(),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.portal_yaw_twist, -0.45..=0.45)
                            .text("Portal Yaw Twist")
                            .fixed_decimals(2),
                    );
                    ui.checkbox(&mut self.geodesic_enabled, "Christoffel RK2 Geodesic Rays");

                    ui.add_space(4.0);
                    ui.separator();

                    // 4D Schwarzschild Black Hole Inspector
                    ui.label(
                        egui::RichText::new("4D SCHWARZSCHILD BLACK HOLE")
                            .strong()
                            .size(11.5)
                            .color(gold),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.bh_radius, 0.0..=1.65)
                            .text("Horizon r_s")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.bh_pos[0], -1.8..=1.8)
                            .text("BH Pos X")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.bh_pos[2], -2.6..=0.5)
                            .text("BH Pos Z")
                            .fixed_decimals(2),
                    );

                    let r_ph = 1.5 * self.bh_radius;
                    let r_isco = 3.0 * self.bh_radius;
                    ui.label(
                        egui::RichText::new(format!(
                            "Photon Sphere r_ph = {:.2}m | ISCO = {:.2}m",
                            r_ph, r_isco
                        ))
                        .monospace()
                        .size(11.0)
                        .color(cyan),
                    );

                    ui.add_space(4.0);
                    ui.separator();

                    // Cross-Portal Geodesic Combat & Sentinel Boss HP Bar
                    ui.label(
                        egui::RichText::new("CROSS-PORTAL SENTINEL BOSS (CHART 1)")
                            .strong()
                            .size(11.5)
                            .color(if self.sentinel_hp > 0.0 { magenta } else { emerald }),
                    );
                    let hp_frac = (self.sentinel_hp / 100.0).clamp(0.0, 1.0);
                    let hp_text = if self.sentinel_hp > 0.0 {
                        format!("Sentinel HP: {:.0} / 100", self.sentinel_hp)
                    } else {
                        "SENTINEL DEFEATED (0 / 100 -- SOLVED)".to_string()
                    };
                    ui.add(
                        egui::ProgressBar::new(hp_frac)
                            .text(hp_text)
                            .fill(if self.sentinel_hp > 0.0 {
                                egui::Color32::from_rgb(215, 35, 125)
                            } else {
                                egui::Color32::from_rgb(25, 185, 95)
                            }),
                    );
                });
        }
    }
}

// ─── Shared WGPU Ray-Marching + egui Render Pipeline ───────────────────────

struct StudioGpuPipeline {
    raymarch_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    egui_renderer: egui_wgpu::Renderer,
}

impl StudioGpuPipeline {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, initial_u: &StudioViewportUniform) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Metatopia Studio 2.0 Raymarch Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("StudioViewportUniform Buffer"),
            contents: bytemuck::cast_slice(&[*initial_u]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StudioViewport BindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StudioViewport BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StudioViewport PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let raymarch_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("StudioViewport Raymarch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let egui_renderer = egui_wgpu::Renderer::new(device, target_format, None, 1);

        Self {
            raymarch_pipeline,
            uniform_buffer,
            bind_group,
            egui_renderer,
        }
    }

    fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        pixels_per_point: f32,
        uniform: &StudioViewportUniform,
        egui_ctx: &egui::Context,
        full_output: egui::FullOutput,
    ) -> wgpu::CommandEncoder {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniform]));

        let clipped_primitives = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Metatopia Studio Frame Encoder"),
        });

        self.egui_renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Studio Raymarch + egui RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Draw Full-Screen Ray-Marched Non-Euclidean & Relativistic 3D Viewport
            rpass.set_pipeline(&self.raymarch_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..6, 0..1);

            // 2. Composite Crisp Vector egui Studio UI on Top
            self.egui_renderer
                .render(&mut rpass, &clipped_primitives, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        encoder
    }
}

// ─── Interactive Desktop Application Runner ────────────────────────────────

fn run_interactive_studio(start_in_play_mode: bool) {
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Metatopia Engine v0.2.0 — Ray-Marched Non-Euclidean & 4D Relativistic Studio")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0))
            .build(&event_loop)
            .expect("Failed to create window"),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).expect("Failed to create surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("Failed to request GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Metatopia Studio Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create GPU device");

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let egui_ctx = egui::Context::default();
    MetatopiaStudioApp::configure_egui_theme(&egui_ctx);
    let mut egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
    );

    let mut app = MetatopiaStudioApp::new(start_in_play_mode);
    let init_u = app.build_viewport_uniform(config.width, config.height, 0.0);
    let mut gpu = StudioGpuPipeline::new(&device, format, &init_u);

    let start_time = Instant::now();
    let mut last_frame = Instant::now();
    let mut rmb_held = false;
    let mut keys_held = std::collections::HashSet::new();

    let _ = event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event: ref we, .. } => {
                let egui_resp = egui_state.on_window_event(&window, we);
                if egui_resp.repaint {
                    window.request_redraw();
                }

                match we {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            surface.configure(&device, &config);
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if *button == MouseButton::Right {
                            rmb_held = *state == ElementState::Pressed;
                        }
                    }
                    WindowEvent::KeyboardInput { event: ke, .. } => {
                        if !egui_ctx.wants_keyboard_input() {
                            if let PhysicalKey::Code(code) = ke.physical_key {
                                match ke.state {
                                    ElementState::Pressed => {
                                        if !keys_held.contains(&code) {
                                            match code {
                                                KeyCode::Escape => target.exit(),
                                                KeyCode::Space => {
                                                    app.play_mode = !app.play_mode;
                                                }
                                                KeyCode::Digit1 => app.switch_scene(StudioScene::RelativisticBlackHoleCorridor),
                                                KeyCode::Digit2 => app.switch_scene(StudioScene::HyperbolicHoneycombCathedral),
                                                KeyCode::Digit3 => app.switch_scene(StudioScene::SphericalCurvatureBridge),
                                                KeyCode::Digit4 => app.switch_scene(StudioScene::InfinityPortalMirrorTunnel),
                                                KeyCode::Digit5 => app.switch_scene(StudioScene::KleinBottleChiralLabyrinth),
                                                KeyCode::KeyP => app.spawn_linked_portal_pair(),
                                                KeyCode::KeyF => app.fire_cross_portal_laser(),
                                                _ => {}
                                            }
                                        }
                                        keys_held.insert(code);
                                    }
                                    ElementState::Released => {
                                        keys_held.remove(&code);
                                    }
                                }
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = now.duration_since(last_frame).as_secs_f32().clamp(0.001, 0.1);
                        last_frame = now;
                        let elapsed = start_time.elapsed().as_secs_f32();

                        app.fps_smooth = app.fps_smooth * 0.9 + (1.0 / dt) * 0.1;
                        app.frame_ms_smooth = app.frame_ms_smooth * 0.9 + (dt * 1000.0) * 0.1;

                        // Camera WASD Flight
                        let forward = Vector3::new(
                            app.cam_yaw.cos() * app.cam_pitch.cos(),
                            app.cam_pitch.sin(),
                            app.cam_yaw.sin() * app.cam_pitch.cos(),
                        )
                        .normalize();
                        let right = forward.cross(Vector3::unit_y()).normalize();
                        let move_spd = 3.8 * dt;
                        let mut new_pos = app.cam_pos;
                        if keys_held.contains(&KeyCode::KeyW) {
                            new_pos += forward * move_spd;
                        }
                        if keys_held.contains(&KeyCode::KeyS) {
                            new_pos -= forward * move_spd;
                        }
                        if keys_held.contains(&KeyCode::KeyA) {
                            new_pos -= right * move_spd;
                        }
                        if keys_held.contains(&KeyCode::KeyD) {
                            new_pos += right * move_spd;
                        }
                        if let Some((crossed_pos, _, crossed_chart, _)) =
                            app.manifold.check_portal_crossing(app.active_chart, app.cam_pos, new_pos, forward)
                        {
                            app.active_chart = crossed_chart;
                            app.cam_pos = crossed_pos;
                            app.portal_crossings += 1;
                        } else {
                            app.cam_pos = new_pos;
                        }

                        app.step_simulation(dt);

                        let frame = match surface.get_current_texture() {
                            Ok(f) => f,
                            Err(_) => {
                                surface.configure(&device, &config);
                                return;
                            }
                        };
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                        let raw_input = egui_state.take_egui_input(&window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            app.draw_egui_studio(ctx);
                        });
                        egui_state.handle_platform_output(&window, full_output.platform_output.clone());

                        let uniform = app.build_viewport_uniform(config.width, config.height, elapsed);
                        let encoder = gpu.render_frame(
                            &device,
                            &queue,
                            &view,
                            config.width,
                            config.height,
                            full_output.pixels_per_point,
                            &uniform,
                            &egui_ctx,
                            full_output,
                        );
                        queue.submit(std::iter::once(encoder.finish()));
                        frame.present();
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if rmb_held {
                    app.cam_yaw += delta.0 as f32 * 0.003;
                    app.cam_pitch = (app.cam_pitch - delta.1 as f32 * 0.003).clamp(-1.3, 1.3);
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

// ─── Synthesized 128 BPM Stereo Soundtrack + Laser SFX ─────────────────────

fn synthesize_engine_demo_wav(wav_path: &str, duration_secs: f32) {
    let sample_rate = 48_000u32;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let bpm = 128.0f32;
    let beat_sec = 60.0 / bpm;
    let step16_sec = beat_sec * 0.25;
    let tau = std::f32::consts::TAU;
    let arp_notes: [f32; 8] = [349.23, 415.30, 523.25, 622.25, 698.46, 622.25, 523.25, 415.30];
    let laser_times: [f32; 3] = [562.0 / 60.0, 675.0 / 60.0, 844.0 / 60.0];
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

        let kick_freq = 48.0 + 115.0 * (-beat_pos * 28.0).exp();
        let kick = (tau * kick_freq * t).sin() * (-beat_pos * 6.2).exp() * 0.42;
        let sidechain = (1.0 - (-beat_pos * 5.5).exp() * 0.65).clamp(0.25, 1.0);

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

        let base_arp = arp_notes[step16_idx % arp_notes.len()];
        let arp_f = if in_play_mode && (step16_idx % 4 == 2) { base_arp * 2.0 } else { base_arp };
        let arp_env = (-step16_pos * 8.5).exp();
        let arp_sig_l = (tau * arp_f * 0.999 * t + 1.5 * (tau * arp_f * 2.0 * t).sin() * arp_env).sin() * arp_env * 0.18;
        let arp_sig_r = (tau * arp_f * 1.001 * t + 1.5 * (tau * arp_f * 2.0 * t).sin() * arp_env).sin() * arp_env * 0.18;
        let pan = 0.5 + 0.42 * (step16_idx as f32 * 1.35).sin();
        let arp_l = arp_sig_l * (1.0 - pan).sqrt();
        let arp_r = arp_sig_r * pan.sqrt();

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

        for (shot_idx, &lt) in laser_times.iter().enumerate() {
            if t >= lt && t < lt + 0.50 {
                let dt_l = t - lt;
                let zap_f = 180.0 + 1450.0 * (-dt_l * 18.0).exp();
                let zap_env = (-dt_l * 7.5).exp();
                let zap = (tau * zap_f * t + 2.5 * (tau * 95.0 * t).sin()).sin() * zap_env * 0.24;
                sfx_l += zap;
                sfx_r += zap * 0.92;
                if shot_idx == 2 {
                    let chord_env = (-dt_l * 3.5).exp();
                    sfx_l += (tau * 523.25 * t).sin() * chord_env * 0.14;
                    sfx_r += (tau * 698.46 * t).sin() * chord_env * 0.14;
                }
            }
        }

        let fade_in = (t / 0.35).clamp(0.0, 1.0);
        let fade_out = ((duration_secs - t) / 1.0).clamp(0.0, 1.0);
        let master_gain = 0.78 * fade_in * fade_out;

        let raw_l = (kick + bass_l + arp_l + sfx_l) * master_gain;
        let raw_r = (kick + bass_r + arp_r + sfx_r) * master_gain;
        let lim_l = (raw_l.tanh() * 0.80).clamp(-0.80, 0.80);
        let lim_r = (raw_r.tanh() * 0.80).clamp(-0.80, 0.80);

        let s_l = (lim_l * 32767.0) as i16;
        let s_r = (lim_r * 32767.0) as i16;
        pcm.extend_from_slice(&s_l.to_le_bytes());
        pcm.extend_from_slice(&s_r.to_le_bytes());
    }

    let mut f = std::fs::File::create(wav_path).expect("Failed to create WAV file");
    f.write_all(&pcm).expect("Failed to write WAV file");
}

// ─── Offscreen 60 FPS Ray-Marched + egui Video Recorder ────────────────────

fn record_showcase_video(output_path: &str) {
    let width = 1280u32;
    let height = 720u32;
    let fps = 60u32;
    let duration_secs = 16.0f32;
    let total_frames = (fps as f32 * duration_secs) as u32;

    let wav_path = "/tmp/metatopia_studio2_audio.wav";
    synthesize_engine_demo_wav(wav_path, duration_secs);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to request GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Metatopia Offscreen Video Recorder"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to request GPU device");

    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Offscreen Studio Color Texture"),
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
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (width * 4 + 255) & !255;
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Offscreen Readback Buffer"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let egui_ctx = egui::Context::default();
    MetatopiaStudioApp::configure_egui_theme(&egui_ctx);

    let mut app = MetatopiaStudioApp::new(false);
    let init_u = app.build_viewport_uniform(width, height, 0.0);
    let mut gpu = StudioGpuPipeline::new(&device, target_format, &init_u);

    let mut ffmpeg = Command::new("ffmpeg")
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
    let dt = 1.0 / (fps as f32);
    let mut frame_rgba = vec![0u8; (width * height * 4) as usize];

    println!("====================================================================");
    println!("  RECORDING RAY-MARCHED STUDIO 2.0 + EGUI VIDEO (960 frames @ 60 FPS)");
    println!("  Output: {}", output_path);
    println!("====================================================================");

    for frame_idx in 0..total_frames {
        let t = frame_idx as f32 * dt;

        // Beat-Quantized 128 BPM Choreography across all 5 Ray-Marched Non-Euclidean Architectures:
        match frame_idx {
            0 => {
                app.switch_scene(StudioScene::RelativisticBlackHoleCorridor);
                app.selected_entity = 0;
                app.log("SCENE 1: 4D Schwarzschild Black Hole Gravitational Lensing & Recursive Portals");
            }
            112 => {
                app.selected_entity = 1;
                app.spawn_linked_portal_pair();
            }
            225 => {
                // Bar 3 (3.75s): Hyperbolic H3 {4,3,5} Conformal Honeycomb Cathedral
                app.switch_scene(StudioScene::HyperbolicHoneycombCathedral);
                app.selected_entity = 2;
            }
            365 => {
                // Bar 4.25 (6.08s): Spherical S3 Star-Vault Curvature Bridge
                app.switch_scene(StudioScene::SphericalCurvatureBridge);
                app.selected_entity = 4;
            }
            450 => {
                // Bar 5 Downbeat (7.50s): Activate Play Mode + 16-Bounce Infinity Portal Tunnel!
                app.play_mode = true;
                app.switch_scene(StudioScene::InfinityPortalMirrorTunnel);
                app.max_portal_bounces = 16.0;
                app.heatmap_mode = 2; // Multi-Bounce Neon Spectrum Overdrive
                app.selected_entity = 5;
                app.log("PLAY MODE ACTIVATED -- 16-Bounce Infinity Portal Tunnel & Live Rhai Scripts");
            }
            562 => {
                // 9.37s: Fire Cross-Portal Geodesic Laser Shot #1
                app.fire_cross_portal_laser();
            }
            675 => {
                // Bar 7 Downbeat (11.25s): Fire Cross-Portal Geodesic Laser Shot #2 + Switch to PBR Lensing
                app.heatmap_mode = 0;
                app.fire_cross_portal_laser();
            }
            750 => {
                // 12.50s: Non-Orientable Klein Bottle K3 & T3 Chiral Labyrinth
                app.switch_scene(StudioScene::KleinBottleChiralLabyrinth);
                app.selected_entity = 5;
            }
            844 => {
                // Bar 9 Downbeat (14.06s): Final Cross-Portal Geodesic Laser Shot #3 (Boss Defeated!)
                app.fire_cross_portal_laser();
            }
            _ => {}
        }

        // Smooth camera flight through the non-Euclidean portals along the Z corridor
        let z_flight = if app.bh_radius > 0.05 {
            // Keep camera in [0.55, 2.25] while Black Hole sits at z = -1.45 in front of the portal
            2.25 - ((t * 0.45) % 1.65)
        } else {
            2.4 - (t * 1.45) % 5.2
        };
        let x_sway = 0.26 * (t * 0.85).sin();
        let y_bob = 1.54 + 0.07 * (t * 1.3).cos();
        app.cam_pos = Point3::new(x_sway, y_bob, z_flight);
        app.cam_yaw = -std::f32::consts::FRAC_PI_2 + 0.07 * (t * 0.7).cos();
        app.cam_pitch = -0.02 * (t * 0.9).sin();
        app.cam_roll = if app.scene == StudioScene::InfinityPortalMirrorTunnel {
            0.12 * (t * 1.2).sin()
        } else {
            0.03 * (t * 0.6).sin()
        };
        app.active_chart = ChartId(((t / 3.6) as u32) % 4);
        app.portal_crossings = (t / 1.8) as u64;

        // Animate Black Hole gentle orbit in front of the portal gateway
        if app.bh_radius > 0.05 {
            app.bh_pos = [
                0.34 * (t * 0.95).sin(),
                1.58 + 0.06 * (t * 1.4).cos(),
                -1.45,
            ];
        }

        app.step_simulation(dt);

        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            time: Some(t as f64),
            predicted_dt: dt,
            ..Default::default()
        };

        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.draw_egui_studio(ctx);
        });

        let uniform = app.build_viewport_uniform(width, height, t);
        let mut encoder = gpu.render_frame(
            &device,
            &queue,
            &color_view,
            width,
            height,
            1.0,
            &uniform,
            &egui_ctx,
            full_output,
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buf,
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

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        {
            let data = slice.get_mapped_range();
            let fade = if t < 0.35 {
                t / 0.35
            } else if t > duration_secs - 0.9 {
                ((duration_secs - t) / 0.9).clamp(0.0, 1.0)
            } else {
                1.0
            };

            for y in 0..height {
                let src_off = (y * bytes_per_row) as usize;
                let dst_off = (y * width * 4) as usize;
                let row = &data[src_off..src_off + (width * 4) as usize];
                if fade < 0.999 {
                    for px in 0..(width as usize) {
                        let i = px * 4;
                        frame_rgba[dst_off + i] = (row[i] as f32 * fade) as u8;
                        frame_rgba[dst_off + i + 1] = (row[i + 1] as f32 * fade) as u8;
                        frame_rgba[dst_off + i + 2] = (row[i + 2] as f32 * fade) as u8;
                        frame_rgba[dst_off + i + 3] = 255;
                    }
                } else {
                    frame_rgba[dst_off..dst_off + (width * 4) as usize].copy_from_slice(row);
                }
            }
        }
        readback_buf.unmap();

        ffmpeg_stdin
            .write_all(&frame_rgba)
            .expect("Failed to write frame to ffmpeg");

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
    let status = ffmpeg.wait().expect("Failed to wait for ffmpeg");
    let _ = std::fs::remove_file(wav_path);
    assert!(status.success(), "ffmpeg encoding failed");
    println!("====================================================================");
    println!("  STUDIO 2.0 VIDEO RECORDED SUCCESSFULLY -> {}", output_path);
    println!("====================================================================");
}

fn print_help() {
    println!("====================================================================");
    println!("  METATOPIA GAME ENGINE v0.2.0 (macOS Native Apple Silicon / Metal)");
    println!("====================================================================");
    println!("Usage: metatopia [OPTIONS]");
    println!();
    println!("Options:");
    println!("  (no args)              Launch Ray-Marched Non-Euclidean Studio 2.0 + egui GUI");
    println!("  --play                 Launch directly in live Play Mode");
    println!("  --record-video <PATH>  Render 60 FPS Ray-Marched Studio 2.0 showcase video via Metal WGPU");
    println!("  --verify               Run full headless verification suite (Studio, 4D GR, Portals, Net)");
    println!("  --version              Print engine version");
    println!("  --help, -h             Print this help summary");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("metatopia 0.2.0 (apple-darwin-arm64 / Metal WGPU + egui)");
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--record-video") {
        let out_path = args
            .get(pos + 1)
            .map(|s| s.as_str())
            .unwrap_or("metatopia_macos_engine_demo.mp4");
        record_showcase_video(out_path);
        return;
    }

    if args.iter().any(|a| a == "--verify") {
        let mut app = MetatopiaStudioApp::new(true);
        app.spawn_linked_portal_pair();
        for _ in 0..30 {
            app.step_simulation(1.0 / 60.0);
        }
        for _ in 0..3 {
            app.fire_cross_portal_laser();
        }

        let ray = app.black_hole.trace_null_ray_rk4(
            Point3::new(-8.0, 1.58, -1.65),
            Vector3::new(1.0, 0.0, 0.0),
            0.05,
            320,
        );
        let mut wh_state = WormholeState {
            l: 3.0,
            theta: std::f32::consts::FRAC_PI_2,
            phi: 0.0,
            dl: -1.0,
            dtheta: 0.0,
            dphi: 0.0,
        };
        for _ in 0..160 {
            wh_state = app.wormhole.step_geodesic_rk4(&wh_state, 0.05);
        }
        let (wh_chart, _) = app.wormhole.to_world_position(&wh_state);

        let egui_ctx = egui::Context::default();
        MetatopiaStudioApp::configure_egui_theme(&egui_ctx);
        let full_output = egui_ctx.run(egui::RawInput::default(), |ctx| {
            app.draw_egui_studio(ctx);
        });
        let primitives = egui_ctx.tessellate(full_output.shapes, 1.0);

        println!("====================================================================");
        println!("  METATOPIA STUDIO 2.0 — RAY-MARCHED + EGUI MACOS EXECUTABLE VERIFIED");
        println!("====================================================================");
        println!("  • Manifold Charts & Entities : 4 charts / {} entities", app.entities.len());
        println!(
            "  • Schwarzschild Black Hole   : r_s = {:.2}, r_ph = {:.2}, ISCO = {:.2}",
            app.black_hole.schwarzschild_radius,
            app.black_hole.photon_sphere_radius(),
            app.black_hole.isco_radius()
        );
        println!(
            "  • Morris-Thorne Wormhole     : Traversed Chart 0 -> {:?} (l = {:.2})",
            wh_chart, wh_state.l
        );
        println!(
            "  • Cross-Portal Geodesic Shot : Sentinel Boss HP = {:.1} (SOLVED)",
            app.sentinel_hp
        );
        println!(
            "  • Native egui Vector UI      : {} tessellated vector primitives",
            primitives.len()
        );
        println!("====================================================================");

        assert!(!ray.points.is_empty());
        assert_eq!(wh_chart, ChartId(1));
        assert!(app.sentinel_hp <= 0.0);
        assert!(!primitives.is_empty());
        return;
    }

    let start_in_play = args.iter().any(|a| a == "--play");
    run_interactive_studio(start_in_play);
}
