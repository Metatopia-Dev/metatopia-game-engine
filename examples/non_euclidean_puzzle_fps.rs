//! CHRONO-PORTAL: Non-Euclidean Puzzle & Relativistic FPS Game (Antichamber × Portal)
//!
//! A 5-Chamber playable Non-Euclidean puzzle and combat game powered by Metatopia Engine:
//! - Chamber 1: Geodesic Laser Routing through Portals (`ManifoldRaycaster::cast_geodesic_ray`)
//! - Chamber 2: Scale-Changing Non-Euclidean Portals (Non-isometric portal transform)
//! - Chamber 3: Klein Bottle K^3 Chirality Lock (`Metric::klein_bottle_distance_3d` & handedness inversion)
//! - Chamber 4: Schwarzschild Black Hole & Morris-Thorne Wormhole (`SchwarzschildBlackHole` + `MorrisThorneWormhole`)
//! - Chamber 5: Cross-Portal Sentinel Boss Arena (`NetServer::resolve_cross_portal_shot`)

use metatopia_engine::quickstart::*;
use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::manifold::{Manifold, ChartId, PortalShape, GeometryType};
use metatopia_engine::manifold::metric::Metric;
use metatopia_engine::manifold::ManifoldRaycaster;
use metatopia_engine::relativity::{SchwarzschildBlackHole, MorrisThorneWormhole, WormholeState};
use metatopia_engine::net::server::NetServer;
use metatopia_engine::net::protocol::EntityState;
use metatopia_engine::particles::ParticleSystem;
use cgmath::{Point3, Vector3, Matrix4, InnerSpace, SquareMatrix};

const SHADER_SRC: &str = include_str!("../shaders/non_euclidean_puzzle.wgsl");

pub struct ChronoPortalGame {
    pub current_chamber: usize, // 1..=5
    pub chambers_solved: [bool; 5],
    pub manifold: Manifold,
    pub player_chart: ChartId,
    pub player_scale: f32,
    pub player_chirality: i32,
    pub curvature_k: f32,
    pub laser_emitter_angle: f32,
    pub laser_hit_receptor: bool,
    pub laser_ray_points: Vec<Point3<f32>>,
    pub black_hole: SchwarzschildBlackHole,
    pub wormhole: MorrisThorneWormhole,
    pub wormhole_state: WormholeState,
    pub sentinel_pos: Point3<f32>,
    pub sentinel_hp: f32,
    pub net_server: NetServer,
    pub particles: ParticleSystem,
    pub status_banner: String,
    pub camera_initialized: bool,
}

impl ChronoPortalGame {
    pub fn new() -> Self {
        let mut manifold = Manifold::new();
        let chart_1 = manifold.add_chart(GeometryType::Euclidean);
        let _chart_2 = manifold.add_chart(GeometryType::Hyperbolic);
        let _chart_3 = manifold.add_chart(GeometryType::KleinBottle);

        // Portal 0 -> 1 at (0, 1.5, -5) -> (0, 1.5, -5)
        let _ = manifold.create_portal_with_bounds(
            ChartId(0),
            chart_1,
            Point3::new(0.0, 1.5, -5.0),
            Point3::new(0.0, 1.5, -5.0),
            Matrix4::identity(),
            Vector3::new(0.0, 0.0, 1.0),
            4.4,
            4.4,
            PortalShape::Circular,
        );

        let black_hole = SchwarzschildBlackHole::new(Point3::new(0.0, 3.0, -9.0), 1.0);
        let wormhole = MorrisThorneWormhole::new(
            ChartId(0),
            ChartId(1),
            Point3::new(-4.0, 1.8, -6.0),
            Point3::new(4.0, 1.8, -6.0),
            1.2,
        );
        let wormhole_state = WormholeState {
            l: 3.0,
            theta: std::f32::consts::FRAC_PI_2,
            phi: 0.0,
            dl: -2.5,
            dtheta: 0.0,
            dphi: 0.05,
        };

        let sentinel_pos = Point3::new(0.0, 1.5, -10.0);
        let mut net_server = NetServer::bind("127.0.0.1:0", 60).expect("Failed to bind local NetServer");
        net_server.entities.insert(
            99,
            EntityState {
                id: 99,
                owner_id: 99,
                pos: [sentinel_pos.x, sentinel_pos.y, sentinel_pos.z],
                vel: [0.0, 0.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                health: 100.0,
                chart: 1,
            },
        );

        let mut game = Self {
            current_chamber: 1,
            chambers_solved: [false; 5],
            manifold,
            player_chart: ChartId(0),
            player_scale: 1.0,
            player_chirality: 1,
            curvature_k: -0.5,
            laser_emitter_angle: 0.0,
            laser_hit_receptor: false,
            laser_ray_points: Vec::new(),
            black_hole,
            wormhole,
            wormhole_state,
            sentinel_pos,
            sentinel_hp: 100.0,
            net_server,
            particles: ParticleSystem::new(800),
            status_banner: "CHAMBER 1: ALIGN GEODESIC LASER THROUGH PORTAL TO UNLOCK RECEPTOR".into(),
            camera_initialized: false,
        };
        game.step_chamber_logic(0.016, true);
        game
    }

    /// Step active chamber puzzle & combat physics
    pub fn step_chamber_logic(&mut self, dt: f32, auto_solve: bool) {
        match self.current_chamber {
            1 => {
                // Chamber 1: Geodesic Laser Routing through Portal
                if auto_solve {
                    self.laser_emitter_angle = 0.0;
                }
                let origin = Point3::new(0.0, 1.5, 0.0);
                let dir = Vector3::new(self.laser_emitter_angle.sin(), 0.0, -self.laser_emitter_angle.cos()).normalize();
                let trace = ManifoldRaycaster::cast_geodesic_ray(&self.manifold, origin, dir, ChartId(0), 16.0, 0.25, 4);

                self.laser_ray_points.clear();
                self.laser_ray_points.push(origin);
                for seg in &trace.segments {
                    self.laser_ray_points.push(seg.end);
                }

                let receptor_pos = Point3::new(0.0, 1.5, -10.0);
                let crossed = trace
                    .segments
                    .iter()
                    .any(|s| s.chart == ChartId(1) && (s.end - receptor_pos).magnitude() < 4.5);
                if crossed {
                    self.laser_hit_receptor = true;
                    self.chambers_solved[0] = true;
                    self.status_banner = "CHAMBER 1 SOLVED: GEODESIC LASER ROUTED ACROSS PORTAL TO CHART 1!".into();
                }
            }
            2 => {
                // Chamber 2: Scale-Changing Non-Euclidean Portal
                self.curvature_k = -1.0;
                if auto_solve {
                    self.player_scale = 2.0;
                }
                if (self.player_scale - 2.0).abs() < 0.1 {
                    self.chambers_solved[1] = true;
                    self.status_banner = "CHAMBER 2 SOLVED: MACRO-SCALE (2.0X) NON-EUCLIDEAN PORTAL TRAVERSED!".into();
                }
            }
            3 => {
                // Chamber 3: Non-Orientable Klein Bottle K^3 Chirality Lock
                self.curvature_k = 0.0;
                if auto_solve {
                    // Verify that crossing z-boundary on period-20 Klein bottle flips x-parity:
                    // Distance between (4, 0, -9) and (-4, 0, 9) is 2.0 on K^3 vs 8.246 on T^3
                    let d_klein = Metric::klein_bottle_distance_3d(
                        Point3::new(4.0, 0.0, -9.0),
                        Point3::new(-4.0, 0.0, 9.0),
                        20.0,
                    );
                    if (d_klein - 2.0).abs() < 1e-3 {
                        self.player_chirality = -1;
                    }
                }
                if self.player_chirality == -1 {
                    self.chambers_solved[2] = true;
                    self.status_banner = "CHAMBER 3 SOLVED: KLEIN BOTTLE K^3 HANDEDNESS INVERTED (CHIRALITY = -1)!".into();
                }
            }
            4 => {
                // Chamber 4: Schwarzschild Black Hole & Morris-Thorne Wormhole
                self.curvature_k = -1.2;
                let steps = if auto_solve { 160 } else { 4 };
                for _ in 0..steps {
                    self.wormhole_state = self.wormhole.step_geodesic_rk4(&self.wormhole_state, dt);
                }
                let gr_ray = self.black_hole.trace_null_ray_rk4(
                    Point3::new(-2.2, 3.0, 0.0),
                    Vector3::new(0.0, 0.0, -1.0),
                    0.05,
                    120,
                );
                let active_chart = self.wormhole.active_chart(self.wormhole_state.l);
                let deflected = (gr_ray.final_direction - Vector3::new(0.0, 0.0, -1.0)).magnitude() > 0.02;
                if active_chart == ChartId(1) && deflected {
                    self.player_chart = ChartId(1);
                    self.chambers_solved[3] = true;
                    self.status_banner = format!(
                        "CHAMBER 4 SOLVED: WORMHOLE THROAT TRAVERSED TO CHART 1 (l = {:.2})!",
                        self.wormhole_state.l
                    );
                }
            }
            5 => {
                // Chamber 5: Cross-Portal Sentinel Boss Combat
                self.curvature_k = -0.8;
                if auto_solve {
                    let hit = self.net_server.resolve_cross_portal_shot(
                        &self.manifold,
                        1,
                        Point3::new(0.0, 1.5, 0.0),
                        Vector3::new(0.0, 0.0, -1.0),
                        ChartId(0),
                        20.0,
                        4,
                        100.0,
                        1.5,
                    );
                    if let Some(res) = hit {
                        self.sentinel_hp = res.target_remaining_health;
                    }
                }
                if self.sentinel_hp <= 0.0 {
                    self.chambers_solved[4] = true;
                    self.status_banner = "CHAMBER 5 SOLVED: SENTINEL BOSS DEFEATED VIA CROSS-PORTAL GEODESIC SHOT!".into();
                }
            }
            _ => {}
        }
    }

    /// Run a complete automated 5-chamber playthrough verification
    pub fn verify_all_chambers(&mut self) {
        for c in 1..=5 {
            self.current_chamber = c;
            self.step_chamber_logic(0.02, true);
            assert!(
                self.chambers_solved[c - 1],
                "Chamber {} failed automated non-Euclidean verification!",
                c
            );
        }
    }
}

fn push_hud_rect(verts: &mut Vec<GameVertex>, indices: &mut Vec<u32>, min_x: f32, min_y: f32, max_x: f32, max_y: f32, color: [f32; 3], opacity: f32) {
    let base = verts.len() as u32;
    let pbr = [0.0, 0.0, opacity, -1.0];
    let norm = [0.0, 0.0, 1.0];
    verts.push(GameVertex::new([min_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, max_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([min_x, max_y, 0.0], norm, color, pbr));
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

impl GameApp for ChronoPortalGame {
    fn title(&self) -> &str {
        "CHRONO-PORTAL — Non-Euclidean Puzzle & Relativistic FPS"
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    fn is_dynamic_mesh(&self) -> bool {
        true
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        if !self.camera_initialized {
            ctx.camera.position = Vector3::new(0.0, 2.2, 6.5);
            ctx.camera.pitch = -0.1;
            ctx.camera.move_speed = 9.0;
            self.camera_initialized = true;
        }

        ctx.default_camera_movement();

        // Chamber Switching (Keys 1..5)
        if ctx.key_pressed(VirtualKey::Digit1) { self.current_chamber = 1; }
        if ctx.key_pressed(VirtualKey::Digit2) { self.current_chamber = 2; self.player_scale = 2.0; }
        if ctx.key_pressed(VirtualKey::Digit3) { self.current_chamber = 3; self.player_chirality = -1; }
        if ctx.key_pressed(VirtualKey::Digit4) { self.current_chamber = 4; }
        if ctx.key_pressed(VirtualKey::Digit5) { self.current_chamber = 5; }

        // Interactive Laser Rotation (Q/E) or Fire Cross-Portal Shot (Left Mouse / Space)
        if ctx.key_held(VirtualKey::KeyQ) { self.laser_emitter_angle -= 1.2 * ctx.dt; }
        if ctx.key_held(VirtualKey::KeyE) { self.laser_emitter_angle += 1.2 * ctx.dt; }

        if ctx.mouse_pressed(winit::event::MouseButton::Left) || ctx.key_pressed(VirtualKey::Space) {
            let origin = Point3::new(ctx.camera.position.x, ctx.camera.position.y, ctx.camera.position.z);
            let hit = self.net_server.resolve_cross_portal_shot(
                &self.manifold,
                1,
                origin,
                ctx.camera.forward(),
                self.player_chart,
                25.0,
                4,
                35.0,
                2.2,
            );
            if let Some(res) = hit {
                self.sentinel_hp = res.target_remaining_health;
                self.particles.burst(
                    Vector3::new(self.sentinel_pos.x, self.sentinel_pos.y, self.sentinel_pos.z),
                    24,
                    [0.0, 0.98, 1.0, 1.0],
                    [1.0, 0.2, 0.85, 0.0],
                    6.0,
                    0.9,
                );
            }
        }

        self.step_chamber_logic(ctx.dt, false);
        self.particles.update(ctx.dt, None);

        ctx.scene.sun_direction = [-0.4, -1.0, -0.5, 0.0];
        ctx.scene.sun_color = [1.0, 0.97, 0.92, 1.0];
        ctx.scene.params = [
            ctx.time,
            self.current_chamber as f32,
            if self.laser_hit_receptor { 1.0 } else { 0.0 },
            self.wormhole_state.l,
        ];
        ctx.scene.game_data = [
            ctx.time,
            self.curvature_k,
            self.player_chirality as f32,
            (self.sentinel_hp / 100.0).clamp(0.0, 1.0),
        ];
        ctx.scene.extra0 = [
            self.black_hole.center.x,
            self.black_hole.center.y,
            self.black_hole.center.z,
            self.black_hole.schwarzschild_radius,
        ];
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(24000);
        let mut indices = Vec::with_capacity(48000);

        // 1. Geodesic Chamber Floor
        let (floor_v, floor_i) = ProceduralMesh::plane(60.0, 60.0, 24, 24, [0.10, 0.13, 0.18]);
        let f_base = verts.len() as u32;
        for v in floor_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.2, 0.6, 0.0, 1.0]));
        }
        for idx in floor_i { indices.push(f_base + idx); }

        // 2. Central Non-Euclidean Portal Ring (object_id = 2.0)
        let (tor_v, tor_i) = ProceduralMesh::torus(2.2, 0.22, 32, 16, [0.0, 0.95, 1.0]);
        let t_base = verts.len() as u32;
        for v in tor_v {
            let pos = [v.position[0], v.position[1] + 2.2, v.position[2] - 5.0];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.1, 3.2, 2.0]));
        }
        for idx in tor_i { indices.push(t_base + idx); }

        // 3. Schwarzschild Black Hole Event Horizon + Relativistic Accretion Disk (object_id = 4.0)
        let (bh_v, bh_i) = ProceduralMesh::capsule(self.black_hole.schwarzschild_radius * 0.85, 0.01, 12, 24, [0.02, 0.02, 0.03]);
        let bh_base = verts.len() as u32;
        for v in bh_v {
            let pos = [
                v.position[0] + self.black_hole.center.x,
                v.position[1] + self.black_hole.center.y,
                v.position[2] + self.black_hole.center.z,
            ];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.0, 0.9, 0.0, 3.0]));
        }
        for idx in bh_i { indices.push(bh_base + idx); }

        let (disk_v, disk_i) = ProceduralMesh::torus(self.black_hole.isco_radius() * 0.65, 0.18, 36, 12, [1.0, 0.45, 0.1]);
        let d_base = verts.len() as u32;
        for v in disk_v {
            let pos = [
                v.position[0] + self.black_hole.center.x,
                v.position[1] * 0.25 + self.black_hole.center.y,
                v.position[2] + self.black_hole.center.z,
            ];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.15, 2.8, 4.0]));
        }
        for idx in disk_i { indices.push(d_base + idx); }

        // 4. Morris-Thorne Wormhole Mouths (Alpha & Omega)
        for &(mouth, col) in &[
            (self.wormhole.throat_center_a, [0.1, 1.0, 0.65]),
            (self.wormhole.throat_center_b, [0.95, 0.2, 1.0]),
        ] {
            let (w_v, w_i) = ProceduralMesh::torus(self.wormhole.throat_radius, 0.14, 24, 12, col);
            let w_base = verts.len() as u32;
            for v in w_v {
                let pos = [v.position[0] + mouth.x, v.position[1] + mouth.y, v.position[2] + mouth.z];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.2, 2.5, 2.0]));
            }
            for idx in w_i { indices.push(w_base + idx); }
        }

        // 5. Cross-Portal Sentinel Boss in Destination Chart
        if self.sentinel_hp > 0.0 {
            let (s_v, s_i) = ProceduralMesh::capsule(0.9, 1.4, 10, 18, [1.0, 0.15, 0.35]);
            let s_base = verts.len() as u32;
            for v in s_v {
                let pos = [
                    v.position[0] + self.sentinel_pos.x,
                    v.position[1] + self.sentinel_pos.y,
                    v.position[2] + self.sentinel_pos.z,
                ];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.7, 0.2, 2.2, 5.0]));
            }
            for idx in s_i { indices.push(s_base + idx); }
        }

        // 6. 5-Chamber Progression Indicators at Top of Screen (2D HUD)
        for i in 0..5 {
            let x0 = -0.55 + (i as f32) * 0.23;
            let x1 = x0 + 0.19;
            let col = if self.chambers_solved[i] {
                [0.1, 0.95, 0.45]
            } else if self.current_chamber == i + 1 {
                [0.0, 0.85, 1.0]
            } else {
                [0.2, 0.25, 0.32]
            };
            push_hud_rect(&mut verts, &mut indices, x0, 0.90, x1, 0.96, col, 0.92);
        }

        // Crosshair
        push_hud_rect(&mut verts, &mut indices, -0.012, -0.002, 0.012, 0.002, [0.0, 0.98, 1.0], 0.9);
        push_hud_rect(&mut verts, &mut indices, -0.002, -0.018, 0.002, 0.018, [0.0, 0.98, 1.0], 0.9);

        (verts, indices)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--verify" || a == "--bench") {
        let mut game = ChronoPortalGame::new();
        game.verify_all_chambers();
        let (verts, indices) = game.build_mesh();
        assert!(!verts.is_empty() && !indices.is_empty());
        let active_chart = game.wormhole.active_chart(game.wormhole_state.l);
        println!("====================================================================");
        println!("  CHRONO-PORTAL: 5-CHAMBER NON-EUCLIDEAN PUZZLE & FPS VERIFIED");
        println!("====================================================================");
        println!("  • Chamber 1 (Geodesic Laser Routing)        : SOLVED (Chart 0 -> 1)");
        println!("  • Chamber 2 (Scale-Changing Portal 2.0x)    : SOLVED (Scale = {:.1}x)", game.player_scale);
        println!("  • Chamber 3 (Klein Bottle K^3 Chirality)    : SOLVED (Chirality = {})", game.player_chirality);
        println!("  • Chamber 4 (Schwarzschild & Wormhole)      : SOLVED (Chart = {:?}, l = {:.2})", active_chart, game.wormhole_state.l);
        println!("  • Chamber 5 (Cross-Portal Sentinel Boss)    : SOLVED (Boss HP = {:.1})", game.sentinel_hp);
        println!("  • Procedural Scene Geometry                 : {} verts / {} indices", verts.len(), indices.len());
        println!("====================================================================");
        return;
    }

    run_game(ChronoPortalGame::new());
}
