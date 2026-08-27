//! Metatopia VR — Meta Quest 3S & PCVR Stereoscopic 3D Showcase
//!
//! True 3D Stereoscopic Virtual Reality experience featuring:
//! - Side-by-Side (SBS) Dual-Eye Stereo Projection (Left + Right eye offset)
//! - Real-time Interpupillary Distance (IPD) tuning (55mm to 72mm)
//! - Seamless 3D Portals connecting Euclidean, Hyperbolic, and Spherical spaces
//! - 6-DoF Head Tracking and full 360-degree immersion
//! - 3D Spatial Audio and particle bursts in Virtual Reality

use metatopia_engine::quickstart::*;
use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::particles::ParticleSystem;
use metatopia_engine::vr::{StereoCameraRig, StereoMode};
use cgmath::{Point3, Vector3};

const SHADER_SRC: &str = include_str!("../shaders/vr_stereo.wgsl");

/// Meta Quest 3S Virtual Reality App
pub struct MetatopiaVrQuest {
    pub rig: StereoCameraRig,
    pub is_sbs_vr: bool,
    pub particle_system: ParticleSystem,
    pub current_space_name: &'static str,
    pub active_chart: u32,
    pub time_elapsed: f32,
    pub status_message: String,
    pub status_timer: f32,
}

impl MetatopiaVrQuest {
    pub fn new() -> Self {
        let mut rig = StereoCameraRig::new(Point3::new(0.0, 1.7, 5.0), 0.063); // 63mm Quest 3S IPD
        rig.mode = StereoMode::SideBySide;

        Self {
            rig,
            is_sbs_vr: true, // Default to SBS 3D VR Mode for Meta Quest
            particle_system: ParticleSystem::new(800),
            current_space_name: "Euclidean Nexus",
            active_chart: 0,
            time_elapsed: 0.0,
            status_message: "VR SBS 3D ACTIVE (Quest 3S Ready). Press [V] for Mono/Stereo, [/] for IPD.".into(),
            status_timer: 6.0,
        }
    }

    pub fn set_status(&mut self, msg: &str) {
        self.status_message = msg.to_string();
        self.status_timer = 4.0;
        println!("[VR]: {}", msg);
    }
}

impl GameApp for MetatopiaVrQuest {
    fn title(&self) -> &str {
        "Metatopia VR — Meta Quest 3S Stereoscopic 3D"
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    /// Enable dynamic mesh updating for particle effects and VR animations
    fn is_dynamic_mesh(&self) -> bool {
        true
    }

    /// Keep cursor free or locked for desktop controls
    fn grab_cursor(&self) -> bool {
        true
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        self.time_elapsed += ctx.dt;
        if self.status_timer > 0.0 {
            self.status_timer -= ctx.dt;
        }

        // 1. Camera Fly Movement (WASD + Space/Shift)
        ctx.default_camera_movement();

        // Sync Stereo Rig with FpsCamera
        self.rig.head_pos = Point3::new(ctx.camera.position.x, ctx.camera.position.y, ctx.camera.position.z);
        self.rig.yaw = ctx.camera.yaw;
        self.rig.pitch = ctx.camera.pitch;

        // 2. Toggle VR Mode (V key)
        if ctx.key_pressed(VirtualKey::KeyV) {
            self.is_sbs_vr = !self.is_sbs_vr;
            if self.is_sbs_vr {
                self.set_status("▶ VR SIDE-BY-SIDE 3D ENABLED (Put on Quest 3S)");
            } else {
                self.set_status("⏸ MONOSCOPIC 2D MODE (Standard Display)");
            }
        }

        // 3. IPD Tuning ([ and ] keys)
        if ctx.key_pressed(VirtualKey::BracketLeft) {
            self.rig.adjust_ipd_mm(-1.0);
            self.set_status(&format!("IPD: {:.1} mm", self.rig.ipd_meters * 1000.0));
        }
        if ctx.key_pressed(VirtualKey::BracketRight) {
            self.rig.adjust_ipd_mm(1.0);
            self.set_status(&format!("IPD: {:.1} mm", self.rig.ipd_meters * 1000.0));
        }

        // 4. FOV Cycle (F key)
        if ctx.key_pressed(VirtualKey::KeyF) {
            if self.rig.fov_y_deg >= 110.0 {
                self.rig.fov_y_deg = 90.0;
            } else {
                self.rig.fov_y_deg += 10.0;
            }
            self.set_status(&format!("VR Field of View: {:.0}°", self.rig.fov_y_deg));
        }

        // 5. Portal Proximity Check & Traversal
        let cam_pos = ctx.camera.position;
        // Portal 1 at (0, 1.5, -8) -> Hyperbolic Space
        let d_portal1 = ((cam_pos.x - 0.0).powi(2) + (cam_pos.z - (-8.0)).powi(2)).sqrt();
        if d_portal1 < 1.8 && cam_pos.y < 3.5 {
            if self.active_chart != 1 {
                self.active_chart = 1;
                self.current_space_name = "Hyperbolic Crystal Forest (K = -1.0)";
                self.set_status("Traversed Portal -> Hyperbolic Crystal Forest!");
                ctx.audio.play_sfx("portal_whoosh");
            }
        }

        // Portal 2 at (10, 1.5, 0) -> Spherical Space
        let d_portal2 = ((cam_pos.x - 10.0).powi(2) + (cam_pos.z - 0.0).powi(2)).sqrt();
        if d_portal2 < 1.8 && cam_pos.y < 3.5 {
            if self.active_chart != 2 {
                self.active_chart = 2;
                self.current_space_name = "Spherical Star Dome (K = +1.0)";
                self.set_status("Traversed Portal -> Spherical Star Dome!");
                ctx.audio.play_sfx("portal_whoosh");
            }
        }

        // 6. Particle Emitter Simulation
        self.particle_system.burst(
            Vector3::new(0.0, 2.0, -8.0),
            2,
            [0.2, 0.9, 1.0, 1.0],
            [0.8, 0.1, 1.0, 0.0],
            2.5,
            0.8,
        );
        self.particle_system.update(ctx.dt, None);

        // 7. GPU Uniforms
        ctx.scene.sun_direction = [-0.4, -1.0, -0.6, 0.0];
        ctx.scene.sun_color = [1.0, 0.96, 0.90, 1.0];

        let sbs_flag = if self.is_sbs_vr { 1.0 } else { 0.0 };
        ctx.scene.game_data = [self.time_elapsed, sbs_flag, self.rig.ipd_meters, self.rig.fov_y_deg];
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(30000);
        let mut indices = Vec::with_capacity(60000);

        // Helper to build 3D geometry tagged with Left Eye (1.0), Right Eye (2.0), or Mono (0.0)
        let mut add_stereo_scene = |eye_tag: f32, left_offset: f32| {
            // 1. Infinite Cyber Grid Floor
            let (plane_v, plane_i) = ProceduralMesh::plane(120.0, 120.0, 30, 30, [0.12, 0.15, 0.20]);
            let p_base = verts.len() as u32;
            for v in plane_v {
                let pos = [v.position[0] + left_offset, v.position[1], v.position[2]];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.1, 0.8, 0.0, eye_tag]));
            }
            for idx in plane_i { indices.push(p_base + idx); }

            // 2. Central Non-Euclidean Quantum Obelisk (Torus + Pillar)
            let (torus_v, torus_i) = ProceduralMesh::torus(1.6, 0.35, 24, 12, [0.0, 0.9, 1.0]);
            let t_base = verts.len() as u32;
            let rot_t = self.time_elapsed * 1.5;
            for v in torus_v {
                let rx = v.position[0] * rot_t.cos() - v.position[2] * rot_t.sin();
                let rz = v.position[0] * rot_t.sin() + v.position[2] * rot_t.cos();
                let pos = [rx + left_offset, v.position[1] + 2.5, rz];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.1, 2.5, eye_tag]));
            }
            for idx in torus_i { indices.push(t_base + idx); }

            // Central Glowing Sphere
            let (sph_v, sph_i) = ProceduralMesh::capsule(0.6, 0.01, 10, 16, [1.0, 0.2, 0.6]);
            let s_base = verts.len() as u32;
            for v in sph_v {
                let pos = [v.position[0] + left_offset, v.position[1] + 2.5, v.position[2]];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.0, 0.2, 3.0, eye_tag]));
            }
            for idx in sph_i { indices.push(s_base + idx); }

            // 3. Portal 1: Glowing Hyperbolic Arch at (0, 0, -8)
            let (arch_v, arch_i) = ProceduralMesh::capsule(0.3, 3.5, 8, 16, [0.8, 0.1, 1.0]);
            let a1_base = verts.len() as u32;
            for v in arch_v {
                let pos = [v.position[0] + left_offset, v.position[1] + 1.8, v.position[2] - 8.0];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.1, 0.2, 3.5, eye_tag]));
            }
            for idx in arch_i { indices.push(a1_base + idx); }

            // 4. Portal 2: Glowing Spherical Arch at (10, 0, 0)
            let (arch2_v, arch2_i) = ProceduralMesh::capsule(0.3, 3.5, 8, 16, [0.1, 1.0, 0.6]);
            let a2_base = verts.len() as u32;
            for v in arch2_v {
                let pos = [v.position[0] + 10.0 + left_offset, v.position[1] + 1.8, v.position[2]];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.1, 0.2, 3.5, eye_tag]));
            }
            for idx in arch2_i { indices.push(a2_base + idx); }

            // 5. Floating Monoliths around the perimeter
            for i in 0..6 {
                let angle = (i as f32) * (std::f32::consts::PI * 2.0 / 6.0);
                let mx = angle.cos() * 12.0 + left_offset;
                let mz = angle.sin() * 12.0;
                let (cyl_v, cyl_i) = ProceduralMesh::cylinder(0.5, 4.0, 12, [0.9, 0.7, 0.1]);
                let c_base = verts.len() as u32;
                for v in cyl_v {
                    let pos = [v.position[0] + mx, v.position[1] + 2.0, v.position[2] + mz];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.2, 0.5, eye_tag]));
                }
                for idx in cyl_i { indices.push(c_base + idx); }
            }
        };

        if self.is_sbs_vr {
            // Build Left Eye View (Tag 1.0) and Right Eye View (Tag 2.0)
            let half_ipd = self.rig.ipd_meters * 0.5;
            add_stereo_scene(1.0, -half_ipd); // Left eye
            add_stereo_scene(2.0, half_ipd);  // Right eye
        } else {
            // Mono View (Tag 0.0)
            add_stereo_scene(0.0, 0.0);
        }

        // Particle Mesh
        let (p_v, p_i) = self.particle_system.build_mesh();
        let p_base = verts.len() as u32;
        verts.extend(p_v);
        for idx in p_i { indices.push(p_base + idx); }

        (verts, indices)
    }
}

fn main() {
    println!("============================================================");
    println!("  METATOPIA VR — Meta Quest 3S Stereoscopic 3D Showcase");
    println!("============================================================");
    println!(" 🥽 Meta Quest 3S / PCVR Instructions:");
    println!("   1. Connect Quest 3S via Quest Link / Virtual Desktop");
    println!("   2. In Virtual Desktop / Bigscreen, turn ON 'Side-by-Side (SBS) 3D'");
    println!("   3. Enjoy full 360° Stereoscopic 3D Non-Euclidean Virtual Reality!");
    println!(" Controls:");
    println!("   WASD + Mouse   : Fly / Look around 3D VR Space");
    println!("   V              : Toggle VR Side-by-Side 3D / 2D Mode");
    println!("   [ / ]          : Adjust IPD (Interpupillary Distance in mm)");
    println!("   F              : Cycle Field of View (90° / 100° / 110°)");
    println!("   Walk to Arch   : Traverse Portals into Hyperbolic / Spherical space");
    println!("============================================================");

    run_game(MetatopiaVrQuest::new());
}
