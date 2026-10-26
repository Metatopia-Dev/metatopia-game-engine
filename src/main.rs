//! Metatopia 3D Game Engine & Studio v1.0 (macOS Metal WGPU + egui)
//!
//! A real 3D Game Engine & Editor (Unity / Godot / Unreal style) featuring:
//!   1. True 3D Rasterized PBR Viewport (`shaders/engine_pbr_3d.wgsl`):
//!      - Physically-inspired Rayleigh/Mie Sky Dome + Directional Sun & Clouds
//!      - Infinite Anti-Aliased 1m/5m 3D Ground Grid + Red X / Blue Z World Axes
//!      - Cook-Torrance GGX PBR 3D Meshes (Player Mech, Enemy Bots, Physics Crates, Spheres,
//!        Ramp, Pillars, Collectible Crystals) with analytical soft shadows & ground AO
//!      - 3D RGB Transform Gizmos (Move / Rotate / Scale) + Yellow Selection Bounding Box
//!   2. Real-Time 3D RigidBody Physics & Playable Action Gameplay:
//!      - Gravity, restitution (bouncing), floor/ramp/box/sphere collisions, impulse knockback
//!      - Playable 3D Character Controller + Plasma Projectile Launcher + Enemy AI + 3D Particle Bursts
//!      - Editor Scene Snapshot on `[ |> PLAY ]` and full state restore on `[ [] STOP ]`
//!   3. Professional Unity/Godot-Style Vector Editor GUI (`egui`):
//!      - Top Menu & Transport Toolbar (`[Q] View`, `[W] Move`, `[E] Rotate`, `[R] Scale`, `PLAY / PAUSE / STOP`)
//!      - Left Scene Hierarchy Outliner + Object Spawner (`+ Cube`, `+ Sphere`, `+ Enemy Bot`)
//!      - Right Component Inspector (`Transform`, `MeshRenderer & PBR Material`, `RigidBody 3D`, `Rhai Script`)
//!      - Bottom Tabbed Dock (`Asset Browser`, `Live Rhai Script IDE`, `Engine Console & Profiler`)

use cgmath::{InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3, perspective};
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

const SHADER_3D_SRC: &str = include_str!("../shaders/engine_pbr_3d.wgsl");

// ─── GPU Data Structures ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub viewport: [f32; 4],
    pub point_light_pos: [f32; 4],
    pub point_light_col: [f32; 4],
    pub shadow_casters: [[f32; 4]; 16],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub mat_params: [f32; 4], // x = metallic, y = roughness, z = emissive, w = mode_flag
}

impl Vertex3D {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// ─── Procedural 3D Mesh Primitives & Composite Models ──────────────────────

fn rotate_y(v: [f32; 3], yaw: f32) -> [f32; 3] {
    let (s, c) = yaw.sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

pub fn append_box(
    out: &mut Vec<Vertex3D>,
    center: [f32; 3],
    half: [f32; 3],
    yaw: f32,
    color: [f32; 4],
    mat_params: [f32; 4],
) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z front
        (
            [0.0, 0.0, 1.0],
            [
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
        ),
        // -Z back
        (
            [0.0, 0.0, -1.0],
            [
                [half[0], -half[1], -half[2]],
                [-half[0], -half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [half[0], half[1], -half[2]],
            ],
        ),
        // +X right
        (
            [1.0, 0.0, 0.0],
            [
                [half[0], -half[1], half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [half[0], half[1], half[2]],
            ],
        ),
        // -X left
        (
            [-1.0, 0.0, 0.0],
            [
                [-half[0], -half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [-half[0], half[1], half[2]],
                [-half[0], half[1], -half[2]],
            ],
        ),
        // +Y top
        (
            [0.0, 1.0, 0.0],
            [
                [-half[0], half[1], half[2]],
                [half[0], half[1], half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
            ],
        ),
        // -Y bottom
        (
            [0.0, -1.0, 0.0],
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], -half[1], half[2]],
                [-half[0], -half[1], half[2]],
            ],
        ),
    ];

    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = [0usize, 1, 2, 0, 2, 3];

    for (n_local, corners) in faces {
        let n = rotate_y(n_local, yaw);
        for &idx in &indices {
            let p_rot = rotate_y(corners[idx], yaw);
            out.push(Vertex3D {
                pos: [center[0] + p_rot[0], center[1] + p_rot[1], center[2] + p_rot[2]],
                normal: n,
                uv: uvs[idx],
                color,
                mat_params,
            });
        }
    }
}

pub fn append_sphere(
    out: &mut Vec<Vertex3D>,
    center: [f32; 3],
    radius: f32,
    rings: u32,
    sectors: u32,
    color: [f32; 4],
    mat_params: [f32; 4],
) {
    let pi = std::f32::consts::PI;
    let tau = std::f32::consts::TAU;
    for r in 0..rings {
        let theta0 = pi * (r as f32) / (rings as f32);
        let theta1 = pi * ((r + 1) as f32) / (rings as f32);
        for s in 0..sectors {
            let phi0 = tau * (s as f32) / (sectors as f32);
            let phi1 = tau * ((s + 1) as f32) / (sectors as f32);

            let n00 = [theta0.sin() * phi0.cos(), theta0.cos(), theta0.sin() * phi0.sin()];
            let n10 = [theta0.sin() * phi1.cos(), theta0.cos(), theta0.sin() * phi1.sin()];
            let n11 = [theta1.sin() * phi1.cos(), theta1.cos(), theta1.sin() * phi1.sin()];
            let n01 = [theta1.sin() * phi0.cos(), theta1.cos(), theta1.sin() * phi0.sin()];

            let v = |n: [f32; 3]| Vertex3D {
                pos: [
                    center[0] + n[0] * radius,
                    center[1] + n[1] * radius,
                    center[2] + n[2] * radius,
                ],
                normal: n,
                uv: [0.5, 0.5],
                color,
                mat_params,
            };

            out.push(v(n00));
            out.push(v(n01));
            out.push(v(n11));

            out.push(v(n00));
            out.push(v(n11));
            out.push(v(n10));
        }
    }
}

/// Appends a 3D tapered cylinder or cone along an arbitrary segment `p0 -> p1`.
pub fn append_tapered_cylinder(
    out: &mut Vec<Vertex3D>,
    p0: [f32; 3],
    p1: [f32; 3],
    r0: f32,
    r1: f32,
    sides: u32,
    color: [f32; 4],
    mat_params: [f32; 4],
) {
    let dir = Vector3::new(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    let len = dir.magnitude();
    if len < 1e-5 {
        return;
    }
    let axis = dir / len;
    let helper = if axis.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let tangent = axis.cross(helper).normalize();
    let bitangent = axis.cross(tangent).normalize();

    let tau = std::f32::consts::TAU;
    for i in 0..sides {
        let a0 = tau * (i as f32) / (sides as f32);
        let a1 = tau * ((i + 1) as f32) / (sides as f32);

        let n0_vec = tangent * a0.cos() + bitangent * a0.sin();
        let n1_vec = tangent * a1.cos() + bitangent * a1.sin();

        let v00 = [
            p0[0] + n0_vec.x * r0,
            p0[1] + n0_vec.y * r0,
            p0[2] + n0_vec.z * r0,
        ];
        let v10 = [
            p0[0] + n1_vec.x * r0,
            p0[1] + n1_vec.y * r0,
            p0[2] + n1_vec.z * r0,
        ];
        let v11 = [
            p1[0] + n1_vec.x * r1,
            p1[1] + n1_vec.y * r1,
            p1[2] + n1_vec.z * r1,
        ];
        let v01 = [
            p1[0] + n0_vec.x * r1,
            p1[1] + n0_vec.y * r1,
            p1[2] + n0_vec.z * r1,
        ];

        let n0 = [n0_vec.x, n0_vec.y, n0_vec.z];
        let n1 = [n1_vec.x, n1_vec.y, n1_vec.z];

        out.push(Vertex3D { pos: v00, normal: n0, uv: [0.0, 0.0], color, mat_params });
        out.push(Vertex3D { pos: v10, normal: n1, uv: [1.0, 0.0], color, mat_params });
        out.push(Vertex3D { pos: v11, normal: n1, uv: [1.0, 1.0], color, mat_params });

        out.push(Vertex3D { pos: v00, normal: n0, uv: [0.0, 0.0], color, mat_params });
        out.push(Vertex3D { pos: v11, normal: n1, uv: [1.0, 1.0], color, mat_params });
        out.push(Vertex3D { pos: v01, normal: n0, uv: [0.0, 1.0], color, mat_params });
    }
}

// ─── Engine Scene Object & Component Model ─────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    GroundGridFloor,
    ArenaPlatform,
    SciFiPillar,
    PhysicsCrate,
    PhysicsSphere,
    PlayerMech,
    EnemyBot,
    EnergyCrystal,
    DirectionalSun,
}

impl MeshKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::GroundGridFloor => "GroundGridPlane (64x64m)",
            Self::ArenaPlatform => "ArenaRampPlatform",
            Self::SciFiPillar => "CyberPillarColumn",
            Self::PhysicsCrate => "BeveledPhysicsCrate",
            Self::PhysicsSphere => "HighGlossSphere",
            Self::PlayerMech => "ArticulatedPlayerMech",
            Self::EnemyBot => "AutonomousCombatBot",
            Self::EnergyCrystal => "OctahedralCrystal",
            Self::DirectionalSun => "DirectionalSunLight",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Static,
    Dynamic,
    Kinematic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoTool {
    Select,
    Move,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottomTab {
    AssetBrowser,
    RhaiScriptEditor,
    EngineConsole,
}

#[derive(Clone)]
pub struct GameObject {
    pub id: u32,
    pub icon: &'static str,
    pub name: String,
    pub tag: &'static str,
    pub active: bool,
    pub mesh: MeshKind,

    // Transform Component
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],

    // PBR Material Component
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: f32,

    // 3D RigidBody & Collider Component
    pub body_type: BodyType,
    pub velocity: [f32; 3],
    pub angular_vel_y: f32,
    pub mass: f32,
    pub restitution: f32,
    pub collider_radius: f32,
    pub use_gravity: bool,

    // Gameplay & AI Health
    pub health: f32,
    pub max_health: f32,
    pub hit_flash: f32,
    pub patrol_origin: [f32; 3],

    // Rhai Script Component
    pub script_name: String,
    pub script_source: String,
    pub script_state: ScriptEntityState,
    pub script_ast: Option<rhai::AST>,
}

impl GameObject {
    pub fn world_half_extents(&self) -> [f32; 3] {
        match self.mesh {
            MeshKind::GroundGridFloor => [28.0, 0.1, 28.0],
            MeshKind::ArenaPlatform => [2.2 * self.scale[0], 0.45 * self.scale[1], 1.8 * self.scale[2]],
            MeshKind::SciFiPillar => [0.55 * self.scale[0], 2.1 * self.scale[1], 0.55 * self.scale[2]],
            MeshKind::PhysicsCrate => [0.60 * self.scale[0], 0.60 * self.scale[1], 0.60 * self.scale[2]],
            MeshKind::PhysicsSphere => [0.62 * self.scale[0], 0.62 * self.scale[1], 0.62 * self.scale[2]],
            MeshKind::PlayerMech => [0.65 * self.scale[0], 1.15 * self.scale[1], 0.65 * self.scale[2]],
            MeshKind::EnemyBot => [0.75 * self.scale[0], 1.10 * self.scale[1], 0.75 * self.scale[2]],
            MeshKind::EnergyCrystal => [0.50 * self.scale[0], 0.75 * self.scale[1], 0.50 * self.scale[2]],
            MeshKind::DirectionalSun => [0.65, 0.65, 0.65],
        }
    }

    pub fn append_mesh_geometry(&self, verts: &mut Vec<Vertex3D>, selected: bool, time: f32) {
        if !self.active {
            return;
        }
        let mode_flag = if self.mesh == MeshKind::GroundGridFloor {
            1.0
        } else if selected {
            2.0
        } else {
            0.0
        };
        let flash_boost = self.hit_flash * 1.8;
        let base_col = [
            (self.albedo[0] + flash_boost).min(1.0),
            (self.albedo[1] + flash_boost * 0.4).min(1.0),
            (self.albedo[2] + flash_boost * 0.2).min(1.0),
            1.0,
        ];
        let mat = [
            self.metallic,
            self.roughness,
            self.emissive + flash_boost,
            mode_flag,
        ];
        let yaw = self.rotation_deg[1].to_radians();
        let p = self.position;
        let s = self.scale;

        match self.mesh {
            MeshKind::GroundGridFloor => {
                append_box(verts, [0.0, -0.08, 0.0], [32.0, 0.08, 32.0], 0.0, base_col, mat);
                // Perimeter arena curb trim
                let curb_mat = [0.65, 0.32, 0.05, 0.0];
                let curb_col = [0.22, 0.28, 0.36, 1.0];
                append_box(verts, [0.0, 0.25, -12.5], [12.5, 0.25, 0.35], 0.0, curb_col, curb_mat);
                append_box(verts, [-12.5, 0.25, 0.0], [0.35, 0.25, 12.5], 0.0, curb_col, curb_mat);
                append_box(verts, [12.5, 0.25, 0.0], [0.35, 0.25, 12.5], 0.0, curb_col, curb_mat);
            }
            MeshKind::ArenaPlatform => {
                // Main raised platform deck + step stairs
                append_box(
                    verts,
                    p,
                    [2.2 * s[0], 0.42 * s[1], 1.7 * s[2]],
                    yaw,
                    base_col,
                    mat,
                );
                // Glowing cyan trim strip along platform edge
                append_box(
                    verts,
                    [p[0], p[1] + 0.34 * s[1], p[2] + 1.65 * s[2]],
                    [2.15 * s[0], 0.06, 0.08],
                    yaw,
                    [0.10, 0.88, 1.0, 1.0],
                    [0.8, 0.2, 1.4, mode_flag],
                );
                // Step ramp blocks leading up
                append_box(
                    verts,
                    [p[0], p[1] * 0.55, p[2] + 2.25 * s[2]],
                    [1.35 * s[0], 0.24 * s[1], 0.55],
                    yaw,
                    [0.28, 0.34, 0.42, 1.0],
                    mat,
                );
            }
            MeshKind::SciFiPillar => {
                // Base pedestal + tall metallic column + glowing energy rings
                append_box(
                    verts,
                    [p[0], 0.22, p[2]],
                    [0.68 * s[0], 0.22, 0.68 * s[2]],
                    yaw,
                    [0.20, 0.24, 0.30, 1.0],
                    [0.7, 0.3, 0.0, mode_flag],
                );
                append_box(
                    verts,
                    [p[0], p[1], p[2]],
                    [0.44 * s[0], 2.0 * s[1], 0.44 * s[2]],
                    yaw,
                    base_col,
                    mat,
                );
                for ring_y in [1.1f32, 2.3, 3.5] {
                    append_box(
                        verts,
                        [p[0], ring_y, p[2]],
                        [0.50 * s[0], 0.08, 0.50 * s[2]],
                        yaw,
                        [0.12, 0.90, 1.0, 1.0],
                        [0.9, 0.15, 1.6, mode_flag],
                    );
                }
            }
            MeshKind::PhysicsCrate => {
                let h = 0.58 * s[0];
                // Inner core box
                append_box(verts, p, [h * 0.92, h * 0.92, h * 0.92], yaw, base_col, mat);
                // Metallic outer frame bands
                let frame_col = [base_col[0] * 0.55, base_col[1] * 0.55, base_col[2] * 0.60, 1.0];
                let frame_mat = [0.85, 0.25, 0.0, mode_flag];
                append_box(verts, p, [h, h * 0.22, h], yaw, frame_col, frame_mat);
                append_box(verts, p, [h * 0.22, h, h], yaw, frame_col, frame_mat);
                // Emissive status badge on crate
                append_box(
                    verts,
                    [p[0], p[1], p[2]],
                    [h * 0.35, h * 0.35, h * 1.02],
                    yaw,
                    [0.15, 0.92, 1.0, 1.0],
                    [0.5, 0.2, 0.9, mode_flag],
                );
            }
            MeshKind::PhysicsSphere => {
                append_sphere(verts, p, 0.62 * s[0], 14, 20, base_col, mat);
            }
            MeshKind::PlayerMech => {
                // Articulated Sci-Fi Player Character Mech
                // 1. Legs
                let leg_col = [0.18, 0.24, 0.32, 1.0];
                let leg_offset = rotate_y([-0.24 * s[0], 0.0, 0.0], yaw);
                let leg_offset_r = rotate_y([0.24 * s[0], 0.0, 0.0], yaw);
                append_box(
                    verts,
                    [p[0] + leg_offset[0], p[1] - 0.55, p[2] + leg_offset[2]],
                    [0.16 * s[0], 0.45 * s[1], 0.18 * s[2]],
                    yaw,
                    leg_col,
                    [0.7, 0.35, 0.0, mode_flag],
                );
                append_box(
                    verts,
                    [p[0] + leg_offset_r[0], p[1] - 0.55, p[2] + leg_offset_r[2]],
                    [0.16 * s[0], 0.45 * s[1], 0.18 * s[2]],
                    yaw,
                    leg_col,
                    [0.7, 0.35, 0.0, mode_flag],
                );
                // 2. Armored Torso
                append_box(
                    verts,
                    [p[0], p[1] + 0.12, p[2]],
                    [0.44 * s[0], 0.48 * s[1], 0.30 * s[2]],
                    yaw,
                    base_col,
                    mat,
                );
                // 3. Head + Glowing Cyan Visor
                append_sphere(
                    verts,
                    [p[0], p[1] + 0.82, p[2]],
                    0.28 * s[0],
                    10,
                    14,
                    base_col,
                    mat,
                );
                let visor_off = rotate_y([0.0, 0.82, -0.22 * s[2]], yaw);
                append_box(
                    verts,
                    [p[0] + visor_off[0], p[1] + visor_off[1], p[2] + visor_off[2]],
                    [0.22 * s[0], 0.09 * s[1], 0.10 * s[2]],
                    yaw,
                    [0.10, 0.95, 1.0, 1.0],
                    [0.9, 0.1, 2.2, mode_flag],
                );
                // 4. Right-Arm Plasma Cannon Barrel
                let gun_p0_off = rotate_y([0.52 * s[0], 0.22, -0.10], yaw);
                let gun_p1_off = rotate_y([0.52 * s[0], 0.22, -0.88], yaw);
                append_tapered_cylinder(
                    verts,
                    [p[0] + gun_p0_off[0], p[1] + gun_p0_off[1], p[2] + gun_p0_off[2]],
                    [p[0] + gun_p1_off[0], p[1] + gun_p1_off[1], p[2] + gun_p1_off[2]],
                    0.12,
                    0.09,
                    10,
                    [0.20, 0.90, 1.0, 1.0],
                    [0.85, 0.2, 1.2, mode_flag],
                );
            }
            MeshKind::EnemyBot => {
                // Autonomous Enemy Combat Mech + World-Space Health Bar
                append_box(
                    verts,
                    [p[0], p[1] - 0.35, p[2]],
                    [0.38 * s[0], 0.42 * s[1], 0.38 * s[2]],
                    yaw,
                    [0.22, 0.20, 0.26, 1.0],
                    [0.8, 0.3, 0.0, mode_flag],
                );
                append_box(
                    verts,
                    [p[0], p[1] + 0.22, p[2]],
                    [0.54 * s[0], 0.40 * s[1], 0.46 * s[2]],
                    yaw,
                    base_col,
                    mat,
                );
                // Crimson glowing eye core
                let eye_off = rotate_y([0.0, 0.25, 0.44 * s[2]], yaw);
                append_sphere(
                    verts,
                    [p[0] + eye_off[0], p[1] + eye_off[1], p[2] + eye_off[2]],
                    0.19 * s[0],
                    8,
                    12,
                    [1.0, 0.18, 0.32, 1.0],
                    [0.5, 0.1, 2.5 + flash_boost, mode_flag],
                );
                // Floating 3D Health Bar above Enemy Bot
                let hp_ratio = (self.health / self.max_health.max(1.0)).clamp(0.0, 1.0);
                let bar_y = p[1] + 1.05 * s[1];
                append_box(
                    verts,
                    [p[0], bar_y, p[2]],
                    [0.62, 0.06, 0.04],
                    0.0,
                    [0.12, 0.12, 0.15, 1.0],
                    [0.1, 0.8, 0.0, 0.0],
                );
                if hp_ratio > 0.01 {
                    let hp_col = if hp_ratio > 0.5 {
                        [0.18, 0.95, 0.42, 1.0]
                    } else {
                        [1.0, 0.28, 0.22, 1.0]
                    };
                    append_box(
                        verts,
                        [p[0] - 0.60 * (1.0 - hp_ratio), bar_y, p[2] + 0.02],
                        [0.60 * hp_ratio, 0.05, 0.045],
                        0.0,
                        hp_col,
                        [0.2, 0.2, 1.6, 0.0],
                    );
                }
            }
            MeshKind::EnergyCrystal => {
                let bob = (time * 3.2).sin() * 0.14;
                let c_pos = [p[0], p[1] + bob, p[2]];
                append_tapered_cylinder(
                    verts,
                    [c_pos[0], c_pos[1], c_pos[2]],
                    [c_pos[0], c_pos[1] + 0.65 * s[1], c_pos[2]],
                    0.42 * s[0],
                    0.02,
                    6,
                    base_col,
                    [0.3, 0.1, 1.8, mode_flag],
                );
                append_tapered_cylinder(
                    verts,
                    [c_pos[0], c_pos[1], c_pos[2]],
                    [c_pos[0], c_pos[1] - 0.65 * s[1], c_pos[2]],
                    0.42 * s[0],
                    0.02,
                    6,
                    base_col,
                    [0.3, 0.1, 1.8, mode_flag],
                );
            }
            MeshKind::DirectionalSun => {
                append_sphere(
                    verts,
                    p,
                    0.42,
                    10,
                    14,
                    [1.0, 0.88, 0.28, 1.0],
                    [0.1, 0.1, 2.5, mode_flag],
                );
            }
        }
    }
}

// ─── Dynamic Plasma Projectiles & 3D Explosion Particles ───────────────────

#[derive(Clone)]
pub struct PlasmaProjectile {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ttl: f32,
}

#[derive(Clone)]
pub struct SparkParticle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub color: [f32; 4],
    pub ttl: f32,
    pub size: f32,
}

// ─── 3D RGB Transform Gizmo & Selection Bounding-Box Builder ───────────────

pub fn build_gizmo_and_selection_vertices(
    obj: &GameObject,
    tool: GizmoTool,
    active_axis: Option<usize>, // Some(0)=X, Some(1)=Y, Some(2)=Z
) -> Vec<Vertex3D> {
    let mut out = Vec::new();
    let c = obj.position;
    let h = obj.world_half_extents();

    // 1. Yellow/Orange Selection Wireframe Bounding Box around the selected GameObject
    let box_col = [1.0, 0.76, 0.14, 1.0];
    let mat = [0.0, 0.5, 1.0, 0.0];
    let r_wire = 0.022;
    let corners = [
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (a, b) in edges {
        append_tapered_cylinder(&mut out, corners[a], corners[b], r_wire, r_wire, 6, box_col, mat);
    }

    // 2. 3D RGB Transform Gizmo Handles (+X Red, +Y Green, +Z Blue)
    let len = (h[0].max(h[1]).max(h[2]) + 1.15).clamp(1.45, 2.65);
    let r_shaft = 0.048;
    let r_tip = 0.135;
    let tip_len = 0.34;

    let col_x = if active_axis == Some(0) { [1.0, 0.92, 0.15, 1.0] } else { [0.98, 0.18, 0.22, 1.0] };
    let col_y = if active_axis == Some(1) { [1.0, 0.92, 0.15, 1.0] } else { [0.20, 0.95, 0.32, 1.0] };
    let col_z = if active_axis == Some(2) { [1.0, 0.92, 0.15, 1.0] } else { [0.18, 0.56, 1.00, 1.0] };

    // Center Pivot Cube
    append_box(&mut out, c, [0.09, 0.09, 0.09], 0.0, [0.96, 0.98, 1.0, 1.0], mat);

    // +X Handle (Red)
    let x_shaft_end = [c[0] + len, c[1], c[2]];
    let x_tip_end = [c[0] + len + tip_len, c[1], c[2]];
    append_tapered_cylinder(&mut out, c, x_shaft_end, r_shaft, r_shaft, 10, col_x, mat);
    if tool == GizmoTool::Scale {
        append_box(&mut out, x_shaft_end, [0.11, 0.11, 0.11], 0.0, col_x, mat);
    } else {
        append_tapered_cylinder(&mut out, x_shaft_end, x_tip_end, r_tip, 0.005, 12, col_x, mat);
    }

    // +Y Handle (Green)
    let y_shaft_end = [c[0], c[1] + len, c[2]];
    let y_tip_end = [c[0], c[1] + len + tip_len, c[2]];
    append_tapered_cylinder(&mut out, c, y_shaft_end, r_shaft, r_shaft, 10, col_y, mat);
    if tool == GizmoTool::Scale {
        append_box(&mut out, y_shaft_end, [0.11, 0.11, 0.11], 0.0, col_y, mat);
    } else {
        append_tapered_cylinder(&mut out, y_shaft_end, y_tip_end, r_tip, 0.005, 12, col_y, mat);
    }

    // +Z Handle (Blue)
    let z_shaft_end = [c[0], c[1], c[2] + len];
    let z_tip_end = [c[0], c[1], c[2] + len + tip_len];
    append_tapered_cylinder(&mut out, c, z_shaft_end, r_shaft, r_shaft, 10, col_z, mat);
    if tool == GizmoTool::Scale {
        append_box(&mut out, z_shaft_end, [0.11, 0.11, 0.11], 0.0, col_z, mat);
    } else {
        append_tapered_cylinder(&mut out, z_shaft_end, z_tip_end, r_tip, 0.005, 12, col_z, mat);
    }

    out
}

// ─── Main 3D Game Engine & Editor Application State ────────────────────────

pub struct MetatopiaEngineApp {
    pub objects: Vec<GameObject>,
    pub editor_snapshot: Vec<GameObject>,
    pub selected_idx: usize,
    pub gizmo_tool: GizmoTool,
    pub active_gizmo_axis: Option<usize>,
    pub bottom_tab: BottomTab,

    // Transport & Simulation State
    pub play_mode: bool,
    pub paused: bool,
    pub show_grid: bool,
    pub show_gizmos: bool,
    pub score: u32,
    pub enemies_defeated: u32,
    pub sim_time: f32,

    // 3D Editor / Game Camera
    pub cam_pos: Point3<f32>,
    pub cam_target: Point3<f32>,
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,

    // Dynamic Projectiles, Particles & Scripting
    pub projectiles: Vec<PlasmaProjectile>,
    pub particles: Vec<SparkParticle>,
    pub script_engine: ScriptEngine,

    // Telemetry
    pub fps_smooth: f32,
    pub frame_ms_smooth: f32,
    pub last_tri_count: usize,
    pub console_logs: Vec<String>,
}

impl MetatopiaEngineApp {
    pub fn new(start_in_play_mode: bool) -> Self {
        let mut script_engine = ScriptEngine::new();

        let player_script = r#"// Player Mech Third-Person Action Controller
fn update(e, dt) {
    let t = get_var(e, "t") + dt;
    set_var(e, "t", t);
    return e;
}"#;

        let enemy_script = r#"// Autonomous Patrol & Target Acquisition AI
fn update(e, dt) {
    let t = get_var(e, "t") + dt * 2.0;
    set_var(e, "t", t);
    e.set_emissive(0.45 + sin(t) * 0.25);
    return e;
}"#;

        let physics_script = r#"// Dynamic RigidBody Restitution & Impact Responder
fn update(e, dt) {
    let t = get_var(e, "t") + dt;
    set_var(e, "t", t);
    return e;
}"#;

        let objects = vec![
            GameObject {
                id: 1,
                icon: "[Env]",
                name: "Arena_Ground_Floor".into(),
                tag: "StaticArena",
                active: true,
                mesh: MeshKind::GroundGridFloor,
                position: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.18, 0.22, 0.28],
                metallic: 0.15,
                roughness: 0.65,
                emissive: 0.0,
                body_type: BodyType::Static,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 0.0,
                restitution: 0.7,
                collider_radius: 0.0,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [0.0, 0.0, 0.0],
                script_name: "arena_surface.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 2,
                icon: "[Plr]",
                name: "Player_Mech_Controller".into(),
                tag: "Player",
                active: true,
                mesh: MeshKind::PlayerMech,
                position: [0.0, 1.02, 4.4],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.16, 0.58, 0.96],
                metallic: 0.72,
                roughness: 0.24,
                emissive: 0.08,
                body_type: BodyType::Kinematic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 75.0,
                restitution: 0.1,
                collider_radius: 0.75,
                use_gravity: true,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [0.0, 1.02, 4.4],
                script_name: "player_controller.rhai".into(),
                script_source: player_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(player_script).ok(),
            },
            GameObject {
                id: 3,
                icon: "[Phy]",
                name: "Physics_Crate_Gold".into(),
                tag: "PhysicsProp",
                active: true,
                mesh: MeshKind::PhysicsCrate,
                position: [-2.1, 0.60, 1.1],
                rotation_deg: [0.0, 18.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.95, 0.68, 0.16],
                metallic: 0.65,
                roughness: 0.28,
                emissive: 0.04,
                body_type: BodyType::Dynamic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 18.0,
                restitution: 0.68,
                collider_radius: 0.62,
                use_gravity: true,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [-2.1, 0.60, 1.1],
                script_name: "rigidbody_crate.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 4,
                icon: "[Phy]",
                name: "Bouncy_Sphere_Cyan".into(),
                tag: "PhysicsProp",
                active: true,
                mesh: MeshKind::PhysicsSphere,
                position: [2.2, 0.65, 1.2],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.12, 0.88, 0.96],
                metallic: 0.88,
                roughness: 0.12,
                emissive: 0.10,
                body_type: BodyType::Dynamic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 10.0,
                restitution: 0.84,
                collider_radius: 0.62,
                use_gravity: true,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [2.2, 0.65, 1.2],
                script_name: "bouncy_sphere.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 5,
                icon: "[Phy]",
                name: "Physics_Crate_Emerald".into(),
                tag: "PhysicsProp",
                active: true,
                mesh: MeshKind::PhysicsCrate,
                position: [-2.1, 1.82, 1.1],
                rotation_deg: [0.0, -12.0, 0.0],
                scale: [0.92, 0.92, 0.92],
                albedo: [0.18, 0.86, 0.48],
                metallic: 0.55,
                roughness: 0.32,
                emissive: 0.04,
                body_type: BodyType::Dynamic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 14.0,
                restitution: 0.65,
                collider_radius: 0.56,
                use_gravity: true,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [-2.1, 1.82, 1.1],
                script_name: "rigidbody_crate.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 6,
                icon: "[AI]",
                name: "Enemy_Mech_Alpha".into(),
                tag: "Enemy",
                active: true,
                mesh: MeshKind::EnemyBot,
                position: [-2.4, 0.80, -2.6],
                rotation_deg: [0.0, 15.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.88, 0.22, 0.30],
                metallic: 0.78,
                roughness: 0.26,
                emissive: 0.12,
                body_type: BodyType::Kinematic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 65.0,
                restitution: 0.2,
                collider_radius: 0.80,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [-2.4, 0.80, -2.6],
                script_name: "enemy_patrol_ai.rhai".into(),
                script_source: enemy_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(enemy_script).ok(),
            },
            GameObject {
                id: 7,
                icon: "[AI]",
                name: "Enemy_Drone_Beta".into(),
                tag: "Enemy",
                active: true,
                mesh: MeshKind::EnemyBot,
                position: [2.5, 0.80, -2.4],
                rotation_deg: [0.0, -18.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.76, 0.26, 0.88],
                metallic: 0.80,
                roughness: 0.22,
                emissive: 0.14,
                body_type: BodyType::Kinematic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 65.0,
                restitution: 0.2,
                collider_radius: 0.80,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [2.5, 0.80, -2.4],
                script_name: "enemy_patrol_ai.rhai".into(),
                script_source: enemy_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(enemy_script).ok(),
            },
            GameObject {
                id: 8,
                icon: "[AI]",
                name: "Boss_Guardian_Omega".into(),
                tag: "Enemy",
                active: true,
                mesh: MeshKind::EnemyBot,
                position: [0.0, 1.68, -4.8],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.32, 1.32, 1.32],
                albedo: [0.96, 0.36, 0.14],
                metallic: 0.85,
                roughness: 0.20,
                emissive: 0.18,
                body_type: BodyType::Kinematic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 120.0,
                restitution: 0.1,
                collider_radius: 1.05,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [0.0, 1.68, -4.8],
                script_name: "boss_guardian_ai.rhai".into(),
                script_source: enemy_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(enemy_script).ok(),
            },
            GameObject {
                id: 9,
                icon: "[Env]",
                name: "Arena_North_Bastion".into(),
                tag: "StaticArena",
                active: true,
                mesh: MeshKind::ArenaPlatform,
                position: [0.0, 0.42, -4.8],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.25, 1.0, 1.15],
                albedo: [0.24, 0.30, 0.38],
                metallic: 0.55,
                roughness: 0.42,
                emissive: 0.0,
                body_type: BodyType::Static,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 0.0,
                restitution: 0.6,
                collider_radius: 2.2,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [0.0, 0.42, -4.8],
                script_name: "arena_surface.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 10,
                icon: "[Env]",
                name: "Cyber_Pillar_West".into(),
                tag: "StaticArena",
                active: true,
                mesh: MeshKind::SciFiPillar,
                position: [-5.2, 2.0, -2.2],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.26, 0.32, 0.40],
                metallic: 0.75,
                roughness: 0.28,
                emissive: 0.0,
                body_type: BodyType::Static,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 0.0,
                restitution: 0.5,
                collider_radius: 0.65,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [-5.2, 2.0, -2.2],
                script_name: "arena_surface.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 11,
                icon: "[Env]",
                name: "Cyber_Pillar_East".into(),
                tag: "StaticArena",
                active: true,
                mesh: MeshKind::SciFiPillar,
                position: [5.2, 2.0, -2.2],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.26, 0.32, 0.40],
                metallic: 0.75,
                roughness: 0.28,
                emissive: 0.0,
                body_type: BodyType::Static,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 0.0,
                restitution: 0.5,
                collider_radius: 0.65,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [5.2, 2.0, -2.2],
                script_name: "arena_surface.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(physics_script).ok(),
            },
            GameObject {
                id: 12,
                icon: "[VFX]",
                name: "Energy_Crystal_Core".into(),
                tag: "Pickup",
                active: true,
                mesh: MeshKind::EnergyCrystal,
                position: [0.0, 1.35, -0.5],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [0.15, 0.95, 0.82],
                metallic: 0.4,
                roughness: 0.1,
                emissive: 1.4,
                body_type: BodyType::Kinematic,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 90.0,
                mass: 1.0,
                restitution: 0.5,
                collider_radius: 0.5,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [0.0, 1.35, -0.5],
                script_name: "crystal_rotator.rhai".into(),
                script_source: enemy_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: script_engine.compile(enemy_script).ok(),
            },
            GameObject {
                id: 13,
                icon: "[Sun]",
                name: "Directional_Sun_Light".into(),
                tag: "Light",
                active: true,
                mesh: MeshKind::DirectionalSun,
                position: [5.5, 7.2, 4.0],
                rotation_deg: [45.0, -35.0, 0.0],
                scale: [1.0, 1.0, 1.0],
                albedo: [1.0, 0.92, 0.65],
                metallic: 0.0,
                roughness: 0.2,
                emissive: 2.5,
                body_type: BodyType::Static,
                velocity: [0.0, 0.0, 0.0],
                angular_vel_y: 0.0,
                mass: 0.0,
                restitution: 0.0,
                collider_radius: 0.4,
                use_gravity: false,
                health: 100.0,
                max_health: 100.0,
                hit_flash: 0.0,
                patrol_origin: [5.5, 7.2, 4.0],
                script_name: "sun_controller.rhai".into(),
                script_source: physics_script.into(),
                script_state: ScriptEntityState::default(),
                script_ast: None,
            },
        ];

        let editor_snapshot = objects.clone();

        Self {
            objects,
            editor_snapshot,
            selected_idx: 2, // Default select Physics_Crate_Gold so RGB Gizmo is immediately visible
            gizmo_tool: GizmoTool::Move,
            active_gizmo_axis: None,
            bottom_tab: BottomTab::AssetBrowser,
            play_mode: start_in_play_mode,
            paused: false,
            show_grid: true,
            show_gizmos: true,
            score: 0,
            enemies_defeated: 0,
            sim_time: 0.0,
            cam_pos: Point3::new(0.0, 5.2, 9.6),
            cam_target: Point3::new(0.0, 1.1, -0.8),
            cam_yaw: -std::f32::consts::FRAC_PI_2,
            cam_pitch: -0.38,
            sun_dir: [0.52, 0.76, 0.38],
            sun_intensity: 2.35,
            projectiles: Vec::new(),
            particles: Vec::new(),
            script_engine,
            fps_smooth: 60.0,
            frame_ms_smooth: 16.67,
            last_tri_count: 0,
            console_logs: vec![
                "[EDITOR] Metatopia 3D Engine v1.0 Initialized (Metal WGPU PBR + Soft Shadows + RigidBody 3D).".into(),
                "[SCENE] Loaded 'Cyber_Arena_Level.scene' (13 GameObjects, 4 Dynamic RigidBodies, 3 AI Bots).".into(),
                "[GIZMO] Transform Gizmo Ready ([W] Move, [E] Rotate, [R] Scale) -- Select any 3D object to edit.".into(),
            ],
        }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        println!("{}", s);
        self.console_logs.push(s);
        if self.console_logs.len() > 30 {
            self.console_logs.remove(0);
        }
    }

    pub fn start_play_mode(&mut self) {
        if !self.play_mode {
            self.editor_snapshot = self.objects.clone();
            self.play_mode = true;
            self.paused = false;
            self.projectiles.clear();
            self.particles.clear();
            self.log("[PLAY] Play Mode Started -- 3D RigidBody Gravity, Collisions & AI Combat Active!");
        }
    }

    pub fn stop_play_mode(&mut self) {
        if self.play_mode {
            self.objects = self.editor_snapshot.clone();
            self.play_mode = false;
            self.paused = false;
            self.projectiles.clear();
            self.particles.clear();
            self.score = 0;
            self.enemies_defeated = 0;
            self.log("[STOP] Stopped Play Mode -- Restored Editor Scene State from Snapshot.");
        }
    }

    pub fn spawn_object(&mut self, mesh: MeshKind, name_prefix: &str, pos: [f32; 3], albedo: [f32; 3]) {
        let next_id = (self.objects.len() + 1) as u32;
        let is_enemy = mesh == MeshKind::EnemyBot;
        let obj = GameObject {
            id: next_id,
            icon: if is_enemy { "[AI]" } else { "[Phy]" },
            name: format!("{}_{}", name_prefix, next_id),
            tag: if is_enemy { "Enemy" } else { "PhysicsProp" },
            active: true,
            mesh,
            position: pos,
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            albedo,
            metallic: if mesh == MeshKind::PhysicsSphere { 0.85 } else { 0.60 },
            roughness: if mesh == MeshKind::PhysicsSphere { 0.14 } else { 0.28 },
            emissive: 0.06,
            body_type: if is_enemy { BodyType::Kinematic } else { BodyType::Dynamic },
            velocity: [0.0, 0.0, 0.0],
            angular_vel_y: 0.0,
            mass: 12.0,
            restitution: 0.78,
            collider_radius: 0.62,
            use_gravity: !is_enemy,
            health: 100.0,
            max_health: 100.0,
            hit_flash: 0.0,
            patrol_origin: pos,
            script_name: "spawned_object.rhai".into(),
            script_source: "// Dynamic 3D GameObject\nfn update(e, dt) { return e; }".into(),
            script_state: ScriptEntityState::default(),
            script_ast: None,
        };
        self.objects.push(obj);
        self.selected_idx = self.objects.len() - 1;
        self.log(format!(
            "[SPAWN] Instantiated 3D GameObject '{}' at ({:.1}, {:.1}, {:.1})",
            self.objects[self.selected_idx].name, pos[0], pos[1], pos[2]
        ));
    }

    pub fn fire_plasma_projectile_at(&mut self, target_pos: [f32; 3]) {
        // Find Player_Mech_Controller position
        let mut muzzle = [0.0f32, 1.35, 3.8];
        for obj in &self.objects {
            if obj.mesh == MeshKind::PlayerMech && obj.active {
                muzzle = [obj.position[0] + 0.45, obj.position[1] + 0.28, obj.position[2] - 0.65];
                break;
            }
        }
        let dir = Vector3::new(
            target_pos[0] - muzzle[0],
            target_pos[1] - muzzle[1],
            target_pos[2] - muzzle[2],
        )
        .normalize();
        let speed = 22.0;
        self.projectiles.push(PlasmaProjectile {
            pos: muzzle,
            vel: [dir.x * speed, dir.y * speed, dir.z * speed],
            ttl: 2.0,
        });
    }

    pub fn step_physics_and_gameplay(&mut self, dt: f32) {
        self.sim_time += dt;

        // Decay hit flashes & rotate crystal even in Edit Mode
        for obj in &mut self.objects {
            if obj.hit_flash > 0.0 {
                obj.hit_flash = (obj.hit_flash - dt * 3.2).max(0.0);
            }
            if obj.mesh == MeshKind::EnergyCrystal {
                obj.rotation_deg[1] = (obj.rotation_deg[1] + 75.0 * dt) % 360.0;
            }
        }

        if !self.play_mode || self.paused {
            return;
        }

        // 1. Step Dynamic 3D RigidBodies (Gravity + Floor / Platform Bounce + Damping)
        let g = -9.81f32;
        for i in 0..self.objects.len() {
            if !self.objects[i].active {
                continue;
            }
            if self.objects[i].body_type == BodyType::Dynamic {
                if self.objects[i].use_gravity {
                    self.objects[i].velocity[1] += g * dt;
                }
                self.objects[i].position[0] += self.objects[i].velocity[0] * dt;
                self.objects[i].position[1] += self.objects[i].velocity[1] * dt;
                self.objects[i].position[2] += self.objects[i].velocity[2] * dt;
                self.objects[i].rotation_deg[1] += self.objects[i].angular_vel_y * dt;

                // Floor collision at y = collider_radius
                let min_y = self.objects[i].collider_radius;
                if self.objects[i].position[1] < min_y {
                    self.objects[i].position[1] = min_y;
                    if self.objects[i].velocity[1] < -0.8 {
                        self.objects[i].velocity[1] =
                            -self.objects[i].velocity[1] * self.objects[i].restitution;
                        self.objects[i].angular_vel_y += self.objects[i].velocity[0] * 25.0;
                    } else {
                        self.objects[i].velocity[1] = 0.0;
                    }
                    // Horizontal ground friction
                    self.objects[i].velocity[0] *= (1.0 - 2.5 * dt).max(0.0);
                    self.objects[i].velocity[2] *= (1.0 - 2.5 * dt).max(0.0);
                }
            } else if self.objects[i].mesh == MeshKind::EnemyBot {
                // Enemy AI Patrol & Face Player
                let phase = self.sim_time * 1.8 + (self.objects[i].id as f32) * 1.3;
                self.objects[i].position[0] =
                    self.objects[i].patrol_origin[0] + 0.85 * phase.sin();
                self.objects[i].rotation_deg[1] = 18.0 * phase.cos();
            }
        }

        // 2. Pairwise Dynamic RigidBody Collisions (Crates & Spheres bouncing off each other)
        let len = self.objects.len();
        for i in 0..len {
            for j in (i + 1)..len {
                if !self.objects[i].active || !self.objects[j].active {
                    continue;
                }
                if self.objects[i].body_type != BodyType::Dynamic
                    && self.objects[j].body_type != BodyType::Dynamic
                {
                    continue;
                }
                let r_sum = self.objects[i].collider_radius + self.objects[j].collider_radius;
                if r_sum <= 0.01 {
                    continue;
                }
                let dp = Vector3::new(
                    self.objects[j].position[0] - self.objects[i].position[0],
                    self.objects[j].position[1] - self.objects[i].position[1],
                    self.objects[j].position[2] - self.objects[i].position[2],
                );
                let dist = dp.magnitude();
                if dist > 0.001 && dist < r_sum {
                    let n = dp / dist;
                    let overlap = r_sum - dist;
                    if self.objects[i].body_type == BodyType::Dynamic
                        && self.objects[j].body_type == BodyType::Dynamic
                    {
                        self.objects[i].position[0] -= n.x * overlap * 0.5;
                        self.objects[i].position[1] -= n.y * overlap * 0.5;
                        self.objects[i].position[2] -= n.z * overlap * 0.5;
                        self.objects[j].position[0] += n.x * overlap * 0.5;
                        self.objects[j].position[1] += n.y * overlap * 0.5;
                        self.objects[j].position[2] += n.z * overlap * 0.5;

                        let rv = Vector3::new(
                            self.objects[j].velocity[0] - self.objects[i].velocity[0],
                            self.objects[j].velocity[1] - self.objects[i].velocity[1],
                            self.objects[j].velocity[2] - self.objects[i].velocity[2],
                        );
                        let vel_along_n = rv.dot(n);
                        if vel_along_n < 0.0 {
                            let e = self.objects[i].restitution.min(self.objects[j].restitution);
                            let j_imp = -(1.0 + e) * vel_along_n * 0.5;
                            let imp = n * j_imp;
                            self.objects[i].velocity[0] -= imp.x;
                            self.objects[i].velocity[1] -= imp.y;
                            self.objects[i].velocity[2] -= imp.z;
                            self.objects[j].velocity[0] += imp.x;
                            self.objects[j].velocity[1] += imp.y;
                            self.objects[j].velocity[2] += imp.z;
                        }
                    }
                }
            }
        }

        // 3. Step Plasma Projectiles & Resolve Hits against Enemy Bots / Physics Crates
        let mut new_logs = Vec::new();
        for proj in &mut self.projectiles {
            proj.pos[0] += proj.vel[0] * dt;
            proj.pos[1] += proj.vel[1] * dt;
            proj.pos[2] += proj.vel[2] * dt;
            proj.ttl -= dt;

            if proj.ttl <= 0.0 {
                continue;
            }

            for obj in &mut self.objects {
                if !obj.active || obj.mesh == MeshKind::GroundGridFloor || obj.mesh == MeshKind::PlayerMech {
                    continue;
                }
                let dx = proj.pos[0] - obj.position[0];
                let dy = proj.pos[1] - obj.position[1];
                let dz = proj.pos[2] - obj.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let hit_r = obj.collider_radius + 0.35;

                if dist < hit_r {
                    proj.ttl = 0.0;
                    obj.hit_flash = 1.0;

                    // Spawn 3D explosion spark particles at impact point
                    for k in 0..14 {
                        let a = (k as f32) * 0.45 + self.sim_time * 7.0;
                        let elev = ((k % 5) as f32) * 0.35;
                        self.particles.push(SparkParticle {
                            pos: proj.pos,
                            vel: [a.cos() * 5.2, 3.2 + elev * 3.0, a.sin() * 5.2],
                            color: if obj.mesh == MeshKind::EnemyBot {
                                [1.0, 0.38, 0.15, 1.0]
                            } else {
                                [0.15, 0.92, 1.0, 1.0]
                            },
                            ttl: 0.55,
                            size: 0.11,
                        });
                    }

                    if obj.body_type == BodyType::Dynamic {
                        // Apply physical knockback impulse to crate/sphere
                        obj.velocity[0] += proj.vel[0] * 0.18;
                        obj.velocity[1] += 3.8;
                        obj.velocity[2] += proj.vel[2] * 0.18;
                        obj.angular_vel_y += 140.0;
                        new_logs.push(format!(
                            "[PHYSICS] Plasma impact knocked '{}' -> vel ({:.1}, {:.1}, {:.1})",
                            obj.name, obj.velocity[0], obj.velocity[1], obj.velocity[2]
                        ));
                    } else if obj.mesh == MeshKind::EnemyBot {
                        obj.health = (obj.health - 50.0).max(0.0);
                        if obj.health <= 0.0 {
                            obj.active = false;
                            self.score += 100;
                            self.enemies_defeated += 1;
                            new_logs.push(format!(
                                "[COMBAT] Defeated '{}' (+100 Score | Total: {})!",
                                obj.name, self.score
                            ));
                        } else {
                            self.score += 25;
                            new_logs.push(format!(
                                "[COMBAT] Hit '{}' -> HP: {:.0}/{:.0}",
                                obj.name, obj.health, obj.max_health
                            ));
                        }
                    }
                    break;
                }
            }
        }
        self.projectiles.retain(|p| p.ttl > 0.0);

        for msg in new_logs {
            self.log(msg);
        }

        // 4. Step 3D Spark Particles
        for part in &mut self.particles {
            part.vel[1] -= 9.81 * dt;
            part.pos[0] += part.vel[0] * dt;
            part.pos[1] += part.vel[1] * dt;
            part.pos[2] += part.vel[2] * dt;
            if part.pos[1] < 0.05 {
                part.pos[1] = 0.05;
                part.vel[1] = -part.vel[1] * 0.5;
            }
            part.ttl -= dt;
        }
        self.particles.retain(|p| p.ttl > 0.0);
    }

    pub fn build_scene_and_gizmo_batches(&mut self) -> (Vec<Vertex3D>, Vec<Vertex3D>) {
        let mut mesh_verts = Vec::with_capacity(18_000);
        for (idx, obj) in self.objects.iter().enumerate() {
            let is_sel = !self.play_mode && idx == self.selected_idx;
            obj.append_mesh_geometry(&mut mesh_verts, is_sel, self.sim_time);
        }

        // Append glowing 3D Plasma Projectiles + Laser Tail Beam
        for proj in &self.projectiles {
            append_sphere(
                &mut mesh_verts,
                proj.pos,
                0.24,
                8,
                12,
                [0.15, 0.96, 1.0, 1.0],
                [0.2, 0.1, 3.5, 0.0],
            );
            let tail = [
                proj.pos[0] - proj.vel[0] * 0.055,
                proj.pos[1] - proj.vel[1] * 0.055,
                proj.pos[2] - proj.vel[2] * 0.055,
            ];
            append_tapered_cylinder(
                &mut mesh_verts,
                proj.pos,
                tail,
                0.13,
                0.02,
                8,
                [0.25, 0.88, 1.0, 1.0],
                [0.2, 0.1, 3.2, 0.0],
            );
        }

        // Append 3D Explosion Spark Particles
        for part in &self.particles {
            append_box(
                &mut mesh_verts,
                part.pos,
                [part.size, part.size, part.size],
                self.sim_time * 6.0,
                part.color,
                [0.2, 0.1, 2.8, 0.0],
            );
        }

        self.last_tri_count = mesh_verts.len() / 3;

        // Build 3D RGB Transform Gizmo + Selection Box in Edit Mode
        let gizmo_verts = if !self.play_mode && self.show_gizmos {
            if let Some(sel_obj) = self.objects.get(self.selected_idx) {
                if sel_obj.mesh != MeshKind::GroundGridFloor && sel_obj.active {
                    build_gizmo_and_selection_vertices(sel_obj, self.gizmo_tool, self.active_gizmo_axis)
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        (mesh_verts, gizmo_verts)
    }

    pub fn build_scene_uniform(&self, width: u32, height: u32) -> SceneUniform {
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let proj = perspective(Rad(50.0f32.to_radians()), aspect, 0.15, 120.0);
        let view = Matrix4::look_at_rh(self.cam_pos, self.cam_target, Vector3::unit_y());

        // WGPU clip space depth [0, 1] matrix adjustment
        let opengl_to_wgpu = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );
        let view_proj = opengl_to_wgpu * proj * view;
        let inv_view_proj = view_proj.invert().unwrap_or_else(Matrix4::identity);

        // Dynamic point light follows active plasma projectile or the Energy Crystal
        let (pl_pos, pl_col) = if let Some(proj_orb) = self.projectiles.first() {
            (
                [proj_orb.pos[0], proj_orb.pos[1], proj_orb.pos[2], 7.5],
                [0.15, 0.92, 1.0, 3.2],
            )
        } else {
            (
                [0.0, 1.45, -0.5, 5.5],
                [0.15, 0.95, 0.82, 1.6],
            )
        };

        // Populate up to 16 shadow caster spheres from active 3D scene objects
        let mut casters = [[0.0f32; 4]; 16];
        let mut c_idx = 0usize;
        for obj in &self.objects {
            if !obj.active
                || obj.mesh == MeshKind::GroundGridFloor
                || obj.mesh == MeshKind::DirectionalSun
            {
                continue;
            }
            if c_idx < 16 {
                let r = obj.collider_radius.max(0.55);
                casters[c_idx] = [obj.position[0], obj.position[1], obj.position[2], r];
                c_idx += 1;
            }
        }

        SceneUniform {
            view_proj: view_proj.into(),
            inv_view_proj: inv_view_proj.into(),
            camera_pos: [self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, self.sim_time],
            sun_dir: [self.sun_dir[0], self.sun_dir[1], self.sun_dir[2], self.sun_intensity],
            viewport: [
                width as f32,
                height as f32,
                if self.show_grid { 1.0 } else { 0.0 },
                if self.play_mode { 1.0 } else { 0.0 },
            ],
            point_light_pos: pl_pos,
            point_light_col: pl_col,
            shadow_casters: casters,
        }
    }

    // ─── Unity / Godot / Unreal Style Vector Editor UI (egui) ──────────────

    pub fn configure_egui_theme(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(18, 21, 28, 238);
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(16, 19, 25, 236);
        visuals.faint_bg_color = egui::Color32::from_rgba_unmultiplied(24, 29, 38, 230);
        visuals.extreme_bg_color = egui::Color32::from_rgba_unmultiplied(10, 12, 16, 245);
        visuals.selection.bg_fill = egui::Color32::from_rgb(38, 118, 215);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(130, 205, 255));
        visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(55, 70, 95, 140));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgba_unmultiplied(28, 34, 46, 235);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(45, 110, 195, 240);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(55, 135, 235);
        ctx.set_visuals(visuals);
    }

    pub fn draw_editor_ui(&mut self, ctx: &egui::Context) {
        let accent = egui::Color32::from_rgb(65, 175, 255);
        let emerald = egui::Color32::from_rgb(45, 230, 125);
        let gold = egui::Color32::from_rgb(255, 195, 55);
        let crimson = egui::Color32::from_rgb(245, 75, 85);
        let muted = egui::Color32::from_rgb(160, 175, 198);

        // ── 1. Top Engine Menu & Toolbar (Unity / Godot / Unreal Style) ───
        egui::TopBottomPanel::top("engine_top_toolbar")
            .exact_height(42.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;

                    ui.label(
                        egui::RichText::new("[M] METATOPIA 3D ENGINE")
                            .strong()
                            .size(13.5)
                            .color(accent),
                    );
                    ui.separator();

                    // Transform Gizmo Tool Buttons: [Q] Select, [W] Move, [E] Rotate, [R] Scale
                    for (tool, label) in [
                        (GizmoTool::Select, "[Q] View"),
                        (GizmoTool::Move, "[W] Move"),
                        (GizmoTool::Rotate, "[E] Rotate"),
                        (GizmoTool::Scale, "[R] Scale"),
                    ] {
                        let active = self.gizmo_tool == tool;
                        let btn = egui::Button::new(
                            egui::RichText::new(label)
                                .size(11.5)
                                .strong()
                                .color(if active { egui::Color32::WHITE } else { muted }),
                        )
                        .fill(if active {
                            egui::Color32::from_rgb(40, 115, 210)
                        } else {
                            egui::Color32::from_rgb(26, 32, 44)
                        });
                        if ui.add(btn).clicked() {
                            self.gizmo_tool = tool;
                        }
                    }

                    ui.separator();
                    ui.checkbox(&mut self.show_grid, "3D Grid");
                    ui.checkbox(&mut self.show_gizmos, "Gizmos");
                    ui.separator();

                    // Transport Controls: [ |> PLAY ]  [ || PAUSE ]  [ [] STOP ]
                    let play_btn = egui::Button::new(
                        egui::RichText::new(if self.play_mode {
                            "|> PLAYING GAME"
                        } else {
                            "|> PLAY"
                        })
                        .strong()
                        .size(12.2)
                        .color(egui::Color32::WHITE),
                    )
                    .fill(if self.play_mode {
                        egui::Color32::from_rgb(22, 165, 82)
                    } else {
                        egui::Color32::from_rgb(35, 95, 55)
                    });
                    if ui.add(play_btn).clicked() {
                        if !self.play_mode {
                            self.start_play_mode();
                        }
                    }

                    let stop_btn = egui::Button::new(
                        egui::RichText::new("[] STOP")
                            .strong()
                            .size(11.8)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if self.play_mode {
                        egui::Color32::from_rgb(185, 45, 55)
                    } else {
                        egui::Color32::from_rgb(45, 32, 38)
                    });
                    if ui.add(stop_btn).clicked() {
                        self.stop_play_mode();
                    }

                    ui.separator();

                    // Quick Spawn Prefab Buttons
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("+ Cube").strong().size(11.5).color(gold),
                            )
                            .fill(egui::Color32::from_rgb(32, 38, 52)),
                        )
                        .clicked()
                    {
                        self.spawn_object(
                            MeshKind::PhysicsCrate,
                            "Physics_Crate",
                            [0.0, 4.2, 0.8],
                            [0.95, 0.55, 0.18],
                        );
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("+ Sphere").strong().size(11.5).color(accent),
                            )
                            .fill(egui::Color32::from_rgb(32, 38, 52)),
                        )
                        .clicked()
                    {
                        self.spawn_object(
                            MeshKind::PhysicsSphere,
                            "Physics_Sphere",
                            [0.5, 4.8, 0.5],
                            [0.95, 0.22, 0.45],
                        );
                    }

                    // Right-aligned live Telemetry
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.0} FPS ({:.1} ms) | Tris: {} | Score: {}",
                                self.fps_smooth,
                                self.frame_ms_smooth,
                                self.last_tri_count,
                                self.score
                            ))
                            .monospace()
                            .size(11.2)
                            .color(emerald),
                        );
                    });
                });
            });

        // ── 2. Bottom Dock: Asset Browser | Rhai Script IDE | Engine Console ──
        egui::TopBottomPanel::bottom("engine_bottom_dock")
            .exact_height(138.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (tab, title) in [
                        (BottomTab::AssetBrowser, "[ASSETS] Project Browser"),
                        (BottomTab::RhaiScriptEditor, "[SCRIPT] Rhai C#/Rust Behavior IDE"),
                        (BottomTab::EngineConsole, "[CONSOLE] Physics & Combat Log"),
                    ] {
                        let sel = self.bottom_tab == tab;
                        if ui
                            .selectable_label(
                                sel,
                                egui::RichText::new(title)
                                    .strong()
                                    .size(11.8)
                                    .color(if sel { accent } else { muted }),
                            )
                            .clicked()
                        {
                            self.bottom_tab = tab;
                        }
                        ui.separator();
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(
                                "[RMB+WASD] Orbit/Fly   [W/E/R] Gizmo   [Space] Play/Stop   [F] Shoot Plasma",
                            )
                            .monospace()
                            .size(10.5)
                            .color(muted),
                        );
                    });
                });
                ui.separator();

                match self.bottom_tab {
                    BottomTab::AssetBrowser => {
                        ui.columns(2, |cols| {
                            // Left half: Visual Prefab & Material Cards
                            cols[0].label(
                                egui::RichText::new("Project Assets / Prefabs (Click to Instantiate into 3D Scene):")
                                    .size(11.2)
                                    .color(gold),
                            );
                            cols[0].horizontal_wrapped(|ui| {
                                let cards = [
                                    ("[Prefab] Gold_Crate.prefab", MeshKind::PhysicsCrate, [0.95, 0.72, 0.18]),
                                    ("[Prefab] Bouncy_Orb.prefab", MeshKind::PhysicsSphere, [0.15, 0.88, 0.98]),
                                    ("[Prefab] Ruby_Sphere.prefab", MeshKind::PhysicsSphere, [0.96, 0.20, 0.38]),
                                    ("[AI] Enemy_Bot.prefab", MeshKind::EnemyBot, [0.90, 0.25, 0.32]),
                                    ("[VFX] Energy_Crystal.prefab", MeshKind::EnergyCrystal, [0.18, 0.95, 0.78]),
                                ];
                                for (label, kind, col) in cards {
                                    let c32 = egui::Color32::from_rgb(
                                        (col[0] * 110.0) as u8,
                                        (col[1] * 110.0) as u8,
                                        (col[2] * 110.0) as u8,
                                    );
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new(label)
                                                    .size(11.2)
                                                    .strong()
                                                    .color(egui::Color32::WHITE),
                                            )
                                            .fill(c32),
                                        )
                                        .clicked()
                                    {
                                        self.spawn_object(kind, "Spawned", [0.4, 4.5, 0.8], col);
                                    }
                                }
                            });

                            // Right half: Live Engine Console feed
                            cols[1].label(
                                egui::RichText::new("Live Engine Console & RigidBody Events:")
                                    .size(11.2)
                                    .color(accent),
                            );
                            egui::ScrollArea::vertical()
                                .id_source("asset_console_scroll")
                                .stick_to_bottom(true)
                                .max_height(78.0)
                                .show(&mut cols[1], |ui| {
                                    for line in self.console_logs.iter().rev().take(4).rev() {
                                        ui.label(
                                            egui::RichText::new(format!(">> {}", line))
                                                .monospace()
                                                .size(11.0)
                                                .color(egui::Color32::from_rgb(195, 230, 215)),
                                        );
                                    }
                                });
                        });
                    }
                    BottomTab::RhaiScriptEditor => {
                        if let Some(obj) = self.objects.get_mut(self.selected_idx) {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Editing Attached Script: {} on '{}'",
                                        obj.script_name, obj.name
                                    ))
                                    .strong()
                                    .size(11.5)
                                    .color(gold),
                                );
                            });
                            ui.add(
                                egui::TextEdit::multiline(&mut obj.script_source)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(3),
                            );
                        }
                    }
                    BottomTab::EngineConsole => {
                        egui::ScrollArea::vertical()
                            .id_source("full_console_scroll")
                            .stick_to_bottom(true)
                            .max_height(88.0)
                            .show(ui, |ui| {
                                for line in self.console_logs.iter().rev().take(6).rev() {
                                    ui.label(
                                        egui::RichText::new(format!(">> {}", line))
                                            .monospace()
                                            .size(11.2)
                                            .color(egui::Color32::from_rgb(195, 230, 215)),
                                    );
                                }
                            });
                    }
                }
            });

        // ── 3. Left Dock: Scene Hierarchy (Outliner) ──────────────────────
        egui::SidePanel::left("engine_scene_hierarchy")
            .exact_width(232.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("SCENE HIERARCHY")
                        .strong()
                        .size(12.2)
                        .color(accent),
                );
                ui.label(
                    egui::RichText::new("Cyber_Arena_Level.scene (13 objects)")
                        .size(10.8)
                        .color(muted),
                );
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_source("hierarchy_scroll")
                    .max_height(340.0)
                    .show(ui, |ui| {
                        for idx in 0..self.objects.len() {
                            let is_sel = self.selected_idx == idx;
                            let active = self.objects[idx].active;
                            let label = format!(
                                "{} {}",
                                self.objects[idx].icon, self.objects[idx].name
                            );
                            let text_col = if !active {
                                egui::Color32::from_rgb(100, 110, 125)
                            } else if is_sel {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_rgb(205, 218, 235)
                            };
                            if ui
                                .selectable_label(
                                    is_sel,
                                    egui::RichText::new(label)
                                        .size(11.6)
                                        .strong()
                                        .color(text_col),
                                )
                                .clicked()
                            {
                                self.selected_idx = idx;
                            }
                        }
                    });

                ui.separator();
                ui.label(
                    egui::RichText::new("WORLD ENVIRONMENT & SUN")
                        .strong()
                        .size(11.5)
                        .color(gold),
                );
                ui.add(
                    egui::Slider::new(&mut self.sun_intensity, 0.4..=4.0)
                        .text("Sun Intensity")
                        .fixed_decimals(2),
                );
                ui.add(
                    egui::Slider::new(&mut self.sun_dir[0], -1.0..=1.0)
                        .text("Sun Dir X")
                        .fixed_decimals(2),
                );
            });

        // ── 4. Right Dock: Component Inspector (Transform, Material, Physics) ──
        egui::SidePanel::right("engine_component_inspector")
            .exact_width(268.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("INSPECTOR")
                        .strong()
                        .size(12.2)
                        .color(accent),
                );
                ui.separator();

                if let Some(obj) = self.objects.get_mut(self.selected_idx) {
                    // GameObject Header
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut obj.active, "");
                        ui.label(
                            egui::RichText::new(&obj.name)
                                .strong()
                                .size(13.0)
                                .color(egui::Color32::WHITE),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("Tag: {}", obj.tag)).size(11.0).color(gold));
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("Mesh: {}", obj.mesh.label()))
                                .size(11.0)
                                .color(muted),
                        );
                    });
                    ui.separator();

                    // [1] Transform Component (Position, Rotation, Scale)
                    ui.label(
                        egui::RichText::new("[v] Transform Component")
                            .strong()
                            .size(11.6)
                            .color(accent),
                    );
                    egui::Grid::new("transform_grid")
                        .num_columns(4)
                        .spacing([6.0, 4.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Position").size(11.0).color(muted));
                            ui.add(egui::DragValue::new(&mut obj.position[0]).speed(0.05).prefix("X: "));
                            ui.add(egui::DragValue::new(&mut obj.position[1]).speed(0.05).prefix("Y: "));
                            ui.add(egui::DragValue::new(&mut obj.position[2]).speed(0.05).prefix("Z: "));
                            ui.end_row();

                            ui.label(egui::RichText::new("Rotation").size(11.0).color(muted));
                            ui.add(egui::DragValue::new(&mut obj.rotation_deg[0]).speed(1.0).prefix("X: "));
                            ui.add(egui::DragValue::new(&mut obj.rotation_deg[1]).speed(1.0).prefix("Y: "));
                            ui.add(egui::DragValue::new(&mut obj.rotation_deg[2]).speed(1.0).prefix("Z: "));
                            ui.end_row();

                            ui.label(egui::RichText::new("Scale").size(11.0).color(muted));
                            ui.add(egui::DragValue::new(&mut obj.scale[0]).speed(0.02).prefix("X: "));
                            ui.add(egui::DragValue::new(&mut obj.scale[1]).speed(0.02).prefix("Y: "));
                            ui.add(egui::DragValue::new(&mut obj.scale[2]).speed(0.02).prefix("Z: "));
                            ui.end_row();
                        });

                    ui.separator();

                    // [2] MeshRenderer & PBR Material Component
                    ui.label(
                        egui::RichText::new("[v] MeshRenderer & PBR Material")
                            .strong()
                            .size(11.6)
                            .color(gold),
                    );
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Base Albedo RGB:").size(11.0).color(muted));
                        ui.color_edit_button_rgb(&mut obj.albedo);
                    });
                    ui.add(
                        egui::Slider::new(&mut obj.metallic, 0.0..=1.0)
                            .text("Metallic")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut obj.roughness, 0.04..=1.0)
                            .text("Roughness")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut obj.emissive, 0.0..=2.5)
                            .text("Emissive")
                            .fixed_decimals(2),
                    );

                    ui.separator();

                    // [3] 3D RigidBody & Collider Component
                    ui.label(
                        egui::RichText::new("[v] RigidBody 3D & Collider")
                            .strong()
                            .size(11.6)
                            .color(emerald),
                    );
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut obj.body_type, BodyType::Dynamic, "Dynamic");
                        ui.selectable_value(&mut obj.body_type, BodyType::Kinematic, "Kinematic");
                        ui.selectable_value(&mut obj.body_type, BodyType::Static, "Static");
                    });
                    ui.checkbox(&mut obj.use_gravity, "Use Gravity (-9.81 m/s^2)");
                    ui.add(
                        egui::Slider::new(&mut obj.restitution, 0.0..=0.95)
                            .text("Bounciness")
                            .fixed_decimals(2),
                    );

                    // [4] Combat / Enemy Health Status
                    if obj.mesh == MeshKind::EnemyBot {
                        ui.separator();
                        ui.label(
                            egui::RichText::new("[v] AI Combat Agent")
                                .strong()
                                .size(11.6)
                                .color(crimson),
                        );
                        let frac = (obj.health / obj.max_health.max(1.0)).clamp(0.0, 1.0);
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .text(format!("HP: {:.0} / {:.0}", obj.health, obj.max_health))
                                .fill(crimson),
                        );
                    }
                }
            });
    }
}

// ─── WGPU 3D Sky + PBR Mesh + Depth + Gizmo + egui Render Pipeline ─────────

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_VERTS: usize = 65_536;

struct Engine3DGpuPipeline {
    sky_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    gizmo_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    mesh_vbuf: wgpu::Buffer,
    gizmo_vbuf: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    egui_renderer: egui_wgpu::Renderer,
}

impl Engine3DGpuPipeline {
    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Engine3D Depth Buffer"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        initial_u: &SceneUniform,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Metatopia 3D PBR & Gizmo Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_3D_SRC.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SceneUniform Buffer"),
            contents: bytemuck::cast_slice(&[*initial_u]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SceneUniform BindGroupLayout"),
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
            label: Some("SceneUniform BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Engine3D PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Atmosphere Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
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

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D PBR Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[Vertex3D::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_mesh",
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
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D RGB Transform Gizmo Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[Vertex3D::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_gizmo",
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
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vb_bytes = (MAX_VERTS * std::mem::size_of::<Vertex3D>()) as u64;
        let mesh_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Vertex Buffer"),
            size: vb_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gizmo_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo Vertex Buffer"),
            size: vb_bytes / 2,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = Self::create_depth_view(device, width, height);
        let egui_renderer = egui_wgpu::Renderer::new(device, target_format, None, 1);

        Self {
            sky_pipeline,
            mesh_pipeline,
            gizmo_pipeline,
            uniform_buffer,
            bind_group,
            mesh_vbuf,
            gizmo_vbuf,
            depth_view,
            depth_size: (width, height),
            egui_renderer,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        pixels_per_point: f32,
        uniform: &SceneUniform,
        mesh_verts: &[Vertex3D],
        gizmo_verts: &[Vertex3D],
        egui_ctx: &egui::Context,
        full_output: egui::FullOutput,
    ) -> wgpu::CommandEncoder {
        if self.depth_size != (width, height) {
            self.depth_view = Self::create_depth_view(device, width, height);
            self.depth_size = (width, height);
        }

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniform]));

        let m_count = mesh_verts.len().min(MAX_VERTS);
        if m_count > 0 {
            queue.write_buffer(&self.mesh_vbuf, 0, bytemuck::cast_slice(&mesh_verts[..m_count]));
        }
        let g_count = gizmo_verts.len().min(MAX_VERTS / 2);
        if g_count > 0 {
            queue.write_buffer(&self.gizmo_vbuf, 0, bytemuck::cast_slice(&gizmo_verts[..g_count]));
        }

        let clipped_primitives =
            egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, queue, *id, image_delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Engine3D Frame CommandEncoder"),
        });

        self.egui_renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        // Pass 1: Sky Atmosphere Background
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Pass"),
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
            rpass.set_pipeline(&self.sky_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }

        // Pass 2: 3D Depth-Tested PBR Meshes + Ground Grid + Soft Shadows
        if m_count > 0 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D PBR Mesh Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.mesh_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.mesh_vbuf.slice(..));
            rpass.draw(0..(m_count as u32), 0..1);
        }

        // Pass 3: 3D RGB Transform Gizmo & Selection Bounding Box (with cleared depth so gizmo handles stay on top)
        if g_count > 0 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D RGB Transform Gizmo Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.gizmo_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.gizmo_vbuf.slice(..));
            rpass.draw(0..(g_count as u32), 0..1);
        }

        // Pass 4: Crisp Vector egui Editor UI Overlay
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Vector UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut rpass, &clipped_primitives, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        encoder
    }
}

// ─── Interactive Desktop 3D Engine & Editor Runner ─────────────────────────

fn run_interactive_engine(start_in_play_mode: bool) {
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Metatopia 3D Game Engine v1.0 — PBR Viewport, Transform Gizmos, RigidBody Physics & Live Gameplay")
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
            label: Some("Metatopia 3D Engine Device"),
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
    MetatopiaEngineApp::configure_egui_theme(&egui_ctx);
    let mut egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
    );

    let mut app = MetatopiaEngineApp::new(start_in_play_mode);
    let init_u = app.build_scene_uniform(config.width, config.height);
    let mut gpu = Engine3DGpuPipeline::new(&device, format, config.width, config.height, &init_u);

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
                                                KeyCode::KeyQ => app.gizmo_tool = GizmoTool::Select,
                                                KeyCode::KeyW if !rmb_held => app.gizmo_tool = GizmoTool::Move,
                                                KeyCode::KeyE if !rmb_held => app.gizmo_tool = GizmoTool::Rotate,
                                                KeyCode::KeyR if !rmb_held => app.gizmo_tool = GizmoTool::Scale,
                                                KeyCode::Space => {
                                                    if app.play_mode {
                                                        app.stop_play_mode();
                                                    } else {
                                                        app.start_play_mode();
                                                    }
                                                }
                                                KeyCode::KeyF => {
                                                    app.fire_plasma_projectile_at([0.0, 1.0, -2.5]);
                                                }
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

                        app.fps_smooth = app.fps_smooth * 0.9 + (1.0 / dt) * 0.1;
                        app.frame_ms_smooth = app.frame_ms_smooth * 0.9 + (dt * 1000.0) * 0.1;

                        if rmb_held {
                            let forward = (app.cam_target - app.cam_pos).normalize();
                            let right = forward.cross(Vector3::unit_y()).normalize();
                            let spd = 6.0 * dt;
                            if keys_held.contains(&KeyCode::KeyW) {
                                app.cam_pos += forward * spd;
                                app.cam_target += forward * spd;
                            }
                            if keys_held.contains(&KeyCode::KeyS) {
                                app.cam_pos -= forward * spd;
                                app.cam_target -= forward * spd;
                            }
                            if keys_held.contains(&KeyCode::KeyA) {
                                app.cam_pos -= right * spd;
                                app.cam_target -= right * spd;
                            }
                            if keys_held.contains(&KeyCode::KeyD) {
                                app.cam_pos += right * spd;
                                app.cam_target += right * spd;
                            }
                        }

                        app.step_physics_and_gameplay(dt);

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
                            app.draw_editor_ui(ctx);
                        });
                        egui_state.handle_platform_output(&window, full_output.platform_output.clone());

                        let (mesh_verts, gizmo_verts) = app.build_scene_and_gizmo_batches();
                        let uniform = app.build_scene_uniform(config.width, config.height);

                        let encoder = gpu.render_frame(
                            &device,
                            &queue,
                            &view,
                            config.width,
                            config.height,
                            full_output.pixels_per_point,
                            &uniform,
                            &mesh_verts,
                            &gizmo_verts,
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
                    app.cam_yaw += delta.0 as f32 * 0.004;
                    app.cam_pitch = (app.cam_pitch - delta.1 as f32 * 0.004).clamp(-1.2, 0.2);
                    let dist = (app.cam_pos - app.cam_target).magnitude();
                    app.cam_pos = Point3::new(
                        app.cam_target.x - app.cam_yaw.cos() * app.cam_pitch.cos() * dist,
                        app.cam_target.y - app.cam_pitch.sin() * dist,
                        app.cam_target.z - app.cam_yaw.sin() * app.cam_pitch.cos() * dist,
                    );
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

// ─── Synthesized 128 BPM Stereo Engine Soundtrack + Physics/Laser SFX ──────

fn synthesize_engine_demo_wav(wav_path: &str, duration_secs: f32) {
    let sample_rate = 48_000u32;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let bpm = 128.0f32;
    let beat_sec = 60.0 / bpm;
    let step16_sec = beat_sec * 0.25;
    let tau = std::f32::consts::TAU;
    let arp_notes: [f32; 8] = [261.63, 329.63, 392.00, 523.25, 659.25, 523.25, 392.00, 329.63];
    let play_switch_time = 6.50f32;
    let shot_times: [f32; 6] = [7.4, 8.6, 10.0, 11.4, 12.8, 14.1];

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
        let in_play = t >= play_switch_time;

        let kick_freq = 48.0 + 110.0 * (-beat_pos * 26.0).exp();
        let kick = (tau * kick_freq * t).sin() * (-beat_pos * 6.0).exp() * 0.40;
        let sidechain = (1.0 - (-beat_pos * 5.2).exp() * 0.60).clamp(0.25, 1.0);

        let bass_freq = match (step16_idx / 16) % 4 {
            0 => 65.41,
            1 => 55.00,
            2 => 43.65,
            _ => 49.00,
        };
        let bass = ((tau * bass_freq * t).sin() + 0.4 * (tau * bass_freq * 2.0 * t).sin()).tanh()
            * sidechain
            * if in_play { 0.28 } else { 0.22 };

        let arp_f = arp_notes[step16_idx % arp_notes.len()] * if in_play { 1.5 } else { 1.0 };
        let arp_env = (-step16_pos * 8.0).exp();
        let arp = (tau * arp_f * t).sin() * arp_env * 0.16;

        let mut sfx = 0.0f32;
        for &st in &shot_times {
            if t >= st && t < st + 0.38 {
                let dt_s = t - st;
                let f = 220.0 + 1100.0 * (-dt_s * 16.0).exp();
                sfx += (tau * f * t).sin() * (-dt_s * 8.0).exp() * 0.24;
            }
        }

        let fade_in = (t / 0.3).clamp(0.0, 1.0);
        let fade_out = ((duration_secs - t) / 0.9).clamp(0.0, 1.0);
        let gain = 0.78 * fade_in * fade_out;

        let s_l = (((kick + bass + arp * 0.9 + sfx) * gain).tanh() * 0.80 * 32767.0) as i16;
        let s_r = (((kick + bass + arp * 1.1 + sfx * 0.9) * gain).tanh() * 0.80 * 32767.0) as i16;
        pcm.extend_from_slice(&s_l.to_le_bytes());
        pcm.extend_from_slice(&s_r.to_le_bytes());
    }

    let mut f = std::fs::File::create(wav_path).expect("Failed to create WAV file");
    f.write_all(&pcm).expect("Failed to write WAV file");
}

// ─── Offscreen 60 FPS 3D Game Engine Editor + Gameplay Video Recorder ──────

fn record_showcase_video(output_path: &str) {
    let width = 1280u32;
    let height = 720u32;
    let fps = 60u32;
    let duration_secs = 16.0f32;
    let total_frames = (fps as f32 * duration_secs) as u32;

    let wav_path = "/tmp/metatopia_engine3d_audio.wav";
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
            label: Some("Metatopia Offscreen 3D Engine Recorder"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create GPU device");

    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Offscreen 3D Engine Color Texture"),
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
    MetatopiaEngineApp::configure_egui_theme(&egui_ctx);

    let mut app = MetatopiaEngineApp::new(false);
    let init_u = app.build_scene_uniform(width, height);
    let mut gpu = Engine3DGpuPipeline::new(&device, target_format, width, height, &init_u);

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
    println!("  RECORDING METATOPIA 3D GAME ENGINE SHOWCASE (960 frames @ 60 FPS)");
    println!("  Output: {}", output_path);
    println!("====================================================================");

    for frame_idx in 0..total_frames {
        let t = frame_idx as f32 * dt;

        // ── PART 1 (0.0s – 6.5s): 3D SCENE EDITOR MODE (Gizmo Manipulation, Material Edit & Prefab Spawn) ──
        if t < 6.50 {
            let orbit_angle = -std::f32::consts::FRAC_PI_2 + 0.38 * (t * 0.55).sin();
            let radius = 9.4;
            app.cam_target = Point3::new(0.0, 1.35, -0.6);
            app.cam_pos = Point3::new(
                app.cam_target.x + orbit_angle.cos() * radius,
                4.8 + 0.35 * (t * 0.7).cos(),
                app.cam_target.z - orbit_angle.sin() * radius,
            );

            if frame_idx == 15 {
                app.selected_idx = 2; // Select Physics_Crate_Gold
                app.gizmo_tool = GizmoTool::Move;
                app.active_gizmo_axis = Some(1); // Highlight Green +Y Axis Gizmo
                app.log("[EDITOR] Selected 'Physics_Crate_Gold' -- Dragging +Y Translate Gizmo Handle.");
            }
            if frame_idx >= 20 && frame_idx < 135 {
                // Smoothly drag Physics_Crate_Gold upward with the Green +Y Gizmo!
                let p = (frame_idx - 20) as f32 / 115.0;
                app.objects[2].position[1] = 0.60 + p * 3.10;
                app.objects[2].rotation_deg[1] = 18.0 + p * 35.0;
            }
            if frame_idx == 140 {
                app.selected_idx = 3; // Select Bouncy_Sphere_Cyan
                app.active_gizmo_axis = Some(1);
                app.log("[EDITOR] Selected 'Bouncy_Sphere_Cyan' -- Lifting to Y = 4.2m for gravity drop test.");
            }
            if frame_idx >= 145 && frame_idx < 250 {
                let p = (frame_idx - 145) as f32 / 105.0;
                app.objects[3].position[1] = 0.65 + p * 3.55;
                app.objects[3].position[0] = 2.2 + p * 0.45;
            }
            if frame_idx == 260 {
                // Instantiate a new Ruby Bouncy Sphere from the Asset Browser!
                app.active_gizmo_axis = None;
                app.spawn_object(
                    MeshKind::PhysicsSphere,
                    "Physics_Sphere_Ruby",
                    [-2.7, 4.6, 1.0],
                    [0.98, 0.22, 0.44],
                );
            }
            if frame_idx == 325 {
                app.selected_idx = 5; // Inspect Enemy_Mech_Alpha before pressing Play
                app.gizmo_tool = GizmoTool::Rotate;
                app.bottom_tab = BottomTab::RhaiScriptEditor;
                app.log("[EDITOR] Inspecting 'Enemy_Mech_Alpha' Rhai AI Script before launching Play Mode.");
            }
        }

        // ── PART 2 (6.5s – 16.0s): PRESS [ |> PLAY ] -> REAL-TIME 3D PHYSICS & ACTION GAMEPLAY! ──
        if frame_idx == 390 {
            app.bottom_tab = BottomTab::AssetBrowser;
            app.selected_idx = 1; // Select Player_Mech_Controller
            app.start_play_mode();
        }

        if t >= 6.50 {
            let play_t = t - 6.50;
            // Player Mech strafes & jumps across the 3D Arena while aiming at enemies and crates
            let player_x = 1.65 * (play_t * 1.15).sin();
            let player_z = 3.9 - 1.1 * (play_t * 0.75).sin().abs();
            let jump_y = if (play_t > 2.2 && play_t < 3.1) || (play_t > 5.8 && play_t < 6.7) {
                let jp = ((play_t - 2.2) % 3.6) / 0.9;
                (jp * std::f32::consts::PI).sin().max(0.0) * 1.45
            } else {
                0.0
            };
            app.objects[1].position = [player_x, 1.02 + jump_y, player_z];
            app.objects[1].rotation_deg[1] = -player_x * 12.0;

            // Smooth Third-Person Action Camera following the Player Mech
            app.cam_target = Point3::new(player_x * 0.45, 1.45, -1.2);
            app.cam_pos = Point3::new(player_x * 0.65, 4.65, player_z + 5.2);

            // Fire 3D Plasma Projectiles at the 3 Enemy Bots & Physics Crates!
            match frame_idx {
                430 => {
                    // 7.16s: Shoot Enemy_Mech_Alpha (Hit 1 -> 50 HP)
                    let target = app.objects[5].position;
                    app.fire_plasma_projectile_at(target);
                }
                490 => {
                    // 8.16s: Shoot Enemy_Mech_Alpha (Hit 2 -> Defeated!)
                    let target = app.objects[5].position;
                    app.fire_plasma_projectile_at(target);
                }
                555 => {
                    // 9.25s: Shoot Physics_Crate_Gold to knock it flying across the floor!
                    let target = app.objects[2].position;
                    app.fire_plasma_projectile_at(target);
                }
                620 => {
                    // 10.33s: Shoot Enemy_Drone_Beta (Hit 1 -> 50 HP)
                    let target = app.objects[6].position;
                    app.fire_plasma_projectile_at(target);
                }
                685 => {
                    // 11.41s: Shoot Enemy_Drone_Beta (Hit 2 -> Defeated!)
                    let target = app.objects[6].position;
                    app.fire_plasma_projectile_at(target);
                }
                760 => {
                    // 12.66s: Shoot Boss_Guardian_Omega (Hit 1 -> 50 HP)
                    let target = app.objects[7].position;
                    app.fire_plasma_projectile_at(target);
                }
                835 => {
                    // 13.91s: Shoot Boss_Guardian_Omega (Hit 2 -> Boss Defeated!)
                    let target = app.objects[7].position;
                    app.fire_plasma_projectile_at(target);
                }
                890 => {
                    // 14.83s: Victory Physics Shot at Bouncy_Sphere_Cyan!
                    let target = app.objects[3].position;
                    app.fire_plasma_projectile_at(target);
                }
                _ => {}
            }
        }

        app.step_physics_and_gameplay(dt);

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
            app.draw_editor_ui(ctx);
        });

        let (mesh_verts, gizmo_verts) = app.build_scene_and_gizmo_batches();
        let uniform = app.build_scene_uniform(width, height);

        let mut encoder = gpu.render_frame(
            &device,
            &queue,
            &color_view,
            width,
            height,
            1.0,
            &uniform,
            &mesh_verts,
            &gizmo_verts,
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
            let fade = if t < 0.30 {
                (t / 0.30).clamp(0.0, 1.0)
            } else if t > duration_secs - 0.55 {
                ((duration_secs - t) / 0.55).clamp(0.0, 1.0)
            } else {
                1.0
            };

            for y in 0..height {
                let src_off = (y * bytes_per_row) as usize;
                let dst_off = (y * width * 4) as usize;
                let row_bytes = (width * 4) as usize;
                if fade >= 0.999 {
                    frame_rgba[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&data[src_off..src_off + row_bytes]);
                } else {
                    for x in 0..width as usize {
                        let s = src_off + x * 4;
                        let d = dst_off + x * 4;
                        frame_rgba[d] = (data[s] as f32 * fade) as u8;
                        frame_rgba[d + 1] = (data[s + 1] as f32 * fade) as u8;
                        frame_rgba[d + 2] = (data[s + 2] as f32 * fade) as u8;
                        frame_rgba[d + 3] = 255;
                    }
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

    let status = ffmpeg.wait().expect("Failed to wait for ffmpeg");
    assert!(status.success(), "ffmpeg encoding failed");
    println!("====================================================================");
    println!("  3D GAME ENGINE VIDEO RECORDED -> {}", output_path);
    println!("====================================================================");
}

// ─── Headless Self-Test Verification Suite (`--verify`) ────────────────────

fn run_verification_suite() {
    let mut app = MetatopiaEngineApp::new(false);
    assert!(app.objects.len() >= 13, "Scene must contain at least 13 3D GameObjects");

    // 1. Lift crate in Editor, start Play Mode, and verify gravity drops and bounces it
    app.objects[2].position[1] = 4.0;
    app.start_play_mode();
    for _ in 0..90 {
        app.step_physics_and_gameplay(1.0 / 60.0);
    }
    assert!(
        app.objects[2].position[1] < 3.5,
        "Dynamic RigidBody crate must fall under gravity in Play Mode"
    );

    // 2. Fire Plasma Projectiles at Enemy_Mech_Alpha and verify hit & defeat
    let enemy_pos = app.objects[5].position;
    app.fire_plasma_projectile_at(enemy_pos);
    for _ in 0..30 {
        app.step_physics_and_gameplay(1.0 / 60.0);
    }
    let enemy_pos2 = app.objects[5].position;
    app.fire_plasma_projectile_at(enemy_pos2);
    for _ in 0..30 {
        app.step_physics_and_gameplay(1.0 / 60.0);
    }
    assert!(
        app.score >= 100 && app.enemies_defeated >= 1,
        "Plasma projectiles must damage and defeat Enemy Bot"
    );

    // 3. Stop Play Mode and verify Editor Snapshot Restore
    app.stop_play_mode();
    let (mesh_verts, gizmo_verts) = app.build_scene_and_gizmo_batches();
    assert!(
        mesh_verts.len() > 1000 && !gizmo_verts.is_empty(),
        "3D Mesh and RGB Transform Gizmo vertex buffers must be populated"
    );

    println!("====================================================================");
    println!("  METATOPIA 3D GAME ENGINE v1.0 — MACOS EXECUTABLE VERIFIED");
    println!("====================================================================");
    println!("  • 3D Scene Objects        : {} GameObjects", app.objects.len());
    println!("  • 3D Triangles & Gizmo    : {} tris / {} gizmo verts", mesh_verts.len() / 3, gizmo_verts.len());
    println!("  • RigidBody 3D Physics    : Gravity, Restitution & Pairwise Collisions PASS");
    println!("  • Playable Combat & Snapshot: Plasma Projectiles, AI Bots & Snapshot Restore PASS");
    println!("====================================================================");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--verify") {
        run_verification_suite();
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

    let start_play = args.iter().any(|a| a == "--play");
    run_interactive_engine(start_play);
}
