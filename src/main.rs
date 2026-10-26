use std::f32::consts::{PI, TAU};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use cgmath::{InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3, perspective};
use egui::{Color32, Context, FontId, Pos2, Rect, RichText, Rounding, Stroke, Vec2};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use metatopia_engine::scripting::{ScriptEngine, ScriptEntityState};

const MAX_VERTS: usize = 262_144;
const SHADOW_MAP_SIZE: u32 = 2048;

// Material IDs matching shaders/engine_pbr_3d.wgsl
const MAT_UNLIT: f32 = -1.0;
const MAT_STANDARD: f32 = 0.0;
const MAT_HEX_FLOOR: f32 = 1.0;
const MAT_BRUSHED: f32 = 2.0;
const MAT_CIRCUIT: f32 = 3.0;
const MAT_WATER: f32 = 4.0;

// ============================================================================
// 1. GPU VERTEX & UNIFORM TYPES
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex3D {
    pos: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    mat: [f32; 4], // [metallic, roughness, emissive, material_id]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],    // xyz = eye, w = time
    sun_dir: [f32; 4],       // xyz = dir toward sun, w = intensity
    viewport: [f32; 4],      // x = width, y = height, z = play_mode, w = grid_alpha
    shadow_params: [f32; 4], // x = 1/2048, y = normal_bias, z = shadow_strength, w = debug_view
    hud_params: [f32; 4],    // x = hp_ratio, y = heat_ratio, z = hit_marker, w = wave_progress
    point_lights: [[f32; 4]; 8], // 4 point lights: even = pos+radius, odd = col+intensity
}

fn mat4_to_array(m: Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m.x.x, m.x.y, m.x.z, m.x.w],
        [m.y.x, m.y.y, m.y.z, m.y.w],
        [m.z.x, m.z.y, m.z.z, m.z.w],
        [m.w.x, m.w.y, m.w.z, m.w.w],
    ]
}

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

fn build_light_view_proj(sun_dir: Vector3<f32>) -> Matrix4<f32> {
    let dir = sun_dir.normalize();
    let half_extent = 25.0_f32;
    let near = 1.0_f32;
    let far = 86.0_f32;
    let eye = Point3::new(dir.x * 40.0, dir.y * 40.0, dir.z * 40.0);
    let up = if dir.y.abs() > 0.95 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let light_view = Matrix4::look_at_rh(eye, Point3::new(0.0, 0.0, 0.0), up);

    let sx = 1.0 / half_extent;
    let sy = 1.0 / half_extent;
    let sz = -2.0 / (far - near);
    let tz = -(far + near) / (far - near);
    let ortho_gl = Matrix4::new(
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        0.0, 0.0, tz, 1.0,
    );
    let mut lvp = OPENGL_TO_WGPU_MATRIX * ortho_gl * light_view;

    // Snap light-space translation to shadow-map texel grid to eliminate edge swimming
    let half_res = (SHADOW_MAP_SIZE as f32) * 0.5;
    lvp.w.x = (lvp.w.x * half_res).round() / half_res;
    lvp.w.y = (lvp.w.y * half_res).round() / half_res;
    lvp
}

// ============================================================================
// 2. 3D SCENE GRAPH, MESH KINDS & PHYSICS ENTITIES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshKind {
    ArenaFloor,
    WaterPool,
    SuspensionBridge,
    SciFiArchway,
    TurretTower,
    CyberPillar,
    EnergyCore,
    PhysicsCrate,
    HoverDrone,
    PlasmaOrb,
    PlayerMech,
    StepPlatform,
    JumpPad,
}

impl MeshKind {
    fn label(&self) -> &'static str {
        match self {
            Self::ArenaFloor => "Hex Cyber-Grid Floor",
            Self::WaterPool => "Reflective Water Basin",
            Self::SuspensionBridge => "Cyber Suspension Bridge",
            Self::SciFiArchway => "Monumental Portal Arch",
            Self::TurretTower => "Autonomous Defense Turret",
            Self::CyberPillar => "Brushed Titanium Pillar",
            Self::EnergyCore => "Neon Circuit Reactor",
            Self::PhysicsCrate => "Rigid-Body Cargo Crate",
            Self::HoverDrone => "Autonomous Hover-Drone",
            Self::PlasmaOrb => "Plasma Energy Pickup",
            Self::PlayerMech => "Vanguard Striker Mech",
            Self::StepPlatform => "Elevated Tactical Deck",
            Self::JumpPad => "Graviton Jump Pad",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GizmoAxis {
    None,
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BottomTab {
    AssetBrowser,
    ConsoleLogs,
    RhaiScriptEditor,
}

#[derive(Clone, Debug)]
struct SceneObject3D {
    id: u32,
    name: String,
    kind: MeshKind,
    position: Vector3<f32>,
    rotation_deg: Vector3<f32>,
    scale: Vector3<f32>,
    velocity: Vector3<f32>,
    color: [f32; 3],
    metallic: f32,
    roughness: f32,
    emissive: f32,
    has_gravity: bool,
    is_static: bool,
    alive: bool,
    hp: f32,
    max_hp: f32,
    ai_phase: f32,
    fire_cooldown: f32,
}

impl SceneObject3D {
    fn world_aabb(&self) -> (Vector3<f32>, Vector3<f32>) {
        let half = match self.kind {
            MeshKind::ArenaFloor => Vector3::new(32.0 * self.scale.x, 0.25 * self.scale.y, 32.0 * self.scale.z),
            MeshKind::WaterPool => Vector3::new(5.5 * self.scale.x, 0.08 * self.scale.y, 5.5 * self.scale.z),
            MeshKind::SuspensionBridge => Vector3::new(5.8 * self.scale.x, 0.55 * self.scale.y, 1.65 * self.scale.z),
            MeshKind::SciFiArchway => Vector3::new(3.2 * self.scale.x, 3.4 * self.scale.y, 0.9 * self.scale.z),
            MeshKind::TurretTower => Vector3::new(0.95 * self.scale.x, 1.55 * self.scale.y, 0.95 * self.scale.z),
            MeshKind::CyberPillar => Vector3::new(0.65 * self.scale.x, 1.9 * self.scale.y, 0.65 * self.scale.z),
            MeshKind::StepPlatform => Vector3::new(1.8 * self.scale.x, 0.35 * self.scale.y, 1.8 * self.scale.z),
            MeshKind::PlayerMech => Vector3::new(0.65 * self.scale.x, 1.05 * self.scale.y, 0.65 * self.scale.z),
            MeshKind::JumpPad => Vector3::new(0.95 * self.scale.x, 0.22 * self.scale.y, 0.95 * self.scale.z),
            MeshKind::HoverDrone => Vector3::new(0.92 * self.scale.x, 0.72 * self.scale.y, 0.92 * self.scale.z),
            _ => Vector3::new(0.75 * self.scale.x, 0.75 * self.scale.y, 0.75 * self.scale.z),
        };
        (self.position - half, self.position + half)
    }
}

#[derive(Clone, Debug)]
struct Projectile3D {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    color: [f32; 3],
    ttl: f32,
    from_enemy: bool,
}

#[derive(Clone, Debug)]
struct Particle3D {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    color: [f32; 3],
    size: f32,
    life: f32,
    max_life: f32,
}

// ============================================================================
// 3. PROCEDURAL 3D MESH & MULTI-PART PREFAB BUILDER
// ============================================================================

fn rotate_xyz(v: Vector3<f32>, rot_deg: Vector3<f32>) -> Vector3<f32> {
    let rx = rot_deg.x.to_radians();
    let ry = rot_deg.y.to_radians();
    let rz = rot_deg.z.to_radians();

    let (sz, cz) = rz.sin_cos();
    let v1 = Vector3::new(v.x * cz - v.y * sz, v.x * sz + v.y * cz, v.z);
    let (sx, cx) = rx.sin_cos();
    let v2 = Vector3::new(v1.x, v1.y * cx - v1.z * sx, v1.y * sx + v1.z * cx);
    let (sy, cy) = ry.sin_cos();
    Vector3::new(v2.x * cy + v2.z * sy, v2.y, -v2.x * sy + v2.z * cy)
}

fn rotate_y(v: Vector3<f32>, yaw_rad: f32) -> Vector3<f32> {
    let (s, c) = yaw_rad.sin_cos();
    Vector3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
}

fn push_quad(
    out: &mut Vec<Vertex3D>,
    p0: Vector3<f32>,
    p1: Vector3<f32>,
    p2: Vector3<f32>,
    p3: Vector3<f32>,
    normal: Vector3<f32>,
    color: [f32; 3],
    mat: [f32; 4],
) {
    if out.len() + 6 > MAX_VERTS {
        return;
    }
    let n = [normal.x, normal.y, normal.z];
    let v0 = Vertex3D { pos: [p0.x, p0.y, p0.z], normal: n, color, mat };
    let v1 = Vertex3D { pos: [p1.x, p1.y, p1.z], normal: n, color, mat };
    let v2 = Vertex3D { pos: [p2.x, p2.y, p2.z], normal: n, color, mat };
    let v3 = Vertex3D { pos: [p3.x, p3.y, p3.z], normal: n, color, mat };
    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

fn append_box_ex(
    out: &mut Vec<Vertex3D>,
    center: Vector3<f32>,
    half: Vector3<f32>,
    rot_deg: Vector3<f32>,
    color: [f32; 3],
    mat: [f32; 4],
) {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]),
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0]]),
        ([1.0, 0.0, 0.0], [[1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]),
        ([-1.0, 0.0, 0.0], [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]),
        ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        ([0.0, 0.0, -1.0], [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]),
    ];

    for (n_raw, corners) in faces {
        let wn = rotate_xyz(Vector3::new(n_raw[0], n_raw[1], n_raw[2]), rot_deg).normalize();
        let pts: Vec<Vector3<f32>> = corners
            .iter()
            .map(|c| {
                let local = Vector3::new(c[0] * half.x, c[1] * half.y, c[2] * half.z);
                center + rotate_xyz(local, rot_deg)
            })
            .collect();
        push_quad(out, pts[0], pts[1], pts[2], pts[3], wn, color, mat);
    }
}

fn append_box(
    out: &mut Vec<Vertex3D>,
    center: Vector3<f32>,
    half: Vector3<f32>,
    yaw_rad: f32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    append_box_ex(out, center, half, Vector3::new(0.0, yaw_rad.to_degrees(), 0.0), color, mat);
}

fn append_sphere(
    out: &mut Vec<Vertex3D>,
    center: Vector3<f32>,
    radii: Vector3<f32>,
    stacks: u32,
    slices: u32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    for i in 0..stacks {
        let v0 = i as f32 / stacks as f32;
        let v1 = (i + 1) as f32 / stacks as f32;
        let phi0 = v0 * PI - PI * 0.5;
        let phi1 = v1 * PI - PI * 0.5;

        for j in 0..slices {
            if out.len() + 6 > MAX_VERTS {
                return;
            }
            let u0 = j as f32 / slices as f32;
            let u1 = (j + 1) as f32 / slices as f32;
            let th0 = u0 * TAU;
            let th1 = u1 * TAU;

            let sample = |phi: f32, th: f32| -> Vertex3D {
                let (sp, cp) = phi.sin_cos();
                let (st, ct) = th.sin_cos();
                let unit = Vector3::new(cp * ct, sp, cp * st);
                let pos = center + Vector3::new(unit.x * radii.x, unit.y * radii.y, unit.z * radii.z);
                let n = Vector3::new(
                    unit.x / radii.x.max(0.01),
                    unit.y / radii.y.max(0.01),
                    unit.z / radii.z.max(0.01),
                )
                .normalize();
                Vertex3D {
                    pos: [pos.x, pos.y, pos.z],
                    normal: [n.x, n.y, n.z],
                    color,
                    mat,
                }
            };

            let p00 = sample(phi0, th0);
            let p10 = sample(phi0, th1);
            let p11 = sample(phi1, th1);
            let p01 = sample(phi1, th0);
            out.extend_from_slice(&[p00, p11, p10, p00, p01, p11]);
        }
    }
}

fn append_cylinder(
    out: &mut Vec<Vertex3D>,
    base: Vector3<f32>,
    axis: Vector3<f32>,
    radius: f32,
    length: f32,
    slices: u32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    let dir = axis.normalize();
    let up_ref = if dir.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let right = dir.cross(up_ref).normalize();
    let forward = right.cross(dir).normalize();
    let top = base + dir * length;

    for i in 0..slices {
        if out.len() + 12 > MAX_VERTS {
            return;
        }
        let a0 = (i as f32 / slices as f32) * TAU;
        let a1 = ((i + 1) as f32 / slices as f32) * TAU;
        let n0 = (right * a0.cos() + forward * a0.sin()).normalize();
        let n1 = (right * a1.cos() + forward * a1.sin()).normalize();

        let b0 = base + n0 * radius;
        let b1 = base + n1 * radius;
        let t1 = top + n1 * radius;
        let t0 = top + n0 * radius;

        let v_b0 = Vertex3D { pos: [b0.x, b0.y, b0.z], normal: [n0.x, n0.y, n0.z], color, mat };
        let v_b1 = Vertex3D { pos: [b1.x, b1.y, b1.z], normal: [n1.x, n1.y, n1.z], color, mat };
        let v_t1 = Vertex3D { pos: [t1.x, t1.y, t1.z], normal: [n1.x, n1.y, n1.z], color, mat };
        let v_t0 = Vertex3D { pos: [t0.x, t0.y, t0.z], normal: [n0.x, n0.y, n0.z], color, mat };
        out.extend_from_slice(&[v_b0, v_b1, v_t1, v_b0, v_t1, v_t0]);

        let top_n = [dir.x, dir.y, dir.z];
        out.extend_from_slice(&[
            Vertex3D { pos: [top.x, top.y, top.z], normal: top_n, color, mat },
            Vertex3D { pos: [t0.x, t0.y, t0.z], normal: top_n, color, mat },
            Vertex3D { pos: [t1.x, t1.y, t1.z], normal: top_n, color, mat },
        ]);
    }
}

fn append_cone(
    out: &mut Vec<Vertex3D>,
    base: Vector3<f32>,
    axis: Vector3<f32>,
    radius: f32,
    length: f32,
    slices: u32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    let dir = axis.normalize();
    let up_ref = if dir.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let right = dir.cross(up_ref).normalize();
    let forward = right.cross(dir).normalize();
    let tip = base + dir * length;

    for i in 0..slices {
        if out.len() + 6 > MAX_VERTS {
            return;
        }
        let a0 = (i as f32 / slices as f32) * TAU;
        let a1 = ((i + 1) as f32 / slices as f32) * TAU;
        let n0 = (right * a0.cos() + forward * a0.sin() + dir * 0.35).normalize();
        let n1 = (right * a1.cos() + forward * a1.sin() + dir * 0.35).normalize();
        let b0 = base + (right * a0.cos() + forward * a0.sin()) * radius;
        let b1 = base + (right * a1.cos() + forward * a1.sin()) * radius;

        out.extend_from_slice(&[
            Vertex3D { pos: [b0.x, b0.y, b0.z], normal: [n0.x, n0.y, n0.z], color, mat },
            Vertex3D { pos: [b1.x, b1.y, b1.z], normal: [n1.x, n1.y, n1.z], color, mat },
            Vertex3D { pos: [tip.x, tip.y, tip.z], normal: [dir.x, dir.y, dir.z], color, mat },
        ]);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_torus_arc(
    out: &mut Vec<Vertex3D>,
    center: Vector3<f32>,
    rot_deg: Vector3<f32>,
    major_r: f32,
    minor_r: f32,
    arc_rad: f32,
    major_seg: u32,
    minor_seg: u32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    for i in 0..major_seg {
        let u0 = (i as f32 / major_seg as f32) * arc_rad;
        let u1 = ((i + 1) as f32 / major_seg as f32) * arc_rad;
        for j in 0..minor_seg {
            if out.len() + 6 > MAX_VERTS {
                return;
            }
            let v0 = (j as f32 / minor_seg as f32) * TAU;
            let v1 = ((j + 1) as f32 / minor_seg as f32) * TAU;

            let sample = |u: f32, v: f32| -> Vertex3D {
                let (su, cu) = u.sin_cos();
                let (sv, cv) = v.sin_cos();
                let r = major_r + minor_r * cv;
                let local_pos = Vector3::new(r * cu, minor_r * sv, r * su);
                let local_n = Vector3::new(cv * cu, sv, cv * su).normalize();
                let wp = center + rotate_xyz(local_pos, rot_deg);
                let wn = rotate_xyz(local_n, rot_deg).normalize();
                Vertex3D {
                    pos: [wp.x, wp.y, wp.z],
                    normal: [wn.x, wn.y, wn.z],
                    color,
                    mat,
                }
            };

            let p00 = sample(u0, v0);
            let p10 = sample(u1, v0);
            let p11 = sample(u1, v1);
            let p01 = sample(u0, v1);
            out.extend_from_slice(&[p00, p10, p11, p00, p11, p01]);
        }
    }
}

fn append_capsule(
    out: &mut Vec<Vertex3D>,
    center: Vector3<f32>,
    rot_deg: Vector3<f32>,
    radius: f32,
    half_height: f32,
    color: [f32; 3],
    mat: [f32; 4],
) {
    let axis = rotate_xyz(Vector3::new(0.0, 1.0, 0.0), rot_deg).normalize();
    let base = center - axis * half_height;
    let top = center + axis * half_height;
    append_cylinder(out, base, axis, radius, half_height * 2.0, 16, color, mat);
    append_sphere(out, base, Vector3::new(radius, radius, radius), 8, 14, color, mat);
    append_sphere(out, top, Vector3::new(radius, radius, radius), 8, 14, color, mat);
}

fn append_scene_object(out: &mut Vec<Vertex3D>, obj: &SceneObject3D, selected: bool, time: f32) {
    if !obj.alive {
        return;
    }
    let rot = obj.rotation_deg;
    let yaw = obj.rotation_deg.y.to_radians();
    let base_mat = [obj.metallic, obj.roughness, obj.emissive, MAT_STANDARD];
    let brushed_mat = [obj.metallic.max(0.78), obj.roughness.min(0.26), obj.emissive, MAT_BRUSHED];
    let circuit_mat = [0.25, 0.18, obj.emissive.max(1.2), MAT_CIRCUIT];

    match obj.kind {
        MeshKind::ArenaFloor => {
            let half = Vector3::new(32.0 * obj.scale.x, 0.25 * obj.scale.y, 32.0 * obj.scale.z);
            let floor_mat = [obj.metallic, obj.roughness, obj.emissive, MAT_HEX_FLOOR];
            append_box(out, obj.position, half, 0.0, obj.color, floor_mat);

            let rail_mat = [0.65, 0.20, 0.85, MAT_CIRCUIT];
            let rail_col = [0.10, 0.72, 1.0];
            append_box(out, Vector3::new(0.0, 0.18, -31.6), Vector3::new(31.6, 0.18, 0.22), 0.0, rail_col, rail_mat);
            append_box(out, Vector3::new(0.0, 0.18, 31.6), Vector3::new(31.6, 0.18, 0.22), 0.0, rail_col, rail_mat);
            append_box(out, Vector3::new(-31.6, 0.18, 0.0), Vector3::new(0.22, 0.18, 31.6), 0.0, rail_col, rail_mat);
            append_box(out, Vector3::new(31.6, 0.18, 0.0), Vector3::new(0.22, 0.18, 31.6), 0.0, rail_col, rail_mat);
        }
        MeshKind::WaterPool => {
            let w_half = Vector3::new(5.4 * obj.scale.x, 0.05 * obj.scale.y, 5.4 * obj.scale.z);
            let water_mat = [0.15, 0.05, 0.08, MAT_WATER];
            append_box(out, obj.position, w_half, yaw, obj.color, water_mat);

            let curb_col = [0.22, 0.28, 0.36];
            let curb_mat = [0.85, 0.22, 0.05, MAT_BRUSHED];
            let sx = 5.55 * obj.scale.x;
            let sz = 5.55 * obj.scale.z;
            append_box(out, obj.position + Vector3::new(0.0, 0.09, -sz), Vector3::new(sx, 0.11, 0.18), 0.0, curb_col, curb_mat);
            append_box(out, obj.position + Vector3::new(0.0, 0.09, sz), Vector3::new(sx, 0.11, 0.18), 0.0, curb_col, curb_mat);
            append_box(out, obj.position + Vector3::new(-sx, 0.09, 0.0), Vector3::new(0.18, 0.11, sz), 0.0, curb_col, curb_mat);
            append_box(out, obj.position + Vector3::new(sx, 0.09, 0.0), Vector3::new(0.18, 0.11, sz), 0.0, curb_col, curb_mat);
        }
        MeshKind::SuspensionBridge => {
            let p = obj.position;
            let s = obj.scale;
            let deck_half = Vector3::new(5.6 * s.x, 0.16 * s.y, 1.55 * s.z);
            append_box_ex(out, p, deck_half, rot, obj.color, brushed_mat);

            let strip_col = [0.12, 0.88, 1.0];
            for z_side in [-1.32_f32, 1.32_f32] {
                let offset = rotate_xyz(Vector3::new(0.0, 0.18 * s.y, z_side * s.z), rot);
                append_box_ex(
                    out,
                    p + offset,
                    Vector3::new(5.5 * s.x, 0.04 * s.y, 0.08 * s.z),
                    rot,
                    strip_col,
                    circuit_mat,
                );
            }

            let pylon_col = [0.28, 0.34, 0.44];
            for x_side in [-3.8_f32, 3.8_f32] {
                for z_side in [-1.45_f32, 1.45_f32] {
                    let local = Vector3::new(x_side * s.x, 0.45 * s.y, z_side * s.z);
                    append_box_ex(
                        out,
                        p + rotate_xyz(local, rot),
                        Vector3::new(0.22 * s.x, 1.65 * s.y, 0.22 * s.z),
                        rot,
                        pylon_col,
                        brushed_mat,
                    );
                }
                let beam_local = Vector3::new(x_side * s.x, 2.02 * s.y, 0.0);
                append_box_ex(
                    out,
                    p + rotate_xyz(beam_local, rot),
                    Vector3::new(0.20 * s.x, 0.14 * s.y, 1.62 * s.z),
                    rot,
                    pylon_col,
                    brushed_mat,
                );
            }

            for (x_end, tilt) in [(-6.35_f32, 16.0_f32), (6.35_f32, -16.0_f32)] {
                let ramp_local = Vector3::new(x_end * s.x, -0.34 * s.y, 0.0);
                let ramp_rot = Vector3::new(rot.x, rot.y, rot.z + tilt);
                append_box_ex(
                    out,
                    p + rotate_xyz(ramp_local, rot),
                    Vector3::new(1.15 * s.x, 0.12 * s.y, 1.45 * s.z),
                    ramp_rot,
                    obj.color,
                    brushed_mat,
                );
            }
        }
        MeshKind::SciFiArchway => {
            let p = obj.position;
            let s = obj.scale;
            for x_side in [-2.65_f32, 2.65_f32] {
                let col_local = Vector3::new(x_side * s.x, 1.85 * s.y, 0.0);
                append_box_ex(
                    out,
                    p + rotate_xyz(col_local, rot),
                    Vector3::new(0.42 * s.x, 1.85 * s.y, 0.52 * s.z),
                    rot,
                    obj.color,
                    brushed_mat,
                );
                let trim_local = Vector3::new((x_side - x_side.signum() * 0.44) * s.x, 1.85 * s.y, 0.0);
                append_box_ex(
                    out,
                    p + rotate_xyz(trim_local, rot),
                    Vector3::new(0.06 * s.x, 1.75 * s.y, 0.25 * s.z),
                    rot,
                    [0.12, 0.88, 1.0],
                    circuit_mat,
                );
            }
            let arch_center = p + rotate_xyz(Vector3::new(0.0, 3.65 * s.y, 0.0), rot);
            let arch_rot = Vector3::new(rot.x + 90.0, rot.y, rot.z);
            append_torus_arc(
                out,
                arch_center,
                arch_rot,
                2.65 * s.x,
                0.36 * s.y,
                PI,
                24,
                14,
                obj.color,
                brushed_mat,
            );
            let keystone_pos = p + rotate_xyz(Vector3::new(0.0, 6.32 * s.y, 0.0), rot);
            append_box_ex(
                out,
                keystone_pos,
                Vector3::new(0.42 * s.x, 0.52 * s.y, 0.58 * s.z),
                rot,
                [0.15, 0.92, 1.0],
                [0.2, 0.12, 1.65, MAT_CIRCUIT],
            );
        }
        MeshKind::TurretTower => {
            let p = obj.position;
            let s = obj.scale;
            append_cylinder(
                out,
                p - Vector3::new(0.0, 1.45 * s.y, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                0.78 * s.x,
                1.85 * s.y,
                6,
                obj.color,
                brushed_mat,
            );
            append_torus_arc(
                out,
                p + Vector3::new(0.0, 0.35 * s.y, 0.0),
                Vector3::new(0.0, time * 55.0, 0.0),
                0.68 * s.x,
                0.09 * s.y,
                TAU,
                20,
                10,
                [1.0, 0.28, 0.14],
                [0.3, 0.15, 1.55, MAT_CIRCUIT],
            );
            let head_pos = p + Vector3::new(0.0, 0.82 * s.y, 0.0);
            append_box_ex(
                out,
                head_pos,
                Vector3::new(0.56 * s.x, 0.38 * s.y, 0.68 * s.z),
                rot,
                [0.24, 0.28, 0.36],
                brushed_mat,
            );
            let fwd = rotate_y(Vector3::new(0.0, 0.0, 1.0), yaw);
            let right = rotate_y(Vector3::new(1.0, 0.0, 0.0), yaw);
            for side in [-0.28_f32, 0.28_f32] {
                let barrel_base = head_pos + right * (side * s.x) + fwd * (0.45 * s.z);
                append_cylinder(
                    out,
                    barrel_base,
                    fwd,
                    0.09 * s.x,
                    0.95 * s.z,
                    12,
                    [0.18, 0.22, 0.28],
                    brushed_mat,
                );
                let muzzle = barrel_base + fwd * (0.96 * s.z);
                append_sphere(
                    out,
                    muzzle,
                    Vector3::new(0.11 * s.x, 0.11 * s.y, 0.11 * s.z),
                    8,
                    10,
                    [1.0, 0.22, 0.12],
                    [0.1, 0.1, 2.2, MAT_CIRCUIT],
                );
            }
        }
        MeshKind::CyberPillar => {
            let p = obj.position;
            let s = obj.scale;
            append_cylinder(
                out,
                p - Vector3::new(0.0, 1.8 * s.y, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                0.56 * s.x,
                3.6 * s.y,
                8,
                obj.color,
                brushed_mat,
            );
            for ring_y in [-0.95_f32, 0.0_f32, 0.95_f32] {
                append_torus_arc(
                    out,
                    p + Vector3::new(0.0, ring_y * s.y, 0.0),
                    Vector3::new(0.0, 0.0, 0.0),
                    0.58 * s.x,
                    0.065 * s.y,
                    TAU,
                    18,
                    10,
                    [0.12, 0.86, 1.0],
                    circuit_mat,
                );
            }
        }
        MeshKind::StepPlatform => {
            let half = Vector3::new(1.8 * obj.scale.x, 0.35 * obj.scale.y, 1.8 * obj.scale.z);
            append_box_ex(out, obj.position, half, rot, obj.color, brushed_mat);
            let trim_half = Vector3::new(1.83 * obj.scale.x, 0.05 * obj.scale.y, 1.83 * obj.scale.z);
            append_box_ex(
                out,
                obj.position + Vector3::new(0.0, half.y * 0.72, 0.0),
                trim_half,
                rot,
                [0.15, 0.85, 1.0],
                circuit_mat,
            );
        }
        MeshKind::JumpPad => {
            let base = obj.position - Vector3::new(0.0, 0.18 * obj.scale.y, 0.0);
            append_cylinder(
                out,
                base,
                Vector3::new(0.0, 1.0, 0.0),
                0.95 * obj.scale.x,
                0.24 * obj.scale.y,
                20,
                [0.18, 0.22, 0.30],
                brushed_mat,
            );
            append_torus_arc(
                out,
                obj.position + Vector3::new(0.0, 0.08 * obj.scale.y, 0.0),
                Vector3::new(0.0, time * 90.0, 0.0),
                0.72 * obj.scale.x,
                0.09 * obj.scale.y,
                TAU,
                22,
                12,
                obj.color,
                [0.2, 0.1, 2.2, MAT_CIRCUIT],
            );
        }
        MeshKind::EnergyCore => {
            let p = obj.position;
            let s = obj.scale;
            append_sphere(
                out,
                p,
                Vector3::new(0.62 * s.x, 0.62 * s.y, 0.62 * s.z),
                16,
                24,
                obj.color,
                [0.2, 0.12, obj.emissive.max(1.5), MAT_CIRCUIT],
            );
            append_torus_arc(
                out,
                p,
                Vector3::new(35.0, time * 65.0, 20.0),
                0.88 * s.x,
                0.065 * s.y,
                TAU,
                24,
                12,
                [0.22, 0.94, 1.0],
                [0.85, 0.15, 1.6, MAT_CIRCUIT],
            );
            append_torus_arc(
                out,
                p,
                Vector3::new(-45.0, -time * 50.0, 60.0),
                1.04 * s.x,
                0.055 * s.y,
                TAU,
                24,
                12,
                [1.0, 0.62, 0.18],
                [0.85, 0.15, 1.3, MAT_BRUSHED],
            );
        }
        MeshKind::PhysicsCrate => {
            let half = Vector3::new(0.65 * obj.scale.x, 0.65 * obj.scale.y, 0.65 * obj.scale.z);
            append_box_ex(out, obj.position, half, rot, obj.color, base_mat);
            let frame_col = [0.22, 0.28, 0.36];
            append_box_ex(
                out,
                obj.position,
                Vector3::new(half.x * 1.03, half.y * 0.24, half.z * 1.03),
                rot,
                frame_col,
                brushed_mat,
            );
            append_box_ex(
                out,
                obj.position,
                Vector3::new(half.x * 0.24, half.y * 1.03, half.z * 1.03),
                rot,
                [0.14, 0.86, 1.0],
                circuit_mat,
            );
        }
        MeshKind::HoverDrone => {
            let p = obj.position;
            let s = obj.scale;
            append_capsule(
                out,
                p,
                Vector3::new(rot.x, rot.y, rot.z + 90.0),
                0.38 * s.y,
                0.24 * s.x,
                obj.color,
                brushed_mat,
            );
            append_torus_arc(
                out,
                p,
                Vector3::new(12.0 * (time * 3.0).sin(), time * 120.0, 8.0 * (time * 2.3).cos()),
                0.78 * s.x,
                0.065 * s.y,
                TAU,
                24,
                12,
                [0.12, 0.90, 1.0],
                [0.3, 0.12, 1.75, MAT_CIRCUIT],
            );
            for angle_deg in [45.0_f32, 135.0, 225.0, 315.0] {
                let a = angle_deg.to_radians() + yaw;
                let pod_pos = p + Vector3::new(a.cos() * 0.74 * s.x, 0.12 * s.y, a.sin() * 0.74 * s.z);
                append_cylinder(
                    out,
                    pod_pos - Vector3::new(0.0, 0.12 * s.y, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                    0.18 * s.x,
                    0.22 * s.y,
                    10,
                    [0.20, 0.24, 0.32],
                    brushed_mat,
                );
                append_sphere(
                    out,
                    pod_pos - Vector3::new(0.0, 0.16 * s.y, 0.0),
                    Vector3::new(0.11 * s.x, 0.06 * s.y, 0.11 * s.z),
                    6,
                    10,
                    [1.0, 0.36, 0.14],
                    [0.1, 0.1, 2.2, MAT_CIRCUIT],
                );
            }
            let eye_dir = rotate_y(Vector3::new(0.0, 0.0, 1.0), yaw);
            let eye_pos = p + eye_dir * (0.46 * s.z);
            append_sphere(
                out,
                eye_pos,
                Vector3::new(0.16 * s.x, 0.12 * s.y, 0.16 * s.z),
                10,
                14,
                [1.0, 0.14, 0.22],
                [0.1, 0.1, 2.8, MAT_CIRCUIT],
            );
        }
        MeshKind::PlasmaOrb => {
            let p = obj.position;
            let s = obj.scale;
            append_sphere(
                out,
                p,
                Vector3::new(0.38 * s.x, 0.38 * s.y, 0.38 * s.z),
                14,
                20,
                obj.color,
                [0.2, 0.08, obj.emissive.max(1.8), MAT_CIRCUIT],
            );
            append_torus_arc(
                out,
                p,
                Vector3::new(45.0, time * 110.0, 0.0),
                0.54 * s.x,
                0.045 * s.y,
                TAU,
                18,
                10,
                [1.0, 0.88, 0.28],
                [0.8, 0.1, 1.8, MAT_CIRCUIT],
            );
        }
        MeshKind::PlayerMech => {
            let s = obj.scale;
            let p = obj.position;
            let speed = Vector3::new(obj.velocity.x, 0.0, obj.velocity.z).magnitude();
            let stride = if speed > 0.25 {
                (time * 10.0).sin() * 0.20 * (speed / 9.0).min(1.0)
            } else {
                0.0
            };

            let dark_metal = [0.65, 0.30, 0.0, MAT_BRUSHED];
            let armor_mat = [obj.metallic.max(0.75), obj.roughness.min(0.24), obj.emissive, MAT_BRUSHED];
            let glow_cyan = [0.2, 0.1, 2.2, MAT_CIRCUIT];

            for (side, leg_stride) in [(-0.26_f32, stride), (0.26_f32, -stride)] {
                let foot_off = rotate_y(Vector3::new(side * s.x, -0.88 * s.y, leg_stride * s.z), yaw);
                append_box(
                    out,
                    p + foot_off,
                    Vector3::new(0.16 * s.x, 0.10 * s.y, 0.26 * s.z),
                    yaw,
                    [0.18, 0.22, 0.28],
                    dark_metal,
                );
                let shin_off = rotate_y(Vector3::new(side * s.x, -0.52 * s.y, leg_stride * 0.55 * s.z), yaw);
                append_box(
                    out,
                    p + shin_off,
                    Vector3::new(0.13 * s.x, 0.32 * s.y, 0.15 * s.z),
                    yaw,
                    [0.24, 0.30, 0.38],
                    dark_metal,
                );
                let knee_off = rotate_y(Vector3::new(side * s.x, -0.22 * s.y, leg_stride * 0.35 * s.z), yaw);
                append_sphere(
                    out,
                    p + knee_off,
                    Vector3::new(0.14 * s.x, 0.14 * s.y, 0.14 * s.z),
                    8,
                    12,
                    [0.12, 0.86, 1.0],
                    glow_cyan,
                );
            }

            let torso_pos = p + Vector3::new(0.0, 0.18 * s.y, 0.0);
            append_box(
                out,
                torso_pos,
                Vector3::new(0.46 * s.x, 0.42 * s.y, 0.32 * s.z),
                yaw,
                obj.color,
                armor_mat,
            );
            let core_pos = p + rotate_y(Vector3::new(0.0, 0.22 * s.y, 0.33 * s.z), yaw);
            append_torus_arc(
                out,
                core_pos,
                Vector3::new(90.0, obj.rotation_deg.y, 0.0),
                0.16 * s.x,
                0.045 * s.y,
                TAU,
                16,
                10,
                [0.15, 0.95, 1.0],
                glow_cyan,
            );

            for side in [-0.58_f32, 0.58_f32] {
                let sh_pos = p + rotate_y(Vector3::new(side * s.x, 0.42 * s.y, 0.0), yaw);
                append_box(
                    out,
                    sh_pos,
                    Vector3::new(0.18 * s.x, 0.20 * s.y, 0.26 * s.z),
                    yaw,
                    obj.color,
                    armor_mat,
                );
            }

            let head_pos = p + Vector3::new(0.0, 0.78 * s.y, 0.0);
            append_sphere(
                out,
                head_pos,
                Vector3::new(0.28 * s.x, 0.25 * s.y, 0.28 * s.z),
                12,
                16,
                [0.88, 0.92, 0.98],
                armor_mat,
            );
            let visor_pos = head_pos + rotate_y(Vector3::new(0.0, 0.02 * s.y, 0.22 * s.z), yaw);
            append_box(
                out,
                visor_pos,
                Vector3::new(0.22 * s.x, 0.07 * s.y, 0.10 * s.z),
                yaw,
                [0.10, 0.95, 1.0],
                glow_cyan,
            );

            let gun_dir = rotate_y(Vector3::new(0.0, 0.0, 1.0), yaw);
            let gun_base = p + rotate_y(Vector3::new(0.56 * s.x, 0.18 * s.y, 0.12 * s.z), yaw);
            append_cylinder(
                out,
                gun_base,
                gun_dir,
                0.095 * s.x,
                0.78 * s.z,
                12,
                [0.20, 0.25, 0.32],
                dark_metal,
            );
            let muzzle_pos = gun_base + gun_dir * (0.80 * s.z);
            append_sphere(
                out,
                muzzle_pos,
                Vector3::new(0.11 * s.x, 0.11 * s.y, 0.11 * s.z),
                8,
                12,
                [0.15, 0.96, 1.0],
                glow_cyan,
            );

            let pack_pos = p + rotate_y(Vector3::new(0.0, 0.22 * s.y, -0.36 * s.z), yaw);
            append_box(
                out,
                pack_pos,
                Vector3::new(0.30 * s.x, 0.28 * s.y, 0.14 * s.z),
                yaw,
                [0.22, 0.26, 0.34],
                dark_metal,
            );
            for jet_x in [-0.16_f32, 0.16_f32] {
                let jet_pos = p + rotate_y(Vector3::new(jet_x * s.x, -0.06 * s.y, -0.38 * s.z), yaw);
                append_sphere(
                    out,
                    jet_pos,
                    Vector3::new(0.08 * s.x, 0.10 * s.y, 0.08 * s.z),
                    8,
                    10,
                    [1.0, 0.52, 0.14],
                    [0.1, 0.1, 2.4, MAT_CIRCUIT],
                );
            }
        }
    }

    if selected {
        let (bmin, bmax) = obj.world_aabb();
        let pad = Vector3::new(0.06, 0.06, 0.06);
        let min = bmin - pad;
        let max = bmax + pad;
        let t = 0.028;
        let wire_col = [1.0, 0.84, 0.18];
        let wire_mat = [0.0, 1.0, 1.0, MAT_UNLIT];

        let edges = [
            (Vector3::new(min.x, min.y, min.z), Vector3::new(max.x, min.y, min.z)),
            (Vector3::new(max.x, min.y, min.z), Vector3::new(max.x, min.y, max.z)),
            (Vector3::new(max.x, min.y, max.z), Vector3::new(min.x, min.y, max.z)),
            (Vector3::new(min.x, min.y, max.z), Vector3::new(min.x, min.y, min.z)),
            (Vector3::new(min.x, max.y, min.z), Vector3::new(max.x, max.y, min.z)),
            (Vector3::new(max.x, max.y, min.z), Vector3::new(max.x, max.y, max.z)),
            (Vector3::new(max.x, max.y, max.z), Vector3::new(min.x, max.y, max.z)),
            (Vector3::new(min.x, max.y, max.z), Vector3::new(min.x, max.y, min.z)),
            (Vector3::new(min.x, min.y, min.z), Vector3::new(min.x, max.y, min.z)),
            (Vector3::new(max.x, min.y, min.z), Vector3::new(max.x, max.y, min.z)),
            (Vector3::new(max.x, min.y, max.z), Vector3::new(max.x, max.y, max.z)),
            (Vector3::new(min.x, min.y, max.z), Vector3::new(min.x, max.y, max.z)),
        ];
        for (a, b) in edges {
            let center = (a + b) * 0.5;
            let span = b - a;
            let half = Vector3::new((span.x.abs() * 0.5).max(t), (span.y.abs() * 0.5).max(t), (span.z.abs() * 0.5).max(t));
            append_box(out, center, half, 0.0, wire_col, wire_mat);
        }
    }
}

fn append_transform_gizmo(
    out: &mut Vec<Vertex3D>,
    origin: Vector3<f32>,
    mode: GizmoMode,
    active_axis: GizmoAxis,
    cam_dist: f32,
) {
    let scale = (cam_dist * 0.13).clamp(0.95, 3.6);
    let shaft_len = 1.45 * scale;
    let shaft_r = 0.042 * scale;
    let tip_r = 0.125 * scale;
    let tip_len = 0.36 * scale;
    let unlit = [0.0, 1.0, 1.0, MAT_UNLIT];

    let x_col = if active_axis == GizmoAxis::X { [1.0, 0.95, 0.35] } else { [0.98, 0.22, 0.28] };
    let y_col = if active_axis == GizmoAxis::Y { [1.0, 0.95, 0.35] } else { [0.22, 0.95, 0.38] };
    let z_col = if active_axis == GizmoAxis::Z { [1.0, 0.95, 0.35] } else { [0.24, 0.58, 1.00] };

    append_sphere(
        out,
        origin,
        Vector3::new(0.095 * scale, 0.095 * scale, 0.095 * scale),
        8,
        12,
        [0.95, 0.96, 1.0],
        unlit,
    );

    if mode == GizmoMode::Rotate {
        append_torus_arc(out, origin, Vector3::new(0.0, 0.0, 90.0), shaft_len * 0.85, shaft_r * 0.85, TAU, 28, 10, x_col, unlit);
        append_torus_arc(out, origin, Vector3::new(0.0, 0.0, 0.0), shaft_len * 0.85, shaft_r * 0.85, TAU, 28, 10, y_col, unlit);
        append_torus_arc(out, origin, Vector3::new(90.0, 0.0, 0.0), shaft_len * 0.85, shaft_r * 0.85, TAU, 28, 10, z_col, unlit);
        return;
    }

    let axes = [
        (Vector3::new(1.0, 0.0, 0.0), x_col),
        (Vector3::new(0.0, 1.0, 0.0), y_col),
        (Vector3::new(0.0, 0.0, 1.0), z_col),
    ];

    for (axis, col) in axes {
        append_cylinder(out, origin, axis, shaft_r, shaft_len, 12, col, unlit);
        let tip_base = origin + axis * shaft_len;
        match mode {
            GizmoMode::Translate => {
                append_cone(out, tip_base, axis, tip_r, tip_len, 14, col, unlit);
            }
            GizmoMode::Scale => {
                append_box(
                    out,
                    tip_base + axis * (tip_r * 0.8),
                    Vector3::new(tip_r * 0.85, tip_r * 0.85, tip_r * 0.85),
                    0.0,
                    col,
                    unlit,
                );
            }
            GizmoMode::Rotate => {}
        }
    }
}

// ============================================================================
// 4. RAYCASTING & 3D PICKING MATH
// ============================================================================

fn screen_to_world_ray(
    cursor_px: (f32, f32),
    screen_size: (f32, f32),
    inv_view_proj: Matrix4<f32>,
) -> (Vector3<f32>, Vector3<f32>) {
    let nx = (2.0 * cursor_px.0 / screen_size.0.max(1.0)) - 1.0;
    let ny = 1.0 - (2.0 * cursor_px.1 / screen_size.1.max(1.0));

    let near_h = inv_view_proj * cgmath::Vector4::new(nx, ny, 0.0, 1.0);
    let far_h = inv_view_proj * cgmath::Vector4::new(nx, ny, 1.0, 1.0);

    let near = near_h.truncate() / near_h.w;
    let far = far_h.truncate() / far_h.w;
    let dir = (far - near).normalize();
    (near, dir)
}

fn ray_aabb_intersect(ro: Vector3<f32>, rd: Vector3<f32>, bmin: Vector3<f32>, bmax: Vector3<f32>) -> Option<f32> {
    let mut tmin = 0.001_f32;
    let mut tmax = 1000.0_f32;

    for i in 0..3 {
        let inv_d = 1.0 / rd[i].abs().max(1e-6) * rd[i].signum();
        let mut t0 = (bmin[i] - ro[i]) * inv_d;
        let mut t1 = (bmax[i] - ro[i]) * inv_d;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmax < tmin {
            return None;
        }
    }
    Some(tmin)
}

fn pick_gizmo_axis(ro: Vector3<f32>, rd: Vector3<f32>, origin: Vector3<f32>, cam_dist: f32) -> GizmoAxis {
    let scale = (cam_dist * 0.13).clamp(0.95, 3.6);
    let len = 1.85 * scale;
    let thick = 0.22 * scale;

    let x_hit = ray_aabb_intersect(
        ro,
        rd,
        origin + Vector3::new(0.1, -thick, -thick),
        origin + Vector3::new(len, thick, thick),
    );
    let y_hit = ray_aabb_intersect(
        ro,
        rd,
        origin + Vector3::new(-thick, 0.1, -thick),
        origin + Vector3::new(thick, len, thick),
    );
    let z_hit = ray_aabb_intersect(
        ro,
        rd,
        origin + Vector3::new(-thick, -thick, 0.1),
        origin + Vector3::new(thick, thick, len),
    );

    let mut best = (GizmoAxis::None, f32::MAX);
    if let Some(t) = x_hit.filter(|&t| t < best.1) {
        best = (GizmoAxis::X, t);
    }
    if let Some(t) = y_hit.filter(|&t| t < best.1) {
        best = (GizmoAxis::Y, t);
    }
    if let Some(t) = z_hit.filter(|&t| t < best.1) {
        best = (GizmoAxis::Z, t);
    }
    best.0
}

// ============================================================================
// 5. WGPU 3D RENDER PIPELINE (SHADOW MAP PASS 0 + SKY/PBR PASS 1 + EGUI PASS 2)
// ============================================================================

struct Engine3DGpuPipeline {
    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,
    shadow_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
}

impl Engine3DGpuPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let shader_src = include_str!("../shaders/engine_pbr_3d.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Metatopia 3D PBR + 2048 PCF Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3D Scene Uniform Buffer"),
            size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("3D Scene Uniform BGL"),
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

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3D Scene Uniform BG"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("2048x2048 Directional Shadow Map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Hardware Bilinear Comparison Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Map BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Map BG"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 36, shader_location: 3 },
            ],
        };

        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&uniform_bgl],
            push_constant_ranges: &[],
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pass 0: 2048x2048 Directional Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_shadow",
                buffers: &[vertex_buffer_layout.clone()],
            },
            fragment: None,
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
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D Procedural Sky Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("3D Mesh PBR + Shadow Pipeline Layout"),
            bind_group_layouts: &[&uniform_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("3D Mesh PBR + 2048 PCF Shadow Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_mesh",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_mesh",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
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

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3D Dynamic Mesh Vertex Buffer"),
            size: (MAX_VERTS * std::mem::size_of::<Vertex3D>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = Self::create_depth_view(device, width.max(1), height.max(1));

        Self {
            shadow_pipeline,
            sky_pipeline,
            mesh_pipeline,
            vertex_buffer,
            uniform_buffer,
            uniform_bg,
            shadow_bg,
            shadow_view,
            depth_view,
            depth_size: (width.max(1), height.max(1)),
        }
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("3D Scene Depth Buffer"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn ensure_depth_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.depth_size != (w, h) {
            self.depth_view = Self::create_depth_view(device, w, h);
            self.depth_size = (w, h);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        uniform: &SceneUniform,
        vertices: &[Vertex3D],
        egui_renderer: &mut egui_wgpu::Renderer,
        paint_jobs: &[egui::ClippedPrimitive],
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
    ) {
        self.ensure_depth_size(device, screen_descriptor.size_in_pixels[0], screen_descriptor.size_in_pixels[1]);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniform));

        let count = vertices.len().min(MAX_VERTS);
        if count > 0 {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices[..count]));
        }

        // Pass 0: 2048x2048 Directional Depth Shadow Map
        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass 0: 2048x2048 Directional Shadow Map"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if count > 0 {
                shadow_pass.set_pipeline(&self.shadow_pipeline);
                shadow_pass.set_bind_group(0, &self.uniform_bg, &[]);
                shadow_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                shadow_pass.draw(0..count as u32, 0..1);
            }
        }

        // Pass 1: 3D Sky + PBR Scene with 9-Tap Hardware Bilinear PCF Shadows
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass 1: 3D Sky + PBR Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.04, b: 0.08, a: 1.0 }),
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

            rpass.set_pipeline(&self.sky_pipeline);
            rpass.set_bind_group(0, &self.uniform_bg, &[]);
            rpass.draw(0..3, 0..1);

            if count > 0 {
                rpass.set_pipeline(&self.mesh_pipeline);
                rpass.set_bind_group(0, &self.uniform_bg, &[]);
                rpass.set_bind_group(1, &self.shadow_bg, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.draw(0..count as u32, 0..1);
            }
        }

        // Pass 2: Egui Editor Panels + 3D Viewport HUD Overlays
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass 2: Egui Editor & 3D HUD Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
            egui_renderer.render(&mut rpass, paint_jobs, screen_descriptor);
        }
    }
}

// ============================================================================
// 6. 3D GAME ENGINE & EDITOR STATE
// ============================================================================

struct Metatopia3DEngineState {
    objects: Vec<SceneObject3D>,
    initial_snapshot: Vec<SceneObject3D>,
    selected_id: Option<u32>,
    next_id: u32,
    play_mode: bool,
    gizmo_mode: GizmoMode,
    active_gizmo_axis: GizmoAxis,
    bottom_tab: BottomTab,
    show_grid: bool,
    show_youtube_overlay: bool,
    shader_debug_view: u32,

    cam_target: Vector3<f32>,
    cam_yaw: f32,
    cam_pitch: f32,
    cam_distance: f32,
    cam_fov_deg: f32,
    viewport_aspect: f32,

    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,

    projectiles: Vec<Projectile3D>,
    particles: Vec<Particle3D>,
    score: u32,
    enemies_defeated: u32,
    wave_number: u32,
    player_hp: f32,
    weapon_heat: f32,
    fire_cooldown: f32,
    hit_marker_timer: f32,

    script_engine: ScriptEngine,
    script_ast: Option<rhai::AST>,
    script_entity: ScriptEntityState,
    script_source: String,
    console_logs: Vec<String>,
    showcase_chapter_title: String,
    showcase_chapter_subtitle: String,
    showcase_progress: f32,
}

impl Metatopia3DEngineState {
    fn new() -> Self {
        let mut objects = Vec::new();
        let mut id = 1;
        let mut push = |mut obj: SceneObject3D| {
            obj.id = id;
            id += 1;
            objects.push(obj);
        };

        // 1. Arena Ground Floor (64m x 64m Chamfered Hex Cyber-Grid with Wet Puddles)
        push(SceneObject3D {
            id: 0,
            name: "Arena_HexFloor_64m".into(),
            kind: MeshKind::ArenaFloor,
            position: Vector3::new(0.0, -0.25, 0.0),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.11, 0.14, 0.20],
            metallic: 0.25,
            roughness: 0.55,
            emissive: 0.0,
            has_gravity: false,
            is_static: true,
            alive: true,
            hp: 9999.0,
            max_hp: 9999.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 2. Central Reflective Water Basin under the Suspension Bridge
        push(SceneObject3D {
            id: 0,
            name: "Central_Reflection_Basin".into(),
            kind: MeshKind::WaterPool,
            position: Vector3::new(0.0, 0.04, -1.8),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.15, 1.0, 0.95),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.05, 0.28, 0.46],
            metallic: 0.15,
            roughness: 0.05,
            emissive: 0.08,
            has_gravity: false,
            is_static: true,
            alive: true,
            hp: 9999.0,
            max_hp: 9999.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 3. Elevated Cyber Suspension Bridge spanning over the Water Basin
        push(SceneObject3D {
            id: 0,
            name: "Cyber_Suspension_Bridge".into(),
            kind: MeshKind::SuspensionBridge,
            position: Vector3::new(0.0, 0.82, -1.8),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.22, 0.27, 0.36],
            metallic: 0.84,
            roughness: 0.22,
            emissive: 0.06,
            has_gravity: false,
            is_static: true,
            alive: true,
            hp: 9999.0,
            max_hp: 9999.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 4. Player Character: Vanguard Striker Mech
        push(SceneObject3D {
            id: 0,
            name: "Player_Vanguard_Mech".into(),
            kind: MeshKind::PlayerMech,
            position: Vector3::new(0.0, 1.05, 6.4),
            rotation_deg: Vector3::new(0.0, 180.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.16, 0.56, 0.96],
            metallic: 0.78,
            roughness: 0.22,
            emissive: 0.16,
            has_gravity: true,
            is_static: false,
            alive: true,
            hp: 100.0,
            max_hp: 100.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 5. Monumental Sci-Fi Portal Archway (North Sector Gate)
        push(SceneObject3D {
            id: 0,
            name: "North_Portal_Archway".into(),
            kind: MeshKind::SciFiArchway,
            position: Vector3::new(0.0, 0.0, -9.8),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.15, 1.10, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.24, 0.30, 0.40],
            metallic: 0.86,
            roughness: 0.20,
            emissive: 0.12,
            has_gravity: false,
            is_static: true,
            alive: true,
            hp: 9999.0,
            max_hp: 9999.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 6. Central Neon Circuit Reactor Core (framed under the Portal Archway)
        push(SceneObject3D {
            id: 0,
            name: "Reactor_Core_Prime".into(),
            kind: MeshKind::EnergyCore,
            position: Vector3::new(0.0, 2.75, -9.8),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.25, 1.25, 1.25),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.08, 0.90, 1.0],
            metallic: 0.20,
            roughness: 0.10,
            emissive: 1.55,
            has_gravity: false,
            is_static: false,
            alive: true,
            hp: 300.0,
            max_hp: 300.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 7. Autonomous Defense Turret Tower (East Bastion — tracks & fires at Player!)
        push(SceneObject3D {
            id: 0,
            name: "Defense_Turret_East".into(),
            kind: MeshKind::TurretTower,
            position: Vector3::new(5.8, 1.45, -5.8),
            rotation_deg: Vector3::new(0.0, -35.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.30, 0.34, 0.42],
            metallic: 0.82,
            roughness: 0.24,
            emissive: 0.35,
            has_gravity: false,
            is_static: false,
            alive: true,
            hp: 175.0,
            max_hp: 175.0,
            ai_phase: 0.8,
            fire_cooldown: 1.1,
        });

        // 8. Graviton Jump Pad (launches Player Mech high over the Water Basin!)
        push(SceneObject3D {
            id: 0,
            name: "JumpPad_South_Launcher".into(),
            kind: MeshKind::JumpPad,
            position: Vector3::new(-3.2, 0.18, 2.8),
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.15, 1.0, 1.15),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: [0.12, 0.98, 0.56],
            metallic: 0.55,
            roughness: 0.18,
            emissive: 1.45,
            has_gravity: false,
            is_static: true,
            alive: true,
            hp: 9999.0,
            max_hp: 9999.0,
            ai_phase: 0.0,
            fire_cooldown: 0.0,
        });

        // 9. Physics Cargo Crates (Selected ID = 9 for Editor Gizmo Drag + Play Mode Gravity Drop!)
        let crate_specs = [
            ("Plasma_Crate_Alpha", Vector3::new(-2.8, 0.85, 0.8), [0.98, 0.50, 0.14]),
            ("Titanium_Crate_Beta", Vector3::new(3.2, 0.85, 1.4), [0.22, 0.78, 0.98]),
            ("Stacked_Crate_Top", Vector3::new(3.2, 2.35, 1.4), [0.96, 0.28, 0.46]),
        ];
        for (name, pos, col) in crate_specs {
            push(SceneObject3D {
                id: 0,
                name: name.into(),
                kind: MeshKind::PhysicsCrate,
                position: pos,
                rotation_deg: Vector3::new(0.0, 18.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                color: col,
                metallic: 0.45,
                roughness: 0.30,
                emissive: 0.14,
                has_gravity: true,
                is_static: false,
                alive: true,
                hp: 65.0,
                max_hp: 65.0,
                ai_phase: 0.0,
                fire_cooldown: 0.0,
            });
        }

        // 10. Autonomous Hover-Drone Enemies (Patrol + Attack AI in Play Mode)
        let drone_specs = [
            ("HoverDrone_Viper_01", Vector3::new(-4.8, 2.25, -4.4), 0.0),
            ("HoverDrone_Viper_02", Vector3::new(0.2, 2.55, -5.2), 2.1),
            ("HoverDrone_Viper_03", Vector3::new(4.6, 2.35, -2.4), 4.2),
        ];
        for (name, pos, phase) in drone_specs {
            push(SceneObject3D {
                id: 0,
                name: name.into(),
                kind: MeshKind::HoverDrone,
                position: pos,
                rotation_deg: Vector3::new(0.0, 0.0, 0.0),
                scale: Vector3::new(1.05, 1.05, 1.05),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                color: [0.92, 0.18, 0.28],
                metallic: 0.80,
                roughness: 0.22,
                emissive: 0.42,
                has_gravity: false,
                is_static: false,
                alive: true,
                hp: 135.0,
                max_hp: 135.0,
                ai_phase: phase,
                fire_cooldown: 1.5,
            });
        }

        // 11. Brushed Titanium Cyber Pillars framing the Arena
        for (name, x, z) in [
            ("Pillar_West_Front", -8.8, 2.8),
            ("Pillar_East_Front", 8.8, 2.8),
            ("Pillar_West_Back", -8.8, -7.8),
            ("Pillar_East_Back", 8.8, -7.8),
        ] {
            push(SceneObject3D {
                id: 0,
                name: name.into(),
                kind: MeshKind::CyberPillar,
                position: Vector3::new(x, 1.8, z),
                rotation_deg: Vector3::new(0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.15, 1.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                color: [0.24, 0.28, 0.36],
                metallic: 0.86,
                roughness: 0.20,
                emissive: 0.06,
                has_gravity: false,
                is_static: true,
                alive: true,
                hp: 9999.0,
                max_hp: 9999.0,
                ai_phase: 0.0,
                fire_cooldown: 0.0,
            });
        }

        // 12. Collectible Golden Plasma Orbs
        for (idx, (x, y, z)) in [(-5.4, 1.25, 0.0), (0.0, 1.65, -1.8), (5.4, 1.25, 0.0)].iter().enumerate() {
            push(SceneObject3D {
                id: 0,
                name: format!("Plasma_Orb_{:02}", idx + 1),
                kind: MeshKind::PlasmaOrb,
                position: Vector3::new(*x, *y, *z),
                rotation_deg: Vector3::new(0.0, 0.0, 0.0),
                scale: Vector3::new(0.95, 0.95, 0.95),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                color: [1.0, 0.84, 0.18],
                metallic: 0.90,
                roughness: 0.10,
                emissive: 1.85,
                has_gravity: false,
                is_static: false,
                alive: true,
                hp: 10.0,
                max_hp: 10.0,
                ai_phase: idx as f32 * 1.4,
                fire_cooldown: 0.0,
            });
        }

        let initial_snapshot = objects.clone();
        let mut script_engine = ScriptEngine::new();
        let script_source = r#"// Metatopia 3D Engine — Live Rhai Gameplay & Shader Controller
fn update(e, dt) {
    e.rotate(1.4 * dt, 0.0, 0.0);
    let pulse = 1.25 + 0.35 * sin(e.get_x() + dt * 3.0);
    e.set_emissive(pulse);
    return e;
}
"#
        .to_string();
        let script_ast = script_engine.compile(&script_source).ok();
        let mut script_entity = ScriptEntityState::new(6, "Reactor_Core_Prime", [0.0, 2.75, -9.8], [0.08, 0.90, 1.0]);
        script_entity.emissive = 1.45;

        Self {
            objects,
            initial_snapshot,
            selected_id: Some(9), // Plasma_Crate_Alpha
            next_id: id,
            play_mode: false,
            gizmo_mode: GizmoMode::Translate,
            active_gizmo_axis: GizmoAxis::None,
            bottom_tab: BottomTab::AssetBrowser,
            show_grid: true,
            show_youtube_overlay: false,
            shader_debug_view: 0,

            cam_target: Vector3::new(0.0, 1.35, -1.2),
            cam_yaw: 0.36,
            cam_pitch: 0.42,
            cam_distance: 15.8,
            cam_fov_deg: 55.0,
            viewport_aspect: 16.0 / 9.0,

            sun_azimuth_deg: 42.0,
            sun_elevation_deg: 46.0,
            sun_intensity: 3.6,

            projectiles: Vec::new(),
            particles: Vec::new(),
            score: 0,
            enemies_defeated: 0,
            wave_number: 1,
            player_hp: 100.0,
            weapon_heat: 0.0,
            fire_cooldown: 0.0,
            hit_marker_timer: 0.0,

            script_engine,
            script_ast,
            script_entity,
            script_source,
            console_logs: vec![
                "[BOOT] Metatopia 3D Engine v1.0.0 (Apple Metal + wgpu + egui)".into(),
                "[SHADOW] Initialized 2048x2048 Depth32Float Directional Shadow Map + 9-Tap Bilinear PCF".into(),
                "[SHADER] Compiled Procedural PBR Materials: HexFloor, BrushedAniso, NeonCircuit, WaterBasin".into(),
                "[SCENE] Loaded Cyber_Arena_Prime.scn (18 3D entities, Suspension Bridge, Turret & 3 HoverDrones)".into(),
            ],
            showcase_chapter_title: "01 // 3D LEVEL EDITOR & 2048x2048 PCF SHADOW MAPPING".into(),
            showcase_chapter_subtitle: "Real-Time Directional Depth Shadows  •  RGB Transform Gizmos  •  Procedural PBR Materials".into(),
            showcase_progress: 0.0,
        }
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.console_logs.push(msg.into());
        if self.console_logs.len() > 60 {
            self.console_logs.remove(0);
        }
    }

    fn camera_eye(&self) -> Vector3<f32> {
        let cp = self.cam_pitch.cos();
        let sp = self.cam_pitch.sin();
        let sy = self.cam_yaw.sin();
        let cy = self.cam_yaw.cos();
        self.cam_target + Vector3::new(self.cam_distance * cp * sy, self.cam_distance * sp, self.cam_distance * cp * cy)
    }

    fn sun_direction(&self) -> Vector3<f32> {
        let az = self.sun_azimuth_deg.to_radians();
        let el = self.sun_elevation_deg.to_radians();
        Vector3::new(el.cos() * az.sin(), el.sin(), el.cos() * az.cos()).normalize()
    }

    fn view_proj_matrices(&self, aspect: f32) -> (Matrix4<f32>, Matrix4<f32>) {
        let eye = self.camera_eye();
        let view = Matrix4::look_at_rh(
            Point3::new(eye.x, eye.y, eye.z),
            Point3::new(self.cam_target.x, self.cam_target.y, self.cam_target.z),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let proj = OPENGL_TO_WGPU_MATRIX * perspective(Rad(self.cam_fov_deg.to_radians()), aspect.max(0.2), 0.1, 220.0);
        let vp = proj * view;
        let inv_vp = vp.invert().unwrap_or_else(Matrix4::identity);
        (vp, inv_vp)
    }

    fn start_play_mode(&mut self) {
        if !self.play_mode {
            self.initial_snapshot = self.objects.clone();
            self.play_mode = true;
            self.projectiles.clear();
            self.particles.clear();
            self.score = 0;
            self.enemies_defeated = 0;
            self.wave_number = 1;
            self.player_hp = 100.0;
            self.weapon_heat = 0.0;
            self.hit_marker_timer = 0.0;
            self.shader_debug_view = 0;
            self.log("[PLAY] ▶ Entered Play Mode: Rigid-Body Physics, Turret & Hover-Drone AI active!");
        }
    }

    fn stop_play_mode(&mut self) {
        if self.play_mode {
            self.objects = self.initial_snapshot.clone();
            self.play_mode = false;
            self.projectiles.clear();
            self.particles.clear();
            self.hit_marker_timer = 0.0;
            self.cam_target = Vector3::new(0.0, 1.35, -1.2);
            self.log("[STOP] ⏹ Restored Scene Snapshot back to Editor state.");
        }
    }

    fn spawn_prefab(&mut self, kind: MeshKind) {
        let id = self.next_id;
        self.next_id += 1;
        let offset = ((id as f32 * 1.7).sin()) * 2.2;
        let (pos, col, grav, stat) = match kind {
            MeshKind::PhysicsCrate => (Vector3::new(offset, 4.2, 0.4), [0.96, 0.56, 0.16], true, false),
            MeshKind::HoverDrone => (Vector3::new(offset, 2.5, -3.6), [0.95, 0.20, 0.30], false, false),
            MeshKind::TurretTower => (Vector3::new(offset, 1.45, -4.5), [0.30, 0.35, 0.44], false, false),
            MeshKind::SciFiArchway => (Vector3::new(offset, 0.0, -5.0), [0.24, 0.30, 0.40], false, true),
            MeshKind::EnergyCore => (Vector3::new(offset, 2.0, -2.0), [0.15, 0.92, 1.00], false, false),
            MeshKind::CyberPillar => (Vector3::new(offset, 1.8, -1.5), [0.26, 0.30, 0.38], false, true),
            MeshKind::PlasmaOrb => (Vector3::new(offset, 1.4, 1.0), [1.0, 0.85, 0.20], false, false),
            MeshKind::StepPlatform => (Vector3::new(offset, 0.8, 1.5), [0.22, 0.28, 0.36], false, true),
            MeshKind::JumpPad => (Vector3::new(offset, 0.18, 2.0), [0.12, 0.98, 0.56], false, true),
            _ => (Vector3::new(0.0, 1.5, 0.0), [0.5, 0.6, 0.7], false, false),
        };
        let name = format!("{}_{:02}", kind.label().replace(' ', "_"), id);
        self.objects.push(SceneObject3D {
            id,
            name: name.clone(),
            kind,
            position: pos,
            rotation_deg: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            color: col,
            metallic: 0.68,
            roughness: 0.24,
            emissive: if matches!(kind, MeshKind::EnergyCore | MeshKind::PlasmaOrb | MeshKind::JumpPad) { 1.45 } else { 0.15 },
            has_gravity: grav,
            is_static: stat,
            alive: true,
            hp: 110.0,
            max_hp: 110.0,
            ai_phase: id as f32,
            fire_cooldown: 1.2,
        });
        self.selected_id = Some(id);
        self.log(format!("[SPAWN] Added 3D Prefab '{}' (ID #{}) at ({:.1}, {:.1}, {:.1})", name, id, pos.x, pos.y, pos.z));
    }

    fn spawn_spark_burst(&mut self, center: Vector3<f32>, color: [f32; 3], count: usize) {
        for i in 0..count {
            let a = (i as f32 / count as f32) * TAU + (center.x + center.z);
            let elev = ((i * 7) % 11) as f32 / 11.0 * 1.2 - 0.15;
            let speed = 3.2 + ((i * 13) % 9) as f32 * 0.75;
            let vel = Vector3::new(a.cos() * speed, elev * speed + 2.8, a.sin() * speed);
            self.particles.push(Particle3D {
                pos: center,
                vel,
                color,
                size: 0.12,
                life: 0.62,
                max_life: 0.62,
            });
        }
    }

    fn fire_player_plasma(&mut self) {
        let Some(player_idx) = self.objects.iter().position(|o| o.kind == MeshKind::PlayerMech && o.alive) else {
            return;
        };
        let p_pos = self.objects[player_idx].position;

        // Find nearest alive hostile (HoverDrone or TurretTower) for 3D tactical lock-on
        let nearest_hostile = self
            .objects
            .iter()
            .filter(|o| o.alive && matches!(o.kind, MeshKind::HoverDrone | MeshKind::TurretTower))
            .min_by(|a, b| {
                (a.position - p_pos)
                    .magnitude2()
                    .partial_cmp(&(b.position - p_pos).magnitude2())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|o| o.position);

        if let Some(target_pos) = nearest_hostile {
            let to_t = target_pos - p_pos;
            self.objects[player_idx].rotation_deg.y = to_t.x.atan2(to_t.z).to_degrees();
        }

        let yaw = self.objects[player_idx].rotation_deg.y.to_radians();
        let fwd = rotate_y(Vector3::new(0.0, 0.0, 1.0), yaw).normalize();
        let muzzle = p_pos + rotate_y(Vector3::new(0.56, 0.20, 0.95), yaw);
        let aim_dir = nearest_hostile
            .map(|t| (t - muzzle).normalize())
            .unwrap_or(fwd);

        self.projectiles.push(Projectile3D {
            pos: muzzle,
            vel: aim_dir * 34.0,
            color: [0.12, 0.95, 1.0],
            ttl: 2.2,
            from_enemy: false,
        });
        self.weapon_heat = (self.weapon_heat + 0.12).min(1.0);
        self.spawn_spark_burst(muzzle, [0.2, 0.95, 1.0], 5);
    }

    fn step_physics_and_gameplay(&mut self, dt: f32, time: f32, held_keys: &[bool; 10]) {
        if let Some(ast) = self.script_ast.as_ref() {
            let _ = self.script_engine.execute_update(ast, &mut self.script_entity, dt);
        }
        let emissive_boost = self.script_entity.emissive.clamp(0.4, 2.5);

        for obj in &mut self.objects {
            if !obj.alive {
                continue;
            }
            match obj.kind {
                MeshKind::EnergyCore => {
                    obj.rotation_deg.y = (obj.rotation_deg.y + dt * 45.0) % 360.0;
                    obj.emissive = 1.35 * emissive_boost;
                    if self.play_mode {
                        obj.position.y = 2.75 + (time * 2.2).sin() * 0.22;
                    }
                }
                MeshKind::PlasmaOrb => {
                    obj.rotation_deg.y = (obj.rotation_deg.y + dt * 95.0) % 360.0;
                    obj.position.y = 1.25 + (time * 3.0 + obj.ai_phase).sin() * 0.16;
                }
                _ => {}
            }
        }

        if !self.play_mode {
            return;
        }

        self.fire_cooldown = (self.fire_cooldown - dt).max(0.0);
        self.weapon_heat = (self.weapon_heat - dt * 0.38).max(0.0);
        self.hit_marker_timer = (self.hit_marker_timer - dt).max(0.0);

        let mut player_pos = Vector3::new(0.0, 1.05, 2.0);
        if let Some(player_idx) = self.objects.iter().position(|o| o.kind == MeshKind::PlayerMech && o.alive) {
            let mut move_vec = Vector3::new(0.0, 0.0, 0.0);
            let cam_fwd = Vector3::new(-self.cam_yaw.sin(), 0.0, -self.cam_yaw.cos()).normalize();
            let cam_right = Vector3::new(self.cam_yaw.cos(), 0.0, -self.cam_yaw.sin()).normalize();

            if held_keys[0] { move_vec += cam_fwd; }
            if held_keys[1] { move_vec -= cam_fwd; }
            if held_keys[2] { move_vec -= cam_right; }
            if held_keys[3] { move_vec += cam_right; }

            if move_vec.magnitude2() > 0.01 {
                let dir = move_vec.normalize();
                self.objects[player_idx].velocity.x = dir.x * 7.5;
                self.objects[player_idx].velocity.z = dir.z * 7.5;
                let target_yaw = dir.x.atan2(dir.z).to_degrees();
                self.objects[player_idx].rotation_deg.y = target_yaw;
            } else {
                self.objects[player_idx].velocity.x *= 0.82;
                self.objects[player_idx].velocity.z *= 0.82;
            }

            if held_keys[4] && self.objects[player_idx].position.y <= 1.35 {
                self.objects[player_idx].velocity.y = 10.2;
                let p = self.objects[player_idx].position;
                self.spawn_spark_burst(p - Vector3::new(0.0, 0.85, 0.0), [1.0, 0.58, 0.14], 14);
            }

            self.objects[player_idx].position.x = self.objects[player_idx].position.x.clamp(-5.8, 5.8);
            self.objects[player_idx].position.z = self.objects[player_idx].position.z.clamp(-6.2, 5.8);
            player_pos = self.objects[player_idx].position;
            self.cam_target = self.cam_target * 0.88 + (player_pos * 0.55 + Vector3::new(0.0, 1.25, -1.8) * 0.45) * 0.12;
        }

        if held_keys[5] && self.fire_cooldown <= 0.0 {
            self.fire_player_plasma();
            self.fire_cooldown = 0.28;
        }

        let jump_pads: Vec<Vector3<f32>> = self
            .objects
            .iter()
            .filter(|o| o.kind == MeshKind::JumpPad && o.alive)
            .map(|o| o.position)
            .collect();

        let mut spark_events: Vec<(Vector3<f32>, [f32; 3], usize)> = Vec::new();
        let mut new_enemy_bolts: Vec<Projectile3D> = Vec::new();

        for i in 0..self.objects.len() {
            if !self.objects[i].alive || self.objects[i].is_static {
                continue;
            }

            if self.objects[i].kind == MeshKind::HoverDrone {
                let phase = self.objects[i].ai_phase;
                let to_player = player_pos - self.objects[i].position;
                let flat_dist = Vector3::new(to_player.x, 0.0, to_player.z).magnitude().max(0.1);
                let dir = Vector3::new(to_player.x / flat_dist, 0.0, to_player.z / flat_dist);
                let orbit = Vector3::new(-dir.z, 0.0, dir.x) * (time * 1.9 + phase).sin();
                let desired_v = (dir * 2.5 + orbit * 2.9).normalize() * 3.8;

                self.objects[i].position.x = (self.objects[i].position.x + desired_v.x * dt).clamp(-5.5, 5.5);
                self.objects[i].position.z = (self.objects[i].position.z + desired_v.z * dt).clamp(-6.2, 2.8);
                self.objects[i].position.y = 2.25 + (time * 3.2 + phase).sin() * 0.35;
                self.objects[i].rotation_deg.y = to_player.x.atan2(to_player.z).to_degrees();
                continue;
            }

            if self.objects[i].kind == MeshKind::TurretTower {
                let to_player = player_pos - self.objects[i].position;
                let target_yaw = to_player.x.atan2(to_player.z).to_degrees();
                self.objects[i].rotation_deg.y = target_yaw;
                self.objects[i].fire_cooldown -= dt;
                if self.objects[i].fire_cooldown <= 0.0 {
                    self.objects[i].fire_cooldown = 1.35;
                    let head_pos = self.objects[i].position + Vector3::new(0.0, 0.82, 0.0);
                    let aim_dir = ((player_pos + Vector3::new(0.0, 0.25, 0.0)) - head_pos).normalize();
                    let muzzle = head_pos + aim_dir * 1.25;
                    new_enemy_bolts.push(Projectile3D {
                        pos: muzzle,
                        vel: aim_dir * 18.5,
                        color: [1.0, 0.22, 0.16],
                        ttl: 2.4,
                        from_enemy: true,
                    });
                    spark_events.push((muzzle, [1.0, 0.30, 0.15], 6));
                }
                continue;
            }

            if self.objects[i].has_gravity {
                self.objects[i].velocity.y -= 20.0 * dt;
            }

            let vel = self.objects[i].velocity;
            self.objects[i].position += vel * dt;

            let pos = self.objects[i].position;
            let on_bridge = pos.x.abs() < 5.6 && (pos.z - (-1.8)).abs() < 1.55 && pos.y >= 0.75;
            let base_ground = if on_bridge { 0.98 } else { 0.0 };

            let min_y = match self.objects[i].kind {
                MeshKind::PlayerMech => base_ground + 1.02,
                MeshKind::PhysicsCrate => base_ground + 0.65 * self.objects[i].scale.y,
                _ => base_ground + 0.5,
            };

            if self.objects[i].position.y < min_y {
                self.objects[i].position.y = min_y;
                if self.objects[i].velocity.y < -2.6 {
                    let bounce_pos = self.objects[i].position - Vector3::new(0.0, 0.5, 0.0);
                    spark_events.push((bounce_pos, [0.35, 0.85, 1.0], 8));
                    self.objects[i].velocity.y = -self.objects[i].velocity.y * 0.44;
                } else {
                    self.objects[i].velocity.y = 0.0;
                }
                self.objects[i].velocity.x *= 0.88;
                self.objects[i].velocity.z *= 0.88;
            }

            for pad_pos in &jump_pads {
                let d_xz = Vector3::new(self.objects[i].position.x - pad_pos.x, 0.0, self.objects[i].position.z - pad_pos.z).magnitude();
                if d_xz < 1.15 && (self.objects[i].position.y - pad_pos.y).abs() < 1.35 && self.objects[i].velocity.y <= 2.0 {
                    self.objects[i].velocity.y = 12.2;
                    spark_events.push((*pad_pos + Vector3::new(0.0, 0.3, 0.0), [0.15, 1.0, 0.58], 18));
                }
            }
        }

        self.projectiles.extend(new_enemy_bolts);
        for (pos, col, count) in spark_events {
            self.spawn_spark_burst(pos, col, count);
        }

        for i in 0..self.objects.len() {
            if self.objects[i].kind == MeshKind::PlasmaOrb && self.objects[i].alive {
                if (self.objects[i].position - player_pos).magnitude() < 1.45 {
                    let orb_pos = self.objects[i].position;
                    let orb_name = self.objects[i].name.clone();
                    self.objects[i].alive = false;
                    self.score += 100;
                    self.player_hp = (self.player_hp + 15.0).min(100.0);
                    self.spawn_spark_burst(orb_pos, [1.0, 0.88, 0.22], 20);
                    self.log(format!("[PICKUP] Collected '{}' (+100 PTS)", orb_name));
                }
            }
        }

        let mut remaining_proj = Vec::new();
        let old_projs = std::mem::take(&mut self.projectiles);
        for mut proj in old_projs {
            proj.pos += proj.vel * dt;
            proj.ttl -= dt;
            if proj.ttl <= 0.0 {
                continue;
            }

            if proj.from_enemy {
                if (proj.pos - player_pos).magnitude() < 0.95 {
                    self.player_hp = (self.player_hp - 6.0).max(25.0);
                    self.spawn_spark_burst(proj.pos, [1.0, 0.25, 0.18], 10);
                    continue;
                }
                remaining_proj.push(proj);
                continue;
            }

            let mut hit = false;
            for i in 0..self.objects.len() {
                if !self.objects[i].alive {
                    continue;
                }
                if matches!(self.objects[i].kind, MeshKind::HoverDrone | MeshKind::TurretTower | MeshKind::PhysicsCrate) {
                    let (bmin, bmax) = self.objects[i].world_aabb();
                    if proj.pos.x >= bmin.x - 0.28
                        && proj.pos.x <= bmax.x + 0.28
                        && proj.pos.y >= bmin.y - 0.28
                        && proj.pos.y <= bmax.y + 0.28
                        && proj.pos.z >= bmin.z - 0.28
                        && proj.pos.z <= bmax.z + 0.28
                    {
                        self.objects[i].hp -= 35.0;
                        let knock = proj.vel.normalize() * 4.5 + Vector3::new(0.0, 2.2, 0.0);
                        if self.objects[i].kind == MeshKind::PhysicsCrate {
                            self.objects[i].velocity += knock;
                        }
                        let hit_pos = proj.pos;
                        self.hit_marker_timer = 0.22;
                        self.spawn_spark_burst(hit_pos, [1.0, 0.55, 0.15], 14);
                        if self.objects[i].hp <= 0.0 {
                            let dead_pos = self.objects[i].position;
                            let dead_name = self.objects[i].name.clone();
                            let dead_kind = self.objects[i].kind;
                            self.objects[i].alive = false;
                            if matches!(dead_kind, MeshKind::HoverDrone | MeshKind::TurretTower) {
                                self.enemies_defeated += 1;
                                self.score += 150;
                            } else {
                                self.score += 50;
                            }
                            self.spawn_spark_burst(dead_pos, [1.0, 0.30, 0.18], 32);
                            self.log(format!("[COMBAT] Destroyed '{}' (+150 PTS)!", dead_name));
                        }
                        hit = true;
                        break;
                    }
                }
            }
            if !hit {
                remaining_proj.push(proj);
            }
        }
        self.projectiles = remaining_proj;

        for p in &mut self.particles {
            p.vel.y -= 9.8 * dt;
            p.pos += p.vel * dt;
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    fn build_vertex_stream(&self, time: f32) -> Vec<Vertex3D> {
        let mut verts = Vec::with_capacity(65_536);

        for obj in &self.objects {
            let is_sel = !self.play_mode && self.selected_id == Some(obj.id);
            append_scene_object(&mut verts, obj, is_sel, time);
        }

        for proj in &self.projectiles {
            append_sphere(
                &mut verts,
                proj.pos,
                Vector3::new(0.16, 0.16, 0.16),
                8,
                12,
                proj.color,
                [0.1, 0.08, 2.8, MAT_CIRCUIT],
            );
        }

        for part in &self.particles {
            let s = part.size * (part.life / part.max_life).clamp(0.15, 1.0);
            append_box(
                &mut verts,
                part.pos,
                Vector3::new(s, s, s),
                time * 6.0,
                part.color,
                [0.1, 0.1, 2.4, MAT_CIRCUIT],
            );
        }

        if !self.play_mode {
            if let Some(sel_id) = self.selected_id {
                if let Some(sel) = self.objects.iter().find(|o| o.id == sel_id && o.alive) {
                    let cam_dist = (self.camera_eye() - sel.position).magnitude();
                    append_transform_gizmo(&mut verts, sel.position, self.gizmo_mode, self.active_gizmo_axis, cam_dist);
                }
            }
        }

        verts
    }

    fn build_uniform(&self, width: f32, height: f32, time: f32) -> SceneUniform {
        let aspect = if self.viewport_aspect > 0.2 {
            self.viewport_aspect
        } else {
            (width / height.max(1.0)).max(0.2)
        };
        let (vp, inv_vp) = self.view_proj_matrices(aspect);
        let eye = self.camera_eye();
        let sun = self.sun_direction();
        let light_vp = build_light_view_proj(sun);

        let mut point_lights = [[0.0_f32; 4]; 8];
        let mut light_slot = 0usize;

        if let Some(core) = self.objects.iter().find(|o| o.kind == MeshKind::EnergyCore && o.alive) {
            point_lights[light_slot * 2] = [core.position.x, core.position.y, core.position.z, 15.0];
            point_lights[light_slot * 2 + 1] = [0.12, 0.90, 1.0, 2.8];
            light_slot += 1;
        }
        if let Some(pad) = self.objects.iter().find(|o| o.kind == MeshKind::JumpPad && o.alive) {
            if light_slot < 4 {
                point_lights[light_slot * 2] = [pad.position.x, pad.position.y + 0.5, pad.position.z, 9.5];
                point_lights[light_slot * 2 + 1] = [0.12, 0.98, 0.56, 2.2];
                light_slot += 1;
            }
        }
        for proj in self.projectiles.iter().take(2) {
            if light_slot < 4 {
                point_lights[light_slot * 2] = [proj.pos.x, proj.pos.y, proj.pos.z, 8.5];
                point_lights[light_slot * 2 + 1] = [proj.color[0], proj.color[1], proj.color[2], 3.0];
                light_slot += 1;
            }
        }
        for orb in self.objects.iter().filter(|o| o.kind == MeshKind::PlasmaOrb && o.alive) {
            if light_slot < 4 {
                point_lights[light_slot * 2] = [orb.position.x, orb.position.y, orb.position.z, 7.5];
                point_lights[light_slot * 2 + 1] = [1.0, 0.84, 0.20, 2.0];
                light_slot += 1;
            }
        }

        let total_hostiles = 4.0_f32;
        let wave_progress = (self.enemies_defeated as f32 / total_hostiles).clamp(0.0, 1.0);

        SceneUniform {
            view_proj: mat4_to_array(vp),
            inv_view_proj: mat4_to_array(inv_vp),
            light_view_proj: mat4_to_array(light_vp),
            camera_pos: [eye.x, eye.y, eye.z, time],
            sun_dir: [sun.x, sun.y, sun.z, self.sun_intensity],
            viewport: [
                width,
                height,
                if self.play_mode { 1.0 } else { 0.0 },
                if self.show_grid { 1.0 } else { 0.0 },
            ],
            shadow_params: [
                1.0 / (SHADOW_MAP_SIZE as f32),
                0.022,
                0.92,
                self.shader_debug_view as f32,
            ],
            hud_params: [
                (self.player_hp / 100.0).clamp(0.0, 1.0),
                self.weapon_heat.clamp(0.0, 1.0),
                (self.hit_marker_timer / 0.22).clamp(0.0, 1.0),
                wave_progress,
            ],
            point_lights,
        }
    }
}

// ============================================================================
// 7. EGUI 3D ENGINE WORKBENCH + 3D VIEWPORT PAINTER OVERLAYS
// ============================================================================

fn draw_3d_viewport_overlays(ctx: &Context, state: &mut Metatopia3DEngineState, viewport_rect: Rect, time: f32) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        egui::Id::new("viewport_3d_overlays"),
    ));

    // 1. Top-Left 3D Viewport Status Pill
    let pill_rect = Rect::from_min_size(
        Pos2::new(viewport_rect.left() + 14.0, viewport_rect.top() + 12.0),
        Vec2::new(430.0, 28.0),
    );
    painter.rect(
        pill_rect,
        Rounding::same(6.0),
        Color32::from_rgba_unmultiplied(11, 15, 24, 215),
        Stroke::new(1.0, Color32::from_rgb(38, 62, 95)),
    );
    let debug_name = match state.shader_debug_view {
        1 => "SHADER VIEW: WORLD NORMALS",
        2 => "SHADER VIEW: METALLIC / ROUGHNESS",
        3 => "SHADER VIEW: 2048x2048 PCF SHADOW MAP",
        _ => "LIT PBR + 2048 PCF SHADOWS",
    };
    let pill_text = format!(
        "PERSP {:.0}°   |   {}   |   ENTITIES: {}",
        state.cam_fov_deg,
        debug_name,
        state.objects.iter().filter(|o| o.alive).count()
    );
    painter.text(
        Pos2::new(pill_rect.left() + 12.0, pill_rect.center().y),
        egui::Align2::LEFT_CENTER,
        pill_text,
        FontId::monospace(11.0),
        Color32::from_rgb(155, 215, 255),
    );

    // 2. Top-Right 3D Orientation Compass Gizmo (X / Y / Z Camera Axes)
    let compass_center = Pos2::new(viewport_rect.right() - 56.0, viewport_rect.top() + 54.0);
    let compass_r = 36.0;
    painter.circle(
        compass_center,
        compass_r + 6.0,
        Color32::from_rgba_unmultiplied(11, 15, 24, 205),
        Stroke::new(1.0, Color32::from_rgb(42, 64, 96)),
    );

    let eye = state.camera_eye();
    let fwd = (state.cam_target - eye).normalize();
    let right = fwd.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
    let up = right.cross(fwd).normalize();

    let mut axes = [
        ("X", Vector3::new(1.0, 0.0, 0.0), Color32::from_rgb(250, 68, 78)),
        ("Y", Vector3::new(0.0, 1.0, 0.0), Color32::from_rgb(65, 238, 105)),
        ("Z", Vector3::new(0.0, 0.0, 1.0), Color32::from_rgb(68, 150, 255)),
    ];
    axes.sort_by(|a, b| a.1.dot(fwd).partial_cmp(&b.1.dot(fwd)).unwrap_or(std::cmp::Ordering::Equal));

    for (label, axis_vec, col) in axes {
        let sx = axis_vec.dot(right);
        let sy = -axis_vec.dot(up);
        let end_pt = Pos2::new(
            compass_center.x + sx * compass_r * 0.78,
            compass_center.y + sy * compass_r * 0.78,
        );
        painter.line_segment([compass_center, end_pt], Stroke::new(2.6, col));
        painter.circle_filled(end_pt, 6.5, col);
        painter.text(
            end_pt,
            egui::Align2::CENTER_CENTER,
            label,
            FontId::proportional(9.5),
            Color32::from_rgb(10, 14, 20),
        );
    }

    // 3. Play Mode Tactical HUD (Crosshair + Hit-Marker + Radar + Hull/Heat Bars)
    if state.play_mode {
        let center = viewport_rect.center();

        let spread = 9.0 + state.weapon_heat * 14.0;
        let reticle_col = Color32::from_rgba_unmultiplied(80, 235, 255, 220);
        painter.circle_stroke(center, spread + 5.0, Stroke::new(1.2, Color32::from_rgba_unmultiplied(60, 190, 255, 110)));
        painter.line_segment([Pos2::new(center.x - spread - 9.0, center.y), Pos2::new(center.x - spread, center.y)], Stroke::new(2.0, reticle_col));
        painter.line_segment([Pos2::new(center.x + spread, center.y), Pos2::new(center.x + spread + 9.0, center.y)], Stroke::new(2.0, reticle_col));
        painter.line_segment([Pos2::new(center.x, center.y - spread - 9.0), Pos2::new(center.x, center.y - spread)], Stroke::new(2.0, reticle_col));
        painter.line_segment([Pos2::new(center.x, center.y + spread), Pos2::new(center.x, center.y + spread + 9.0)], Stroke::new(2.0, reticle_col));
        painter.circle_filled(center, 2.2, Color32::from_rgb(255, 240, 160));

        if state.hit_marker_timer > 0.0 {
            let alpha = ((state.hit_marker_timer / 0.22) * 255.0).clamp(0.0, 255.0) as u8;
            let hit_col = Color32::from_rgba_unmultiplied(255, 210, 55, alpha);
            let d1 = 6.0;
            let d2 = 16.0;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                painter.line_segment(
                    [Pos2::new(center.x + sx * d1, center.y + sy * d1), Pos2::new(center.x + sx * d2, center.y + sy * d2)],
                    Stroke::new(2.8, hit_col),
                );
            }
        }

        let hostiles_remaining = state
            .objects
            .iter()
            .filter(|o| o.alive && matches!(o.kind, MeshKind::HoverDrone | MeshKind::TurretTower))
            .count();
        let top_banner = Rect::from_center_size(
            Pos2::new(viewport_rect.center().x, viewport_rect.top() + 60.0),
            Vec2::new(470.0, 34.0),
        );
        painter.rect(
            top_banner,
            Rounding::same(7.0),
            Color32::from_rgba_unmultiplied(10, 18, 28, 230),
            Stroke::new(1.5, if hostiles_remaining == 0 { Color32::from_rgb(50, 240, 130) } else { Color32::from_rgb(55, 195, 255) }),
        );
        let status_txt = if hostiles_remaining == 0 {
            format!("★ SECTOR CLEARED ★   |   SCORE: {:06}   |   HOSTILES ELIMINATED: 4/4", state.score)
        } else {
            format!(
                "PLAY MODE ACTIVE   |   SCORE: {:06}   |   HOSTILES REMAINING: {}",
                state.score, hostiles_remaining
            )
        };
        painter.text(
            top_banner.center(),
            egui::Align2::CENTER_CENTER,
            status_txt,
            FontId::monospace(12.5),
            if hostiles_remaining == 0 {
                Color32::from_rgb(90, 255, 155)
            } else {
                Color32::from_rgb(220, 245, 255)
            },
        );

        let radar_r = 56.0;
        let radar_center = Pos2::new(viewport_rect.left() + radar_r + 18.0, viewport_rect.bottom() - radar_r - 18.0);
        painter.circle(
            radar_center,
            radar_r,
            Color32::from_rgba_unmultiplied(8, 16, 26, 225),
            Stroke::new(1.8, Color32::from_rgb(48, 165, 230)),
        );
        painter.circle_stroke(radar_center, radar_r * 0.66, Stroke::new(0.8, Color32::from_rgba_unmultiplied(48, 145, 200, 90)));
        painter.circle_stroke(radar_center, radar_r * 0.33, Stroke::new(0.8, Color32::from_rgba_unmultiplied(48, 145, 200, 90)));

        let sweep_a = time * 2.6;
        let sweep_end = Pos2::new(radar_center.x + sweep_a.cos() * radar_r, radar_center.y + sweep_a.sin() * radar_r);
        painter.line_segment(
            [radar_center, sweep_end],
            Stroke::new(1.6, Color32::from_rgba_unmultiplied(70, 245, 180, 180)),
        );

        let player_pos = state
            .objects
            .iter()
            .find(|o| o.kind == MeshKind::PlayerMech && o.alive)
            .map(|o| o.position)
            .unwrap_or(Vector3::new(0.0, 0.0, 0.0));

        for obj in &state.objects {
            if !obj.alive || matches!(obj.kind, MeshKind::ArenaFloor | MeshKind::WaterPool | MeshKind::SuspensionBridge | MeshKind::PlayerMech) {
                continue;
            }
            let rel = obj.position - player_pos;
            let bx = (rel.x / 18.0).clamp(-0.92, 0.92) * radar_r;
            let bz = (rel.z / 18.0).clamp(-0.92, 0.92) * radar_r;
            let blip_pos = Pos2::new(radar_center.x + bx, radar_center.y + bz);
            let (blip_col, blip_r) = match obj.kind {
                MeshKind::HoverDrone => (Color32::from_rgb(255, 68, 82), 4.2),
                MeshKind::TurretTower => (Color32::from_rgb(255, 145, 40), 4.8),
                MeshKind::EnergyCore | MeshKind::PlasmaOrb => (Color32::from_rgb(255, 225, 65), 3.6),
                MeshKind::JumpPad => (Color32::from_rgb(65, 250, 145), 3.8),
                _ => (Color32::from_rgb(90, 175, 255), 2.8),
            };
            painter.circle_filled(blip_pos, blip_r, blip_col);
        }
        painter.circle_filled(radar_center, 4.2, Color32::from_rgb(80, 240, 255));

        let hud_box = Rect::from_center_size(
            Pos2::new(viewport_rect.center().x, viewport_rect.bottom() - 34.0),
            Vec2::new(340.0, 44.0),
        );
        painter.rect(
            hud_box,
            Rounding::same(7.0),
            Color32::from_rgba_unmultiplied(10, 16, 26, 220),
            Stroke::new(1.2, Color32::from_rgb(44, 78, 118)),
        );

        let hp_bar_bg = Rect::from_min_size(Pos2::new(hud_box.left() + 14.0, hud_box.top() + 8.0), Vec2::new(200.0, 11.0));
        let hp_fill = Rect::from_min_size(hp_bar_bg.min, Vec2::new(200.0 * (state.player_hp / 100.0).clamp(0.0, 1.0), 11.0));
        painter.rect_filled(hp_bar_bg, Rounding::same(3.0), Color32::from_rgb(24, 34, 48));
        painter.rect_filled(hp_fill, Rounding::same(3.0), Color32::from_rgb(45, 225, 135));
        painter.text(
            Pos2::new(hp_bar_bg.right() + 10.0, hp_bar_bg.center().y),
            egui::Align2::LEFT_CENTER,
            format!("HULL {:3.0}%", state.player_hp),
            FontId::monospace(10.5),
            Color32::from_rgb(130, 245, 185),
        );

        let heat_bar_bg = Rect::from_min_size(Pos2::new(hud_box.left() + 14.0, hud_box.top() + 24.0), Vec2::new(200.0, 9.0));
        let heat_fill = Rect::from_min_size(heat_bar_bg.min, Vec2::new(200.0 * state.weapon_heat.clamp(0.0, 1.0), 9.0));
        painter.rect_filled(heat_bar_bg, Rounding::same(3.0), Color32::from_rgb(24, 34, 48));
        painter.rect_filled(heat_fill, Rounding::same(3.0), Color32::from_rgb(255, 145, 45));
        painter.text(
            Pos2::new(heat_bar_bg.right() + 10.0, heat_bar_bg.center().y),
            egui::Align2::LEFT_CENTER,
            format!("PLASMA {:2.0}%", state.weapon_heat * 100.0),
            FontId::monospace(10.5),
            Color32::from_rgb(255, 185, 95),
        );
    }

    // 4. YouTube Broadcast Chapter Lower-Third Overlay
    if state.show_youtube_overlay {
        let card_w = 560.0_f32.min(viewport_rect.width() - 32.0);
        let card_h = 56.0;
        let y_offset = if state.play_mode { 92.0 } else { 16.0 };
        let card_rect = Rect::from_min_size(
            Pos2::new(viewport_rect.left() + 16.0, viewport_rect.bottom() - card_h - y_offset),
            Vec2::new(card_w, card_h),
        );
        painter.rect(
            card_rect,
            Rounding::same(8.0),
            Color32::from_rgba_unmultiplied(9, 13, 22, 232),
            Stroke::new(1.4, Color32::from_rgb(48, 175, 245)),
        );
        let accent = Rect::from_min_size(card_rect.min, Vec2::new(5.0, card_h));
        painter.rect_filled(accent, Rounding::same(4.0), Color32::from_rgb(45, 215, 255));

        painter.text(
            Pos2::new(card_rect.left() + 16.0, card_rect.top() + 17.0),
            egui::Align2::LEFT_CENTER,
            &state.showcase_chapter_title,
            FontId::proportional(14.5),
            Color32::from_rgb(240, 250, 255),
        );
        painter.text(
            Pos2::new(card_rect.left() + 16.0, card_rect.top() + 37.0),
            egui::Align2::LEFT_CENTER,
            &state.showcase_chapter_subtitle,
            FontId::monospace(10.8),
            Color32::from_rgb(125, 205, 250),
        );

        let prog_bg = Rect::from_min_size(Pos2::new(card_rect.left() + 8.0, card_rect.bottom() - 4.0), Vec2::new(card_w - 16.0, 2.5));
        let prog_fg = Rect::from_min_size(prog_bg.min, Vec2::new((card_w - 16.0) * state.showcase_progress.clamp(0.0, 1.0), 2.5));
        painter.rect_filled(prog_bg, Rounding::same(1.0), Color32::from_rgb(28, 42, 62));
        painter.rect_filled(prog_fg, Rounding::same(1.0), Color32::from_rgb(55, 225, 255));
    }
}

fn draw_engine_editor_ui(ctx: &Context, state: &mut Metatopia3DEngineState, fps: f32, time: f32) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = Color32::from_rgba_unmultiplied(15, 19, 27, 245);
    visuals.panel_fill = Color32::from_rgba_unmultiplied(15, 19, 27, 242);
    ctx.set_visuals(visuals);

    // 1. TOP TOOLBAR
    egui::TopBottomPanel::top("top_editor_toolbar")
        .exact_height(44.0)
        .frame(egui::Frame::none().fill(Color32::from_rgb(13, 17, 24)).inner_margin(egui::Margin::symmetric(12.0, 6.0)))
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(
                    RichText::new("◆ METATOPIA 3D ENGINE")
                        .strong()
                        .size(15.0)
                        .color(Color32::from_rgb(75, 210, 255)),
                );
                ui.separator();

                ui.label(RichText::new("Gizmo:").size(12.0).color(Color32::GRAY));
                if ui.selectable_label(state.gizmo_mode == GizmoMode::Translate, "✢ Move (W)").clicked() {
                    state.gizmo_mode = GizmoMode::Translate;
                }
                if ui.selectable_label(state.gizmo_mode == GizmoMode::Rotate, "↻ Rotate (E)").clicked() {
                    state.gizmo_mode = GizmoMode::Rotate;
                }
                if ui.selectable_label(state.gizmo_mode == GizmoMode::Scale, "⤢ Scale (R)").clicked() {
                    state.gizmo_mode = GizmoMode::Scale;
                }

                ui.separator();
                ui.checkbox(&mut state.show_grid, "3D Grid");

                ui.separator();
                for (mode_id, label) in [(0, "Lit PBR"), (1, "Normals"), (3, "2048 PCF Map")] {
                    if ui.selectable_label(state.shader_debug_view == mode_id, label).clicked() {
                        state.shader_debug_view = mode_id;
                    }
                }

                ui.add_space(16.0);
                let play_btn = egui::Button::new(
                    RichText::new(if state.play_mode { "⏹ STOP (Restore Editor)" } else { "▶ PLAY 3D GAME" })
                        .strong()
                        .size(13.5)
                        .color(if state.play_mode {
                            Color32::from_rgb(255, 115, 115)
                        } else {
                            Color32::from_rgb(95, 250, 155)
                        }),
                )
                .min_size(Vec2::new(155.0, 28.0));

                if ui.add(play_btn).clicked() {
                    if state.play_mode {
                        state.stop_play_mode();
                    } else {
                        state.start_play_mode();
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("Apple Metal • {:5.1} FPS • 2048 PCF Shadows", fps))
                            .monospace()
                            .size(11.5)
                            .color(Color32::from_rgb(125, 225, 170)),
                    );
                });
            });
        });

    // 2. BOTTOM DOCK
    egui::TopBottomPanel::bottom("bottom_editor_dock")
        .exact_height(150.0)
        .frame(egui::Frame::none().fill(Color32::from_rgb(13, 17, 24)).inner_margin(egui::Margin::symmetric(12.0, 8.0)))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(state.bottom_tab == BottomTab::AssetBrowser, "📦 3D Prefab & Asset Browser").clicked() {
                    state.bottom_tab = BottomTab::AssetBrowser;
                }
                if ui.selectable_label(state.bottom_tab == BottomTab::ConsoleLogs, "🖥 Engine Telemetry Console").clicked() {
                    state.bottom_tab = BottomTab::ConsoleLogs;
                }
                if ui.selectable_label(state.bottom_tab == BottomTab::RhaiScriptEditor, "📜 Live Rhai Script IDE").clicked() {
                    state.bottom_tab = BottomTab::RhaiScriptEditor;
                }
                ui.separator();
                ui.label(
                    RichText::new("LMB: Pick & Drag 3D RGB Gizmo  |  RMB Drag: Orbit 3D Camera  |  Play Mode: WASD + Space + LMB/F Fire")
                        .size(11.0)
                        .color(Color32::from_rgb(140, 165, 200)),
                );
            });
            ui.separator();

            match state.bottom_tab {
                BottomTab::AssetBrowser => {
                    ui.horizontal_wrapped(|ui| {
                        let prefabs = [
                            ("📦 Rigid-Body Crate", MeshKind::PhysicsCrate, Color32::from_rgb(245, 145, 55)),
                            ("🛸 Hover-Drone AI", MeshKind::HoverDrone, Color32::from_rgb(245, 75, 90)),
                            ("🗼 Defense Turret", MeshKind::TurretTower, Color32::from_rgb(255, 130, 60)),
                            ("⛩ Portal Archway", MeshKind::SciFiArchway, Color32::from_rgb(135, 185, 245)),
                            ("⚛ Neon Reactor Core", MeshKind::EnergyCore, Color32::from_rgb(60, 225, 255)),
                            ("🏛 Titanium Pillar", MeshKind::CyberPillar, Color32::from_rgb(155, 175, 210)),
                            ("⚡ Graviton Jump Pad", MeshKind::JumpPad, Color32::from_rgb(65, 245, 145)),
                            ("🧱 Tactical Step Deck", MeshKind::StepPlatform, Color32::from_rgb(130, 165, 220)),
                            ("✨ Plasma Orb Pickup", MeshKind::PlasmaOrb, Color32::from_rgb(255, 215, 65)),
                        ];
                        for (title, kind, tint) in prefabs {
                            let btn = egui::Button::new(RichText::new(title).strong().size(12.2).color(tint))
                                .min_size(Vec2::new(150.0, 68.0));
                            if ui.add(btn).clicked() {
                                state.spawn_prefab(kind);
                            }
                        }
                    });
                }
                BottomTab::ConsoleLogs => {
                    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                        for line in &state.console_logs {
                            ui.label(RichText::new(line).monospace().size(11.5).color(Color32::from_rgb(185, 215, 245)));
                        }
                    });
                }
                BottomTab::RhaiScriptEditor => {
                    ui.horizontal(|ui| {
                        if ui.button("⚡ Compile & Hot-Reload Rhai Script").clicked() {
                            match state.script_engine.compile(&state.script_source) {
                                Ok(ast) => {
                                    state.script_ast = Some(ast);
                                    state.log("[RHAI] Compiled & hot-reloaded live Rhai script AST successfully.");
                                }
                                Err(err) => {
                                    state.log(format!("[RHAI ERROR] {}", err));
                                }
                            }
                        }
                        ui.label(
                            RichText::new(format!(
                                "Live Emissive Boost: {:.2}x",
                                state.script_entity.emissive
                            ))
                            .monospace()
                            .size(11.5)
                            .color(Color32::from_rgb(95, 235, 185)),
                        );
                    });
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut state.script_source)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(4),
                        );
                    });
                }
            }
        });

    // 3. LEFT PANEL: 3D SCENE OUTLINER
    egui::SidePanel::left("left_scene_outliner")
        .exact_width(245.0)
        .frame(egui::Frame::none().fill(Color32::from_rgba_unmultiplied(15, 20, 29, 238)).inner_margin(egui::Margin::same(10.0)))
        .show(ctx, |ui| {
            ui.label(RichText::new("SCENE OUTLINER (3D)").strong().size(13.0).color(Color32::from_rgb(120, 195, 255)));
            ui.label(RichText::new("Cyber_Arena_Prime.scn").monospace().size(11.0).color(Color32::GRAY));
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut clicked_id = None;
                for obj in &state.objects {
                    if !obj.alive {
                        continue;
                    }
                    let icon = match obj.kind {
                        MeshKind::ArenaFloor => "🌐",
                        MeshKind::WaterPool => "🌊",
                        MeshKind::SuspensionBridge => "🌉",
                        MeshKind::SciFiArchway => "⛩",
                        MeshKind::TurretTower => "🗼",
                        MeshKind::PlayerMech => "🤖",
                        MeshKind::EnergyCore => "⚛",
                        MeshKind::PhysicsCrate => "📦",
                        MeshKind::HoverDrone => "🛸",
                        MeshKind::CyberPillar => "🏛",
                        MeshKind::PlasmaOrb => "✨",
                        MeshKind::StepPlatform => "🧱",
                        MeshKind::JumpPad => "⚡",
                    };
                    let selected = state.selected_id == Some(obj.id);
                    let label = format!("{}  {}", icon, obj.name);
                    if ui.selectable_label(selected, RichText::new(label).size(12.0)).clicked() {
                        clicked_id = Some(obj.id);
                    }
                }
                if let Some(id) = clicked_id {
                    state.selected_id = Some(id);
                }
            });
        });

    // 4. RIGHT PANEL: 3D INSPECTOR
    egui::SidePanel::right("right_inspector_panel")
        .exact_width(290.0)
        .frame(egui::Frame::none().fill(Color32::from_rgba_unmultiplied(15, 20, 29, 238)).inner_margin(egui::Margin::same(10.0)))
        .show(ctx, |ui| {
            ui.label(RichText::new("INSPECTOR").strong().size(13.0).color(Color32::from_rgb(120, 195, 255)));
            ui.separator();

            if let Some(sel_id) = state.selected_id {
                if let Some(obj) = state.objects.iter_mut().find(|o| o.id == sel_id && o.alive) {
                    ui.label(RichText::new(&obj.name).strong().size(14.0).color(Color32::WHITE));
                    ui.label(RichText::new(obj.kind.label()).size(11.5).color(Color32::from_rgb(110, 200, 255)));
                    ui.add_space(6.0);

                    ui.collapsing("Transform (3D World Space)", |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Pos X").color(Color32::from_rgb(255, 95, 95)));
                            ui.add(egui::DragValue::new(&mut obj.position.x).speed(0.1));
                            ui.label(RichText::new("Y").color(Color32::from_rgb(95, 255, 115)));
                            ui.add(egui::DragValue::new(&mut obj.position.y).speed(0.1));
                            ui.label(RichText::new("Z").color(Color32::from_rgb(95, 155, 255)));
                            ui.add(egui::DragValue::new(&mut obj.position.z).speed(0.1));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Rot Y°");
                            ui.add(egui::DragValue::new(&mut obj.rotation_deg.y).speed(1.5));
                            ui.label("Pitch X°");
                            ui.add(egui::DragValue::new(&mut obj.rotation_deg.x).speed(1.5));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Scale");
                            ui.add(egui::DragValue::new(&mut obj.scale.x).speed(0.05).clamp_range(0.2..=5.0));
                            ui.add(egui::DragValue::new(&mut obj.scale.y).speed(0.05).clamp_range(0.2..=5.0));
                            ui.add(egui::DragValue::new(&mut obj.scale.z).speed(0.05).clamp_range(0.2..=5.0));
                        });
                    });

                    ui.collapsing("PBR Surface Material", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Albedo RGB");
                            ui.color_edit_button_rgb(&mut obj.color);
                        });
                        ui.add(egui::Slider::new(&mut obj.metallic, 0.0..=1.0).text("Metallic"));
                        ui.add(egui::Slider::new(&mut obj.roughness, 0.04..=1.0).text("Roughness"));
                        ui.add(egui::Slider::new(&mut obj.emissive, 0.0..=3.0).text("Emissive Glow"));
                    });

                    ui.collapsing("Rigid-Body Physics & AI", |ui| {
                        ui.checkbox(&mut obj.has_gravity, "Use Gravity (9.81 m/s²)");
                        ui.checkbox(&mut obj.is_static, "Static Collider");
                        ui.label(format!(
                            "Velocity: ({:.1}, {:.1}, {:.1})",
                            obj.velocity.x, obj.velocity.y, obj.velocity.z
                        ));
                        ui.add(egui::ProgressBar::new((obj.hp / obj.max_hp.max(1.0)).clamp(0.0, 1.0)).text(format!("HP {:.0}/{:.0}", obj.hp, obj.max_hp)));
                    });
                }
            }

            ui.separator();
            ui.collapsing("Directional Sun & 2048 Shadow Map", |ui| {
                ui.add(egui::Slider::new(&mut state.sun_azimuth_deg, -180.0..=180.0).text("Sun Azimuth°"));
                ui.add(egui::Slider::new(&mut state.sun_elevation_deg, 12.0..=85.0).text("Sun Elevation°"));
                ui.add(egui::Slider::new(&mut state.sun_intensity, 0.5..=6.0).text("Sun Intensity"));
                ui.add(egui::Slider::new(&mut state.cam_fov_deg, 35.0..=85.0).text("Camera FOV°"));
            });
        });

    // 5. COMPUTE EXACT UNOBSCURED 3D VIEWPORT RECT & DRAW PAINTER OVERLAYS
    let viewport_rect = ctx.available_rect();
    if viewport_rect.width() > 64.0 && viewport_rect.height() > 64.0 {
        state.viewport_aspect = viewport_rect.width() / viewport_rect.height();
        draw_3d_viewport_overlays(ctx, state, viewport_rect, time);
    }
}

// ============================================================================
// 8. LIVE INTERACTIVE DESKTOP 3D ENGINE & EDITOR RUNNER (WINIT 0.29)
// ============================================================================

fn run_interactive_engine() {
    let event_loop = EventLoop::new().expect("create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Metatopia 3D Game Engine & Editor — Apple Metal / wgpu (2048 PCF Shadows)")
            .with_inner_size(LogicalSize::new(1440.0, 860.0))
            .build(&event_loop)
            .expect("create window"),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).expect("create surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("request adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Metatopia 3D Engine Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("request device");

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

    let mut pipeline = Engine3DGpuPipeline::new(&device, format, config.width, config.height);
    let egui_ctx = Context::default();
    let mut egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
    );
    let mut egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);
    let mut state = Metatopia3DEngineState::new();

    let start_time = Instant::now();
    let mut last_frame = start_time;
    let mut fps_smooth = 60.0_f32;
    let mut cursor_pos = (640.0_f32, 360.0_f32);
    let mut rmb_down = false;
    let mut lmb_down = false;
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut held_keys = [false; 10];

    let _ = event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event: ref we, .. } => {
                let resp = egui_state.on_window_event(&window, we);
                if resp.repaint {
                    window.request_redraw();
                }
                if resp.consumed && !matches!(we, WindowEvent::RedrawRequested | WindowEvent::Resized(_)) {
                    return;
                }

                match we {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                    }
                    WindowEvent::KeyboardInput { event: ke, .. } => {
                        let pressed = ke.state == ElementState::Pressed;
                        if let PhysicalKey::Code(code) = ke.physical_key {
                            match code {
                                KeyCode::KeyW | KeyCode::ArrowUp => {
                                    held_keys[0] = pressed;
                                    if pressed && !state.play_mode && !rmb_down {
                                        state.gizmo_mode = GizmoMode::Translate;
                                    }
                                }
                                KeyCode::KeyS | KeyCode::ArrowDown => held_keys[1] = pressed,
                                KeyCode::KeyA | KeyCode::ArrowLeft => held_keys[2] = pressed,
                                KeyCode::KeyD | KeyCode::ArrowRight => held_keys[3] = pressed,
                                KeyCode::Space => held_keys[4] = pressed,
                                KeyCode::KeyF => held_keys[5] = pressed,
                                KeyCode::KeyE if pressed && !state.play_mode => {
                                    state.gizmo_mode = GizmoMode::Rotate;
                                }
                                KeyCode::KeyR if pressed && !state.play_mode => {
                                    state.gizmo_mode = GizmoMode::Scale;
                                }
                                KeyCode::KeyP if pressed => {
                                    if state.play_mode {
                                        state.stop_play_mode();
                                    } else {
                                        state.start_play_mode();
                                    }
                                }
                                KeyCode::Escape if pressed => {
                                    if state.play_mode {
                                        state.stop_play_mode();
                                    } else {
                                        target.exit();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(_, y) => *y,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                        };
                        state.cam_distance = (state.cam_distance - scroll * 0.9).clamp(3.0, 48.0);
                    }
                    WindowEvent::MouseInput { state: btn_state, button, .. } => {
                        let pressed = *btn_state == ElementState::Pressed;
                        match button {
                            MouseButton::Right => {
                                rmb_down = pressed;
                                last_mouse = None;
                            }
                            MouseButton::Left => {
                                lmb_down = pressed;
                                held_keys[5] = pressed && state.play_mode;
                                if !pressed {
                                    state.active_gizmo_axis = GizmoAxis::None;
                                    last_mouse = None;
                                } else if !state.play_mode {
                                    let screen = (config.width as f32, config.height as f32);
                                    let (_, inv_vp) = state.view_proj_matrices(screen.0 / screen.1.max(1.0));
                                    let (ro, rd) = screen_to_world_ray(cursor_pos, screen, inv_vp);

                                    let mut gizmo_hit = GizmoAxis::None;
                                    if let Some(sel_id) = state.selected_id {
                                        if let Some(sel) = state.objects.iter().find(|o| o.id == sel_id && o.alive) {
                                            let cam_dist = (state.camera_eye() - sel.position).magnitude();
                                            gizmo_hit = pick_gizmo_axis(ro, rd, sel.position, cam_dist);
                                        }
                                    }

                                    if gizmo_hit != GizmoAxis::None {
                                        state.active_gizmo_axis = gizmo_hit;
                                    } else {
                                        let mut best: Option<(u32, f32)> = None;
                                        for obj in &state.objects {
                                            if !obj.alive || obj.kind == MeshKind::ArenaFloor {
                                                continue;
                                            }
                                            let (bmin, bmax) = obj.world_aabb();
                                            if let Some(t) = ray_aabb_intersect(ro, rd, bmin, bmax) {
                                                if best.map_or(true, |(_, bt)| t < bt) {
                                                    best = Some((obj.id, t));
                                                }
                                            }
                                        }
                                        if let Some((id, _)) = best {
                                            state.selected_id = Some(id);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let cur = (position.x as f32, position.y as f32);
                        if let Some(prev) = last_mouse {
                            let dx = cur.0 - prev.0;
                            let dy = cur.1 - prev.1;
                            if rmb_down {
                                state.cam_yaw -= dx * 0.0065;
                                state.cam_pitch = (state.cam_pitch + dy * 0.0065).clamp(0.08, 1.35);
                            } else if lmb_down && !state.play_mode && state.active_gizmo_axis != GizmoAxis::None {
                                if let Some(sel_id) = state.selected_id {
                                    if let Some(obj) = state.objects.iter_mut().find(|o| o.id == sel_id && o.alive) {
                                        let delta = (dx - dy) * 0.032;
                                        match (state.gizmo_mode, state.active_gizmo_axis) {
                                            (GizmoMode::Translate, GizmoAxis::X) => obj.position.x += dx * 0.035,
                                            (GizmoMode::Translate, GizmoAxis::Y) => obj.position.y = (obj.position.y - dy * 0.035).max(0.2),
                                            (GizmoMode::Translate, GizmoAxis::Z) => obj.position.z += dx * 0.035,
                                            (GizmoMode::Rotate, _) => obj.rotation_deg.y += dx * 0.8,
                                            (GizmoMode::Scale, GizmoAxis::X) => obj.scale.x = (obj.scale.x + delta).clamp(0.3, 4.0),
                                            (GizmoMode::Scale, GizmoAxis::Y) => obj.scale.y = (obj.scale.y + delta).clamp(0.3, 4.0),
                                            (GizmoMode::Scale, GizmoAxis::Z) => obj.scale.z = (obj.scale.z + delta).clamp(0.3, 4.0),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        cursor_pos = cur;
                        last_mouse = Some(cur);
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - last_frame).as_secs_f32().clamp(0.001, 0.05);
                        last_frame = now;
                        fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1;
                        let elapsed = (now - start_time).as_secs_f32();

                        if rmb_down && !state.play_mode {
                            let fwd = Vector3::new(-state.cam_yaw.sin(), 0.0, -state.cam_yaw.cos()).normalize();
                            let right = Vector3::new(state.cam_yaw.cos(), 0.0, -state.cam_yaw.sin()).normalize();
                            if held_keys[0] { state.cam_target += fwd * 9.0 * dt; }
                            if held_keys[1] { state.cam_target -= fwd * 9.0 * dt; }
                            if held_keys[2] { state.cam_target -= right * 9.0 * dt; }
                            if held_keys[3] { state.cam_target += right * 9.0 * dt; }
                        }

                        state.step_physics_and_gameplay(dt, elapsed, &held_keys);

                        let frame = match surface.get_current_texture() {
                            Ok(f) => f,
                            Err(_) => return,
                        };
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                        let raw_input = egui_state.take_egui_input(&window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            draw_engine_editor_ui(ctx, &mut state, fps_smooth, elapsed);
                        });
                        egui_state.handle_platform_output(&window, full_output.platform_output);

                        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [config.width, config.height],
                            pixels_per_point: full_output.pixels_per_point,
                        };

                        for (id, delta) in &full_output.textures_delta.set {
                            egui_renderer.update_texture(&device, &queue, *id, delta);
                        }

                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("3D Frame Command Encoder"),
                        });
                        egui_renderer.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen_descriptor);

                        let uniform = state.build_uniform(config.width as f32, config.height as f32, elapsed);
                        let verts = state.build_vertex_stream(elapsed);
                        pipeline.render_frame(
                            &device,
                            &queue,
                            &mut encoder,
                            &view,
                            &uniform,
                            &verts,
                            &mut egui_renderer,
                            &paint_jobs,
                            &screen_descriptor,
                        );

                        for id in &full_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }

                        queue.submit(std::iter::once(encoder.finish()));
                        frame.present();
                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

// ============================================================================
// 9. HEADLESS 60 FPS YOUTUBE SHOWCASE RECORDER & PROCEDURAL SOUNDTRACK
// ============================================================================

fn generate_3d_engine_soundtrack_wav(wav_path: &Path, duration_secs: f32) -> Result<(), String> {
    let sample_rate = 44_100u32;
    let total_samples = (duration_secs * sample_rate as f32) as usize;
    let mut pcm: Vec<i16> = Vec::with_capacity(total_samples * 2);

    let bass_freqs = [55.0_f32, 65.41, 73.42, 82.41];
    let arp_freqs = [220.0_f32, 261.63, 329.63, 392.0, 440.0, 523.25, 659.25, 523.25];

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        let in_play = t >= 7.8;

        let bar = ((t * 0.65) as usize) % bass_freqs.len();
        let bf = bass_freqs[bar];
        let sub = (t * bf * TAU).sin() * 0.24;
        let saw = ((t * bf * 2.0).fract() * 2.0 - 1.0) * (if in_play { 0.14 } else { 0.08 });

        let step = ((t * 6.0) as usize) % arp_freqs.len();
        let af = arp_freqs[step];
        let env = (1.0 - (t * 6.0).fract()).powf(2.1);
        let arp = (t * af * TAU).sin() * env * (if in_play { 0.18 } else { 0.11 });

        let mut sfx = 0.0_f32;
        if (t - 7.8).abs() < 0.45 {
            let u = 1.0 - (t - 7.8).abs() / 0.45;
            sfx += (t * (240.0 + u * 640.0) * TAU).sin() * u * 0.26;
        }

        if in_play && t < 18.8 {
            let fire_phase = (t * 4.5).fract();
            if fire_phase < 0.22 {
                let chirp = 920.0 - fire_phase * 2800.0;
                sfx += (t * chirp.max(140.0) * TAU).sin() * (0.22 - fire_phase) * 0.72;
            }
            let beat = (t * (128.0 / 60.0)).fract();
            if beat < 0.18 {
                let k_freq = 115.0 * (-beat * 18.0).exp() + 42.0;
                sfx += (t * k_freq * TAU).sin() * (1.0 - beat / 0.18) * 0.24;
            }
        }

        let fade_in = (t / 0.5).clamp(0.0, 1.0);
        let fade_out = ((duration_secs - t) / 0.8).clamp(0.0, 1.0);
        let master = (sub + saw + arp + sfx) * fade_in * fade_out;

        let pan = (t * 0.85).sin() * 0.14;
        let left = ((master * (1.0 - pan)).clamp(-0.95, 0.95) * 32767.0) as i16;
        let right = ((master * (1.0 + pan)).clamp(-0.95, 0.95) * 32767.0) as i16;
        pcm.push(left);
        pcm.push(right);
    }

    let data_bytes = (pcm.len() * 2) as u32;
    let mut f = std::fs::File::create(wav_path).map_err(|e| e.to_string())?;
    f.write_all(b"RIFF").map_err(|e| e.to_string())?;
    f.write_all(&(36 + data_bytes).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(b"WAVEfmt ").map_err(|e| e.to_string())?;
    f.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&2u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&sample_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(sample_rate * 4).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&4u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&16u16.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(b"data").map_err(|e| e.to_string())?;
    f.write_all(&data_bytes.to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(bytemuck::cast_slice(&pcm)).map_err(|e| e.to_string())?;
    Ok(())
}

fn record_3d_engine_showcase_video(output_mp4: &Path) -> Result<(), String> {
    let width = 1920u32;
    let height = 1080u32;
    let fps = 60u32;
    let duration_secs = 20.0_f32;
    let total_frames = (fps as f32 * duration_secs) as u32;

    println!(
        "🎬 Recording 20s 60fps YouTube Showcase Video ({}x{}, {} frames) -> {}",
        width,
        height,
        total_frames,
        output_mp4.display()
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("Failed to find GPU adapter")?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Headless 1080p Recorder Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .map_err(|e| format!("device error: {e}"))?;

    let color_format = wgpu::TextureFormat::Rgba8Unorm;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Offscreen 1080p Render Target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = width * 4;
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("1080p Readback Buffer"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut pipeline = Engine3DGpuPipeline::new(&device, color_format, width, height);
    let egui_ctx = Context::default();
    let mut egui_renderer = egui_wgpu::Renderer::new(&device, color_format, None, 1);
    let mut state = Metatopia3DEngineState::new();
    state.show_youtube_overlay = true;

    let wav_path = output_mp4.with_extension("wav");
    generate_3d_engine_soundtrack_wav(&wav_path, duration_secs)?;

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", "1920x1080",
            "-r", "60",
            "-i", "-",
            "-i", wav_path.to_str().unwrap_or("audio.wav"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_mp4.to_str().unwrap_or("out.mp4"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {e}"))?;

    let mut stdin = ffmpeg.stdin.take().ok_or("Failed to open ffmpeg stdin")?;
    let dt = 1.0 / (fps as f32);

    for frame_idx in 0..total_frames {
        let t = frame_idx as f32 * dt;
        let mut held_keys = [false; 10];
        state.showcase_progress = (t / duration_secs).clamp(0.0, 1.0);

        if t < 4.2 {
            // CHAPTER 1 (0.0s - 4.2s): 3D Level Editor, 2048 PCF Shadow Map & RGB Translate Gizmo
            state.showcase_chapter_title = "01 // 3D LEVEL EDITOR & 2048x2048 PCF SHADOW MAPPING".into();
            state.showcase_chapter_subtitle = "Apple Metal wgpu  •  9-Tap Hardware Bilinear PCF Shadows  •  RGB Transform Gizmos".into();
            state.cam_yaw = 0.42 - t * 0.16;
            state.cam_pitch = 0.40 + (t * 0.7).sin() * 0.04;
            state.cam_distance = 15.6 - t * 0.45;
            state.gizmo_mode = GizmoMode::Translate;
            state.selected_id = Some(9); // Plasma_Crate_Alpha
            state.active_gizmo_axis = if t > 1.2 { GizmoAxis::Y } else { GizmoAxis::None };
            state.sun_azimuth_deg = 38.0 + t * 8.0;

            if t > 1.2 {
                let u = ((t - 1.2) / 2.8).clamp(0.0, 1.0);
                if let Some(crate_obj) = state.objects.iter_mut().find(|o| o.id == 9) {
                    crate_obj.position.y = 0.85 + u * 4.35;
                }
            }
        } else if t < 7.8 {
            // CHAPTER 2 (4.2s - 7.8s): Shader Debug Views (Normals / 2048 Shadow Map), Rotate Gizmo & Spawner
            state.showcase_chapter_title = "02 // PBR SHADER INSPECTOR, GIZMO ROTATION & PREFAB SPAWNER".into();
            state.showcase_chapter_subtitle = "Live Shader Views (Lit PBR / Normals / 2048 Shadow Map)  •  Rhai Scripting IDE".into();
            state.cam_yaw = -0.25 + (t - 4.2) * 0.19;
            state.cam_pitch = 0.36;
            state.cam_distance = 13.8;
            state.selected_id = Some(6); // Reactor_Core_Prime
            state.gizmo_mode = GizmoMode::Rotate;
            state.active_gizmo_axis = GizmoAxis::Y;

            if (4.6..5.4).contains(&t) {
                state.shader_debug_view = 1;
            } else if (5.4..6.2).contains(&t) {
                state.shader_debug_view = 3;
            } else {
                state.shader_debug_view = 0;
            }

            if frame_idx == 330 {
                state.spawn_prefab(MeshKind::HoverDrone);
                state.bottom_tab = BottomTab::RhaiScriptEditor;
            }
            if frame_idx == 410 {
                state.bottom_tab = BottomTab::ConsoleLogs;
            }
        } else if t < 15.0 {
            // CHAPTER 3 (7.8s - 15.0s): Press PLAY! Rigid-Body Physics Drop + Mech vs. Turret/Drone Combat
            if frame_idx == 468 {
                state.start_play_mode();
            }
            state.showcase_chapter_title = "03 // PLAY MODE: RIGID-BODY PHYSICS & MECH VS. TURRET COMBAT".into();
            state.showcase_chapter_subtitle = "Gravity Drop & Elastic Collisions  •  Autonomous Turret & Drone AI  •  Tactical Radar HUD".into();
            state.cam_yaw = -0.35 + (t - 7.8) * 0.12;
            state.cam_pitch = 0.36;
            state.cam_distance = 13.2;

            // Keep Vanguard Mech patrolling right across the Suspension Bridge over the Water Basin!
            if let Some(player_idx) = state.objects.iter().position(|o| o.kind == MeshKind::PlayerMech && o.alive) {
                let u = t - 7.8;
                let target_x = (u * 1.15).sin() * 3.4;
                let target_z = (2.8 - u * 0.95).max(-1.8);
                let cur = state.objects[player_idx].position;
                state.objects[player_idx].velocity.x = (target_x - cur.x) * 4.5;
                state.objects[player_idx].velocity.z = (target_z - cur.z) * 4.5;
            }
            held_keys[5] = true;
        } else if t < 18.6 {
            // CHAPTER 4 (15.0s - 18.6s): Graviton Jump Pad Launch, Aerial Combat & Sector Cleared!
            state.showcase_chapter_title = "04 // GRAVITON JUMP PAD LAUNCH & SECTOR CLEARED".into();
            state.showcase_chapter_subtitle = "Vertical Jump-Jet Thrusters  •  3D Particle Explosions  •  100% Native Rust + WGPU".into();
            state.cam_yaw = 0.52 - (t - 15.0) * 0.18;
            state.cam_pitch = 0.38;
            state.cam_distance = 13.6;

            if frame_idx == 902 {
                if let Some(player_idx) = state.objects.iter().position(|o| o.kind == MeshKind::PlayerMech && o.alive) {
                    state.objects[player_idx].position = Vector3::new(-3.2, 1.05, 2.8);
                    state.objects[player_idx].velocity = Vector3::new(1.8, 12.6, -2.6);
                    let p = state.objects[player_idx].position;
                    state.spawn_spark_burst(p, [0.15, 1.0, 0.58], 22);
                }
            }
            held_keys[5] = true;
        } else {
            // OUTRO (18.6s - 20.0s): Return to Editor Mode Hero Shot
            if frame_idx == 1116 {
                state.stop_play_mode();
                state.bottom_tab = BottomTab::AssetBrowser;
            }
            state.showcase_chapter_title = "METATOPIA 3D ENGINE — OPEN SOURCE RUST + WGPU + APPLE METAL".into();
            state.showcase_chapter_subtitle = "Subscribe to Metatopia on YouTube  •  github.com/metatopia/metatopia-game-engine".into();
            state.cam_yaw = 0.32 + (t - 18.8) * 0.12;
            state.cam_pitch = 0.42;
            state.cam_distance = 15.2;
        }

        state.step_physics_and_gameplay(dt, t, &held_keys);

        let mut raw_input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(width as f32, height as f32))),
            ..Default::default()
        };
        raw_input.time = Some(t as f64);

        let full_output = egui_ctx.run(raw_input, |ctx| {
            draw_engine_editor_ui(ctx, &mut state, 60.0, t);
        });
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, 1.0);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.0,
        };

        for (id, delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(&device, &queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Video Frame Encoder"),
        });
        egui_renderer.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen_descriptor);

        let uniform = state.build_uniform(width as f32, height as f32, t);
        let verts = state.build_vertex_stream(t);
        pipeline.render_frame(
            &device,
            &queue,
            &mut encoder,
            &target_view,
            &uniform,
            &verts,
            &mut egui_renderer,
            &paint_jobs,
            &screen_descriptor,
        );

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target_tex,
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
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("map_async error: {e}"))?;

        {
            let mapped = slice.get_mapped_range();
            stdin.write_all(&mapped).map_err(|e| format!("ffmpeg write error: {e}"))?;
        }
        readback_buf.unmap();

        if (frame_idx + 1) % 180 == 0 || frame_idx + 1 == total_frames {
            println!("  • Encoded frame {}/{} ({:.1}s)", frame_idx + 1, total_frames, t + dt);
        }
    }

    drop(stdin);
    let status = ffmpeg.wait().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&wav_path);
    if !status.success() {
        return Err("ffmpeg exited with non-zero status".into());
    }
    println!("✅ Saved YouTube 3D Engine Showcase Video -> {}", output_mp4.display());
    Ok(())
}

fn run_headless_verification() -> Result<(), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("Failed to find GPU adapter")?;

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Verification Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .map_err(|e| format!("device error: {e}"))?;

    let mut pipeline = Engine3DGpuPipeline::new(&device, wgpu::TextureFormat::Rgba8Unorm, 1280, 720);
    let state = Metatopia3DEngineState::new();
    let verts = state.build_vertex_stream(1.0);
    let uniform = state.build_uniform(1280.0, 720.0, 1.0);
    queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    pipeline.ensure_depth_size(&device, 1280, 720);

    println!("============================================================");
    println!("✅ METATOPIA 3D GAME ENGINE & EDITOR — VERIFICATION PASSED");
    println!("   GPU Adapter      : {} ({:?})", info.name, info.backend);
    println!("   Shadow Map       : {}x{} Depth32Float + 9-Tap Bilinear PCF", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    println!("   3D Entities      : {}", state.objects.len());
    println!("   Generated Verts  : {} / {}", verts.len(), MAX_VERTS);
    println!("============================================================");
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--verify") {
        if let Err(e) = run_headless_verification() {
            eprintln!("Verification failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--record-video") {
        let out_path = args
            .get(pos + 1)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("dist/macos/metatopia_macos_engine_demo.mp4"));
        if let Err(e) = record_3d_engine_showcase_video(&out_path) {
            eprintln!("Video recording failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    run_interactive_engine();
}
