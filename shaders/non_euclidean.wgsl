// ─── Physically-Based Non-Euclidean Shader ─────────────────────────────────
// Cook-Torrance BRDF · Geometry-dependent vertex warping · Interactions

const PI: f32 = 3.14159265359;

// ─── Uniforms ──────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,  // xyz = position, w = space_type
}

struct SceneUniform {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
    params: vec4<f32>,             // x=time, y=exposure, z=ambient, w=num_lights
    sphere_pos: vec4<f32>,         // xyz = position, w = glow
    orb0_pos: vec4<f32>,
    orb1_pos: vec4<f32>,
    orb2_pos: vec4<f32>,
    orb3_pos: vec4<f32>,
    interaction: vec4<f32>,        // x=collected, y=total, z=sphere_dist, w=unused
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> scene: SceneUniform;

// ─── Vertex I/O ────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) pbr_params: vec4<f32>,  // x=metallic, y=roughness, z=ao, w=emissive
}

// ─── Non-Euclidean Geometry Warping ────────────────────────────────────────

fn apply_curvature(p: vec3<f32>, space_type: u32) -> vec3<f32> {
    if (space_type == 0u) { return p; }

    let r = length(p.xz);
    var out = p;

    if (space_type == 1u) {
        // ── Hyperbolic (K < 0): saddle geometry ───────────────────
        // Floor curves downward at edges, space expands outward
        out.y -= r * r * 0.005;
        // Radial expansion: far objects stretch outward
        if (r > 0.1) {
            let expand = 1.0 + r * 0.012;
            out.x = p.x * expand;
            out.z = p.z * expand;
        }
    } else {
        // ── Spherical (K > 0): dome geometry ──────────────────────
        // Floor curves upward at edges, space contracts inward
        out.y += r * r * 0.008;
        // Radial contraction: far objects bend inward
        if (r > 0.1) {
            let contract = 1.0 / (1.0 + r * 0.009);
            out.x = p.x * contract;
            out.z = p.z * contract;
        }
    }
    return out;
}

// Warp normal to approximate the curvature tangent plane
fn warp_normal(n: vec3<f32>, p: vec3<f32>, space_type: u32) -> vec3<f32> {
    if (space_type == 0u) { return n; }

    var out = n;
    let r = length(p.xz);

    if (space_type == 1u) {
        // Hyperbolic: floor normal tilts outward at edges
        if (n.y > 0.5 && r > 1.0) {
            let tilt = r * 0.01;
            out = normalize(vec3<f32>(n.x + p.x * tilt * 0.1, n.y, n.z + p.z * tilt * 0.1));
        }
    } else {
        // Spherical: floor normal tilts inward at edges
        if (n.y > 0.5 && r > 1.0) {
            let tilt = r * 0.016;
            out = normalize(vec3<f32>(n.x - p.x * tilt * 0.1, n.y, n.z - p.z * tilt * 0.1));
        }
    }
    return out;
}

// ─── Vertex Shader ─────────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let room_size = 10.0;
    var position  = vec3<f32>(0.0);
    var normal    = vec3<f32>(0.0, 1.0, 0.0);
    var base_color = vec3<f32>(0.5);
    var metallic  = 0.04;
    var roughness = 0.5;
    var ao        = 1.0;
    var emissive  = 0.0;

    let space_type = u32(camera.view_position.w);
    let time = scene.params.x;

    // ── Shared geometry data ───────────────────────────────────────
    var qv = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
    );
    var oct = array<vec3<f32>, 6>(
        vec3<f32>( 0.0,  1.0,  0.0), vec3<f32>( 0.0, -1.0,  0.0),
        vec3<f32>( 1.0,  0.0,  0.0), vec3<f32>(-1.0,  0.0,  0.0),
        vec3<f32>( 0.0,  0.0,  1.0), vec3<f32>( 0.0,  0.0, -1.0)
    );
    var fi = array<array<u32, 3>, 8>(
        array<u32, 3>(0u,2u,4u), array<u32, 3>(0u,4u,3u),
        array<u32, 3>(0u,3u,5u), array<u32, 3>(0u,5u,2u),
        array<u32, 3>(1u,4u,2u), array<u32, 3>(1u,3u,4u),
        array<u32, 3>(1u,5u,3u), array<u32, 3>(1u,2u,5u)
    );

    // ────────────────────────────────────────────────────────────────
    //   0..35   Room  │  36..131  Sphere  │  132..227  Orbs (×4)
    // ────────────────────────────────────────────────────────────────

    if (idx < 36u) {
        // ── Room ──────────────────────────────────────────────────
        let face = idx / 6u;
        let v = qv[idx % 6u];

        if (face == 0u) {
            position  = vec3<f32>(v.x * room_size, -2.0, v.y * room_size);
            normal    = vec3<f32>(0.0, 1.0, 0.0);
            base_color = vec3<f32>(0.15, 0.14, 0.13);
            metallic = 0.02; roughness = 0.4;
        } else if (face == 1u) {
            position  = vec3<f32>(v.x * room_size, 5.0, v.y * room_size);
            normal    = vec3<f32>(0.0, -1.0, 0.0);
            base_color = vec3<f32>(0.22, 0.21, 0.20);
            roughness = 0.92;
        } else if (face == 2u) {
            position = vec3<f32>(v.x * room_size, v.y * 3.5 + 1.5, room_size);
            normal   = vec3<f32>(0.0, 0.0, -1.0);
            base_color = vec3<f32>(0.28, 0.26, 0.24); roughness = 0.65;
        } else if (face == 3u) {
            position = vec3<f32>(v.x * room_size, v.y * 3.5 + 1.5, -room_size);
            normal   = vec3<f32>(0.0, 0.0, 1.0);
            base_color = vec3<f32>(0.28, 0.26, 0.24); roughness = 0.65;
        } else if (face == 4u) {
            position = vec3<f32>(-room_size, v.y * 3.5 + 1.5, v.x * room_size);
            normal   = vec3<f32>(1.0, 0.0, 0.0);
            base_color = vec3<f32>(0.28, 0.26, 0.24); roughness = 0.65;
        } else {
            let pv = v * 0.4;
            position = vec3<f32>(room_size, pv.y * 3.5 + 1.5, pv.x * room_size * 0.3);
            normal   = vec3<f32>(-1.0, 0.0, 0.0);
            base_color = vec3<f32>(0.7, 0.3, 0.9);
            metallic = 0.8; roughness = 0.1; emissive = 3.0;
        }

        // Space tint (non-portal)
        if (face != 5u) {
            if (space_type == 1u)      { base_color *= vec3<f32>(0.85, 0.78, 1.15); }
            else if (space_type == 2u) { base_color *= vec3<f32>(1.15, 0.95, 0.78); }
        }

        // ─── Apply geometric curvature to room surfaces ──────────
        position = apply_curvature(position, space_type);
        normal = warp_normal(normal, position, space_type);

    } else if (idx < 132u) {
        // ── Sphere (subdivided octahedron) ────────────────────────
        let li = idx - 36u;
        let sphere_center = scene.sphere_pos.xyz;
        let sphere_glow   = scene.sphere_pos.w;
        let sphere_radius = 1.2;

        let ot = li / 12u; let si = li % 12u; let st2 = si / 3u; let vi = si % 3u;
        let fid = fi[ot];
        let A = oct[fid[0]]; let B = oct[fid[1]]; let C = oct[fid[2]];
        let mAB = normalize((A + B) * 0.5);
        let mBC = normalize((B + C) * 0.5);
        let mCA = normalize((C + A) * 0.5);

        var vert = vec3<f32>(0.0);
        if (st2 == 0u) { if (vi == 0u) { vert = A; } else if (vi == 1u) { vert = mAB; } else { vert = mCA; } }
        else if (st2 == 1u) { if (vi == 0u) { vert = mAB; } else if (vi == 1u) { vert = B; } else { vert = mBC; } }
        else if (st2 == 2u) { if (vi == 0u) { vert = mCA; } else if (vi == 1u) { vert = mBC; } else { vert = C; } }
        else { if (vi == 0u) { vert = mAB; } else if (vi == 1u) { vert = mBC; } else { vert = mCA; } }

        let n = normalize(vert);
        position = sphere_center + n * sphere_radius;
        normal = n;

        // Space-specific metal (gold / chrome / copper)
        if (space_type == 0u) { base_color = vec3<f32>(1.0, 0.86, 0.57); }
        else if (space_type == 1u) { base_color = vec3<f32>(0.90, 0.90, 0.95); }
        else { base_color = vec3<f32>(0.98, 0.72, 0.58); }
        metallic = 1.0; roughness = 0.12;
        emissive = sphere_glow * 2.5;

    } else if (idx < 228u) {
        // ── Collectible orbs (4 × 24 verts) ──────────────────────
        let ol = idx - 132u;
        let oi = ol / 24u;
        let inner = ol % 24u;
        let tri = inner / 3u;
        let vi = inner % 3u;

        var od: vec4<f32>;
        if (oi == 0u) { od = scene.orb0_pos; }
        else if (oi == 1u) { od = scene.orb1_pos; }
        else if (oi == 2u) { od = scene.orb2_pos; }
        else { od = scene.orb3_pos; }

        let orb_center = od.xyz;
        let orb_active = od.w;
        let orb_radius = 0.35 * orb_active;

        let fid = fi[tri];
        var vert: vec3<f32>;
        if (vi == 0u) { vert = oct[fid[0]]; }
        else if (vi == 1u) { vert = oct[fid[1]]; }
        else { vert = oct[fid[2]]; }

        let n = normalize(vert);
        position = orb_center + n * orb_radius;
        normal = n;

        if (oi == 0u) { base_color = vec3<f32>(0.2, 0.9, 1.0); }
        else if (oi == 1u) { base_color = vec3<f32>(1.0, 0.3, 0.8); }
        else if (oi == 2u) { base_color = vec3<f32>(1.0, 0.9, 0.2); }
        else { base_color = vec3<f32>(0.3, 1.0, 0.4); }

        metallic = 0.7; roughness = 0.2;
        let pd = length(camera.view_position.xyz - orb_center);
        let ps = mix(1.5, 6.0, clamp(1.0 - pd / 5.0, 0.0, 1.0));
        emissive = (sin(time * ps + f32(oi) * 1.5) * 0.4 + 0.6) * 4.0 * orb_active;
    }

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_pos     = position;
    out.normal        = normal;
    out.base_color    = base_color;
    out.pbr_params    = vec4<f32>(metallic, roughness, ao, emissive);
    return out;
}

// ─── PBR Functions ─────────────────────────────────────────────────────────

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let NdH = max(dot(N, H), 0.0);
    let d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}
fn geometry_schlick(NdV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, r: f32) -> f32 {
    return geometry_schlick(max(dot(N, V), 0.0), r) * geometry_schlick(max(dot(N, L), 0.0), r);
}
fn fresnel_schlick(ct: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - ct, 0.0, 1.0), 5.0);
}
fn fresnel_schlick_r(ct: f32, F0: vec3<f32>, r: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - r), F0) - F0) * pow(clamp(1.0 - ct, 0.0, 1.0), 5.0);
}
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn to_srgb(c: vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(1.0 / 2.2)); }

fn pbr_direct(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, rad: vec3<f32>, alb: vec3<f32>, met: f32, rou: f32) -> vec3<f32> {
    let H = normalize(V + L);
    let F0 = mix(vec3<f32>(0.04), alb, met);
    let D = distribution_ggx(N, H, rou);
    let G = geometry_smith(N, V, L, rou);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let NdV = max(dot(N, V), 0.0); let NdL = max(dot(N, L), 0.0);
    let spec = (D * G * F) / (4.0 * NdV * NdL + 0.0001);
    return ((vec3<f32>(1.0) - F) * (1.0 - met) * alb / PI + spec) * rad * NdL;
}
fn pt_light(N: vec3<f32>, V: vec3<f32>, wp: vec3<f32>, lp: vec3<f32>, lc: vec3<f32>, li: f32, a: vec3<f32>, m: f32, r: f32) -> vec3<f32> {
    let d = length(lp - wp);
    return pbr_direct(N, V, normalize(lp - wp), lc * li / (d * d + 1.0), a, m, r);
}

// ─── Fragment Shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);

    var alb = in.base_color;
    let met = in.pbr_params.x;
    var rou = clamp(in.pbr_params.y, 0.045, 1.0);
    let ao  = in.pbr_params.z;
    let em  = in.pbr_params.w;

    let st  = u32(camera.view_position.w);
    let time = scene.params.x;
    let expo = scene.params.y;
    let ambi = scene.params.z;
    let nlit = u32(scene.params.w);

    // ── Floor checker ─────────────────────────────────────────────
    if (N.y > 0.5 && em == 0.0) {
        let cx = floor(in.world_pos.x / 2.0 + 0.001);
        let cz = floor(in.world_pos.z / 2.0 + 0.001);
        if (((i32(cx) + i32(cz)) % 2 + 2) % 2 == 0) { rou = 0.12; alb *= 1.35; }
        else { rou = 0.55; }
    }

    // ── Lighting accumulation ─────────────────────────────────────
    var Lo = pbr_direct(N, V, normalize(scene.sun_direction.xyz), scene.sun_color.xyz * scene.sun_direction.w, alb, met, rou);
    if (nlit >= 1u) { Lo += pt_light(N, V, in.world_pos, scene.light0_pos.xyz, scene.light0_color.xyz, scene.light0_color.w, alb, met, rou); }
    if (nlit >= 2u) { Lo += pt_light(N, V, in.world_pos, scene.light1_pos.xyz, scene.light1_color.xyz, scene.light1_color.w, alb, met, rou); }
    if (nlit >= 3u) { Lo += pt_light(N, V, in.world_pos, scene.light2_pos.xyz, scene.light2_color.xyz, scene.light2_color.w, alb, met, rou); }
    if (nlit >= 4u) { Lo += pt_light(N, V, in.world_pos, scene.light3_pos.xyz, scene.light3_color.xyz, scene.light3_color.w, alb, met, rou); }

    // ── Ambient ───────────────────────────────────────────────────
    let F0a = mix(vec3<f32>(0.04), alb, met);
    let kSa = fresnel_schlick_r(max(dot(N, V), 0.0), F0a, rou);
    let kDa = (vec3<f32>(1.0) - kSa) * (1.0 - met);
    var sky = vec3<f32>(0.06, 0.09, 0.14); var gnd = vec3<f32>(0.02, 0.02, 0.03);
    if (st == 1u) { sky = vec3<f32>(0.09, 0.04, 0.16); gnd = vec3<f32>(0.03, 0.01, 0.06); }
    else if (st == 2u) { sky = vec3<f32>(0.16, 0.09, 0.04); gnd = vec3<f32>(0.06, 0.03, 0.01); }
    let hemi = mix(gnd, sky, N.y * 0.5 + 0.5);
    var color = Lo + kDa * alb * hemi * ambi * ao + kSa * hemi * ambi * 0.3;

    // ── Emissive ──────────────────────────────────────────────────
    if (em > 0.0) {
        if (in.world_pos.x > 9.5) {
            let center = vec2<f32>(0.0, 1.5);
            let dist = length(in.world_pos.yz - center);
            let ripple = sin(dist * 6.0 - time * 3.5) * 0.5 + 0.5;
            let pulse = sin(time * 1.5) * 0.15 + 0.85;
            let glow = exp(-dist * 0.35);
            var ec: vec3<f32>;
            if (st == 0u) { ec = vec3<f32>(0.4, 0.65, 1.0); }
            else if (st == 1u) { ec = vec3<f32>(0.8, 0.35, 1.0); }
            else { ec = vec3<f32>(1.0, 0.6, 0.2); }
            color += ec * (1.0 + ripple * 0.6) * em * pulse * glow;
            if (dist < 1.5) { let c2 = 1.0 - dist / 1.5; color += ec * c2 * c2 * 6.0; }
        } else {
            color += alb * em;
        }
    }

    // ── Curvature-aware grid ──────────────────────────────────────
    if (em == 0.0) {
        var gc: vec2<f32>;
        if (abs(N.y) > 0.5) { gc = in.world_pos.xz; }
        else if (abs(N.z) > 0.5) { gc = in.world_pos.xy; }
        else { gc = in.world_pos.zy; }

        var gs = 2.0;
        // Hyperbolic: grid spacing increases at edges (exponential expansion)
        if (st == 1u) {
            let r = length(gc);
            gs = 2.0 * (1.0 + r * 0.03);
        }
        // Spherical: grid spacing decreases at edges (convergence)
        else if (st == 2u) {
            let r = length(gc);
            gs = 2.0 / (1.0 + r * 0.02);
        }

        let lw = 0.025;
        let gx = fract(gc.x / gs); let gy = fract(gc.y / gs);
        if ((gx < lw || gx > 1.0 - lw) || (gy < lw || gy > 1.0 - lw)) {
            var gl: vec3<f32>;
            if (st == 0u) { gl = vec3<f32>(0.15, 0.25, 0.55); }
            else if (st == 1u) { gl = vec3<f32>(0.40, 0.15, 0.55); }
            else { gl = vec3<f32>(0.55, 0.30, 0.10); }
            color += gl * 0.35;
        }
    }

    // ── Distance fog ──────────────────────────────────────────────
    let vd = length(camera.view_position.xyz - in.world_pos);
    // Hyperbolic: fog grows exponentially faster (space expands)
    // Spherical: fog grows slower (space contracts)
    var fog_rate = 0.012;
    if (st == 1u) { fog_rate = 0.018; }
    else if (st == 2u) { fog_rate = 0.007; }
    let fog_f = exp(-vd * fog_rate);
    var fog_c: vec3<f32>;
    if (st == 0u) { fog_c = vec3<f32>(0.015, 0.02, 0.05); }
    else if (st == 1u) { fog_c = vec3<f32>(0.03, 0.015, 0.06); }
    else { fog_c = vec3<f32>(0.06, 0.03, 0.015); }
    color = mix(fog_c, color, fog_f);

    color = aces(color * expo);
    color = to_srgb(color);
    return vec4<f32>(color, 1.0);
}