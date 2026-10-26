// ============================================================================
// METATOPIA 3D ENGINE — AAA PBR + 2048x2048 PCF SHADOW MAP + PROCEDURAL SHADER
// Co-designed by GX10 Blackwell (Qwen 3.6 35B), Claude Fable 5.1 & GPT-6 Astra
// ============================================================================

struct SceneUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,       // xyz = eye pos, w = time
    sun_dir: vec4<f32>,          // xyz = normalized sun dir (toward sun), w = sun intensity
    viewport: vec4<f32>,         // x = width, y = height, z = play_mode (0=Edit, 1=Play), w = grid_alpha
    shadow_params: vec4<f32>,    // x = shadow_texel (1/2048), y = normal_bias, z = shadow_strength, w = debug_view
    hud_params: vec4<f32>,       // x = hp_ratio, y = heat_ratio, z = hit_marker, w = wave_progress
    point_lights: array<vec4<f32>, 8>, // even = xyz pos + radius, odd = rgb color + intensity
};

@group(0) @binding(0)
var<uniform> u: SceneUniform;

@group(1) @binding(0)
var shadow_map: texture_depth_2d;

@group(1) @binding(1)
var shadow_sampler: sampler_comparison;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat: vec4<f32>, // x = metallic, y = roughness, z = emissive, w = material_id / flags
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) mat: vec4<f32>,
};

struct SkyOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

const PI: f32 = 3.14159265359;

// Material IDs encoded in floor(mat.w) when mat.w >= 0.0:
//  -1.0 = Unlit Gizmo / Ray / UI line
//   0.0 = Standard Cook-Torrance PBR + micro-surface detail
//   1.0 = Chamfered Hexagonal Cyber-Tile Floor + Wet Reflective Puddles + Editor Grid
//   2.0 = Anisotropic Brushed Titanium / Mech Armor
//   3.0 = Animated Neon Circuit Traces + Energy Core
//   4.0 = Animated Reflective Water Pool (multi-wave normal perturbation + sun glint)

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u2 = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u2.x), mix(c, d, u2.x), u2.y);
}

// Unified physical sky + clouds + horizon glow used by Sky Dome, IBL reflections, and Fog
fn sky_radiance(rd: vec3<f32>, include_clouds: bool) -> vec3<f32> {
    let sun_dir = normalize(u.sun_dir.xyz);
    let y = clamp(rd.y, -1.0, 1.0);

    let zenith = vec3<f32>(0.025, 0.065, 0.165);
    let horizon = vec3<f32>(0.11, 0.24, 0.42);
    let ground_nadir = vec3<f32>(0.02, 0.028, 0.045);

    var sky = mix(horizon, zenith, pow(clamp(y, 0.0, 1.0), 0.52));
    if y < 0.0 {
        sky = mix(horizon * 0.42, ground_nadir, clamp(-y * 3.2, 0.0, 1.0));
    }

    // Atmospheric Mie scattering & sunset/cyber horizon rim
    let sun_dot = max(dot(rd, sun_dir), 0.0);
    let horizon_band = exp(-abs(y) * 7.5);
    sky += vec3<f32>(0.05, 0.16, 0.28) * horizon_band * 0.55;
    sky += vec3<f32>(1.0, 0.62, 0.28) * pow(sun_dot, 6.0) * 0.28;
    sky += vec3<f32>(1.0, 0.90, 0.72) * pow(sun_dot, 64.0) * 0.95;
    sky += vec3<f32>(1.0, 0.98, 0.92) * smoothstep(0.9993, 0.99985, sun_dot) * 4.5;

    if include_clouds && y > 0.02 {
        let wind = vec2<f32>(u.camera_pos.w * 0.14, u.camera_pos.w * 0.06);
        let uv = rd.xz / (y + 0.16) * 1.15 + wind;
        let c1 = noise2d(uv * 1.6);
        let c2 = noise2d(uv * 3.4 + vec2<f32>(5.2, 1.3));
        let c3 = noise2d(uv * 7.1 - wind * 1.5);
        let cloud = smoothstep(0.47, 0.78, c1 * 0.55 + c2 * 0.30 + c3 * 0.15);
        let sun_lit = pow(sun_dot, 3.0);
        let cloud_col = mix(
            vec3<f32>(0.18, 0.27, 0.42),
            vec3<f32>(1.0, 0.84, 0.66),
            0.35 + 0.65 * sun_lit
        );
        sky = mix(sky, cloud_col, cloud * clamp((y - 0.02) * 3.5, 0.0, 0.58));
    }

    return sky;
}

// ============================================================================
// PASS 0: SHADOW MAP VERTEX SHADER (2048x2048 Depth32Float)
// ============================================================================
@vertex
fn vs_shadow(in: VertexIn) -> @builtin(position) vec4<f32> {
    // Unlit gizmo geometry (mat.w < -0.5) should not cast shadows: move offscreen
    if in.mat.w < -0.5 {
        return vec4<f32>(0.0, 0.0, -2.0, 1.0);
    }
    return u.light_view_proj * vec4<f32>(in.pos, 1.0);
}

// ============================================================================
// PASS 1A: SKY DOME SHADERS
// ============================================================================
@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[vid];
    var out: SkyOut;
    out.clip_pos = vec4<f32>(p.x, p.y, 0.9999, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let clip_near = vec4<f32>(in.ndc.x, in.ndc.y, 0.1, 1.0);
    let clip_far  = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world_near_h = u.inv_view_proj * clip_near;
    let world_far_h  = u.inv_view_proj * clip_far;
    let world_near = world_near_h.xyz / world_near_h.w;
    let world_far  = world_far_h.xyz / world_far_h.w;
    let rd = normalize(world_far - world_near);

    let sky = sky_radiance(rd, true);
    let mapped = pow(aces_tonemap(sky), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(mapped, 1.0);
}

// ============================================================================
// PASS 1B: 3D MESH PBR + 9-TAP PCF SHADOWS + PROCEDURAL MATERIALS
// ============================================================================
@vertex
fn vs_mesh(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    out.mat = in.mat;
    return out;
}

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = max(a * a, 0.0001);
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 1e-6);
}

fn d_ggx_aniso(
    n_dot_h: f32,
    h: vec3<f32>,
    t: vec3<f32>,
    b: vec3<f32>,
    ax: f32,
    ay: f32
) -> f32 {
    let t_dot_h = dot(t, h);
    let b_dot_h = dot(b, h);
    let a2 = ax * ay;
    let v = vec3<f32>(ay * t_dot_h, ax * b_dot_h, a2 * n_dot_h);
    let s = dot(v, v);
    if s <= 1e-7 {
        return 0.0;
    }
    let w = a2 / s;
    return a2 * w * w * (1.0 / PI);
}

fn g_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-6);
}

fn g_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return g_schlick_ggx(n_dot_v, roughness) * g_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// 9-tap hardware bilinear PCF directional shadow sampling (effective 36-texel filter)
fn sample_shadow_pcf9(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let cos_theta = clamp(dot(N, L), 0.0, 1.0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let normal_bias = u.shadow_params.y * (1.0 + 2.2 * sin_theta);
    let biased_pos = world_pos + N * normal_bias;

    let light_clip = u.light_view_proj * vec4<f32>(biased_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z - 0.0012;

    if uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || depth <= 0.0 || depth >= 1.0 {
        return 1.0;
    }

    let texel = u.shadow_params.x;
    var acc = 0.0;
    var weight_sum = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let w = select(1.0, 2.0, x == 0) * select(1.0, 2.0, y == 0);
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            acc += w * textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, depth);
            weight_sum += w;
        }
    }
    let raw_shadow = acc / weight_sum;

    // Fade shadow cleanly near the edge of the ortho frustum
    let edge = max(abs(ndc.x), abs(ndc.y));
    let edge_fade = smoothstep(0.88, 0.99, edge);
    let strength = u.shadow_params.z;
    return mix(1.0, mix(raw_shadow, 1.0, edge_fade), strength);
}

// Hexagonal grid cell helper: returns (local_x, local_y, edge_distance, cell_hash)
fn hex_cell(p: vec2<f32>, scale: f32) -> vec4<f32> {
    let s = vec2<f32>(1.0, 1.7320508);
    let h = s * 0.5;
    let q = p * scale;
    let a = (q - s * floor(q / s)) - h;
    let b = (q - h - s * floor((q - h) / s)) - h;
    let gv = select(b, a, dot(a, a) < dot(b, b));
    let id = q - gv;
    let hex_d = max(abs(gv.x), dot(abs(gv), vec2<f32>(0.5, 0.8660254)));
    let edge_dist = 0.5 - hex_d;
    return vec4<f32>(gv.x, gv.y, edge_dist, hash21(floor(id * 2.0)));
}

// Procedural circuit pattern for sci-fi cores and energy trims
fn circuit_pattern(world_pos: vec3<f32>, N: vec3<f32>, time: f32) -> f32 {
    let abs_n = abs(N);
    var uv = world_pos.xz;
    if abs_n.x > abs_n.y && abs_n.x > abs_n.z {
        uv = world_pos.yz;
    } else if abs_n.z > abs_n.x && abs_n.z > abs_n.y {
        uv = world_pos.xy;
    }
    let scaled = uv * 4.5;
    let cell = floor(scaled);
    let f = fract(scaled) - vec2<f32>(0.5);
    let h = hash21(cell);
    let dir = select(vec2<f32>(f.x, f.y), vec2<f32>(f.y, -f.x), h > 0.5);
    let line1 = smoothstep(0.08, 0.02, abs(dir.x));
    let line2 = smoothstep(0.08, 0.02, abs(dir.y)) * step(0.45, h);
    let node = smoothstep(0.18, 0.09, length(f));
    let pulse = 0.55 + 0.45 * sin(time * 4.2 + h * 18.0 + (uv.x + uv.y) * 2.5);
    return clamp(max(max(line1, line2), node) * pulse, 0.0, 1.0);
}

// Multi-wave procedural water normal for the central reflection basin
fn water_normal(xz: vec2<f32>, time: f32) -> vec3<f32> {
    let p1 = xz * 2.2 + vec2<f32>(time * 1.35, -time * 0.85);
    let p2 = xz * 4.7 + vec2<f32>(-time * 1.75, time * 1.15);
    let p3 = xz * 8.3 + vec2<f32>(time * 2.4, time * 1.9);
    let dx = cos(p1.x + sin(p1.y)) * 0.032
           + cos(p2.x * 1.2 - p2.y) * 0.020
           + sin(p3.x + p3.y) * 0.011;
    let dz = sin(p1.y + cos(p1.x)) * 0.032
           + sin(p2.y * 1.1 + p2.x) * 0.020
           + cos(p3.y - p3.x) * 0.011;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

@fragment
fn fs_mesh(in: VertexOut) -> @location(0) vec4<f32> {
    // 1. Unlit Gizmo / Ray / UI Pass-Through
    if in.mat.w < -0.5 {
        return vec4<f32>(in.color, 1.0);
    }

    let time = u.camera_pos.w;
    let mat_id = i32(floor(in.mat.w + 0.1));
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let L = normalize(u.sun_dir.xyz);

    var N = normalize(in.world_normal);
    var albedo = in.color;
    var metallic = clamp(in.mat.x, 0.0, 1.0);
    var roughness = clamp(in.mat.y, 0.04, 1.0);
    var emissive_strength = max(in.mat.z, 0.0);

    // ------------------------------------------------------------------------
    // Procedural Material Specialization
    // ------------------------------------------------------------------------
    if mat_id == 1 && N.y > 0.85 {
        // MAT_HEX_FLOOR: Chamfered hexagonal cyber tiles + wet puddles + Editor grid
        let hc = hex_cell(in.world_pos.xz, 1.25);
        let seam = smoothstep(0.055, 0.012, hc.z);
        let bevel_n = normalize(vec3<f32>(hc.x * 0.18, 1.0, hc.y * 0.18));
        N = normalize(mix(N, bevel_n, seam * 0.75));

        // Tile-to-tile subtle value variation and glowing cyan sub-seams
        albedo = albedo * (0.88 + 0.24 * hc.w) * (1.0 - 0.45 * seam);
        let pulse_seam = seam * smoothstep(0.72, 0.98, sin(length(in.world_pos.xz) * 0.45 - time * 1.8));
        albedo = mix(albedo, vec3<f32>(0.08, 0.65, 0.92), pulse_seam * 0.35);
        emissive_strength += pulse_seam * 0.28;

        // Wet puddle patches reflecting the procedural sky & neon lights
        let puddle_noise = noise2d(in.world_pos.xz * 0.22 + vec2<f32>(3.1, 7.4));
        let puddle_mask = smoothstep(0.56, 0.72, puddle_noise) * (1.0 - seam * 0.5);
        roughness = mix(roughness, 0.065, puddle_mask);
        metallic = mix(metallic, 0.25, puddle_mask * 0.5);
        N = normalize(mix(N, vec3<f32>(0.0, 1.0, 0.0), puddle_mask * 0.85));

        // Anti-aliased 1m / 5m Editor Grid + World X/Z Axes
        let grid_alpha = u.viewport.w;
        if grid_alpha > 0.01 {
            let coord = in.world_pos.xz;
            let deriv = max(fwidth(coord), vec2<f32>(0.0005));
            let grid_1m = abs(fract(coord - 0.5) - 0.5) / deriv;
            let line_1m = 1.0 - min(min(grid_1m.x, grid_1m.y), 1.0);

            let coord_5m = coord * 0.2;
            let deriv_5m = max(fwidth(coord_5m), vec2<f32>(0.0001));
            let grid_5m = abs(fract(coord_5m - 0.5) - 0.5) / deriv_5m;
            let line_5m = 1.0 - min(min(grid_5m.x, grid_5m.y), 1.0);

            let dist = length(in.world_pos.xz - u.camera_pos.xz);
            let fade = clamp(1.0 - dist / 62.0, 0.0, 1.0) * grid_alpha;

            albedo = mix(albedo, vec3<f32>(0.20, 0.34, 0.52), line_1m * 0.25 * fade);
            albedo = mix(albedo, vec3<f32>(0.28, 0.68, 0.96), line_5m * 0.58 * fade);

            let axis_z = 1.0 - min(abs(coord.x) / (deriv.x * 1.8), 1.0);
            let axis_x = 1.0 - min(abs(coord.y) / (deriv.y * 1.8), 1.0);
            albedo = mix(albedo, vec3<f32>(0.98, 0.24, 0.30), axis_x * 0.88 * fade);
            albedo = mix(albedo, vec3<f32>(0.24, 0.56, 1.00), axis_z * 0.88 * fade);
        }
    } else if mat_id == 2 {
        // MAT_BRUSHED: Anisotropic brushed titanium & mech armor with panel seams
        let brush = sin(in.world_pos.y * 85.0 + in.world_pos.x * 25.0) * 0.5 + 0.5;
        roughness = clamp(roughness + (brush - 0.5) * 0.08, 0.06, 0.92);
        let panel_line = smoothstep(0.96, 0.995, abs(sin(in.world_pos.y * 6.28318)));
        albedo = albedo * (1.0 - panel_line * 0.22);
    } else if mat_id == 3 {
        // MAT_CIRCUIT: Animated neon circuit traces on dark carbon/energy substrate
        let trace = circuit_pattern(in.world_pos, N, time);
        albedo = mix(albedo * 0.45, in.color * 1.35, trace);
        emissive_strength = emissive_strength * (0.45 + 0.95 * trace);
    } else if mat_id == 4 {
        // MAT_WATER: Reflective animated water basin under the suspension bridge
        N = water_normal(in.world_pos.xz, time);
        roughness = 0.05;
        metallic = 0.15;
        let caustic = noise2d(in.world_pos.xz * 3.5 + vec2<f32>(time * 0.9, -time * 0.7));
        albedo = mix(vec3<f32>(0.03, 0.16, 0.28), vec3<f32>(0.08, 0.48, 0.68), caustic * 0.45);
    } else {
        // MAT_STANDARD: Subtle micro-normal grain so surfaces never look flat plastic
        let grain = noise2d(in.world_pos.xz * 14.0 + in.world_pos.yy * 9.0) - 0.5;
        roughness = clamp(roughness + grain * 0.035, 0.04, 0.98);
    }

    // ------------------------------------------------------------------------
    // Shader Debug View Modes (accessible from the 3D Viewport Toolbar!)
    //   0 = Full Lit PBR + 2048 PCF Shadows
    //   1 = World Normals Visualization
    //   2 = Metallic (R) / Roughness (G) / Emissive (B) Inspection
    //   3 = Real-Time 2048x2048 PCF Shadow Map Factor Visualization
    // ------------------------------------------------------------------------
    let debug_mode = i32(floor(u.shadow_params.w + 0.1));
    let sun_shadow = sample_shadow_pcf9(in.world_pos, N, L);

    if debug_mode == 1 {
        return vec4<f32>(N * 0.5 + vec3<f32>(0.5), 1.0);
    } else if debug_mode == 2 {
        return vec4<f32>(metallic, roughness, clamp(emissive_strength * 0.35, 0.0, 1.0), 1.0);
    } else if debug_mode == 3 {
        let n_dot_l_dbg = max(dot(N, L), 0.15);
        let vis = sun_shadow * n_dot_l_dbg;
        let col = mix(vec3<f32>(0.12, 0.18, 0.34), vec3<f32>(0.98, 0.94, 0.82), vis);
        return vec4<f32>(col, 1.0);
    }

    // ------------------------------------------------------------------------
    // Direct Sun Cook-Torrance / Anisotropic GGX Lighting
    // ------------------------------------------------------------------------
    let H = normalize(V + L);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let n_dot_v = max(dot(N, V), 0.001);
    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);

    var ndf = d_ggx(n_dot_h, roughness);
    if mat_id == 2 {
        let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.y) > 0.92);
        let T = normalize(cross(up_ref, N));
        let B = normalize(cross(N, T));
        let ax = max(roughness * 0.45, 0.04);
        let ay = min(roughness * 1.65, 0.95);
        ndf = d_ggx_aniso(n_dot_h, H, T, B, ax, ay);
    }

    let g = g_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);
    let specular = (ndf * g * f) / (4.0 * n_dot_v * n_dot_l + 0.001);
    let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);

    let sun_radiance = vec3<f32>(1.0, 0.95, 0.86) * u.sun_dir.w;
    var lo = (k_d * albedo / PI + specular) * sun_radiance * n_dot_l * sun_shadow;

    // ------------------------------------------------------------------------
    // 4 Dynamic Point Lights (Reactor Cores, Plasma Projectiles, Beacons)
    // ------------------------------------------------------------------------
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lpos_rad = u.point_lights[i * 2u];
        let lcol_int = u.point_lights[i * 2u + 1u];
        let radius = max(lpos_rad.w, 0.001);
        let intensity = lcol_int.w;
        if intensity > 0.001 {
            let to_light = lpos_rad.xyz - in.world_pos;
            let dist = length(to_light);
            if dist < radius {
                let PL = to_light / max(dist, 0.001);
                let PH = normalize(V + PL);
                let atten = pow(clamp(1.0 - dist / radius, 0.0, 1.0), 2.0) * intensity;
                let pnl = max(dot(N, PL), 0.0);
                let pnh = max(dot(N, PH), 0.0);
                let phv = max(dot(PH, V), 0.0);
                let p_ndf = d_ggx(pnh, roughness);
                let p_g = g_smith(n_dot_v, pnl, roughness);
                let p_f = fresnel_schlick(phv, f0);
                let p_spec = (p_ndf * p_g * p_f) / (4.0 * n_dot_v * pnl + 0.001);
                let p_kd = (vec3<f32>(1.0) - p_f) * (1.0 - metallic);
                lo += (p_kd * albedo / PI + p_spec) * lcol_int.rgb * pnl * atten;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Split-Sum Environment IBL + Real Sky Specular Reflection + Rim Light
    // ------------------------------------------------------------------------
    let R = reflect(-V, N);
    let env_sky = sky_radiance(R, false);
    let env_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let env_spec = env_sky * env_fresnel * pow(1.0 - roughness * 0.72, 2.0) * (0.45 + 0.65 * metallic);

    let hemi_t = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient_irr = mix(vec3<f32>(0.04, 0.06, 0.10), vec3<f32>(0.15, 0.24, 0.38), hemi_t);
    let ambient_kd = (vec3<f32>(1.0) - env_fresnel) * (1.0 - metallic);
    // Floor contact & shadow-coupled ambient occlusion
    let contact_ao = clamp(0.62 + 0.38 * clamp(in.world_pos.y * 1.4, 0.0, 1.0), 0.62, 1.0)
                   * (0.72 + 0.28 * sun_shadow);
    let ambient = (ambient_kd * albedo * ambient_irr + env_spec) * contact_ao;

    // Subtle Grazing Fresnel Rim for silhouette clarity
    let rim = pow(1.0 - n_dot_v, 4.0) * (0.18 + 0.32 * (1.0 - roughness));
    let rim_light = vec3<f32>(0.14, 0.42, 0.72) * rim * max(N.y + 0.3, 0.0);

    let emissive = in.color * emissive_strength * 2.5;
    var color = ambient + lo + rim_light + emissive;

    // ------------------------------------------------------------------------
    // Seamless Atmospheric Distance Fog (matches sky_radiance(rd) at horizon!)
    // ------------------------------------------------------------------------
    let cam_dist = length(in.world_pos - u.camera_pos.xyz);
    let view_ray = normalize(in.world_pos - u.camera_pos.xyz);
    let fog_factor = clamp(1.0 - exp(-max(cam_dist - 22.0, 0.0) * 0.015), 0.0, 0.88);
    let fog_color = sky_radiance(view_ray, false);
    color = mix(color, fog_color, fog_factor);

    let mapped = pow(aces_tonemap(color), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(mapped, 1.0);
}
