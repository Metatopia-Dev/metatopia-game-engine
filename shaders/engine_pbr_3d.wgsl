// ============================================================================
// Metatopia 3D Game Engine — PBR Mesh, Sky Atmosphere, Ground Grid & 3D Gizmo Shader
// ============================================================================

struct SceneUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,      // xyz = eye pos, w = time
    sun_dir: vec4<f32>,         // xyz = normalized direction to sun, w = sun intensity
    viewport: vec4<f32>,        // x = width, y = height, z = show_grid (1/0), w = play_mode (1/0)
    point_light_pos: vec4<f32>, // xyz = dynamic projectile/crystal light pos, w = radius
    point_light_col: vec4<f32>, // rgb = color, w = intensity
    shadow_casters: array<vec4<f32>, 16>, // xyz = caster center, w = caster radius (for soft ground & object shadows)
};

@group(0) @binding(0)
var<uniform> u: SceneUniform;

// ─── 1. Fullscreen Sky Atmosphere + Procedural Horizon + 3D Ground Grid ────

struct SkyVsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyVsOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    var out: SkyVsOut;
    out.clip_pos = vec4<f32>(pos[vid], 0.9999, 1.0);
    out.ndc = pos[vid];
    return out;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_sky(in: SkyVsOut) -> @location(0) vec4<f32> {
    let near_h = u.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far_h  = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p  = far_h.xyz / far_h.w;
    let rd = normalize(far_p - near_p);

    let sun_d = normalize(u.sun_dir.xyz);
    let up = clamp(rd.y, 0.0, 1.0);
    let down = clamp(-rd.y, 0.0, 1.0);

    // Physically-inspired Rayleigh + Mie sky gradient
    let sky_zenith = vec3<f32>(0.07, 0.16, 0.34);
    let sky_horizon = vec3<f32>(0.36, 0.54, 0.74);
    let ground_nadir = vec3<f32>(0.08, 0.10, 0.14);

    var col = mix(sky_horizon, sky_zenith, pow(up, 0.55));
    col = mix(col, ground_nadir, pow(down, 0.45));

    // Horizon atmospheric haze band
    let horizon_glow = exp(-abs(rd.y) * 10.0);
    col += vec3<f32>(0.22, 0.28, 0.36) * horizon_glow * 0.45;

    // Directional Sun disc & Mie halo
    let sun_cos = max(dot(rd, sun_d), 0.0);
    let sun_halo = pow(sun_cos, 16.0) * 0.35 + pow(sun_cos, 128.0) * 0.9;
    let sun_disc = smoothstep(0.9992, 0.9998, sun_cos) * 3.5;
    col += vec3<f32>(1.0, 0.90, 0.72) * (sun_halo + sun_disc);

    // Subtle procedural high-altitude clouds
    if (rd.y > 0.02) {
        let cloud_uv = rd.xz / (rd.y + 0.15) * 2.2 + vec2<f32>(u.camera_pos.w * 0.02, 0.0);
        let c1 = sin(cloud_uv.x * 2.5) * cos(cloud_uv.y * 2.1);
        let c2 = sin(cloud_uv.x * 5.3 + 1.2) * cos(cloud_uv.y * 4.7 - 0.8) * 0.5;
        let cloud = smoothstep(0.25, 0.85, (c1 + c2) * 0.5 + 0.5) * smoothstep(0.02, 0.25, rd.y);
        col = mix(col, vec3<f32>(0.88, 0.92, 0.98), cloud * 0.18);
    }

    let mapped = linear_to_srgb(aces_tonemap(col));
    return vec4<f32>(mapped, 1.0);
}

// ─── 2. 3D PBR Lit & Shadowed Mesh Pipeline ────────────────────────────────

struct MeshVertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,      // rgb = base albedo, a = alpha / flags
    @location(4) mat_params: vec4<f32>, // x = metallic, y = roughness, z = emissive, w = is_ground_or_selected
};

struct MeshVsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) mat_params: vec4<f32>,
};

@vertex
fn vs_mesh(in: MeshVertexIn) -> MeshVsOut {
    var out: MeshVsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.world_normal = normalize(in.normal);
    out.uv = in.uv;
    out.color = in.color;
    out.mat_params = in.mat_params;
    return out;
}

// Soft ray-sphere shadow occlusion against the 16 active scene shadow casters
fn compute_soft_shadow(p: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    var shadow = 1.0;
    let ro = p + light_dir * 0.08;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let c = u.shadow_casters[i];
        if (c.w <= 0.001) {
            continue;
        }
        let oc = c.xyz - ro;
        // Skip self-shadowing if point is inside caster radius
        if (length(p - c.xyz) < c.w * 1.05) {
            continue;
        }
        let proj = dot(oc, light_dir);
        if (proj > 0.0) {
            let perp2 = max(dot(oc, oc) - proj * proj, 0.0);
            let r2 = c.w * c.w;
            if (perp2 < r2) {
                shadow = min(shadow, 0.18);
            } else {
                let penumbra = clamp((sqrt(perp2) - c.w) / (c.w * 0.55), 0.0, 1.0);
                shadow = min(shadow, mix(0.22, 1.0, penumbra * penumbra));
            }
        }
    }
    return shadow;
}

// Ambient occlusion on ground plane from nearby objects
fn compute_ground_ao(p: vec3<f32>) -> f32 {
    var ao = 1.0;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let c = u.shadow_casters[i];
        if (c.w <= 0.001) {
            continue;
        }
        let dxz = length(p.xz - c.xz);
        let h = max(c.y - p.y, 0.0);
        if (h < c.w * 2.5) {
            let falloff = clamp(dxz / (c.w * 1.45), 0.0, 1.0);
            ao *= mix(0.45, 1.0, falloff);
        }
    }
    return ao;
}

@fragment
fn fs_mesh(in: MeshVsOut) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(u.camera_pos.xyz - in.world_pos);
    let L = normalize(u.sun_dir.xyz);
    let H = normalize(V + L);

    var albedo = in.color.rgb;
    let metallic = clamp(in.mat_params.x, 0.0, 1.0);
    let roughness = clamp(in.mat_params.y, 0.04, 1.0);
    let emissive = in.mat_params.z;
    let mode_flag = in.mat_params.w; // 1.0 = ground plane, 2.0 = selected object highlight

    // ── Procedural Checkerboard & Editor 3D Grid on Ground Plane (mode_flag == 1.0) ──
    if (mode_flag > 0.5 && mode_flag < 1.5) {
        let gx = floor(in.world_pos.x);
        let gz = floor(in.world_pos.z);
        let checker = abs((gx + gz) % 2.0);
        let tile_a = vec3<f32>(0.15, 0.18, 0.23);
        let tile_b = vec3<f32>(0.19, 0.23, 0.29);
        albedo = mix(tile_a, tile_b, checker);

        // Crisp 1m and 5m Editor Grid + World Origin Red X / Blue Z Axes
        let dist = length(in.world_pos.xz - u.camera_pos.xz);
        let fade = 1.0 - smoothstep(14.0, 48.0, dist);

        // 1m minor grid lines
        let fx = abs(fract(in.world_pos.x + 0.5) - 0.5);
        let fz = abs(fract(in.world_pos.z + 0.5) - 0.5);
        let line_1m = 1.0 - smoothstep(0.012, 0.032, min(fx, fz));

        // 5m major grid lines
        let fx5 = abs(fract(in.world_pos.x * 0.2 + 0.5) - 0.5) * 5.0;
        let fz5 = abs(fract(in.world_pos.z * 0.2 + 0.5) - 0.5) * 5.0;
        let line_5m = 1.0 - smoothstep(0.018, 0.048, min(fx5, fz5));

        if (u.viewport.z > 0.5) {
            albedo = mix(albedo, vec3<f32>(0.32, 0.39, 0.48), line_1m * 0.55 * fade);
            albedo = mix(albedo, vec3<f32>(0.50, 0.62, 0.78), line_5m * 0.75 * fade);

            // World X Axis (Red) along z == 0
            let z_axis_dist = abs(in.world_pos.z);
            if (z_axis_dist < 0.055) {
                let ax_w = 1.0 - smoothstep(0.015, 0.055, z_axis_dist);
                albedo = mix(albedo, vec3<f32>(0.95, 0.22, 0.26), ax_w * fade);
            }
            // World Z Axis (Blue) along x == 0
            let x_axis_dist = abs(in.world_pos.x);
            if (x_axis_dist < 0.055) {
                let az_w = 1.0 - smoothstep(0.015, 0.055, x_axis_dist);
                albedo = mix(albedo, vec3<f32>(0.20, 0.55, 0.98), az_w * fade);
            }
        }
    }

    // ── Cook-Torrance PBR Direct Sun Lighting + Soft Shadows ──
    let ndotl = max(dot(N, L), 0.0);
    let ndotv = max(dot(N, V), 0.001);
    let ndoth = max(dot(N, H), 0.0);
    let vdoth = max(dot(V, H), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vdoth, 5.0);

    // GGX Normal Distribution
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / max(3.14159265 * denom * denom, 0.0001);

    // Schlick-GGX Geometry
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g1_v = ndotv / (ndotv * (1.0 - k) + k);
    let g1_l = ndotl / (ndotl * (1.0 - k) + k);
    let G = g1_v * g1_l;

    let specular = (D * G * fresnel) / max(4.0 * ndotv * ndotl, 0.001);
    let kd = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic);
    let diffuse = kd * albedo / 3.14159265;

    let shadow = compute_soft_shadow(in.world_pos, L);
    let ao = compute_ground_ao(in.world_pos);
    let sun_color = vec3<f32>(1.0, 0.95, 0.86) * u.sun_dir.w;

    var color = (diffuse + specular) * sun_color * ndotl * shadow;

    // ── Hemisphere Sky / Ground Ambient + Specular Environment Reflection ──
    let hemi = N.y * 0.5 + 0.5;
    let sky_amb = mix(vec3<f32>(0.10, 0.12, 0.16), vec3<f32>(0.28, 0.40, 0.58), hemi);
    let R = reflect(-V, N);
    let env_ref = mix(vec3<f32>(0.12, 0.15, 0.20), vec3<f32>(0.38, 0.56, 0.78), clamp(R.y * 0.5 + 0.5, 0.0, 1.0));
    color += (albedo * (1.0 - metallic) * sky_amb * 0.65 + env_ref * fresnel * (1.0 - roughness * 0.75) * 0.55) * ao;

    // ── Dynamic Point Light (Plasma Projectiles / Energy Crystal) ──
    if (u.point_light_pos.w > 0.01) {
        let to_pl = u.point_light_pos.xyz - in.world_pos;
        let pl_dist = length(to_pl);
        let pl_dir = to_pl / max(pl_dist, 0.001);
        let atten = clamp(1.0 - pl_dist / u.point_light_pos.w, 0.0, 1.0);
        let pl_ndotl = max(dot(N, pl_dir), 0.0);
        color += albedo * u.point_light_col.rgb * u.point_light_col.w * pl_ndotl * atten * atten;
    }

    // ── Emissive & Selected Rim Highlight ──
    color += albedo * emissive;

    if (mode_flag > 1.5) {
        // Selected GameObject warm orange/cyan Fresnel rim highlight (like Unity/Unreal selection outline)
        let rim = pow(1.0 - ndotv, 2.5);
        color += vec3<f32>(1.0, 0.62, 0.12) * rim * 0.85;
    }

    // Distance atmospheric fog blend
    let cam_dist = length(in.world_pos - u.camera_pos.xyz);
    let fog = smoothstep(28.0, 68.0, cam_dist);
    color = mix(color, vec3<f32>(0.28, 0.42, 0.58), fog);

    let out_srgb = linear_to_srgb(aces_tonemap(color));
    return vec4<f32>(out_srgb, in.color.a);
}

// ─── 3. Unlit 3D RGB Transform Gizmo & Wireframe Selection Overlay ──────────

@fragment
fn fs_gizmo(in: MeshVsOut) -> @location(0) vec4<f32> {
    // Crisp high-visibility RGB Gizmo with subtle directional shading so 3D cones/shafts pop
    let N = normalize(in.world_normal);
    let shade = 0.78 + 0.22 * max(dot(N, normalize(vec3<f32>(0.5, 0.8, 0.4))), 0.0);
    let col = clamp(in.color.rgb * shade, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(col, in.color.a);
}
