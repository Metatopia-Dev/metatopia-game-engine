// ─── Template Game Shader ──────────────────────────────────────────────────
// Starter shader for new games using metatopia_engine::quickstart.
//
// This shader receives:
//   - CameraUniform: view_proj matrix, camera position
//   - SceneUniform: lighting, params (time, etc.), game_data, hud_info
//   - GameVertex: position, normal, uv, color, pbr params
//
// Modify the fragment shader to change how things look!

// ─── Uniforms ──────────────────────────────────────────────────────────────

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
}

struct SceneUniform {
    sun_direction: vec4<f32>,   // xyz=direction, w=intensity
    sun_color: vec4<f32>,       // rgb=color
    light0_pos: vec4<f32>,      // xyz=position, w=range
    light0_color: vec4<f32>,    // rgb=color, w=intensity
    params: vec4<f32>,          // x=time, y=exposure, z=ambient, w=custom
    game_data: vec4<f32>,       // YOUR game values (score, level, etc.)
    extra0: vec4<f32>,          // 4 more floats for your use
    extra1: vec4<f32>,
    extra2: vec4<f32>,
    extra3: vec4<f32>,
    hud_info: vec4<f32>,        // x=resolution_x, y=resolution_y, z=custom, w=custom
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;

// ─── Vertex I/O ────────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>,   // x=metallic, y=roughness, z=emission, w=custom
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>,
}

// ─── Vertex Shader ─────────────────────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.uv = in.uv;
    out.color = in.color;
    out.pbr = in.pbr;
    return out;
}

// ─── Lighting Helpers ──────────────────────────────────────────────────────

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

// ─── Fragment Shader ───────────────────────────────────────────────────────
// This is where you customize the look of your game!

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = scene.params.x;

    // ── Base color from vertex ───────────────────────────────────
    let albedo = in.color;
    let N = normalize(in.world_normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);

    // ── Sun light (directional) ──────────────────────────────────
    let sun_dir = normalize(scene.sun_direction.xyz);
    let sun_int = scene.sun_direction.w;
    let sun_col = scene.sun_color.rgb;
    let NdotL = max(dot(N, sun_dir), 0.0);
    let diffuse = albedo * sun_col * sun_int * NdotL;

    // ── Specular (Blinn-Phong) ───────────────────────────────────
    let H = normalize(sun_dir + V);
    let roughness = in.pbr.y;
    let shininess = max(2.0 / (roughness * roughness + 0.001) - 2.0, 1.0);
    let spec = pow(max(dot(N, H), 0.0), shininess) * sun_int * 0.3;

    // ── Point light ──────────────────────────────────────────────
    let lp = scene.light0_pos.xyz;
    let lc = scene.light0_color.rgb;
    let li = scene.light0_color.w;
    let to_light = lp - in.world_pos;
    let d = length(to_light);
    let atten = li / (d * d + 1.0);
    let NdotL2 = max(dot(N, normalize(to_light)), 0.0);
    let point_diffuse = albedo * lc * atten * NdotL2;

    // ── Ambient ──────────────────────────────────────────────────
    let ambient = albedo * 0.15;

    // ── Grid pattern on floor ────────────────────────────────────
    var grid = 0.0;
    if (N.y > 0.9) {
        let gs = 2.0;
        let lw = 0.03;
        let gx = fract(in.world_pos.x / gs);
        let gz = fract(in.world_pos.z / gs);
        if ((gx < lw || gx > 1.0 - lw) || (gz < lw || gz > 1.0 - lw)) {
            grid = 0.15;
        }
    }

    // ── Combine ──────────────────────────────────────────────────
    var color = diffuse + point_diffuse + ambient + vec3(spec) + vec3(grid);

    // Emission
    let emission = in.pbr.z;
    if (emission > 0.0) { color += albedo * emission; }

    // Tone-map + gamma
    color = aces(color * 1.8);
    color = to_srgb(color);

    // ── Crosshair overlay ────────────────────────────────────────
    // Small dot + ring at screen center for FPS aiming.
    let res = vec2<f32>(scene.hud_info.x, scene.hud_info.y);
    if (res.x > 0.0) {
        let px = in.clip_position.xy;
        let center = res * 0.5;
        let d = length(px - center);
        // Dot (radius 2px)
        if (d < 2.0) {
            color = vec3(1.0, 1.0, 1.0);
        }
        // Ring (radius 8px, thickness 1px)
        let ring = abs(d - 8.0);
        if (ring < 1.0) {
            let alpha = 1.0 - ring;
            color = mix(color, vec3(1.0, 1.0, 1.0), alpha * 0.5);
        }
    }

    return vec4<f32>(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
