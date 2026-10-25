// CHRONO-PORTAL: Non-Euclidean Puzzle & Relativistic FPS Shader
// Combines Riemannian curvature vertex projection, Schwarzschild accretion Doppler glow,
// recursive portal aperture interferometry, and 2D tactical HUD rendering.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
};

struct SceneUniform {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    params: vec4<f32>,         // x: time, y: chamber_idx, z: laser_locked, w: wormhole_throat
    game_data: vec4<f32>,      // x: time, y: curvature_k, z: chirality, w: boss_hp_ratio
    extra0: vec4<f32>,         // x: bh_pos.x, y: bh_pos.y, z: bh_pos.z, w: r_s
    extra1: vec4<f32>,
    extra2: vec4<f32>,
    extra3: vec4<f32>,
    hud_info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>, // x: metallic, y: roughness, z: emissive, w: object_id (< -0.5 = 2D HUD)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.uv;
    out.color = in.color;
    out.pbr = in.pbr;

    if (in.pbr.w < -0.5) {
        out.world_pos = in.position;
        out.normal = in.normal;
        out.clip_position = vec4<f32>(in.position.xy, 0.0, 1.0);
        return out;
    }

    let k = scene.game_data.y;
    let chirality = scene.game_data.z;
    let rel = in.position - camera.view_position.xyz;
    let r = length(rel);

    var warped_pos = in.position;
    if (abs(k) > 0.01 && r > 0.001) {
        let scale_factor = 1.0 / (1.0 + 0.018 * abs(k) * r * r);
        warped_pos = camera.view_position.xyz + rel * mix(1.0, scale_factor, 0.45);
    }

    // Apply subtle Klein bottle chiral twist when chirality < 0
    if (chirality < 0.0) {
        let twist = 0.04 * sin(scene.game_data.x * 1.5 + in.position.z * 0.25);
        warped_pos.x = warped_pos.x * cos(twist) - warped_pos.y * sin(twist);
    }

    out.world_pos = warped_pos;
    out.normal = normalize(in.normal);
    out.clip_position = camera.view_proj * vec4<f32>(warped_pos, 1.0);
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_id = in.pbr.w;
    if (object_id < -0.5) {
        return vec4<f32>(in.color, in.pbr.z);
    }

    let t = scene.game_data.x;
    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    let L = normalize(-scene.sun_direction.xyz);
    let H = normalize(L + V);

    let metallic = in.pbr.x;
    let roughness = max(in.pbr.y, 0.05);
    let emissive = in.pbr.z;

    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 0.0);

    let F0 = mix(vec3<f32>(0.04), in.color, metallic);
    let F = F0 + (1.0 - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    let spec_power = (2.0 / (roughness * roughness)) - 2.0;
    let spec = pow(NdotH, max(spec_power, 1.0)) * (metallic * 0.8 + 0.2);

    var base_color = in.color;

    // Portal Aperture Recursive Interferometry (object_id == 2.0)
    if (abs(object_id - 2.0) < 0.1) {
        let wave = sin(length(in.world_pos.xy) * 8.0 - t * 6.0) * 0.5 + 0.5;
        base_color = mix(in.color, vec3<f32>(0.0, 0.98, 1.0), wave * 0.65);
    }

    // Schwarzschild Accretion Disk Relativistic Doppler Beaming (object_id == 4.0)
    if (abs(object_id - 4.0) < 0.1) {
        let doppler = clamp(1.0 + 0.45 * sin(atan2(in.world_pos.z, in.world_pos.x) + t * 4.0), 0.3, 2.2);
        let beaming = pow(doppler, 3.0);
        base_color = in.color * beaming;
    }

    let diffuse = base_color * (1.0 - metallic) * NdotL;
    let ambient = base_color * vec3<f32>(0.16, 0.20, 0.28) * 0.45;
    var hdr = ambient + (diffuse + spec * F) * scene.sun_color.rgb + base_color * emissive;

    // Geodesic Floor Grid Lines
    if (abs(in.world_pos.y) < 0.05 && abs(N.y) > 0.85) {
        let coord = in.world_pos.xz * 0.5;
        let grid = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), vec2<f32>(0.001));
        let line = 1.0 - min(min(grid.x, grid.y), 1.0);
        let grid_tint = mix(vec3<f32>(0.0, 0.85, 1.0), vec3<f32>(1.0, 0.2, 0.85), clamp(-scene.game_data.y * 0.5, 0.0, 1.0));
        hdr += grid_tint * line * 0.45;
    }

    let mapped = aces_tonemap(hdr);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
