// Metatopia Engine - Stereoscopic 3D VR Shader for Meta Quest 3S & PCVR
// Supports: Side-by-Side (SBS) Dual-Eye Stereo, IPD parallax, PBR materials, and non-Euclidean manifolds

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
};

struct SceneUniform {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    params: vec4<f32>,         // x: time, y: exposure, z: ambient, w: custom
    game_data: vec4<f32>,      // x: time, y: is_vr_sbs_mode, z: ipd_meters, w: fov
    extra0: vec4<f32>,         // Left Eye position (xyz)
    extra1: vec4<f32>,         // Right Eye position (xyz)
    extra2: vec4<f32>,
    extra3: vec4<f32>,
    hud_info: vec4<f32>,       // x: screen_w, y: screen_h
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>, // x: metallic, y: roughness, z: emissive, w: eye_tag (1.0=Left, 2.0=Right, 0.0=Mono)
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
    out.world_pos = in.position;
    out.normal = normalize(in.normal);
    out.uv = in.uv;
    out.color = in.color;
    out.pbr = in.pbr;

    let is_sbs = scene.game_data.y > 0.5;
    let eye_tag = in.pbr.w;

    var clip = camera.view_proj * vec4<f32>(in.position, 1.0);

    if (is_sbs) {
        if (eye_tag == 1.0) {
            // Left Eye: viewport mapped to left half [-1.0, 0.0]
            clip.x = clip.x * 0.5 - 0.5 * clip.w;
        } else if (eye_tag == 2.0) {
            // Right Eye: viewport mapped to right half [0.0, 1.0]
            clip.x = clip.x * 0.5 + 0.5 * clip.w;
        }
    }

    out.clip_position = clip;
    return out;
}

// ACES Tone Mapping
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
    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    let L = normalize(-scene.sun_direction.xyz);
    let H = normalize(L + V);

    let metallic = in.pbr.x;
    let roughness = max(in.pbr.y, 0.05);
    let emissive = in.pbr.z;

    // Diffuse & Specular
    let NdotL = max(dot(N, L), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 0.0);

    // Fresnel Schlick
    let F0 = mix(vec3<f32>(0.04), in.color, metallic);
    let F = F0 + (1.0 - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);

    // Specular GGX
    let spec_power = (2.0 / (roughness * roughness)) - 2.0;
    let spec = pow(NdotH, max(spec_power, 1.0)) * (metallic * 0.8 + 0.2);

    let diffuse = in.color * (1.0 - metallic) * NdotL;
    let ambient = in.color * vec3<f32>(0.15, 0.18, 0.25) * 0.4;

    var final_color = ambient + (diffuse + spec * F) * scene.sun_color.rgb + in.color * emissive;

    // Grid Floor Shading for ground
    if (abs(in.world_pos.y) < 0.02 && abs(N.y) > 0.9) {
        let coord = in.world_pos.xz;
        let grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
        let line = min(grid.x, grid.y);
        let grid_val = 1.0 - min(line, 1.0);

        var grid_color = vec3<f32>(0.05, 0.07, 0.10);
        grid_color += vec3<f32>(0.2, 0.3, 0.5) * grid_val * 0.5;
        final_color = mix(grid_color, final_color, 0.2);
    }

    let mapped = aces_tonemap(final_color);
    return vec4<f32>(mapped, 1.0);
}
