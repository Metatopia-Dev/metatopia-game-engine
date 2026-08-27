// Metatopia Studio - Professional 3D Game Editor Shader
// Features: Infinite grid floor, PBR metallic/roughness lighting, selection outlines, and 2D Screen UI overlay

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
    game_data: vec4<f32>,      // x: time, y: is_play_mode, z: selected_id, w: hover_id
    extra0: vec4<f32>,
    extra1: vec4<f32>,
    extra2: vec4<f32>,
    extra3: vec4<f32>,
    hud_info: vec4<f32>,       // x: screen_w, y: screen_h, z: mouse_x, w: mouse_y
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>, // x: metallic, y: roughness, z: emissive, w: object_id (w < -0.5 is 2D UI)
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

    if (in.pbr.w < -0.5) {
        // 2D Screen UI overlay: coordinates are already in [-1.0, 1.0] NDC space
        out.clip_position = vec4<f32>(in.position.xy, 0.0, 1.0);
    } else {
        out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    }
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
    let object_id = in.pbr.w;

    // ── 2D UI Overlay Rendering ──────────────────────────────────────────
    if (object_id < -0.5) {
        let opacity = in.pbr.z;
        return vec4<f32>(in.color, opacity);
    }

    // ── 3D Viewport Rendering ─────────────────────────────────────────────
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

    // Diffuse color
    let diffuse = in.color * (1.0 - metallic) * NdotL;
    let ambient = in.color * vec3<f32>(0.2, 0.22, 0.28) * 0.4;

    var final_color = ambient + (diffuse + spec * F) * scene.sun_color.rgb + in.color * emissive;

    // Grid Floor Shading
    if (abs(in.world_pos.y) < 0.02 && abs(N.y) > 0.9) {
        let coord = in.world_pos.xz;
        let grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
        let line = min(grid.x, grid.y);
        let grid_val = 1.0 - min(line, 1.0);

        // Major grid lines every 5 units
        let grid_major = abs(fract(coord * 0.2 - 0.5) - 0.5) / fwidth(coord * 0.2);
        let line_major = min(grid_major.x, grid_major.y);
        let major_val = 1.0 - min(line_major, 1.0);

        var grid_color = vec3<f32>(0.08, 0.10, 0.14);
        grid_color += vec3<f32>(0.25, 0.28, 0.35) * grid_val * 0.4;
        grid_color += vec3<f32>(0.4, 0.6, 0.9) * major_val * 0.6;

        // X and Z axes colored lines
        if (abs(in.world_pos.z) < 0.05) { grid_color = vec3<f32>(0.9, 0.2, 0.2); } // X Axis (Red)
        if (abs(in.world_pos.x) < 0.05) { grid_color = vec3<f32>(0.2, 0.4, 0.9); } // Z Axis (Blue)

        final_color = mix(grid_color, final_color, 0.15);
    }

    // Selection Highlight (Glowing Golden Amber Rim)
    let selected_id = scene.game_data.z;
    if (abs(object_id - selected_id) < 0.1 && selected_id > 0.0) {
        let fresnel_rim = pow(1.0 - NdotV, 3.0);
        let selection_pulse = sin(scene.game_data.x * 6.0) * 0.2 + 0.8;
        let selection_color = vec3<f32>(1.0, 0.7, 0.2) * selection_pulse;
        final_color += selection_color * (fresnel_rim * 1.8 + 0.3);
    }

    // Tonemap & Gamma Correction
    let mapped = aces_tonemap(final_color);
    return vec4<f32>(mapped, 1.0);
}
