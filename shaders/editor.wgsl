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

    if (in.pbr.w < -2.5) {
        // 3D Transform Gizmo overlay: project in 3D space with near-plane depth bias so it is never buried inside meshes
        let clip = camera.view_proj * vec4<f32>(in.position, 1.0);
        out.clip_position = vec4<f32>(clip.xy, clamp(clip.z * 0.02, 0.002 * clip.w, 0.99 * clip.w), clip.w);
    } else if (in.pbr.w < -1.5) {
        // 2D Screen UI Text: always in front of UI panels
        out.clip_position = vec4<f32>(in.position.xy, 0.0001, 1.0);
    } else if (in.pbr.w < -0.5) {
        // 2D Screen UI Panel: in front of 3D scene and gizmos
        out.clip_position = vec4<f32>(in.position.xy, 0.001, 1.0);
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

    // ── 2D UI Overlay Rendering (sRGB-authored colors converted to linear for Rgba8UnormSrgb target)
    if (object_id < -0.5 && object_id > -2.5) {
        let opacity = in.pbr.z;
        let ui_linear = pow(max(in.color, vec3<f32>(0.0)), vec3<f32>(2.2));
        return vec4<f32>(ui_linear, opacity);
    }

    // ── 3D Transform Gizmo Rendering (unlit vivid axes) ───────────────────
    if (object_id < -2.5) {
        let gizmo_lin = pow(max(in.color, vec3<f32>(0.0)), vec3<f32>(2.2));
        return vec4<f32>(gizmo_lin, 1.0);
    }

    // ── 3D Viewport Rendering ─────────────────────────────────────────────
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

    let diffuse = in.color * (1.0 - metallic) * NdotL;
    let ambient = in.color * vec3<f32>(0.18, 0.20, 0.26) * 0.45;

    var final_color = ambient + (diffuse + spec * F) * scene.sun_color.rgb + in.color * emissive;

    // Distance-Faded Infinite Grid Floor Shading
    if (abs(in.world_pos.y) < 0.02 && abs(N.y) > 0.9) {
        let dist = length(in.world_pos - camera.view_position.xyz);
        let line_fade = 1.0 - smoothstep(12.0, 42.0, dist);
        let horizon_fade = smoothstep(22.0, 52.0, dist);

        let coord = in.world_pos.xz;
        let grid = abs(fract(coord - 0.5) - 0.5) / max(fwidth(coord), vec2<f32>(0.0001));
        let line = min(grid.x, grid.y);
        let grid_val = (1.0 - min(line, 1.0)) * line_fade;

        let grid_major = abs(fract(coord * 0.2 - 0.5) - 0.5) / max(fwidth(coord * 0.2), vec2<f32>(0.0001));
        let line_major = min(grid_major.x, grid_major.y);
        let major_val = (1.0 - min(line_major, 1.0)) * line_fade;

        var grid_color = vec3<f32>(0.05, 0.065, 0.095);
        grid_color += vec3<f32>(0.14, 0.20, 0.28) * grid_val * 0.45;
        grid_color += vec3<f32>(0.0, 0.58, 0.85) * major_val * 0.55;

        if (abs(in.world_pos.z) < 0.045) {
            grid_color = mix(grid_color, vec3<f32>(0.90, 0.20, 0.22), line_fade);
        }
        if (abs(in.world_pos.x) < 0.045) {
            grid_color = mix(grid_color, vec3<f32>(0.20, 0.48, 0.95), line_fade);
        }

        let sky_srgb = vec3<f32>(0.05, 0.06, 0.09);
        final_color = mix(grid_color, sky_srgb, horizon_fade);
    }

    // Selection Highlight (Glowing Golden Amber Rim)
    let selected_id = scene.game_data.z;
    if (abs(object_id - selected_id) < 0.1 && selected_id > 0.0) {
        let fresnel_rim = pow(1.0 - NdotV, 2.5);
        let selection_pulse = sin(scene.game_data.x * 6.0) * 0.2 + 0.8;
        let selection_color = vec3<f32>(1.0, 0.72, 0.18) * selection_pulse;
        final_color += selection_color * (fresnel_rim * 1.4 + 0.22);
    }

    // ACES Tonemap -> convert to linear for hardware sRGB framebuffer
    let mapped_srgb = aces_tonemap(final_color);
    let mapped_linear = pow(max(mapped_srgb, vec3<f32>(0.0)), vec3<f32>(2.2));
    return vec4<f32>(mapped_linear, 1.0);
}
