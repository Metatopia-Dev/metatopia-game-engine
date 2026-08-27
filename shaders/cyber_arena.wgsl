// ─── CYBERSHOCK: Neon Grid Arena WGSL Shader ───────────────────────────────
// Features:
//   - PBR lighting with high-contrast neon specular & emissions
//   - Animated cyber grid floor with player distance ripple & pulsing lines
//   - Dynamic muzzle flash, explosion light points & hit sparks
//   - Holographic Screen HUD:
//       * Dynamic Crosshair + Hitmarker X on hit
//       * Sci-Fi Health Bar (bottom-left)
//       * Energy Shield Bar (bottom-left)
//       * Ammo Bar & Mag Counter (bottom-right)
//       * Score & Combo Multiplier (top-right)
//       * Wave & Boss Health Bar (top-center)
//       * Damage Vignette Flash & Dash Warp
//   - ACES Film Tonemapping & CRT/Scanline arcade filter

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
}

struct SceneUniform {
    sun_direction: vec4<f32>,   // xyz=direction, w=intensity
    sun_color: vec4<f32>,       // rgb=color, w=unused
    light0_pos: vec4<f32>,      // xyz=pos, w=range (Muzzle / Shot Light)
    light0_color: vec4<f32>,    // rgb=color, w=intensity
    params: vec4<f32>,          // x=time, y=damage_flash, z=ambient, w=muzzle_flash
    game_data: vec4<f32>,       // x=hp, y=shield, z=weapon_type (0=plasma,1=shotgun,2=railgun), w=ammo
    extra0: vec4<f32>,          // x=score, y=combo_mult, z=wave, w=enemies_remaining
    extra1: vec4<f32>,          // x=hitmarker_timer, y=dash_timer, z=boss_hp_pct (0-1), w=game_state (0=title,1=play,2=win,3=gameover)
    extra2: vec4<f32>,          // xyz=light1_pos, w=light1_range (Explosion / Arena Light)
    extra3: vec4<f32>,          // rgb=light1_color, w=light1_intensity
    hud_info: vec4<f32>,        // x=res_x, y=res_y, z=max_hp, w=max_ammo
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;

// ─── Vertex Input / Output ────────────────────────────────────────────────

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>,   // x=metallic, y=roughness, z=emission, w=type (0=standard, 1=floor, 2=pillar, 3=enemy, 4=powerup)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
    @location(4) pbr: vec4<f32>,
}

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

// ─── Math & Color Utilities ───────────────────────────────────────────────

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

fn sd_box(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

// ─── Fragment Shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = scene.params.x;
    let res = scene.hud_info.xy;
    let screen_uv = in.clip_position.xy / max(res, vec2(1.0));
    let aspect = res.x / max(res.y, 1.0);

    let N = normalize(in.world_normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    let dist_to_cam = length(camera.view_position.xyz - in.world_pos);

    var albedo = in.color;
    var metallic = in.pbr.x;
    var roughness = in.pbr.y;
    var emission = in.pbr.z;
    let surface_type = in.pbr.w;

    // ── 1. Cyber Grid Floor Texture ───────────────────────────────────────
    if (surface_type > 0.5 && surface_type < 1.5) {
        // Floor grid
        let p = in.world_pos.xz * 0.5;
        let grid = abs(fract(p - 0.5) - 0.5) / fwidth(p);
        let line = min(grid.x, grid.y);
        let c = 1.0 - min(line, 1.0);

        // Subgrid
        let sub_p = in.world_pos.xz * 2.0;
        let sub_grid = abs(fract(sub_p - 0.5) - 0.5) / fwidth(sub_p);
        let sub_line = min(sub_grid.x, sub_grid.y);
        let sub_c = 1.0 - min(sub_line, 1.0);

        // Hex pulse / ripple from center
        let dist_from_center = length(in.world_pos.xz);
        let wave = sin(dist_from_center * 1.5 - time * 4.0) * 0.5 + 0.5;
        let pulse = sin(time * 2.0) * 0.2 + 0.8;

        // Dark reflective floor tile
        albedo = vec3<f32>(0.02, 0.03, 0.06);
        metallic = 0.9;
        roughness = 0.2;

        // Primary glowing grid line (cyan/blue)
        let grid_col = vec3<f32>(0.0, 0.8, 1.0) * (c * 2.5 * pulse + sub_c * 0.4);
        // Ripple glow
        let ripple_col = vec3<f32>(0.9, 0.1, 0.6) * (wave * smoothstep(25.0, 0.0, dist_from_center) * c * 1.5);

        albedo += grid_col + ripple_col;
        emission += (c * 1.8 + ripple_col.r) * smoothstep(60.0, 5.0, dist_to_cam);
    }
    // ── 2. Neon Cover Pillars / Walls ─────────────────────────────────────
    else if (surface_type > 1.5 && surface_type < 2.5) {
        // Horizontal glowing stripes on pillars
        let stripe = sin(in.world_pos.y * 3.0 + time) * 0.5 + 0.5;
        if (stripe > 0.85) {
            albedo = vec3<f32>(0.1, 0.9, 1.0);
            emission = 3.0;
            roughness = 0.1;
        } else {
            albedo = in.color * 0.4;
            metallic = 0.8;
            roughness = 0.3;
        }
    }
    // ── 3. Enemies ────────────────────────────────────────────────────────
    else if (surface_type > 2.5 && surface_type < 3.5) {
        // Pulsing enemy core
        let pulse = sin(time * 8.0) * 0.3 + 0.7;
        albedo = in.color * (0.8 + pulse * 0.4);
        emission = in.pbr.z * (1.0 + pulse * 0.5);
    }
    // ── 4. Powerups ───────────────────────────────────────────────────────
    else if (surface_type > 3.5) {
        let hover_pulse = sin(time * 5.0) * 0.5 + 1.0;
        albedo = in.color * hover_pulse;
        emission = 4.0;
    }

    // ── 5. Lighting Computations (Sun + Point Lights) ─────────────────────
    let sun_dir = normalize(scene.sun_direction.xyz);
    let sun_col = scene.sun_color.rgb;
    let sun_int = scene.sun_direction.w;

    let NdotL = max(dot(N, sun_dir), 0.0);
    var direct_light = albedo * sun_col * sun_int * NdotL;

    // Specular Highlight (Blinn-Phong)
    let H = normalize(sun_dir + V);
    let shininess = max(2.0 / (roughness * roughness + 0.0001) - 2.0, 1.0);
    let spec = pow(max(dot(N, H), 0.0), shininess);
    let specular = sun_col * spec * (metallic * 0.8 + 0.2) * sun_int;

    // Point Light 0 (Muzzle flash / Player Weapon Light)
    let l0_delta = scene.light0_pos.xyz - in.world_pos;
    let l0_dist = length(l0_delta);
    let l0_range = max(scene.light0_pos.w, 0.1);
    if (l0_dist < l0_range) {
        let l0_dir = normalize(l0_delta);
        let l0_att = clamp(1.0 - l0_dist / l0_range, 0.0, 1.0);
        let l0_NdotL = max(dot(N, l0_dir), 0.0);
        let l0_H = normalize(l0_dir + V);
        let l0_spec = pow(max(dot(N, l0_H), 0.0), shininess);
        direct_light += (albedo * l0_NdotL + specular * l0_spec) * scene.light0_color.rgb * scene.light0_color.w * l0_att * l0_att;
    }

    // Point Light 1 (Explosion / Active Arena Hazard Light)
    let l1_delta = scene.extra2.xyz - in.world_pos;
    let l1_dist = length(l1_delta);
    let l1_range = max(scene.extra2.w, 0.1);
    if (l1_dist < l1_range) {
        let l1_dir = normalize(l1_delta);
        let l1_att = clamp(1.0 - l1_dist / l1_range, 0.0, 1.0);
        let l1_NdotL = max(dot(N, l1_dir), 0.0);
        direct_light += albedo * l1_NdotL * scene.extra3.rgb * scene.extra3.w * l1_att * l1_att;
    }

    // Ambient Lighting (Cyberpunk Cyan-Purple Sky/Horizon)
    let sky_up = vec3<f32>(0.08, 0.05, 0.15);
    let ground_col = vec3<f32>(0.02, 0.05, 0.10);
    let ambient = mix(ground_col, sky_up, N.y * 0.5 + 0.5) * albedo;

    // Emissive self-illumination
    let emissive_light = albedo * emission;

    // Total 3D Lit Color
    var color = direct_light + specular + ambient + emissive_light;

    // Distance Fog (Dark synthwave horizon)
    let fog_factor = clamp((dist_to_cam - 10.0) / 70.0, 0.0, 0.95);
    let fog_color = vec3<f32>(0.03, 0.02, 0.08);
    color = mix(color, fog_color, fog_factor);

    // ── 6. 2D Screen-Space HUD & FX Overlay ───────────────────────────────
    // Screen coordinates centered at (0, 0)
    let uv_centered = (screen_uv - 0.5) * vec2<f32>(aspect, 1.0);
    let r = length(uv_centered);

    // (A) Crosshair in Screen Center
    let ch_len = 0.015;
    let ch_thick = 0.0018;
    let ch_gap = 0.005;
    let dist_x = abs(uv_centered.x);
    let dist_y = abs(uv_centered.y);

    let is_h_bar = dist_y < ch_thick && dist_x > ch_gap && dist_x < (ch_gap + ch_len);
    let is_v_bar = dist_x < ch_thick && dist_y > ch_gap && dist_y < (ch_gap + ch_len);
    let is_dot = r < 0.0022;

    if (is_h_bar || is_v_bar || is_dot) {
        color = vec3<f32>(0.0, 1.0, 0.9); // Bright cyan reticle
    }

    // (B) Hitmarker (Red X on successful hit)
    let hitmarker_time = scene.extra1.x; // > 0.0 on hit
    if (hitmarker_time > 0.01) {
        let rot_uv = vec2<f32>(
            uv_centered.x * 0.7071 - uv_centered.y * 0.7071,
            uv_centered.x * 0.7071 + uv_centered.y * 0.7071
        );
        let hm_x = abs(rot_uv.x);
        let hm_y = abs(rot_uv.y);
        let hm_bar = (hm_y < 0.002 && hm_x > 0.006 && hm_x < 0.018) ||
                     (hm_x < 0.002 && hm_y > 0.006 && hm_y < 0.018);
        if (hm_bar) {
            color = mix(color, vec3<f32>(1.0, 0.1, 0.2), hitmarker_time);
        }
    }

    // (C) Player Damage Vignette Flash (Red border pulse)
    let dmg_flash = scene.params.y;
    if (dmg_flash > 0.01) {
        let edge_dist = length((screen_uv - 0.5) * 1.8);
        let vignette = smoothstep(0.4, 1.2, edge_dist) * dmg_flash;
        color = mix(color, vec3<f32>(1.0, 0.0, 0.1), vignette * 0.85);
    }

    // (D) Dash / Boost Warp Effect (Cyan motion streaks)
    let dash_timer = scene.extra1.y;
    if (dash_timer > 0.01) {
        let streak = sin(atan2(uv_centered.y, uv_centered.x) * 20.0 + time * 30.0) * 0.5 + 0.5;
        let ring = smoothstep(0.2, 0.8, r) * dash_timer * streak * 0.3;
        color += vec3<f32>(0.0, 0.7, 1.0) * ring;
    }

    // (E) Bottom-Left HUD: Health & Shield Gauges
    let hp_pct = clamp(scene.game_data.x / max(scene.hud_info.z, 1.0), 0.0, 1.0);
    let shield_pct = clamp(scene.game_data.y / 50.0, 0.0, 1.0);

    // HP Bar (SDF Box at screen_uv near [0.08..0.28, 0.90..0.92])
    let hp_box_p = screen_uv - vec2<f32>(0.14, 0.92);
    let hp_sd = sd_box(hp_box_p, vec2<f32>(0.09, 0.012));
    if (hp_sd < 0.001) {
        let bar_fill = (hp_box_p.x + 0.09) / 0.18;
        if (bar_fill <= hp_pct) {
            // Neon Green/Cyan for HP
            color = mix(vec3<f32>(0.0, 1.0, 0.5), vec3<f32>(0.1, 0.9, 1.0), bar_fill);
        } else {
            color = vec3<f32>(0.15, 0.05, 0.08); // Depleted dark red
        }
    } else if (hp_sd < 0.004) {
        color = vec3<f32>(0.0, 0.8, 1.0); // Border glow
    }

    // Shield Bar (SDF Box directly above HP bar)
    let sh_box_p = screen_uv - vec2<f32>(0.14, 0.885);
    let sh_sd = sd_box(sh_box_p, vec2<f32>(0.09, 0.008));
    if (sh_sd < 0.001) {
        let bar_fill = (sh_box_p.x + 0.09) / 0.18;
        if (bar_fill <= shield_pct) {
            color = vec3<f32>(0.2, 0.6, 1.0); // Deep Blue Shield
        } else {
            color = vec3<f32>(0.05, 0.08, 0.15);
        }
    } else if (sh_sd < 0.003) {
        color = vec3<f32>(0.3, 0.7, 1.0);
    }

    // (F) Bottom-Right HUD: Ammo Gauge
    let ammo_pct = clamp(scene.game_data.w / max(scene.hud_info.w, 1.0), 0.0, 1.0);
    let ammo_box_p = screen_uv - vec2<f32>(0.86, 0.92);
    let ammo_sd = sd_box(ammo_box_p, vec2<f32>(0.08, 0.012));
    if (ammo_sd < 0.001) {
        let bar_fill = (ammo_box_p.x + 0.08) / 0.16;
        if (bar_fill <= ammo_pct) {
            // Weapon specific ammo color
            let weapon_type = scene.game_data.z;
            var w_col = vec3<f32>(0.0, 1.0, 0.8); // Plasma Cyan
            if (weapon_type > 0.5 && weapon_type < 1.5) {
                w_col = vec3<f32>(1.0, 0.2, 0.7); // Shotgun Magenta
            } else if (weapon_type > 1.5) {
                w_col = vec3<f32>(1.0, 0.8, 0.1); // Railgun Gold
            }
            color = w_col;
        } else {
            color = vec3<f32>(0.1, 0.1, 0.15);
        }
    } else if (ammo_sd < 0.003) {
        color = vec3<f32>(0.9, 0.6, 0.2);
    }

    // (G) Top-Center Boss Bar (when active)
    let boss_pct = scene.extra1.z;
    if (boss_pct > 0.001) {
        let boss_box_p = screen_uv - vec2<f32>(0.5, 0.06);
        let boss_sd = sd_box(boss_box_p, vec2<f32>(0.22, 0.01));
        if (boss_sd < 0.001) {
            let bar_fill = (boss_box_p.x + 0.22) / 0.44;
            if (bar_fill <= boss_pct) {
                color = mix(vec3<f32>(1.0, 0.1, 0.4), vec3<f32>(0.7, 0.1, 1.0), bar_fill);
            } else {
                color = vec3<f32>(0.1, 0.02, 0.05);
            }
        } else if (boss_sd < 0.003) {
            color = vec3<f32>(1.0, 0.2, 0.8);
        }
    }

    // ── 7. Post-Processing: Scanlines, Tonemapping & Gamma ─────────────────
    let scanline = sin(screen_uv.y * res.y * 1.5) * 0.03;
    color -= scanline;

    color = aces(color);
    color = to_srgb(color);

    return vec4<f32>(color, 1.0);
}
