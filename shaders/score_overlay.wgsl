// ─── Score Overlay — SDF 7-Segment Digit Renderer ──────────────────────────
// Reusable WGSL functions for rendering score HUDs.
// Copy the needed functions into your game shader's fragment stage.
//
// Usage:
//   let overlay = score_overlay(screen_uv, resolution, score, level, combo, time_left);
//   color = mix(color, overlay.rgb, overlay.a);

// ── SDF Primitives ─────────────────────────────────────────────────────────

fn sdf_box(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdf_segment_h(p: vec2<f32>, w: f32, t: f32) -> f32 {
    return sdf_box(p, vec2<f32>(w, t));
}

fn sdf_segment_v(p: vec2<f32>, h: f32, t: f32) -> f32 {
    return sdf_box(p, vec2<f32>(t, h));
}

// ── 7-Segment Digit ────────────────────────────────────────────────────────
// Renders digit 0–9 at origin. Digit cell is ~1.0 wide × 1.8 tall.
//   Segment layout:
//     ─A─
//    |   |
//    F   B
//    |   |
//     ─G─
//    |   |
//    E   C
//    |   |
//     ─D─

fn segment_mask(digit: u32) -> u32 {
    // Bitmask: bit0=A, bit1=B, bit2=C, bit3=D, bit4=E, bit5=F, bit6=G
    // 0=0x3F, 1=0x06, 2=0x5B, 3=0x4F, 4=0x66, 5=0x6D, 6=0x7D, 7=0x07, 8=0x7F, 9=0x6F
    var masks = array<u32, 10>(
        0x3Fu, 0x06u, 0x5Bu, 0x4Fu, 0x66u,
        0x6Du, 0x7Du, 0x07u, 0x7Fu, 0x6Fu
    );
    if (digit > 9u) { return 0x00u; }
    return masks[digit];
}

fn render_digit(p: vec2<f32>, digit: u32) -> f32 {
    let sw = 0.32;  // segment half-width
    let sh = 0.06;  // segment half-thickness
    let vw = 0.06;  // vertical segment half-thickness
    let vh = 0.28;  // vertical segment half-height

    let mask = segment_mask(digit);
    var d = 999.0;

    // A: top horizontal
    if ((mask & 0x01u) != 0u) { d = min(d, sdf_segment_h(p - vec2<f32>(0.0, 0.7), sw, sh)); }
    // B: top-right vertical
    if ((mask & 0x02u) != 0u) { d = min(d, sdf_segment_v(p - vec2<f32>(0.3, 0.35), vh, vw)); }
    // C: bottom-right vertical
    if ((mask & 0x04u) != 0u) { d = min(d, sdf_segment_v(p - vec2<f32>(0.3, -0.35), vh, vw)); }
    // D: bottom horizontal
    if ((mask & 0x08u) != 0u) { d = min(d, sdf_segment_h(p - vec2<f32>(0.0, -0.7), sw, sh)); }
    // E: bottom-left vertical
    if ((mask & 0x10u) != 0u) { d = min(d, sdf_segment_v(p - vec2<f32>(-0.3, -0.35), vh, vw)); }
    // F: top-left vertical
    if ((mask & 0x20u) != 0u) { d = min(d, sdf_segment_v(p - vec2<f32>(-0.3, 0.35), vh, vw)); }
    // G: middle horizontal
    if ((mask & 0x40u) != 0u) { d = min(d, sdf_segment_h(p - vec2<f32>(0.0, 0.0), sw, sh)); }

    return smoothstep(0.02, -0.02, d);
}

// ── Multi-Digit Number ─────────────────────────────────────────────────────
// Renders a number right-aligned. Each digit cell is 0.9 units wide.

fn render_number(p: vec2<f32>, value: u32, max_digits: u32) -> f32 {
    var v = value;
    var result = 0.0;
    let spacing = 0.9;

    for (var i = 0u; i < max_digits; i++) {
        let digit = v % 10u;
        v = v / 10u;
        let offset = f32(max_digits - 1u - i) * spacing;
        let dp = p - vec2<f32>(offset, 0.0);
        result = max(result, render_digit(dp, digit));
        if (v == 0u && i > 0u) { break; }
    }
    return result;
}

// ── Colon Separator ────────────────────────────────────────────────────────

fn render_colon(p: vec2<f32>) -> f32 {
    let d1 = length(p - vec2<f32>(0.0, 0.25)) - 0.07;
    let d2 = length(p - vec2<f32>(0.0, -0.25)) - 0.07;
    return smoothstep(0.02, -0.02, min(d1, d2));
}

// ── Score HUD Overlay ──────────────────────────────────────────────────────
// Call from fragment shader. Returns vec4 (rgb, alpha).
// screen_uv: normalized [0,1] screen coordinates (origin top-left)
// resolution: screen resolution in pixels

fn score_hud_pest(screen_uv: vec2<f32>, aspect: f32, score: f32, level: f32, pests: f32, time_left: f32, combo: f32) -> vec4<f32> {
    // HUD region: top-right corner
    let hud_x = screen_uv.x;
    let hud_y = 1.0 - screen_uv.y; // flip so y goes up

    // Panel bounds (top-right)
    let panel_right = 0.99;
    let panel_top = 0.99;
    let panel_w = 0.38;
    let panel_h = 0.14;

    let panel_left = panel_right - panel_w;
    let panel_bottom = panel_top - panel_h;

    // Check if we're in the panel region
    if (hud_x < panel_left || hud_x > panel_right || hud_y < panel_bottom || hud_y > panel_top) {
        return vec4<f32>(0.0);
    }

    // Normalized coordinates within panel [0,1]
    let px = (hud_x - panel_left) / panel_w;
    let py = (hud_y - panel_bottom) / panel_h;

    // Background
    let bg_alpha = 0.55;
    let bg_color = vec3<f32>(0.02, 0.02, 0.04);

    // Rounded corners
    let corner_r = 0.06;
    let cp = vec2<f32>(px, py) - 0.5;
    let corner_d = sdf_box(cp, vec2<f32>(0.5 - corner_r, 0.5 - corner_r));
    if (corner_d > corner_r) { return vec4<f32>(0.0); }

    var color = bg_color;
    var alpha = bg_alpha;

    // Score display: top row
    // "SCORE" label area (left) + number (right)
    let score_val = u32(score);
    let digit_scale = 9.0;
    let score_p = (vec2<f32>(px, py) - vec2<f32>(0.55, 0.70)) * digit_scale;
    let score_d = render_number(score_p, score_val, 6u);

    // Combo indicator
    var text_color = vec3<f32>(0.9, 0.85, 0.7);
    if (combo > 1.5) {
        // Combo active — gold/orange glow
        text_color = vec3<f32>(1.0, 0.8, 0.2);
    }

    if (score_d > 0.0) {
        color = text_color;
        alpha = score_d;
    }

    // Level display: bottom-left
    let level_val = u32(level);
    let level_p = (vec2<f32>(px, py) - vec2<f32>(0.12, 0.28)) * digit_scale * 1.1;
    let level_d = render_number(level_p, level_val, 2u);
    if (level_d > 0.0) {
        color = vec3<f32>(0.5, 0.8, 1.0);
        alpha = level_d;
    }

    // Timer display: bottom-right
    let time_val = u32(max(time_left, 0.0));
    let minutes = time_val / 60u;
    let seconds = time_val % 60u;
    let min_p = (vec2<f32>(px, py) - vec2<f32>(0.58, 0.28)) * digit_scale * 1.1;
    let min_d = render_number(min_p, minutes, 1u);
    let colon_p = (vec2<f32>(px, py) - vec2<f32>(0.70, 0.28)) * digit_scale * 1.1;
    let colon_d = render_colon(colon_p);
    let sec_p = (vec2<f32>(px, py) - vec2<f32>(0.82, 0.28)) * digit_scale * 1.1;
    let sec_d = render_number(sec_p, seconds, 2u);

    var timer_color = vec3<f32>(0.7, 0.9, 0.7);
    if (time_left < 30.0) { timer_color = vec3<f32>(1.0, 0.5, 0.2); }
    if (time_left < 10.0) { timer_color = vec3<f32>(1.0, 0.2, 0.2); }

    let timer_d = max(max(min_d, sec_d), colon_d);
    if (timer_d > 0.0) {
        color = timer_color;
        alpha = timer_d;
    }

    // Pests alive: bottom-center
    let pests_val = u32(pests);
    let pests_p = (vec2<f32>(px, py) - vec2<f32>(0.36, 0.28)) * digit_scale * 1.1;
    let pests_d = render_number(pests_p, pests_val, 2u);
    if (pests_d > 0.0) {
        color = vec3<f32>(1.0, 0.6, 0.5);
        alpha = pests_d;
    }

    // Combo multiplier badge (if combo > 1)
    if (combo > 1.5) {
        let combo_val = u32(min(combo, 5.0));
        let combo_p = (vec2<f32>(px, py) - vec2<f32>(0.12, 0.70)) * digit_scale * 0.9;
        let combo_d = render_number(combo_p, combo_val, 1u);
        if (combo_d > 0.0) {
            color = vec3<f32>(1.0, 0.6, 0.1);
            alpha = combo_d;
        }
    }

    return vec4<f32>(color, alpha);
}

// ── Collection HUD (for basic_game) ────────────────────────────────────────

fn score_hud_collect(screen_uv: vec2<f32>, collected: f32, total: f32) -> vec4<f32> {
    let hud_x = screen_uv.x;
    let hud_y = 1.0 - screen_uv.y;

    // Compact panel top-right
    let panel_right = 0.99;
    let panel_top = 0.97;
    let panel_w = 0.18;
    let panel_h = 0.08;

    let panel_left = panel_right - panel_w;
    let panel_bottom = panel_top - panel_h;

    if (hud_x < panel_left || hud_x > panel_right || hud_y < panel_bottom || hud_y > panel_top) {
        return vec4<f32>(0.0);
    }

    let px = (hud_x - panel_left) / panel_w;
    let py = (hud_y - panel_bottom) / panel_h;

    // Rounded background
    let corner_r = 0.08;
    let cp = vec2<f32>(px, py) - 0.5;
    let corner_d = sdf_box(cp, vec2<f32>(0.5 - corner_r, 0.5 - corner_r));
    if (corner_d > corner_r) { return vec4<f32>(0.0); }

    var color = vec3<f32>(0.02, 0.02, 0.04);
    var alpha = 0.55;

    // "N / M" display
    let digit_scale = 14.0;
    let coll_val = u32(collected);
    let total_val = u32(total);

    // Collected number (left)
    let coll_p = (vec2<f32>(px, py) - vec2<f32>(0.22, 0.50)) * digit_scale;
    let coll_d = render_number(coll_p, coll_val, 1u);

    // Slash separator
    // Simple diagonal line approximation
    let slash_p = (vec2<f32>(px, py) - vec2<f32>(0.48, 0.50)) * digit_scale;
    let slash_d = smoothstep(0.12, 0.0, abs(slash_p.x + slash_p.y * 0.4) + abs(slash_p.x - slash_p.y * 0.4) - 0.5);

    // Total number (right)
    let total_p = (vec2<f32>(px, py) - vec2<f32>(0.74, 0.50)) * digit_scale;
    let total_d = render_number(total_p, total_val, 1u);

    // Color based on progress
    var text_col = vec3<f32>(0.5, 0.8, 1.0);
    if (collected >= total && total > 0.5) {
        text_col = vec3<f32>(0.3, 1.0, 0.4); // All collected = green
    }

    let combined = max(max(coll_d, total_d), slash_d);
    if (combined > 0.0) {
        color = text_col;
        alpha = combined;
    }

    return vec4<f32>(color, alpha);
}
