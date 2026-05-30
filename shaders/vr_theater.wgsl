// ─── VR Netflix Hyperbolic Theater ─────────────────────────────────────────
// Fullscreen ray-marched cinema in 5 non-Euclidean theater spaces.
// PBR screens, Poincaré disk floor, procedural audio drones.

const PI: f32 = 3.14159265359;

struct Uniform {
    camera_pos: vec4<f32>,       // xyz=pos, w=fov
    camera_target: vec4<f32>,    // xyz=target, w=time
    resolution: vec4<f32>,       // x=w, y=h, z=space_type, w=selected_screen
    params: vec4<f32>,           // x=screen_count, y=ambient, z=highlights, w=transition
    screen0: vec4<f32>,          // xyz=position, w=playing
    screen1: vec4<f32>,
    screen2: vec4<f32>,
    screen3: vec4<f32>,
    screen4: vec4<f32>,
    screen5: vec4<f32>,
    screen6: vec4<f32>,
    screen7: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// ── Vertex ────────────────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0),
        vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,1.0),  vec2<f32>(-1.0,1.0)
    );
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = positions[idx];
    return out;
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Poincaré Disk Hyperbolic Tiling ───────────────────────────────────────

fn poincare_disk_pattern(p: vec2<f32>, time: f32) -> vec3<f32> {
    let r = length(p);
    if (r > 0.99) { return vec3<f32>(0.02); } // boundary
    
    // Hyperbolic distance from center
    let hyp_r = 2.0 * atanh(r);
    
    // Concentric hyperbolic rings
    let ring = sin(hyp_r * 4.0 - time * 0.3) * 0.5 + 0.5;
    
    // Angular sectors (7-fold symmetry for {7,3} tiling feel)
    let angle = atan2(p.y, p.x);
    let sector = sin(angle * 7.0 + time * 0.1) * 0.5 + 0.5;
    
    // Warp with hyperbolic metric
    let warp = 1.0 / (1.0 - r * r + 0.001);
    let detail = sin(hyp_r * 12.0 + angle * 3.0 - time * 0.5) * 0.5 + 0.5;
    
    // Colors
    let base = pal(ring * 0.3 + sector * 0.2,
        vec3<f32>(0.15, 0.10, 0.20),
        vec3<f32>(0.15, 0.12, 0.18),
        vec3<f32>(1.0, 0.8, 0.6),
        vec3<f32>(0.0, 0.15, 0.30));
    
    let edge = smoothstep(0.48, 0.50, fract(hyp_r * 2.0));
    let sector_edge = smoothstep(0.48, 0.50, fract(angle * 7.0 / 6.28318));
    let grid = max(edge, sector_edge) * 0.15;
    
    return base + grid * vec3<f32>(0.4, 0.3, 0.6);
}

// ── Spherical Dome Stars ──────────────────────────────────────────────────

fn star_field(rd: vec3<f32>, time: f32) -> vec3<f32> {
    var stars = vec3<f32>(0.0);
    // Layer 1: bright stars
    let p1 = rd * 50.0;
    let cell1 = floor(p1);
    let f1 = fract(p1) - 0.5;
    let h1 = hash31(cell1);
    if (h1 > 0.97) {
        let d = length(f1);
        let twinkle = sin(time * 3.0 + h1 * 100.0) * 0.3 + 0.7;
        stars += vec3<f32>(1.0, 0.95, 0.8) * smoothstep(0.15, 0.0, d) * twinkle;
    }
    // Layer 2: dim stars
    let p2 = rd * 120.0;
    let cell2 = floor(p2);
    let f2 = fract(p2) - 0.5;
    let h2 = hash31(cell2);
    if (h2 > 0.95) {
        let d = length(f2);
        stars += vec3<f32>(0.6, 0.7, 1.0) * smoothstep(0.1, 0.0, d) * 0.3;
    }
    return stars;
}

// ── Screen Rendering ──────────────────────────────────────────────────────

fn render_screen(ro: vec3<f32>, rd: vec3<f32>, screen_pos: vec3<f32>, screen_idx: f32,
                 selected: f32, playing: f32, time: f32) -> vec4<f32> {
    // Screen faces the camera (billboard)
    let to_screen = screen_pos - ro;
    let screen_dist = length(to_screen);
    let screen_dir = to_screen / screen_dist;
    
    // Project ray onto screen plane
    let t_hit = dot(to_screen, screen_dir) / dot(rd, screen_dir);
    if (t_hit < 0.0 || t_hit > 50.0) { return vec4<f32>(0.0); }
    
    let hit = ro + rd * t_hit;
    let local = hit - screen_pos;
    
    // Screen dimensions (16:9)
    let sw = 1.8;
    let sh = 1.0;
    
    // Build screen-local coordinate system
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(screen_dir, up));
    let sup = cross(right, screen_dir);
    
    let lu = dot(local, right);
    let lv = dot(local, sup);
    
    if (abs(lu) > sw || abs(lv) > sh) { return vec4<f32>(0.0); }
    
    // Screen UV (0-1)
    let suv = vec2<f32>((lu / sw) * 0.5 + 0.5, (lv / sh) * 0.5 + 0.5);
    
    // ── Movie poster / playback ──────────────────────────────
    var screen_color = vec3<f32>(0.0);
    
    if (playing > 0.5) {
        // Animated "video" - procedural movie simulation
        let movie_t = time * 0.5 + screen_idx * 2.0;
        let scene1 = pal(suv.x * 0.5 + movie_t * 0.1,
            vec3<f32>(0.5), vec3<f32>(0.5), vec3<f32>(1.0,0.7,0.4), vec3<f32>(0.0,0.15,0.2));
        let scene2 = pal(suv.y * 0.3 + movie_t * 0.15,
            vec3<f32>(0.5), vec3<f32>(0.5), vec3<f32>(2.0,1.0,0.0), vec3<f32>(0.5,0.2,0.25));
        let blend = sin(movie_t * 0.3) * 0.5 + 0.5;
        screen_color = mix(scene1, scene2, blend);
        
        // Scanlines
        let scanline = sin(suv.y * 200.0) * 0.03;
        screen_color += scanline;
    } else {
        // Movie poster - procedural
        let poster_hash = hash21(vec2<f32>(screen_idx * 7.0, 3.0));
        let gradient = pal(suv.y * 0.5 + poster_hash,
            vec3<f32>(0.3), vec3<f32>(0.4),
            vec3<f32>(1.0, 0.5 + poster_hash * 0.5, 0.3),
            vec3<f32>(0.0 + poster_hash * 0.3, 0.1, 0.2));
        
        // Title bar
        let title_band = smoothstep(0.15, 0.20, suv.y) * (1.0 - smoothstep(0.30, 0.35, suv.y));
        screen_color = gradient * 0.7 + vec3<f32>(0.8, 0.75, 0.6) * title_band * 0.5;
        
        // Rating stars
        if (suv.y > 0.05 && suv.y < 0.12) {
            let star_x = suv.x * 5.0;
            let star_rating = 3.0 + poster_hash * 2.0;
            if (star_x < star_rating) {
                screen_color += vec3<f32>(1.0, 0.85, 0.2) * 0.3;
            }
        }
    }
    
    // ── Screen frame (border) ────────────────────────────────
    let border = 0.03;
    let bx = smoothstep(0.0, border, suv.x) * smoothstep(0.0, border, 1.0 - suv.x);
    let by = smoothstep(0.0, border, suv.y) * smoothstep(0.0, border, 1.0 - suv.y);
    let frame_mask = bx * by;
    
    // Selection glow
    var glow = vec3<f32>(0.0);
    if (selected > 0.5) {
        let edge_glow = (1.0 - frame_mask) * 0.8;
        glow = vec3<f32>(0.3, 0.5, 1.0) * edge_glow * (sin(time * 3.0) * 0.3 + 0.7);
    }
    
    // Emissive screen with frame
    let final_color = screen_color * frame_mask * 1.5 + glow;
    
    // Distance attenuation
    let atten = 1.0 / (1.0 + screen_dist * screen_dist * 0.02);
    
    return vec4<f32>(final_color * atten, frame_mask * atten);
}

// ── Camera ────────────────────────────────────────────────────────────────

fn camera_ray(uv: vec2<f32>, ro: vec3<f32>, ta: vec3<f32>, fov: f32) -> vec3<f32> {
    let ww = normalize(ta - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = cross(uu, ww);
    return normalize(uv.x * uu + uv.y * vv + fov * ww);
}

// ── Scene SDF for theater room ────────────────────────────────────────────

fn theater_sdf(p: vec3<f32>, space: u32) -> f32 {
    if (space == 0u) {
        // Hyperbolic lobby: cylindrical room
        let cyl = length(p.xz) - 12.0;
        let floor_d = -p.y;
        let ceil_d = p.y - 6.0;
        return max(max(cyl, floor_d), ceil_d);
    } else if (space == 1u) {
        // Spherical dome
        return length(p) - 15.0;
    } else if (space == 2u) {
        // Escher: box room with warp
        let b = abs(p) - vec3<f32>(10.0, 5.0, 10.0);
        return max(max(b.x, b.y), b.z);
    } else if (space == 3u) {
        // Personal pocket: small cozy room
        let b = abs(p) - vec3<f32>(4.0, 3.0, 5.0);
        return max(max(b.x, b.y), b.z);
    } else {
        // Social hub
        return length(p) - 10.0;
    }
}

fn theater_normal(p: vec3<f32>, space: u32) -> vec3<f32> {
    let e = 0.001;
    let dx = theater_sdf(p + vec3<f32>(e,0.0,0.0), space) - theater_sdf(p - vec3<f32>(e,0.0,0.0), space);
    let dy = theater_sdf(p + vec3<f32>(0.0,e,0.0), space) - theater_sdf(p - vec3<f32>(0.0,e,0.0), space);
    let dz = theater_sdf(p + vec3<f32>(0.0,0.0,e), space) - theater_sdf(p - vec3<f32>(0.0,0.0,e), space);
    return normalize(vec3<f32>(dx, dy, dz));
}

// ── Fragment ──────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let res = vec2<f32>(u.resolution.x, u.resolution.y);
    let space = u32(u.resolution.z);
    let selected = u.resolution.w;
    let time = u.camera_target.w;
    let screen_count = u32(u.params.x);
    let ambient_level = u.params.y;
    let transition = u.params.w;

    let aspect = res.x / res.y;
    var uv = in.uv;
    uv.x *= aspect;

    let ro = u.camera_pos.xyz;
    let ta = u.camera_target.xyz;
    let fov = u.camera_pos.w;
    let rd = camera_ray(uv, ro, ta, fov);

    // ── Theater environment ──────────────────────────────────
    var env_color = vec3<f32>(0.0);

    if (space == 0u) {
        // Hyperbolic Lobby: dark purple + Poincaré disk floor
        env_color = vec3<f32>(0.03, 0.02, 0.06);
        
        // Floor ray intersection (y=0 plane)
        if (rd.y < -0.01) {
            let t_floor = -ro.y / rd.y;
            let floor_pos = ro + rd * t_floor;
            let fp = floor_pos.xz / 12.0; // normalize to unit disk
            if (length(fp) < 1.0) {
                env_color = poincare_disk_pattern(fp, time) * ambient_level;
                // Reflection glow
                env_color += vec3<f32>(0.05, 0.03, 0.08) * exp(-t_floor * 0.05);
            }
        }
        // Ceiling subtle glow
        if (rd.y > 0.1) {
            let ceiling_t = (6.0 - ro.y) / rd.y;
            if (ceiling_t > 0.0) {
                let cp = ro + rd * ceiling_t;
                let cr = length(cp.xz) / 12.0;
                env_color += vec3<f32>(0.04, 0.02, 0.08) * (1.0 - cr) * 0.5;
            }
        }
    } else if (space == 1u) {
        // Spherical Dome: starfield
        env_color = star_field(rd, time) + vec3<f32>(0.01, 0.01, 0.02);
        // Subtle nebula
        let nebula = pal(rd.y * 0.5 + 0.5 + time * 0.01,
            vec3<f32>(0.02), vec3<f32>(0.03),
            vec3<f32>(1.0, 0.5, 0.3), vec3<f32>(0.0, 0.3, 0.6));
        env_color += nebula * 0.3;
    } else if (space == 2u) {
        // Escher Theater: tessellated impossible pattern
        let p = ro + rd * 5.0;
        let grid = abs(sin(p.x * 2.0 + p.y * 1.5 + time * 0.2))
                 * abs(sin(p.z * 2.0 - p.y * 1.5 - time * 0.15));
        env_color = vec3<f32>(0.08, 0.06, 0.04) * (0.5 + grid * 0.5);
        // Penrose-like tiling hint
        let angle = atan2(rd.z, rd.x);
        let sector = fract(angle * 5.0 / 6.28318);
        env_color += vec3<f32>(0.03, 0.02, 0.01) * step(0.48, sector) * step(sector, 0.52);
    } else if (space == 3u) {
        // Personal Pocket: warm cozy
        env_color = vec3<f32>(0.06, 0.04, 0.02);
        // Warm walls
        if (rd.y < -0.01) {
            let t = -ro.y / rd.y;
            let fp = ro + rd * t;
            let checker = step(0.0, sin(fp.x * 2.0) * sin(fp.z * 2.0));
            env_color = mix(vec3<f32>(0.12, 0.08, 0.04), vec3<f32>(0.08, 0.05, 0.03), checker);
        }
    } else {
        // Social Hub: soft blue-purple
        env_color = vec3<f32>(0.04, 0.03, 0.06);
        env_color += star_field(rd, time) * 0.3;
    }

    // ── Render screens ───────────────────────────────────────
    var final_color = env_color;

    var screens = array<vec4<f32>, 8>(
        u.screen0, u.screen1, u.screen2, u.screen3,
        u.screen4, u.screen5, u.screen6, u.screen7
    );

    for (var i = 0u; i < screen_count; i++) {
        let s = screens[i];
        if (s.y < -50.0) { continue; } // hidden
        
        let is_selected = select(0.0, 1.0, f32(i) == selected);
        let is_playing = s.w;
        let screen_result = render_screen(ro, rd, s.xyz, f32(i), is_selected, is_playing, time);
        
        if (screen_result.w > 0.01) {
            final_color = mix(final_color, screen_result.xyz, screen_result.w);
        }
    }

    // ── Space transition flash ────────────────────────────────
    if (transition > 0.0) {
        final_color = mix(final_color, vec3<f32>(0.5, 0.3, 0.8), transition);
    }

    // ── Vignette ──────────────────────────────────────────────
    let vig = 1.0 - dot(in.uv * 0.4, in.uv * 0.4) * 0.5;
    final_color *= vig;

    // ── Tonemap ───────────────────────────────────────────────
    final_color = aces(final_color * 2.0);
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    // ── Film grain ────────────────────────────────────────────
    let grain = hash21(in.clip_pos.xy * 0.5 + vec2<f32>(time * 97.0, 0.0));
    final_color += (grain - 0.5) * 0.02;

    return vec4<f32>(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
