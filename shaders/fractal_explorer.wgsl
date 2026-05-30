// ─── Mandelbulb Fractal Explorer ───────────────────────────────────────────
// Ray-marched 3D fractal with PBR lighting, orbit trap coloring,
// glow, AO, film grain, and 5 color palettes.

const PI: f32 = 3.14159265359;

struct FractalUniform {
    camera_pos: vec4<f32>,      // xyz=position, w=fov
    camera_target: vec4<f32>,   // xyz=look-at, w=zoom
    resolution: vec4<f32>,      // x=width, y=height, z=time, w=power
    params: vec4<f32>,          // x=palette, y=audio, z=max_iter, w=warp
}

@group(0) @binding(0) var<uniform> u: FractalUniform;

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

// ── Noise ─────────────────────────────────────────────────────────────────

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ── IQ Cosine Palettes ────────────────────────────────────────────────────

fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

fn get_palette(t: f32, idx: u32) -> vec3<f32> {
    if (idx == 0u) {
        // Fire
        return pal(t, vec3<f32>(0.5,0.5,0.5), vec3<f32>(0.5,0.5,0.5),
            vec3<f32>(1.0,0.7,0.4), vec3<f32>(0.0,0.15,0.20));
    } else if (idx == 1u) {
        // Ocean
        return pal(t, vec3<f32>(0.5,0.5,0.5), vec3<f32>(0.5,0.5,0.5),
            vec3<f32>(1.0,1.0,1.0), vec3<f32>(0.0,0.33,0.67));
    } else if (idx == 2u) {
        // Nebula
        return pal(t, vec3<f32>(0.5,0.5,0.5), vec3<f32>(0.5,0.5,0.5),
            vec3<f32>(2.0,1.0,0.0), vec3<f32>(0.5,0.2,0.25));
    } else if (idx == 3u) {
        // Earth
        return pal(t, vec3<f32>(0.5,0.5,0.5), vec3<f32>(0.5,0.5,0.5),
            vec3<f32>(1.0,1.0,0.5), vec3<f32>(0.8,0.9,0.30));
    } else {
        // Monochrome
        let v = 0.3 + t * 0.6;
        return vec3<f32>(v * 0.9, v * 0.95, v);
    }
}

// ── Mandelbulb SDF ────────────────────────────────────────────────────────

struct BulbResult {
    dist: f32,
    orbit_trap: f32,     // min distance to origin during iteration
    iterations: f32,     // normalized iteration count
}

fn mandelbulb(p: vec3<f32>, power: f32, max_iter: i32) -> BulbResult {
    var z = p;
    var dr = 1.0;
    var r = 0.0;
    var trap = 1e10;
    var i = 0;

    for (; i < max_iter; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        // Track orbit trap (min distance to planes)
        trap = min(trap, length(z.xz));
        trap = min(trap, abs(z.y));

        // Spherical coordinates
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // Scale and rotate
        let zr = pow(r, power);
        let t = theta * power;
        let p2 = phi * power;
        z = zr * vec3<f32>(sin(t) * cos(p2), sin(p2) * sin(t), cos(t));
        z += p;
    }

    var res: BulbResult;
    res.dist = 0.5 * log(r) * r / dr;
    res.orbit_trap = trap;
    res.iterations = f32(i) / f32(max_iter);
    return res;
}

// ── Ray Marching ──────────────────────────────────────────────────────────

struct MarchResult {
    hit: bool,
    pos: vec3<f32>,
    dist: f32,
    steps: f32,
    trap: f32,
    iter_ratio: f32,
    glow: f32,
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>, power: f32, max_iter: i32) -> MarchResult {
    var res: MarchResult;
    res.hit = false;
    res.glow = 0.0;

    var t = 0.0;
    let max_t = 20.0;
    let eps = 0.0005;
    let max_steps = 120;

    for (var i = 0; i < max_steps; i++) {
        let p = ro + rd * t;
        let bulb = mandelbulb(p, power, max_iter);

        // Accumulate glow from near misses
        res.glow += 0.1 / (1.0 + bulb.dist * bulb.dist * 500.0);

        if (bulb.dist < eps) {
            res.hit = true;
            res.pos = p;
            res.dist = t;
            res.steps = f32(i) / f32(max_steps);
            res.trap = bulb.orbit_trap;
            res.iter_ratio = bulb.iterations;
            return res;
        }

        t += bulb.dist * 0.8; // slight under-step for safety
        if (t > max_t) { break; }
    }

    res.dist = t;
    return res;
}

// ── Normal via central differences ────────────────────────────────────────

fn calc_normal(p: vec3<f32>, power: f32, max_iter: i32) -> vec3<f32> {
    let e = 0.0005;
    let dx = mandelbulb(p + vec3<f32>(e,0.0,0.0), power, max_iter).dist
           - mandelbulb(p - vec3<f32>(e,0.0,0.0), power, max_iter).dist;
    let dy = mandelbulb(p + vec3<f32>(0.0,e,0.0), power, max_iter).dist
           - mandelbulb(p - vec3<f32>(0.0,e,0.0), power, max_iter).dist;
    let dz = mandelbulb(p + vec3<f32>(0.0,0.0,e), power, max_iter).dist
           - mandelbulb(p - vec3<f32>(0.0,0.0,e), power, max_iter).dist;
    return normalize(vec3<f32>(dx, dy, dz));
}

// ── Soft shadow ───────────────────────────────────────────────────────────

fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, power: f32, max_iter: i32) -> f32 {
    var t = 0.01;
    var res = 1.0;
    for (var i = 0; i < 32; i++) {
        let d = mandelbulb(ro + rd * t, power, max_iter).dist;
        res = min(res, 8.0 * d / t);
        t += clamp(d, 0.005, 0.5);
        if (t > 4.0 || d < 0.0001) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

// ── Camera ray ────────────────────────────────────────────────────────────

fn camera_ray(uv: vec2<f32>, ro: vec3<f32>, ta: vec3<f32>, fov: f32) -> vec3<f32> {
    let ww = normalize(ta - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = cross(uu, ww);
    return normalize(uv.x * uu + uv.y * vv + fov * ww);
}

// ── ACES Tonemapping ──────────────────────────────────────────────────────

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Fragment ──────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let res = vec2<f32>(u.resolution.x, u.resolution.y);
    let time = u.resolution.z;
    let power = u.resolution.w;
    let palette_idx = u32(u.params.x);
    let audio = u.params.y;
    let max_iter = i32(u.params.z);
    let warp = u.params.w;

    // Aspect-corrected UV
    let aspect = res.x / res.y;
    var uv = in.uv;
    uv.x *= aspect;

    // Camera
    let ro = u.camera_pos.xyz;
    let ta = u.camera_target.xyz;
    let fov = u.camera_pos.w;
    let rd = camera_ray(uv, ro, ta, fov);

    // Audio-reactive power pulse
    let eff_power = power + audio * sin(time * 2.0) * 0.3;

    // March
    let result = ray_march(ro, rd, eff_power, max_iter);

    var color = vec3<f32>(0.0);

    if (result.hit) {
        let p = result.pos;
        let N = calc_normal(p, eff_power, max_iter);

        // ── Coloring from orbit trap + iterations ──────────────
        let trap_color = get_palette(result.trap * 2.0, palette_idx);
        let iter_color = get_palette(result.iter_ratio, palette_idx);
        let base_color = mix(trap_color, iter_color, 0.4);

        // ── Lighting ───────────────────────────────────────────
        let light_dir = normalize(vec3<f32>(0.6, 0.8, -0.4));
        let light2_dir = normalize(vec3<f32>(-0.3, 0.5, 0.7));
        let V = normalize(ro - p);
        let H = normalize(V + light_dir);

        let NdL = max(dot(N, light_dir), 0.0);
        let NdL2 = max(dot(N, light2_dir), 0.0);
        let NdH = max(dot(N, H), 0.0);

        // Soft shadow
        let shad = soft_shadow(p + N * 0.002, light_dir, eff_power, max_iter);

        // Diffuse
        let diff = NdL * shad * 0.8 + NdL2 * 0.25;

        // Specular (Blinn-Phong)
        let spec = pow(NdH, 32.0) * shad * 0.5;

        // AO from steps
        let ao = 1.0 - result.steps * 0.7;

        // Fresnel rim
        let fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);
        let rim = fresnel * 0.15;

        // Combine
        let ambient = base_color * 0.15;
        color = ambient + base_color * diff * ao + vec3<f32>(1.0) * spec + rim * base_color;

        // Distance fade
        let fade = exp(-result.dist * 0.15);
        color *= fade;

    } else {
        // ── Background: dark with subtle nebula ────────────────
        let bg_t = length(in.uv) * 0.3;
        color = get_palette(bg_t + time * 0.02, palette_idx) * 0.03;
    }

    // ── Glow ──────────────────────────────────────────────────
    let glow_color = get_palette(0.5 + time * 0.05, palette_idx);
    color += glow_color * result.glow * 0.04;

    // ── Vignette ──────────────────────────────────────────────
    let vig = 1.0 - dot(in.uv * 0.5, in.uv * 0.5) * 0.4;
    color *= vig;

    // ── Tonemap + gamma ───────────────────────────────────────
    color = aces(color * 2.5);
    color = pow(color, vec3<f32>(1.0 / 2.2));

    // ── Film grain ────────────────────────────────────────────
    let grain = hash21(in.clip_pos.xy * 0.5 + vec2<f32>(time * 97.0, 0.0));
    color += (grain - 0.5) * 0.015;

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
