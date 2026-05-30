// ─── Pest Control Kitchen Shader ───────────────────────────────────────────
// PBR · Kitchen room · 8 dynamic pests · Crosshair overlay · Hit effects

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
}

struct SceneUniform {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    light0_pos: vec4<f32>,
    light0_color: vec4<f32>,
    light1_pos: vec4<f32>,
    light1_color: vec4<f32>,
    params: vec4<f32>,          // x=time, y=exposure, z=ambient, w=firing_flash
    game_info: vec4<f32>,       // x=score, y=level, z=pests_alive, w=time_left
    pest0: vec4<f32>,
    pest1: vec4<f32>,
    pest2: vec4<f32>,
    pest3: vec4<f32>,
    pest4: vec4<f32>,
    pest5: vec4<f32>,
    pest6: vec4<f32>,
    pest7: vec4<f32>,
    pest_flash: vec4<f32>,      // hit_flash 0-3
    pest_flash2: vec4<f32>,     // hit_flash 4-7
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> scene: SceneUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) pbr_params: vec4<f32>,  // x=metallic, y=roughness, z=ao, w=emissive
}

// ── Pest helpers ──────────────────────────────────────────────────────────

fn pest_radius(t: f32) -> f32 {
    if (t < 0.5) { return 0.0; }
    if (t < 1.5) { return 0.18; }  // cockroach
    if (t < 2.5) { return 0.10; }  // ant
    if (t < 3.5) { return 0.15; }  // spider
    if (t < 4.5) { return 0.35; }  // rat
    return 0.20;                    // wasp
}

fn pest_base_color(t: f32) -> vec3<f32> {
    if (t < 1.5) { return vec3<f32>(0.40, 0.22, 0.10); }  // cockroach: brown
    if (t < 2.5) { return vec3<f32>(0.12, 0.04, 0.02); }  // ant: dark red
    if (t < 3.5) { return vec3<f32>(0.18, 0.18, 0.22); }  // spider: dark gray
    if (t < 4.5) { return vec3<f32>(0.38, 0.28, 0.18); }  // rat: brown-gray
    return vec3<f32>(0.85, 0.75, 0.10);                     // wasp: yellow
}

fn pest_metallic(t: f32) -> f32 {
    if (t < 1.5) { return 0.6; }   // cockroach: shiny chitin
    if (t < 2.5) { return 0.3; }   // ant
    if (t < 3.5) { return 0.4; }   // spider
    if (t < 4.5) { return 0.1; }   // rat: fur
    return 0.5;                     // wasp: shiny
}

// ── Vertex Shader ─────────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let room = 6.0;
    var position  = vec3<f32>(0.0);
    var normal    = vec3<f32>(0.0, 1.0, 0.0);
    var base_color = vec3<f32>(0.5);
    var metallic  = 0.04;
    var roughness = 0.5;
    var ao        = 1.0;
    var emissive  = 0.0;
    let time = scene.params.x;

    var qv = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0),
        vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,1.0),  vec2<f32>(-1.0,1.0)
    );
    var oct = array<vec3<f32>, 6>(
        vec3<f32>(0.0,1.0,0.0), vec3<f32>(0.0,-1.0,0.0),
        vec3<f32>(1.0,0.0,0.0), vec3<f32>(-1.0,0.0,0.0),
        vec3<f32>(0.0,0.0,1.0), vec3<f32>(0.0,0.0,-1.0)
    );
    var fi = array<array<u32,3>,8>(
        array<u32,3>(0u,2u,4u), array<u32,3>(0u,4u,3u),
        array<u32,3>(0u,3u,5u), array<u32,3>(0u,5u,2u),
        array<u32,3>(1u,4u,2u), array<u32,3>(1u,3u,4u),
        array<u32,3>(1u,5u,3u), array<u32,3>(1u,2u,5u)
    );

    // ────────────────────────────────────────────────────────────────
    //  0..35: Room  │  36..227: 8 Pests  │  228..239: Crosshair
    // ────────────────────────────────────────────────────────────────

    if (idx < 36u) {
        // ── Kitchen Room ──────────────────────────────────────────
        let face = idx / 6u;
        let v = qv[idx % 6u];

        if (face == 0u) {
            // Floor – terracotta tile
            position = vec3<f32>(v.x * room, 0.0, v.y * room);
            normal = vec3<f32>(0.0, 1.0, 0.0);
            base_color = vec3<f32>(0.55, 0.40, 0.30);
            metallic = 0.02; roughness = 0.35;
        } else if (face == 1u) {
            // Ceiling – warm white
            position = vec3<f32>(v.x * room, 3.0, v.y * room);
            normal = vec3<f32>(0.0, -1.0, 0.0);
            base_color = vec3<f32>(0.88, 0.86, 0.82);
            roughness = 0.9;
        } else if (face == 2u) {
            // Back wall (+Z) – dark wood cabinets
            position = vec3<f32>(v.x * room, v.y * 1.5 + 1.5, room);
            normal = vec3<f32>(0.0, 0.0, -1.0);
            base_color = vec3<f32>(0.35, 0.25, 0.15);
            metallic = 0.05; roughness = 0.55;
        } else if (face == 3u) {
            // Front wall (-Z) – window wall, lighter
            position = vec3<f32>(v.x * room, v.y * 1.5 + 1.5, -room);
            normal = vec3<f32>(0.0, 0.0, 1.0);
            base_color = vec3<f32>(0.75, 0.72, 0.65);
            roughness = 0.7;
            // Window glow near center
            let wx = abs(v.x);
            let wy = v.y;
            if (wx < 0.5 && wy > -0.2) {
                emissive = 1.5;
                base_color = vec3<f32>(0.7, 0.8, 0.95);
            }
        } else if (face == 4u) {
            // Left wall (-X) – cream paint
            position = vec3<f32>(-room, v.y * 1.5 + 1.5, v.x * room);
            normal = vec3<f32>(1.0, 0.0, 0.0);
            base_color = vec3<f32>(0.72, 0.68, 0.58);
            roughness = 0.65;
        } else {
            // Right wall (+X) – tile backsplash / counter
            position = vec3<f32>(room, v.y * 1.5 + 1.5, v.x * room);
            normal = vec3<f32>(-1.0, 0.0, 0.0);
            base_color = vec3<f32>(0.50, 0.55, 0.52);
            metallic = 0.15; roughness = 0.25;
        }

        // Floor checker
        if (face == 0u) {
            let cx = floor(position.x / 1.5 + 0.001);
            let cz = floor(position.z / 1.5 + 0.001);
            if (((i32(cx) + i32(cz)) % 2 + 2) % 2 == 0) {
                base_color = vec3<f32>(0.65, 0.55, 0.42);
                roughness = 0.15;
            }
        }

        out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
        out.world_pos = position;
        out.normal = normal;
        out.base_color = base_color;
        out.pbr_params = vec4<f32>(metallic, roughness, ao, emissive);
        return out;

    } else if (idx < 228u) {
        // ── Pests ─────────────────────────────────────────────────
        let pest_local = idx - 36u;
        let pest_idx = pest_local / 24u;
        let vert_local = pest_local % 24u;

        // Get pest data
        var pd: vec4<f32>;
        if (pest_idx == 0u) { pd = scene.pest0; }
        else if (pest_idx == 1u) { pd = scene.pest1; }
        else if (pest_idx == 2u) { pd = scene.pest2; }
        else if (pest_idx == 3u) { pd = scene.pest3; }
        else if (pest_idx == 4u) { pd = scene.pest4; }
        else if (pest_idx == 5u) { pd = scene.pest5; }
        else if (pest_idx == 6u) { pd = scene.pest6; }
        else { pd = scene.pest7; }

        let center = pd.xyz;
        let pest_type = pd.w;
        let radius = pest_radius(pest_type);

        // Get hit flash
        var flash = 0.0;
        if (pest_idx == 0u) { flash = scene.pest_flash.x; }
        else if (pest_idx == 1u) { flash = scene.pest_flash.y; }
        else if (pest_idx == 2u) { flash = scene.pest_flash.z; }
        else if (pest_idx == 3u) { flash = scene.pest_flash.w; }
        else if (pest_idx == 4u) { flash = scene.pest_flash2.x; }
        else if (pest_idx == 5u) { flash = scene.pest_flash2.y; }
        else if (pest_idx == 6u) { flash = scene.pest_flash2.z; }
        else { flash = scene.pest_flash2.w; }

        // Octahedron vertex
        let tri = vert_local / 3u;
        let vi = vert_local % 3u;
        let fid = fi[tri];
        var vert: vec3<f32>;
        if (vi == 0u) { vert = oct[fid[0]]; }
        else if (vi == 1u) { vert = oct[fid[1]]; }
        else { vert = oct[fid[2]]; }

        let n = normalize(vert);
        // Squash vertically for ground pests to look bug-like
        var scale = vec3<f32>(1.0, 0.5, 1.0);
        if (pest_type > 3.5 && pest_type < 4.5) { scale = vec3<f32>(1.2, 0.6, 0.8); } // rat: wider
        if (pest_type > 4.5) { scale = vec3<f32>(0.7, 1.0, 1.2); } // wasp: elongated
        let scaled_vert = n * scale;
        position = center + normalize(scaled_vert) * radius;
        normal = normalize(scaled_vert);

        base_color = pest_base_color(pest_type);
        metallic = pest_metallic(pest_type);
        roughness = 0.35;

        // Hit flash: turn white-red
        if (flash > 0.0) {
            base_color = mix(base_color, vec3<f32>(1.0, 0.3, 0.15), flash);
            emissive = flash * 8.0;
        }

        // Alive glow (subtle)
        if (pest_type > 0.5) {
            let pd2 = length(camera.view_position.xyz - center);
            emissive += max(0.0, 1.0 - pd2 / 4.0) * 0.5;
        }

        out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
        out.world_pos = position;
        out.normal = normal;
        out.base_color = base_color;
        out.pbr_params = vec4<f32>(metallic, roughness, ao, emissive);
        return out;

    } else {
        // ── Crosshair ─────────────────────────────────────────────
        let ch = idx - 228u;
        let bar = ch / 6u;
        let vi = ch % 6u;
        let s = 0.018;
        let t2 = 0.0025;
        var px = array<f32,6>(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
        var py = array<f32,6>(-1.0,-1.0, 1.0, -1.0, 1.0,  1.0);

        var cx: f32; var cy: f32;
        if (bar == 0u) { cx = px[vi]*s; cy = py[vi]*t2; }
        else           { cx = px[vi]*t2; cy = py[vi]*s; }

        // Dot at center
        let firing = scene.params.w;
        var ch_color = vec3<f32>(1.0, 1.0, 1.0);
        if (firing > 0.1) { ch_color = vec3<f32>(1.0, 0.4, 0.2); }

        out.clip_position = vec4<f32>(cx, cy, 0.0005, 1.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 0.0, 1.0);
        out.base_color = ch_color;
        out.pbr_params = vec4<f32>(0.0, 0.5, 1.0, 20.0); // pure emissive
        return out;
    }
}

// ─── PBR Functions ─────────────────────────────────────────────────────────

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, r: f32) -> f32 {
    let a = r*r; let a2 = a*a; let d = max(dot(N,H),0.0);
    let x = d*d*(a2-1.0)+1.0; return a2/(PI*x*x+0.0001);
}
fn geometry_schlick(d: f32, r: f32) -> f32 {
    let k = ((r+1.0)*(r+1.0))/8.0; return d/(d*(1.0-k)+k);
}
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, r: f32) -> f32 {
    return geometry_schlick(max(dot(N,V),0.0),r)*geometry_schlick(max(dot(N,L),0.0),r);
}
fn fresnel_schlick(ct: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0+(1.0-F0)*pow(clamp(1.0-ct,0.0,1.0),5.0);
}
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),vec3<f32>(0.0),vec3<f32>(1.0));
}
fn to_srgb(c: vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(1.0/2.2)); }

fn pbr_direct(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, rad: vec3<f32>, alb: vec3<f32>, met: f32, rou: f32) -> vec3<f32> {
    let H = normalize(V+L); let F0 = mix(vec3<f32>(0.04),alb,met);
    let D = distribution_ggx(N,H,rou); let G = geometry_smith(N,V,L,rou);
    let F = fresnel_schlick(max(dot(H,V),0.0),F0);
    let NdV = max(dot(N,V),0.0); let NdL = max(dot(N,L),0.0);
    let spec = (D*G*F)/(4.0*NdV*NdL+0.0001);
    return ((vec3<f32>(1.0)-F)*(1.0-met)*alb/PI+spec)*rad*NdL;
}
fn pt_light(N: vec3<f32>, V: vec3<f32>, wp: vec3<f32>, lp: vec3<f32>, lc: vec3<f32>, li: f32, a: vec3<f32>, m: f32, r: f32) -> vec3<f32> {
    let d = length(lp-wp); return pbr_direct(N,V,normalize(lp-wp),lc*li/(d*d+1.0),a,m,r);
}

// ─── Fragment Shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let em = in.pbr_params.w;

    // Crosshair: pure emissive, no PBR
    if (em >= 15.0) {
        return vec4<f32>(in.base_color, 1.0);
    }

    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    let alb = in.base_color;
    let met = in.pbr_params.x;
    let rou = clamp(in.pbr_params.y, 0.045, 1.0);
    let ao  = in.pbr_params.z;
    let time = scene.params.x;
    let expo = scene.params.y;
    let ambi = scene.params.z;
    let firing = scene.params.w;

    // ── Lighting ──────────────────────────────────────────────────
    var Lo = pbr_direct(N,V,normalize(scene.sun_direction.xyz),
        scene.sun_color.xyz * scene.sun_direction.w, alb, met, rou);
    Lo += pt_light(N,V,in.world_pos, scene.light0_pos.xyz,
        scene.light0_color.xyz, scene.light0_color.w, alb, met, rou);
    Lo += pt_light(N,V,in.world_pos, scene.light1_pos.xyz,
        scene.light1_color.xyz, scene.light1_color.w, alb, met, rou);

    // ── Ambient ───────────────────────────────────────────────────
    let F0a = mix(vec3<f32>(0.04), alb, met);
    let kSa = fresnel_schlick(max(dot(N,V),0.0), F0a);
    let kDa = (vec3<f32>(1.0) - kSa) * (1.0 - met);
    let hemi = mix(vec3<f32>(0.04, 0.035, 0.03),
                   vec3<f32>(0.12, 0.11, 0.09), N.y * 0.5 + 0.5);
    var color = Lo + kDa * alb * hemi * ambi * ao + kSa * hemi * ambi * 0.2;

    // ── Emissive ──────────────────────────────────────────────────
    if (em > 0.0) { color += alb * em; }

    // ── Window light glow (face 3) ────────────────────────────────
    if (em > 1.0 && in.world_pos.z < -5.5) {
        let pulse = sin(time * 0.5) * 0.05 + 0.95;
        color *= pulse;
    }

    // ── Firing flash overlay ──────────────────────────────────────
    if (firing > 0.0) {
        color += vec3<f32>(0.6, 0.4, 0.1) * firing * 0.15;
    }

    color = aces(color * expo);
    color = to_srgb(color);
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
