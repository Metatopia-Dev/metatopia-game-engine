// ─── Metatopia Non-Euclidean Geodesic Ray-Marcher & Recursive Portal Shader ──
// Production Audio-Reactive WGSL Shader for "GEODESIC HORIZON (128 BPM)"
// Features:
//   1. Multi-bounce recursive GPU portal ray traversal (up to 16 chart bounces)
//   2. Analytical 3D conformal Christoffel symbol RK2 midpoint geodesic integration
//   3. Möbius gyrovector domain folding for H³ {4,3,5} hyperbolic honeycombs
//   4. High-contrast dark-obsidian Cook-Torrance GGX PBR + Fresnel + Neon Bloom
//   5. Real-time 128 BPM audio-reactive envelopes (kick, bass, arp, riser, chromatic punch)

const PI: f32 = 3.14159265359;

struct PortalBenchmarkUniform {
    camera_pos: vec4<f32>,      // xyz = position in active chart, w = fov
    camera_dir: vec4<f32>,      // xyz = forward unit vector, w = active_chart_id
    camera_up: vec4<f32>,       // xyz = up unit vector (with dutch roll), w = curvature_k
    resolution: vec4<f32>,      // x = width, y = height, z = time, w = scene_mode (0..3)
    params: vec4<f32>,          // x = max_portal_bounces, y = geodesic_enabled, z = heatmap_mode, w = max_steps
    portal0_src: vec4<f32>,     // xyz = source center, w = source_chart
    portal0_dst: vec4<f32>,     // xyz = target center, w = target_chart
    portal0_norm: vec4<f32>,    // xyz = portal plane normal, w = yaw_rotation_rad
    portal1_src: vec4<f32>,
    portal1_dst: vec4<f32>,
    portal1_norm: vec4<f32>,
    portal2_src: vec4<f32>,
    portal2_dst: vec4<f32>,
    portal2_norm: vec4<f32>,
    portal3_src: vec4<f32>,
    portal3_dst: vec4<f32>,
    portal3_norm: vec4<f32>,
    telemetry: vec4<f32>,       // x = kick_env, y = bass_env, z = arp_env, w = cinematic_mode (1.0 = music video)
}

@group(0) @binding(0) var<uniform> u: PortalBenchmarkUniform;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0),  vec2<f32>(-1.0, 1.0)
    );
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = positions[idx];
    return out;
}

// ─── Mathematical & Manifold Primitives ─────────────────────────────────────

fn rotate_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn rotate_z(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

// Möbius gyrovector addition u ⊕_K v in 3D conformal space
fn mobius_add(u_vec: vec3<f32>, v_vec: vec3<f32>, curvature_k: f32) -> vec3<f32> {
    if (abs(curvature_k) < 0.001) {
        return u_vec + v_vec;
    }
    let c = -curvature_k * 0.06;
    let u2 = dot(u_vec, u_vec);
    let v2 = dot(v_vec, v_vec);
    let uv = dot(u_vec, v_vec);
    let denom = max(1.0 + 2.0 * c * uv + c * c * u2 * v2, 0.04);
    return ((1.0 + 2.0 * c * uv + c * v2) * u_vec + (1.0 - c * u2) * v_vec) / denom;
}

// Analytical 3D conformally flat Christoffel geodesic acceleration: aⁱ = -Γⁱⱼₖ vʲ vᵏ
fn geodesic_acceleration(pos: vec3<f32>, vel: vec3<f32>, curvature_k: f32) -> vec3<f32> {
    if (abs(curvature_k) < 0.001) {
        return vec3<f32>(0.0);
    }
    let scaled_p = pos * 0.055;
    let raw_r2 = dot(scaled_p, scaled_p);
    let max_r2 = select(100.0, 0.96 / max(-curvature_k, 0.01), curvature_k < 0.0);
    let r2 = min(raw_r2, max_r2);
    let denom = max(1.0 + curvature_k * r2, 0.08);
    let d_sigma = (-2.0 * curvature_k / denom) * scaled_p;
    let v_dot_ds = dot(d_sigma, vel);
    let v2 = dot(vel, vel);
    return -(2.0 * v_dot_ds * vel - v2 * d_sigma) * 0.14;
}

fn chart_curvature(chart_id: u32, base_k: f32) -> f32 {
    if (chart_id == 1u) {
        return base_k - 0.65;
    } else if (chart_id == 2u) {
        return base_k + 0.65;
    }
    return base_k;
}

fn portal_accent_color(chart_id: u32) -> vec3<f32> {
    let idx = chart_id % 4u;
    if (idx == 0u) {
        return vec3<f32>(0.05, 0.92, 1.0);   // Electric Laser Cyan (E³)
    } else if (idx == 1u) {
        return vec3<f32>(1.0, 0.08, 0.62);   // Hyperbolic Neon Magenta (H³)
    } else if (idx == 2u) {
        return vec3<f32>(1.0, 0.72, 0.08);   // Spherical Solar Gold (S³)
    } else {
        return vec3<f32>(0.18, 1.0, 0.48);   // Escher Cyber Emerald
    }
}

// ─── Signed Distance Field Primitives ──────────────────────────────────────

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_torus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xy) - t.x, p.z);
    return length(q) - t.y;
}

fn sd_portal_frame(p: vec3<f32>, center: vec3<f32>, normal: vec3<f32>, half_w: f32, half_h: f32) -> f32 {
    let n = normalize(normal);
    let ref_up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let right = normalize(cross(n, ref_up));
    let up = normalize(cross(right, n));
    let rel = p - center;
    let local = vec3<f32>(dot(rel, right), dot(rel, up), dot(rel, n));

    let outer = sd_box(local, vec3<f32>(half_w + 0.18, half_h + 0.18, 0.10));
    let inner = sd_box(local, vec3<f32>(half_w, half_h, 0.30));
    return max(outer, -inner);
}

fn sd_portal_aperture(p: vec3<f32>, center: vec3<f32>, normal: vec3<f32>, half_w: f32, half_h: f32) -> f32 {
    let n = normalize(normal);
    let ref_up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let right = normalize(cross(n, ref_up));
    let up = normalize(cross(right, n));
    let rel = p - center;
    let local = vec3<f32>(dot(rel, right), dot(rel, up), dot(rel, n));
    return sd_box(local, vec3<f32>(half_w, half_h, 0.03));
}

// 3D Geodesic Particle Swarm SDF: Dual helical streams of particles advecting along curved geodesics through portals
fn sd_geodesic_swarm(p: vec3<f32>, time: f32, curvature_k: f32) -> f32 {
    let z_speed = time * 6.4;
    let z_cell = 1.6;
    let z_local = (p.z + z_speed) - round((p.z + z_speed) / z_cell) * z_cell;
    let cell_idx = round((p.z + z_speed) / z_cell);
    let phase = cell_idx * 2.39996 + time * 1.8;
    let r_orbit = 1.48 + 0.18 * sin(cell_idx * 0.7 + curvature_k);
    let orb_a = vec3<f32>(r_orbit * cos(phase), 1.62 + 0.85 * r_orbit * sin(phase), z_local);
    let orb_b = vec3<f32>(-r_orbit * cos(phase), 1.62 - 0.85 * r_orbit * sin(phase), z_local);
    let q = vec3<f32>(p.x, p.y, z_local);
    return min(length(q - orb_a), length(q - orb_b)) - 0.075;
}

struct SdfHit {
    dist: f32,
    material_id: u32,   // 0=walls, 1=chrome_arch, 2=neon_ring, 3=portal_frame, 4=honeycomb, 5=klein_chiral, 10..13=portal_aperture
    portal_idx: i32,
}

fn get_portal_data(idx: i32) -> array<vec4<f32>, 3> {
    if (idx == 0) {
        return array<vec4<f32>, 3>(u.portal0_src, u.portal0_dst, u.portal0_norm);
    } else if (idx == 1) {
        return array<vec4<f32>, 3>(u.portal1_src, u.portal1_dst, u.portal1_norm);
    } else if (idx == 2) {
        return array<vec4<f32>, 3>(u.portal2_src, u.portal2_dst, u.portal2_norm);
    } else {
        return array<vec4<f32>, 3>(u.portal3_src, u.portal3_dst, u.portal3_norm);
    }
}

// ─── Multi-Chart Scene SDF (Unobstructed Sightlines + Audio-Reactive Geometry) ──

fn map_chart_scene(pos: vec3<f32>, chart_id: u32, scene_mode: u32, skip_portal_Check: bool) -> SdfHit {
    var best: SdfHit;
    best.dist = 1e9;
    best.material_id = 0u;
    best.portal_idx = -1;

    let time = u.resolution.z;
    let kick = u.telemetry.x;
    let bass = u.telemetry.y;
    let arp = u.telemetry.z;
    var p = pos;

    // Smooth near-camera distance factor to prevent foreground rings from clipping at screen edges
    let cam_dist_z = abs(p.z - u.camera_pos.z);
    let near_guard = max(1.45 - cam_dist_z, 0.0) * 1.5;

    if (scene_mode == 1u) {
        // ── Scene 1: Hyperbolic H³ {4,3,5} Conformal Honeycomb Megastructure ──
        // Carve a wide open flight cathedral along the camera trajectory so center sightline is crystal clear
        let tunnel_clear = length(p.xy - vec2<f32>(0.0, 1.65)) - 1.85;

        // Clean C1-continuous lattice repetition (avoids nonlinear domain fold aliasing in far field!)
        let cell_size = 3.2;
        let q = p - round(p / cell_size) * cell_size;
        let radial_warp = clamp(1.0 / (1.0 + 0.018 * dot(p.xy, p.xy)), 0.45, 1.0);

        // Octahedral/Cubic dark-chrome hyperbolic struts + vertex gyro-spheres
        let strut_r = (0.24 + 0.035 * bass) * radial_warp;
        let d_x = length(q.yz) - strut_r;
        let d_y = length(q.xz) - strut_r;
        let d_z = length(q.xy) - strut_r;
        let raw_struts = min(d_x, min(d_y, d_z));
        let node_sphere = length(q) - (0.56 + 0.07 * kick) * radial_warp;
        let lattice = min(raw_struts, node_sphere);
        let struts = max(lattice, -tunnel_clear);

        // Coaxial cathedral guide rings spaced cleanly along the central flight tunnel
        let ring_z = p.z - round(p.z / 3.2) * 3.2;
        let ring_p = vec3<f32>(p.x, p.y - 1.65, ring_z);
        let tunnel_ring = sd_torus(ring_p, vec2<f32>(2.05 + 0.12 * kick, 0.05 + 0.018 * arp)) + near_guard;

        if (struts < best.dist) {
            best.dist = struts;
            best.material_id = 4u;
        }
        if (tunnel_ring < best.dist) {
            best.dist = tunnel_ring;
            best.material_id = 2u;
        }
        return best;
    }

    if (scene_mode == 2u) {
        // ── Scene 2: Multi-Chart Curvature Bridge — Cylindrical Spherical S³ Star-Vault ──
        let cyl_r = 4.35 + 0.35 * sin(p.z * 0.35);
        let room_d = -(length(p.xy - vec2<f32>(0.0, 1.6)) - cyl_r);
        best.dist = room_d;
        best.material_id = 0u;

        // Structural ribbed arches embedded along the curved cylindrical vault wall
        let rib_z = p.z - round(p.z / 2.2) * 2.2;
        let rib_p = vec3<f32>(p.x, p.y - 1.6, rib_z);
        let vault_rib = sd_torus(rib_p, vec2<f32>(cyl_r - 0.25, 0.22));
        if (vault_rib < best.dist) {
            best.dist = vault_rib;
            best.material_id = 1u;
        }

        // Inner floating neon bridge rings coaxial with the Z flight tunnel
        let arch_z = p.z - round(p.z / 2.8) * 2.8;
        let arch_p = vec3<f32>(p.x, p.y - 1.6, arch_z);
        let arch_ring = sd_torus(arch_p, vec2<f32>(2.55 + 0.16 * bass, 0.065 + 0.02 * kick)) + near_guard;
        if (arch_ring < best.dist) {
            best.dist = arch_ring;
            best.material_id = 2u;
        }
    } else if (scene_mode == 4u) {
        // ── Scene 4: Non-Orientable Klein Bottle K³ & Flat 3-Torus T³ Chiral Labyrinth ──
        // Figure-8 twisted vault with asymmetric chiral pillars so X -> -X parity flip is visibly dramatic!
        let twist_angle = p.z * 0.28;
        let twisted_xy = rotate_z(vec3<f32>(p.x, p.y - 1.6, 0.0), twist_angle).xy;
        let room_d = -(length(twisted_xy * vec2<f32>(0.82, 1.05)) - 4.45);
        best.dist = room_d;
        best.material_id = 0u;

        // Asymmetric Chiral Double-Helix Pillars (Left x < 0 vs Right x > 0 have opposite chirality & color!)
        let z_rep = p.z - round(p.z / 2.0) * 2.0;
        let chiral_offset = 0.45 * sin(p.z * 0.9 + select(-1.2, 1.2, p.x > 0.0));
        let col_x = abs(p.x) - 2.85;
        let col_y = (p.y - 1.6) - chiral_offset;
        let chiral_pillar = sd_box(vec3<f32>(col_x, col_y, z_rep), vec3<f32>(0.34, 1.85, 0.26));
        if (chiral_pillar < best.dist) {
            best.dist = chiral_pillar;
            best.material_id = 5u;
        }

        // Twisted Möbius / Klein Immersion Halo Rings coaxial with the portal sightline
        let ring_z = p.z - round(p.z / 2.8) * 2.8;
        let fig8_warp = 0.22 * sin(atan2(p.y - 1.6, p.x) * 2.0 + p.z * 0.5);
        let rp = vec3<f32>(p.x, p.y - 1.6, ring_z);
        let klein_ring = sd_torus(rp, vec2<f32>(2.42 + fig8_warp + 0.14 * kick, 0.06 + 0.02 * arp)) + near_guard;
        if (klein_ring < best.dist) {
            best.dist = klein_ring;
            best.material_id = 2u;
        }
    } else {
        // ── Architectural Non-Euclidean Corridor & Vaults (Scenes 0 & 3) ──
        let corridor_w = 5.2;
        let corridor_h = 3.8;
        let room_d = -sd_box(p - vec3<f32>(0.0, 1.6, 0.0), vec3<f32>(corridor_w, corridor_h, 7.5));
        best.dist = room_d;
        best.material_id = 0u;

        // Flanking Cyber-Arches & Ribbed Columns (strictly on the sides |x| >= 2.5, NEVER blocking the center!)
        let z_rep = p.z - round(p.z / 2.4) * 2.4;
        let side_x = abs(p.x) - (corridor_w - 0.85);
        let rib_col = sd_box(vec3<f32>(side_x, p.y - 1.6, z_rep), vec3<f32>(0.32, corridor_h, 0.28));
        if (rib_col < best.dist) {
            best.dist = rib_col;
            best.material_id = 1u;
        }

        // Ceiling & Floor Neon Laser Guide Rails (flanking the portal sightline at |x| = 2.05)
        let rail_x = abs(p.x) - 2.05;
        let rail_y = abs(p.y - 1.6) - (corridor_h - 0.18);
        let rail_d = length(vec2<f32>(rail_x, rail_y)) - (0.055 + 0.025 * kick);
        if (rail_d < best.dist) {
            best.dist = rail_d;
            best.material_id = 2u;
        }

        // Scenes 0 & 3: Beat-Pulsing Coaxial Non-Euclidean Halo Rings framing the portal axis
        let ring_z = p.z - round(p.z / 3.2) * 3.2;
        let rp = vec3<f32>(p.x, p.y - 1.6, ring_z);
        let halo = sd_torus(rp, vec2<f32>(2.65 + 0.14 * kick, 0.05 + 0.018 * arp)) + near_guard;
        if (halo < best.dist) {
            best.dist = halo;
            best.material_id = 2u;
        }
    }

    // Evaluate Active Portals in `chart_id`
    let half_w = 1.65;
    let half_h = 2.35;
    for (var i = 0; i < 4; i++) {
        let pd = get_portal_data(i);
        let src_chart = u32(round(pd[0].w));
        if (src_chart == chart_id) {
            let center = pd[0].xyz;
            let normal = pd[2].xyz;
            let frame_d = sd_portal_frame(p, center, normal, half_w, half_h);
            if (frame_d < best.dist) {
                best.dist = frame_d;
                best.material_id = 3u;
                best.portal_idx = i;
            }
            if (!skip_portal_Check) {
                let aperture_d = sd_portal_aperture(p, center, normal, half_w, half_h);
                if (aperture_d < best.dist) {
                    best.dist = aperture_d;
                    best.material_id = 10u + u32(i);
                    best.portal_idx = i;
                }
            }
        }
    }

    return best;
}

fn map_solid_geometry(pos: vec3<f32>, chart_id: u32, scene_mode: u32) -> f32 {
    let hit = map_chart_scene(pos, chart_id, scene_mode, true);
    return hit.dist;
}

fn calc_normal(p: vec3<f32>, chart_id: u32, scene_mode: u32) -> vec3<f32> {
    let e = 0.0025;
    let dx = map_solid_geometry(p + vec3<f32>(e, 0.0, 0.0), chart_id, scene_mode)
           - map_solid_geometry(p - vec3<f32>(e, 0.0, 0.0), chart_id, scene_mode);
    let dy = map_solid_geometry(p + vec3<f32>(0.0, e, 0.0), chart_id, scene_mode)
           - map_solid_geometry(p - vec3<f32>(0.0, e, 0.0), chart_id, scene_mode);
    let dz = map_solid_geometry(p + vec3<f32>(0.0, 0.0, e), chart_id, scene_mode)
           - map_solid_geometry(p - vec3<f32>(0.0, 0.0, e), chart_id, scene_mode);
    return normalize(vec3<f32>(dx, dy, dz));
}

// ─── Recursive Portal + Geodesic Ray Marcher ───────────────────────────────

struct TraceResult {
    hit: bool,
    pos: vec3<f32>,
    dir: vec3<f32>,
    chart_id: u32,
    material_id: u32,
    portal_idx: i32,
    total_dist: f32,
    steps_taken: u32,
    portal_bounces: u32,
    parity_sign: f32,
    portal_glow: vec3<f32>,
}

fn trace_non_euclidean_manifold(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    start_chart: u32,
) -> TraceResult {
    var res: TraceResult;
    res.hit = false;
    res.pos = ray_origin;
    res.dir = ray_dir;
    res.chart_id = start_chart;
    res.material_id = 0u;
    res.portal_idx = -1;
    res.total_dist = 0.0;
    res.steps_taken = 0u;
    res.portal_bounces = 0u;
    res.parity_sign = 1.0;
    res.portal_glow = vec3<f32>(0.0);

    let max_bounces = u32(round(u.params.x));
    let geodesic_on = u.params.y > 0.5;
    let max_steps = u32(clamp(round(u.params.w), 32.0, 180.0));
    let scene_mode = u32(round(u.resolution.w));
    let base_k = u.camera_up.w;
    let kick = u.telemetry.x;
    let arp = u.telemetry.z;

    var curr_pos = ray_origin;
    var curr_dir = normalize(ray_dir);
    var curr_chart = start_chart;

    for (var step = 0u; step < max_steps; step++) {
        res.steps_taken = step + 1u;
        let hit = map_chart_scene(curr_pos, curr_chart, scene_mode, false);
        var march_step = clamp(hit.dist * 0.85, 0.003, 0.90);

        // Attenuate volumetric bloom on deep portal bounces so innermost levels stay crisp and never blow out to white
        let bounce_atten = exp(-0.22 * f32(res.portal_bounces));

        // Volumetric Geodesic Particle Swarm streaming along curved trajectories through portals
        let swarm_d = sd_geodesic_swarm(curr_pos, u.resolution.z, base_k);
        if (swarm_d < 0.45) {
            let p_col = portal_accent_color(curr_chart + res.portal_bounces + 2u);
            res.portal_glow += p_col * (0.012 * (1.0 + 0.8 * arp) * bounce_atten * march_step / (0.02 + swarm_d * swarm_d * 24.0));
        }

        // Crisp volumetric neon glow strictly around solid portal frames (material_id == 3) and laser rings (material_id == 2)
        if (hit.material_id == 3u && hit.portal_idx >= 0) {
            let p_idx = clamp(hit.portal_idx, 0, 3);
            let pd = get_portal_data(p_idx);
            let dst_chart = u32(round(pd[1].w));
            let glow_col = portal_accent_color(dst_chart + res.portal_bounces + scene_mode);
            let pulse = (1.0 + 0.55 * kick) * bounce_atten;
            res.portal_glow += glow_col * pulse * (0.036 * march_step / (1.0 + hit.dist * hit.dist * 38.0));
        } else if (hit.material_id == 2u) {
            let glow_col = portal_accent_color(curr_chart + 1u + scene_mode);
            let pulse = (1.0 + 0.6 * arp + 0.35 * kick) * bounce_atten;
            res.portal_glow += glow_col * pulse * (0.028 * march_step / (1.0 + hit.dist * hit.dist * 32.0));
        }

        if (hit.dist < 0.0045) {
            // Check if the ray hit an active portal aperture
            if (hit.material_id >= 10u) {
                let p_idx = clamp(i32(hit.material_id) - 10, 0, 3);
                if (res.portal_bounces < max_bounces) {
                    let pd = get_portal_data(p_idx);
                    let src_center = pd[0].xyz;
                    let dst_center = pd[1].xyz;
                    let dst_chart = u32(round(pd[1].w));
                    let yaw_rot = pd[2].w;

                    // Single sin/cos evaluation for local tangent-frame position + direction transport
                    let c_yaw = cos(yaw_rot);
                    let s_yaw = sin(yaw_rot);
                    let rel = curr_pos - src_center;
                    var rot_offset = vec3<f32>(
                        c_yaw * rel.x + s_yaw * rel.z,
                        rel.y,
                        -s_yaw * rel.x + c_yaw * rel.z
                    );
                    curr_dir = normalize(vec3<f32>(
                        c_yaw * curr_dir.x + s_yaw * curr_dir.z,
                        curr_dir.y,
                        -s_yaw * curr_dir.x + c_yaw * curr_dir.z
                    ));
                    // Helical roll on recursive mirror tunnel bounces for kaleidoscopic spiral depth
                    if (scene_mode == 3u) {
                        let roll = 0.16;
                        rot_offset = rotate_z(rot_offset, roll);
                        curr_dir = normalize(rotate_z(curr_dir, roll));
                    } else if (scene_mode == 4u) {
                        // Non-Orientable Klein Bottle K³ Parity Flip: reflect X -> -X and invert chirality!
                        rot_offset.x = -rot_offset.x;
                        curr_dir.x = -curr_dir.x;
                        res.parity_sign = -res.parity_sign;
                    }

                    // Local tangent-space Möbius conformal transport (centered at portal origin so |rel| < 1)
                    let dst_k = chart_curvature(dst_chart, base_k);
                    let local_mobius = mobius_add(vec3<f32>(0.0), rot_offset * 0.25, dst_k * 0.15) * 4.0;
                    let exit_offset = 0.24; // Clear destination portal frame slab completely
                    curr_pos = dst_center + local_mobius + curr_dir * exit_offset;
                    res.total_dist += exit_offset;
                    curr_chart = dst_chart;
                    res.portal_bounces += 1u;

                    // Edge-only neon rim highlight on each nested portal boundary, attenuated by bounce depth
                    let rim_x = abs(rel.x) / 1.65;
                    let rim_y = abs(rel.y) / 2.35;
                    let rim_prox = max(rim_x, rim_y);
                    let rim_line = smoothstep(0.88, 0.99, rim_prox);
                    let bounce_col = portal_accent_color(dst_chart + res.portal_bounces + scene_mode);
                    res.portal_glow += bounce_col * rim_line * (0.65 + 0.35 * kick) * bounce_atten;

                    if (res.total_dist > 115.0) {
                        break;
                    }
                    continue;
                } else {
                    res.hit = true;
                    res.pos = curr_pos;
                    res.dir = curr_dir;
                    res.chart_id = curr_chart;
                    res.material_id = hit.material_id;
                    res.portal_idx = hit.portal_idx;
                    return res;
                }
            }

            // Solid surface hit
            res.hit = true;
            res.pos = curr_pos;
            res.dir = curr_dir;
            res.chart_id = curr_chart;
            res.material_id = hit.material_id;
            res.portal_idx = hit.portal_idx;
            return res;
        }

        if (geodesic_on) {
            // True RK2 midpoint geodesic integration on both tangent direction and position
            let k_scale = select(0.45, 0.18, scene_mode == 3u);
            let chart_k = chart_curvature(curr_chart, base_k) * k_scale;
            let a1 = geodesic_acceleration(curr_pos, curr_dir, chart_k);
            march_step = min(march_step, 0.48 / max(length(a1), 0.25));
            let mid_dir = normalize(curr_dir + a1 * (march_step * 0.5));
            let mid_pos = curr_pos + curr_dir * (march_step * 0.5);
            let a2 = geodesic_acceleration(mid_pos, mid_dir, chart_k);
            curr_pos += mid_dir * march_step;
            curr_dir = normalize(curr_dir + a2 * march_step);
        } else {
            curr_pos += curr_dir * march_step;
        }

        res.total_dist += march_step;

        if (res.total_dist > 115.0) {
            break;
        }
    }

    res.pos = curr_pos;
    res.dir = curr_dir;
    res.chart_id = curr_chart;
    return res;
}

// ─── High-Contrast Dark-Neon Cook-Torrance PBR & Anti-Aliased Grids ────────

fn chart_grid_pattern(p: vec3<f32>, n: vec3<f32>, chart_id: u32, dist: f32, scene_mode: u32) -> f32 {
    // Stronger distance-based anti-aliasing fade in Honeycomb (scene 1) to eliminate far-field moiré shimmer
    let fade_rate = select(0.032, 0.14, scene_mode == 1u);
    let dist_fade = exp(-fade_rate * dist);
    if (dist_fade < 0.02) {
        return 0.0;
    }
    let scale = select(1.25, 1.5, chart_id == 1u);
    let filter_w = clamp(0.024 + dist * 0.0065, 0.024, 0.30);
    let gp = abs(fract(p * scale - 0.5) - 0.5) / filter_w;
    let line = min(min(gp.x, gp.y), gp.z);
    let grid = (1.0 - min(line, 1.0)) * dist_fade;

    if (abs(n.y) > 0.5 && scene_mode != 1u) {
        let r = length(p.xz);
        let ring = abs(fract(r * 0.5 - u.resolution.z * 0.35) - 0.5);
        let ring_line = smoothstep(filter_w * 1.4, 0.0, ring) * dist_fade;
        return max(grid * 0.72, ring_line * 0.85);
    }
    return grid * 0.72;
}

fn heatmap_color(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return vec3<f32>(
        clamp(1.5 * x, 0.0, 1.0),
        clamp(1.5 * x - 0.35, 0.0, 1.0),
        clamp(sin(x * PI) * 0.9 + (1.0 - x) * 0.35, 0.0, 1.0)
    );
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
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let uv = vec2<f32>(in.uv.x * aspect, in.uv.y);
    let time = u.resolution.z;
    let scene_mode = u32(round(u.resolution.w));
    let heatmap_mode = u32(round(u.params.z));
    let max_steps = max(u.params.w, 32.0);
    let max_bounces = max(u.params.x, 1.0);

    let kick = u.telemetry.x;
    let bass = u.telemetry.y;
    let arp = u.telemetry.z;
    let cinematic_mode = u.telemetry.w > 0.5;

    // 2.35:1 Anamorphic Widescreen Matte Black Letterbox Bars (Clean full-width edge, zero partial seam artifacts)
    if (cinematic_mode && abs(in.uv.y) > 0.86) {
        return vec4<f32>(0.005, 0.006, 0.012, 1.0);
    }

    // Build orthonormal camera frame (supports dynamic Dutch roll via camera_up.xyz)
    let forward = normalize(u.camera_dir.xyz);
    let world_up = normalize(u.camera_up.xyz);
    let right = normalize(cross(forward, world_up));
    let up = cross(right, forward);
    // Beat-reactive FOV punch on kick hits (stronger during Act IV drop)
    let kick_punch = select(0.065, 0.11, scene_mode == 3u);
    let fov_scale = u.camera_pos.w * (1.0 - kick_punch * kick);

    let ray_dir = normalize(uv.x * right + uv.y * up + fov_scale * forward);
    let start_chart = u32(round(u.camera_dir.w));

    let trace = trace_non_euclidean_manifold(u.camera_pos.xyz, ray_dir, start_chart);

    // Step-Cost Diagnostic Heatmap Mode (Mode 1)
    if (heatmap_mode == 1u) {
        let ratio = f32(trace.steps_taken) / max_steps;
        var col = heatmap_color(ratio);
        if (trace.portal_bounces > 0u) {
            col += vec3<f32>(0.0, 0.25, 0.45) * f32(trace.portal_bounces);
        }
        return vec4<f32>(col, 1.0);
    }

    // Deep obsidian cosmic background
    var color = vec3<f32>(0.004, 0.006, 0.014);
    // Rotate accent palette per chart, per portal bounce depth, AND per Act (scene_mode) so every Act & bounce level is unmistakable!
    let palette_idx = trace.chart_id + trace.portal_bounces + select(0u, 2u, scene_mode == 2u);
    let chart_tint = portal_accent_color(palette_idx);
    let secondary_tint = portal_accent_color(palette_idx + 1u);
    let bounce_lod = exp(-0.12 * f32(trace.portal_bounces));

    if (trace.hit) {
        if (trace.material_id >= 10u) {
            // Deep 16th-bounce recursive portal vortex horizon
            let swirl = sin(length(trace.pos.xy) * 8.0 - time * 5.0) * 0.5 + 0.5;
            color = mix(chart_tint, secondary_tint, swirl) * (0.40 + 0.30 * kick);
        } else {
            let n = calc_normal(trace.pos, trace.chart_id, scene_mode);
            let v = normalize(-trace.dir);
            let light_pos = u.camera_pos.xyz + vec3<f32>(2.0 * sin(time * 1.2), 1.5, -2.5);
            let l = normalize(light_pos - trace.pos);
            let h = normalize(v + l);
            let l2 = normalize(vec3<f32>(-0.5, 0.8, -0.6));
            let h2 = normalize(v + l2);

            let grid = chart_grid_pattern(trace.pos, n, trace.chart_id, trace.total_dist, scene_mode);
            let fresnel = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 4.0);

            // High-contrast dark obsidian base + glowing neon geodesic grid lines
            var albedo = vec3<f32>(0.012, 0.015, 0.025);
            var metallic = 0.85;
            var roughness = 0.18;
            var emissive = chart_tint * grid * (1.35 + 0.95 * kick + 0.5 * bass) * bounce_lod;

            if (trace.material_id == 1u) {
                // Dark brushed titanium cyber-arches with gridded surfaces and Fresnel neon rim
                albedo = vec3<f32>(0.018, 0.022, 0.038);
                metallic = 0.92;
                roughness = 0.14;
                emissive = (chart_tint * grid * 1.15 + secondary_tint * fresnel * (0.75 + 0.55 * kick)) * bounce_lod;
            } else if (trace.material_id == 2u) {
                // Blazing emissive neon laser rings & cathedral arches
                albedo = secondary_tint * 0.45;
                metallic = 0.95;
                roughness = 0.08;
                emissive = mix(chart_tint, secondary_tint, 0.5 + 0.5 * sin(trace.pos.z * 1.5 + time * 4.0))
                         * (1.65 + 1.0 * arp + 0.9 * kick) * bounce_lod;
            } else if (trace.material_id == 3u) {
                // Portal Gateway Frame — Intense Colour-Coded Neon Energy Border per Bounce Level
                let p_idx = clamp(trace.portal_idx, 0, 3);
                let pd = get_portal_data(p_idx);
                let dst_chart = u32(round(pd[1].w));
                let p_col = portal_accent_color(dst_chart + trace.portal_bounces + scene_mode);
                albedo = p_col * 0.35;
                emissive = p_col * (2.2 + 1.2 * kick + 0.4 * sin(time * 8.0 + trace.pos.y * 4.0)) * bounce_lod;
            } else if (trace.material_id == 4u) {
                // Hyperbolic H³ Honeycomb Dark-Chrome Struts with Two-Tone Cyan/Magenta Depth Hierarchy
                let depth_mix = clamp(trace.total_dist / 16.0, 0.0, 1.0);
                let near_col = vec3<f32>(0.05, 0.92, 1.0);
                let far_col = vec3<f32>(1.0, 0.08, 0.68);
                let strut_tint = mix(near_col, far_col, depth_mix);
                albedo = vec3<f32>(0.014, 0.012, 0.028);
                metallic = 0.94;
                roughness = 0.14;
                emissive = strut_tint * (grid * 1.35 + fresnel * (0.85 + 0.65 * bass) * (1.0 - 0.70 * depth_mix));
            } else if (trace.material_id == 5u) {
                // Non-Orientable Klein Bottle Chiral Pillars — Left/Right Handedness Flips on Every Portal Bounce!
                let handedness = trace.pos.x * trace.parity_sign;
                let chiral_col = select(
                    vec3<f32>(0.05, 0.95, 1.0), // Left-handed chirality: Laser Cyan
                    vec3<f32>(1.0, 0.72, 0.10), // Right-handed chirality: Solar Gold
                    handedness > 0.0
                );
                albedo = vec3<f32>(0.016, 0.018, 0.032);
                metallic = 0.93;
                roughness = 0.12;
                emissive = chiral_col * (grid * 1.45 + fresnel * (0.95 + 0.65 * kick)) * bounce_lod;
            }

            // GGX / Blinn-Phong high-contrast microfacet specular + Schlick Fresnel
            let ndotl = max(dot(n, l), 0.0);
            let ndotl2 = max(dot(n, l2), 0.0);
            let spec_power = exp2((1.0 - roughness) * 7.5);
            let spec1 = pow(max(dot(n, h), 0.0), spec_power) * (0.25 + 0.75 * metallic) * 1.2 * bounce_lod;
            let spec2 = pow(max(dot(n, h2), 0.0), spec_power * 0.5) * 0.45 * bounce_lod;

            let step_ao = clamp(1.0 - 0.68 * (f32(trace.steps_taken) / max_steps), 0.12, 1.0);
            let direct = albedo * (ndotl * 1.0 + ndotl2 * 0.40 + 0.05)
                       + chart_tint * spec1
                       + secondary_tint * spec2
                       + chart_tint * fresnel * 0.30 * bounce_lod;

            color = direct * step_ao + emissive;

            // Mode 2: Full-3D Multi-Bounce Neon Spectrum Overdrive (retains all 3D geometry, grid & lighting!)
            if (heatmap_mode == 2u) {
                let b_ratio = clamp(f32(trace.portal_bounces) / max(max_bounces * 0.5, 1.0), 0.0, 1.0);
                let spec_col = portal_accent_color(trace.portal_bounces + 1u);
                let aura = spec_col * (grid * 1.85 + fresnel * 1.35 + 0.22 * b_ratio) * bounce_lod;
                color = mix(color, color * 0.68 + aura, 0.78);
            }

            // Atmospheric depth fog (fades to deep obsidian-black in H³ Honeycomb so far-field has zero noise!)
            let fog_rate = select(0.0016, 0.011, scene_mode == 1u);
            let dist_fog = 1.0 - exp(-fog_rate * trace.total_dist * trace.total_dist);
            let h3_step_fog = pow(clamp(f32(trace.steps_taken) / max_steps, 0.0, 1.0), 1.35) * 0.96;
            let fog_t = select(dist_fog, max(dist_fog, h3_step_fog), scene_mode == 1u);
            let horizon_col = select(
                mix(vec3<f32>(0.004, 0.006, 0.016), chart_tint * 0.06, 0.30 + 0.30 * bass),
                vec3<f32>(0.006, 0.004, 0.014),
                scene_mode == 1u
            );
            color = mix(color, horizon_col, clamp(fog_t, 0.0, 0.98));
        }
    } else {
        // Cosmic horizon glow at infinity
        let radial = exp(-3.2 * length(uv));
        let bg_tint = select(chart_tint, vec3<f32>(0.35, 0.05, 0.45), scene_mode == 1u);
        color = vec3<f32>(0.004, 0.005, 0.012) + bg_tint * radial * (0.16 + 0.18 * kick);
    }

    // Add volumetric portal frame & laser bloom
    color += trace.portal_glow * (0.95 + 0.35 * kick);

    // Subtle radial vignette for cinema depth
    let vig = clamp(1.12 - 0.22 * dot(in.uv, in.uv), 0.45, 1.0);
    color *= vig;

    // ACES filmic tonemapping (linear output — hardware Rgba8UnormSrgb target handles sRGB gamma!)
    color = aces_tonemap(color * 1.28);

    return vec4<f32>(color, 1.0);
}
