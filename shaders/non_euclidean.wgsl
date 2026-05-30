// ─── Non-Euclidean PBR Shader with Stochastic Rendering ────────────────────
// Cook-Torrance BRDF · Geometry warping · Procedural noise · Film grain
// Non-deterministic: sparkles, quantum wobble, stochastic roughness, fog noise

const PI: f32 = 3.14159265359;

// ─── Uniforms ──────────────────────────────────────────────────────────────

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
    light2_pos: vec4<f32>,
    light2_color: vec4<f32>,
    light3_pos: vec4<f32>,
    light3_color: vec4<f32>,
    params: vec4<f32>,             // x=time, y=exposure, z=ambient, w=num_lights
    sphere_pos: vec4<f32>,         // xyz=pos, w=glow
    orb0_pos: vec4<f32>,
    orb1_pos: vec4<f32>,
    orb2_pos: vec4<f32>,
    orb3_pos: vec4<f32>,
    interaction: vec4<f32>,        // x=collected, y=total, z=sphere_dist, w=noise_seed
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> scene: SceneUniform;

// ─── Noise Functions ───────────────────────────────────────────────────────

fn hash11(p: f32) -> f32 {
    var x = fract(p * 0.1031);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
}

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

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// Smooth value noise in 3D
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash31(i);
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let x0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    let x1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(x0, x1, u.z);
}

// Fractal Brownian Motion (3 octaves)
fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pp = p;
    for (var i = 0; i < 3; i++) {
        v += a * value_noise(pp);
        pp = pp * 2.01;
        a *= 0.5;
    }
    return v;
}

// ─── Vertex I/O ────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) pbr_params: vec4<f32>,  // x=metallic, y=roughness, z=ao, w=emissive
}

// ─── Geometry Warping ──────────────────────────────────────────────────────

fn apply_curvature(p: vec3<f32>, st: u32) -> vec3<f32> {
    if (st == 0u) { return p; }
    let r = length(p.xz);
    var out = p;
    if (st == 1u) {
        out.y -= r * r * 0.005;
        if (r > 0.1) { let e = 1.0 + r * 0.012; out.x = p.x * e; out.z = p.z * e; }
    } else {
        out.y += r * r * 0.008;
        if (r > 0.1) { let c = 1.0 / (1.0 + r * 0.009); out.x = p.x * c; out.z = p.z * c; }
    }
    return out;
}

fn warp_normal(n: vec3<f32>, p: vec3<f32>, st: u32) -> vec3<f32> {
    if (st == 0u) { return n; }
    let r = length(p.xz);
    if (n.y > 0.5 && r > 1.0) {
        let s = select(-1.0, 1.0, st == 1u);
        let t = r * select(0.016, 0.01, st == 1u);
        return normalize(vec3<f32>(n.x + s * p.x * t * 0.1, n.y, n.z + s * p.z * t * 0.1));
    }
    return n;
}

// ─── Vertex Shader ─────────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let room_size = 10.0;
    var position  = vec3<f32>(0.0);
    var normal    = vec3<f32>(0.0, 1.0, 0.0);
    var base_color = vec3<f32>(0.5);
    var metallic  = 0.04;
    var roughness = 0.5;
    var ao        = 1.0;
    var emissive  = 0.0;

    let st = u32(camera.view_position.w);
    let time = scene.params.x;
    let seed = scene.interaction.w;

    // ── Shared geometry ───────────────────────────────────────────
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

    // Noise intensity per geometry
    let noise_strength = select(select(0.015, 0.06, st == 1u), 0.025, st == 2u);

    if (idx < 36u) {
        // ── Room ──────────────────────────────────────────────────
        let face = idx / 6u;
        let v = qv[idx % 6u];

        if (face == 0u) {
            position = vec3<f32>(v.x*room_size, -2.0, v.y*room_size);
            normal = vec3<f32>(0.0,1.0,0.0);
            base_color = vec3<f32>(0.15,0.14,0.13); metallic = 0.02; roughness = 0.4;
        } else if (face == 1u) {
            position = vec3<f32>(v.x*room_size, 5.0, v.y*room_size);
            normal = vec3<f32>(0.0,-1.0,0.0);
            base_color = vec3<f32>(0.22,0.21,0.20); roughness = 0.92;
        } else if (face == 2u) {
            position = vec3<f32>(v.x*room_size, v.y*3.5+1.5, room_size);
            normal = vec3<f32>(0.0,0.0,-1.0);
            base_color = vec3<f32>(0.28,0.26,0.24); roughness = 0.65;
        } else if (face == 3u) {
            position = vec3<f32>(v.x*room_size, v.y*3.5+1.5, -room_size);
            normal = vec3<f32>(0.0,0.0,1.0);
            base_color = vec3<f32>(0.28,0.26,0.24); roughness = 0.65;
        } else if (face == 4u) {
            position = vec3<f32>(-room_size, v.y*3.5+1.5, v.x*room_size);
            normal = vec3<f32>(1.0,0.0,0.0);
            base_color = vec3<f32>(0.28,0.26,0.24); roughness = 0.65;
        } else {
            let pv = v * 0.4;
            position = vec3<f32>(room_size, pv.y*3.5+1.5, pv.x*room_size*0.3);
            normal = vec3<f32>(-1.0,0.0,0.0);
            base_color = vec3<f32>(0.7,0.3,0.9);
            metallic = 0.8; roughness = 0.1; emissive = 3.0;
        }
        if (face != 5u) {
            if (st == 1u) { base_color *= vec3<f32>(0.85,0.78,1.15); }
            else if (st == 2u) { base_color *= vec3<f32>(1.15,0.95,0.78); }
        }

        // ── Stochastic vertex displacement (geometry uncertainty) ─
        let disp_noise = value_noise(position * 0.4 + vec3<f32>(time * 1.5, seed * 10.0, time * 0.7));
        position += normal * (disp_noise - 0.5) * noise_strength * 2.0;

        position = apply_curvature(position, st);
        normal = warp_normal(normal, position, st);

    } else if (idx < 132u) {
        // ── Quantum Sphere (subdivided octahedron) ────────────────
        let li = idx - 36u;
        let sphere_center = scene.sphere_pos.xyz;
        let sphere_glow = scene.sphere_pos.w;
        let sphere_radius = 1.2;

        let ot = li/12u; let si = li%12u; let st2 = si/3u; let vi = si%3u;
        let fid = fi[ot];
        let A = oct[fid[0]]; let B = oct[fid[1]]; let C = oct[fid[2]];
        let mAB = normalize((A+B)*0.5); let mBC = normalize((B+C)*0.5); let mCA = normalize((C+A)*0.5);
        var vert = vec3<f32>(0.0);
        if (st2==0u){if(vi==0u){vert=A;}else if(vi==1u){vert=mAB;}else{vert=mCA;}}
        else if(st2==1u){if(vi==0u){vert=mAB;}else if(vi==1u){vert=B;}else{vert=mBC;}}
        else if(st2==2u){if(vi==0u){vert=mCA;}else if(vi==1u){vert=mBC;}else{vert=C;}}
        else{if(vi==0u){vert=mAB;}else if(vi==1u){vert=mBC;}else{vert=mCA;}}

        let n = normalize(vert);

        // ── Quantum wobble: non-deterministic vertex displacement ─
        let wobble_freq = select(select(3.0, 6.0, st == 1u), 2.0, st == 2u);
        let wobble_amp  = select(select(0.05, 0.15, st == 1u), 0.03, st == 2u);
        let wobble = value_noise(n * wobble_freq + vec3<f32>(time * 3.0, seed * 20.0, time * 1.7));
        let q_radius = sphere_radius + (wobble - 0.5) * wobble_amp;

        position = sphere_center + n * q_radius;
        normal = n;

        if (st == 0u) { base_color = vec3<f32>(1.0,0.86,0.57); }
        else if (st == 1u) { base_color = vec3<f32>(0.90,0.90,0.95); }
        else { base_color = vec3<f32>(0.98,0.72,0.58); }

        // Stochastic color shift on sphere
        let color_noise = hash33(n * 5.0 + vec3<f32>(floor(time * 8.0), seed, 0.0));
        base_color += (color_noise - 0.5) * 0.08;

        metallic = 1.0; roughness = 0.12;
        emissive = sphere_glow * 2.5;

    } else if (idx < 228u) {
        // ── Orbs ──────────────────────────────────────────────────
        let ol = idx - 132u; let oi = ol/24u;
        let inner = ol%24u; let tri = inner/3u; let vi = inner%3u;
        var od: vec4<f32>;
        if(oi==0u){od=scene.orb0_pos;}else if(oi==1u){od=scene.orb1_pos;}
        else if(oi==2u){od=scene.orb2_pos;}else{od=scene.orb3_pos;}

        let orb_center = od.xyz; let orb_active = od.w;
        let orb_radius = 0.35 * orb_active;
        let fid = fi[tri];
        var vert: vec3<f32>;
        if(vi==0u){vert=oct[fid[0]];}else if(vi==1u){vert=oct[fid[1]];}else{vert=oct[fid[2]];}
        let n = normalize(vert);

        // Stochastic orb size pulsing
        let size_noise = value_noise(vec3<f32>(f32(oi) * 7.0, time * 4.0, seed * 15.0));
        let noisy_radius = orb_radius * (1.0 + (size_noise - 0.5) * 0.2);
        position = orb_center + n * noisy_radius;
        normal = n;

        if(oi==0u){base_color=vec3<f32>(0.2,0.9,1.0);}
        else if(oi==1u){base_color=vec3<f32>(1.0,0.3,0.8);}
        else if(oi==2u){base_color=vec3<f32>(1.0,0.9,0.2);}
        else{base_color=vec3<f32>(0.3,1.0,0.4);}

        metallic = 0.7; roughness = 0.2;
        let pd = length(camera.view_position.xyz - orb_center);
        let ps = mix(1.5, 6.0, clamp(1.0-pd/5.0, 0.0, 1.0));
        emissive = (sin(time*ps + f32(oi)*1.5)*0.4+0.6) * 4.0 * orb_active;
    }

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    out.base_color = base_color;
    out.pbr_params = vec4<f32>(metallic, roughness, ao, emissive);
    return out;
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
fn fresnel_schlick_r(ct: f32, F0: vec3<f32>, r: f32) -> vec3<f32> {
    return F0+(max(vec3<f32>(1.0-r),F0)-F0)*pow(clamp(1.0-ct,0.0,1.0),5.0);
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
    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    var alb = in.base_color;
    let met = in.pbr_params.x;
    var rou = clamp(in.pbr_params.y, 0.045, 1.0);
    let ao  = in.pbr_params.z;
    let em  = in.pbr_params.w;
    let st  = u32(camera.view_position.w);
    let time = scene.params.x;
    let expo = scene.params.y;
    let ambi = scene.params.z;
    let nlit = u32(scene.params.w);
    let seed = scene.interaction.w;

    // ── Stochastic roughness micro-variation ──────────────────────
    let rough_noise = value_noise(in.world_pos * 8.0 + vec3<f32>(seed * 5.0, 0.0, 0.0));
    rou = clamp(rou + (rough_noise - 0.5) * 0.08, 0.045, 1.0);

    // ── Floor checker ─────────────────────────────────────────────
    if (N.y > 0.5 && em == 0.0) {
        let cx = floor(in.world_pos.x/2.0+0.001);
        let cz = floor(in.world_pos.z/2.0+0.001);
        if (((i32(cx)+i32(cz))%2+2)%2==0) { rou = 0.12; alb *= 1.35; }
        else { rou = 0.55; }
    }

    // ── Lighting ──────────────────────────────────────────────────
    var Lo = pbr_direct(N,V,normalize(scene.sun_direction.xyz),scene.sun_color.xyz*scene.sun_direction.w,alb,met,rou);
    if(nlit>=1u){Lo+=pt_light(N,V,in.world_pos,scene.light0_pos.xyz,scene.light0_color.xyz,scene.light0_color.w,alb,met,rou);}
    if(nlit>=2u){Lo+=pt_light(N,V,in.world_pos,scene.light1_pos.xyz,scene.light1_color.xyz,scene.light1_color.w,alb,met,rou);}
    if(nlit>=3u){Lo+=pt_light(N,V,in.world_pos,scene.light2_pos.xyz,scene.light2_color.xyz,scene.light2_color.w,alb,met,rou);}
    if(nlit>=4u){Lo+=pt_light(N,V,in.world_pos,scene.light3_pos.xyz,scene.light3_color.xyz,scene.light3_color.w,alb,met,rou);}

    // ── Ambient ───────────────────────────────────────────────────
    let F0a = mix(vec3<f32>(0.04),alb,met);
    let kSa = fresnel_schlick_r(max(dot(N,V),0.0),F0a,rou);
    let kDa = (vec3<f32>(1.0)-kSa)*(1.0-met);
    var sky = vec3<f32>(0.06,0.09,0.14); var gnd = vec3<f32>(0.02,0.02,0.03);
    if(st==1u){sky=vec3<f32>(0.09,0.04,0.16);gnd=vec3<f32>(0.03,0.01,0.06);}
    else if(st==2u){sky=vec3<f32>(0.16,0.09,0.04);gnd=vec3<f32>(0.06,0.03,0.01);}
    let hemi = mix(gnd, sky, N.y*0.5+0.5);
    var color = Lo + kDa*alb*hemi*ambi*ao + kSa*hemi*ambi*0.3;

    // ── Metallic sparkles (non-deterministic) ─────────────────────
    if (met > 0.5) {
        let sparkle_cell = floor(in.world_pos * 12.0);
        let sparkle_val = hash31(sparkle_cell + vec3<f32>(floor(time * 12.0), seed * 100.0, 0.0));
        if (sparkle_val > 0.96) {
            let intensity = (sparkle_val - 0.96) * 25.0;
            color += alb * intensity * 3.0;
        }
    }

    // ── Emissive ──────────────────────────────────────────────────
    if (em > 0.0) {
        if (in.world_pos.x > 9.5) {
            // Portal with quantum distortion
            let center = vec2<f32>(0.0, 1.5);
            let dist = length(in.world_pos.yz - center);

            // Deterministic base
            let ripple = sin(dist*6.0 - time*3.5)*0.5+0.5;
            let pulse = sin(time*1.5)*0.15+0.85;
            let glow = exp(-dist*0.35);

            // Non-deterministic distortion
            let portal_noise = fbm(vec3<f32>(in.world_pos.yz * 3.0, time * 2.5 + seed * 50.0));
            let quantum_flicker = value_noise(vec3<f32>(in.world_pos.yz * 8.0, time * 6.0 + seed * 30.0));

            var ec: vec3<f32>;
            if(st==0u){ec=vec3<f32>(0.4,0.65,1.0);}
            else if(st==1u){ec=vec3<f32>(0.8,0.35,1.0);}
            else{ec=vec3<f32>(1.0,0.6,0.2);}

            // Stochastic color shift in portal
            let hue_shift = hash21(vec2<f32>(floor(time*4.0), seed*7.0));
            ec = mix(ec, ec.yzx, hue_shift * 0.3);

            ec *= (1.0 + ripple*0.6 + portal_noise*0.4);
            let em_final = em * pulse * glow * (1.0 + quantum_flicker * 0.3);
            color += ec * em_final;
            if (dist < 1.5) { let c2 = 1.0-dist/1.5; color += ec * c2*c2 * 6.0 * (0.7 + quantum_flicker * 0.3); }
        } else {
            // Sphere / orb emissive with stochastic pulsing
            let em_noise = value_noise(vec3<f32>(in.world_pos * 4.0 + vec3<f32>(time*5.0, seed*20.0, 0.0)));
            color += alb * em * (0.8 + em_noise * 0.4);
        }
    }

    // ── Curvature-aware grid ──────────────────────────────────────
    if (em == 0.0) {
        var gc: vec2<f32>;
        if(abs(N.y)>0.5){gc=in.world_pos.xz;}
        else if(abs(N.z)>0.5){gc=in.world_pos.xy;}
        else{gc=in.world_pos.zy;}
        var gs = 2.0;
        if(st==1u){gs=2.0*(1.0+length(gc)*0.03);}
        else if(st==2u){gs=2.0/(1.0+length(gc)*0.02);}
        let lw = 0.025;
        let gx = fract(gc.x/gs); let gy = fract(gc.y/gs);
        if ((gx<lw||gx>1.0-lw)||(gy<lw||gy>1.0-lw)) {
            var gl: vec3<f32>;
            if(st==0u){gl=vec3<f32>(0.15,0.25,0.55);}
            else if(st==1u){gl=vec3<f32>(0.40,0.15,0.55);}
            else{gl=vec3<f32>(0.55,0.30,0.10);}

            // Stochastic grid brightness
            let grid_noise = hash21(floor(gc / gs) + vec2<f32>(floor(time * 2.0), seed * 3.0));
            color += gl * (0.25 + grid_noise * 0.2);
        }
    }

    // ── Noise-modulated fog ───────────────────────────────────────
    let vd = length(camera.view_position.xyz - in.world_pos);
    var fog_rate = select(select(0.012, 0.018, st==1u), 0.007, st==2u);
    // Non-deterministic fog density variation
    let fog_noise = fbm(in.world_pos * 0.15 + vec3<f32>(time * 0.3, 0.0, seed * 5.0));
    fog_rate *= (0.8 + fog_noise * 0.4);
    let fog_f = exp(-vd * fog_rate);
    var fog_c: vec3<f32>;
    if(st==0u){fog_c=vec3<f32>(0.015,0.02,0.05);}
    else if(st==1u){fog_c=vec3<f32>(0.03,0.015,0.06);}
    else{fog_c=vec3<f32>(0.06,0.03,0.015);}
    // Stochastic fog color tint
    let fog_tint = hash33(vec3<f32>(floor(in.world_pos.xz * 0.5), floor(time * 0.5)));
    fog_c += (fog_tint - 0.5) * 0.01;
    color = mix(fog_c, color, fog_f);

    // ── Tone-map + gamma ──────────────────────────────────────────
    color = aces(color * expo);
    color = to_srgb(color);

    // ── Film grain (non-deterministic post-process) ───────────────
    let screen_uv = in.clip_position.xy;
    let grain = hash21(screen_uv * 0.7 + vec2<f32>(time * 137.0, seed * 251.0));
    let grain_strength = select(select(0.025, 0.045, st==1u), 0.018, st==2u);
    color += (grain - 0.5) * grain_strength;

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}