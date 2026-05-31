// ─── Pest Control Shader v2 ────────────────────────────────────────────────
// PBR · 6 location themes · Death animations · Film grain · Noise

const PI: f32 = 3.14159265359;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,  // xyz=pos, w=location_id (0-5)
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
    pest_flash: vec4<f32>,      // >0 = hit flash, <0 = death progress (-1 = fully dead)
    pest_flash2: vec4<f32>,
    hud_info: vec4<f32>,         // x=resolution_x, y=resolution_y, z=combo_count, w=ammo
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> scene: SceneUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) pbr_params: vec4<f32>,
}

// ── Noise ─────────────────────────────────────────────────────────────────

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

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p); let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i), hash31(i + vec3<f32>(1.0,0.0,0.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0,1.0,0.0)), hash31(i + vec3<f32>(1.0,1.0,0.0)), u.x), u.y),
        mix(mix(hash31(i + vec3<f32>(0.0,0.0,1.0)), hash31(i + vec3<f32>(1.0,0.0,1.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0,1.0,1.0)), hash31(i + vec3<f32>(1.0,1.0,1.0)), u.x), u.y),
        u.z);
}

// ── Pest helpers ──────────────────────────────────────────────────────────

fn pest_radius(t: f32) -> f32 {
    if (t < 0.5) { return 0.0; }
    if (t < 1.5) { return 0.18; }
    if (t < 2.5) { return 0.10; }
    if (t < 3.5) { return 0.15; }
    if (t < 4.5) { return 0.35; }
    return 0.20;
}

fn pest_base_color(t: f32) -> vec3<f32> {
    if (t < 1.5) { return vec3<f32>(0.40, 0.22, 0.10); }
    if (t < 2.5) { return vec3<f32>(0.12, 0.04, 0.02); }
    if (t < 3.5) { return vec3<f32>(0.18, 0.18, 0.22); }
    if (t < 4.5) { return vec3<f32>(0.38, 0.28, 0.18); }
    return vec3<f32>(0.85, 0.75, 0.10);
}

fn pest_metallic(t: f32) -> f32 {
    if (t < 1.5) { return 0.6; }
    if (t < 2.5) { return 0.3; }
    if (t < 3.5) { return 0.4; }
    if (t < 4.5) { return 0.1; }
    return 0.5;
}

// ── Location-based room colors ────────────────────────────────────────────

struct RoomTheme {
    floor_a: vec3<f32>,
    floor_b: vec3<f32>,
    ceiling: vec3<f32>,
    wall_back: vec3<f32>,
    wall_front: vec3<f32>,
    wall_left: vec3<f32>,
    wall_right: vec3<f32>,
    floor_metallic: f32,
    floor_roughness_a: f32,
    floor_roughness_b: f32,
}

fn get_theme(loc: u32) -> RoomTheme {
    var t: RoomTheme;
    if (loc == 0u) {
        // Kitchen – warm terracotta tile
        t.floor_a = vec3<f32>(0.65, 0.55, 0.42);
        t.floor_b = vec3<f32>(0.55, 0.40, 0.30);
        t.ceiling = vec3<f32>(0.88, 0.86, 0.82);
        t.wall_back = vec3<f32>(0.35, 0.25, 0.15);
        t.wall_front = vec3<f32>(0.75, 0.72, 0.65);
        t.wall_left = vec3<f32>(0.72, 0.68, 0.58);
        t.wall_right = vec3<f32>(0.50, 0.55, 0.52);
        t.floor_metallic = 0.02; t.floor_roughness_a = 0.15; t.floor_roughness_b = 0.35;
    } else if (loc == 1u) {
        // Bathroom – cool blue-white tile
        t.floor_a = vec3<f32>(0.78, 0.82, 0.85);
        t.floor_b = vec3<f32>(0.60, 0.68, 0.75);
        t.ceiling = vec3<f32>(0.92, 0.92, 0.92);
        t.wall_back = vec3<f32>(0.65, 0.72, 0.78);
        t.wall_front = vec3<f32>(0.80, 0.82, 0.85);
        t.wall_left = vec3<f32>(0.75, 0.78, 0.82);
        t.wall_right = vec3<f32>(0.70, 0.75, 0.80);
        t.floor_metallic = 0.15; t.floor_roughness_a = 0.08; t.floor_roughness_b = 0.25;
    } else if (loc == 2u) {
        // Basement – dark concrete
        t.floor_a = vec3<f32>(0.30, 0.28, 0.26);
        t.floor_b = vec3<f32>(0.22, 0.20, 0.18);
        t.ceiling = vec3<f32>(0.35, 0.33, 0.30);
        t.wall_back = vec3<f32>(0.28, 0.26, 0.24);
        t.wall_front = vec3<f32>(0.32, 0.30, 0.28);
        t.wall_left = vec3<f32>(0.25, 0.23, 0.22);
        t.wall_right = vec3<f32>(0.30, 0.28, 0.25);
        t.floor_metallic = 0.0; t.floor_roughness_a = 0.7; t.floor_roughness_b = 0.85;
    } else if (loc == 3u) {
        // Attic – warm wood
        t.floor_a = vec3<f32>(0.55, 0.40, 0.25);
        t.floor_b = vec3<f32>(0.45, 0.32, 0.18);
        t.ceiling = vec3<f32>(0.50, 0.38, 0.22);
        t.wall_back = vec3<f32>(0.48, 0.35, 0.20);
        t.wall_front = vec3<f32>(0.60, 0.50, 0.35);
        t.wall_left = vec3<f32>(0.52, 0.40, 0.25);
        t.wall_right = vec3<f32>(0.52, 0.40, 0.25);
        t.floor_metallic = 0.05; t.floor_roughness_a = 0.3; t.floor_roughness_b = 0.55;
    } else if (loc == 4u) {
        // Garden – green/natural
        t.floor_a = vec3<f32>(0.30, 0.42, 0.22);
        t.floor_b = vec3<f32>(0.25, 0.35, 0.18);
        t.ceiling = vec3<f32>(0.55, 0.65, 0.80);
        t.wall_back = vec3<f32>(0.35, 0.45, 0.30);
        t.wall_front = vec3<f32>(0.40, 0.50, 0.35);
        t.wall_left = vec3<f32>(0.38, 0.48, 0.32);
        t.wall_right = vec3<f32>(0.38, 0.48, 0.32);
        t.floor_metallic = 0.0; t.floor_roughness_a = 0.6; t.floor_roughness_b = 0.8;
    } else {
        // Restaurant – polished dark
        t.floor_a = vec3<f32>(0.15, 0.12, 0.10);
        t.floor_b = vec3<f32>(0.25, 0.20, 0.15);
        t.ceiling = vec3<f32>(0.20, 0.18, 0.15);
        t.wall_back = vec3<f32>(0.30, 0.15, 0.10);
        t.wall_front = vec3<f32>(0.22, 0.18, 0.14);
        t.wall_left = vec3<f32>(0.25, 0.20, 0.15);
        t.wall_right = vec3<f32>(0.25, 0.20, 0.15);
        t.floor_metallic = 0.3; t.floor_roughness_a = 0.05; t.floor_roughness_b = 0.15;
    }
    return t;
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
    let loc = u32(camera.view_position.w);
    let theme = get_theme(loc);

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

    if (idx < 36u) {
        // ── Room ──────────────────────────────────────────────────
        let face = idx / 6u;
        let v = qv[idx % 6u];

        if (face == 0u) {
            position = vec3<f32>(v.x*room, 0.0, v.y*room);
            normal = vec3<f32>(0.0,1.0,0.0);
            // Checker floor using theme colors
            let cx = floor(position.x / 1.5 + 0.001);
            let cz = floor(position.z / 1.5 + 0.001);
            if (((i32(cx)+i32(cz))%2+2)%2 == 0) {
                base_color = theme.floor_a; roughness = theme.floor_roughness_a;
            } else {
                base_color = theme.floor_b; roughness = theme.floor_roughness_b;
            }
            metallic = theme.floor_metallic;
        } else if (face == 1u) {
            position = vec3<f32>(v.x*room, 3.0, v.y*room);
            normal = vec3<f32>(0.0,-1.0,0.0);
            base_color = theme.ceiling; roughness = 0.9;
        } else if (face == 2u) {
            position = vec3<f32>(v.x*room, v.y*1.5+1.5, room);
            normal = vec3<f32>(0.0,0.0,-1.0);
            base_color = theme.wall_back; roughness = 0.55;
        } else if (face == 3u) {
            position = vec3<f32>(v.x*room, v.y*1.5+1.5, -room);
            normal = vec3<f32>(0.0,0.0,1.0);
            base_color = theme.wall_front; roughness = 0.7;
            // Window glow
            if (abs(v.x) < 0.5 && v.y > -0.2) {
                emissive = 1.5;
                base_color = mix(base_color, vec3<f32>(0.7,0.8,0.95), 0.6);
            }
        } else if (face == 4u) {
            position = vec3<f32>(-room, v.y*1.5+1.5, v.x*room);
            normal = vec3<f32>(1.0,0.0,0.0);
            base_color = theme.wall_left; roughness = 0.65;
        } else {
            position = vec3<f32>(room, v.y*1.5+1.5, v.x*room);
            normal = vec3<f32>(-1.0,0.0,0.0);
            base_color = theme.wall_right;
            metallic = 0.15; roughness = 0.25;
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

        var pd: vec4<f32>;
        if (pest_idx==0u){pd=scene.pest0;} else if(pest_idx==1u){pd=scene.pest1;}
        else if(pest_idx==2u){pd=scene.pest2;} else if(pest_idx==3u){pd=scene.pest3;}
        else if(pest_idx==4u){pd=scene.pest4;} else if(pest_idx==5u){pd=scene.pest5;}
        else if(pest_idx==6u){pd=scene.pest6;} else{pd=scene.pest7;}

        let center = pd.xyz;
        let pest_type = pd.w;
        var radius = pest_radius(pest_type);

        var flash = 0.0;
        if(pest_idx==0u){flash=scene.pest_flash.x;} else if(pest_idx==1u){flash=scene.pest_flash.y;}
        else if(pest_idx==2u){flash=scene.pest_flash.z;} else if(pest_idx==3u){flash=scene.pest_flash.w;}
        else if(pest_idx==4u){flash=scene.pest_flash2.x;} else if(pest_idx==5u){flash=scene.pest_flash2.y;}
        else if(pest_idx==6u){flash=scene.pest_flash2.z;} else{flash=scene.pest_flash2.w;}

        // Death animation: flash < 0 means dying
        var death_factor = 1.0;
        if (flash < 0.0) {
            death_factor = 1.0 + flash; // goes from 1.0 → 0.0 as flash goes from 0 → -1
            death_factor = max(death_factor, 0.0);
            radius *= death_factor;
        }

        let tri = vert_local / 3u;
        let vi = vert_local % 3u;
        let fid = fi[tri];
        var vert: vec3<f32>;
        if(vi==0u){vert=oct[fid[0]];} else if(vi==1u){vert=oct[fid[1]];} else{vert=oct[fid[2]];}

        let n = normalize(vert);
        var scale = vec3<f32>(1.0, 0.5, 1.0);
        if (pest_type > 3.5 && pest_type < 4.5) { scale = vec3<f32>(1.2,0.6,0.8); }
        if (pest_type > 4.5) { scale = vec3<f32>(0.7,1.0,1.2); }

        // Death spin
        if (flash < 0.0) {
            let spin = (1.0 - death_factor) * 12.0;
            let cs = cos(spin); let sn = sin(spin);
            scale = vec3<f32>(scale.x * cs - scale.z * sn, scale.y * death_factor, scale.x * sn + scale.z * cs);
        }

        let sv = n * scale;
        position = center + normalize(sv) * radius;
        normal = normalize(sv);

        base_color = pest_base_color(pest_type);
        metallic = pest_metallic(pest_type);
        roughness = 0.35;

        // Hit flash: white-orange flash
        if (flash > 0.0) {
            base_color = mix(base_color, vec3<f32>(1.0, 0.4, 0.15), flash);
            emissive = flash * 8.0;
        }
        // Death: turn dark red
        if (flash < 0.0) {
            base_color = mix(base_color, vec3<f32>(0.6, 0.05, 0.0), 1.0 - death_factor);
            emissive = (1.0 - death_factor) * 3.0;
        }

        // Proximity glow
        if (pest_type > 0.5 && flash >= 0.0) {
            let d = length(camera.view_position.xyz - center);
            emissive += max(0.0, 1.0 - d / 4.0) * 0.5;
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
        let s = 0.018; let t2 = 0.0025;
        var px = array<f32,6>(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
        var py = array<f32,6>(-1.0,-1.0, 1.0, -1.0, 1.0,  1.0);
        var cx: f32; var cy: f32;
        if (bar == 0u) { cx = px[vi]*s; cy = py[vi]*t2; }
        else           { cx = px[vi]*t2; cy = py[vi]*s; }

        let firing = scene.params.w;
        var ch_color = vec3<f32>(1.0);
        if (firing > 0.1) { ch_color = vec3<f32>(1.0, 0.4, 0.2); }

        out.clip_position = vec4<f32>(cx, cy, 0.0005, 1.0);
        out.world_pos = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0,0.0,1.0);
        out.base_color = ch_color;
        out.pbr_params = vec4<f32>(0.0, 0.5, 1.0, 20.0);
        return out;
    }
}

// ─── PBR ──────────────────────────────────────────────────────────────────

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, r: f32) -> f32 {
    let a=r*r;let a2=a*a;let d=max(dot(N,H),0.0);let x=d*d*(a2-1.0)+1.0;return a2/(PI*x*x+0.0001);
}
fn geometry_schlick(d: f32, r: f32) -> f32 {
    let k=((r+1.0)*(r+1.0))/8.0;return d/(d*(1.0-k)+k);
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
fn to_srgb(c: vec3<f32>) -> vec3<f32> { return pow(c,vec3<f32>(1.0/2.2)); }

fn pbr_direct(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, rad: vec3<f32>, alb: vec3<f32>, met: f32, rou: f32) -> vec3<f32> {
    let H=normalize(V+L);let F0=mix(vec3<f32>(0.04),alb,met);
    let D=distribution_ggx(N,H,rou);let G=geometry_smith(N,V,L,rou);
    let F=fresnel_schlick(max(dot(H,V),0.0),F0);
    let NdV=max(dot(N,V),0.0);let NdL=max(dot(N,L),0.0);
    let spec=(D*G*F)/(4.0*NdV*NdL+0.0001);
    return ((vec3<f32>(1.0)-F)*(1.0-met)*alb/PI+spec)*rad*NdL;
}
fn pt_light(N: vec3<f32>, V: vec3<f32>, wp: vec3<f32>, lp: vec3<f32>, lc: vec3<f32>, li: f32, a: vec3<f32>, m: f32, r: f32) -> vec3<f32> {
    let d=length(lp-wp);return pbr_direct(N,V,normalize(lp-wp),lc*li/(d*d+1.0),a,m,r);
}

// ─── SDF Score HUD Overlay ────────────────────────────────────────────────

fn sdf_box_hud(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdf_seg_h(p: vec2<f32>, w: f32, t: f32) -> f32 { return sdf_box_hud(p, vec2<f32>(w, t)); }
fn sdf_seg_v(p: vec2<f32>, h: f32, t: f32) -> f32 { return sdf_box_hud(p, vec2<f32>(t, h)); }

fn seg_mask(digit: u32) -> u32 {
    var m = array<u32, 10>(0x3Fu,0x06u,0x5Bu,0x4Fu,0x66u,0x6Du,0x7Du,0x07u,0x7Fu,0x6Fu);
    if (digit > 9u) { return 0x00u; }
    return m[digit];
}

fn hud_digit(p: vec2<f32>, digit: u32) -> f32 {
    let sw=0.32; let sh=0.06; let vw=0.06; let vh=0.28;
    let mask = seg_mask(digit);
    var d = 999.0;
    if((mask&0x01u)!=0u){d=min(d,sdf_seg_h(p-vec2<f32>(0.0,0.7),sw,sh));}
    if((mask&0x02u)!=0u){d=min(d,sdf_seg_v(p-vec2<f32>(0.3,0.35),vh,vw));}
    if((mask&0x04u)!=0u){d=min(d,sdf_seg_v(p-vec2<f32>(0.3,-0.35),vh,vw));}
    if((mask&0x08u)!=0u){d=min(d,sdf_seg_h(p-vec2<f32>(0.0,-0.7),sw,sh));}
    if((mask&0x10u)!=0u){d=min(d,sdf_seg_v(p-vec2<f32>(-0.3,-0.35),vh,vw));}
    if((mask&0x20u)!=0u){d=min(d,sdf_seg_v(p-vec2<f32>(-0.3,0.35),vh,vw));}
    if((mask&0x40u)!=0u){d=min(d,sdf_seg_h(p-vec2<f32>(0.0,0.0),sw,sh));}
    return smoothstep(0.02,-0.02,d);
}

fn hud_number(p: vec2<f32>, value: u32, max_digits: u32) -> f32 {
    var v = value; var result = 0.0;
    for (var i = 0u; i < max_digits; i++) {
        let digit = v % 10u; v = v / 10u;
        let offset = f32(max_digits-1u-i) * 0.9;
        result = max(result, hud_digit(p - vec2<f32>(offset, 0.0), digit));
        if (v == 0u && i > 0u) { break; }
    }
    return result;
}

fn hud_colon(p: vec2<f32>) -> f32 {
    let d1 = length(p - vec2<f32>(0.0, 0.25)) - 0.07;
    let d2 = length(p - vec2<f32>(0.0, -0.25)) - 0.07;
    return smoothstep(0.02, -0.02, min(d1, d2));
}

fn score_hud(frag_pos: vec2<f32>, res: vec2<f32>, score_v: f32, level_v: f32, pests_v: f32, time_v: f32, combo_v: f32, firing: f32, ammo_v: f32) -> vec4<f32> {
    // Panel in pixels: top-right corner
    let panel_w = 360.0;
    let panel_h = 120.0;
    let margin = 12.0;
    let corner_r = 10.0;

    let pl = res.x - panel_w - margin;
    let pt = margin;
    let pr = res.x - margin;
    let pb = pt + panel_h;

    let fx = frag_pos.x;
    let fy = frag_pos.y;

    // Bounds check
    if (fx < pl || fx > pr || fy < pt || fy > pb) { return vec4<f32>(0.0); }

    // Pixel coords within panel (origin top-left)
    let lx = fx - pl;
    let ly = fy - pt;

    // Rounded corners
    let dx = max(corner_r - lx, lx - (panel_w - corner_r));
    let dy = max(corner_r - ly, ly - (panel_h - corner_r));
    if (dx > 0.0 && dy > 0.0 && length(vec2<f32>(dx, dy)) > corner_r) {
        return vec4<f32>(0.0);
    }

    // Background
    var col = vec3<f32>(0.03, 0.03, 0.06);
    var alp = 0.7;

    // Border glow (2px edge)
    let edge = min(min(lx, panel_w - lx), min(ly, panel_h - ly));
    if (edge < 2.5) {
        var border_col = vec3<f32>(0.15, 0.3, 0.5);
        if (firing > 0.05) {
            border_col = mix(border_col, vec3<f32>(1.0, 0.5, 0.15), firing);
        }
        col = mix(border_col, col, smoothstep(0.0, 2.5, edge));
    }

    // Hit flash: whole panel pulses when firing
    if (firing > 0.05) {
        col += vec3<f32>(0.15, 0.06, 0.02) * firing;
    }

    // SDF coordinates: 1 unit = 20px, y-up, origin at panel bottom-left
    let unit = 20.0;
    let sx = lx / unit;
    let sy = (panel_h - ly) / unit;

    // ── Score: large digits, top row, center-right ──
    var score_col = vec3<f32>(0.95, 0.9, 0.7);
    if (combo_v > 1.5) { score_col = vec3<f32>(1.0, 0.75, 0.15); }
    let score_p = vec2<f32>(sx, sy) - vec2<f32>(9.5, 4.2);
    let score_d = hud_number(score_p, u32(score_v), 6u);
    if (score_d > 0.0) { col = score_col; alp = score_d; }

    // ── Level: bottom-left ──
    let level_p = vec2<f32>(sx, sy) - vec2<f32>(1.2, 1.5);
    let level_d = hud_number(level_p, u32(level_v), 2u);
    if (level_d > 0.0) { col = vec3<f32>(0.4, 0.8, 1.0); alp = level_d; }

    // ── Pests alive: bottom-center ──
    let pests_p = vec2<f32>(sx, sy) - vec2<f32>(7.0, 1.5);
    let pests_d = hud_number(pests_p, u32(pests_v), 2u);
    if (pests_d > 0.0) { col = vec3<f32>(1.0, 0.5, 0.4); alp = pests_d; }

    // ── Timer: bottom-right (M:SS) ──
    let tv = u32(max(time_v, 0.0));
    let tm = tv / 60u;
    let ts = tv % 60u;
    let min_p = vec2<f32>(sx, sy) - vec2<f32>(12.5, 1.5);
    let min_d = hud_number(min_p, tm, 1u);
    let colon_p = vec2<f32>(sx, sy) - vec2<f32>(13.6, 1.5);
    let colon_d = hud_colon(colon_p);
    let sec_p = vec2<f32>(sx, sy) - vec2<f32>(14.5, 1.5);
    let sec_d = hud_number(sec_p, ts, 2u);

    var timer_col = vec3<f32>(0.5, 0.9, 0.5);
    if (time_v < 30.0) { timer_col = vec3<f32>(1.0, 0.55, 0.2); }
    if (time_v < 10.0) { timer_col = vec3<f32>(1.0, 0.2, 0.15); }
    let timer_d = max(max(min_d, sec_d), colon_d);
    if (timer_d > 0.0) { col = timer_col; alp = timer_d; }

    // ── Combo badge: top-left "xN" ──
    if (combo_v > 1.5) {
        let cv = u32(min(combo_v, 5.0));
        let combo_p = vec2<f32>(sx, sy) - vec2<f32>(1.2, 4.2);
        let combo_d = hud_number(combo_p, cv, 1u);
        if (combo_d > 0.0) { col = vec3<f32>(1.0, 0.65, 0.1); alp = combo_d; }
    }

    // ── Ammo: top-left area (below combo or standalone) ──
    let ammo_p = vec2<f32>(sx, sy) - vec2<f32>(3.5, 4.2);
    let ammo_d = hud_number(ammo_p, u32(ammo_v), 3u);
    var ammo_col = vec3<f32>(0.6, 0.8, 0.5);
    if (ammo_v < 10.0) { ammo_col = vec3<f32>(1.0, 0.3, 0.2); }
    else if (ammo_v < 30.0) { ammo_col = vec3<f32>(1.0, 0.7, 0.3); }
    if (ammo_d > 0.0) { col = ammo_col; alp = ammo_d; }

    return vec4<f32>(col, alp);
}

// ─── Fragment Shader ──────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let em = in.pbr_params.w;
    if (em >= 15.0) { return vec4<f32>(in.base_color, 1.0); } // crosshair

    let N = normalize(in.normal);
    let V = normalize(camera.view_position.xyz - in.world_pos);
    var alb = in.base_color;
    let met = in.pbr_params.x;
    var rou = clamp(in.pbr_params.y, 0.045, 1.0);
    let ao  = in.pbr_params.z;
    let time = scene.params.x;
    let expo = scene.params.y;
    let ambi = scene.params.z;
    let firing = scene.params.w;
    let loc = u32(camera.view_position.w);

    // Micro-roughness noise
    let rn = value_noise(in.world_pos * 6.0);
    rou = clamp(rou + (rn - 0.5) * 0.06, 0.045, 1.0);

    // ── Lighting ──────────────────────────────────────────────────
    var Lo = pbr_direct(N,V,normalize(scene.sun_direction.xyz),
        scene.sun_color.xyz*scene.sun_direction.w, alb, met, rou);
    Lo += pt_light(N,V,in.world_pos, scene.light0_pos.xyz,
        scene.light0_color.xyz, scene.light0_color.w, alb, met, rou);
    Lo += pt_light(N,V,in.world_pos, scene.light1_pos.xyz,
        scene.light1_color.xyz, scene.light1_color.w, alb, met, rou);

    // ── Ambient ───────────────────────────────────────────────────
    let F0a = mix(vec3<f32>(0.04), alb, met);
    let kSa = fresnel_schlick(max(dot(N,V),0.0), F0a);
    let kDa = (vec3<f32>(1.0)-kSa)*(1.0-met);

    var sky = vec3<f32>(0.12,0.11,0.09); var gnd = vec3<f32>(0.04,0.035,0.03);
    // Location-specific ambient
    if (loc == 2u) { sky = vec3<f32>(0.05,0.04,0.04); gnd = vec3<f32>(0.02,0.02,0.02); }
    else if (loc == 4u) { sky = vec3<f32>(0.15,0.18,0.22); gnd = vec3<f32>(0.06,0.08,0.05); }
    else if (loc == 5u) { sky = vec3<f32>(0.08,0.06,0.04); gnd = vec3<f32>(0.03,0.02,0.02); }
    let hemi = mix(gnd, sky, N.y*0.5+0.5);
    var color = Lo + kDa*alb*hemi*ambi*ao + kSa*hemi*ambi*0.2;

    if (em > 0.0) { color += alb * em; }

    // ── Grid pattern on floor ─────────────────────────────────────
    if (N.y > 0.5 && em == 0.0) {
        let gs = 1.5; let lw = 0.02;
        let gx = fract(in.world_pos.x/gs); let gz = fract(in.world_pos.z/gs);
        if ((gx<lw||gx>1.0-lw)||(gz<lw||gz>1.0-lw)) {
            color += vec3<f32>(0.08, 0.06, 0.04);
        }
    }

    // ── Firing flash ──────────────────────────────────────────────
    if (firing > 0.0) {
        color += vec3<f32>(0.6,0.4,0.1) * firing * 0.2;
    }

    // ── Distance fog ──────────────────────────────────────────────
    let vd = length(camera.view_position.xyz - in.world_pos);
    var fog_c = vec3<f32>(0.06,0.055,0.05);
    if (loc == 2u) { fog_c = vec3<f32>(0.03,0.03,0.03); }
    else if (loc == 4u) { fog_c = vec3<f32>(0.08,0.10,0.12); }
    let fog_f = exp(-vd * 0.02);
    color = mix(fog_c, color, fog_f);

    color = aces(color * expo);
    color = to_srgb(color);

    // ── Film grain ────────────────────────────────────────────────
    let grain = hash21(in.clip_position.xy * 0.7 + vec2<f32>(time * 137.0, 0.0));
    color += (grain - 0.5) * 0.02;

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    // ── Score HUD overlay ─────────────────────────────────────────
    let res = scene.hud_info.xy;
    let combo_hud = scene.hud_info.z;
    let ammo_hud = scene.hud_info.w;
    if (res.x > 0.0 && res.y > 0.0) {
        let gi = scene.game_info;
        let hud = score_hud(in.clip_position.xy, res, gi.x, gi.y, gi.z, gi.w, combo_hud, firing, ammo_hud);
        color = mix(color, hud.rgb, hud.a);
    }

    return vec4<f32>(color, 1.0);
}
