// Camera uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Non-Euclidean vertex shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Create a cube room (6 faces * 6 vertices each = 36 vertices)
    let face = in_vertex_index / 6u;
    let vertex = in_vertex_index % 6u;
    
    // Define quad vertices for each face
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    
    let v = quad_verts[vertex];
    let room_size = 10.0;
    
    var position: vec3<f32>;
    var normal: vec3<f32>;
    
    // Build each face of the room
    if (face == 0u) { // Floor
        position = vec3<f32>(v.x * room_size, -2.0, v.y * room_size);
        normal = vec3<f32>(0.0, 1.0, 0.0);
    } else if (face == 1u) { // Ceiling
        position = vec3<f32>(v.x * room_size, 5.0, v.y * room_size);
        normal = vec3<f32>(0.0, -1.0, 0.0);
    } else if (face == 2u) { // Front wall
        position = vec3<f32>(v.x * room_size, v.y * 3.5 + 1.5, room_size);
        normal = vec3<f32>(0.0, 0.0, -1.0);
    } else if (face == 3u) { // Back wall
        position = vec3<f32>(v.x * room_size, v.y * 3.5 + 1.5, -room_size);
        normal = vec3<f32>(0.0, 0.0, 1.0);
    } else if (face == 4u) { // Left wall
        position = vec3<f32>(-room_size, v.y * 3.5 + 1.5, v.x * room_size);
        normal = vec3<f32>(1.0, 0.0, 0.0);
    } else { // Right wall (portal wall)
        position = vec3<f32>(room_size, v.y * 3.5 + 1.5, v.x * room_size);
        normal = vec3<f32>(-1.0, 0.0, 0.0);
    }
    
    // Transform to clip space
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    
    // Different colors for each wall
    if (face == 0u) { // Floor - dark gray
        out.color = vec4<f32>(0.2, 0.2, 0.2, 1.0);
    } else if (face == 1u) { // Ceiling - lighter gray
        out.color = vec4<f32>(0.3, 0.3, 0.3, 1.0);
    } else if (face == 5u) { // Right wall (portal) - purple tint
        out.color = vec4<f32>(0.6, 0.4, 0.8, 1.0);
    } else { // Other walls - blue-gray
        out.color = vec4<f32>(0.4, 0.4, 0.5, 1.0);
    }
    
    return out;
}

// Fragment shader with non-Euclidean effects
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create a grid pattern on surfaces
    let grid_scale = 2.0;
    var grid_coord: vec2<f32>;
    
    // Choose grid coordinates based on surface normal
    if (abs(in.normal.y) > 0.5) { // Floor/ceiling
        grid_coord = in.world_pos.xz;
    } else if (abs(in.normal.z) > 0.5) { // Front/back walls
        grid_coord = in.world_pos.xy;
    } else { // Left/right walls
        grid_coord = in.world_pos.zy;
    }
    
    let grid_x = fract(grid_coord.x / grid_scale);
    let grid_y = fract(grid_coord.y / grid_scale);
    
    // Create grid lines
    let line_width = 0.02;
    let is_grid_line = (grid_x < line_width || grid_x > 1.0 - line_width) ||
                       (grid_y < line_width || grid_y > 1.0 - line_width);
    
    var final_color = in.color;
    
    // Basic lighting
    let light_dir = normalize(vec3<f32>(0.5, 0.7, 0.3));
    let diffuse = max(dot(in.normal, light_dir), 0.0) * 0.5 + 0.5;
    final_color = final_color * diffuse;
    
    if (is_grid_line) {
        // Make grid lines slightly brighter
        final_color = final_color * 1.3;
    }
    
    // Distance fog for depth perception
    let view_distance = length(camera.view_position.xyz - in.world_pos);
    let fog_factor = exp(-view_distance * 0.02);
    let fog_color = vec4<f32>(0.05, 0.05, 0.1, 1.0);
    final_color = mix(fog_color, final_color, fog_factor);
    
    return final_color;
}