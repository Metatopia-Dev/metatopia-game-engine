// Camera uniform buffer
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>, // xyz = position, w = space_type (0=Euclidean, 1=Hyperbolic, 2=Spherical)
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
    } else { // Right wall (portal wall to hyperbolic space)
        // Create a portal area in the center of the wall
        let portal_size = 0.4; // Make portal smaller portion of wall
        let portal_v = v * portal_size;
        position = vec3<f32>(room_size, portal_v.y * 3.5 + 1.5, portal_v.x * room_size * 0.3);
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
    } else if (face == 5u) { // Right wall (portal) - glowing purple portal
        out.color = vec4<f32>(0.9, 0.4, 1.0, 1.0);
    } else { // Other walls - blue-gray
        out.color = vec4<f32>(0.4, 0.4, 0.5, 1.0);
    }
    
    return out;
}

// Fragment shader with non-Euclidean effects
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = in.color;
    
    // Check if this is a portal (bright purple color)
    let is_portal = in.color.r > 0.8 && in.color.b > 0.9;
    
    if (is_portal) {
        // Animated portal effect
        let time = camera.view_position.w; // We can use w component for time later
        let center = vec2<f32>(0.0, 1.5); // Portal center in world space
        let dist_from_center = length(in.world_pos.yz - center);
        
        // Create ripple effect
        let ripple = sin(dist_from_center * 3.0 - time * 2.0) * 0.5 + 0.5;
        
        // Glowing edge effect
        let edge_glow = 1.0 - smoothstep(0.0, 3.0, dist_from_center);
        
        // Combine effects
        final_color = vec4<f32>(
            0.6 + ripple * 0.4,
            0.2 + edge_glow * 0.3,
            0.9 + ripple * 0.1,
            1.0
        );
        
        // Add bright center
        if (dist_from_center < 1.5) {
            let center_brightness = 1.0 - (dist_from_center / 1.5);
            final_color += vec4<f32>(center_brightness * 0.3, center_brightness * 0.5, center_brightness * 0.2, 0.0);
        }
    } else {
        // Regular surface with grid pattern
        // Get space type from camera uniform (w component)
        let space_type = u32(camera.view_position.w);
        
        var grid_scale = 2.0;
        var line_width = 0.02;
        var grid_coord: vec2<f32>;
        
        // Choose grid coordinates based on surface normal
        if (abs(in.normal.y) > 0.5) { // Floor/ceiling
            grid_coord = in.world_pos.xz;
        } else if (abs(in.normal.z) > 0.5) { // Front/back walls
            grid_coord = in.world_pos.xy;
        } else { // Left/right walls
            grid_coord = in.world_pos.zy;
        }
        
        var is_grid_line = false;
        
        // Different patterns for different spaces
        if (space_type == 0u) { // Euclidean - rectangular grid
            let grid_x = fract(grid_coord.x / grid_scale);
            let grid_y = fract(grid_coord.y / grid_scale);
            is_grid_line = (grid_x < line_width || grid_x > 1.0 - line_width) ||
                          (grid_y < line_width || grid_y > 1.0 - line_width);
        } else if (space_type == 1u) { // Hyperbolic - radial pattern
            let center = vec2<f32>(0.0, 0.0);
            let dist = length(grid_coord - center);
            let angle = atan2(grid_coord.y - center.y, grid_coord.x - center.x);
            
            // Radial lines
            let radial_lines = fract(angle / (3.14159 / 6.0)); // 12 radial lines
            let circular_lines = fract(dist / grid_scale);
            
            is_grid_line = (radial_lines < line_width * 2.0 || radial_lines > 1.0 - line_width * 2.0) ||
                          (circular_lines < line_width || circular_lines > 1.0 - line_width);
        } else { // Spherical - hexagonal pattern
            // Simplified hex grid
            let hex_x = grid_coord.x * 1.15;
            let hex_y = grid_coord.y;
            let col = floor(hex_x / grid_scale);
            let row = floor(hex_y / grid_scale);
            
            let hex_grid_x = fract(hex_x / grid_scale);
            let hex_grid_y = fract(hex_y / grid_scale);
            
            // Create diamond pattern as simplified hex
            let diamond = abs(hex_grid_x - 0.5) + abs(hex_grid_y - 0.5);
            is_grid_line = diamond < line_width * 2.0;
        }
        
        // Apply space-specific color schemes
        if (space_type == 0u) { // Euclidean - blue tones
            final_color = final_color * vec4<f32>(0.7, 0.8, 1.0, 1.0);
        } else if (space_type == 1u) { // Hyperbolic - purple tones
            final_color = final_color * vec4<f32>(1.0, 0.7, 1.0, 1.0);
        } else { // Spherical - orange/warm tones
            final_color = final_color * vec4<f32>(1.0, 0.9, 0.7, 1.0);
        }
        
        // Basic lighting
        let light_dir = normalize(vec3<f32>(0.5, 0.7, 0.3));
        let diffuse = max(dot(in.normal, light_dir), 0.0) * 0.5 + 0.5;
        final_color = final_color * diffuse;
        
        if (is_grid_line) {
            // Make grid lines brighter and colored based on space
            if (space_type == 0u) {
                final_color = vec4<f32>(0.5, 0.7, 1.0, 1.0); // Bright blue lines
            } else if (space_type == 1u) {
                final_color = vec4<f32>(1.0, 0.5, 1.0, 1.0); // Bright purple lines
            } else {
                final_color = vec4<f32>(1.0, 0.8, 0.4, 1.0); // Bright orange lines
            }
        }
        
        // Distance fog for depth perception
        let view_distance = length(camera.view_position.xyz - in.world_pos);
        let fog_factor = exp(-view_distance * 0.02);
        let fog_color = vec4<f32>(0.05, 0.05, 0.1, 1.0);
        final_color = mix(fog_color, final_color, fog_factor);
    }
    
    return final_color;
}