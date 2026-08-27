//! Dynamic Multi-Light Manager for GPU Shaders
//!
//! Manages Point Lights, Spot Lights, and Directional Sun Lights,
//! providing distance culling and GPU uniform/storage buffer alignment.

use bytemuck::{Pod, Zeroable};

/// GPU-aligned representation of a single light for WGSL shaders
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    /// Position (xyz) and Light Type (w: 0.0 = Point, 1.0 = Spot, 2.0 = Directional)
    pub pos_type: [f32; 4],
    /// Color (rgb) and Intensity (w)
    pub color_intensity: [f32; 4],
    /// Direction (xyz) and Radius/Range (w)
    pub dir_radius: [f32; 4],
    /// Spot Angles (x: inner cos, y: outer cos, z: unused, w: unused)
    pub spot_params: [f32; 4],
}

/// Point Light emitting in all directions with spherical attenuation
#[derive(Debug, Clone)]
pub struct PointLight {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

impl PointLight {
    pub fn new(pos: [f32; 3], color: [f32; 3], intensity: f32, radius: f32) -> Self {
        Self { pos, color, intensity, radius }
    }

    pub fn to_gpu(&self) -> GpuLight {
        GpuLight {
            pos_type: [self.pos[0], self.pos[1], self.pos[2], 0.0],
            color_intensity: [self.color[0], self.color[1], self.color[2], self.intensity],
            dir_radius: [0.0, 0.0, 0.0, self.radius],
            spot_params: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Conical Spot Light emitting in a specific direction
#[derive(Debug, Clone)]
pub struct SpotLight {
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_angle_rad: f32,
    pub outer_angle_rad: f32,
}

impl SpotLight {
    pub fn new(pos: [f32; 3], dir: [f32; 3], color: [f32; 3], intensity: f32, range: f32, inner_rad: f32, outer_rad: f32) -> Self {
        Self { pos, dir, color, intensity, range, inner_angle_rad: inner_rad, outer_angle_rad: outer_rad }
    }

    pub fn to_gpu(&self) -> GpuLight {
        let len = (self.dir[0]*self.dir[0] + self.dir[1]*self.dir[1] + self.dir[2]*self.dir[2]).sqrt().max(0.001);
        let norm_dir = [self.dir[0]/len, self.dir[1]/len, self.dir[2]/len];
        GpuLight {
            pos_type: [self.pos[0], self.pos[1], self.pos[2], 1.0],
            color_intensity: [self.color[0], self.color[1], self.color[2], self.intensity],
            dir_radius: [norm_dir[0], norm_dir[1], norm_dir[2], self.range],
            spot_params: [self.inner_angle_rad.cos(), self.outer_angle_rad.cos(), 0.0, 0.0],
        }
    }
}

/// Global Directional Light (Sun / Moon)
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub dir: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

impl DirectionalLight {
    pub fn new(dir: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self { dir, color, intensity }
    }

    pub fn to_gpu(&self) -> GpuLight {
        let len = (self.dir[0]*self.dir[0] + self.dir[1]*self.dir[1] + self.dir[2]*self.dir[2]).sqrt().max(0.001);
        let norm_dir = [self.dir[0]/len, self.dir[1]/len, self.dir[2]/len];
        GpuLight {
            pos_type: [0.0, 0.0, 0.0, 2.0],
            color_intensity: [self.color[0], self.color[1], self.color[2], self.intensity],
            dir_radius: [norm_dir[0], norm_dir[1], norm_dir[2], 1000.0],
            spot_params: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Central Dynamic Light Manager
#[derive(Debug, Clone)]
pub struct LightManager {
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
    pub directional_lights: Vec<DirectionalLight>,
    pub ambient_color: [f32; 3],
    pub ambient_intensity: f32,
}

impl Default for LightManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LightManager {
    pub fn new() -> Self {
        Self {
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            directional_lights: Vec::new(),
            ambient_color: [0.05, 0.05, 0.08],
            ambient_intensity: 1.0,
        }
    }

    pub fn add_point_light(&mut self, light: PointLight) -> usize {
        let id = self.point_lights.len();
        self.point_lights.push(light);
        id
    }

    pub fn add_spot_light(&mut self, light: SpotLight) -> usize {
        let id = self.spot_lights.len();
        self.spot_lights.push(light);
        id
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) -> usize {
        let id = self.directional_lights.len();
        self.directional_lights.push(light);
        id
    }

    pub fn clear_dynamic_lights(&mut self) {
        self.point_lights.clear();
        self.spot_lights.clear();
    }

    /// Pack all active lights into an array of GPU-ready structures up to `max_lights`
    pub fn build_gpu_lights(&self, max_lights: usize) -> (Vec<GpuLight>, u32) {
        let mut gpu_lights = Vec::with_capacity(max_lights);

        for dir in &self.directional_lights {
            if gpu_lights.len() >= max_lights { break; }
            gpu_lights.push(dir.to_gpu());
        }

        for pt in &self.point_lights {
            if gpu_lights.len() >= max_lights { break; }
            gpu_lights.push(pt.to_gpu());
        }

        for spot in &self.spot_lights {
            if gpu_lights.len() >= max_lights { break; }
            gpu_lights.push(spot.to_gpu());
        }

        let active_count = gpu_lights.len() as u32;

        // Pad remaining entries with zeroes
        while gpu_lights.len() < max_lights {
            gpu_lights.push(GpuLight {
                pos_type: [0.0, 0.0, 0.0, -1.0],
                color_intensity: [0.0, 0.0, 0.0, 0.0],
                dir_radius: [0.0, 0.0, 0.0, 0.0],
                spot_params: [0.0, 0.0, 0.0, 0.0],
            });
        }

        (gpu_lights, active_count)
    }

    /// Cull point and spot lights that are outside the camera view sphere
    pub fn cull_lights(&self, camera_pos: [f32; 3], max_radius: f32) -> Vec<GpuLight> {
        let mut culled = Vec::new();

        for dir in &self.directional_lights {
            culled.push(dir.to_gpu());
        }

        for pt in &self.point_lights {
            let dx = pt.pos[0] - camera_pos[0];
            let dy = pt.pos[1] - camera_pos[1];
            let dz = pt.pos[2] - camera_pos[2];
            let dist_sq = dx*dx + dy*dy + dz*dz;
            let range = max_radius + pt.radius;
            if dist_sq <= range * range {
                culled.push(pt.to_gpu());
            }
        }

        for spot in &self.spot_lights {
            let dx = spot.pos[0] - camera_pos[0];
            let dy = spot.pos[1] - camera_pos[1];
            let dz = spot.pos[2] - camera_pos[2];
            let dist_sq = dx*dx + dy*dy + dz*dz;
            let range = max_radius + spot.range;
            if dist_sq <= range * range {
                culled.push(spot.to_gpu());
            }
        }

        culled
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_gpu_packing() {
        let mut mgr = LightManager::new();
        mgr.add_directional_light(DirectionalLight::new([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0));
        mgr.add_point_light(PointLight::new([0.0, 5.0, 0.0], [1.0, 0.0, 0.0], 2.0, 10.0));

        let (gpu_lights, count) = mgr.build_gpu_lights(16);
        assert_eq!(count, 2);
        assert_eq!(gpu_lights.len(), 16);
        assert_eq!(gpu_lights[0].pos_type[3], 2.0); // Directional
        assert_eq!(gpu_lights[1].pos_type[3], 0.0); // Point
    }

    #[test]
    fn test_light_culling() {
        let mut mgr = LightManager::new();
        // Near light
        mgr.add_point_light(PointLight::new([0.0, 0.0, 5.0], [1.0, 1.0, 1.0], 1.0, 2.0));
        // Far light
        mgr.add_point_light(PointLight::new([0.0, 0.0, 100.0], [1.0, 1.0, 1.0], 1.0, 2.0));

        let visible = mgr.cull_lights([0.0, 0.0, 0.0], 10.0);
        assert_eq!(visible.len(), 1);
    }
}
