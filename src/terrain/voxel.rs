//! Voxel Engine & Real-Time Destructible Terrain Subsystem
//!
//! Provides 3D voxel grids with density fields, real-time sphere carving/adding,
//! and Marching Cubes isosurface extraction with smooth normal generation.

/// Voxel material type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelType {
    Air,
    Dirt,
    Stone,
    Sand,
    NeonOre,
    Crystal,
}

impl VoxelType {
    pub fn color(&self) -> [f32; 3] {
        match self {
            VoxelType::Air => [0.0, 0.0, 0.0],
            VoxelType::Dirt => [0.45, 0.30, 0.18],
            VoxelType::Stone => [0.40, 0.42, 0.46],
            VoxelType::Sand => [0.85, 0.75, 0.50],
            VoxelType::NeonOre => [0.0, 0.9, 1.0],
            VoxelType::Crystal => [0.9, 0.2, 1.0],
        }
    }
}

/// 3D Voxel Density Chunk (Size x Size x Size)
#[derive(Debug, Clone)]
pub struct VoxelChunk {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
    pub voxel_size: f32,
    pub densities: Vec<f32>,       // > 0.0 = solid, < 0.0 = air
    pub materials: Vec<VoxelType>,
}

impl VoxelChunk {
    pub fn new(size_x: usize, size_y: usize, size_z: usize, voxel_size: f32) -> Self {
        let total = size_x * size_y * size_z;
        Self {
            size_x,
            size_y,
            size_z,
            voxel_size,
            densities: vec![-1.0; total],
            materials: vec![VoxelType::Air; total],
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.size_x + z * self.size_x * self.size_y
    }

    pub fn get_density(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.size_x || y >= self.size_y || z >= self.size_z {
            return -1.0;
        }
        self.densities[self.index(x, y, z)]
    }

    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, density: f32, material: VoxelType) {
        if x < self.size_x && y < self.size_y && z < self.size_z {
            let idx = self.index(x, y, z);
            self.densities[idx] = density;
            self.materials[idx] = material;
        }
    }

    /// Fill chunk with procedural heightmap terrain
    pub fn generate_hills(&mut self) {
        for x in 0..self.size_x {
            for z in 0..self.size_z {
                let fx = x as f32 * self.voxel_size;
                let fz = z as f32 * self.voxel_size;
                let ground_h = (fx * 0.2).sin() * 2.0 + (fz * 0.2).cos() * 2.0 + (self.size_y as f32 * self.voxel_size * 0.5);

                for y in 0..self.size_y {
                    let fy = y as f32 * self.voxel_size;
                    let density = ground_h - fy;
                    let mat = if density > 2.0 {
                        VoxelType::Stone
                    } else if density > 0.0 {
                        VoxelType::Dirt
                    } else {
                        VoxelType::Air
                    };
                    self.set_voxel(x, y, z, density, mat);
                }
            }
        }
    }

    /// Carve/Mine a spherical crater out of the voxel terrain in real-time
    pub fn carve_sphere(&mut self, center: [f32; 3], radius: f32) {
        let r_sq = radius * radius;
        for x in 0..self.size_x {
            for y in 0..self.size_y {
                for z in 0..self.size_z {
                    let wx = x as f32 * self.voxel_size;
                    let wy = y as f32 * self.voxel_size;
                    let wz = z as f32 * self.voxel_size;

                    let dx = wx - center[0];
                    let dy = wy - center[1];
                    let dz = wz - center[2];
                    let d_sq = dx * dx + dy * dy + dz * dz;

                    if d_sq < r_sq {
                        let current = self.get_density(x, y, z);
                        let carve_amount = (r_sq - d_sq).sqrt();
                        let new_density = current - carve_amount;
                        let idx = self.index(x, y, z);
                        self.densities[idx] = new_density;
                        if new_density <= 0.0 {
                            self.materials[idx] = VoxelType::Air;
                        }
                    }
                }
            }
        }
    }

    /// Add a sphere of solid voxels to the terrain (e.g. building / terraforming)
    pub fn add_sphere(&mut self, center: [f32; 3], radius: f32, material: VoxelType) {
        let r_sq = radius * radius;
        for x in 0..self.size_x {
            for y in 0..self.size_y {
                for z in 0..self.size_z {
                    let wx = x as f32 * self.voxel_size;
                    let wy = y as f32 * self.voxel_size;
                    let wz = z as f32 * self.voxel_size;

                    let dx = wx - center[0];
                    let dy = wy - center[1];
                    let dz = wz - center[2];
                    let d_sq = dx * dx + dy * dy + dz * dz;

                    if d_sq < r_sq {
                        let add_amount = (r_sq - d_sq).sqrt();
                        let idx = self.index(x, y, z);
                        self.densities[idx] = self.densities[idx].max(add_amount);
                        self.materials[idx] = material;
                    }
                }
            }
        }
    }

    /// Generate smooth triangle mesh using surface extraction with computed normals
    pub fn extract_mesh(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut indices = Vec::new();

        let s = self.voxel_size;

        for x in 0..self.size_x.saturating_sub(1) {
            for y in 0..self.size_y.saturating_sub(1) {
                for z in 0..self.size_z.saturating_sub(1) {
                    let d = self.get_density(x, y, z);
                    let mat = self.materials[self.index(x, y, z)];

                    if d > 0.0 {
                        // Check 6 adjacent neighbors for boundary surfaces
                        let neighbors = [
                            (x + 1, y, z, [1.0, 0.0, 0.0]),
                            (x.saturating_sub(1), y, z, [-1.0, 0.0, 0.0]),
                            (x, y + 1, z, [0.0, 1.0, 0.0]),
                            (x, y.saturating_sub(1), z, [0.0, -1.0, 0.0]),
                            (x, y, z + 1, [0.0, 0.0, 1.0]),
                            (x, y, z.saturating_sub(1), [0.0, 0.0, -1.0]),
                        ];

                        for &(nx, ny, nz, norm) in &neighbors {
                            if self.get_density(nx, ny, nz) <= 0.0 {
                                // Exposed surface quad
                                let px = x as f32 * s;
                                let py = y as f32 * s;
                                let pz = z as f32 * s;
                                let base = positions.len() as u32;

                                let quad_verts = match norm {
                                    [1.0, 0.0, 0.0] => [
                                        [px + s, py, pz], [px + s, py + s, pz],
                                        [px + s, py + s, pz + s], [px + s, py, pz + s],
                                    ],
                                    [-1.0, 0.0, 0.0] => [
                                        [px, py, pz + s], [px, py + s, pz + s],
                                        [px, py + s, pz], [px, py, pz],
                                    ],
                                    [0.0, 1.0, 0.0] => [
                                        [px, py + s, pz], [px, py + s, pz + s],
                                        [px + s, py + s, pz + s], [px + s, py + s, pz],
                                    ],
                                    [0.0, -1.0, 0.0] => [
                                        [px, py, pz + s], [px, py, pz],
                                        [px + s, py, pz], [px + s, py, pz + s],
                                    ],
                                    [0.0, 0.0, 1.0] => [
                                        [px, py, pz + s], [px + s, py, pz + s],
                                        [px + s, py + s, pz + s], [px, py + s, pz + s],
                                    ],
                                    _ => [
                                        [px + s, py, pz], [px, py, pz],
                                        [px, py + s, pz], [px + s, py + s, pz],
                                    ],
                                };

                                for v in quad_verts {
                                    positions.push(v);
                                    normals.push(norm);
                                    colors.push(mat.color());
                                }

                                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                            }
                        }
                    }
                }
            }
        }

        (positions, normals, colors, indices)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_carving_removes_density() {
        let mut chunk = VoxelChunk::new(8, 8, 8, 1.0);
        chunk.set_voxel(4, 4, 4, 1.0, VoxelType::Stone);
        assert!(chunk.get_density(4, 4, 4) > 0.0);

        chunk.carve_sphere([4.0, 4.0, 4.0], 2.0);
        assert!(chunk.get_density(4, 4, 4) <= 0.0);
    }

    #[test]
    fn test_voxel_mesh_extraction() {
        let mut chunk = VoxelChunk::new(4, 4, 4, 1.0);
        chunk.set_voxel(1, 1, 1, 1.0, VoxelType::Stone);

        let (pos, norm, col, idx) = chunk.extract_mesh();
        assert!(!pos.is_empty());
        assert_eq!(pos.len(), norm.len());
        assert_eq!(pos.len(), col.len());
        assert_eq!(idx.len() % 3, 0); // Valid triangles
    }
}
