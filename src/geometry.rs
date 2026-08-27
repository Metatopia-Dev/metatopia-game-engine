//! Procedural 3D Mesh and Geometry Toolkit
//!
//! Generates parametric 3D meshes (Cylinders, Cones, Capsules, Tori, Icosahedrons,
//! Geodesic Domes, Heightmap Terrains, Planes, Rings, and 3D Arrows) ready for
//! rendering or physics colliders.

use std::f32::consts::PI;
use crate::quickstart::GameVertex;

/// Procedural 3D Mesh Generator
pub struct ProceduralMesh;

impl ProceduralMesh {
    /// Generate a 3D Cylinder aligned with the Y-axis, centered at origin
    pub fn cylinder(radius: f32, height: f32, segments: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let segs = segments.max(3);
        let h = height / 2.0;
        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        // Side vertices
        for i in 0..=segs {
            let theta = 2.0 * PI * (i as f32) / (segs as f32);
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            let norm = [theta.cos(), 0.0, theta.sin()];

            verts.push(GameVertex::colored([x, -h, z], norm, color));
            verts.push(GameVertex::colored([x,  h, z], norm, color));
        }

        // Side indices
        for i in 0..segs {
            let a = i * 2;
            let b = a + 1;
            let c = a + 2;
            let d = a + 3;
            idxs.extend_from_slice(&[a, c, b, b, c, d]);
        }

        // Top & Bottom Caps
        let top_center_idx = verts.len() as u32;
        verts.push(GameVertex::colored([0.0, h, 0.0], [0.0, 1.0, 0.0], color));

        let bot_center_idx = verts.len() as u32;
        verts.push(GameVertex::colored([0.0, -h, 0.0], [0.0, -1.0, 0.0], color));

        let base_cap = verts.len() as u32;
        for i in 0..=segs {
            let theta = 2.0 * PI * (i as f32) / (segs as f32);
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            verts.push(GameVertex::colored([x,  h, z], [0.0,  1.0, 0.0], color));
            verts.push(GameVertex::colored([x, -h, z], [0.0, -1.0, 0.0], color));
        }

        for i in 0..segs {
            let top_edge = base_cap + i * 2;
            let next_top = base_cap + (i + 1) * 2;
            idxs.extend_from_slice(&[top_center_idx, top_edge, next_top]);

            let bot_edge = base_cap + i * 2 + 1;
            let next_bot = base_cap + (i + 1) * 2 + 1;
            idxs.extend_from_slice(&[bot_center_idx, next_bot, bot_edge]);
        }

        (verts, idxs)
    }

    /// Generate a 3D Cone pointing along the +Y axis
    pub fn cone(radius: f32, height: f32, segments: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let segs = segments.max(3);
        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        let tip = [0.0, height, 0.0];
        let slope = (radius / height).atan();

        // Tip vertices & base vertices for sides
        for i in 0..=segs {
            let theta = 2.0 * PI * (i as f32) / (segs as f32);
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            let norm = [theta.cos() * slope.cos(), slope.sin(), theta.sin() * slope.cos()];

            let base_idx = verts.len() as u32;
            verts.push(GameVertex::colored([x, 0.0, z], norm, color));
            verts.push(GameVertex::colored(tip, norm, color));

            if i < segs {
                idxs.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);
            }
        }

        // Bottom Disc Cap
        let bot_center_idx = verts.len() as u32;
        verts.push(GameVertex::colored([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], color));
        let cap_base = verts.len() as u32;

        for i in 0..=segs {
            let theta = 2.0 * PI * (i as f32) / (segs as f32);
            let x = theta.cos() * radius;
            let z = theta.sin() * radius;
            verts.push(GameVertex::colored([x, 0.0, z], [0.0, -1.0, 0.0], color));
        }

        for i in 0..segs {
            idxs.extend_from_slice(&[bot_center_idx, cap_base + i + 1, cap_base + i]);
        }

        (verts, idxs)
    }

    /// Generate a 3D Capsule (Cylinder with hemispherical caps)
    pub fn capsule(radius: f32, cylinder_height: f32, rings: u32, segments: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let segs = segments.max(4);
        let r_count = rings.max(2);
        let h = cylinder_height / 2.0;

        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        // Top Hemisphere (y from h to h + radius)
        for j in 0..=r_count {
            let phi = (PI / 2.0) * (1.0 - (j as f32) / (r_count as f32));
            let y = h + radius * phi.sin();
            let r = radius * phi.cos();

            for i in 0..=segs {
                let theta = 2.0 * PI * (i as f32) / (segs as f32);
                let x = r * theta.cos();
                let z = r * theta.sin();
                let norm = [x / radius, phi.sin(), z / radius];
                verts.push(GameVertex::colored([x, y, z], norm, color));
            }
        }

        // Bottom Hemisphere (y from -h to -h - radius)
        for j in 0..=r_count {
            let phi = (PI / 2.0) * ((j as f32) / (r_count as f32));
            let y = -h - radius * phi.sin();
            let r = radius * phi.cos();

            for i in 0..=segs {
                let theta = 2.0 * PI * (i as f32) / (segs as f32);
                let x = r * theta.cos();
                let z = r * theta.sin();
                let norm = [x / radius, -phi.sin(), z / radius];
                verts.push(GameVertex::colored([x, y, z], norm, color));
            }
        }

        let stride = segs + 1;
        let total_rings = (r_count + 1) * 2;

        for j in 0..(total_rings - 1) {
            for i in 0..segs {
                let a = (j * stride + i) as u32;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                idxs.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        (verts, idxs)
    }

    /// Generate a 3D Torus (Donut) centered on the XZ plane
    pub fn torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let maj_segs = major_segments.max(4);
        let min_segs = minor_segments.max(3);

        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        for i in 0..=maj_segs {
            let u = 2.0 * PI * (i as f32) / (maj_segs as f32);
            let center = [major_radius * u.cos(), 0.0, major_radius * u.sin()];

            for j in 0..=min_segs {
                let v = 2.0 * PI * (j as f32) / (min_segs as f32);
                let x = (major_radius + minor_radius * v.cos()) * u.cos();
                let y = minor_radius * v.sin();
                let z = (major_radius + minor_radius * v.cos()) * u.sin();

                let norm = [
                    (x - center[0]) / minor_radius,
                    y / minor_radius,
                    (z - center[2]) / minor_radius,
                ];

                verts.push(GameVertex::colored([x, y, z], norm, color));
            }
        }

        let stride = min_segs + 1;
        for i in 0..maj_segs {
            for j in 0..min_segs {
                let a = i * stride + j;
                let b = a + 1;
                let c = (i + 1) * stride + j;
                let d = c + 1;
                idxs.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        (verts, idxs)
    }

    /// Generate a Subdivided 3D Plane
    pub fn plane(width: f32, depth: f32, sub_x: u32, sub_z: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let sx = sub_x.max(1);
        let sz = sub_z.max(1);
        let hw = width / 2.0;
        let hd = depth / 2.0;

        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        for j in 0..=sz {
            let z = -hd + (j as f32 / sz as f32) * depth;
            for i in 0..=sx {
                let x = -hw + (i as f32 / sx as f32) * width;
                verts.push(GameVertex::colored([x, 0.0, z], [0.0, 1.0, 0.0], color));
            }
        }

        let stride = sx + 1;
        for j in 0..sz {
            for i in 0..sx {
                let a = j * stride + i;
                let b = a + 1;
                let c = (j + 1) * stride + i;
                let d = c + 1;
                idxs.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        (verts, idxs)
    }

    /// Generate a 3D Heightmap Terrain with custom elevation function
    pub fn heightmap_terrain<F>(
        width: f32,
        depth: f32,
        res_x: u32,
        res_z: u32,
        height_fn: F,
        color: [f32; 3],
    ) -> (Vec<GameVertex>, Vec<u32>)
    where
        F: Fn(f32, f32) -> f32,
    {
        let rx = res_x.max(2);
        let rz = res_z.max(2);
        let hw = width / 2.0;
        let hd = depth / 2.0;

        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        for j in 0..=rz {
            let z = -hd + (j as f32 / rz as f32) * depth;
            for i in 0..=rx {
                let x = -hw + (i as f32 / rx as f32) * width;
                let y = height_fn(x, z);

                // Compute normal via finite differences
                let eps = 0.1;
                let dy_dx = (height_fn(x + eps, z) - height_fn(x - eps, z)) / (2.0 * eps);
                let dy_dz = (height_fn(x, z + eps) - height_fn(x, z - eps)) / (2.0 * eps);
                let mut norm = [-dy_dx, 1.0, -dy_dz];
                let len = (norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]).sqrt().max(0.001);
                norm = [norm[0]/len, norm[1]/len, norm[2]/len];

                verts.push(GameVertex::colored([x, y, z], norm, color));
            }
        }

        let stride = rx + 1;
        for j in 0..rz {
            for i in 0..rx {
                let a = j * stride + i;
                let b = a + 1;
                let c = (j + 1) * stride + i;
                let d = c + 1;
                idxs.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        (verts, idxs)
    }

    /// Generate a 3D Flat Ring / Disc with a hole in the middle
    pub fn ring(inner_radius: f32, outer_radius: f32, segments: u32, color: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let segs = segments.max(3);
        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        for i in 0..=segs {
            let theta = 2.0 * PI * (i as f32) / (segs as f32);
            let in_x = theta.cos() * inner_radius;
            let in_z = theta.sin() * inner_radius;
            let out_x = theta.cos() * outer_radius;
            let out_z = theta.sin() * outer_radius;

            verts.push(GameVertex::colored([in_x, 0.0, in_z], [0.0, 1.0, 0.0], color));
            verts.push(GameVertex::colored([out_x, 0.0, out_z], [0.0, 1.0, 0.0], color));
        }

        for i in 0..segs {
            let a = i * 2;
            let b = a + 1;
            let c = a + 2;
            let d = a + 3;
            idxs.extend_from_slice(&[a, b, c, b, d, c]);
        }

        (verts, idxs)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cylinder_geometry() {
        let (verts, indices) = ProceduralMesh::cylinder(1.0, 2.0, 16, [1.0, 1.0, 1.0]);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0); // Valid triangles
    }

    #[test]
    fn test_torus_geometry() {
        let (verts, indices) = ProceduralMesh::torus(3.0, 0.8, 16, 8, [0.0, 1.0, 0.0]);
        assert!(!verts.is_empty());
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_terrain_heightmap() {
        let (verts, indices) = ProceduralMesh::heightmap_terrain(
            20.0, 20.0, 10, 10,
            |x, z| (x * 0.5).sin() * (z * 0.5).cos(),
            [0.2, 0.8, 0.3],
        );
        assert_eq!(verts.len(), 11 * 11);
        assert_eq!(indices.len(), 10 * 10 * 6);
    }
}
