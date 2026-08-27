//! Projective Decal and Surface Impact System
//!
//! Projects surface marks (bullet holes, blast scorches, blood splatters, glowing neon glyphs)
//! oriented along collision surface normals with z-fighting offset and lifetime fading.

use crate::quickstart::GameVertex;

/// Visual type of decal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecalType {
    BulletHole,
    ScorchMark,
    BloodSplatter,
    NeonRune,
    Custom,
}

/// A surface-projected impact decal
#[derive(Debug, Clone)]
pub struct Decal {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub emissive: f32,
    pub lifetime: f32,
    pub max_life: f32,
    pub decal_type: DecalType,
}

/// Central Decal Management System
#[derive(Debug, Clone)]
pub struct DecalSystem {
    pub decals: Vec<Decal>,
    pub max_decals: usize,
}

impl Default for DecalSystem {
    fn default() -> Self {
        Self::new(200)
    }
}

impl DecalSystem {
    pub fn new(max_decals: usize) -> Self {
        Self {
            decals: Vec::with_capacity(max_decals),
            max_decals,
        }
    }

    /// Spawn a surface-aligned decal at an impact point
    pub fn spawn_decal(
        &mut self,
        pos: [f32; 3],
        normal: [f32; 3],
        size: f32,
        color: [f32; 4],
        emissive: f32,
        lifetime: f32,
        decal_type: DecalType,
    ) {
        if self.decals.len() >= self.max_decals {
            self.decals.remove(0); // Recycle oldest
        }

        self.decals.push(Decal {
            pos,
            normal,
            size,
            color,
            emissive,
            lifetime: 0.0,
            max_life: lifetime.max(0.5),
            decal_type,
        });
    }

    /// Step simulation and decay lifetimes
    pub fn update(&mut self, dt: f32) {
        for d in &mut self.decals {
            d.lifetime += dt;
        }
        self.decals.retain(|d| d.lifetime < d.max_life);
    }

    /// Clear all active decals
    pub fn clear(&mut self) {
        self.decals.clear();
    }

    /// Build vertex mesh with tangent/bitangent alignment and z-offset to prevent z-fighting
    pub fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.decals.len() * 4);
        let mut idxs = Vec::with_capacity(self.decals.len() * 6);

        for d in &self.decals {
            let t = (d.lifetime / d.max_life).clamp(0.0, 1.0);
            let alpha = (d.color[3] * (1.0 - t * t)).max(0.0);
            if alpha <= 0.001 { continue; }

            let n = d.normal;
            let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt().max(0.001);
            let norm = [n[0]/len, n[1]/len, n[2]/len];

            // Build orthogonal tangent & bitangent
            let up = if norm[1].abs() < 0.99 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };

            // tangent = up x norm
            let mut tangent = [
                up[1] * norm[2] - up[2] * norm[1],
                up[2] * norm[0] - up[0] * norm[2],
                up[0] * norm[1] - up[1] * norm[0],
            ];
            let t_len = (tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]).sqrt().max(0.001);
            tangent = [tangent[0]/t_len * d.size * 0.5, tangent[1]/t_len * d.size * 0.5, tangent[2]/t_len * d.size * 0.5];

            // bitangent = norm x tangent
            let bitangent = [
                (norm[1] * tangent[2] - norm[2] * tangent[1]),
                (norm[2] * tangent[0] - norm[0] * tangent[2]),
                (norm[0] * tangent[1] - norm[1] * tangent[0]),
            ];

            // Slight offset along normal to avoid z-fighting
            let offset_dist = 0.005;
            let center = [
                d.pos[0] + norm[0] * offset_dist,
                d.pos[1] + norm[1] * offset_dist,
                d.pos[2] + norm[2] * offset_dist,
            ];

            let p0 = [center[0] - tangent[0] - bitangent[0], center[1] - tangent[1] - bitangent[1], center[2] - tangent[2] - bitangent[2]];
            let p1 = [center[0] + tangent[0] - bitangent[0], center[1] + tangent[1] - bitangent[1], center[2] + tangent[2] - bitangent[2]];
            let p2 = [center[0] + tangent[0] + bitangent[0], center[1] + tangent[1] + bitangent[1], center[2] + tangent[2] + bitangent[2]];
            let p3 = [center[0] - tangent[0] + bitangent[0], center[1] - tangent[1] + bitangent[1], center[2] - tangent[2] + bitangent[2]];

            let rgb = [d.color[0], d.color[1], d.color[2]];
            let pbr = [0.0, 0.5, d.emissive * alpha, 0.0];

            let base = verts.len() as u32;
            verts.push(GameVertex::new(p0, norm, rgb, pbr));
            verts.push(GameVertex::new(p1, norm, rgb, pbr));
            verts.push(GameVertex::new(p2, norm, rgb, pbr));
            verts.push(GameVertex::new(p3, norm, rgb, pbr));

            idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        (verts, idxs)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decal_spawning_and_mesh() {
        let mut ds = DecalSystem::new(10);
        ds.spawn_decal(
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], // Floor normal
            0.5,
            [0.1, 0.1, 0.1, 1.0],
            0.0,
            5.0,
            DecalType::ScorchMark,
        );

        assert_eq!(ds.decals.len(), 1);
        let (verts, indices) = ds.build_mesh();
        assert_eq!(verts.len(), 4);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_decal_recycling() {
        let mut ds = DecalSystem::new(2);
        ds.spawn_decal([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, [1.0, 0.0, 0.0, 1.0], 0.0, 2.0, DecalType::BulletHole);
        ds.spawn_decal([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, [1.0, 0.0, 0.0, 1.0], 0.0, 2.0, DecalType::BulletHole);
        ds.spawn_decal([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.2, [1.0, 0.0, 0.0, 1.0], 0.0, 2.0, DecalType::BulletHole);

        assert_eq!(ds.decals.len(), 2);
        assert_eq!(ds.decals[0].pos[0], 1.0); // Oldest (0.0) was recycled
    }
}
