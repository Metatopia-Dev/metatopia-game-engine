//! 3D Model & Mesh Asset Loader
//!
//! Parses 3D meshes (Wavefront OBJ, custom binary formats, procedural buffers)
//! with vertex positions, normals, UVs, vertex colors, and index buffers.

/// Parsed 3D mesh data ready for GPU upload
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl LoadedMesh {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Automatically compute smooth face normals if normals are missing
    pub fn compute_normals(&mut self) {
        if self.positions.is_empty() || self.indices.is_empty() { return; }

        self.normals = vec![[0.0, 0.0, 0.0]; self.positions.len()];

        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            if i0 >= self.positions.len() || i1 >= self.positions.len() || i2 >= self.positions.len() {
                continue;
            }

            let p0 = self.positions[i0];
            let p1 = self.positions[i1];
            let p2 = self.positions[i2];

            let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];

            for &idx in &[i0, i1, i2] {
                self.normals[idx][0] += normal[0];
                self.normals[idx][1] += normal[1];
                self.normals[idx][2] += normal[2];
            }
        }

        // Normalize
        for n in &mut self.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-6 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                *n = [0.0, 1.0, 0.0];
            }
        }
    }
}

/// 3D Model Loader
pub struct ModelLoader;

impl ModelLoader {
    /// Parse a 3D Wavefront `.obj` model file
    pub fn parse_obj(content: &str, mesh_name: &str) -> Result<LoadedMesh, String> {
        let mut mesh = LoadedMesh::new(mesh_name);
        let mut temp_positions = Vec::new();
        let mut temp_normals = Vec::new();
        let mut temp_uvs = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.split_whitespace();
            let prefix = parts.next().unwrap_or("");

            match prefix {
                "v" => {
                    let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    temp_positions.push([x, y, z]);
                }
                "vn" => {
                    let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    temp_normals.push([x, y, z]);
                }
                "vt" => {
                    let u: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let v: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    temp_uvs.push([u, v]);
                }
                "f" => {
                    let face_verts: Vec<&str> = parts.collect();
                    if face_verts.len() < 3 { continue; }

                    // Triangulate polygon fan
                    for i in 1..face_verts.len() - 1 {
                        for &v_spec in &[face_verts[0], face_verts[i], face_verts[i + 1]] {
                            let mut v_indices = v_spec.split('/');
                            let p_idx: usize = v_indices.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).saturating_sub(1);
                            let uv_idx: Option<usize> = v_indices.next().and_then(|s| if s.is_empty() { None } else { s.parse::<usize>().ok() }).map(|i: usize| i.saturating_sub(1));
                            let n_idx: Option<usize> = v_indices.next().and_then(|s| if s.is_empty() { None } else { s.parse::<usize>().ok() }).map(|i: usize| i.saturating_sub(1));

                            if p_idx < temp_positions.len() {
                                let new_idx = mesh.positions.len() as u32;
                                mesh.positions.push(temp_positions[p_idx]);

                                if let Some(ni) = n_idx {
                                    if ni < temp_normals.len() {
                                        mesh.normals.push(temp_normals[ni]);
                                    } else {
                                        mesh.normals.push([0.0, 1.0, 0.0]);
                                    }
                                } else {
                                    mesh.normals.push([0.0, 1.0, 0.0]);
                                }

                                if let Some(ui) = uv_idx {
                                    if ui < temp_uvs.len() {
                                        mesh.uvs.push(temp_uvs[ui]);
                                    } else {
                                        mesh.uvs.push([0.0, 0.0]);
                                    }
                                } else {
                                    mesh.uvs.push([0.0, 0.0]);
                                }

                                mesh.colors.push([1.0, 1.0, 1.0]);
                                mesh.indices.push(new_idx);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if mesh.normals.is_empty() || mesh.normals.iter().all(|n| *n == [0.0, 1.0, 0.0]) {
            mesh.compute_normals();
        }

        Ok(mesh)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obj_parser_cube() {
        let obj_data = r#"
            # Wavefront OBJ Quad Cube
            v -1.0 -1.0 1.0
            v 1.0 -1.0 1.0
            v 1.0 1.0 1.0
            v -1.0 1.0 1.0
            f 1 2 3 4
        "#;

        let mesh = ModelLoader::parse_obj(obj_data, "TestQuad").expect("Parse failed");
        assert_eq!(mesh.positions.len(), 6); // 1 quad = 2 triangles = 6 vertices
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_compute_normals() {
        let mut mesh = LoadedMesh::new("Triangle");
        mesh.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        mesh.indices = vec![0, 1, 2];
        mesh.compute_normals();

        assert_eq!(mesh.normals.len(), 3);
        assert!((mesh.normals[0][2] - 1.0).abs() < 1e-4); // Points +Z
    }
}
