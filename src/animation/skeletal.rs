//! Skeletal Animation Rigging, GPU Skinning & Two-Bone Inverse Kinematics (IK)
//!
//! Provides bone hierarchy forward kinematics, skinning matrices for vertex shaders,
//! and analytical Two-Bone Inverse Kinematics for procedural foot placement and arm reaching.

use cgmath::{Vector3, Matrix4, SquareMatrix, InnerSpace};

/// An animated joint/bone in a skeletal hierarchy
#[derive(Debug, Clone, PartialEq)]
pub struct Bone {
    pub id: usize,
    pub name: String,
    pub parent_id: Option<usize>,
    pub local_pos: [f32; 3],
    pub local_rot: [f32; 3], // Euler angles (Yaw, Pitch, Roll in radians)
    pub local_scale: [f32; 3],
    pub local_matrix: [[f32; 4]; 4],
    pub global_matrix: [[f32; 4]; 4],
    pub inverse_bind_matrix: [[f32; 4]; 4],
}

impl Bone {
    pub fn new(id: usize, name: impl Into<String>, parent_id: Option<usize>, local_pos: [f32; 3]) -> Self {
        let identity: [[f32; 4]; 4] = Matrix4::identity().into();
        Self {
            id,
            name: name.into(),
            parent_id,
            local_pos,
            local_rot: [0.0, 0.0, 0.0],
            local_scale: [1.0, 1.0, 1.0],
            local_matrix: identity,
            global_matrix: identity,
            inverse_bind_matrix: identity,
        }
    }

    /// Compute local 4x4 transform from pos, rot, scale
    pub fn compute_local_matrix(&mut self) {
        let cy = self.local_rot[0].cos();
        let sy = self.local_rot[0].sin();
        let cp = self.local_rot[1].cos();
        let sp = self.local_rot[1].sin();
        let cr = self.local_rot[2].cos();
        let sr = self.local_rot[2].sin();

        let sx = self.local_scale[0];
        let sy_scale = self.local_scale[1];
        let sz = self.local_scale[2];

        // Combined Rotation (Yaw-Pitch-Roll) * Scale
        let r00 = (cy * cr + sy * sp * sr) * sx;
        let r01 = (sr * cp) * sx;
        let r02 = (-sy * cr + cy * sp * sr) * sx;

        let r10 = (-cy * sr + sy * sp * cr) * sy_scale;
        let r11 = (cr * cp) * sy_scale;
        let r12 = (sr * sy + cy * sp * cr) * sy_scale;

        let r20 = (sy * cp) * sz;
        let r21 = (-sp) * sz;
        let r22 = (cy * cp) * sz;

        self.local_matrix = [
            [r00, r01, r02, 0.0],
            [r10, r11, r12, 0.0],
            [r20, r21, r22, 0.0],
            [self.local_pos[0], self.local_pos[1], self.local_pos[2], 1.0],
        ];
    }
}

/// Character Skeleton & Rig
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub name: String,
    pub bones: Vec<Bone>,
}

impl Skeleton {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bones: Vec::new(),
        }
    }

    pub fn add_bone(&mut self, name: impl Into<String>, parent_id: Option<usize>, local_pos: [f32; 3]) -> usize {
        let id = self.bones.len();
        let bone = Bone::new(id, name, parent_id, local_pos);
        self.bones.push(bone);
        id
    }

    /// Evaluates Forward Kinematics (FK) down the bone tree
    pub fn evaluate_forward_kinematics(&mut self) {
        for i in 0..self.bones.len() {
            self.bones[i].compute_local_matrix();
            let parent_id = self.bones[i].parent_id;

            if let Some(pid) = parent_id {
                let parent_global: Matrix4<f32> = self.bones[pid].global_matrix.into();
                let child_local: Matrix4<f32> = self.bones[i].local_matrix.into();
                let combined = parent_global * child_local;
                self.bones[i].global_matrix = combined.into();
            } else {
                self.bones[i].global_matrix = self.bones[i].local_matrix;
            }
        }
    }

    /// Store current global matrices as Inverse Bind Pose matrices
    pub fn capture_bind_pose(&mut self) {
        self.evaluate_forward_kinematics();
        for bone in &mut self.bones {
            let m: Matrix4<f32> = bone.global_matrix.into();
            if let Some(inv) = m.invert() {
                bone.inverse_bind_matrix = inv.into();
            }
        }
    }

    /// Compute final GPU skinning palette: $M_{\text{skin}} = M_{\text{global}} \cdot M_{\text{inv\_bind}}$
    pub fn compute_skinning_palette(&self) -> Vec<[[f32; 4]; 4]> {
        self.bones.iter().map(|b| {
            let g: Matrix4<f32> = b.global_matrix.into();
            let inv: Matrix4<f32> = b.inverse_bind_matrix.into();
            (g * inv).into()
        }).collect()
    }
}

/// Analytical Closed-Form Two-Bone Inverse Kinematics (IK) Solver
pub struct TwoBoneIk;

impl TwoBoneIk {
    /// Solves joint positions for a 2-bone chain (Root -> Middle/Knee -> End/Foot)
    /// Returns `Some((middle_pos, end_pos))` or clamps if target is out of reach
    pub fn solve(
        root_pos: [f32; 3],
        target_pos: [f32; 3],
        pole_target: [f32; 3],
        len_a: f32, // Upper bone length
        len_b: f32, // Lower bone length
    ) -> ([f32; 3], [f32; 3]) {
        let r = Vector3::new(root_pos[0], root_pos[1], root_pos[2]);
        let t = Vector3::new(target_pos[0], target_pos[1], target_pos[2]);
        let p = Vector3::new(pole_target[0], pole_target[1], pole_target[2]);

        let to_target = t - r;
        let dist = to_target.magnitude();

        // Max extension
        let max_len = (len_a + len_b) * 0.9999;
        let min_len = (len_a - len_b).abs() * 1.0001;
        let clamped_dist = dist.clamp(min_len, max_len);

        let dir = if dist > 1e-5 { to_target / dist } else { Vector3::new(0.0, -1.0, 0.0) };

        // Law of Cosines to find angle at root
        let cos_alpha = ((len_a * len_a + clamped_dist * clamped_dist - len_b * len_b) / (2.0 * len_a * clamped_dist)).clamp(-1.0, 1.0);
        let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

        // Bend direction towards pole target (Gram-Schmidt projection perpendicular to dir)
        let to_pole = p - r;
        let pole_proj = to_pole - dir * to_pole.dot(dir);
        let bend_dir = if pole_proj.magnitude2() > 1e-5 {
            pole_proj.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };

        // Middle joint position
        let middle = r + dir * (len_a * cos_alpha) + bend_dir * (len_a * sin_alpha);
        let end = r + dir * clamped_dist;

        (
            [middle.x, middle.y, middle.z],
            [end.x, end.y, end.z],
        )
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_fk_evaluation() {
        let mut skel = Skeleton::new("Humanoid");
        let hip = skel.add_bone("Hip", None, [0.0, 1.0, 0.0]);
        let leg = skel.add_bone("Leg", Some(hip), [0.0, -0.5, 0.0]);

        skel.evaluate_forward_kinematics();

        // Global pos of leg should be (0.0, 0.5, 0.0)
        let leg_global = skel.bones[leg].global_matrix;
        assert!((leg_global[3][1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_two_bone_ik_reaches_target() {
        let root = [0.0, 1.0, 0.0];
        let target = [0.0, 0.0, 0.0];
        let pole = [0.0, 0.5, 1.0]; // Bend forward (+Z)

        let (mid, end) = TwoBoneIk::solve(root, target, pole, 0.5, 0.5);

        // Middle joint bends forward in +Z
        assert!(mid[2] > 0.0);
        // End reaches target (0.0, 0.0, 0.0)
        assert!((end[0] - target[0]).abs() < 1e-3);
        assert!((end[1] - target[1]).abs() < 1e-3);
        assert!((end[2] - target[2]).abs() < 1e-3);
    }
}
