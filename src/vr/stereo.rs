//! Stereoscopic Dual-Eye VR Camera Rig and Interpupillary Distance (IPD) Math
//!
//! Provides stereoscopic camera matrices for Meta Quest 3S / PCVR headsets,
//! supporting Side-by-Side (SBS), Top-Bottom, and Anaglyph 3D rendering.

use cgmath::{Point3, Vector3, Matrix4, Deg, perspective, InnerSpace};

/// Stereoscopic display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoMode {
    /// Standard 2D monoscopic view
    Mono,
    /// Side-by-Side (SBS) Dual-Eye Stereo (Left on left half, Right on right half)
    SideBySide,
    /// Over/Under (Top/Bottom) Stereo
    TopBottom,
    /// Red/Cyan Anaglyph 3D (for standard monitors with 3D glasses)
    AnaglyphRedCyan,
}

/// Eye identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye {
    Left,
    Right,
}

/// Stereoscopic Dual-Eye Camera Rig
#[derive(Debug, Clone)]
pub struct StereoCameraRig {
    /// Center of the user's head in world coordinates
    pub head_pos: Point3<f32>,
    /// Head orientation in radians (Yaw, Pitch, Roll)
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Interpupillary distance in meters (e.g. 0.063 for 63mm Quest 3S average IPD)
    pub ipd_meters: f32,
    /// Vertical Field of View in degrees (e.g. 95.0 to 110.0 for Quest 3S)
    pub fov_y_deg: f32,
    /// Aspect ratio of a single eye view (default ~ 0.9 for SBS on 16:9 screens)
    pub aspect_ratio: f32,
    /// Near clip plane in meters
    pub near: f32,
    /// Far clip plane in meters
    pub far: f32,
    /// Active stereoscopic rendering mode
    pub mode: StereoMode,
}

impl Default for StereoCameraRig {
    fn default() -> Self {
        Self {
            head_pos: Point3::new(0.0, 1.7, 0.0), // 1.7m average standing eye height
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            ipd_meters: 0.063, // 63mm default IPD
            fov_y_deg: 95.0,
            aspect_ratio: 0.8888, // 16:9 screen split into two 8:9 eye viewports
            near: 0.05,
            far: 1000.0,
            mode: StereoMode::SideBySide,
        }
    }
}

impl StereoCameraRig {
    pub fn new(head_pos: Point3<f32>, ipd_meters: f32) -> Self {
        Self {
            head_pos,
            ipd_meters,
            ..Default::default()
        }
    }

    /// Forward direction vector of the headset
    pub fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    /// Right direction vector perpendicular to the head forward orientation
    pub fn right(&self) -> Vector3<f32> {
        let fwd = self.forward();
        Vector3::new(-fwd.z, 0.0, fwd.x).normalize()
    }

    /// Up direction vector of the headset
    pub fn up(&self) -> Vector3<f32> {
        self.right().cross(self.forward()).normalize()
    }

    /// Compute world position of the specified eye offset by half the IPD
    pub fn eye_position(&self, eye: Eye) -> Point3<f32> {
        let half_ipd = self.ipd_meters * 0.5;
        let right = self.right();
        match eye {
            Eye::Left => self.head_pos - right * half_ipd,
            Eye::Right => self.head_pos + right * half_ipd,
        }
    }

    /// Compute view matrix for the specified eye
    pub fn view_matrix(&self, eye: Eye) -> Matrix4<f32> {
        let eye_pos = self.eye_position(eye);
        let fwd = self.forward();
        let target = eye_pos + fwd;
        Matrix4::look_at_rh(eye_pos, target, self.up())
    }

    /// Compute projection matrix for one eye
    pub fn proj_matrix(&self) -> Matrix4<f32> {
        perspective(Deg(self.fov_y_deg), self.aspect_ratio, self.near, self.far)
    }

    /// Compute combined View-Projection matrix for specified eye
    pub fn view_proj(&self, eye: Eye) -> Matrix4<f32> {
        self.proj_matrix() * self.view_matrix(eye)
    }

    /// Returns Left and Right eye matrices formatted as `[[f32; 4]; 4]` for GPU uniforms
    pub fn dual_matrices(&self) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
        let left_vp: [[f32; 4]; 4] = self.view_proj(Eye::Left).into();
        let right_vp: [[f32; 4]; 4] = self.view_proj(Eye::Right).into();
        (left_vp, right_vp)
    }

    /// Adjust IPD in millimeters (e.g. `adjust_ipd_mm(1.0)` or `adjust_ipd_mm(-1.0)`)
    pub fn adjust_ipd_mm(&mut self, delta_mm: f32) {
        let new_mm = (self.ipd_meters * 1000.0 + delta_mm).clamp(52.0, 75.0);
        self.ipd_meters = new_mm / 1000.0;
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_eye_separation() {
        let rig = StereoCameraRig::new(Point3::new(0.0, 1.7, 0.0), 0.064); // 64mm IPD
        let left_pos = rig.eye_position(Eye::Left);
        let right_pos = rig.eye_position(Eye::Right);

        let distance = (right_pos - left_pos).magnitude();
        assert!((distance - 0.064).abs() < 1e-4);
    }

    #[test]
    fn test_dual_eye_parallax() {
        let rig = StereoCameraRig::default();
        let left_vp = rig.view_proj(Eye::Left);
        let right_vp = rig.view_proj(Eye::Right);

        // Left and Right matrices must produce distinct x-parallax
        assert_ne!(left_vp, right_vp);
    }

    #[test]
    fn test_ipd_adjustment_clamps() {
        let mut rig = StereoCameraRig::default();
        rig.adjust_ipd_mm(100.0); // Attempt to exceed max
        assert_eq!(rig.ipd_meters, 0.075); // Clamped to 75mm

        rig.adjust_ipd_mm(-100.0); // Attempt to go below min
        assert_eq!(rig.ipd_meters, 0.052); // Clamped to 52mm
    }
}
