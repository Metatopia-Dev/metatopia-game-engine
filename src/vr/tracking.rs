//! 6-DoF VR Head Tracking and Meta Quest Touch Controller Input Mapping
//!
//! Provides head orientation poses and dual-controller tracking for VR interaction.

/// 6-Degrees-of-Freedom VR Headset Pose
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VrHeadPose {
    pub pos: [f32; 3],
    pub linear_vel: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

impl Default for VrHeadPose {
    fn default() -> Self {
        Self {
            pos: [0.0, 1.7, 0.0],
            linear_vel: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }
}

/// Hand controller identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrHand {
    Left,
    Right,
}

/// Quest Touch Plus / VR Hand Controller State
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VrController {
    pub hand: VrHand,
    pub pos: [f32; 3],
    pub rot: [f32; 3], // Yaw, Pitch, Roll
    pub trigger: f32,  // 0.0 to 1.0 index finger trigger
    pub grip: f32,     // 0.0 to 1.0 squeeze grip
    pub thumbstick: (f32, f32), // -1.0 to 1.0 X and Y
    pub btn_primary: bool,   // A on Right, X on Left
    pub btn_secondary: bool, // B on Right, Y on Left
    pub thumbstick_click: bool,
}

impl VrController {
    pub fn new(hand: VrHand) -> Self {
        let default_x = match hand {
            VrHand::Left => -0.25,
            VrHand::Right => 0.25,
        };
        Self {
            hand,
            pos: [default_x, 1.2, -0.4],
            rot: [0.0, 0.0, 0.0],
            trigger: 0.0,
            grip: 0.0,
            thumbstick: (0.0, 0.0),
            btn_primary: false,
            btn_secondary: false,
            thumbstick_click: false,
        }
    }
}

/// Active VR Tracking State
#[derive(Debug, Clone)]
pub struct VrTrackingContext {
    pub head: VrHeadPose,
    pub left_controller: VrController,
    pub right_controller: VrController,
    pub recenter_offset: [f32; 3],
}

impl Default for VrTrackingContext {
    fn default() -> Self {
        Self {
            head: VrHeadPose::default(),
            left_controller: VrController::new(VrHand::Left),
            right_controller: VrController::new(VrHand::Right),
            recenter_offset: [0.0, 0.0, 0.0],
        }
    }
}

impl VrTrackingContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recenter the VR origin to current head position
    pub fn recenter(&mut self) {
        self.recenter_offset = self.head.pos;
    }

    /// Get recentered head position
    pub fn calibrated_head_pos(&self) -> [f32; 3] {
        [
            self.head.pos[0] - self.recenter_offset[0],
            self.head.pos[1],
            self.head.pos[2] - self.recenter_offset[2],
        ]
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vr_controllers_default_positions() {
        let left = VrController::new(VrHand::Left);
        let right = VrController::new(VrHand::Right);

        assert!(left.pos[0] < 0.0);
        assert!(right.pos[0] > 0.0);
        assert_eq!(left.trigger, 0.0);
    }

    #[test]
    fn test_vr_recenter() {
        let mut ctx = VrTrackingContext::new();
        ctx.head.pos = [5.0, 1.7, -3.0];
        ctx.recenter();

        let cal = ctx.calibrated_head_pos();
        assert_eq!(cal[0], 0.0);
        assert_eq!(cal[2], 0.0);
    }
}
