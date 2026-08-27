//! Virtual Reality (VR) & Stereoscopic 3D Subsystem
//!
//! Provides stereoscopic dual-eye rendering for Meta Quest 3S / PCVR headsets:
//! - Dual-Eye View-Projection matrices with IPD adjustment (`StereoCameraRig`)
//! - Side-by-Side (SBS), Top-Bottom, and Anaglyph 3D modes
//! - 6-DoF Head Pose tracking & Quest Touch controller state (`VrTrackingContext`)

pub mod stereo;
pub mod tracking;

pub use stereo::{StereoCameraRig, StereoMode, Eye};
pub use tracking::{VrHeadPose, VrController, VrHand, VrTrackingContext};
