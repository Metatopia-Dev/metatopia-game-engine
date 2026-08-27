//! Atmosphere, Day/Night Cycle & Volumetric Scattering Subsystem
//!
//! Provides Rayleigh/Mie atmospheric lighting, dynamic solar motion, and height fog.

pub mod sky;

pub use sky::{TimeOfDay, VolumetricFog, AtmosphereController};
