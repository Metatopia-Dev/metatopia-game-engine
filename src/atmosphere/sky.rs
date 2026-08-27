//! Dynamic Atmosphere, Day/Night Cycle & Volumetric Scattering Model
//!
//! Simulates Rayleigh/Mie atmospheric scattering, dynamic solar/lunar motion,
//! color temperature shifts, and volumetric height-based fog.

use std::f32::consts::PI;

/// Diurnal Time of Day Progression
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeOfDay {
    /// Current time in 24-hour format (0.0 to 24.0)
    pub time_hours: f32,
    /// Simulation speed multiplier (e.g. 60.0 = 1 hour per minute)
    pub time_scale: f32,
    /// Latitude angle in radians (controls solar trajectory incline)
    pub latitude_rad: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self {
            time_hours: 12.0, // Start at solar noon
            time_scale: 1.0,
            latitude_rad: 45.0 * (PI / 180.0),
        }
    }
}

impl TimeOfDay {
    pub fn new(initial_hours: f32, time_scale: f32) -> Self {
        Self {
            time_hours: initial_hours.rem_euclid(24.0),
            time_scale,
            latitude_rad: 45.0 * (PI / 180.0),
        }
    }

    /// Advance time by `dt` seconds
    pub fn update(&mut self, dt_seconds: f32) {
        let delta_hours = (dt_seconds * self.time_scale) / 3600.0;
        self.time_hours = (self.time_hours + delta_hours).rem_euclid(24.0);
    }

    pub fn is_day(&self) -> bool {
        self.time_hours >= 6.0 && self.time_hours <= 18.0
    }

    /// Solar elevation angle in radians (positive = above horizon, negative = night)
    pub fn solar_elevation(&self) -> f32 {
        let solar_time = (self.time_hours - 12.0) * (PI / 12.0); // -PI at midnight, 0 at noon, PI at midnight
        solar_time.cos() * self.latitude_rad.cos()
    }

    /// Unit direction pointing towards the Sun
    pub fn sun_direction(&self) -> [f32; 3] {
        let solar_time = (self.time_hours - 12.0) * (PI / 12.0);
        let x = solar_time.sin() * self.latitude_rad.cos();
        let y = self.solar_elevation();
        let z = -solar_time.cos() * self.latitude_rad.sin();

        let len = (x * x + y * y + z * z).sqrt().max(1e-5);
        [x / len, y / len, z / len]
    }

    /// Sun light color based on atmospheric thickness & Rayleigh scattering
    pub fn sun_color(&self) -> [f32; 3] {
        let elev = self.solar_elevation();
        if elev <= -0.1 {
            // Night Moon Light (Subtle cold blue)
            [0.08, 0.12, 0.25]
        } else if elev < 0.2 {
            // Sunrise / Sunset Golden Hour (Warm orange / deep red)
            let t = ((elev + 0.1) / 0.3).clamp(0.0, 1.0);
            let sunset = [1.0, 0.45, 0.15];
            let morning = [1.0, 0.85, 0.60];
            [
                sunset[0] * (1.0 - t) + morning[0] * t,
                sunset[1] * (1.0 - t) + morning[1] * t,
                sunset[2] * (1.0 - t) + morning[2] * t,
            ]
        } else {
            // Midday Sun (Pure warm white)
            let t = ((elev - 0.2) / 0.8).clamp(0.0, 1.0);
            let morning = [1.0, 0.85, 0.60];
            let noon = [1.0, 0.98, 0.92];
            [
                morning[0] * (1.0 - t) + noon[0] * t,
                morning[1] * (1.0 - t) + noon[1] * t,
                morning[2] * (1.0 - t) + noon[2] * t,
            ]
        }
    }

    /// Sky Zenith (top) color
    pub fn sky_zenith_color(&self) -> [f32; 3] {
        let elev = self.solar_elevation();
        if elev <= 0.0 {
            // Starry Deep Space Night
            [0.01, 0.02, 0.05]
        } else {
            let t = elev.clamp(0.0, 1.0);
            let dusk = [0.15, 0.10, 0.30];
            let noon = [0.10, 0.35, 0.85];
            [
                dusk[0] * (1.0 - t) + noon[0] * t,
                dusk[1] * (1.0 - t) + noon[1] * t,
                dusk[2] * (1.0 - t) + noon[2] * t,
            ]
        }
    }

    /// Sky Horizon color
    pub fn sky_horizon_color(&self) -> [f32; 3] {
        let elev = self.solar_elevation();
        if elev <= 0.0 {
            [0.03, 0.05, 0.10]
        } else if elev < 0.3 {
            let t = (elev / 0.3).clamp(0.0, 1.0);
            let glow = [0.95, 0.50, 0.20];
            let day = [0.65, 0.75, 0.90];
            [
                glow[0] * (1.0 - t) + day[0] * t,
                glow[1] * (1.0 - t) + day[1] * t,
                glow[2] * (1.0 - t) + day[2] * t,
            ]
        } else {
            [0.65, 0.75, 0.90]
        }
    }
}

/// Volumetric Exponential Height Fog
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumetricFog {
    pub density: f32,
    pub height_falloff: f32,
    pub inscattering_color: [f32; 3],
    pub anisotropy_g: f32,
}

impl Default for VolumetricFog {
    fn default() -> Self {
        Self {
            density: 0.01,
            height_falloff: 0.15,
            inscattering_color: [0.6, 0.7, 0.85],
            anisotropy_g: 0.65,
        }
    }
}

impl VolumetricFog {
    /// Calculate fog extinction optical depth at a given height
    pub fn sample_density(&self, world_height: f32) -> f32 {
        self.density * (-world_height * self.height_falloff).exp()
    }
}

/// Comprehensive Atmosphere Subsystem Controller
#[derive(Debug, Clone)]
pub struct AtmosphereController {
    pub time_of_day: TimeOfDay,
    pub fog: VolumetricFog,
    pub cloud_coverage: f32,
    pub wind_speed: [f32; 2],
}

impl Default for AtmosphereController {
    fn default() -> Self {
        Self {
            time_of_day: TimeOfDay::default(),
            fog: VolumetricFog::default(),
            cloud_coverage: 0.3,
            wind_speed: [1.5, 0.5],
        }
    }
}

impl AtmosphereController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, dt: f32) {
        self.time_of_day.update(dt);
        // Sync fog inscattering with horizon color
        self.fog.inscattering_color = self.time_of_day.sky_horizon_color();
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_cycle_noon_and_midnight() {
        let noon = TimeOfDay::new(12.0, 1.0);
        let midnight = TimeOfDay::new(0.0, 1.0);

        assert!(noon.is_day());
        assert!(!midnight.is_day());
        assert!(noon.solar_elevation() > 0.0);
        assert!(midnight.solar_elevation() < 0.0);
    }

    #[test]
    fn test_fog_height_falloff() {
        let fog = VolumetricFog::default();
        let ground_density = fog.sample_density(0.0);
        let high_density = fog.sample_density(50.0);

        assert!(ground_density > high_density);
    }
}
