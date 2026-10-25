//! 4D Lorentzian Spacetime Metrics, Schwarzschild Black Holes & Morris-Thorne Wormholes
//!
//! Extends Metatopia's Riemannian manifold subsystem (`src/manifold/`) to 4D Lorentzian
//! spacetimes (-,+,+,+) for general-relativistic ray-tracing, gravitational lensing,
//! accretion disk Doppler beaming, and traversable Morris-Thorne wormholes:
//!
//! 1. **Schwarzschild Metric** (in geometric units $G = c = 1$, $r_s = 2M$):
//!    $$ds^2 = -\left(1 - \frac{r_s}{r}\right) dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2 (d\theta^2 + \sin^2\theta\,d\phi^2)$$
//!    - Event Horizon: $r_{\text{eh}} = r_s = 2M$
//!    - Photon Sphere (unstable circular null orbit): $r_{\text{ph}} = \frac{3}{2} r_s = 3M$
//!    - Innermost Stable Circular Orbit (ISCO): $r_{\text{ISCO}} = 3 r_s = 6M$
//!    - Exact 3D null geodesic acceleration in isotropic/Cartesian coordinates:
//!      $$\frac{d^2 \mathbf{x}}{d\lambda^2} = -\frac{3}{2} r_s \frac{\|\mathbf{x} \times \mathbf{v}\|^2}{\|\mathbf{x}\|^5} \mathbf{x}$$
//!
//! 2. **Morris-Thorne Traversable Wormhole Metric**:
//!    $$ds^2 = -dt^2 + dl^2 + (b_0^2 + l^2)(d\theta^2 + \sin^2\theta\,d\phi^2)$$
//!    where $l \in (-\infty, +\infty)$ is the proper radial distance through the wormhole throat
//!    of minimum radius $b_0 > 0$, connecting Universe A ($l > 0$, `chart_a`) and
//!    Universe B ($l < 0$, `chart_b`).

use cgmath::{InnerSpace, Point3, Vector3};
use crate::manifold::ChartId;

/// Schwarzschild Black Hole configuration and relativistic optics calculator
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildBlackHole {
    /// Center position in local chart coordinates
    pub center: Point3<f32>,
    /// Schwarzschild radius $r_s = 2GM/c^2$ (Event Horizon radius)
    pub schwarzschild_radius: f32,
    /// Inner accretion disk radius (typically $r_{\text{ISCO}} = 3 r_s$)
    pub accretion_inner_radius: f32,
    /// Outer accretion disk radius (typically $8 r_s .. 14 r_s$)
    pub accretion_outer_radius: f32,
    /// Accretion disk normal vector (spin axis)
    pub disk_normal: Vector3<f32>,
}

impl SchwarzschildBlackHole {
    /// Construct a Schwarzschild black hole with standard relativistic radii
    pub fn new(center: Point3<f32>, schwarzschild_radius: f32) -> Self {
        let rs = schwarzschild_radius.max(1e-4);
        Self {
            center,
            schwarzschild_radius: rs,
            accretion_inner_radius: 3.0 * rs, // ISCO = 3 r_s = 6M
            accretion_outer_radius: 9.0 * rs,
            disk_normal: Vector3::new(0.0, 1.0, 0.0),
        }
    }

    /// Unstable circular photon sphere radius $r_{\text{ph}} = 1.5 r_s = 3M$
    #[inline]
    pub fn photon_sphere_radius(&self) -> f32 {
        1.5 * self.schwarzschild_radius
    }

    /// Innermost Stable Circular Orbit (ISCO) for massive particles $r_{\text{ISCO}} = 3.0 r_s = 6M$
    #[inline]
    pub fn isco_radius(&self) -> f32 {
        3.0 * self.schwarzschild_radius
    }

    /// Critical impact parameter for photon capture $b_{\text{crit}} = \frac{3\sqrt{3}}{2} r_s$
    #[inline]
    pub fn critical_impact_parameter(&self) -> f32 {
        2.598_076_2 * self.schwarzschild_radius
    }

    /// Gravitational time dilation / redshift factor $\sqrt{1 - r_s / r}$ for $r > r_s$
    #[inline]
    pub fn lapse_function(&self, r: f32) -> f32 {
        if r <= self.schwarzschild_radius {
            0.0
        } else {
            (1.0 - self.schwarzschild_radius / r).sqrt()
        }
    }

    /// Exact general-relativistic null geodesic acceleration in 3D Cartesian coordinates
    /// derived from Binet's null geodesic equation $\frac{d^2 u}{d\phi^2} + u = \frac{3}{2} r_s u^2$:
    /// $$\mathbf{a}_{\text{GR}}(\mathbf{x}, \mathbf{v}) = -\frac{3}{2} r_s \frac{\|\mathbf{r} \times \mathbf{v}\|^2}{\|\mathbf{r}\|^5} \mathbf{r}$$
    ///
    /// Notice that for a photon injected tangentially at the photon sphere $r = 1.5 r_s$
    /// with $\|\mathbf{v}\| = 1$, $\|\mathbf{r} \times \mathbf{v}\|^2 = r^2$, so:
    /// $\|\mathbf{a}_{\text{GR}}\| = \frac{3}{2} r_s \frac{r^2}{r^4} = \frac{1.5 r_s}{r^2} = \frac{r}{r^2} = \frac{1}{r} = \frac{\|\mathbf{v}\|^2}{r}$,
    /// which is the exact centripetal acceleration required for a circular photon orbit at $r = 1.5 r_s$!
    pub fn null_geodesic_acceleration(&self, pos: Point3<f32>, dir: Vector3<f32>) -> Vector3<f32> {
        let rel = pos - self.center;
        let r2 = rel.magnitude2();
        let min_r2 = (self.schwarzschild_radius * 0.5).powi(2).max(1e-6);
        if r2 < min_r2 {
            return Vector3::new(0.0, 0.0, 0.0);
        }
        let r = r2.sqrt();
        let r5 = r2 * r2 * r;
        let h_vec = rel.cross(dir);
        let h2 = h_vec.magnitude2();
        let scale = -1.5 * self.schwarzschild_radius * h2 / r5;
        rel * scale
    }

    /// Weak-field Einstein gravitational deflection angle $\Delta\phi = \frac{2 r_s}{b}$ (radians)
    /// for impact parameter $b \gg r_s$
    pub fn weak_field_deflection_angle(&self, impact_parameter: f32) -> f32 {
        2.0 * self.schwarzschild_radius / impact_parameter.max(self.critical_impact_parameter())
    }

    /// Relativistic Keplerian orbital speed $\beta = v/c = \sqrt{\frac{r_s}{2(r - r_s)}}$
    /// and Doppler beaming factor $\delta = \frac{\sqrt{1 - r_s/r}}{\gamma (1 - \beta \cos\theta)}$
    /// for an accretion disk element at position `pos` viewed along ray direction `ray_dir`.
    /// Returns `(doppler_delta, specific_intensity_boost)` where $I_{\text{obs}} / I_{\text{emit}} = \delta^4$.
    pub fn accretion_disk_doppler(&self, pos: Point3<f32>, ray_dir: Vector3<f32>) -> (f32, f32) {
        let rel = pos - self.center;
        let r = rel.magnitude().max(self.schwarzschild_radius * 1.55);
        // Tangential orbital direction in disk plane
        let radial_in_plane = rel - self.disk_normal * rel.dot(self.disk_normal);
        let tangent = self.disk_normal.cross(radial_in_plane);
        if tangent.magnitude2() < 1e-8 {
            return (1.0, 1.0);
        }
        let v_hat = tangent.normalize();
        // Exact Schwarzschild local-frame circular orbital velocity beta = sqrt(r_s / (2(r - r_s)))
        let denom = (2.0 * (r - self.schwarzschild_radius)).max(self.schwarzschild_radius);
        let beta = (self.schwarzschild_radius / denom).sqrt().clamp(0.0, 0.75);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        // Photon propagation from disk to observer is -ray_dir
        let photon_to_obs = -ray_dir.normalize();
        let cos_theta = v_hat.dot(photon_to_obs).clamp(-0.99, 0.99);
        let grav_redshift = self.lapse_function(r).clamp(0.15, 1.0);
        let delta = (grav_redshift / (gamma * (1.0 - beta * cos_theta))).clamp(0.1, 4.0);
        let intensity_boost = delta.powi(4);
        (delta, intensity_boost)
    }

    /// Trace a photon ray in the Schwarzschild spacetime using 4th-Order Runge-Kutta (RK4).
    /// Note: For constant-speed null curve integration where $\|\mathbf{v}\| = 1$ at $t=0$,
    /// we re-normalize $\mathbf{v}$ at each step so affine parameter $\lambda$ corresponds to
    /// Euclidean arc length along the spatial trajectory.
    pub fn trace_null_ray_rk4(
        &self,
        origin: Point3<f32>,
        direction: Vector3<f32>,
        dt: f32,
        steps: usize,
    ) -> SchwarzschildRayResult {
        let mut x = origin;
        let mut v = direction.normalize();
        let mut points = Vec::with_capacity(steps + 1);
        points.push(x);
        let mut captured_by_horizon = false;
        let mut closest_approach = (x - self.center).magnitude();

        for _ in 0..steps {
            let r = (x - self.center).magnitude();
            if r < closest_approach {
                closest_approach = r;
            }
            if r <= self.schwarzschild_radius * 1.01 {
                captured_by_horizon = true;
                break;
            }

            let k1_x = v;
            let k1_v = self.null_geodesic_acceleration(x, v);

            let x2 = x + k1_x * (0.5 * dt);
            let v2 = (v + k1_v * (0.5 * dt)).normalize();
            let k2_x = v2;
            let k2_v = self.null_geodesic_acceleration(x2, v2);

            let x3 = x + k2_x * (0.5 * dt);
            let v3 = (v + k2_v * (0.5 * dt)).normalize();
            let k3_x = v3;
            let k3_v = self.null_geodesic_acceleration(x3, v3);

            let x4 = x + k3_x * dt;
            let v4 = (v + k3_v * dt).normalize();
            let k4_x = v4;
            let k4_v = self.null_geodesic_acceleration(x4, v4);

            x = x + (k1_x + k2_x * 2.0 + k3_x * 2.0 + k4_x) * (dt / 6.0);
            v = (v + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0)).normalize();
            points.push(x);
        }

        SchwarzschildRayResult {
            points,
            final_direction: v,
            captured_by_horizon,
            closest_approach,
        }
    }
}

/// Result of tracing a relativistic photon trajectory around a Schwarzschild black hole
#[derive(Debug, Clone)]
pub struct SchwarzschildRayResult {
    pub points: Vec<Point3<f32>>,
    pub final_direction: Vector3<f32>,
    pub captured_by_horizon: bool,
    pub closest_approach: f32,
}

/// Ellis / Morris-Thorne Traversable Wormhole connecting two manifold charts (`chart_a` at $l > 0$
/// and `chart_b` at $l < 0$) through a smooth zero-tidal-force spherical throat of radius $b_0$.
///
/// Metric:
/// $$ds^2 = -dt^2 + dl^2 + (b_0^2 + l^2)(d\theta^2 + \sin^2\theta\,d\phi^2)$$
#[derive(Debug, Clone, Copy)]
pub struct MorrisThorneWormhole {
    pub chart_a: ChartId,
    pub chart_b: ChartId,
    pub throat_center_a: Point3<f32>,
    pub throat_center_b: Point3<f32>,
    /// Minimum throat radius $b_0 > 0$
    pub throat_radius: f32,
}

/// State of a particle or photon in Morris-Thorne wormhole coordinates $(l, \theta, \phi)$
#[derive(Debug, Clone, Copy)]
pub struct WormholeState {
    /// Proper radial coordinate $l \in (-\infty, +\infty)$ ($l > 0$ in `chart_a`, $l < 0$ in `chart_b`)
    pub l: f32,
    /// Polar angle $\theta \in (0, \pi)$
    pub theta: f32,
    /// Azimuthal angle $\phi \in (-\pi, \pi]$
    pub phi: f32,
    /// Radial velocity $\dot{l}$
    pub dl: f32,
    /// Polar angular velocity $\dot{\theta}$
    pub dtheta: f32,
    /// Azimuthal angular velocity $\dot{\phi}$
    pub dphi: f32,
}

impl MorrisThorneWormhole {
    pub fn new(
        chart_a: ChartId,
        chart_b: ChartId,
        throat_center_a: Point3<f32>,
        throat_center_b: Point3<f32>,
        throat_radius: f32,
    ) -> Self {
        Self {
            chart_a,
            chart_b,
            throat_center_a,
            throat_center_b,
            throat_radius: throat_radius.max(0.05),
        }
    }

    /// Circumferential areal radius $r(l) = \sqrt{b_0^2 + l^2} \ge b_0$
    #[inline]
    pub fn circumferential_radius(&self, l: f32) -> f32 {
        (self.throat_radius * self.throat_radius + l * l).sqrt()
    }

    /// Embedding diagram profile $z(r) = b_0 \operatorname{arcosh}(r / b_0)$ for $r \ge b_0$
    #[inline]
    pub fn embedding_z(&self, r: f32) -> f32 {
        let ratio = (r / self.throat_radius).max(1.0);
        self.throat_radius * ratio.acosh()
    }

    /// Active manifold chart for proper coordinate $l$ (`chart_a` when $l \ge 0$, `chart_b` when $l < 0$)
    #[inline]
    pub fn active_chart(&self, l: f32) -> ChartId {
        if l >= 0.0 {
            self.chart_a
        } else {
            self.chart_b
        }
    }

    /// Convert wormhole state $(l, \theta, \phi)$ into 3D world position and active `ChartId`
    pub fn to_world_position(&self, state: &WormholeState) -> (ChartId, Point3<f32>) {
        let r_areal = self.circumferential_radius(state.l);
        let sin_t = state.theta.sin();
        let offset = Vector3::new(
            r_areal * sin_t * state.phi.cos(),
            r_areal * state.theta.cos(),
            r_areal * sin_t * state.phi.sin(),
        );
        if state.l >= 0.0 {
            (self.chart_a, self.throat_center_a + offset)
        } else {
            (self.chart_b, self.throat_center_b + offset)
        }
    }

    /// Evaluate the exact geodesic ODE right-hand side on the Morris-Thorne wormhole:
    /// - $\ddot{l} = l (\dot{\theta}^2 + \sin^2\theta\,\dot{\phi}^2)$
    /// - $\ddot{\theta} = -\frac{2 l}{b_0^2 + l^2} \dot{l}\dot{\theta} + \sin\theta\cos\theta\,\dot{\phi}^2$
    /// - $\ddot{\phi} = -\frac{2 l}{b_0^2 + l^2} \dot{l}\dot{\phi} - 2 \cot\theta\,\dot{\theta}\dot{\phi}$
    fn geodesic_rhs(&self, s: &WormholeState) -> (f32, f32, f32, f32, f32, f32) {
        let r2 = self.throat_radius * self.throat_radius + s.l * s.l;
        let sin_t = s.theta.sin().abs().max(1e-4);
        let cos_t = s.theta.cos();
        let cot_t = cos_t / sin_t;

        let ddl = s.l * (s.dtheta * s.dtheta + sin_t * sin_t * s.dphi * s.dphi);
        let ddtheta = (-2.0 * s.l / r2) * s.dl * s.dtheta + sin_t * cos_t * s.dphi * s.dphi;
        let ddphi = (-2.0 * s.l / r2) * s.dl * s.dphi - 2.0 * cot_t * s.dtheta * s.dphi;
        (s.dl, s.dtheta, s.dphi, ddl, ddtheta, ddphi)
    }

    /// Advance a particle or photon state by `dt` along a Morris-Thorne wormhole geodesic via RK4
    pub fn step_geodesic_rk4(&self, state: &WormholeState, dt: f32) -> WormholeState {
        let add = |s: &WormholeState, k: (f32, f32, f32, f32, f32, f32), scale: f32| WormholeState {
            l: s.l + k.0 * scale,
            theta: (s.theta + k.1 * scale).clamp(0.05, std::f32::consts::PI - 0.05),
            phi: s.phi + k.2 * scale,
            dl: s.dl + k.3 * scale,
            dtheta: s.dtheta + k.4 * scale,
            dphi: s.dphi + k.5 * scale,
        };

        let k1 = self.geodesic_rhs(state);
        let s2 = add(state, k1, 0.5 * dt);
        let k2 = self.geodesic_rhs(&s2);
        let s3 = add(state, k2, 0.5 * dt);
        let k3 = self.geodesic_rhs(&s3);
        let s4 = add(state, k3, dt);
        let k4 = self.geodesic_rhs(&s4);

        let w = dt / 6.0;
        let raw_phi = state.phi + (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) * w;
        let wrapped_phi = (raw_phi + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        WormholeState {
            l: state.l + (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) * w,
            theta: (state.theta + (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) * w)
                .clamp(0.05, std::f32::consts::PI - 0.05),
            phi: wrapped_phi,
            dl: state.dl + (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3) * w,
            dtheta: state.dtheta + (k1.4 + 2.0 * k2.4 + 2.0 * k3.4 + k4.4) * w,
            dphi: state.dphi + (k1.5 + 2.0 * k2.5 + 2.0 * k3.5 + k4.5) * w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_photon_sphere_circular_orbit() {
        let bh = SchwarzschildBlackHole::new(Point3::new(0.0, 0.0, 0.0), 1.0);
        let r_ph = bh.photon_sphere_radius(); // 1.5 * r_s = 1.5
        assert!((r_ph - 1.5).abs() < 1e-6);

        // Inject photon tangentially on the photon sphere r = 1.5
        let res = bh.trace_null_ray_rk4(
            Point3::new(r_ph, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            0.01,
            200,
        );
        assert!(!res.captured_by_horizon);
        let final_r = (res.points.last().unwrap() - bh.center).magnitude();
        // Analytical circular orbit at r = 1.5 r_s must remain at r = 1.5 to high precision
        assert!(
            (final_r - r_ph).abs() < 1e-3,
            "Photon sphere orbit drifted: final_r = {} vs r_ph = {}",
            final_r,
            r_ph
        );
    }

    #[test]
    fn test_schwarzschild_horizon_capture_and_doppler_asymmetry() {
        let bh = SchwarzschildBlackHole::new(Point3::new(0.0, 0.0, 0.0), 1.0);
        // Radial ingoing ray is captured by the event horizon r = r_s
        let res = bh.trace_null_ray_rk4(
            Point3::new(0.0, 0.0, 3.0),
            Vector3::new(0.0, 0.0, -1.0),
            0.05,
            100,
        );
        assert!(res.captured_by_horizon);

        // Accretion disk Doppler beaming: approaching side (+x or -x depending on spin) vs receding side
        let (_, boost_pos_x) = bh.accretion_disk_doppler(
            Point3::new(4.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        );
        let (_, boost_neg_x) = bh.accretion_disk_doppler(
            Point3::new(-4.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        );
        assert!(
            (boost_pos_x - boost_neg_x).abs() > 0.5,
            "Relativistic Doppler beaming must produce strong approaching/receding asymmetry"
        );
    }

    #[test]
    fn test_morris_thorne_wormhole_throat_traversal() {
        let wh = MorrisThorneWormhole::new(
            ChartId(0),
            ChartId(1),
            Point3::new(0.0, 1.5, -5.0),
            Point3::new(0.0, 1.5, 5.0),
            1.2,
        );
        // Particle starts in Universe A (l = +2.5) moving through throat (dl = -1.0) with angular momentum
        let mut state = WormholeState {
            l: 2.5,
            theta: std::f32::consts::FRAC_PI_2,
            phi: 0.0,
            dl: -1.5,
            dtheta: 0.0,
            dphi: 0.25,
        };
        assert_eq!(wh.active_chart(state.l), ChartId(0));

        // Conserved angular momentum J = (b0^2 + l^2) * dphi in equatorial plane (theta = pi/2)
        let j_initial = (wh.throat_radius * wh.throat_radius + state.l * state.l) * state.dphi;

        for _ in 0..80 {
            state = wh.step_geodesic_rk4(&state, 0.05);
        }

        // Must have smoothly traversed throat into Universe B (l < 0, ChartId(1))
        assert!(state.l < -0.5, "Expected state.l < -0.5, got {}", state.l);
        assert_eq!(wh.active_chart(state.l), ChartId(1));

        let j_final = (wh.throat_radius * wh.throat_radius + state.l * state.l) * state.dphi;
        assert!(
            (j_initial - j_final).abs() < 1e-3,
            "Angular momentum conservation violated: {} vs {}",
            j_initial,
            j_final
        );
    }
}
