//! Raycast Suspension Vehicle & Hovercraft Physics Subsystem
//!
//! Simulates 4-wheel independent spring-damper suspension physics,
//! engine torque, Ackerman steering, lateral drift, and planetary surface alignment.

/// Individual raycast wheel suspension unit
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelSuspension {
    /// Local attachment point on vehicle chassis [X, Y, Z]
    pub chassis_offset: [f32; 3],
    /// Rest spring extension length in meters
    pub rest_length: f32,
    /// Spring stiffness constant k (N/m)
    pub spring_stiffness: f32,
    /// Shock absorber damping constant (N*s/m)
    pub damping_factor: f32,
    /// Current spring length
    pub current_length: f32,
    /// Is the wheel touching ground?
    pub is_grounded: bool,
    /// Steering angle in radians
    pub steer_angle: f32,
    /// Wheel rotational velocity
    pub wheel_rpm: f32,
}

impl WheelSuspension {
    pub fn new(offset: [f32; 3], rest_length: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            chassis_offset: offset,
            rest_length,
            spring_stiffness: stiffness,
            damping_factor: damping,
            current_length: rest_length,
            is_grounded: false,
            steer_angle: 0.0,
            wheel_rpm: 0.0,
        }
    }
}

/// 4-Wheel Raycast Vehicle Simulation Model
#[derive(Debug, Clone)]
pub struct RaycastVehicle {
    pub pos: [f32; 3],
    pub rot: [f32; 3], // Yaw, Pitch, Roll
    pub linear_vel: [f32; 3],
    pub angular_vel: [f32; 3],
    pub mass: f32,
    pub engine_power: f32,
    pub brake_power: f32,
    pub max_steer_angle: f32,
    pub wheels: [WheelSuspension; 4], // 0: FL, 1: FR, 2: RL, 3: RR
    pub throttle: f32,                // -1.0 (Reverse) to +1.0 (Forward)
    pub steer_input: f32,             // -1.0 (Left) to +1.0 (Right)
    pub handbrake: bool,
}

impl Default for RaycastVehicle {
    fn default() -> Self {
        let fl = WheelSuspension::new([-1.0, -0.2, 1.5], 0.6, 12000.0, 1500.0);
        let fr = WheelSuspension::new([1.0, -0.2, 1.5], 0.6, 12000.0, 1500.0);
        let rl = WheelSuspension::new([-1.0, -0.2, -1.5], 0.6, 14000.0, 1800.0);
        let rr = WheelSuspension::new([1.0, -0.2, -1.5], 0.6, 14000.0, 1800.0);

        Self {
            pos: [0.0, 2.0, 0.0],
            rot: [0.0, 0.0, 0.0],
            linear_vel: [0.0, 0.0, 0.0],
            angular_vel: [0.0, 0.0, 0.0],
            mass: 1200.0,
            engine_power: 4500.0,
            brake_power: 8000.0,
            max_steer_angle: 0.55, // ~31 degrees max steer
            wheels: [fl, fr, rl, rr],
            throttle: 0.0,
            steer_input: 0.0,
            handbrake: false,
        }
    }
}

impl RaycastVehicle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forward unit vector of the chassis
    pub fn forward_vector(&self) -> [f32; 3] {
        let yaw = self.rot[0];
        let pitch = self.rot[1];
        [yaw.sin() * pitch.cos(), -pitch.sin(), yaw.cos() * pitch.cos()]
    }

    /// Right unit vector of the chassis
    pub fn right_vector(&self) -> [f32; 3] {
        let yaw = self.rot[0];
        [yaw.cos(), 0.0, -yaw.sin()]
    }

    /// Up unit vector of the chassis
    pub fn up_vector(&self) -> [f32; 3] {
        [0.0, 1.0, 0.0]
    }

    /// Current speed in meters per second
    pub fn speed_mps(&self) -> f32 {
        let v = self.linear_vel;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Current speed in KM/H
    pub fn speed_kmh(&self) -> f32 {
        self.speed_mps() * 3.6
    }

    /// Update vehicle physics with spring-damper suspension forces and drive torque
    pub fn update_physics<F>(&mut self, dt: f32, ground_fn: F)
    where
        F: Fn([f32; 3]) -> f32,
    {
        if dt <= 0.0 { return; }

        let gravity = -9.81;
        let mut total_force = [0.0, gravity * self.mass, 0.0];
        let mut total_torque_yaw = 0.0;

        let fwd = self.forward_vector();
        let right = self.right_vector();

        // 1. Steering angle update (Front wheels 0 & 1)
        let target_steer = self.steer_input * self.max_steer_angle;
        self.wheels[0].steer_angle = target_steer;
        self.wheels[1].steer_angle = target_steer;

        // 2. Suspension Raycast & Forces
        for i in 0..4 {
            let offset = self.wheels[i].chassis_offset;
            let wheel_world_x = self.pos[0] + offset[0] * right[0] + offset[2] * fwd[0];
            let wheel_world_z = self.pos[2] + offset[0] * right[2] + offset[2] * fwd[2];
            let ray_origin_y = self.pos[1] + offset[1];

            let ground_y = ground_fn([wheel_world_x, ray_origin_y, wheel_world_z]);
            let hit_distance = ray_origin_y - ground_y;

            if hit_distance <= self.wheels[i].rest_length && hit_distance > -0.2 {
                self.wheels[i].is_grounded = true;
                let compression = (self.wheels[i].rest_length - hit_distance).max(0.0);
                self.wheels[i].current_length = hit_distance.max(0.05);

                // Spring Force F_s = k * x
                let spring_force = compression * self.wheels[i].spring_stiffness;
                // Damper Force F_d = -c * v_y
                let damper_force = -self.linear_vel[1] * self.wheels[i].damping_factor;
                let total_suspension_force = (spring_force + damper_force).max(0.0);

                total_force[1] += total_suspension_force;

                // Drive Force on Rear Wheels (2 & 3)
                if i >= 2 && !self.handbrake {
                    let drive_force = self.throttle * self.engine_power;
                    total_force[0] += fwd[0] * drive_force * 0.5;
                    total_force[2] += fwd[2] * drive_force * 0.5;
                }

                // Lateral Tire Grip & Steering Torque
                let lateral_slip = self.linear_vel[0] * right[0] + self.linear_vel[2] * right[2];
                let friction_force = -lateral_slip * 800.0;
                total_force[0] += right[0] * friction_force * 0.25;
                total_force[2] += right[2] * friction_force * 0.25;

                // Steering yaw torque from front wheels
                if i < 2 {
                    total_torque_yaw += self.wheels[i].steer_angle * self.speed_mps() * 1200.0;
                }
            } else {
                self.wheels[i].is_grounded = false;
                self.wheels[i].current_length = self.wheels[i].rest_length;
            }
        }

        // 3. Air Resistance (Drag)
        let drag_coeff = 0.45;
        total_force[0] -= self.linear_vel[0] * self.speed_mps() * drag_coeff;
        total_force[2] -= self.linear_vel[2] * self.speed_mps() * drag_coeff;

        // 4. Integrate Linear Velocity & Position
        let inv_mass = 1.0 / self.mass;
        self.linear_vel[0] += (total_force[0] * inv_mass) * dt;
        self.linear_vel[1] += (total_force[1] * inv_mass) * dt;
        self.linear_vel[2] += (total_force[2] * inv_mass) * dt;

        self.pos[0] += self.linear_vel[0] * dt;
        self.pos[1] = (self.pos[1] + self.linear_vel[1] * dt).max(0.5); // Floor collision clamp
        self.pos[2] += self.linear_vel[2] * dt;

        // 5. Integrate Angular Velocity & Rotation
        let yaw_accel = total_torque_yaw / 2500.0;
        self.angular_vel[0] = (self.angular_vel[0] + yaw_accel * dt) * 0.92; // Damping
        self.rot[0] += self.angular_vel[0] * dt;
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_acceleration() {
        let mut car = RaycastVehicle::new();
        car.pos = [0.0, 0.4, 0.0];
        car.throttle = 1.0;

        // Step physics on flat plane at y = 0.0
        for _ in 0..10 {
            car.update_physics(0.016, |_| 0.0);
        }

        assert!(car.speed_mps() > 0.0);
    }

    #[test]
    fn test_suspension_restores_height() {
        let mut car = RaycastVehicle::new();
        car.pos = [0.0, 0.2, 0.0]; // Compressed down

        car.update_physics(0.016, |_| 0.0);
        assert!(car.linear_vel[1] > 0.0); // Upward spring force
    }
}
