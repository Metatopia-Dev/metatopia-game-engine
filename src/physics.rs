//! Rigid body physics simulation engine
//!
//! Provides dynamic, static, and kinematic rigid bodies with mass, velocity,
//! impulses, friction, restitution (bouncing), swept collision resolution,
//! and non-Euclidean curved gravity fields.

use cgmath::{InnerSpace, Vector3};
use crate::collision::{AABB, SphereCollider, Ray, RayHit};

/// Type of rigid body mobility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Dynamic,
    Static,
    Kinematic,
}

/// Geometric collider attached to a rigid body
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsCollider {
    Box(AABB),
    Sphere(SphereCollider),
    Capsule { radius: f32, height: f32 },
}

/// Gravity field modes (supports standard directional, point attractor, and spherical manifold curvature)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GravityField {
    Constant(Vector3<f32>),
    PointAttractor { center: Vector3<f32>, strength: f32 },
    SphericalManifold { center: Vector3<f32>, surface_radius: f32, strength: f32 },
}

/// Rigid body entity with physical attributes
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: usize,
    pub body_type: BodyType,
    pub pos: Vector3<f32>,
    pub vel: Vector3<f32>,
    pub force: Vector3<f32>,
    pub mass: f32,
    pub inv_mass: f32,
    pub restitution: f32, // Bounciness (0.0 to 1.0)
    pub friction: f32,    // Friction coefficient (0.0 to 1.0)
    pub linear_damping: f32,
    pub collider: PhysicsCollider,
    pub gravity_scale: f32,
    pub is_grounded: bool,
    pub tag: String,
}

impl RigidBody {
    /// Create a new dynamic sphere rigid body
    pub fn new_sphere(id: usize, pos: Vector3<f32>, radius: f32, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            id,
            body_type: BodyType::Dynamic,
            pos,
            vel: Vector3::new(0.0, 0.0, 0.0),
            force: Vector3::new(0.0, 0.0, 0.0),
            mass,
            inv_mass,
            restitution: 0.5,
            friction: 0.2,
            linear_damping: 0.01,
            collider: PhysicsCollider::Sphere(SphereCollider::new([pos.x, pos.y, pos.z], radius)),
            gravity_scale: 1.0,
            is_grounded: false,
            tag: String::new(),
        }
    }

    /// Create a new static or dynamic box rigid body
    pub fn new_box(id: usize, pos: Vector3<f32>, half_extents: [f32; 3], body_type: BodyType, mass: f32) -> Self {
        let inv_mass = if body_type == BodyType::Dynamic && mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            id,
            body_type,
            pos,
            vel: Vector3::new(0.0, 0.0, 0.0),
            force: Vector3::new(0.0, 0.0, 0.0),
            mass,
            inv_mass,
            restitution: 0.3,
            friction: 0.4,
            linear_damping: 0.02,
            collider: PhysicsCollider::Box(AABB::from_center([pos.x, pos.y, pos.z], half_extents)),
            gravity_scale: 1.0,
            is_grounded: false,
            tag: String::new(),
        }
    }

    /// Apply an instantaneous velocity change (impulse)
    pub fn apply_impulse(&mut self, impulse: Vector3<f32>) {
        if self.body_type == BodyType::Dynamic {
            self.vel += impulse * self.inv_mass;
        }
    }

    /// Apply a continuous force over time
    pub fn apply_force(&mut self, force: Vector3<f32>) {
        if self.body_type == BodyType::Dynamic {
            self.force += force;
        }
    }

    /// Update internal collider position to match body position
    pub fn sync_collider(&mut self) {
        match &mut self.collider {
            PhysicsCollider::Sphere(s) => {
                s.center = [self.pos.x, self.pos.y, self.pos.z];
            }
            PhysicsCollider::Box(b) => {
                let half = b.half_extents();
                *b = AABB::from_center([self.pos.x, self.pos.y, self.pos.z], half);
            }
            PhysicsCollider::Capsule { .. } => {}
        }
    }
}

/// Physics world managing bodies, gravity, and collision responses
pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    pub gravity: GravityField,
    pub floor_y: Option<f32>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            gravity: GravityField::Constant(Vector3::new(0.0, -9.81, 0.0)),
            floor_y: Some(0.0),
        }
    }

    /// Add a rigid body to the physics simulation
    pub fn add_body(&mut self, mut body: RigidBody) -> usize {
        let id = self.bodies.len();
        body.id = id;
        body.sync_collider();
        self.bodies.push(body);
        id
    }

    /// Step simulation forward by dt seconds
    pub fn step(&mut self, dt: f32) {
        let dt_clamped = dt.min(0.05);

        // 1. Apply Gravity & Forces
        for body in &mut self.bodies {
            if body.body_type != BodyType::Dynamic { continue; }

            // Compute gravitational acceleration
            let g_acc = match self.gravity {
                GravityField::Constant(g) => g,
                GravityField::PointAttractor { center, strength } => {
                    let delta = center - body.pos;
                    let dist = delta.magnitude().max(0.1);
                    (delta / dist) * (strength / (dist * dist)).min(50.0)
                }
                GravityField::SphericalManifold { center, surface_radius, strength } => {
                    let delta = center - body.pos;
                    let dist = delta.magnitude();
                    let dir = if dist > 0.0001 { delta / dist } else { Vector3::new(0.0, -1.0, 0.0) };
                    dir * strength * (dist / surface_radius).clamp(0.1, 2.0)
                }
            };

            body.vel += (g_acc * body.gravity_scale + body.force * body.inv_mass) * dt_clamped;
            body.vel *= 1.0 - (body.linear_damping * dt_clamped).clamp(0.0, 1.0);
            body.pos += body.vel * dt_clamped;
            body.force = Vector3::new(0.0, 0.0, 0.0);
            body.sync_collider();
        }

        // 2. Floor Collisions
        if let Some(floor) = self.floor_y {
            for body in &mut self.bodies {
                if body.body_type != BodyType::Dynamic { continue; }

                let radius = match &body.collider {
                    PhysicsCollider::Sphere(s) => s.radius,
                    PhysicsCollider::Box(b) => b.half_extents()[1],
                    PhysicsCollider::Capsule { radius, height } => radius + height * 0.5,
                };

                if body.pos.y - radius <= floor {
                    body.pos.y = floor + radius;
                    if body.vel.y < 0.0 {
                        body.vel.y = -body.vel.y * body.restitution;
                        if body.vel.y.abs() < 0.2 {
                            body.vel.y = 0.0;
                            body.is_grounded = true;
                        }
                    }
                    body.vel.x *= 1.0 - body.friction.clamp(0.0, 1.0) * 0.5;
                    body.vel.z *= 1.0 - body.friction.clamp(0.0, 1.0) * 0.5;
                    body.sync_collider();
                } else {
                    body.is_grounded = false;
                }
            }
        }

        // 3. Body vs Body Collisions
        let num_bodies = self.bodies.len();
        for i in 0..num_bodies {
            for j in (i + 1)..num_bodies {
                let (b1, b2) = {
                    let (left, right) = self.bodies.split_at_mut(j);
                    (&mut left[i], &mut right[0])
                };

                if b1.body_type == BodyType::Static && b2.body_type == BodyType::Static {
                    continue;
                }

                Self::resolve_pair_collision(b1, b2);
            }
        }
    }

    fn resolve_pair_collision(b1: &mut RigidBody, b2: &mut RigidBody) {
        let col1 = b1.collider.clone();
        let col2 = b2.collider.clone();

        match (col1, col2) {
            (PhysicsCollider::Sphere(s1), PhysicsCollider::Sphere(s2)) => {
                let delta = b2.pos - b1.pos;
                let dist = delta.magnitude();
                let min_dist = s1.radius + s2.radius;

                if dist < min_dist && dist > 0.0001 {
                    let normal = delta / dist;
                    let penetration = min_dist - dist;

                    let total_inv = b1.inv_mass + b2.inv_mass;
                    if total_inv > 0.0 {
                        b1.pos -= normal * (penetration * (b1.inv_mass / total_inv));
                        b2.pos += normal * (penetration * (b2.inv_mass / total_inv));
                        b1.sync_collider();
                        b2.sync_collider();
                    }

                    let rel_vel = b2.vel - b1.vel;
                    let vel_along_norm = rel_vel.dot(normal);

                    if vel_along_norm < 0.0 {
                        let e = b1.restitution.min(b2.restitution);
                        let j = -(1.0 + e) * vel_along_norm / total_inv;
                        let impulse = normal * j;

                        if b1.body_type == BodyType::Dynamic { b1.vel -= impulse * b1.inv_mass; }
                        if b2.body_type == BodyType::Dynamic { b2.vel += impulse * b2.inv_mass; }
                    }
                }
            }
            (PhysicsCollider::Sphere(s), PhysicsCollider::Box(b)) => {
                let center = b1.pos;
                let mut closest = center;
                closest.x = closest.x.clamp(b.min[0], b.max[0]);
                closest.y = closest.y.clamp(b.min[1], b.max[1]);
                closest.z = closest.z.clamp(b.min[2], b.max[2]);

                let delta = center - closest;
                let dist = delta.magnitude();
                if dist < s.radius && dist > 0.0001 {
                    let normal = delta / dist;
                    let penetration = s.radius - dist;
                    let total_inv = b1.inv_mass + b2.inv_mass;

                    if total_inv > 0.0 {
                        b1.pos += normal * (penetration * (b1.inv_mass / total_inv));
                        b2.pos -= normal * (penetration * (b2.inv_mass / total_inv));
                        b1.sync_collider();
                        b2.sync_collider();

                        let rel_vel = b1.vel - b2.vel;
                        let vel_along_norm = rel_vel.dot(normal);
                        if vel_along_norm < 0.0 {
                            let e = b1.restitution.min(b2.restitution);
                            let j = -(1.0 + e) * vel_along_norm / total_inv;
                            let impulse = normal * j;
                            if b1.body_type == BodyType::Dynamic { b1.vel += impulse * b1.inv_mass; }
                            if b2.body_type == BodyType::Dynamic { b2.vel -= impulse * b2.inv_mass; }
                        }
                    }
                }
            }
            (PhysicsCollider::Box(b), PhysicsCollider::Sphere(s)) => {
                let center = b2.pos;
                let mut closest = center;
                closest.x = closest.x.clamp(b.min[0], b.max[0]);
                closest.y = closest.y.clamp(b.min[1], b.max[1]);
                closest.z = closest.z.clamp(b.min[2], b.max[2]);

                let delta = center - closest;
                let dist = delta.magnitude();
                if dist < s.radius && dist > 0.0001 {
                    let normal = delta / dist;
                    let penetration = s.radius - dist;
                    let total_inv = b1.inv_mass + b2.inv_mass;

                    if total_inv > 0.0 {
                        b2.pos += normal * (penetration * (b2.inv_mass / total_inv));
                        b1.pos -= normal * (penetration * (b1.inv_mass / total_inv));
                        b1.sync_collider();
                        b2.sync_collider();

                        let rel_vel = b2.vel - b1.vel;
                        let vel_along_norm = rel_vel.dot(normal);
                        if vel_along_norm < 0.0 {
                            let e = b1.restitution.min(b2.restitution);
                            let j = -(1.0 + e) * vel_along_norm / total_inv;
                            let impulse = normal * j;
                            if b2.body_type == BodyType::Dynamic { b2.vel += impulse * b2.inv_mass; }
                            if b1.body_type == BodyType::Dynamic { b1.vel -= impulse * b1.inv_mass; }
                        }
                    }
                }
            }
            (PhysicsCollider::Box(b1_box), PhysicsCollider::Box(b2_box)) => {
                if b1_box.intersects(&b2_box) {
                    let delta = b2.pos - b1.pos;
                    let h1 = b1_box.half_extents();
                    let h2 = b2_box.half_extents();

                    let ox = (h1[0] + h2[0]) - delta.x.abs();
                    let oy = (h1[1] + h2[1]) - delta.y.abs();
                    let oz = (h1[2] + h2[2]) - delta.z.abs();

                    if ox > 0.0 && oy > 0.0 && oz > 0.0 {
                        let normal = if ox < oy && ox < oz {
                            Vector3::new(delta.x.signum(), 0.0, 0.0)
                        } else if oy < oz {
                            Vector3::new(0.0, delta.y.signum(), 0.0)
                        } else {
                            Vector3::new(0.0, 0.0, delta.z.signum())
                        };

                        let penetration = ox.min(oy).min(oz);
                        let total_inv = b1.inv_mass + b2.inv_mass;
                        if total_inv > 0.0 {
                            b1.pos -= normal * (penetration * (b1.inv_mass / total_inv));
                            b2.pos += normal * (penetration * (b2.inv_mass / total_inv));
                            b1.sync_collider();
                            b2.sync_collider();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Cast a ray into the physics world and return the closest hit body
    pub fn raycast(&self, ray: &Ray, max_dist: f32) -> Option<(usize, RayHit)> {
        let mut nearest: Option<(usize, RayHit)> = None;

        for body in &self.bodies {
            let hit = match &body.collider {
                PhysicsCollider::Box(b) => ray.intersects_aabb(b, max_dist),
                PhysicsCollider::Sphere(s) => ray.intersects_sphere(s, max_dist),
                PhysicsCollider::Capsule { radius, height } => {
                    let aabb = AABB::from_center(
                        [body.pos.x, body.pos.y, body.pos.z],
                        [*radius, radius + height * 0.5, *radius]
                    );
                    ray.intersects_aabb(&aabb, max_dist)
                }
            };

            if let Some(h) = hit {
                if nearest.as_ref().map_or(true, |(_, nh)| h.t < nh.t) {
                    nearest = Some((body.id, h));
                }
            }
        }

        nearest
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_freefall_and_bounce() {
        let mut world = PhysicsWorld::new();
        let body_id = world.add_body(RigidBody::new_sphere(
            0,
            Vector3::new(0.0, 5.0, 0.0),
            0.5,
            1.0,
        ));

        // Step for 1 second
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }

        let body = &world.bodies[body_id];
        assert!(body.pos.y >= 0.5); // Stayed above floor
    }

    #[test]
    fn test_point_gravity_attractor() {
        let mut world = PhysicsWorld::new();
        world.floor_y = None; // Space mode
        world.gravity = GravityField::PointAttractor {
            center: Vector3::new(0.0, 0.0, 0.0),
            strength: 50.0,
        };

        let body_id = world.add_body(RigidBody::new_sphere(
            0,
            Vector3::new(0.0, 10.0, 0.0),
            0.5,
            1.0,
        ));

        world.step(0.1);
        let body = &world.bodies[body_id];
        assert!(body.vel.y < 0.0); // Pulled down toward center
    }
}
