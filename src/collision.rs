//! Collision detection primitives and spatial queries.
//!
//! Provides AABB, Sphere, and Ray colliders plus a `CollisionWorld` that
//! supports point queries, ray casting, and overlap tests. All pure math —
//! no GPU dependency.
//!
//! # Example
//! ```
//! use metatopia_engine::collision::*;
//!
//! let box_a = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
//! let sphere = SphereCollider::new([0.0, 0.0, 0.0], 0.5);
//! assert!(box_a.contains_point([0.0, 0.0, 0.0]));
//! assert!(box_a.intersects_sphere(&sphere));
//! ```

use cgmath::Vector3;

// ─── AABB ──────────────────────────────────────────────────────────────────

/// Axis-Aligned Bounding Box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl AABB {
    /// Create an AABB from min/max corners.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Create an AABB centered at `center` with given half-extents.
    pub fn from_center(center: [f32; 3], half: [f32; 3]) -> Self {
        Self {
            min: [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            max: [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
        }
    }

    /// Create an AABB from a cube centered at `center` with given `size`.
    pub fn cube(center: [f32; 3], size: f32) -> Self {
        let h = size / 2.0;
        Self::from_center(center, [h, h, h])
    }

    /// Center of the AABB.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Half-extents.
    pub fn half_extents(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Check if a point is inside this AABB.
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0]
            && p[1] >= self.min[1] && p[1] <= self.max[1]
            && p[2] >= self.min[2] && p[2] <= self.max[2]
    }

    /// Check if this AABB overlaps another.
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0] && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1] && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2] && self.max[2] >= other.min[2]
    }

    /// Check if this AABB intersects a sphere.
    pub fn intersects_sphere(&self, sphere: &SphereCollider) -> bool {
        let mut dist_sq = 0.0_f32;
        for i in 0..3 {
            let v = sphere.center[i];
            if v < self.min[i] {
                dist_sq += (self.min[i] - v) * (self.min[i] - v);
            } else if v > self.max[i] {
                dist_sq += (v - self.max[i]) * (v - self.max[i]);
            }
        }
        dist_sq <= sphere.radius * sphere.radius
    }

    /// Expand this AABB to include a point.
    pub fn expand_to(&mut self, p: [f32; 3]) {
        for i in 0..3 {
            if p[i] < self.min[i] { self.min[i] = p[i]; }
            if p[i] > self.max[i] { self.max[i] = p[i]; }
        }
    }

    /// Compute the union of two AABBs.
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
}

// ─── Sphere ────────────────────────────────────────────────────────────────

/// Sphere collider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereCollider {
    pub center: [f32; 3],
    pub radius: f32,
}

impl SphereCollider {
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self { center, radius }
    }

    /// Check if a point is inside this sphere.
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }

    /// Check if this sphere overlaps another.
    pub fn intersects(&self, other: &SphereCollider) -> bool {
        let dx = other.center[0] - self.center[0];
        let dy = other.center[1] - self.center[1];
        let dz = other.center[2] - self.center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r_sum = self.radius + other.radius;
        dist_sq <= r_sum * r_sum
    }

    /// Distance from the sphere center to a point.
    pub fn distance_to(&self, p: [f32; 3]) -> f32 {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Bounding AABB for this sphere.
    pub fn bounding_aabb(&self) -> AABB {
        AABB::from_center(self.center, [self.radius, self.radius, self.radius])
    }
}

// ─── Ray ───────────────────────────────────────────────────────────────────

/// Ray defined by origin and normalized direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

/// Result of a ray intersection test.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// World-space hit point.
    pub point: [f32; 3],
    /// Normal at the hit point (approximate for AABBs).
    pub normal: [f32; 3],
}

impl Ray {
    /// Create a ray from origin and direction (will be normalized).
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        let len = (direction[0] * direction[0] + direction[1] * direction[1]
            + direction[2] * direction[2]).sqrt();
        let dir = if len > 0.0 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, 0.0, -1.0]
        };
        Self { origin, direction: dir }
    }

    /// Create a ray from cgmath types.
    pub fn from_vectors(origin: Vector3<f32>, direction: Vector3<f32>) -> Self {
        Self::new(
            [origin.x, origin.y, origin.z],
            [direction.x, direction.y, direction.z],
        )
    }

    /// Point along the ray at distance `t`.
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
            self.origin[2] + self.direction[2] * t,
        ]
    }

    /// Ray-AABB intersection (slab method). Returns `Some(RayHit)` if hit within `max_dist`.
    pub fn intersects_aabb(&self, aabb: &AABB, max_dist: f32) -> Option<RayHit> {
        let mut tmin = 0.0_f32;
        let mut tmax = max_dist;
        let mut hit_axis = 0usize;
        let mut hit_sign = 1.0_f32;

        for i in 0..3 {
            if self.direction[i].abs() < 1e-8 {
                // Parallel to slab
                if self.origin[i] < aabb.min[i] || self.origin[i] > aabb.max[i] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / self.direction[i];
                let mut t1 = (aabb.min[i] - self.origin[i]) * inv_d;
                let mut t2 = (aabb.max[i] - self.origin[i]) * inv_d;
                let mut sign = -1.0_f32;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                    sign = 1.0;
                }
                if t1 > tmin { tmin = t1; hit_axis = i; hit_sign = sign; }
                if t2 < tmax { tmax = t2; }
                if tmin > tmax { return None; }
            }
        }

        if tmin < 0.0 { return None; }

        let point = self.at(tmin);
        let mut normal = [0.0; 3];
        normal[hit_axis] = hit_sign;

        Some(RayHit { t: tmin, point, normal })
    }

    /// Ray-sphere intersection. Returns `Some(RayHit)` for the nearest hit within `max_dist`.
    pub fn intersects_sphere(&self, sphere: &SphereCollider, max_dist: f32) -> Option<RayHit> {
        let oc = [
            self.origin[0] - sphere.center[0],
            self.origin[1] - sphere.center[1],
            self.origin[2] - sphere.center[2],
        ];
        let a = self.direction[0] * self.direction[0]
            + self.direction[1] * self.direction[1]
            + self.direction[2] * self.direction[2];
        let b = 2.0 * (oc[0] * self.direction[0] + oc[1] * self.direction[1] + oc[2] * self.direction[2]);
        let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere.radius * sphere.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { return None; }

        let sqrt_disc = disc.sqrt();
        let mut t = (-b - sqrt_disc) / (2.0 * a);
        if t < 0.0 { t = (-b + sqrt_disc) / (2.0 * a); }
        if t < 0.0 || t > max_dist { return None; }

        let point = self.at(t);
        let normal = [
            (point[0] - sphere.center[0]) / sphere.radius,
            (point[1] - sphere.center[1]) / sphere.radius,
            (point[2] - sphere.center[2]) / sphere.radius,
        ];

        Some(RayHit { t, point, normal })
    }
}

// ─── Collider Enum ─────────────────────────────────────────────────────────

/// A collider that can be either an AABB or a Sphere.
#[derive(Debug, Clone, Copy)]
pub enum Collider {
    Box(AABB),
    Sphere(SphereCollider),
}

impl Collider {
    pub fn bounding_aabb(&self) -> AABB {
        match self {
            Collider::Box(aabb) => *aabb,
            Collider::Sphere(s) => s.bounding_aabb(),
        }
    }

    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        match self {
            Collider::Box(aabb) => aabb.contains_point(p),
            Collider::Sphere(s) => s.contains_point(p),
        }
    }
}

// ─── CollisionWorld ────────────────────────────────────────────────────────

/// An entry in the collision world.
#[derive(Debug, Clone)]
pub struct CollisionEntry {
    pub id: u32,
    pub collider: Collider,
    pub tag: &'static str,
}

/// Query result from the collision world.
#[derive(Debug, Clone)]
pub struct QueryHit {
    pub id: u32,
    pub tag: &'static str,
    pub ray_hit: Option<RayHit>,
}

/// Spatial collision world holding all colliders. Rebuilt each frame (or as needed).
pub struct CollisionWorld {
    entries: Vec<CollisionEntry>,
    next_id: u32,
}

impl Default for CollisionWorld {
    fn default() -> Self { Self::new() }
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self { entries: Vec::new(), next_id: 0 }
    }

    /// Remove all entries. Call at the start of each frame if rebuilding.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 0;
    }

    /// Add a collider. Returns its ID.
    pub fn add(&mut self, collider: Collider, tag: &'static str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(CollisionEntry { id, collider, tag });
        id
    }

    /// Add an AABB collider. Convenience method.
    pub fn add_aabb(&mut self, aabb: AABB, tag: &'static str) -> u32 {
        self.add(Collider::Box(aabb), tag)
    }

    /// Add a sphere collider. Convenience method.
    pub fn add_sphere(&mut self, center: [f32; 3], radius: f32, tag: &'static str) -> u32 {
        self.add(Collider::Sphere(SphereCollider::new(center, radius)), tag)
    }

    /// Find all colliders containing a point.
    pub fn query_point(&self, p: [f32; 3]) -> Vec<QueryHit> {
        self.entries.iter()
            .filter(|e| e.collider.contains_point(p))
            .map(|e| QueryHit { id: e.id, tag: e.tag, ray_hit: None })
            .collect()
    }

    /// Cast a ray and return all hits sorted by distance.
    pub fn query_ray(&self, ray: &Ray, max_dist: f32) -> Vec<QueryHit> {
        let mut hits: Vec<QueryHit> = self.entries.iter().filter_map(|e| {
            let ray_hit = match &e.collider {
                Collider::Box(aabb) => ray.intersects_aabb(aabb, max_dist),
                Collider::Sphere(s) => ray.intersects_sphere(s, max_dist),
            };
            ray_hit.map(|rh| QueryHit { id: e.id, tag: e.tag, ray_hit: Some(rh) })
        }).collect();
        hits.sort_by(|a, b| {
            let ta = a.ray_hit.as_ref().map(|h| h.t).unwrap_or(f32::MAX);
            let tb = b.ray_hit.as_ref().map(|h| h.t).unwrap_or(f32::MAX);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    /// Cast a ray and return the nearest hit.
    pub fn raycast(&self, ray: &Ray, max_dist: f32) -> Option<QueryHit> {
        self.query_ray(ray, max_dist).into_iter().next()
    }

    /// Find all entries whose bounding AABB overlaps a given sphere.
    pub fn query_sphere(&self, center: [f32; 3], radius: f32) -> Vec<QueryHit> {
        let test = SphereCollider::new(center, radius);
        self.entries.iter().filter(|e| {
            match &e.collider {
                Collider::Box(aabb) => aabb.intersects_sphere(&test),
                Collider::Sphere(s) => s.intersects(&test),
            }
        }).map(|e| QueryHit { id: e.id, tag: e.tag, ray_hit: None }).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize { self.entries.len() }

    /// Check if empty.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_contains_point() {
        let b = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(b.contains_point([0.0, 0.0, 0.0]));
        assert!(b.contains_point([1.0, 1.0, 1.0]));
        assert!(!b.contains_point([1.1, 0.0, 0.0]));
    }

    #[test]
    fn aabb_intersects() {
        let a = AABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = AABB::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let c = AABB::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn sphere_contains_point() {
        let s = SphereCollider::new([0.0, 0.0, 0.0], 1.0);
        assert!(s.contains_point([0.5, 0.0, 0.0]));
        assert!(!s.contains_point([1.5, 0.0, 0.0]));
    }

    #[test]
    fn sphere_intersects() {
        let a = SphereCollider::new([0.0, 0.0, 0.0], 1.0);
        let b = SphereCollider::new([1.5, 0.0, 0.0], 1.0);
        let c = SphereCollider::new([5.0, 0.0, 0.0], 1.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_sphere_intersection() {
        let b = AABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let s1 = SphereCollider::new([1.0, 1.0, 1.0], 0.5);
        let s2 = SphereCollider::new([5.0, 5.0, 5.0], 0.5);
        assert!(b.intersects_sphere(&s1));
        assert!(!b.intersects_sphere(&s2));
    }

    #[test]
    fn ray_aabb_hit() {
        let b = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = ray.intersects_aabb(&b, 100.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 0.01);
    }

    #[test]
    fn ray_aabb_miss() {
        let b = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let ray = Ray::new([5.0, 5.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(ray.intersects_aabb(&b, 100.0).is_none());
    }

    #[test]
    fn ray_sphere_hit() {
        let s = SphereCollider::new([0.0, 0.0, 0.0], 1.0);
        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = ray.intersects_sphere(&s, 100.0);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.t - 4.0).abs() < 0.01);
    }

    #[test]
    fn ray_sphere_miss() {
        let s = SphereCollider::new([0.0, 0.0, 0.0], 1.0);
        let ray = Ray::new([5.0, 5.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(ray.intersects_sphere(&s, 100.0).is_none());
    }

    #[test]
    fn collision_world_raycast() {
        let mut world = CollisionWorld::new();
        world.add_sphere([0.0, 0.0, 0.0], 1.0, "target");
        world.add_sphere([0.0, 0.0, -10.0], 1.0, "far");

        let ray = Ray::new([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = world.raycast(&ray, 100.0);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().tag, "target");
    }

    #[test]
    fn collision_world_point_query() {
        let mut world = CollisionWorld::new();
        world.add_aabb(AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), "box");
        world.add_sphere([5.0, 0.0, 0.0], 1.0, "sphere");

        let hits = world.query_point([0.0, 0.0, 0.0]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, "box");
    }

    #[test]
    fn collision_world_sphere_query() {
        let mut world = CollisionWorld::new();
        world.add_aabb(AABB::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), "near");
        world.add_aabb(AABB::new([100.0, 0.0, 0.0], [101.0, 1.0, 1.0]), "far");

        let hits = world.query_sphere([1.0, 1.0, 1.0], 3.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, "near");
    }

    #[test]
    fn aabb_from_center() {
        let b = AABB::from_center([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        assert_eq!(b.min, [-1.0, -2.0, -3.0]);
        assert_eq!(b.max, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn aabb_union() {
        let a = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = AABB::new([-1.0, 2.0, -1.0], [0.5, 3.0, 0.5]);
        let u = a.union(&b);
        assert_eq!(u.min, [-1.0, 0.0, -1.0]);
        assert_eq!(u.max, [1.0, 3.0, 1.0]);
    }
}
