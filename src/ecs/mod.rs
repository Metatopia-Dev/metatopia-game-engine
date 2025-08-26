//! Entity Component System for the non-Euclidean engine

use std::any::{Any, TypeId};
use std::collections::HashMap;
use cgmath::{Point3, Quaternion, InnerSpace};
use crate::manifold::{ManifoldPosition, ManifoldOrientation, ChartId};

/// Entity identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub u32);

/// Component trait that all components must implement
pub trait Component: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Transform component for non-Euclidean spaces
#[derive(Debug, Clone)]
pub struct Transform {
    pub position: ManifoldPosition,
    pub orientation: ManifoldOrientation,
    pub scale: f32,
}

impl Transform {
    pub fn new(chart_id: ChartId, position: Point3<f32>) -> Self {
        Self {
            position: ManifoldPosition::new(chart_id, position),
            orientation: ManifoldOrientation::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            scale: 1.0,
        }
    }
}

impl Component for Transform {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Velocity component for physics
#[derive(Debug, Clone)]
pub struct Velocity {
    pub linear: cgmath::Vector3<f32>,
    pub angular: cgmath::Vector3<f32>,
}

impl Component for Velocity {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Renderable component
#[derive(Debug, Clone)]
pub struct Renderable {
    pub mesh_id: String,
    pub shader_id: String,
    pub visible: bool,
}

impl Component for Renderable {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Portal entity marker
#[derive(Debug, Clone)]
pub struct PortalEntity {
    pub portal_id: crate::manifold::PortalId,
    pub active: bool,
}

impl Component for PortalEntity {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

/// Component storage
struct ComponentStorage {
    components: HashMap<TypeId, HashMap<Entity, Box<dyn Component>>>,
}

impl ComponentStorage {
    fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }
    
    fn add_component<T: Component + 'static>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        self.components
            .entry(type_id)
            .or_insert_with(HashMap::new)
            .insert(entity, Box::new(component));
    }
    
    fn get_component<T: Component + 'static>(&self, entity: Entity) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)?
            .get(&entity)?
            .as_any()
            .downcast_ref::<T>()
    }
    
    fn get_component_mut<T: Component + 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)?
            .get_mut(&entity)?
            .as_any_mut()
            .downcast_mut::<T>()
    }
    
    fn remove_component<T: Component + 'static>(&mut self, entity: Entity) -> Option<Box<dyn Component>> {
        let type_id = TypeId::of::<T>();
        self.components
            .get_mut(&type_id)?
            .remove(&entity)
    }
    
    fn remove_all_components(&mut self, entity: Entity) {
        for (_, components) in self.components.iter_mut() {
            components.remove(&entity);
        }
    }
}

/// ECS World containing all entities and components
pub struct World {
    entities: Vec<Entity>,
    next_entity_id: u32,
    components: ComponentStorage,
    systems: Vec<Box<dyn System>>,
}

impl World {
    /// Create a new empty world
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            next_entity_id: 0,
            components: ComponentStorage::new(),
            systems: Vec::new(),
        }
    }
    
    /// Create a new entity
    pub fn create_entity(&mut self) -> Entity {
        let entity = Entity(self.next_entity_id);
        self.next_entity_id += 1;
        self.entities.push(entity);
        entity
    }
    
    /// Destroy an entity and all its components
    pub fn destroy_entity(&mut self, entity: Entity) {
        if let Some(pos) = self.entities.iter().position(|&e| e == entity) {
            self.entities.remove(pos);
            self.components.remove_all_components(entity);
        }
    }
    
    /// Add a component to an entity
    pub fn add_component<T: Component + 'static>(&mut self, entity: Entity, component: T) {
        self.components.add_component(entity, component);
    }
    
    /// Get a component from an entity
    pub fn get_component<T: Component + 'static>(&self, entity: Entity) -> Option<&T> {
        self.components.get_component(entity)
    }
    
    /// Get a mutable component from an entity
    pub fn get_component_mut<T: Component + 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.components.get_component_mut(entity)
    }
    
    /// Remove a component from an entity
    pub fn remove_component<T: Component + 'static>(&mut self, entity: Entity) {
        self.components.remove_component::<T>(entity);
    }
    
    /// Query entities with specific components
    pub fn query<T: Component + 'static>(&self) -> Vec<Entity> {
        let type_id = TypeId::of::<T>();
        if let Some(components) = self.components.components.get(&type_id) {
            components.keys().copied().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Query entities with two component types
    pub fn query2<T1: Component + 'static, T2: Component + 'static>(&self) -> Vec<Entity> {
        let entities1 = self.query::<T1>();
        entities1
            .into_iter()
            .filter(|&e| self.get_component::<T2>(e).is_some())
            .collect()
    }
    
    /// Add a system to the world
    pub fn add_system(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
    }
    
    /// Update all systems
    pub fn update(&mut self, dt: f32) {
        // Clone systems to avoid borrow issues
        let systems = self.systems.clone();
        for system in systems.iter() {
            system.update(self, dt);
        }
    }
    
    /// Get all entities
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
}

/// System trait for ECS systems
pub trait System: Send + Sync {
    fn update(&self, world: &mut World, dt: f32);
    fn clone_box(&self) -> Box<dyn System>;
}

impl Clone for Box<dyn System> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Transform system that handles manifold positions
pub struct TransformSystem;

impl System for TransformSystem {
    fn update(&self, world: &mut World, _dt: f32) {
        let entities = world.query::<Transform>();
        for entity in entities {
            // Update transform positions based on manifold rules
            if let Some(transform) = world.get_component_mut::<Transform>(entity) {
                // Wrap coordinates if needed
                // This would interact with the manifold system
            }
        }
    }
    
    fn clone_box(&self) -> Box<dyn System> {
        Box::new(TransformSystem)
    }
}

/// Portal transition system
pub struct PortalTransitionSystem {
    manifold: std::sync::Arc<std::sync::RwLock<crate::manifold::Manifold>>,
}

impl PortalTransitionSystem {
    pub fn new(manifold: std::sync::Arc<std::sync::RwLock<crate::manifold::Manifold>>) -> Self {
        Self { manifold }
    }
}

impl System for PortalTransitionSystem {
    fn update(&self, world: &mut World, _dt: f32) {
        let entities = world.query2::<Transform, Velocity>();
        
        for entity in entities {
            // Collect portal transition data first to avoid borrow conflicts
            let transition_data = {
                let transform = world.get_component::<Transform>(entity);
                let velocity = world.get_component::<Velocity>(entity);
                
                if let (Some(transform), Some(velocity)) = (transform, velocity) {
                    if let Ok(manifold) = self.manifold.read() {
                        let position = transform.position.local.to_point();
                        let direction = velocity.linear.normalize();
                        
                        if let Some((_portal_id, intersection, new_chart)) =
                            manifold.ray_portal_intersection(position, direction, transform.position.chart_id) {
                            
                            let path = manifold.compute_geodesic(
                                position,
                                intersection,
                                transform.position.chart_id,
                                10
                            );
                            
                            Some((new_chart, intersection, path))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            
            // Apply the transition if needed
            if let Some((new_chart, intersection, path)) = transition_data {
                if let Some(transform_mut) = world.get_component_mut::<Transform>(entity) {
                    transform_mut.position.chart_id = new_chart;
                    transform_mut.position.local = crate::manifold::LocalCoordinate::from_point(intersection);
                    
                    // Update orientation with parallel transport
                    if let Some(path) = path {
                        if let Ok(manifold) = self.manifold.read() {
                            transform_mut.orientation.transport_along(&path, &manifold, new_chart);
                        }
                    }
                }
            }
        }
    }
    
    fn clone_box(&self) -> Box<dyn System> {
        Box::new(PortalTransitionSystem::new(self.manifold.clone()))
    }
}