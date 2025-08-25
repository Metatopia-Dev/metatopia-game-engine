//! Resource and asset management for the non-Euclidean engine

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::any::{Any, TypeId};

/// Asset loader trait
pub trait AssetLoader: Send + Sync {
    type Asset: Any + Send + Sync;
    
    fn load(&self, path: &Path) -> Result<Self::Asset, Box<dyn std::error::Error>>;
    fn extensions(&self) -> &[&str];
}

/// Resource handle
#[derive(Debug, Clone)]
pub struct ResourceHandle<T> {
    pub id: String,
    pub data: Arc<RwLock<T>>,
}

impl<T> ResourceHandle<T> {
    pub fn new(id: String, data: T) -> Self {
        Self {
            id,
            data: Arc::new(RwLock::new(data)),
        }
    }
    
    pub fn read(&self) -> std::sync::RwLockReadGuard<T> {
        self.data.read().unwrap()
    }
    
    pub fn write(&self) -> std::sync::RwLockWriteGuard<T> {
        self.data.write().unwrap()
    }
}

/// Resource storage
struct ResourceStorage {
    resources: HashMap<TypeId, HashMap<String, Box<dyn Any + Send + Sync>>>,
}

impl ResourceStorage {
    fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }
    
    fn insert<T: Any + Send + Sync + 'static>(&mut self, id: String, resource: T) {
        let type_id = TypeId::of::<T>();
        self.resources
            .entry(type_id)
            .or_insert_with(HashMap::new)
            .insert(id, Box::new(resource));
    }
    
    fn get<T: Any + Send + Sync + 'static>(&self, id: &str) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        self.resources
            .get(&type_id)?
            .get(id)?
            .downcast_ref::<T>()
    }
    
    fn get_mut<T: Any + Send + Sync + 'static>(&mut self, id: &str) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        self.resources
            .get_mut(&type_id)?
            .get_mut(id)?
            .downcast_mut::<T>()
    }
    
    fn remove<T: Any + Send + Sync + 'static>(&mut self, id: &str) -> Option<Box<T>> {
        let type_id = TypeId::of::<T>();
        self.resources
            .get_mut(&type_id)?
            .remove(id)?
            .downcast::<T>()
            .ok()
    }
}

/// Resource manager
pub struct ResourceManager {
    storage: Arc<RwLock<ResourceStorage>>,
    asset_path: PathBuf,
    loaders: HashMap<String, Box<dyn AssetLoader<Asset = Box<dyn Any + Send + Sync>>>>,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(ResourceStorage::new())),
            asset_path: PathBuf::from("assets"),
            loaders: HashMap::new(),
        }
    }
    
    /// Set the base asset path
    pub fn set_asset_path(&mut self, path: impl Into<PathBuf>) {
        self.asset_path = path.into();
    }
    
    /// Load a resource from file
    pub fn load<T: Any + Send + Sync + Clone + 'static>(
        &mut self,
        id: &str,
        path: &str,
    ) -> Result<ResourceHandle<T>, Box<dyn std::error::Error>> {
        let full_path = self.asset_path.join(path);
        
        // Check if already loaded
        if let Ok(storage) = self.storage.read() {
            if let Some(resource) = storage.get::<T>(id) {
                return Ok(ResourceHandle::new(id.to_string(), resource.clone()));
            }
        }
        
        // Load from file
        let extension = full_path.extension()
            .and_then(|ext| ext.to_str())
            .ok_or("No file extension")?;
        
        if let Some(loader) = self.loaders.get(extension) {
            let asset = loader.load(&full_path)?;
            
            if let Ok(resource) = asset.downcast::<T>() {
                let resource = *resource;
                self.storage.write().unwrap().insert(id.to_string(), resource.clone());
                Ok(ResourceHandle::new(id.to_string(), resource))
            } else {
                Err("Type mismatch".into())
            }
        } else {
            Err(format!("No loader for extension: {}", extension).into())
        }
    }
    
    /// Add a resource directly
    pub fn add<T: Any + Send + Sync + Clone + 'static>(&mut self, id: &str, resource: T) -> ResourceHandle<T> {
        self.storage.write().unwrap().insert(id.to_string(), resource.clone());
        ResourceHandle::new(id.to_string(), resource)
    }
    
    /// Get a resource by ID
    pub fn get<T: Any + Send + Sync + Clone + 'static>(&self, id: &str) -> Option<ResourceHandle<T>> {
        self.storage.read().unwrap().get::<T>(id)
            .cloned()
            .map(|resource| ResourceHandle::new(id.to_string(), resource))
    }
    
    /// Remove a resource
    pub fn remove<T: Any + Send + Sync + 'static>(&mut self, id: &str) -> Option<Box<T>> {
        self.storage.write().unwrap().remove::<T>(id)
    }
    
    /// Check if a resource exists
    pub fn exists<T: Any + Send + Sync + 'static>(&self, id: &str) -> bool {
        self.storage.read().unwrap().get::<T>(id).is_some()
    }
}

/// Mesh resource
#[derive(Clone)]
pub struct MeshResource {
    pub vertices: Vec<crate::graphics::Vertex>,
    pub indices: Vec<u16>,
}

/// Shader resource
#[derive(Clone)]
pub struct ShaderResource {
    pub vertex_source: String,
    pub fragment_source: String,
    pub geometry_type: crate::manifold::GeometryType,
}

/// Texture resource
#[derive(Clone)]
pub struct TextureResource {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: TextureFormat,
}

#[derive(Debug, Clone, Copy)]
pub enum TextureFormat {
    Rgba8,
    Rgb8,
    R8,
    Rgba16F,
}

/// World resource for non-Euclidean levels
#[derive(Clone)]
pub struct WorldResource {
    pub manifold: crate::manifold::Manifold,
    pub spawn_points: Vec<(crate::manifold::ChartId, cgmath::Point3<f32>)>,
    pub metadata: WorldMetadata,
}

#[derive(Clone)]
pub struct WorldMetadata {
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: String,
}

impl Default for WorldMetadata {
    fn default() -> Self {
        Self {
            name: "Untitled World".to_string(),
            description: String::new(),
            author: "Unknown".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}