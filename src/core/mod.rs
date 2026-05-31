//! Core engine module
//!
//! Provides the main Engine struct, configuration, and game state trait.

use crate::ecs::World;
use crate::time::Time;

/// Configuration for the engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Window title
    pub title: String,
    /// Window width in pixels
    pub width: u32,
    /// Window height in pixels
    pub height: u32,
    /// Enable vertical sync
    pub vsync: bool,
    /// Target frames per second (None = unlimited)
    pub target_fps: Option<u32>,
    /// Whether the window is resizable
    pub resizable: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            title: "Metatopia Engine".to_string(),
            width: 1280,
            height: 720,
            vsync: true,
            target_fps: None,
            resizable: true,
        }
    }
}

/// Trait for implementing game states
pub trait GameState {
    /// Called once when the game state is initialised
    fn on_init(&mut self, engine: &mut Engine);

    /// Called every frame with the delta time in seconds
    fn on_update(&mut self, engine: &mut Engine, dt: f32);

    /// Called every frame for rendering
    fn on_render(&mut self, _engine: &mut Engine, _renderer: &mut crate::graphics::Renderer) {}

    /// Called when the game state is being cleaned up
    fn on_cleanup(&mut self, _engine: &mut Engine) {}
}

/// The main engine struct that ties all subsystems together
pub struct Engine {
    /// Engine configuration
    pub config: EngineConfig,
    /// ECS world
    pub world: World,
    /// Time management
    pub time: Time,
    /// Whether the engine is currently running
    running: bool,
}

impl Engine {
    /// Create a new engine instance with the given configuration
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            world: World::new(),
            time: Time::new(),
            running: true,
        }
    }

    /// Check if the engine is still running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Signal the engine to quit
    pub fn quit(&mut self) {
        self.running = false;
    }
}
