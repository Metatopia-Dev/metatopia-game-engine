//! Window management module

use winit::{
    event::{Event, WindowEvent as WinitWindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
    window::{Window as WinitWindow, WindowBuilder as WinitWindowBuilder},
    dpi::LogicalSize,
};
use std::sync::Arc;

/// Window event types
#[derive(Debug, Clone)]
pub enum WindowEvent {
    Resized(u32, u32),
    Moved(i32, i32),
    CloseRequested,
    Focused(bool),
    KeyboardInput {
        key: String,
        pressed: bool,
    },
    MouseInput {
        button: MouseButton,
        pressed: bool,
    },
    MouseMoved(f32, f32),
    MouseWheel(f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u16),
}

/// Window builder for configuring window creation
pub struct WindowBuilder {
    title: String,
    width: u32,
    height: u32,
    resizable: bool,
    maximized: bool,
    fullscreen: bool,
    vsync: bool,
}

impl Default for WindowBuilder {
    fn default() -> Self {
        Self {
            title: "Metatopia Engine Window".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
            maximized: false,
            fullscreen: false,
            vsync: true,
        }
    }
}

impl WindowBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
    
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    
    pub fn with_resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }
    
    pub fn with_maximized(mut self, maximized: bool) -> Self {
        self.maximized = maximized;
        self
    }
    
    pub fn with_fullscreen(mut self, fullscreen: bool) -> Self {
        self.fullscreen = fullscreen;
        self
    }
    
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }
    
    pub fn build(self) -> Result<Window, Box<dyn std::error::Error>> {
        Window::from_builder(self)
    }
}

/// Event loop wrapper that can be extracted
pub struct EventLoopWrapper {
    event_loop: EventLoop<()>,
}

impl EventLoopWrapper {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            event_loop: EventLoopBuilder::new().build()?,
        })
    }
    
    pub fn create_window(&self, builder: &WindowBuilder) -> Result<Arc<WinitWindow>, Box<dyn std::error::Error>> {
        let window = WinitWindowBuilder::new()
            .with_title(&builder.title)
            .with_inner_size(LogicalSize::new(builder.width, builder.height))
            .with_resizable(builder.resizable)
            .with_maximized(builder.maximized)
            .build(&self.event_loop)?;
        
        Ok(Arc::new(window))
    }
    
    pub fn run<F>(self, mut event_handler: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(Event<()>, &winit::event_loop::EventLoopWindowTarget<()>) + 'static,
    {
        self.event_loop.run(move |event, target| {
            event_handler(event, target);
        })?;
        Ok(())
    }
}

/// Main window struct
pub struct Window {
    window: Arc<WinitWindow>,
    events: Vec<WindowEvent>,
    should_close: bool,
    width: u32,
    height: u32,
}

impl Window {
    /// Create a new window with default settings
    pub fn new(config: &crate::core::EngineConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let builder = WindowBuilder::new()
            .with_title(&config.title)
            .with_dimensions(config.width, config.height)
            .with_resizable(config.resizable)
            .with_vsync(config.vsync);
        
        Self::from_builder(builder)
    }
    
    /// Create a window from a builder (requires event loop to be created separately)
    pub fn from_builder(builder: WindowBuilder) -> Result<Self, Box<dyn std::error::Error>> {
        // Create a temporary event loop just for window creation
        let event_loop = EventLoop::new()?;
        
        let window = WinitWindowBuilder::new()
            .with_title(&builder.title)
            .with_inner_size(LogicalSize::new(builder.width, builder.height))
            .with_resizable(builder.resizable)
            .with_maximized(builder.maximized)
            .build(&event_loop)?;
        
        let window = Arc::new(window);
        
        // Note: The event loop is dropped here, which is not ideal but allows
        // the window to be created. In production, use with_event_loop instead.
        
        Ok(Self {
            window,
            events: Vec::new(),
            should_close: false,
            width: builder.width,
            height: builder.height,
        })
    }
    
    /// Create window with existing event loop
    pub fn with_event_loop(builder: WindowBuilder, event_loop: &EventLoop<()>) -> Result<Self, Box<dyn std::error::Error>> {
        let window = WinitWindowBuilder::new()
            .with_title(&builder.title)
            .with_inner_size(LogicalSize::new(builder.width, builder.height))
            .with_resizable(builder.resizable)
            .with_maximized(builder.maximized)
            .build(event_loop)?;
        
        let window = Arc::new(window);
        
        Ok(Self {
            window,
            events: Vec::new(),
            should_close: false,
            width: builder.width,
            height: builder.height,
        })
    }
    
    /// Poll and process window events (stub for compatibility)
    pub fn poll_events(&mut self) {
        // Events are now handled in the main event loop
        // This is kept for API compatibility
    }
    
    /// Process a winit event
    pub fn handle_event(&mut self, event: &Event<()>) {
        self.events.clear();
        
        if let Event::WindowEvent { event, window_id } = event {
            if window_id == &self.window.id() {
                match event {
                    WinitWindowEvent::CloseRequested => {
                        self.should_close = true;
                        self.events.push(WindowEvent::CloseRequested);
                    }
                    WinitWindowEvent::Resized(size) => {
                        self.width = size.width;
                        self.height = size.height;
                        self.events.push(WindowEvent::Resized(size.width, size.height));
                    }
                    WinitWindowEvent::Focused(focused) => {
                        self.events.push(WindowEvent::Focused(*focused));
                    }
                    _ => {}
                }
            }
        }
    }
    
    /// Get pending window events
    pub fn events(&self) -> &[WindowEvent] {
        &self.events
    }
    
    /// Check if the window should close
    pub fn should_close(&self) -> bool {
        self.should_close
    }
    
    /// Get window dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    
    /// Get window width
    pub fn width(&self) -> u32 {
        self.width
    }
    
    /// Get window height
    pub fn height(&self) -> u32 {
        self.height
    }
    
    /// Set window title
    pub fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }
    
    /// Get the underlying winit window for wgpu surface creation
    pub fn winit_window(&self) -> &WinitWindow {
        &self.window
    }
    
    /// Get window handle as Arc for wgpu
    pub fn window_arc(&self) -> Arc<WinitWindow> {
        self.window.clone()
    }
    
    /// Request a redraw
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}