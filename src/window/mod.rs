//! Window management module

use winit::{
    event::{Event, WindowEvent as WinitWindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window as WinitWindow, WindowBuilder as WinitWindowBuilder},
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

/// Main window struct
pub struct Window {
    window: Arc<WinitWindow>,
    event_loop: Option<EventLoop<()>>,
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
    
    /// Create a window from a builder
    pub fn from_builder(builder: WindowBuilder) -> Result<Self, Box<dyn std::error::Error>> {
        let event_loop = EventLoop::new()?;
        
        let window = WinitWindowBuilder::new()
            .with_title(&builder.title)
            .with_inner_size(winit::dpi::LogicalSize::new(builder.width, builder.height))
            .with_resizable(builder.resizable)
            .with_maximized(builder.maximized)
            .build(&event_loop)?;
        
        let window = Arc::new(window);
        
        Ok(Self {
            window,
            event_loop: Some(event_loop),
            events: Vec::new(),
            should_close: false,
            width: builder.width,
            height: builder.height,
        })
    }
    
    /// Poll and process window events
    pub fn poll_events(&mut self) {
        self.events.clear();
        
        if let Some(event_loop) = self.event_loop.take() {
            let window = self.window.clone();
            let events = &mut self.events;
            let should_close = &mut self.should_close;
            let width = &mut self.width;
            let height = &mut self.height;
            
            event_loop.run(move |event, elwt| {
                elwt.set_control_flow(ControlFlow::Poll);
                
                match event {
                    Event::WindowEvent { event, .. } => {
                        match event {
                            WinitWindowEvent::CloseRequested => {
                                *should_close = true;
                                events.push(WindowEvent::CloseRequested);
                                elwt.exit();
                            }
                            WinitWindowEvent::Resized(size) => {
                                *width = size.width;
                                *height = size.height;
                                events.push(WindowEvent::Resized(size.width, size.height));
                            }
                            WinitWindowEvent::Focused(focused) => {
                                events.push(WindowEvent::Focused(focused));
                            }
                            _ => {}
                        }
                    }
                    Event::AboutToWait => {
                        elwt.exit();
                    }
                    _ => {}
                }
            }).ok();
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
}