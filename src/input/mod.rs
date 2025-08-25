//! Input handling for the non-Euclidean engine

use std::collections::{HashMap, HashSet};
use cgmath::{Point2, Vector2};

/// Input event types
#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyPressed(KeyCode),
    KeyReleased(KeyCode),
    MouseButtonPressed(MouseButton),
    MouseButtonReleased(MouseButton),
    MouseMoved(f32, f32),
    MouseWheel(f32),
    GamepadButtonPressed(GamepadButton),
    GamepadButtonReleased(GamepadButton),
    GamepadAxisMoved(GamepadAxis, f32),
}

/// Keyboard key codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Space, Enter, Escape, Tab, Backspace, Delete,
    Up, Down, Left, Right,
    LeftShift, RightShift, LeftCtrl, RightCtrl, LeftAlt, RightAlt,
    Unknown,
}

/// Mouse buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Extra1,
    Extra2,
}

/// Gamepad buttons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadButton {
    A, B, X, Y,
    LeftBumper, RightBumper,
    LeftTrigger, RightTrigger,
    Select, Start,
    LeftStick, RightStick,
    DpadUp, DpadDown, DpadLeft, DpadRight,
}

/// Gamepad axes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamepadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

/// Input manager for handling all input events
pub struct InputManager {
    pressed_keys: HashSet<KeyCode>,
    pressed_mouse_buttons: HashSet<MouseButton>,
    mouse_position: Point2<f32>,
    mouse_delta: Vector2<f32>,
    gamepad_buttons: HashSet<GamepadButton>,
    gamepad_axes: HashMap<GamepadAxis, f32>,
    events: Vec<InputEvent>,
}

impl InputManager {
    /// Create a new input manager
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            pressed_mouse_buttons: HashSet::new(),
            mouse_position: Point2::new(0.0, 0.0),
            mouse_delta: Vector2::new(0.0, 0.0),
            gamepad_buttons: HashSet::new(),
            gamepad_axes: HashMap::new(),
            events: Vec::new(),
        }
    }
    
    /// Process an input event
    pub fn process_event(&mut self, event: InputEvent) {
        match event.clone() {
            InputEvent::KeyPressed(key) => {
                self.pressed_keys.insert(key);
            }
            InputEvent::KeyReleased(key) => {
                self.pressed_keys.remove(&key);
            }
            InputEvent::MouseButtonPressed(button) => {
                self.pressed_mouse_buttons.insert(button);
            }
            InputEvent::MouseButtonReleased(button) => {
                self.pressed_mouse_buttons.remove(&button);
            }
            InputEvent::MouseMoved(x, y) => {
                let new_pos = Point2::new(x, y);
                self.mouse_delta = new_pos - self.mouse_position;
                self.mouse_position = new_pos;
            }
            InputEvent::GamepadButtonPressed(button) => {
                self.gamepad_buttons.insert(button);
            }
            InputEvent::GamepadButtonReleased(button) => {
                self.gamepad_buttons.remove(&button);
            }
            InputEvent::GamepadAxisMoved(axis, value) => {
                self.gamepad_axes.insert(axis, value);
            }
            _ => {}
        }
        
        self.events.push(event);
    }
    
    /// Poll and return all pending events
    pub fn poll_events(&mut self) -> Vec<InputEvent> {
        let events = self.events.clone();
        self.events.clear();
        self.mouse_delta = Vector2::new(0.0, 0.0);
        events
    }
    
    /// Check if a key is currently pressed
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }
    
    /// Check if a mouse button is currently pressed
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_mouse_buttons.contains(&button)
    }
    
    /// Get current mouse position
    pub fn mouse_position(&self) -> Point2<f32> {
        self.mouse_position
    }
    
    /// Get mouse movement delta
    pub fn mouse_delta(&self) -> Vector2<f32> {
        self.mouse_delta
    }
    
    /// Check if a gamepad button is pressed
    pub fn is_gamepad_button_pressed(&self, button: GamepadButton) -> bool {
        self.gamepad_buttons.contains(&button)
    }
    
    /// Get gamepad axis value
    pub fn gamepad_axis(&self, axis: GamepadAxis) -> f32 {
        self.gamepad_axes.get(&axis).copied().unwrap_or(0.0)
    }
    
    /// Clear all input state
    pub fn clear(&mut self) {
        self.pressed_keys.clear();
        self.pressed_mouse_buttons.clear();
        self.gamepad_buttons.clear();
        self.events.clear();
        self.mouse_delta = Vector2::new(0.0, 0.0);
    }
}

/// Input action mapping for gameplay
pub struct InputAction {
    pub name: String,
    pub keys: Vec<KeyCode>,
    pub mouse_buttons: Vec<MouseButton>,
    pub gamepad_buttons: Vec<GamepadButton>,
}

impl InputAction {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            keys: Vec::new(),
            mouse_buttons: Vec::new(),
            gamepad_buttons: Vec::new(),
        }
    }
    
    pub fn with_key(mut self, key: KeyCode) -> Self {
        self.keys.push(key);
        self
    }
    
    pub fn with_mouse_button(mut self, button: MouseButton) -> Self {
        self.mouse_buttons.push(button);
        self
    }
    
    pub fn with_gamepad_button(mut self, button: GamepadButton) -> Self {
        self.gamepad_buttons.push(button);
        self
    }
    
    pub fn is_pressed(&self, input: &InputManager) -> bool {
        for key in &self.keys {
            if input.is_key_pressed(*key) {
                return true;
            }
        }
        
        for button in &self.mouse_buttons {
            if input.is_mouse_button_pressed(*button) {
                return true;
            }
        }
        
        for button in &self.gamepad_buttons {
            if input.is_gamepad_button_pressed(*button) {
                return true;
            }
        }
        
        false
    }
}