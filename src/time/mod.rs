//! Time management for the non-Euclidean engine

use std::time::{Duration, Instant};

/// Time tracking for the engine
#[derive(Debug, Clone)]
pub struct Time {
    start_time: Instant,
    last_frame_time: Instant,
    current_time: Instant,
    delta_time: f32,
    total_time: f32,
    frame_count: u64,
    fps: f32,
    fps_update_time: Instant,
    fps_frame_count: u32,
}

impl Time {
    /// Create a new time tracker
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_frame_time: now,
            current_time: now,
            delta_time: 0.0,
            total_time: 0.0,
            frame_count: 0,
            fps: 0.0,
            fps_update_time: now,
            fps_frame_count: 0,
        }
    }
    
    /// Update time tracking
    pub fn update(&mut self, dt: f32) {
        self.current_time = Instant::now();
        self.delta_time = dt;
        self.total_time += dt;
        self.frame_count += 1;
        self.fps_frame_count += 1;
        
        // Update FPS every second
        let fps_elapsed = self.current_time.duration_since(self.fps_update_time);
        if fps_elapsed >= Duration::from_secs(1) {
            self.fps = self.fps_frame_count as f32 / fps_elapsed.as_secs_f32();
            self.fps_update_time = self.current_time;
            self.fps_frame_count = 0;
        }
        
        self.last_frame_time = self.current_time;
    }
    
    /// Get delta time in seconds
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }
    
    /// Get total elapsed time in seconds
    pub fn total_time(&self) -> f32 {
        self.total_time
    }
    
    /// Get current FPS
    pub fn fps(&self) -> f32 {
        self.fps
    }
    
    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
    
    /// Get time since engine start
    pub fn elapsed(&self) -> Duration {
        self.current_time.duration_since(self.start_time)
    }
}

/// Timer for measuring intervals
#[derive(Debug, Clone)]
pub struct Timer {
    start_time: Instant,
    duration: Duration,
    paused: bool,
    pause_time: Option<Instant>,
    accumulated_pause: Duration,
}

impl Timer {
    /// Create a new timer with specified duration
    pub fn new(duration: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            duration,
            paused: false,
            pause_time: None,
            accumulated_pause: Duration::ZERO,
        }
    }
    
    /// Create a timer from seconds
    pub fn from_seconds(seconds: f32) -> Self {
        Self::new(Duration::from_secs_f32(seconds))
    }
    
    /// Reset the timer
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.paused = false;
        self.pause_time = None;
        self.accumulated_pause = Duration::ZERO;
    }
    
    /// Pause the timer
    pub fn pause(&mut self) {
        if !self.paused {
            self.paused = true;
            self.pause_time = Some(Instant::now());
        }
    }
    
    /// Resume the timer
    pub fn resume(&mut self) {
        if self.paused {
            if let Some(pause_time) = self.pause_time {
                self.accumulated_pause += Instant::now().duration_since(pause_time);
            }
            self.paused = false;
            self.pause_time = None;
        }
    }
    
    /// Check if timer has finished
    pub fn finished(&self) -> bool {
        self.elapsed() >= self.duration
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        let total_elapsed = if self.paused {
            self.pause_time.unwrap_or(Instant::now()).duration_since(self.start_time)
        } else {
            Instant::now().duration_since(self.start_time)
        };
        
        total_elapsed.saturating_sub(self.accumulated_pause)
    }
    
    /// Get remaining time
    pub fn remaining(&self) -> Duration {
        self.duration.saturating_sub(self.elapsed())
    }
    
    /// Get progress as a value between 0.0 and 1.0
    pub fn progress(&self) -> f32 {
        let elapsed = self.elapsed().as_secs_f32();
        let duration = self.duration.as_secs_f32();
        if duration > 0.0 {
            (elapsed / duration).min(1.0)
        } else {
            1.0
        }
    }
}

/// Fixed timestep accumulator for physics
pub struct FixedTimestep {
    accumulator: f32,
    fixed_dt: f32,
    max_steps: u32,
}

impl FixedTimestep {
    /// Create a new fixed timestep with target rate (e.g., 60 Hz)
    pub fn new(rate: f32) -> Self {
        Self {
            accumulator: 0.0,
            fixed_dt: 1.0 / rate,
            max_steps: 10, // Prevent spiral of death
        }
    }
    
    /// Update and return number of fixed steps to perform
    pub fn update(&mut self, dt: f32) -> u32 {
        self.accumulator += dt;
        
        let mut steps = 0;
        while self.accumulator >= self.fixed_dt && steps < self.max_steps {
            self.accumulator -= self.fixed_dt;
            steps += 1;
        }
        
        // Clamp accumulator to prevent spiral of death
        if steps >= self.max_steps {
            self.accumulator = 0.0;
        }
        
        steps
    }
    
    /// Get the fixed timestep value
    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }
    
    /// Get interpolation alpha for rendering
    pub fn alpha(&self) -> f32 {
        self.accumulator / self.fixed_dt
    }
}