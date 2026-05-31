//! Scoring system module
//!
//! Provides a reusable `ScoreTracker` with combo mechanics, score events,
//! and a GPU-uploadable `HudData` struct for rendering in-game HUD overlays.

/// Events emitted by the scoring system
#[derive(Debug, Clone)]
pub enum ScoreEvent {
    /// Points were awarded (base amount, multiplied amount, current combo)
    PointsAwarded { base: u32, multiplied: u32, combo: u32 },
    /// A new combo streak started
    ComboStarted,
    /// Combo streak was dropped (timer expired)
    ComboDropped,
    /// A new high score was reached
    HighScoreBeaten(u32),
}

/// Tracks score, combo streaks, and high score across a game session.
///
/// # Example
/// ```
/// use metatopia_engine::scoring::ScoreTracker;
///
/// let mut tracker = ScoreTracker::new();
/// let events = tracker.add_points(100);
/// assert_eq!(tracker.score(), 100);
/// tracker.tick(0.016); // advance combo timer
/// ```
pub struct ScoreTracker {
    score: u32,
    high_score: u32,
    combo_count: u32,
    combo_timer: f32,
    combo_window: f32,
    combo_multiplier_cap: u32,
}

impl Default for ScoreTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ScoreTracker {
    /// Create a new score tracker with default settings.
    /// Combo window is 2.0 seconds, max multiplier is 5x.
    pub fn new() -> Self {
        Self {
            score: 0,
            high_score: 0,
            combo_count: 0,
            combo_timer: 0.0,
            combo_window: 2.0,
            combo_multiplier_cap: 5,
        }
    }

    /// Create a score tracker with custom combo settings.
    pub fn with_combo(combo_window_secs: f32, max_multiplier: u32) -> Self {
        Self {
            combo_window: combo_window_secs,
            combo_multiplier_cap: max_multiplier,
            ..Self::new()
        }
    }

    /// Add points, applying the current combo multiplier.
    /// Returns a list of score events that occurred.
    pub fn add_points(&mut self, base: u32) -> Vec<ScoreEvent> {
        let mut events = Vec::new();

        // Update combo
        if self.combo_timer > 0.0 {
            self.combo_count += 1;
        } else {
            if self.combo_count > 0 {
                // Had a combo that expired, now starting fresh
            }
            self.combo_count = 1;
            events.push(ScoreEvent::ComboStarted);
        }
        self.combo_timer = self.combo_window;

        // Calculate multiplied score
        let multiplier = (self.combo_count).min(self.combo_multiplier_cap);
        let multiplied = base * multiplier;
        self.score += multiplied;

        events.push(ScoreEvent::PointsAwarded {
            base,
            multiplied,
            combo: self.combo_count,
        });

        // Check high score
        if self.score > self.high_score {
            self.high_score = self.score;
            events.push(ScoreEvent::HighScoreBeaten(self.high_score));
        }

        events
    }

    /// Advance the combo timer. Call once per frame with delta time.
    /// Returns `Some(ScoreEvent::ComboDropped)` if the combo just expired.
    pub fn tick(&mut self, dt: f32) -> Option<ScoreEvent> {
        if self.combo_timer > 0.0 {
            self.combo_timer = (self.combo_timer - dt).max(0.0);
            if self.combo_timer <= 0.0 && self.combo_count > 1 {
                let old_combo = self.combo_count;
                self.combo_count = 0;
                let _ = old_combo; // combo info available if needed
                return Some(ScoreEvent::ComboDropped);
            }
        }
        None
    }

    /// Current score
    pub fn score(&self) -> u32 { self.score }

    /// Session high score
    pub fn high_score(&self) -> u32 { self.high_score }

    /// Current combo count
    pub fn combo_count(&self) -> u32 { self.combo_count }

    /// Remaining combo timer
    pub fn combo_timer(&self) -> f32 { self.combo_timer }

    /// Current effective multiplier
    pub fn multiplier(&self) -> u32 {
        if self.combo_timer > 0.0 {
            self.combo_count.min(self.combo_multiplier_cap)
        } else {
            1
        }
    }

    /// Reset score (preserves high_score)
    pub fn reset(&mut self) {
        self.score = 0;
        self.combo_count = 0;
        self.combo_timer = 0.0;
    }

    /// Set score directly (for migration from hand-rolled scoring)
    pub fn set_score(&mut self, score: u32) {
        self.score = score;
        if self.score > self.high_score {
            self.high_score = self.score;
        }
    }

    /// Set combo state directly (for migration from hand-rolled scoring)
    pub fn set_combo(&mut self, count: u32, timer: f32) {
        self.combo_count = count;
        self.combo_timer = timer;
    }
}

/// GPU-uploadable HUD data for score overlay rendering.
///
/// Pack this into a uniform buffer and pass to shaders.
/// Layout: two `vec4<f32>` (32 bytes total).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HudData {
    /// x=score, y=high_score, z=combo_count, w=combo_timer
    pub score_info: [f32; 4],
    /// x=level, y=time_remaining, z=collected, w=total
    pub game_info: [f32; 4],
}

impl Default for HudData {
    fn default() -> Self {
        Self {
            score_info: [0.0; 4],
            game_info: [0.0; 4],
        }
    }
}

impl HudData {
    /// Create HUD data from a ScoreTracker and optional game state.
    pub fn from_tracker(tracker: &ScoreTracker, level: u32, time_remaining: f32) -> Self {
        Self {
            score_info: [
                tracker.score() as f32,
                tracker.high_score() as f32,
                tracker.combo_count() as f32,
                tracker.combo_timer(),
            ],
            game_info: [level as f32, time_remaining, 0.0, 0.0],
        }
    }

    /// Create HUD data for a collection-style game (like basic_game).
    pub fn from_collection(collected: u32, total: u32) -> Self {
        Self {
            score_info: [0.0; 4],
            game_info: [0.0, 0.0, collected as f32, total as f32],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_scoring() {
        let mut tracker = ScoreTracker::new();
        tracker.add_points(100);
        assert_eq!(tracker.score(), 100);
        assert_eq!(tracker.combo_count(), 1);
    }

    #[test]
    fn combo_multiplier() {
        let mut tracker = ScoreTracker::new();
        tracker.add_points(10); // 1x = 10
        tracker.add_points(10); // 2x = 20
        tracker.add_points(10); // 3x = 30
        assert_eq!(tracker.score(), 60); // 10 + 20 + 30
        assert_eq!(tracker.combo_count(), 3);
    }

    #[test]
    fn combo_cap() {
        let mut tracker = ScoreTracker::with_combo(2.0, 3);
        for _ in 0..5 {
            tracker.add_points(10);
        }
        // 10 + 20 + 30 + 30 + 30 = 120 (capped at 3x)
        assert_eq!(tracker.score(), 120);
    }

    #[test]
    fn combo_decay() {
        let mut tracker = ScoreTracker::new();
        tracker.add_points(10); // combo_count = 1
        tracker.add_points(10); // combo_count = 2
        assert_eq!(tracker.combo_timer(), 2.0);
        tracker.tick(1.0);
        assert_eq!(tracker.combo_timer(), 1.0);
        let event = tracker.tick(1.5);
        assert!(matches!(event, Some(ScoreEvent::ComboDropped)));
    }

    #[test]
    fn high_score_tracking() {
        let mut tracker = ScoreTracker::new();
        tracker.add_points(100);
        assert_eq!(tracker.high_score(), 100);
        tracker.reset();
        assert_eq!(tracker.score(), 0);
        assert_eq!(tracker.high_score(), 100);
    }

    #[test]
    fn hud_data_pod() {
        let hud = HudData::default();
        let _bytes: &[u8] = bytemuck::bytes_of(&hud);
        assert_eq!(std::mem::size_of::<HudData>(), 32);
    }
}
