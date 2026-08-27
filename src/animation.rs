//! Tweening and Animation Curve Engine
//!
//! Provides 18 easing curves (Linear, Quad, Cubic, Sine, Expo, Elastic, Bounce, Back),
//! property tweeners (`f32`, vectors, colors), keyframe tracks, and loop modes.

use std::f32::consts::PI;
use cgmath::{Point3, Vector3};

/// Mathematical easing curve functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EaseFunction {
    Linear,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseInSine,
    EaseOutSine,
    EaseInOutSine,
    EaseInExpo,
    EaseOutExpo,
    EaseInOutExpo,
    EaseInElastic,
    EaseOutElastic,
    EaseInOutElastic,
    EaseInBounce,
    EaseOutBounce,
    EaseInOutBounce,
    EaseInBack,
    EaseOutBack,
    EaseInOutBack,
}

impl EaseFunction {
    /// Evaluate the easing function for parameter t in [0.0, 1.0]
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EaseFunction::Linear => t,
            EaseFunction::EaseInQuad => t * t,
            EaseFunction::EaseOutQuad => t * (2.0 - t),
            EaseFunction::EaseInOutQuad => {
                if t < 0.5 { 2.0 * t * t } else { -1.0 + (4.0 - 2.0 * t) * t }
            }
            EaseFunction::EaseInCubic => t * t * t,
            EaseFunction::EaseOutCubic => {
                let f = t - 1.0;
                f * f * f + 1.0
            }
            EaseFunction::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let f = 2.0 * t - 2.0;
                    0.5 * f * f * f + 1.0
                }
            }
            EaseFunction::EaseInSine => 1.0 - ((t * PI / 2.0).cos()),
            EaseFunction::EaseOutSine => (t * PI / 2.0).sin(),
            EaseFunction::EaseInOutSine => 0.5 * (1.0 - (PI * t).cos()),
            EaseFunction::EaseInExpo => {
                if t == 0.0 { 0.0 } else { (10.0 * (t - 1.0)).exp2() }
            }
            EaseFunction::EaseOutExpo => {
                if t == 1.0 { 1.0 } else { 1.0 - (-10.0 * t).exp2() }
            }
            EaseFunction::EaseInOutExpo => {
                if t == 0.0 { 0.0 }
                else if t == 1.0 { 1.0 }
                else if t < 0.5 { 0.5 * (20.0 * t - 10.0).exp2() }
                else { 0.5 * (2.0 - (-20.0 * t + 10.0).exp2()) }
            }
            EaseFunction::EaseInElastic => {
                if t == 0.0 { 0.0 }
                else if t == 1.0 { 1.0 }
                else {
                    let c4 = (2.0 * PI) / 3.0;
                    -((10.0 * t - 10.0).exp2()) * ((t * 10.0 - 10.75) * c4).sin()
                }
            }
            EaseFunction::EaseOutElastic => {
                if t == 0.0 { 0.0 }
                else if t == 1.0 { 1.0 }
                else {
                    let c4 = (2.0 * PI) / 3.0;
                    (10.0 * -t).exp2() * ((t * 10.0 - 0.75) * c4).sin() + 1.0
                }
            }
            EaseFunction::EaseInOutElastic => {
                if t == 0.0 { 0.0 }
                else if t == 1.0 { 1.0 }
                else {
                    let c5 = (2.0 * PI) / 4.5;
                    if t < 0.5 {
                        -0.5 * ((20.0 * t - 10.0).exp2()) * ((20.0 * t - 11.125) * c5).sin()
                    } else {
                        0.5 * ((-20.0 * t + 10.0).exp2()) * ((20.0 * t - 11.125) * c5).sin() + 1.0
                    }
                }
            }
            EaseFunction::EaseOutBounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t2 = t - 1.5 / d1;
                    n1 * t2 * t2 + 0.75
                } else if t < 2.5 / d1 {
                    let t2 = t - 2.25 / d1;
                    n1 * t2 * t2 + 0.9375
                } else {
                    let t2 = t - 2.625 / d1;
                    n1 * t2 * t2 + 0.984375
                }
            }
            EaseFunction::EaseInBounce => 1.0 - EaseFunction::EaseOutBounce.evaluate(1.0 - t),
            EaseFunction::EaseInOutBounce => {
                if t < 0.5 {
                    0.5 * EaseFunction::EaseInBounce.evaluate(t * 2.0)
                } else {
                    0.5 * EaseFunction::EaseOutBounce.evaluate(t * 2.0 - 1.0) + 0.5
                }
            }
            EaseFunction::EaseInBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                c3 * t * t * t - c1 * t * t
            }
            EaseFunction::EaseOutBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                let t1 = t - 1.0;
                1.0 + c3 * t1 * t1 * t1 + c1 * t1 * t1
            }
            EaseFunction::EaseInOutBack => {
                let c1 = 1.70158;
                let c2 = c1 * 1.525;
                if t < 0.5 {
                    0.5 * ((2.0 * t).powi(2) * ((c2 + 1.0) * 2.0 * t - c2))
                } else {
                    0.5 * ((2.0 * t - 2.0).powi(2) * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0)
                }
            }
        }
    }
}

/// Linear interpolation trait for animatable types
pub trait Interpolate: Clone {
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Interpolate for f32 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Interpolate for [f32; 3] {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0] + (other[0] - self[0]) * t,
            self[1] + (other[1] - self[1]) * t,
            self[2] + (other[2] - self[2]) * t,
        ]
    }
}

impl Interpolate for [f32; 4] {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0] + (other[0] - self[0]) * t,
            self[1] + (other[1] - self[1]) * t,
            self[2] + (other[2] - self[2]) * t,
            self[3] + (other[3] - self[3]) * t,
        ]
    }
}

impl Interpolate for Vector3<f32> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Interpolate for Point3<f32> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Point3::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }
}

/// Loop behavior of an animation tween
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    Once,
    Loop,
    PingPong,
}

/// Property tween interpolator
#[derive(Debug, Clone)]
pub struct Tween<T: Interpolate> {
    pub start: T,
    pub end: T,
    pub current: T,
    pub duration: f32,
    pub elapsed: f32,
    pub delay: f32,
    pub ease: EaseFunction,
    pub loop_mode: LoopMode,
    pub is_finished: bool,
    is_reverse: bool,
}

impl<T: Interpolate> Tween<T> {
    pub fn new(start: T, end: T, duration: f32, ease: EaseFunction) -> Self {
        Self {
            current: start.clone(),
            start,
            end,
            duration: duration.max(0.001),
            elapsed: 0.0,
            delay: 0.0,
            ease,
            loop_mode: LoopMode::Once,
            is_finished: false,
            is_reverse: false,
        }
    }

    pub fn with_loop(mut self, loop_mode: LoopMode) -> Self {
        self.loop_mode = loop_mode;
        self
    }

    pub fn with_delay(mut self, delay: f32) -> Self {
        self.delay = delay;
        self
    }

    /// Step the tween forward by dt seconds
    pub fn update(&mut self, dt: f32) -> &T {
        if self.is_finished {
            return &self.current;
        }

        if self.delay > 0.0 {
            self.delay -= dt;
            return &self.current;
        }

        self.elapsed += dt;

        if self.elapsed >= self.duration {
            match self.loop_mode {
                LoopMode::Once => {
                    self.elapsed = self.duration;
                    self.is_finished = true;
                    self.current = self.end.clone();
                }
                LoopMode::Loop => {
                    self.elapsed %= self.duration;
                }
                LoopMode::PingPong => {
                    self.elapsed %= self.duration;
                    self.is_reverse = !self.is_reverse;
                }
            }
        }

        let raw_t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        let t = if self.is_reverse { 1.0 - raw_t } else { raw_t };
        let eased_t = self.ease.evaluate(t);

        self.current = self.start.lerp(&self.end, eased_t);
        &self.current
    }
}

/// A keyframe at a specific point in time
#[derive(Debug, Clone)]
pub struct Keyframe<T: Interpolate> {
    pub time: f32,
    pub value: T,
    pub ease: EaseFunction,
}

/// Timeline keyframe sequence track
#[derive(Debug, Clone)]
pub struct KeyframeTrack<T: Interpolate> {
    pub keyframes: Vec<Keyframe<T>>,
    pub duration: f32,
    pub elapsed: f32,
    pub looping: bool,
}

impl<T: Interpolate> KeyframeTrack<T> {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            duration: 0.0,
            elapsed: 0.0,
            looping: false,
        }
    }

    pub fn add_keyframe(&mut self, time: f32, value: T, ease: EaseFunction) {
        if time > self.duration {
            self.duration = time;
        }
        self.keyframes.push(Keyframe { time, value, ease });
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Sample the track value at time t
    pub fn sample(&self, t: f32) -> Option<T> {
        if self.keyframes.is_empty() { return None; }
        if self.keyframes.len() == 1 || t <= self.keyframes[0].time {
            return Some(self.keyframes[0].value.clone());
        }

        let total_time = if self.looping && self.duration > 0.0 {
            t % self.duration
        } else {
            t.min(self.duration)
        };

        for i in 0..(self.keyframes.len() - 1) {
            let k0 = &self.keyframes[i];
            let k1 = &self.keyframes[i + 1];

            if total_time >= k0.time && total_time <= k1.time {
                let seg_duration = k1.time - k0.time;
                let local_t = if seg_duration > 0.0 { (total_time - k0.time) / seg_duration } else { 0.0 };
                let eased_t = k0.ease.evaluate(local_t);
                return Some(k0.value.lerp(&k1.value, eased_t));
            }
        }

        self.keyframes.last().map(|k| k.value.clone())
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_easing_boundaries() {
        let easings = [
            EaseFunction::Linear,
            EaseFunction::EaseInQuad,
            EaseFunction::EaseOutQuad,
            EaseFunction::EaseInOutQuad,
            EaseFunction::EaseInSine,
            EaseFunction::EaseOutSine,
            EaseFunction::EaseInOutSine,
            EaseFunction::EaseInBounce,
            EaseFunction::EaseOutBounce,
            EaseFunction::EaseInBack,
            EaseFunction::EaseOutBack,
        ];

        for ease in easings {
            assert!((ease.evaluate(0.0) - 0.0).abs() < 1e-4, "{:?} at 0 failed", ease);
            assert!((ease.evaluate(1.0) - 1.0).abs() < 1e-4, "{:?} at 1 failed", ease);
        }
    }

    #[test]
    fn test_tween_interpolation() {
        let mut tween = Tween::new(0.0_f32, 100.0_f32, 1.0, EaseFunction::Linear);
        tween.update(0.5);
        assert!((tween.current - 50.0).abs() < 1e-4);

        tween.update(0.5);
        assert!((tween.current - 100.0).abs() < 1e-4);
        assert!(tween.is_finished);
    }

    #[test]
    fn test_keyframe_track() {
        let mut track = KeyframeTrack::new();
        track.add_keyframe(0.0, 0.0_f32, EaseFunction::Linear);
        track.add_keyframe(1.0, 10.0_f32, EaseFunction::Linear);
        track.add_keyframe(2.0, 30.0_f32, EaseFunction::Linear);

        assert!((track.sample(0.5).unwrap() - 5.0).abs() < 1e-4);
        assert!((track.sample(1.5).unwrap() - 20.0).abs() < 1e-4);
    }
}
