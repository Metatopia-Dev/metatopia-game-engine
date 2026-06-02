//! Audio engine wrapping rodio for easy sound playback.
//!
//! Provides fire-and-forget sound effects, looping background music,
//! volume control, and sound preloading.
//!
//! # Example
//! ```no_run
//! use metatopia_engine::audio::AudioEngine;
//!
//! let mut audio = AudioEngine::new();
//! audio.play_sfx("sounds/hit.wav");
//! audio.play_music("music/theme.mp3");
//! audio.set_music_volume(0.5);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::io::BufReader;

/// Handle to a playing sound effect (for stop/volume control).
pub struct SfxHandle {
    sink: rodio::Sink,
}

impl SfxHandle {
    /// Stop this sound effect.
    pub fn stop(&self) { self.sink.stop(); }
    /// Set volume (0.0 = silent, 1.0 = normal).
    pub fn set_volume(&self, vol: f32) { self.sink.set_volume(vol); }
    /// Check if still playing.
    pub fn is_playing(&self) -> bool { !self.sink.empty() }
}

/// Main audio engine.
pub struct AudioEngine {
    #[allow(dead_code)] // Must keep alive — dropping OutputStream kills audio
    stream: Option<rodio::OutputStream>,
    stream_handle: Option<rodio::OutputStreamHandle>,
    music_sink: Option<rodio::Sink>,
    music_volume: f32,
    sfx_volume: f32,
    preloaded: HashMap<String, PathBuf>,
    initialized: bool,
}

impl Default for AudioEngine {
    fn default() -> Self { Self::new() }
}

impl AudioEngine {
    /// Create a new audio engine. Initializes the audio output device.
    pub fn new() -> Self {
        match rodio::OutputStream::try_default() {
            Ok((stream, handle)) => {
                Self {
                    stream: Some(stream),
                    stream_handle: Some(handle),
                    music_sink: None,
                    music_volume: 0.7,
                    sfx_volume: 1.0,
                    preloaded: HashMap::new(),
                    initialized: true,
                }
            }
            Err(e) => {
                eprintln!("⚠️ Audio init failed: {e}. Running without sound.");
                Self {
                    stream: None,
                    stream_handle: None,
                    music_sink: None,
                    music_volume: 0.7,
                    sfx_volume: 1.0,
                    preloaded: HashMap::new(),
                    initialized: false,
                }
            }
        }
    }

    /// Whether audio is available.
    pub fn is_available(&self) -> bool { self.initialized }

    /// Preload a sound for instant playback later.
    /// ```no_run
    /// # use metatopia_engine::audio::AudioEngine;
    /// # let mut audio = AudioEngine::new();
    /// audio.preload("hit", "sounds/hit.wav");
    /// audio.play("hit"); // instant playback
    /// ```
    pub fn preload(&mut self, name: &str, path: &str) {
        self.preloaded.insert(name.to_string(), PathBuf::from(path));
    }

    /// Play a preloaded sound by name. Returns None if not preloaded or audio unavailable.
    pub fn play(&self, name: &str) -> Option<SfxHandle> {
        let path = self.preloaded.get(name)?;
        self.play_sfx_path(path)
    }

    /// Play a one-shot sound effect from a file path.
    pub fn play_sfx(&self, path: &str) -> Option<SfxHandle> {
        self.play_sfx_path(Path::new(path))
    }

    fn play_sfx_path(&self, path: &Path) -> Option<SfxHandle> {
        let handle = self.stream_handle.as_ref()?;
        let file = std::fs::File::open(path).ok()?;
        let source = rodio::Decoder::new(BufReader::new(file)).ok()?;
        let sink = rodio::Sink::try_new(handle).ok()?;
        sink.set_volume(self.sfx_volume);
        sink.append(source);
        Some(SfxHandle { sink })
    }

    /// Play looping background music. Stops any currently playing music.
    pub fn play_music(&mut self, path: &str) {
        self.stop_music();
        let handle = match self.stream_handle.as_ref() {
            Some(h) => h,
            None => return,
        };
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => { eprintln!("⚠️ Music load failed: {e}"); return; }
        };
        let source = match rodio::Decoder::new(BufReader::new(file)) {
            Ok(s) => s,
            Err(e) => { eprintln!("⚠️ Music decode failed: {e}"); return; }
        };
        let sink = match rodio::Sink::try_new(handle) {
            Ok(s) => s,
            Err(_) => return,
        };
        sink.set_volume(self.music_volume);
        sink.append(rodio::source::Source::repeat_infinite(source));
        self.music_sink = Some(sink);
    }

    /// Stop background music.
    pub fn stop_music(&mut self) {
        if let Some(sink) = self.music_sink.take() {
            sink.stop();
        }
    }

    /// Pause background music.
    pub fn pause_music(&self) {
        if let Some(ref sink) = self.music_sink {
            sink.pause();
        }
    }

    /// Resume background music.
    pub fn resume_music(&self) {
        if let Some(ref sink) = self.music_sink {
            sink.play();
        }
    }

    /// Set music volume (0.0–1.0).
    pub fn set_music_volume(&mut self, vol: f32) {
        self.music_volume = vol.clamp(0.0, 1.0);
        if let Some(ref sink) = self.music_sink {
            sink.set_volume(self.music_volume);
        }
    }

    /// Set sound effects volume (0.0–1.0).
    pub fn set_sfx_volume(&mut self, vol: f32) {
        self.sfx_volume = vol.clamp(0.0, 1.0);
    }

    /// Get current music volume.
    pub fn music_volume(&self) -> f32 { self.music_volume }

    /// Get current SFX volume.
    pub fn sfx_volume(&self) -> f32 { self.sfx_volume }

    /// Check if music is currently playing.
    pub fn is_music_playing(&self) -> bool {
        self.music_sink.as_ref().map(|s| !s.empty() && !s.is_paused()).unwrap_or(false)
    }

    /// Stop all audio.
    pub fn stop_all(&mut self) {
        self.stop_music();
    }
}
