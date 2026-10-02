//! Metatopia Engine — GPU to Integrated Silicon Photonics (SiPh) Simulator
//!
//! An interactive 3D in-engine physics and architectural simulator modeling the
//! transition from legacy electronic GPU interconnects (copper SerDes, NVLink, PCIe Gen6)
//! to Integrated Silicon Photonics (SiPh), Co-Packaged Optics (CPO), and Photonic Tensor Cores.
//!
//! Interactive Controls:
//! - [1] Electronic GPU Baseline (Copper SerDes, HBM3 interposer, 14.5 pJ/bit)
//! - [2] Co-Packaged Optics (CPO) Hybrid (Electronic GPU + Silicon Photonic I/O Chiplets, 1.8 pJ/bit)
//! - [3] Integrated Photonic AI Accelerator (Coherent Photonic Tensor Core + Light-Speed MZI GEMM)
//! - [L] / [Shift+L] : Laser Power -/+ (mW)
//! - [C] / [Shift+C] : WDM Wavelength Channels -/+ (1 to 64 lambda)
//! - [B] / [Shift+B] : Baud Rate -/+ (Gbps)
//! - [T] / [Shift+T] : Die Temperature -/+ (°C)
//! - [H]             : Toggle Closed-Loop Thermal Auto-Lock (demonstrates thermo-optic ring detuning)
//! - [M] / [Shift+M] : GEMM Workload Dimension N (512 to 8192)
//! - [P] / [Shift+P] : Photonic Cores -/+ (1 to 32 tiles)
//! - [Space]         : Toggle Dynamic Thermal Drift Perturbation
//! - [R]             : Reset Simulator to defaults
//! - Mouse Left Click: Clickable onscreen buttons and steppers
//! - Mouse Right Drag: Free Orbit & Fly Camera
//! - WASD            : Move Camera

use metatopia_engine::quickstart::*;
use metatopia_engine::geometry::ProceduralMesh;
use metatopia_engine::particles::ParticleSystem;
use metatopia_engine::photonics::{PhotonicsSimulator, PhotonicsConfig, ArchitectureMode};
use cgmath::{Vector3, InnerSpace};

const SHADER_SRC: &str = include_str!("../shaders/editor.wgsl");

/// Master 3D GPU to Silicon Photonics Simulator Application
pub struct GpuPhotonicsSimApp {
    pub sim: PhotonicsSimulator,
    pub particle_system: ParticleSystem,
    pub camera_initialized: bool,
    pub pulse_timer: f32,
    pub status_message: String,
    pub status_timer: f32,
}

impl GpuPhotonicsSimApp {
    pub fn new() -> Self {
        let sim = PhotonicsSimulator::new(PhotonicsConfig::default());
        Self {
            sim,
            particle_system: ParticleSystem::new(2000),
            camera_initialized: false,
            pulse_timer: 0.0,
            status_message: "GPU to Silicon Photonics Simulator Ready. Press [1], [2], [3] to switch modes.".into(),
            status_timer: 6.0,
        }
    }

    pub fn set_status(&mut self, msg: &str) {
        self.status_message = msg.to_string();
        self.status_timer = 4.0;
        println!("[PhotonicsSim]: {}", msg);
    }

    /// Helper to convert screen pixel coords to NDC `[-1.0, 1.0]`
    fn pixel_to_ndc(&self, px: f32, py: f32, width: u32, height: u32) -> (f32, f32) {
        let ndc_x = (px / width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height as f32) * 2.0;
        (ndc_x, ndc_y)
    }
}

// ─── 5x7 Vector/Bitmap Font Glyph Table ─────────────────────────────────────

fn get_glyph_bits(c: char) -> [u8; 7] {
    let uc = c.to_ascii_uppercase();
    match uc {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00110, 0b01000, 0b10000, 0b10000, 0b10000, 0b01000, 0b00110],
        ')' => [0b01100, 0b00010, 0b00001, 0b00001, 0b00001, 0b00010, 0b01100],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b01011, 0b10011, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        '=' => [0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        _   => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}

fn push_ui_rect(verts: &mut Vec<GameVertex>, indices: &mut Vec<u32>, min_x: f32, min_y: f32, max_x: f32, max_y: f32, color: [f32; 3], opacity: f32) {
    let base_idx = verts.len() as u32;
    let pbr = [0.0, 0.0, opacity, -1.0]; // w = -1.0 flags 2D screen UI overlay
    let norm = [0.0, 0.0, 1.0];

    verts.push(GameVertex::new([min_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, min_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([max_x, max_y, 0.0], norm, color, pbr));
    verts.push(GameVertex::new([min_x, max_y, 0.0], norm, color, pbr));

    indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
}

fn push_text(verts: &mut Vec<GameVertex>, indices: &mut Vec<u32>, text: &str, start_x: f32, start_y: f32, char_w: f32, char_h: f32, color: [f32; 3]) {
    let mut cur_x = start_x;
    let pix_w = char_w / 6.0;
    let pix_h = char_h / 8.0;

    for ch in text.chars() {
        if ch == ' ' {
            cur_x += char_w;
            continue;
        }
        let bits = get_glyph_bits(ch);
        for (row, byte) in bits.iter().enumerate() {
            let py = start_y + char_h - ((row + 1) as f32 * pix_h);
            for col in 0..5 {
                if (byte & (1 << (4 - col))) != 0 {
                    let px = cur_x + (col as f32 * pix_w);
                    push_ui_rect(verts, indices, px, py, px + pix_w, py + pix_h, color, 1.0);
                }
            }
        }
        cur_x += char_w + pix_w;
    }
}

// ─── GameApp Implementation ────────────────────────────────────────────────

impl GameApp for GpuPhotonicsSimApp {
    fn title(&self) -> &str {
        "Metatopia — GPU to Integrated Silicon Photonics Simulator"
    }

    fn window_size(&self) -> (u32, u32) {
        (1440, 860)
    }

    fn shader_source(&self) -> String {
        SHADER_SRC.to_string()
    }

    fn is_dynamic_mesh(&self) -> bool {
        true
    }

    fn grab_cursor(&self) -> bool {
        false
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // 1. Initial Camera Position (Elevated isometric view of chip package)
        if !self.camera_initialized {
            ctx.camera.position = Vector3::new(0.0, 11.0, 15.0);
            ctx.camera.pitch = -0.58;
            ctx.camera.yaw = -1.57; // Look forward along -Z
            ctx.camera.move_speed = 10.0;
            self.camera_initialized = true;
        }

        // Lighting & Atmosphere
        ctx.scene.sun_direction = [-0.4, -1.0, -0.5, 0.0];
        ctx.scene.sun_color = [1.0, 0.98, 0.95, 1.0];

        // Tick simulation physics
        self.sim.update(ctx.dt);
        self.pulse_timer += ctx.dt;

        if self.status_timer > 0.0 {
            self.status_timer -= ctx.dt;
        }

        // 2. Camera Navigation
        if ctx.mouse_held(winit::event::MouseButton::Right) {
            ctx.default_camera_movement();
        } else {
            // WASD translation only
            let speed = ctx.camera.move_speed * ctx.dt;
            let fwd = ctx.camera.forward();
            let right = ctx.camera.right();
            let flat_fwd = Vector3::new(fwd.x, 0.0, fwd.z).normalize() * speed;
            let flat_right = right * speed;
            if ctx.key_held(VirtualKey::KeyW) { ctx.camera.position += flat_fwd; }
            if ctx.key_held(VirtualKey::KeyS) { ctx.camera.position -= flat_fwd; }
            if ctx.key_held(VirtualKey::KeyA) { ctx.camera.position -= flat_right; }
            if ctx.key_held(VirtualKey::KeyD) { ctx.camera.position += flat_right; }
        }

        // 3. Keyboard Shortcuts
        if ctx.key_pressed(VirtualKey::Digit1) { self.sim.set_mode(ArchitectureMode::ElectronicGpu); self.set_status("Switched to Electronic GPU Baseline (Copper SerDes)"); }
        if ctx.key_pressed(VirtualKey::Digit2) { self.sim.set_mode(ArchitectureMode::CpoHybrid); self.set_status("Switched to Co-Packaged Optics (CPO Hybrid)"); }
        if ctx.key_pressed(VirtualKey::Digit3) { self.sim.set_mode(ArchitectureMode::PhotonicAccelerator); self.set_status("Switched to Integrated Photonic AI Accelerator"); }

        let shift = ctx.key_held(VirtualKey::ShiftLeft) || ctx.key_held(VirtualKey::ShiftRight);

        // Laser Power: L
        if ctx.key_pressed(VirtualKey::KeyL) {
            let delta = if shift { 5.0 } else { -5.0 };
            self.sim.adjust_laser_power(delta);
            self.set_status(&format!("Laser Power: {:.1} mW", self.sim.config.laser_power_mw));
        }

        // WDM Channels: C
        if ctx.key_pressed(VirtualKey::KeyC) {
            self.sim.adjust_wdm_channels(shift);
            self.set_status(&format!("WDM Channels: {} wavelengths", self.sim.config.wdm_channels));
        }

        // Baud Rate: B
        if ctx.key_pressed(VirtualKey::KeyB) {
            let delta = if shift { 14.0 } else { -14.0 };
            self.sim.adjust_baud_rate(delta);
            self.set_status(&format!("Baud Rate: {:.1} Gbps", self.sim.config.baud_rate_gbps));
        }

        // Die Temperature: T
        if ctx.key_pressed(VirtualKey::KeyT) {
            let delta = if shift { 5.0 } else { -5.0 };
            self.sim.adjust_temperature(delta);
            self.set_status(&format!("Operating Temperature: {:.1} °C", self.sim.config.die_temperature_c));
        }

        // Thermal Lock: H
        if ctx.key_pressed(VirtualKey::KeyH) {
            self.sim.toggle_thermal_lock();
            let msg = if self.sim.config.auto_thermal_lock { "Auto-Thermal Lock: ON" } else { "Auto-Thermal Lock: OFF (Thermal Drift Exposed!)" };
            self.set_status(msg);
        }

        // Matrix Workload Dimension: M
        if ctx.key_pressed(VirtualKey::KeyM) {
            self.sim.adjust_matrix_dim(shift);
            self.set_status(&format!("Matrix GEMM Dim: {}x{}", self.sim.config.matrix_dim_n, self.sim.config.matrix_dim_n));
        }

        // Photonic Cores: P
        if ctx.key_pressed(VirtualKey::KeyP) {
            self.sim.adjust_photonic_cores(shift);
            self.set_status(&format!("Photonic Cores: {} tiles", self.sim.config.num_photonic_cores));
        }

        // Space: Toggle thermal perturbation drift
        if ctx.key_pressed(VirtualKey::Space) {
            self.sim.thermal_drift_enabled = !self.sim.thermal_drift_enabled;
            let msg = if self.sim.thermal_drift_enabled { "Dynamic Thermal Fluctuations: ACTIVE" } else { "Dynamic Thermal Fluctuations: PAUSED" };
            self.set_status(msg);
        }

        // Reset: R
        if ctx.key_pressed(VirtualKey::KeyR) {
            self.sim = PhotonicsSimulator::new(PhotonicsConfig::default());
            self.set_status("Reset simulator to defaults.");
        }

        // 4. Mouse Clicking on UI Panels
        if ctx.mouse_pressed(winit::event::MouseButton::Left) {
            let (m_ndc_x, m_ndc_y) = self.pixel_to_ndc(ctx.mouse_pos().0, ctx.mouse_pos().1, ctx.resolution.0, ctx.resolution.1);

            // A. Top Bar Mode Buttons
            if m_ndc_y >= 0.86 && m_ndc_y <= 0.96 {
                if m_ndc_x >= -0.52 && m_ndc_x <= -0.22 {
                    self.sim.set_mode(ArchitectureMode::ElectronicGpu);
                    self.set_status("Selected Electronic GPU Baseline");
                } else if m_ndc_x >= -0.20 && m_ndc_x <= 0.10 {
                    self.sim.set_mode(ArchitectureMode::CpoHybrid);
                    self.set_status("Selected Co-Packaged Optics (CPO Hybrid)");
                } else if m_ndc_x >= 0.12 && m_ndc_x <= 0.45 {
                    self.sim.set_mode(ArchitectureMode::PhotonicAccelerator);
                    self.set_status("Selected Photonic AI Accelerator");
                }
            }

            // B. Left Control Panel Buttons (Steppers)
            if m_ndc_x >= -0.98 && m_ndc_x <= -0.66 {
                // Laser Power +/- (Row 1: y ~ 0.58)
                if m_ndc_y >= 0.56 && m_ndc_y <= 0.62 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_laser_power(-5.0); }
                    else { self.sim.adjust_laser_power(5.0); }
                }
                // WDM Channels +/- (Row 2: y ~ 0.48)
                else if m_ndc_y >= 0.46 && m_ndc_y <= 0.52 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_wdm_channels(false); }
                    else { self.sim.adjust_wdm_channels(true); }
                }
                // Baud Rate +/- (Row 3: y ~ 0.38)
                else if m_ndc_y >= 0.36 && m_ndc_y <= 0.42 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_baud_rate(-14.0); }
                    else { self.sim.adjust_baud_rate(14.0); }
                }
                // Temperature +/- (Row 4: y ~ 0.26)
                else if m_ndc_y >= 0.24 && m_ndc_y <= 0.30 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_temperature(-5.0); }
                    else { self.sim.adjust_temperature(5.0); }
                }
                // Thermal Lock Toggle (Row 5: y ~ 0.14)
                else if m_ndc_y >= 0.12 && m_ndc_y <= 0.18 {
                    self.sim.toggle_thermal_lock();
                }
                // Photonic Cores +/- (Row 6: y ~ 0.02)
                else if m_ndc_y >= 0.00 && m_ndc_y <= 0.06 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_photonic_cores(false); }
                    else { self.sim.adjust_photonic_cores(true); }
                }
                // Matrix Size +/- (Row 7: y ~ -0.10)
                else if m_ndc_y >= -0.12 && m_ndc_y <= -0.06 {
                    if m_ndc_x <= -0.73 { self.sim.adjust_matrix_dim(false); }
                    else { self.sim.adjust_matrix_dim(true); }
                }
            }
        }

        // 5. Dynamic Particle Simulation
        let dt = ctx.dt;
        match self.sim.config.mode {
            ArchitectureMode::ElectronicGpu => {
                // Electronic Copper SerDes sparks & thermal haze rising from die
                let temp_factor = (self.sim.config.die_temperature_c - 40.0).max(0.0) / 60.0;
                let burst_count = (2.0 + temp_factor * 8.0) as usize;
                self.particle_system.burst(
                    Vector3::new(0.0, 0.4, 0.0),
                    burst_count,
                    [1.0, 0.4, 0.1, 0.8], // Orange heat sparks
                    [0.3, 0.3, 0.3, 0.0],
                    1.5,
                    0.6,
                );
            }
            ArchitectureMode::CpoHybrid | ArchitectureMode::PhotonicAccelerator => {
                // Photonic pulse packets along optical waveguides
                let is_detuned = self.sim.metrics.resonance_transmission_factor < 0.5;
                if is_detuned {
                    // Detuning radiation: red/orange light spills out from microrings
                    self.particle_system.burst(
                        Vector3::new(-3.5, 0.3, 0.0),
                        6,
                        [1.0, 0.1, 0.2, 0.9],
                        [0.5, 0.0, 0.0, 0.0],
                        2.5,
                        0.4,
                    );
                } else {
                    // Healthy optical streaming (Cyan, Violet, Neon Green)
                    let p_col = match (self.sim.config.wdm_channels) % 3 {
                        0 => [0.0, 0.9, 1.0, 0.9], // Cyan
                        1 => [0.8, 0.2, 1.0, 0.9], // Violet
                        _ => [0.1, 1.0, 0.5, 0.9], // Green
                    };
                    self.particle_system.burst(
                        Vector3::new(-4.0 + (self.pulse_timer * 12.0) % 8.0, 0.25, -2.5),
                        2,
                        p_col,
                        [0.0, 0.2, 0.4, 0.0],
                        1.0,
                        0.3,
                    );
                }
            }
        }
        self.particle_system.update(dt, None);

        // Upload scene uniform
        let is_photonic = match self.sim.config.mode {
            ArchitectureMode::ElectronicGpu => 0.0,
            ArchitectureMode::CpoHybrid => 1.0,
            ArchitectureMode::PhotonicAccelerator => 2.0,
        };
        ctx.scene.game_data = [ctx.time, is_photonic, self.sim.metrics.active_bandwidth_tbps, self.sim.config.die_temperature_c];
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(60000);
        let mut indices = Vec::with_capacity(120000);

        let mode = self.sim.config.mode;
        let metrics = &self.sim.metrics;
        let cfg = &self.sim.config;

        // ── 1. Clean Metallic Grid Floor Plane (100x100) ────────────────────
        let (plane_v, plane_i) = ProceduralMesh::plane(100.0, 100.0, 20, 20, [0.08, 0.10, 0.13]);
        let base = verts.len() as u32;
        for v in plane_v {
            verts.push(GameVertex::new(v.position, v.normal, v.color, [0.1, 0.9, 0.0, 0.0]));
        }
        for idx in plane_i { indices.push(base + idx); }

        // ── 2. Silicon Substrate / BGA Package Base ─────────────────────────
        // Main Package PCB Substrate (14.0 x 0.4 x 14.0)
        let (sub_v, sub_i) = ProceduralMesh::cylinder(7.2, 0.35, 32, [0.05, 0.22, 0.18]); // Emerald green package substrate
        let sub_base = verts.len() as u32;
        for v in sub_v {
            let pos = [v.position[0], v.position[1] + 0.18, v.position[2]];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.2, 0.7, 0.0, 0.0]));
        }
        for idx in sub_i { indices.push(sub_base + idx); }

        // Gold Ring Edge Bezel on Package
        let (bezel_v, bezel_i) = ProceduralMesh::torus(7.2, 0.12, 32, 12, [0.95, 0.80, 0.20]);
        let bz_base = verts.len() as u32;
        for v in bezel_v {
            let pos = [v.position[0], v.position[1] + 0.35, v.position[2]];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.15, 0.0, 0.0]));
        }
        for idx in bezel_i { indices.push(bz_base + idx); }

        // Silicon Interposer Tile (10.0 x 0.15 x 10.0)
        let (interposer_v, interposer_i) = ProceduralMesh::cylinder(5.2, 0.15, 32, [0.18, 0.20, 0.25]); // Silicon gray
        let int_base = verts.len() as u32;
        for v in interposer_v {
            let pos = [v.position[0], v.position[1] + 0.45, v.position[2]];
            verts.push(GameVertex::new(pos, v.normal, v.color, [0.7, 0.3, 0.0, 0.0]));
        }
        for idx in interposer_i { indices.push(int_base + idx); }

        // ── 3. Central Compute Die (GPU vs Photonic Tensor Cores) ───────────
        match mode {
            ArchitectureMode::ElectronicGpu => {
                // Large Monolithic Electronic GPU Die (6.0 x 0.3 x 6.0)
                let temp_ratio = ((cfg.die_temperature_c - 40.0) / 60.0).clamp(0.0, 1.0);
                // Color shifts from sleek cyber-cyan to fiery orange-red as temperature rises!
                let die_col = [
                    0.2 + temp_ratio * 0.75,
                    0.8 - temp_ratio * 0.55,
                    1.0 - temp_ratio * 0.85,
                ];

                let (die_v, die_i) = ProceduralMesh::cylinder(3.2, 0.3, 24, die_col);
                let d_base = verts.len() as u32;
                for v in die_v {
                    let pos = [v.position[0], v.position[1] + 0.65, v.position[2]];
                    let emissive = 0.5 + temp_ratio * 1.5;
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.2, emissive, 0.0]));
                }
                for idx in die_i { indices.push(d_base + idx); }

                // 8x8 SM Core Tiles on GPU Die
                for row in 0..6 {
                    for col in 0..6 {
                        let cx = -1.8 + col as f32 * 0.72;
                        let cz = -1.8 + row as f32 * 0.72;
                        let (sm_v, sm_i) = ProceduralMesh::cylinder(0.24, 0.08, 8, [0.85, 0.55, 0.15]); // Copper SM
                        let sm_base = verts.len() as u32;
                        for v in sm_v {
                            let pos = [cx + v.position[0], v.position[1] + 0.85, cz + v.position[2]];
                            verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.2, 0.8, 0.0]));
                        }
                        for idx in sm_i { indices.push(sm_base + idx); }
                    }
                }

                // Electrical Copper Traces radiating to package edge
                for i in 0..16 {
                    let angle = (i as f32 / 16.0) * std::f32::consts::PI * 2.0;
                    let (tx_v, tx_i) = ProceduralMesh::cylinder(0.04, 2.5, 6, [0.9, 0.6, 0.2]); // Copper SerDes lines
                    let t_base = verts.len() as u32;
                    let rad = 4.2;
                    let px = angle.cos() * rad;
                    let pz = angle.sin() * rad;
                    for v in tx_v {
                        let pos = [px + v.position[0], v.position[1] + 0.55, pz + v.position[2]];
                        verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.1, 0.5, 0.0]));
                    }
                    for idx in tx_i { indices.push(t_base + idx); }
                }
            }

            ArchitectureMode::CpoHybrid => {
                // Central GPU Compute Die (Smaller 4.0 x 4.0)
                let (die_v, die_i) = ProceduralMesh::cylinder(2.2, 0.3, 24, [0.1, 0.6, 0.9]);
                let d_base = verts.len() as u32;
                for v in die_v {
                    let pos = [v.position[0], v.position[1] + 0.65, v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.2, 1.0, 0.0]));
                }
                for idx in die_i { indices.push(d_base + idx); }

                // 4 Co-Packaged Silicon Photonic I/O Chiplets (PIC Tiles) around perimeter
                let pic_offsets = [
                    (3.6, 0.0), (-3.6, 0.0), (0.0, 3.6), (0.0, -3.6)
                ];

                for &(px, pz) in &pic_offsets {
                    // Optical Transceiver PIC Tile
                    let (pic_v, pic_i) = ProceduralMesh::cylinder(1.0, 0.25, 16, [0.15, 0.85, 0.75]); // Turquoise SiPh
                    let p_base = verts.len() as u32;
                    for v in pic_v {
                        let pos = [px + v.position[0], v.position[1] + 0.62, pz + v.position[2]];
                        verts.push(GameVertex::new(pos, v.normal, v.color, [0.4, 0.3, 1.5, 0.0]));
                    }
                    for idx in pic_i { indices.push(p_base + idx); }

                    // Optical Fiber Ribbon V-Groove Connector to edge
                    let (fib_v, fib_i) = ProceduralMesh::cylinder(0.18, 1.8, 8, [0.05, 0.45, 0.95]); // Blue glass fibers
                    let f_base = verts.len() as u32;
                    let dir_x = px.signum();
                    let dir_z = pz.signum();
                    for v in fib_v {
                        let pos = [
                            px + dir_x * 1.5 + v.position[0],
                            v.position[1] + 0.60,
                            pz + dir_z * 1.5 + v.position[2],
                        ];
                        verts.push(GameVertex::new(pos, v.normal, v.color, [0.1, 0.1, 2.5, 0.0]));
                    }
                    for idx in fib_i { indices.push(f_base + idx); }
                }
            }

            ArchitectureMode::PhotonicAccelerator => {
                // 8 Coherent Photonic Tensor Core Tiles with visible MZI optical meshes
                let core_count = cfg.num_photonic_cores.min(16);
                for i in 0..core_count {
                    let angle = (i as f32 / core_count as f32) * std::f32::consts::PI * 2.0;
                    let rad = 2.6;
                    let cx = angle.cos() * rad;
                    let cz = angle.sin() * rad;

                    // MZI Mesh Core Tile
                    let (core_v, core_i) = ProceduralMesh::cylinder(0.75, 0.28, 16, [0.8, 0.2, 1.0]); // Neon Purple Photonic Tile
                    let c_base = verts.len() as u32;
                    for v in core_v {
                        let pos = [cx + v.position[0], v.position[1] + 0.65, cz + v.position[2]];
                        verts.push(GameVertex::new(pos, v.normal, v.color, [0.3, 0.2, 2.0, 0.0]));
                    }
                    for idx in core_i { indices.push(c_base + idx); }

                    // Glowing MZI Phase Shifter Lattice Node in center of tile
                    let (node_v, node_i) = ProceduralMesh::capsule(0.18, 0.01, 8, 8, [0.1, 1.0, 0.8]);
                    let n_base = verts.len() as u32;
                    for v in node_v {
                        let pos = [cx + v.position[0], v.position[1] + 0.88, cz + v.position[2]];
                        verts.push(GameVertex::new(pos, v.normal, v.color, [0.0, 0.1, 3.5, 0.0]));
                    }
                    for idx in node_i { indices.push(n_base + idx); }
                }

                // Central Optical Crossbar Routing Core
                let (hub_v, hub_i) = ProceduralMesh::cylinder(1.2, 0.35, 24, [0.0, 0.9, 1.0]);
                let h_base = verts.len() as u32;
                for v in hub_v {
                    let pos = [v.position[0], v.position[1] + 0.70, v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.2, 0.1, 2.5, 0.0]));
                }
                for idx in hub_i { indices.push(h_base + idx); }
            }
        }

        // ── 4. High-Bandwidth Memory (HBM3/HBM3e) 3D Cubes ──────────────────
        let hbm_coords = [
            (-3.8, -3.8), (3.8, -3.8), (-3.8, 3.8), (3.8, 3.8)
        ];
        for &(hx, hz) in &hbm_coords {
            // Stacked DRAM Cubes (Gold & Deep Charcoal)
            let (hbm_v, hbm_i) = ProceduralMesh::cylinder(0.7, 0.85, 12, [0.12, 0.14, 0.18]);
            let h_base = verts.len() as u32;
            for v in hbm_v {
                let pos = [hx + v.position[0], v.position[1] + 0.90, hz + v.position[2]];
                verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.3, 0.1, 0.0]));
            }
            for idx in hbm_i { indices.push(h_base + idx); }

            // Gold DRAM Sub-layer Stacking Rings
            for layer in 1..4 {
                let (ly_v, ly_i) = ProceduralMesh::torus(0.72, 0.02, 16, 6, [0.95, 0.80, 0.20]);
                let ly_base = verts.len() as u32;
                let ly_y = 0.55 + layer as f32 * 0.22;
                for v in ly_v {
                    let pos = [hx + v.position[0], v.position[1] + ly_y, hz + v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.1, 0.5, 0.0]));
                }
                for idx in ly_i { indices.push(ly_base + idx); }
            }
        }

        // ── 5. Silicon Photonic Waveguides & Microring Resonator (MRR) Banks ──
        if mode != ArchitectureMode::ElectronicGpu {
            // A. Continuous Wave (CW) Laser Source Arrays (DFB Laser bars)
            let laser_pos = [(-5.2, -1.5), (-5.2, 1.5)];
            for &(lx, lz) in &laser_pos {
                let (las_v, las_i) = ProceduralMesh::cylinder(0.35, 0.45, 10, [1.0, 0.1, 0.3]); // Red DFB laser
                let l_base = verts.len() as u32;
                for v in las_v {
                    let pos = [lx + v.position[0], v.position[1] + 0.65, lz + v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.1, 3.0, 0.0]));
                }
                for idx in las_i { indices.push(l_base + idx); }
            }

            // B. Silicon Optical Bus Waveguides (Translucent Glowing Channels)
            let bus_y = 0.58;
            let (wg1_v, wg1_i) = ProceduralMesh::cylinder(0.045, 7.5, 8, [0.0, 0.9, 1.0]); // Cyan bus
            let wg1_base = verts.len() as u32;
            for v in wg1_v {
                let pos = [v.position[1], bus_y, -2.5 + v.position[0]];
                verts.push(GameVertex::new(pos, [0.0, 1.0, 0.0], [0.0, 0.9, 1.0], [0.1, 0.1, 2.5, 0.0]));
            }
            for idx in wg1_i { indices.push(wg1_base + idx); }

            let (wg2_v, wg2_i) = ProceduralMesh::cylinder(0.045, 7.5, 8, [0.8, 0.2, 1.0]); // Purple bus
            let wg2_base = verts.len() as u32;
            for v in wg2_v {
                let pos = [v.position[1], bus_y, 2.5 + v.position[0]];
                verts.push(GameVertex::new(pos, [0.0, 1.0, 0.0], [0.8, 0.2, 1.0], [0.1, 0.1, 2.5, 0.0]));
            }
            for idx in wg2_i { indices.push(wg2_base + idx); }

            // C. Bank of Microring Resonators (MRRs) with WDM Spectral Colors!
            let ring_count = cfg.wdm_channels.min(8);
            let wdm_colors = [
                [0.0, 0.9, 1.0], // 1530 nm (Cyan)
                [0.2, 0.5, 1.0], // 1535 nm (Blue)
                [0.1, 1.0, 0.5], // 1540 nm (Green)
                [0.9, 0.9, 0.1], // 1545 nm (Yellow)
                [1.0, 0.5, 0.1], // 1550 nm (Amber)
                [1.0, 0.1, 0.5], // 1555 nm (Magenta)
                [0.8, 0.1, 1.0], // 1560 nm (Violet)
                [0.3, 1.0, 0.9], // 1565 nm (Seafoam)
            ];

            let is_tuned = metrics.resonance_transmission_factor > 0.4;
            let emission_mult = if is_tuned { 2.5 } else { 0.3 };

            for i in 0..ring_count {
                let rx = -2.8 + i as f32 * 0.8;
                let rz = -2.15; // Coupled right alongside the optical bus waveguide
                let col = wdm_colors[i % wdm_colors.len()];

                // Circular Silicon Microring
                let (ring_v, ring_i) = ProceduralMesh::torus(0.28, 0.035, 20, 8, col);
                let r_base = verts.len() as u32;
                for v in ring_v {
                    let pos = [rx + v.position[0], bus_y + v.position[1] + 0.04, rz + v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.2, 0.1, emission_mult, 0.0]));
                }
                for idx in ring_i { indices.push(r_base + idx); }

                // Micro-Heater Thermal Tuning Pad on Ring Center
                let heater_col = if cfg.auto_thermal_lock {
                    [1.0, 0.8, 0.2] // Glowing warm gold when active
                } else {
                    [0.4, 0.4, 0.4] // Inactive gray
                };
                let (ht_v, ht_i) = ProceduralMesh::cylinder(0.12, 0.04, 8, heater_col);
                let ht_base = verts.len() as u32;
                for v in ht_v {
                    let pos = [rx + v.position[0], bus_y + v.position[1] + 0.05, rz + v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.8, 0.3, if cfg.auto_thermal_lock { 1.5 } else { 0.0 }, 0.0]));
                }
                for idx in ht_i { indices.push(ht_base + idx); }

                // Drop Port Photodetector (Germanium on Silicon)
                let (det_v, det_i) = ProceduralMesh::cylinder(0.10, 0.06, 8, [0.1, 0.15, 0.2]);
                let d_base = verts.len() as u32;
                for v in det_v {
                    let pos = [rx + v.position[0], bus_y + v.position[1] + 0.05, rz + 0.55 + v.position[2]];
                    verts.push(GameVertex::new(pos, v.normal, v.color, [0.9, 0.2, 0.4, 0.0]));
                }
                for idx in det_i { indices.push(d_base + idx); }
            }
        }

        // ── 6. Particle Mesh ───────────────────────────────────────────────
        let (p_v, p_i) = self.particle_system.build_mesh();
        let p_base = verts.len() as u32;
        verts.extend(p_v);
        for idx in p_i { indices.push(p_base + idx); }

        // ── 7. 2D Real-Time In-Engine HUD & Telemetry Dashboard ─────────────

        // A. Top Navigation Header & Architecture Selector Bar
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.84, 1.0, 1.0, [0.05, 0.06, 0.08], 0.95);
        push_ui_rect(&mut verts, &mut indices, -1.0, 0.835, 1.0, 0.84, [0.18, 0.22, 0.30], 1.0); // Border

        push_text(&mut verts, &mut indices, "SILICON PHOTONICS SIMULATOR", -0.98, 0.95, 0.015, 0.028, [0.0, 0.9, 1.0]);

        // Architecture Mode Tabs: [1: ELECTRICAL GPU], [2: CPO HYBRID], [3: PHOTONIC AI]
        let btn1_bg = if mode == ArchitectureMode::ElectronicGpu { [0.8, 0.35, 0.1] } else { [0.12, 0.15, 0.20] };
        push_ui_rect(&mut verts, &mut indices, -0.52, 0.86, -0.22, 0.97, btn1_bg, 0.95);
        push_text(&mut verts, &mut indices, "[1] ELECTRICAL GPU", -0.50, 0.90, 0.012, 0.024, [1.0, 1.0, 1.0]);

        let btn2_bg = if mode == ArchitectureMode::CpoHybrid { [0.1, 0.7, 0.8] } else { [0.12, 0.15, 0.20] };
        push_ui_rect(&mut verts, &mut indices, -0.20, 0.86, 0.10, 0.97, btn2_bg, 0.95);
        push_text(&mut verts, &mut indices, "[2] CPO HYBRID", -0.17, 0.90, 0.012, 0.024, [1.0, 1.0, 1.0]);

        let btn3_bg = if mode == ArchitectureMode::PhotonicAccelerator { [0.7, 0.15, 0.9] } else { [0.12, 0.15, 0.20] };
        push_ui_rect(&mut verts, &mut indices, 0.12, 0.86, 0.45, 0.97, btn3_bg, 0.95);
        push_text(&mut verts, &mut indices, "[3] PHOTONIC AI CORE", 0.15, 0.90, 0.012, 0.024, [1.0, 1.0, 1.0]);

        // Reset Button
        push_ui_rect(&mut verts, &mut indices, 0.82, 0.86, 0.97, 0.97, [0.4, 0.12, 0.15], 0.9);
        push_text(&mut verts, &mut indices, "RESET [R]", 0.84, 0.90, 0.012, 0.024, [1.0, 1.0, 1.0]);

        // B. Left Parameter Controls Panel (Interactive Steppers)
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.65, -0.62, 0.80, [0.06, 0.07, 0.10], 0.92);
        push_ui_rect(&mut verts, &mut indices, -0.98, 0.74, -0.62, 0.80, [0.14, 0.18, 0.26], 1.0);
        push_text(&mut verts, &mut indices, "PHYSICS CONTROLS", -0.95, 0.755, 0.014, 0.025, [1.0, 0.8, 0.2]);

        // 1. Laser Optical Power
        let laser_str = format!("LASER: {:.0} MW [L]", cfg.laser_power_mw);
        push_text(&mut verts, &mut indices, &laser_str, -0.95, 0.64, 0.012, 0.022, [0.2, 0.8, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.57, -0.80, 0.62, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[-] 5MW", -0.94, 0.58, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, 0.57, -0.64, 0.62, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[+] 5MW", -0.76, 0.58, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // 2. WDM Channels
        let wdm_str = format!("WDM: {} CHANNELS [C]", cfg.wdm_channels);
        push_text(&mut verts, &mut indices, &wdm_str, -0.95, 0.53, 0.012, 0.022, [0.2, 0.8, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.46, -0.80, 0.51, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[/2] CH", -0.94, 0.47, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, 0.46, -0.64, 0.51, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[*2] CH", -0.76, 0.47, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // 3. Baud Rate
        let baud_str = format!("RATE: {:.0} GBPS [B]", cfg.baud_rate_gbps);
        push_text(&mut verts, &mut indices, &baud_str, -0.95, 0.42, 0.012, 0.022, [0.2, 0.8, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.35, -0.80, 0.40, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[-] RATE", -0.94, 0.36, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, 0.35, -0.64, 0.40, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[+] RATE", -0.76, 0.36, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // 4. Operating Temperature
        let temp_str = format!("TEMP: {:.1} C [T]", cfg.die_temperature_c);
        let temp_col = if cfg.die_temperature_c > 85.0 { [1.0, 0.2, 0.2] } else { [1.0, 0.7, 0.2] };
        push_text(&mut verts, &mut indices, &temp_str, -0.95, 0.31, 0.012, 0.022, temp_col);
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.24, -0.80, 0.29, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[-] 5C", -0.94, 0.25, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, 0.24, -0.64, 0.29, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[+] 5C", -0.76, 0.25, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // 5. Thermal Auto-Lock Button
        let lock_label = if cfg.auto_thermal_lock { "LOCK: AUTO [H]" } else { "LOCK: OFF [H]" };
        let lock_bg = if cfg.auto_thermal_lock { [0.1, 0.6, 0.3] } else { [0.7, 0.2, 0.2] };
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.14, -0.64, 0.21, lock_bg, 0.9);
        push_text(&mut verts, &mut indices, lock_label, -0.92, 0.16, 0.011, 0.022, [1.0, 1.0, 1.0]);

        // 6. Photonic Cores
        let core_str = format!("OPT CORES: {} [P]", cfg.num_photonic_cores);
        push_text(&mut verts, &mut indices, &core_str, -0.95, 0.09, 0.012, 0.022, [0.8, 0.4, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.95, 0.02, -0.80, 0.07, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[-] CORE", -0.94, 0.03, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, 0.02, -0.64, 0.07, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[+] CORE", -0.76, 0.03, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // 7. Matrix Workload N
        let mat_str = format!("GEMM: {}x{} [M]", cfg.matrix_dim_n, cfg.matrix_dim_n);
        push_text(&mut verts, &mut indices, &mat_str, -0.95, -0.03, 0.012, 0.022, [0.2, 0.9, 0.6]);
        push_ui_rect(&mut verts, &mut indices, -0.95, -0.10, -0.80, -0.05, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[-] DIM", -0.94, -0.09, 0.010, 0.018, [1.0, 1.0, 1.0]);
        push_ui_rect(&mut verts, &mut indices, -0.78, -0.10, -0.64, -0.05, [0.15, 0.20, 0.30], 0.9);
        push_text(&mut verts, &mut indices, "[+] DIM", -0.76, -0.09, 0.010, 0.018, [1.0, 1.0, 1.0]);

        // Quick Hints
        push_text(&mut verts, &mut indices, "HINT: SPACE=DRIFT", -0.95, -0.20, 0.010, 0.018, [0.7, 0.7, 0.8]);
        push_text(&mut verts, &mut indices, "RIGHT DRAG=ORBIT", -0.95, -0.24, 0.010, 0.018, [0.7, 0.7, 0.8]);
        push_text(&mut verts, &mut indices, "WASD=FLY CAMERA", -0.95, -0.28, 0.010, 0.018, [0.7, 0.7, 0.8]);

        // C. Right Comparison & Metrics Dashboard
        push_ui_rect(&mut verts, &mut indices, 0.54, -0.65, 0.98, 0.80, [0.06, 0.07, 0.10], 0.92);
        push_ui_rect(&mut verts, &mut indices, 0.54, 0.74, 0.98, 0.80, [0.14, 0.18, 0.26], 1.0);
        push_text(&mut verts, &mut indices, "OPTICAL VS ELECTRICAL BENCHMARK", 0.56, 0.755, 0.013, 0.024, [0.0, 0.9, 1.0]);

        // 1. Bandwidth Comparison
        let bw_opt_str = format!("OPTICAL BW: {:.1} TBPS", metrics.optical_effective_bandwidth_tbps);
        let bw_elec_str = format!("COPPER BW:  {:.1} TBPS", metrics.electrical_bandwidth_tbps);
        push_text(&mut verts, &mut indices, &bw_opt_str, 0.56, 0.69, 0.012, 0.022, [0.0, 0.9, 1.0]);
        push_text(&mut verts, &mut indices, &bw_elec_str, 0.56, 0.65, 0.012, 0.022, [0.8, 0.5, 0.2]);

        // Bandwidth Horizontal Progress Meter
        let opt_bar_w = (metrics.optical_effective_bandwidth_tbps / 30.0).clamp(0.02, 0.38);
        push_ui_rect(&mut verts, &mut indices, 0.56, 0.62, 0.56 + opt_bar_w, 0.64, [0.0, 0.9, 1.0], 1.0);
        let elec_bar_w = (metrics.electrical_bandwidth_tbps / 30.0).clamp(0.02, 0.38);
        push_ui_rect(&mut verts, &mut indices, 0.56, 0.59, 0.56 + elec_bar_w, 0.61, [0.8, 0.5, 0.2], 1.0);

        // 2. Energy per Bit Comparison (pJ/bit)
        let pj_opt_str = format!("OPTICAL ENERGY:   {:.2} PJ/B", metrics.optical_energy_per_bit_pj);
        let pj_elec_str = format!("COPPER SERDES:   {:.2} PJ/B", metrics.electrical_energy_per_bit_pj);
        push_text(&mut verts, &mut indices, &pj_opt_str, 0.56, 0.53, 0.012, 0.022, [0.2, 1.0, 0.4]);
        push_text(&mut verts, &mut indices, &pj_elec_str, 0.56, 0.49, 0.012, 0.022, [1.0, 0.4, 0.3]);

        let sav_str = format!("ENERGY SAVINGS:  {:.1}%", metrics.power_savings_percent);
        push_text(&mut verts, &mut indices, &sav_str, 0.56, 0.45, 0.013, 0.024, [1.0, 0.9, 0.1]);

        // 3. Compute Throughput (TOPS)
        let comp_str = format!("ACTIVE COMPUTE:   {:.0} TOPS", metrics.active_compute_tops);
        let spd_str = format!("ACCELERATION:    {:.1}X", metrics.compute_speedup_x);
        push_text(&mut verts, &mut indices, &comp_str, 0.56, 0.39, 0.012, 0.022, [0.8, 0.3, 1.0]);
        push_text(&mut verts, &mut indices, &spd_str, 0.56, 0.35, 0.012, 0.022, [0.9, 0.3, 0.9]);

        // 4. Energy Efficiency (TOPS / Watt)
        let eff_opt_str = format!("PHOTONIC EFF:    {:.1} TOPS/W", metrics.photonic_efficiency_tops_w);
        let eff_elec_str = format!("GPU CMOS EFF:    {:.2} TOPS/W", metrics.electronic_efficiency_tops_w);
        push_text(&mut verts, &mut indices, &eff_opt_str, 0.56, 0.29, 0.012, 0.022, [0.1, 1.0, 0.6]);
        push_text(&mut verts, &mut indices, &eff_elec_str, 0.56, 0.25, 0.012, 0.022, [0.7, 0.7, 0.7]);

        // 5. Total System Power & Interconnect Power
        let pwr_tot_str = format!("TOTAL SYSTEM:    {:.1} W", metrics.total_system_power_watts);
        let pwr_io_str = format!("INTERCONNECT:    {:.1} W", metrics.active_interconnect_power_watts);
        push_text(&mut verts, &mut indices, &pwr_tot_str, 0.56, 0.19, 0.012, 0.022, [1.0, 0.8, 0.3]);
        push_text(&mut verts, &mut indices, &pwr_io_str, 0.56, 0.15, 0.012, 0.022, [1.0, 0.6, 0.2]);

        // 6. Optical Link Health & Photometrics
        let loss_str = format!("TOTAL LOSS:      {:.1} DB", metrics.optical_insertion_loss_db);
        let rx_str = format!("RX OPT POWER:    {:.1} DBM", metrics.received_power_dbm);
        let ber_str = format!("ESTIMATED BER:   {:.1E}", metrics.optical_ber);
        push_text(&mut verts, &mut indices, &loss_str, 0.56, 0.09, 0.011, 0.020, [0.7, 0.8, 0.9]);
        push_text(&mut verts, &mut indices, &rx_str, 0.56, 0.05, 0.011, 0.020, [0.7, 0.8, 0.9]);
        push_text(&mut verts, &mut indices, &ber_str, 0.56, 0.01, 0.011, 0.020, if metrics.optical_ber > 1e-3 { [1.0, 0.2, 0.2] } else { [0.2, 1.0, 0.4] });

        // 7. Thermal Resonator State
        let drift_str = format!("THERMAL DRIFT:   {:.3} NM", metrics.thermal_wavelength_drift_nm);
        let trans_str = format!("RING RESONANCE:  {:.1}%", metrics.resonance_transmission_factor * 100.0);
        let tune_str = format!("HEATER POWER:    {:.2} W", metrics.thermal_tuning_power_watts);
        push_text(&mut verts, &mut indices, &drift_str, 0.56, -0.05, 0.011, 0.020, [1.0, 0.8, 0.2]);
        push_text(&mut verts, &mut indices, &trans_str, 0.56, -0.09, 0.011, 0.020, if metrics.resonance_transmission_factor > 0.5 { [0.2, 1.0, 0.4] } else { [1.0, 0.2, 0.2] });
        push_text(&mut verts, &mut indices, &tune_str, 0.56, -0.13, 0.011, 0.020, [0.8, 0.8, 0.9]);

        // D. Bottom Status Bar & Telemetry Console
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.96, 0.98, -0.70, [0.05, 0.06, 0.08], 0.95);
        push_ui_rect(&mut verts, &mut indices, -0.98, -0.71, 0.98, -0.70, [0.18, 0.22, 0.30], 1.0);

        push_text(&mut verts, &mut indices, "TELEMETRY LOG & EVENT MONITOR:", -0.96, -0.68, 0.012, 0.022, [0.0, 0.9, 1.0]);
        let log_msg = format!("> {}", self.status_message);
        push_text(&mut verts, &mut indices, &log_msg, -0.96, -0.76, 0.013, 0.024, [0.2, 1.0, 0.5]);

        let mode_summary = format!(
            "ACTIVE ARCHITECTURE: {} | WORKLOAD: {} TFLOPS | LATENCY SPEEDUP: {:.1}X",
            mode.name(),
            metrics.workload_teraflops,
            metrics.latency_speedup_x
        );
        push_text(&mut verts, &mut indices, &mode_summary, -0.96, -0.84, 0.011, 0.020, [0.9, 0.8, 0.3]);

        let hotkey_bar = "HOTKEYS: [1] GPU  [2] CPO  [3] PHOTONIC | [L] LASER | [C] WDM | [B] BAUD | [T] TEMP | [H] LOCK | [P] CORES | [M] GEMM";
        push_text(&mut verts, &mut indices, hotkey_bar, -0.96, -0.92, 0.010, 0.018, [0.6, 0.7, 0.8]);

        (verts, indices)
    }
}

fn main() {
    println!("============================================================");
    println!("  METATOPIA — GPU TO INTEGRATED SILICON PHOTONICS SIMULATOR ");
    println!("============================================================");
    println!(" Simulating physical transition from Electronic GPU to Photonic AI:");
    println!("   • Co-Packaged Optics (CPO) Hybrid & TeraPHY Transceivers");
    println!("   • Dense Wavelength Division Multiplexing (DWDM) in Si Waveguides");
    println!("   • Microring Resonators (MRR) & Thermo-Optic Closed-Loop Heaters");
    println!("   • Coherent MZI Mesh Matrix-Vector Multipliers (GEMM at light speed)");
    println!("   • Real-Time Energy-per-Bit (pJ/bit), TOPS, and TOPS/Watt Analytics");
    println!(" Controls:");
    println!("   [1] / [2] / [3] : Switch Architectural Mode (GPU / CPO / Photonic)");
    println!("   [L] / [Shift+L] : Step Laser Power Up / Down (mW)");
    println!("   [C] / [Shift+C] : Step WDM Channels (1 to 64 wavelengths)");
    println!("   [B] / [Shift+B] : Step Optical Baud Rate (Gbps)");
    println!("   [T] / [Shift+T] : Step Die Temperature Up / Down (°C)");
    println!("   [H]             : Toggle Closed-Loop Thermal Auto-Lock");
    println!("   [M] / [Shift+M] : Adjust GEMM Workload Dimension N");
    println!("   [P] / [Shift+P] : Adjust Photonic Tensor Core Tile Count");
    println!("   [Space]         : Toggle Continuous Thermal Drift Perturbation");
    println!("   [R]             : Reset Simulator to Defaults");
    println!("   Mouse Left      : Click On-Screen Buttons & Steppers");
    println!("   Mouse Right Drag: Free Orbit & Fly Camera");
    println!("   WASD            : Move Camera in 3D Space");
    println!("============================================================");

    run_game(GpuPhotonicsSimApp::new());
}
