//! Integrated Silicon Photonics & GPU Optical Interconnect Simulator
//!
//! Models the architectural transition from legacy electrical GPU interconnects (copper SerDes,
//! PCIe Gen6, NVLink) to Integrated Silicon Photonics (SiPh), including:
//! - Co-Packaged Optics (CPO) optical I/O engines (e.g. Ayar Labs TeraPHY, Celestial AI, Intel OIO)
//! - Dense Wavelength Division Multiplexing (DWDM) over single-mode silicon waveguides
//! - Microring Resonator (MRR) banks with thermo-optic closed-loop heater tuning (dn/dT = 1.86e-4 K^-1)
//! - Optical power budgets (waveguide loss, grating couplers, crosstalk, insertion loss)
//! - Photonic Tensor Core coherent Matrix-Vector Multipliers (MZI meshes / MRR weight banks)
//! - Real-time energy-per-bit (pJ/bit), bandwidth density (Tbps/mm), TOPS, and TOPS/Watt benchmarks


/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f32 = 299_792_458.0;

/// Silicon group index for 220nm x 450nm strip waveguide at 1550nm
pub const SILICON_GROUP_INDEX: f32 = 3.96;

/// Silicon thermo-optic coefficient dn/dT (K^-1)
pub const SILICON_THERMO_OPTIC_COEFF: f32 = 1.86e-4;

/// Architecture simulation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureMode {
    /// Legacy Electronic GPU: CMOS SMs, HBM3 via Silicon Interposer, Copper SerDes / NVLink
    ElectronicGpu,
    /// Co-Packaged Optics (CPO) Hybrid: Electronic GPU compute die + Silicon Photonic Optical I/O Chiplets
    CpoHybrid,
    /// Fully Integrated Photonic AI Accelerator: Coherent Photonic Tensor Core + All-Optical Interconnect
    PhotonicAccelerator,
}

impl ArchitectureMode {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ElectronicGpu => "1. Electronic GPU (Copper SerDes)",
            Self::CpoHybrid => "2. Co-Packaged Optics (CPO Hybrid)",
            Self::PhotonicAccelerator => "3. Integrated Photonic Accelerator",
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            Self::ElectronicGpu => "ELECTRICAL GPU",
            Self::CpoHybrid => "CPO HYBRID",
            Self::PhotonicAccelerator => "PHOTONIC AI",
        }
    }
}

/// Simulation configuration and adjustable parameters
#[derive(Debug, Clone)]
pub struct PhotonicsConfig {
    /// Active architectural mode
    pub mode: ArchitectureMode,

    // ─── Optical Interconnect & Laser ───────────────────────────────────────
    /// Laser optical output power per wavelength channel (mW)
    pub laser_power_mw: f32,
    /// Laser Wall-Plug Efficiency (WPE), ratio of optical power to electrical DC power (0.10 - 0.35)
    pub laser_wpe: f32,
    /// Number of WDM wavelength channels per bus waveguide (1, 4, 8, 16, 32, 64)
    pub wdm_channels: usize,
    /// Baud rate per wavelength channel (Gbps) [e.g. 25.0, 56.0, 112.0 Gbps PAM4]
    pub baud_rate_gbps: f32,
    /// Number of optical waveguide I/O ports / fiber lanes (e.g. 8, 16, 32, 64)
    pub optical_ports: usize,
    /// Waveguide interconnect length (cm)
    pub waveguide_length_cm: f32,
    /// Waveguide optical propagation loss (dB/cm) [typical 1.0 - 2.5 dB/cm for Si strip]
    pub waveguide_loss_db_per_cm: f32,
    /// Fiber-to-chip grating coupler or edge coupler loss per facet (dB)
    pub coupler_loss_db: f32,
    /// Microring resonator insertion / drop loss (dB)
    pub ring_drop_loss_db: f32,
    /// Photodetector responsivity (A/W) [Ge-on-Si PIN diode]
    pub photodetector_responsivity: f32,
    /// Energy consumption of optical modulator driver + TIA receiver (pJ/bit)
    pub optical_eo_oe_energy_pj_b: f32,

    // ─── Thermal & Microring Resonator Parameters ───────────────────────────
    /// Chip junction operating temperature (°C)
    pub die_temperature_c: f32,
    /// Nominal calibration reference temperature (°C)
    pub nominal_temperature_c: f32,
    /// Microring Resonator Quality factor (Q = lambda / FWHM)
    pub mrr_q_factor: f32,
    /// Center operating wavelength (nm) [e.g. 1550.0 nm C-Band or 1310.0 nm O-Band]
    pub center_wavelength_nm: f32,
    /// Active closed-loop thermal microheater tuning enabled
    pub auto_thermal_lock: bool,
    /// Microheater tuning efficiency (mW of electrical heater power per nm of wavelength shift)
    pub tuning_efficiency_mw_per_nm: f32,
    /// Manual thermal heater power override (mW) when auto_thermal_lock is false
    pub manual_tuning_power_mw: f32,

    // ─── Electronic GPU Baseline Parameters ─────────────────────────────────
    /// Electrical SerDes energy per bit (pJ/bit) [copper PHY + retimer + equalization]
    pub electronic_serdes_energy_pj_b: f32,
    /// Number of electrical SerDes differential pairs / lanes (e.g. 128 lanes for NVLink/PCIe)
    pub electronic_serdes_lanes: usize,
    /// Electrical lane signaling rate (Gbps) [e.g. 56.0 Gbps PAM4]
    pub electronic_lane_rate_gbps: f32,
    /// Number of Electronic Streaming Multiprocessors / Tensor Cores
    pub gpu_sm_count: usize,
    /// Electronic GPU core clock frequency (GHz)
    pub gpu_clock_ghz: f32,
    /// Electronic GPU compute die TDP (Watts)
    pub gpu_compute_tdp_watts: f32,

    // ─── Compute Workload & Photonic Tensor Core ────────────────────────────
    /// Square matrix dimension N for GEMM benchmark (N x N @ FP16 / INT8)
    pub matrix_dim_n: usize,
    /// Batch size for inference / training pass
    pub batch_size: usize,
    /// Number of integrated Photonic Tensor Core tiles (e.g. 4, 8, 16)
    pub num_photonic_cores: usize,
    /// Coherent Photonic MZI mesh matrix dimension (e.g. 64x64 or 128x128)
    pub mzi_mesh_dim: usize,
    /// High-speed electro-optic DAC/ADC sampling rate for photonic compute (GHz)
    pub photonic_compute_clock_ghz: f32,
}

impl Default for PhotonicsConfig {
    fn default() -> Self {
        Self {
            mode: ArchitectureMode::CpoHybrid,

            // Optical link defaults (state-of-the-art SiPh CPO)
            laser_power_mw: 20.0,
            laser_wpe: 0.22, // 22% wall-plug efficiency
            wdm_channels: 16,
            baud_rate_gbps: 56.0,
            optical_ports: 16,
            waveguide_length_cm: 6.0,
            waveguide_loss_db_per_cm: 1.5,
            coupler_loss_db: 1.2,
            ring_drop_loss_db: 0.8,
            photodetector_responsivity: 0.95,
            optical_eo_oe_energy_pj_b: 1.6, // 1.6 pJ/bit

            // Thermal defaults
            die_temperature_c: 65.0,
            nominal_temperature_c: 45.0,
            mrr_q_factor: 12_000.0,
            center_wavelength_nm: 1550.0,
            auto_thermal_lock: true,
            tuning_efficiency_mw_per_nm: 2.2, // 2.2 mW/nm
            manual_tuning_power_mw: 0.0,

            // Electronic baseline (PCIe Gen6 / NVLink 4 level)
            electronic_serdes_energy_pj_b: 14.5, // 14.5 pJ/bit
            electronic_serdes_lanes: 64,
            electronic_lane_rate_gbps: 56.0,
            gpu_sm_count: 128,
            gpu_clock_ghz: 1.8,
            gpu_compute_tdp_watts: 550.0,

            // Workload & Photonic Compute
            matrix_dim_n: 2048,
            batch_size: 16,
            num_photonic_cores: 8,
            mzi_mesh_dim: 64,
            photonic_compute_clock_ghz: 10.0,
        }
    }
}

/// Real-time computed physical and architectural metrics
#[derive(Debug, Clone, Default)]
pub struct SimMetrics {
    // ─── Interconnect Metrics ───────────────────────────────────────────────
    /// Maximum raw optical interconnect bandwidth (Tbps)
    pub optical_raw_bandwidth_tbps: f32,
    /// Effective optical bandwidth accounting for thermal detuning / transmission loss (Tbps)
    pub optical_effective_bandwidth_tbps: f32,
    /// Electronic SerDes interconnect bandwidth (Tbps)
    pub electrical_bandwidth_tbps: f32,
    /// Active bandwidth of current architectural mode (Tbps)
    pub active_bandwidth_tbps: f32,

    // ─── Optical Loss Budget & Photometrics ─────────────────────────────────
    /// Total optical insertion loss from laser to photodetector (dB)
    pub optical_insertion_loss_db: f32,
    /// Received optical power at photodetector (dBm)
    pub received_power_dbm: f32,
    /// Received optical power in microwatts (uW)
    pub received_power_uw: f32,
    /// Generated photocurrent (mA)
    pub photocurrent_ma: f32,
    /// Estimated optical Bit Error Rate (BER)
    pub optical_ber: f32,

    // ─── Thermal & Microring Resonator Tracking ─────────────────────────────
    /// Thermal resonance wavelength drift without tuning (nm)
    pub thermal_wavelength_drift_nm: f32,
    /// Microring Full-Width at Half-Maximum resonance linewidth (nm)
    pub ring_fwhm_nm: f32,
    /// Normalized resonance transmission factor (0.0 = completely detuned, 1.0 = perfect resonance)
    pub resonance_transmission_factor: f32,
    /// Total electrical power consumed by microheaters for thermal stabilization (Watts)
    pub thermal_tuning_power_watts: f32,

    // ─── Power & Energy Efficiency ──────────────────────────────────────────
    /// Optical interconnect total power consumption (Watts) [Laser electrical + Drivers + Heaters]
    pub optical_interconnect_power_watts: f32,
    /// Electronic SerDes total interconnect power consumption (Watts)
    pub electrical_interconnect_power_watts: f32,
    /// Active interconnect power (Watts)
    pub active_interconnect_power_watts: f32,
    /// Optical interconnect energy per bit (pJ/bit)
    pub optical_energy_per_bit_pj: f32,
    /// Electrical interconnect energy per bit (pJ/bit)
    pub electrical_energy_per_bit_pj: f32,
    /// Interconnect power reduction percentage relative to pure electrical (%)
    pub power_savings_percent: f32,

    // ─── Latency ────────────────────────────────────────────────────────────
    /// Optical time-of-flight latency across package waveguide (picoseconds)
    pub optical_flight_latency_ps: f32,
    /// Electrical copper propagation + SerDes FIFO latency (picoseconds)
    pub electrical_latency_ps: f32,
    /// End-to-end communication latency speedup factor (Optical vs Electrical)
    pub latency_speedup_x: f32,

    // ─── Compute Performance (GEMM Benchmark) ───────────────────────────────
    /// Total floating-point operations in current matrix workload (TeraFLOPs)
    pub workload_teraflops: f32,
    /// Electronic GPU compute throughput (TOPS / TFLOPs)
    pub electronic_compute_tops: f32,
    /// Photonic Tensor Core compute throughput (TOPS)
    pub photonic_compute_tops: f32,
    /// Active compute throughput (TOPS)
    pub active_compute_tops: f32,
    /// Electronic compute energy efficiency (TOPS / Watt)
    pub electronic_efficiency_tops_w: f32,
    /// Photonic compute energy efficiency (TOPS / Watt)
    pub photonic_efficiency_tops_w: f32,
    /// Active compute energy efficiency (TOPS / Watt)
    pub active_efficiency_tops_w: f32,
    /// Total system power (Compute die + Interconnect) in Watts
    pub total_system_power_watts: f32,
    /// Compute acceleration factor over electronic GPU baseline
    pub compute_speedup_x: f32,
}

/// Silicon Photonics & GPU Optical Interconnect Physics Simulator
pub struct PhotonicsSimulator {
    pub config: PhotonicsConfig,
    pub metrics: SimMetrics,
    /// Dynamic animation time accumulator
    pub sim_time: f32,
    /// Thermal perturbation sine wave frequency for realistic dynamic temperature drifting
    pub thermal_drift_enabled: bool,
    /// Status notification message
    pub status_message: String,
}

impl PhotonicsSimulator {
    pub fn new(config: PhotonicsConfig) -> Self {
        let mut sim = Self {
            config,
            metrics: SimMetrics::default(),
            sim_time: 0.0,
            thermal_drift_enabled: false,
            status_message: "Silicon Photonics Simulator Initialized".into(),
        };
        sim.recalculate();
        sim
    }

    /// Update simulation state with elapsed delta time (seconds)
    pub fn update(&mut self, dt: f32) {
        self.sim_time += dt;

        if self.thermal_drift_enabled {
            // Simulate subtle die thermal fluctuation +/- 4.5 °C over time
            let delta_t = (self.sim_time * 0.8).sin() * 4.5;
            self.config.die_temperature_c = (self.config.nominal_temperature_c + 20.0 + delta_t).clamp(25.0, 110.0);
        }

        self.recalculate();
    }

    /// Recalculate all physical models and metrics based on current configuration
    pub fn recalculate(&mut self) {
        let cfg = &self.config;

        // ── 1. Interconnect Bandwidths ─────────────────────────────────────
        // Optical: Channels * BaudRate * Ports
        let opt_raw_gbps = (cfg.wdm_channels as f32) * cfg.baud_rate_gbps * (cfg.optical_ports as f32);
        let optical_raw_bandwidth_tbps = opt_raw_gbps / 1000.0;

        // Electrical: Lanes * LaneRate
        let elec_raw_gbps = (cfg.electronic_serdes_lanes as f32) * cfg.electronic_lane_rate_gbps;
        let electrical_bandwidth_tbps = elec_raw_gbps / 1000.0;

        // ── 2. Thermal Tuning & Microring Resonator Resonance ───────────────
        // Temperature difference from nominal calibration point
        let delta_t = cfg.die_temperature_c - cfg.nominal_temperature_c;

        // Thermo-optic wavelength shift: dlambda = (lambda / n_g) * (dn/dT) * DeltaT
        // For Si at 1550nm: (1550 / 3.96) * 1.86e-4 ~ 0.0728 nm/K (~ 0.08 nm/°C)
        let thermal_wavelength_drift_nm = (cfg.center_wavelength_nm / SILICON_GROUP_INDEX)
            * SILICON_THERMO_OPTIC_COEFF
            * delta_t;

        // Ring Resonator FWHM: FWHM = lambda / Q
        let ring_fwhm_nm = cfg.center_wavelength_nm / cfg.mrr_q_factor.max(100.0);

        // Required microheater tuning:
        let (effective_detuning_nm, thermal_tuning_power_watts) = if cfg.auto_thermal_lock {
            // Closed-loop heater brings detuning back to zero (locked)
            let tuning_shift_needed_nm = thermal_wavelength_drift_nm.abs();
            // Total tuning power = (mW/nm) * nm * TotalRings
            let total_rings = (cfg.wdm_channels * cfg.optical_ports) as f32;
            let heater_power_mw = cfg.tuning_efficiency_mw_per_nm * tuning_shift_needed_nm * total_rings;
            (0.0f32, heater_power_mw / 1000.0)
        } else {
            // Manual tuning
            let manual_shift_nm = cfg.manual_tuning_power_mw / cfg.tuning_efficiency_mw_per_nm.max(0.1);
            let residual_detuning = (thermal_wavelength_drift_nm - manual_shift_nm).abs();
            let total_rings = (cfg.wdm_channels * cfg.optical_ports) as f32;
            let heater_power_watts = (cfg.manual_tuning_power_mw * total_rings) / 1000.0;
            (residual_detuning, heater_power_watts)
        };

        // Lorentzian Resonance Transmission Drop Profile:
        // T(delta) = (FWHM/2)^2 / (delta^2 + (FWHM/2)^2)
        let half_fwhm = ring_fwhm_nm * 0.5;
        let resonance_transmission_factor = (half_fwhm * half_fwhm) / (effective_detuning_nm * effective_detuning_nm + half_fwhm * half_fwhm);

        // Effective optical bandwidth with thermal loss factor
        let optical_effective_bandwidth_tbps = optical_raw_bandwidth_tbps * resonance_transmission_factor.clamp(0.01, 1.0);

        // ── 3. Optical Power Budget & Insertion Loss ───────────────────────
        // Total insertion loss:
        // L_total = 2 * CouplerLoss + WaveguideLoss * Length + RingDropLoss + CrosstalkMargin(0.5dB)
        let total_waveguide_loss = cfg.waveguide_loss_db_per_cm * cfg.waveguide_length_cm;
        let optical_insertion_loss_db = 2.0 * cfg.coupler_loss_db + total_waveguide_loss + cfg.ring_drop_loss_db + 0.5;

        // Laser input power (dBm): P(dBm) = 10 * log10(P(mW))
        let laser_power_dbm = 10.0 * cfg.laser_power_mw.max(0.001).log10();
        let received_power_dbm = laser_power_dbm - optical_insertion_loss_db;
        let received_power_mw = 10.0f32.powf(received_power_dbm / 10.0);
        let received_power_uw = received_power_mw * 1000.0;

        // Photocurrent: I_ph = R * P_rx
        let photocurrent_ma = cfg.photodetector_responsivity * received_power_mw;

        // Bit Error Rate (BER) model:
        // As received power drops below receiver sensitivity (-18 dBm), BER degrades rapidly
        let snr_db = received_power_dbm - (-22.0); // Sensitivity threshold at -22 dBm
        let optical_ber = if snr_db < 0.0 {
            0.5 // Complete link failure
        } else {
            let q_factor = (snr_db * 0.6).max(0.1);
            0.5 * (-q_factor * 0.707).exp()
        }.clamp(1e-15, 0.5);

        // ── 4. Energy per Bit & Power Dissipation ───────────────────────────
        // Electrical SerDes Power: BW * E_bit
        let electrical_interconnect_power_watts = (electrical_bandwidth_tbps * 1000.0) * cfg.electronic_serdes_energy_pj_b * 1e-3;

        // Optical Interconnect Power:
        // Laser electrical power = (OpticalPower per channel * Channels * Ports) / WPE
        let total_optical_channels = (cfg.wdm_channels * cfg.optical_ports) as f32;
        let total_laser_optical_mw = cfg.laser_power_mw * total_optical_channels;
        let laser_electrical_power_watts = (total_laser_optical_mw / cfg.laser_wpe.max(0.05)) / 1000.0;

        // E/O and O/E transceiver circuit power:
        let transceiver_circuits_watts = (optical_raw_bandwidth_tbps * 1000.0) * cfg.optical_eo_oe_energy_pj_b * 1e-3;

        // Total optical interconnect power
        let optical_interconnect_power_watts = laser_electrical_power_watts + transceiver_circuits_watts + thermal_tuning_power_watts;

        // Optical effective energy per bit:
        let optical_energy_per_bit_pj = if optical_effective_bandwidth_tbps > 0.001 {
            (optical_interconnect_power_watts / (optical_effective_bandwidth_tbps * 1000.0)) * 1e3
        } else {
            999.0
        };

        // Active mode assignment
        let (active_bandwidth_tbps, active_interconnect_power_watts) = match cfg.mode {
            ArchitectureMode::ElectronicGpu => (electrical_bandwidth_tbps, electrical_interconnect_power_watts),
            ArchitectureMode::CpoHybrid => (optical_effective_bandwidth_tbps, optical_interconnect_power_watts),
            ArchitectureMode::PhotonicAccelerator => (optical_effective_bandwidth_tbps, optical_interconnect_power_watts),
        };

        let power_savings_percent = if cfg.electronic_serdes_energy_pj_b > 0.0 {
            ((cfg.electronic_serdes_energy_pj_b - optical_energy_per_bit_pj) / cfg.electronic_serdes_energy_pj_b) * 100.0
        } else {
            0.0
        };

        // ── 5. Latency Modeling ─────────────────────────────────────────────
        // Waveguide light speed: v = c / n_g ~ 3e8 / 3.96 = 7.57e7 m/s (13.2 ps/mm)
        let waveguide_length_mm = cfg.waveguide_length_cm * 10.0;
        let ps_per_mm = (SILICON_GROUP_INDEX / SPEED_OF_LIGHT) * 1e12 * 1e-3;
        let optical_flight_latency_ps = waveguide_length_mm * ps_per_mm + 45.0; // +45ps for E/O & O/E conversion delay

        // Electrical latency: copper RC delay (~25 ps/mm) + SerDes FIFO / retimer latency (~1800 ps)
        let electrical_latency_ps = waveguide_length_mm * 25.0 + 1850.0;
        let latency_speedup_x = electrical_latency_ps / optical_flight_latency_ps.max(1.0);

        // ── 6. Compute & Acceleration (GEMM Workload) ───────────────────────
        // GEMM operations: 2 * N^3 * Batch
        let n = cfg.matrix_dim_n as f32;
        let batch = cfg.batch_size as f32;
        let total_flops = 2.0 * n * n * n * batch;
        let workload_teraflops = total_flops / 1e12;

        // Electronic GPU compute TOPS (Tensor Cores):
        // SMs * (FMA_per_cycle * 2) * Clock
        // e.g. 128 SMs * 512 FP16 ops/clock * 1.8 GHz = 117.9 TFLOPs (Sparse: ~235 TOPS)
        let electronic_compute_tops = (cfg.gpu_sm_count as f32) * 1024.0 * cfg.gpu_clock_ghz * 1e-3;
        let electronic_efficiency_tops_w = electronic_compute_tops / cfg.gpu_compute_tdp_watts.max(10.0);

        // Photonic Tensor Core compute TOPS:
        // Uses MZI interferometer coherent mesh (Reck / Clements mesh).
        // Light travels passively through the mesh! In one pass: 2 * (MZI_dim^2) MAC ops
        // Operating at photonic clock (DAC/ADC rate) across WDM wavelengths and parallel optical core tiles:
        let mzi = cfg.mzi_mesh_dim as f32;
        let macs_per_cycle = 2.0 * mzi * mzi;
        let parallel_wavelengths = (cfg.wdm_channels / 2).max(1) as f32;
        let num_cores = cfg.num_photonic_cores as f32;
        let photonic_compute_tops = (macs_per_cycle * cfg.photonic_compute_clock_ghz * parallel_wavelengths * num_cores) * 1e-3;

        // Photonic compute power: passive optical propagation consumes ~0W runtime!
        // Only high-speed ADC/DAC converters + laser power contribute (~45W base + 4.5W per core)
        let photonic_compute_power_watts = 45.0 + (num_cores * 4.5);
        let photonic_efficiency_tops_w = photonic_compute_tops / photonic_compute_power_watts;

        // Active mode compute assignments
        let (active_compute_tops, active_compute_power_watts, active_efficiency_tops_w) = match cfg.mode {
            ArchitectureMode::ElectronicGpu => (electronic_compute_tops, cfg.gpu_compute_tdp_watts, electronic_efficiency_tops_w),
            ArchitectureMode::CpoHybrid => (electronic_compute_tops * 1.35, cfg.gpu_compute_tdp_watts, electronic_efficiency_tops_w * 1.35), // Interconnect unbottlenecked speedup
            ArchitectureMode::PhotonicAccelerator => (photonic_compute_tops, photonic_compute_power_watts, photonic_efficiency_tops_w),
        };

        let total_system_power_watts = active_compute_power_watts + active_interconnect_power_watts;
        let compute_speedup_x = active_compute_tops / electronic_compute_tops.max(0.1);

        self.metrics = SimMetrics {
            optical_raw_bandwidth_tbps,
            optical_effective_bandwidth_tbps,
            electrical_bandwidth_tbps,
            active_bandwidth_tbps,

            optical_insertion_loss_db,
            received_power_dbm,
            received_power_uw,
            photocurrent_ma,
            optical_ber,

            thermal_wavelength_drift_nm,
            ring_fwhm_nm,
            resonance_transmission_factor,
            thermal_tuning_power_watts,

            optical_interconnect_power_watts,
            electrical_interconnect_power_watts,
            active_interconnect_power_watts,
            optical_energy_per_bit_pj,
            electrical_energy_per_bit_pj: cfg.electronic_serdes_energy_pj_b,
            power_savings_percent,

            optical_flight_latency_ps,
            electrical_latency_ps,
            latency_speedup_x,

            workload_teraflops,
            electronic_compute_tops,
            photonic_compute_tops,
            active_compute_tops,
            electronic_efficiency_tops_w,
            photonic_efficiency_tops_w,
            active_efficiency_tops_w,
            total_system_power_watts,
            compute_speedup_x,
        };
    }

    /// Set architectural mode
    pub fn set_mode(&mut self, mode: ArchitectureMode) {
        self.config.mode = mode;
        self.status_message = format!("Switched to: {}", mode.name());
        self.recalculate();
    }

    /// Step laser power up/down (mW)
    pub fn adjust_laser_power(&mut self, delta_mw: f32) {
        self.config.laser_power_mw = (self.config.laser_power_mw + delta_mw).clamp(2.0, 100.0);
        self.status_message = format!("Laser Power set to {:.1} mW", self.config.laser_power_mw);
        self.recalculate();
    }

    /// Step WDM channels (powers of 2)
    pub fn adjust_wdm_channels(&mut self, increase: bool) {
        let current = self.config.wdm_channels;
        let next = if increase {
            match current {
                1 => 4,
                4 => 8,
                8 => 16,
                16 => 32,
                _ => 64,
            }
        } else {
            match current {
                64 => 32,
                32 => 16,
                16 => 8,
                8 => 4,
                _ => 1,
            }
        };
        self.config.wdm_channels = next;
        self.status_message = format!("WDM Channels set to {} wavelengths", next);
        self.recalculate();
    }

    /// Step baud rate (Gbps)
    pub fn adjust_baud_rate(&mut self, delta_gbps: f32) {
        self.config.baud_rate_gbps = (self.config.baud_rate_gbps + delta_gbps).clamp(10.0, 112.0);
        self.status_message = format!("Baud Rate set to {:.1} Gbps", self.config.baud_rate_gbps);
        self.recalculate();
    }

    /// Step die operating temperature (°C)
    pub fn adjust_temperature(&mut self, delta_c: f32) {
        self.config.die_temperature_c = (self.config.die_temperature_c + delta_c).clamp(20.0, 120.0);
        self.status_message = format!("Die Temperature set to {:.1} °C", self.config.die_temperature_c);
        self.recalculate();
    }

    /// Toggle automatic closed-loop thermal lock
    pub fn toggle_thermal_lock(&mut self) {
        self.config.auto_thermal_lock = !self.config.auto_thermal_lock;
        let state = if self.config.auto_thermal_lock { "ENABLED" } else { "DISABLED (MANUAL TUNING)" };
        self.status_message = format!("Closed-Loop Thermal Lock: {}", state);
        self.recalculate();
    }

    /// Adjust manual heater power when auto-lock is off
    pub fn adjust_manual_heater(&mut self, delta_mw: f32) {
        self.config.manual_tuning_power_mw = (self.config.manual_tuning_power_mw + delta_mw).clamp(0.0, 80.0);
        self.status_message = format!("Manual Heater Power: {:.1} mW", self.config.manual_tuning_power_mw);
        self.recalculate();
    }

    /// Step waveguide loss (dB/cm)
    pub fn adjust_waveguide_loss(&mut self, delta_db: f32) {
        self.config.waveguide_loss_db_per_cm = (self.config.waveguide_loss_db_per_cm + delta_db).clamp(0.2, 5.0);
        self.status_message = format!("Waveguide Loss set to {:.2} dB/cm", self.config.waveguide_loss_db_per_cm);
        self.recalculate();
    }

    /// Step GEMM matrix dimension N
    pub fn adjust_matrix_dim(&mut self, increase: bool) {
        let current = self.config.matrix_dim_n;
        let next = if increase {
            match current {
                512 => 1024,
                1024 => 2048,
                2048 => 4096,
                _ => 8192,
            }
        } else {
            match current {
                8192 => 4096,
                4096 => 2048,
                2048 => 1024,
                _ => 512,
            }
        };
        self.config.matrix_dim_n = next;
        self.status_message = format!("GEMM Workload N = {}x{}", next, next);
        self.recalculate();
    }

    /// Step number of parallel Photonic Tensor Core tiles (1 to 32)
    pub fn adjust_photonic_cores(&mut self, increase: bool) {
        let current = self.config.num_photonic_cores;
        let next = if increase {
            match current {
                1 => 2,
                2 => 4,
                4 => 8,
                8 => 16,
                _ => 32,
            }
        } else {
            match current {
                32 => 16,
                16 => 8,
                8 => 4,
                4 => 2,
                _ => 1,
            }
        };
        self.config.num_photonic_cores = next;
        self.status_message = format!("Photonic Cores set to {} tiles", next);
        self.recalculate();
    }
}

// ─── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_initialization_defaults() {
        let sim = PhotonicsSimulator::new(PhotonicsConfig::default());
        assert_eq!(sim.config.mode, ArchitectureMode::CpoHybrid);
        assert!(sim.metrics.optical_raw_bandwidth_tbps > 10.0);
        assert!(sim.metrics.optical_energy_per_bit_pj < 5.0);
        assert!(sim.metrics.power_savings_percent > 50.0);
    }

    #[test]
    fn test_optical_vs_electrical_bandwidth_and_power() {
        let mut sim = PhotonicsSimulator::new(PhotonicsConfig::default());
        sim.recalculate();

        // 16 channels * 56 Gbps * 16 ports = 14,336 Gbps = 14.336 Tbps
        assert!((sim.metrics.optical_raw_bandwidth_tbps - 14.336).abs() < 0.01);

        // 64 lanes * 56 Gbps = 3,584 Gbps = 3.584 Tbps
        assert!((sim.metrics.electrical_bandwidth_tbps - 3.584).abs() < 0.01);

        // Optical energy per bit must be significantly lower than 14.5 pJ/bit
        assert!(sim.metrics.optical_energy_per_bit_pj < sim.metrics.electrical_energy_per_bit_pj);
    }

    #[test]
    fn test_thermo_optic_detuning_without_lock() {
        let mut config = PhotonicsConfig::default();
        config.auto_thermal_lock = false;
        config.manual_tuning_power_mw = 0.0;
        config.nominal_temperature_c = 40.0;
        config.die_temperature_c = 75.0; // 35°C delta

        let mut sim = PhotonicsSimulator::new(config);
        sim.recalculate();

        // 35°C delta * ~0.0728 nm/°C ~ 2.55 nm drift
        assert!(sim.metrics.thermal_wavelength_drift_nm > 2.0);

        // Without tuning, ring is heavily detuned: resonance transmission must drop significantly
        assert!(sim.metrics.resonance_transmission_factor < 0.05);

        // Turn on auto-thermal lock: resonance restored to 1.0!
        sim.toggle_thermal_lock();
        assert_eq!(sim.metrics.resonance_transmission_factor, 1.0);
        assert!(sim.metrics.thermal_tuning_power_watts > 0.5);
    }

    #[test]
    fn test_photonic_compute_tops_and_efficiency() {
        let mut config = PhotonicsConfig::default();
        config.mode = ArchitectureMode::PhotonicAccelerator;
        let mut sim = PhotonicsSimulator::new(config);
        sim.recalculate();

        // Photonic accelerator efficiency must surpass electronic GPU baseline (> 50 TOPS/W and > 50x better)
        assert!(sim.metrics.photonic_efficiency_tops_w > 50.0);
        assert!(sim.metrics.photonic_efficiency_tops_w > sim.metrics.electronic_efficiency_tops_w * 50.0);
        assert!(sim.metrics.compute_speedup_x > 1.0);
    }

    #[test]
    fn test_optical_loss_and_ber() {
        let mut sim = PhotonicsSimulator::new(PhotonicsConfig::default());
        sim.adjust_laser_power(10.0); // 30 mW
        assert!(sim.metrics.received_power_dbm > -15.0);
        assert!(sim.metrics.optical_ber < 1e-4);

        // Extreme waveguide loss causes severe attenuation
        sim.config.waveguide_loss_db_per_cm = 5.0;
        sim.config.waveguide_length_cm = 15.0;
        sim.recalculate();
        assert!(sim.metrics.optical_insertion_loss_db > 50.0);
        assert_eq!(sim.metrics.optical_ber, 0.5); // Link failure
    }
}
