/**
 * Metatopia — GPU to Integrated Silicon Photonics Simulator
 * Complete interactive 3D Web physics simulator and benchmark dashboard.
 */

// ─── Physical Constants & Defaults ───
const SILICON_GROUP_INDEX = 3.96;
const SILICON_THERMO_OPTIC_COEFF = 1.86e-4; // K^-1
const CENTER_WAVELENGTH_NM = 1550.0;

// Simulation State
const simState = {
  mode: 1, // 0: Electronic GPU, 1: CPO Hybrid, 2: Photonic AI Core
  laserPowerMw: 20.0,
  laserWpe: 0.22,
  wdmChannels: 16,
  baudRateGbps: 56.0,
  opticalPorts: 16,
  dieTempC: 65.0,
  nominalTempC: 45.0,
  autoThermalLock: true,
  mrrQFactor: 12000.0,
  tuningEfficiencyMwPerNm: 2.2,
  manualTuningPowerMw: 0.0,
  waveguideLossDbPerCm: 1.5,
  waveguideLengthCm: 6.0,
  couplerLossDb: 1.2,
  ringDropLossDb: 0.8,
  opticalEoOeEnergyPjB: 1.6,
  electronicSerdesEnergyPjB: 14.5,
  electronicLanes: 64,
  electronicLaneRateGbps: 56.0,
  photonicCores: 8,
  mziMeshDim: 64,
  photonicClockGhz: 10.0,
  matrixDimN: 2048,
  demoTourActive: false,
  demoStage: 0,
  demoTimer: 0.0,
};

// Computed Physical Metrics
let metrics = {};

// ─── Physical Engine Calculations ───
function recalculatePhysics() {
  const s = simState;

  // 1. Interconnect Bandwidth
  const optRawGbps = s.wdmChannels * s.baudRateGbps * s.opticalPorts;
  const opticalRawTbps = optRawGbps / 1000.0;
  const electricalTbps = (s.electronicLanes * s.electronicLaneRateGbps) / 1000.0;

  // 2. Thermal Tuning & Microring Resonator Resonance
  const deltaT = s.dieTempC - s.nominalTempC;
  const thermalWavelengthDriftNm = (CENTER_WAVELENGTH_NM / SILICON_GROUP_INDEX) * SILICON_THERMO_OPTIC_COEFF * deltaT;
  const ringFwhmNm = CENTER_WAVELENGTH_NM / s.mrrQFactor;

  let effectiveDetuningNm = 0.0;
  let heaterPowerWatts = 0.0;
  const totalRings = s.wdmChannels * s.opticalPorts;

  if (s.autoThermalLock) {
    const shiftNeededNm = Math.abs(thermalWavelengthDriftNm);
    heaterPowerWatts = (s.tuningEfficiencyMwPerNm * shiftNeededNm * totalRings) / 1000.0;
    effectiveDetuningNm = 0.0;
  } else {
    const manualShiftNm = s.manualTuningPowerMw / s.tuningEfficiencyMwPerNm;
    effectiveDetuningNm = Math.abs(thermalWavelengthDriftNm - manualShiftNm);
    heaterPowerWatts = (s.manualTuningPowerMw * totalRings) / 1000.0;
  }

  // Lorentzian Resonance Transmission
  const halfFwhm = ringFwhmNm * 0.5;
  const resonanceTransmissionFactor = (halfFwhm * halfFwhm) / (effectiveDetuningNm * effectiveDetuningNm + halfFwhm * halfFwhm);
  const opticalEffectiveTbps = opticalRawTbps * Math.max(0.01, Math.min(1.0, resonanceTransmissionFactor));

  // 3. Optical Link Budget & Loss
  const totalWaveguideLoss = s.waveguideLossDbPerCm * s.waveguideLengthCm;
  const opticalInsertionLossDb = 2.0 * s.couplerLossDb + totalWaveguideLoss + s.ringDropLossDb + 0.5;

  const laserPowerDbm = 10.0 * Math.log10(Math.max(0.001, s.laserPowerMw));
  const receivedPowerDbm = laserPowerDbm - opticalInsertionLossDb;
  const receivedPowerMw = Math.pow(10.0, receivedPowerDbm / 10.0);
  const photocurrentMa = 0.95 * receivedPowerMw;

  const snrDb = receivedPowerDbm - (-22.0);
  let opticalBer = 1e-15;
  if (snrDb < 0.0) {
    opticalBer = 0.5;
  } else {
    const qFactor = Math.max(0.1, snrDb * 0.6);
    opticalBer = Math.min(0.5, Math.max(1e-15, 0.5 * Math.exp(-qFactor * 0.707)));
  }

  // 4. Power & Energy per Bit
  const electricalPowerWatts = (electricalTbps * 1000.0) * s.electronicSerdesEnergyPjB * 1e-3;
  const totalLaserOpticalMw = s.laserPowerMw * totalRings;
  const laserElectricalWatts = (totalLaserOpticalMw / s.laserWpe) / 1000.0;
  const transceiverWatts = (opticalRawTbps * 1000.0) * s.opticalEoOeEnergyPjB * 1e-3;
  const opticalPowerWatts = laserElectricalWatts + transceiverWatts + heaterPowerWatts;

  const opticalEnergyPerBitPj = opticalEffectiveTbps > 0.001
    ? (opticalPowerWatts / (opticalEffectiveTbps * 1000.0)) * 1e3
    : 999.0;

  const energySavingsPercent = ((s.electronicSerdesEnergyPjB - opticalEnergyPerBitPj) / s.electronicSerdesEnergyPjB) * 100.0;

  // 5. Compute & Acceleration
  const electronicTops = 128 * 1024 * 1.8 * 1e-3; // ~236 TOPS
  const electronicEffTopsW = electronicTops / 550.0; // ~0.43 TOPS/W

  const macsPerCycle = 2.0 * s.mziMeshDim * s.mziMeshDim; // 8192
  const parallelWavelengths = Math.max(1, s.wdmChannels / 2);
  const photonicTops = (macsPerCycle * s.photonicClockGhz * parallelWavelengths * s.photonicCores) * 1e-3;
  const photonicPowerWatts = 45.0 + (s.photonicCores * 4.5);
  const photonicEffTopsW = photonicTops / photonicPowerWatts;

  let activeTbps, activePwr, activeTops, activeEff;
  if (s.mode === 0) {
    activeTbps = electricalTbps;
    activePwr = electricalPowerWatts;
    activeTops = electronicTops;
    activeEff = electronicEffTopsW;
  } else if (s.mode === 1) {
    activeTbps = opticalEffectiveTbps;
    activePwr = opticalPowerWatts;
    activeTops = electronicTops * 1.35;
    activeEff = electronicEffTopsW * 1.35;
  } else {
    activeTbps = opticalEffectiveTbps;
    activePwr = opticalPowerWatts;
    activeTops = photonicTops;
    activeEff = photonicEffTopsW;
  }

  const speedupX = activeTops / Math.max(0.1, electronicTops);

  metrics = {
    opticalRawTbps,
    opticalEffectiveTbps,
    electricalTbps,
    activeTbps,
    thermalWavelengthDriftNm,
    resonanceTransmissionFactor,
    heaterPowerWatts,
    opticalInsertionLossDb,
    receivedPowerDbm,
    photocurrentMa,
    opticalBer,
    opticalPowerWatts,
    electricalPowerWatts,
    opticalEnergyPerBitPj,
    energySavingsPercent,
    photonicTops,
    photonicEffTopsW,
    activeTops,
    activeEff,
    speedupX,
  };
}

// ─── UI Sync & Updates ───
function updateUI() {
  const s = simState;
  const m = metrics;

  // Controls Values
  document.getElementById('val-laser-power').textContent = `${s.laserPowerMw.toFixed(0)} mW`;
  document.getElementById('slider-laser-power').value = s.laserPowerMw;

  document.getElementById('val-wdm-channels').textContent = `${s.wdmChannels} λ`;
  document.getElementById('slider-wdm-channels').value = s.wdmChannels;

  document.getElementById('val-baud-rate').textContent = `${s.baudRateGbps.toFixed(0)} Gbps`;
  document.getElementById('slider-baud-rate').value = s.baudRateGbps;

  document.getElementById('val-temp').textContent = `${s.dieTempC.toFixed(1)} °C`;
  document.getElementById('slider-temp').value = s.dieTempC;

  document.getElementById('val-cores').textContent = `${s.photonicCores} Tiles`;
  document.getElementById('slider-cores').value = s.photonicCores;

  document.getElementById('val-matrix-dim').textContent = `${s.matrixDimN} x ${s.matrixDimN}`;
  document.getElementById('slider-matrix-dim').value = s.matrixDimN;

  document.getElementById('val-wg-loss').textContent = `${s.waveguideLossDbPerCm.toFixed(2)} dB/cm`;
  document.getElementById('slider-wg-loss').value = s.waveguideLossDbPerCm;

  const lockBtn = document.getElementById('btn-toggle-lock');
  const lockState = document.getElementById('val-lock-state');
  if (s.autoThermalLock) {
    lockBtn.className = 'toggle-btn on';
    lockBtn.textContent = '🔒 THERMAL AUTO-LOCK: ON';
    lockState.textContent = 'LOCKED (0.0 nm)';
    lockState.style.color = 'var(--emerald-glow)';
  } else {
    lockBtn.className = 'toggle-btn off';
    lockBtn.textContent = '⚠️ THERMAL AUTO-LOCK: OFF';
    lockState.textContent = `DETUNED (+${m.thermalWavelengthDriftNm.toFixed(2)} nm)`;
    lockState.style.color = 'var(--ruby-glow)';
  }

  // Badges
  const modeNames = [
    'ARCH: 1. ELECTRONIC GPU BASELINE',
    'ARCH: 2. CO-PACKAGED OPTICS (CPO HYBRID)',
    'ARCH: 3. INTEGRATED PHOTONIC AI CORE',
  ];
  document.getElementById('badge-active-arch').textContent = modeNames[s.mode];

  const photonBadge = document.getElementById('badge-photons-flowing');
  if (s.mode === 0) {
    photonBadge.textContent = 'COPPER SERDES: 64 DIFFERENTIAL CHANNELS';
    photonBadge.style.borderColor = 'var(--amber-glow)';
    photonBadge.style.color = 'var(--amber-glow)';
  } else {
    if (m.resonanceTransmissionFactor < 0.3) {
      photonBadge.textContent = '⚠️ OPTICAL LINK DETUNED: HIGH PACKET LOSS';
      photonBadge.style.borderColor = 'var(--ruby-glow)';
      photonBadge.style.color = 'var(--ruby-glow)';
    } else {
      photonBadge.textContent = `PHOTON STREAMS: ${s.wdmChannels * s.opticalPorts} DWDM CHANNELS ACTIVE`;
      photonBadge.style.borderColor = 'var(--emerald-glow)';
      photonBadge.style.color = 'var(--emerald-glow)';
    }
  }

  // Right Panel Telemetry
  document.getElementById('val-metric-bw').textContent = `${m.activeTbps.toFixed(1)} Tbps`;
  document.getElementById('bar-optical-bw').style.width = `${Math.min(100, (m.activeTbps / 25.0) * 100)}%`;
  document.getElementById('table-opt-bw').textContent = `${m.opticalEffectiveTbps.toFixed(2)} Tbps`;
  document.getElementById('table-elec-bw').textContent = `${m.electricalTbps.toFixed(2)} Tbps`;

  document.getElementById('val-metric-pj').textContent = `${m.opticalEnergyPerBitPj.toFixed(2)} pJ/b`;
  document.getElementById('bar-energy-sav').style.width = `${Math.max(5, Math.min(100, m.energySavingsPercent))}%`;
  document.getElementById('table-opt-pj').textContent = `${m.opticalEnergyPerBitPj.toFixed(2)} pJ/bit`;
  document.getElementById('table-elec-pj').textContent = `${s.electronicSerdesEnergyPjB.toFixed(2)} pJ/bit`;
  document.getElementById('table-energy-savings').textContent = `${m.energySavingsPercent.toFixed(1)}% SAVINGS`;

  document.getElementById('val-metric-tops').textContent = `${m.activeTops.toFixed(0)} TOPS`;
  document.getElementById('table-speedup').textContent = `${m.speedupX.toFixed(1)}x FASTER`;
  document.getElementById('table-opt-eff').textContent = `${m.photonicEffTopsW.toFixed(1)} TOPS/Watt`;
  document.getElementById('table-elec-eff').textContent = `${(electronicEff = 0.43).toFixed(2)} TOPS/Watt`;

  document.getElementById('val-metric-ber').textContent = m.opticalBer < 1e-12 ? 'BER < 10⁻¹²' : `BER: ${m.opticalBer.toExponential(1)}`;
  document.getElementById('table-loss-db').textContent = `${m.opticalInsertionLossDb.toFixed(1)} dB`;
  document.getElementById('table-rx-power').textContent = `${m.receivedPowerDbm > 0 ? '+' : ''}${m.receivedPowerDbm.toFixed(1)} dBm`;
  document.getElementById('table-mrr-lock').textContent = `${(m.resonanceTransmissionFactor * 100).toFixed(1)}% TRANSMISSION`;
  document.getElementById('table-heater-power').textContent = `${m.heaterPowerWatts.toFixed(2)} W`;

  // Mode Buttons Active State
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.mode) === s.mode);
  });
}

function logMessage(msg) {
  const logEl = document.getElementById('log-text');
  if (logEl) {
    logEl.textContent = msg;
  }
  console.log(`[Simulator]: ${msg}`);
}

// ─── 3D Canvas Visualizer ───
const canvas = document.getElementById('render-canvas');
const ctx = canvas.getContext('2d');

let cameraYaw = -0.4;
let cameraPitch = 0.55;
let cameraDistance = 380;
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

// Particle stream system
const particles = [];
for (let i = 0; i < 90; i++) {
  particles.push({
    progress: Math.random(),
    lane: Math.floor(Math.random() * 4),
    speed: 0.008 + Math.random() * 0.008,
    color: ['#00e5ff', '#bd34fe', '#05ffa1', '#ffb703'][i % 4],
  });
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Mouse Orbit Controls
canvas.addEventListener('mousedown', (e) => {
  isDragging = true;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
});

window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - lastMouseX;
  const dy = e.clientY - lastMouseY;
  cameraYaw += dx * 0.008;
  cameraPitch = Math.max(0.1, Math.min(1.4, cameraPitch + dy * 0.008));
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
});

window.addEventListener('mouseup', () => { isDragging = false; });

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  cameraDistance = Math.max(220, Math.min(650, cameraDistance + e.deltaY * 0.4));
});

// 3D Point Projection Helper
function project3D(x, y, z, cx, cy) {
  // Rotate by Yaw around Y
  const cosY = Math.cos(cameraYaw);
  const sinY = Math.sin(cameraYaw);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  // Rotate by Pitch around X
  const cosP = Math.cos(cameraPitch);
  const sinP = Math.sin(cameraPitch);
  const y2 = y * cosP - z1 * sinP;
  const z2 = y * sinP + z1 * cosP;

  // Perspective Division
  const depth = z2 + cameraDistance;
  const fov = 480;
  const scale = fov / Math.max(10, depth);

  return {
    px: cx + x1 * scale,
    py: cy - y2 * scale,
    depth: z2,
    scale,
  };
}

// Render 3D Scene
let animTime = 0;
function render3D() {
  const w = canvas.width / window.devicePixelRatio;
  const h = canvas.height / window.devicePixelRatio;
  const cx = w * 0.5;
  const cy = h * 0.52;

  ctx.clearRect(0, 0, w, h);

  // Background Grid Glow
  const grad = ctx.createRadialGradient(cx, cy, 50, cx, cy, 400);
  grad.addColorStop(0, 'rgba(18, 26, 42, 0.4)');
  grad.addColorStop(1, 'rgba(6, 8, 13, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // 1. Ceramic BGA Substrate Base (Isometric Octagon)
  const subRadius = 140;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 + Math.PI / 8;
    const pt = project3D(Math.cos(a) * subRadius, -8, Math.sin(a) * subRadius, cx, cy);
    if (i === 0) ctx.moveTo(pt.px, pt.py);
    else ctx.lineTo(pt.px, pt.py);
  }
  ctx.closePath();
  ctx.fillStyle = '#0a221e'; // Emerald-green package
  ctx.fill();
  ctx.strokeStyle = '#00e5ff33';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 2. Silicon Interposer Platform
  const intRadius = 105;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 + Math.PI / 8;
    const pt = project3D(Math.cos(a) * intRadius, 4, Math.sin(a) * intRadius, cx, cy);
    if (i === 0) ctx.moveTo(pt.px, pt.py);
    else ctx.lineTo(pt.px, pt.py);
  }
  ctx.closePath();
  ctx.fillStyle = '#1c222c';
  ctx.fill();
  ctx.strokeStyle = '#00e5ff66';
  ctx.stroke();

  // 3. Central Compute Architecture (Depends on Mode)
  if (simState.mode === 0) {
    // ── Mode 0: Monolithic Electronic GPU ──
    const gpuSize = 55;
    const tempRatio = Math.max(0, Math.min(1, (simState.dieTempC - 40) / 60));
    const rCol = Math.floor(40 + tempRatio * 215);
    const gCol = Math.floor(180 - tempRatio * 130);
    const bCol = Math.floor(255 - tempRatio * 230);

    // Main GPU Die
    const corners = [
      project3D(-gpuSize, 14, -gpuSize, cx, cy),
      project3D(gpuSize, 14, -gpuSize, cx, cy),
      project3D(gpuSize, 14, gpuSize, cx, cy),
      project3D(-gpuSize, 14, gpuSize, cx, cy),
    ];
    ctx.beginPath();
    ctx.moveTo(corners[0].px, corners[0].py);
    corners.forEach((c) => ctx.lineTo(c.px, c.py));
    ctx.closePath();
    ctx.fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
    ctx.shadowColor = `rgb(${rCol}, ${gCol}, ${bCol})`;
    ctx.shadowBlur = 18;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#fff';
    ctx.stroke();

    // SM Core Tiles (Grid of 6x6 cores)
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const sx = -40 + c * 20;
        const sz = -40 + r * 20;
        const smPt = project3D(sx, 16, sz, cx, cy);
        ctx.fillStyle = '#ffb703';
        ctx.fillRect(smPt.px - 3, smPt.py - 3, 6, 6);
      }
    }

    // Copper Traces Radiating Outward
    ctx.strokeStyle = 'rgba(255, 160, 40, 0.4)';
    ctx.lineWidth = 1.2;
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      const p1 = project3D(Math.cos(a) * 58, 8, Math.sin(a) * 58, cx, cy);
      const p2 = project3D(Math.cos(a) * 125, 2, Math.sin(a) * 125, cx, cy);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
    }
  } else if (simState.mode === 1) {
    // ── Mode 1: Co-Packaged Optics (CPO Hybrid) ──
    // Center GPU Die (Optimized)
    const gpuSize = 36;
    const corners = [
      project3D(-gpuSize, 14, -gpuSize, cx, cy),
      project3D(gpuSize, 14, -gpuSize, cx, cy),
      project3D(gpuSize, 14, gpuSize, cx, cy),
      project3D(-gpuSize, 14, gpuSize, cx, cy),
    ];
    ctx.beginPath();
    ctx.moveTo(corners[0].px, corners[0].py);
    corners.forEach((c) => ctx.lineTo(c.px, c.py));
    ctx.closePath();
    ctx.fillStyle = '#00a3ff';
    ctx.shadowColor = '#00e5ff';
    ctx.shadowBlur = 16;
    ctx.fill();
    ctx.shadowBlur = 0;

    // 4 Perimeter CPO Photonic Integrated Circuit (PIC) Tiles
    const picOffsets = [
      [75, 0], [-75, 0], [0, 75], [0, -75]
    ];
    picOffsets.forEach(([px, pz]) => {
      const pPt = project3D(px, 12, pz, cx, cy);
      ctx.fillStyle = '#05ffa1';
      ctx.shadowColor = '#05ffa1';
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(pPt.px, pPt.py, 14 * pPt.scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Optical Waveguide Link to Center GPU
      const cPt = project3D(px * 0.45, 12, pz * 0.45, cx, cy);
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(pPt.px, pPt.py);
      ctx.lineTo(cPt.px, cPt.py);
      ctx.stroke();

      // Optical Fiber Ribbon Off-Chip
      const fPt = project3D(px * 1.5, 6, pz * 1.5, cx, cy);
      ctx.strokeStyle = '#bd34fe';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(pPt.px, pPt.py);
      ctx.lineTo(fPt.px, fPt.py);
      ctx.stroke();
    });
  } else {
    // ── Mode 2: Integrated Photonic AI Core ──
    const coreCount = simState.photonicCores;
    for (let i = 0; i < coreCount; i++) {
      const a = (i / coreCount) * Math.PI * 2;
      const rad = 52;
      const pt = project3D(Math.cos(a) * rad, 16, Math.sin(a) * rad, cx, cy);

      ctx.fillStyle = '#bd34fe';
      ctx.shadowColor = '#bd34fe';
      ctx.shadowBlur = 20;
      ctx.beginPath();
      ctx.arc(pt.px, pt.py, 15 * pt.scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Internal MZI Lattice Node
      ctx.fillStyle = '#05ffa1';
      ctx.beginPath();
      ctx.arc(pt.px, pt.py, 5 * pt.scale, 0, Math.PI * 2);
      ctx.fill();

      // Optical Crossbar Waveguides
      const centerPt = project3D(0, 16, 0, cx, cy);
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(pt.px, pt.py);
      ctx.lineTo(centerPt.px, centerPt.py);
      ctx.stroke();
    }
  }

  // 4. HBM3 Memory Stacks (4 Cubes at Corners)
  const hbmCoords = [
    [-65, -65], [65, -65], [-65, 65], [65, 65]
  ];
  hbmCoords.forEach(([hx, hz]) => {
    const pt = project3D(hx, 18, hz, cx, cy);
    ctx.fillStyle = '#222733';
    ctx.strokeStyle = '#ffb703';
    ctx.lineWidth = 1.8;
    ctx.fillRect(pt.px - 10 * pt.scale, pt.py - 12 * pt.scale, 20 * pt.scale, 24 * pt.scale);
    ctx.strokeRect(pt.px - 10 * pt.scale, pt.py - 12 * pt.scale, 20 * pt.scale, 24 * pt.scale);
  });

  // 5. Silicon Photonic Waveguides & Microrings (MRRs)
  if (simState.mode !== 0) {
    const isTuned = metrics.resonanceTransmissionFactor > 0.4;
    const ringColors = ['#00e5ff', '#05ffa1', '#ffb703', '#bd34fe'];

    for (let r = 0; r < 4; r++) {
      const rx = -50 + r * 34;
      const rz = -35;
      const rPt = project3D(rx, 14, rz, cx, cy);

      ctx.beginPath();
      ctx.arc(rPt.px, rPt.py, 9 * rPt.scale, 0, Math.PI * 2);
      ctx.strokeStyle = ringColors[r];
      ctx.lineWidth = 2.5;

      if (isTuned) {
        ctx.shadowColor = ringColors[r];
        ctx.shadowBlur = 14;
        ctx.stroke();
        ctx.shadowBlur = 0;
      } else {
        // Detuned MRR warning
        ctx.strokeStyle = '#ff2a6d';
        ctx.shadowColor = '#ff2a6d';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
  }

  // 6. Dynamic Particle Stream (Photons vs Electrons)
  particles.forEach((p) => {
    p.progress = (p.progress + p.speed) % 1.0;

    let px, py, pz;
    if (simState.mode === 0) {
      // Jittery electrical sparks on GPU
      const a = (p.lane / 4) * Math.PI * 2;
      px = Math.cos(a) * (20 + p.progress * 85);
      pz = Math.sin(a) * (20 + p.progress * 85);
      py = 14 + Math.sin(animTime * 12 + p.lane) * 4;

      const pt = project3D(px, py, pz, cx, cy);
      ctx.fillStyle = '#ff781e';
      ctx.shadowColor = '#ff781e';
      ctx.shadowBlur = 8;
      ctx.fillRect(pt.px - 2, pt.py - 2, 4, 4);
      ctx.shadowBlur = 0;
    } else {
      // Streaming photons along waveguides
      const isDetuned = metrics.resonanceTransmissionFactor < 0.4;
      if (isDetuned && p.progress > 0.4 && p.progress < 0.7) {
        // Red photon scatter burst
        px = -50 + (p.progress * 80) + (Math.random() - 0.5) * 20;
        pz = -35 + (Math.random() - 0.5) * 20;
        py = 14 + (Math.random() - 0.5) * 12;

        const pt = project3D(px, py, pz, cx, cy);
        ctx.fillStyle = '#ff2a6d';
        ctx.shadowColor = '#ff2a6d';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        px = -65 + p.progress * 130;
        pz = -35;
        py = 14;

        const pt = project3D(px, py, pz, cx, cy);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, 3.0 * pt.scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  });

  animTime += 0.016;

  // Run Demo Tour if active
  if (simState.demoTourActive) {
    runDemoTour();
  }

  requestAnimationFrame(render3D);
}

// ─── Automated Demo Tour ───
function runDemoTour() {
  simState.demoTimer += 0.016;
  cameraYaw += 0.005; // Gentle cinematic orbit

  if (simState.demoStage === 0 && simState.demoTimer > 2.5) {
    simState.demoStage = 1;
    simState.mode = 1; // Switch to CPO
    logMessage('DEMO TOUR [1/4]: Switched to Co-Packaged Optics (CPO) — 14.3 Tbps Bandwidth Activated!');
    recalculatePhysics();
    updateUI();
  } else if (simState.demoStage === 1 && simState.demoTimer > 5.5) {
    simState.demoStage = 2;
    simState.wdmChannels = 32;
    logMessage('DEMO TOUR [2/4]: Scaling to 32 WDM Channels — 28.6 Tbps Optical Throughput!');
    recalculatePhysics();
    updateUI();
  } else if (simState.demoStage === 2 && simState.demoTimer > 8.5) {
    simState.demoStage = 3;
    simState.autoThermalLock = false;
    simState.dieTempC = 85.0;
    logMessage('DEMO TOUR [3/4]: Thermal Lock Disabled & Die Heated to 85°C — Watch Microrings Detune and Red Photon Loss!');
    recalculatePhysics();
    updateUI();
  } else if (simState.demoStage === 3 && simState.demoTimer > 12.0) {
    simState.demoStage = 4;
    simState.autoThermalLock = true;
    logMessage('DEMO TOUR [4/4]: Closed-Loop Thermal Lock Re-engaged — Micro-heaters Lock Resonance back to 100%!');
    recalculatePhysics();
    updateUI();
  } else if (simState.demoStage === 4 && simState.demoTimer > 15.0) {
    simState.demoStage = 5;
    simState.mode = 2; // Photonic AI Core
    logMessage('DEMO TOUR COMPLETE: Integrated Photonic AI Core Activated — 5,243 TOPS at 64.7 TOPS/Watt!');
    recalculatePhysics();
    updateUI();
    simState.demoTourActive = false;
    document.getElementById('btn-demo-tour').classList.remove('active');
  }
}

// ─── Event Listeners Setup ───
function setupEvents() {
  // Mode Buttons
  document.getElementById('btn-mode-gpu').addEventListener('click', () => {
    simState.mode = 0;
    logMessage('Architecture: Switched to Electronic GPU Baseline (Copper SerDes)');
    recalculatePhysics();
    updateUI();
  });

  document.getElementById('btn-mode-cpo').addEventListener('click', () => {
    simState.mode = 1;
    logMessage('Architecture: Switched to Co-Packaged Optics (CPO Hybrid)');
    recalculatePhysics();
    updateUI();
  });

  document.getElementById('btn-mode-accel').addEventListener('click', () => {
    simState.mode = 2;
    logMessage('Architecture: Switched to Integrated Photonic AI Core (MZI GEMM)');
    recalculatePhysics();
    updateUI();
  });

  // Demo Tour Button
  const demoBtn = document.getElementById('btn-demo-tour');
  demoBtn.addEventListener('click', () => {
    simState.demoTourActive = !simState.demoTourActive;
    if (simState.demoTourActive) {
      simState.demoStage = 0;
      simState.demoTimer = 0.0;
      simState.mode = 0;
      demoBtn.classList.add('active');
      logMessage('Starting Automated Feature Tour...');
    } else {
      demoBtn.classList.remove('active');
      logMessage('Automated Feature Tour stopped.');
    }
    recalculatePhysics();
    updateUI();
  });

  // Reset Button
  document.getElementById('btn-reset').addEventListener('click', () => {
    simState.mode = 1;
    simState.laserPowerMw = 20.0;
    simState.wdmChannels = 16;
    simState.baudRateGbps = 56.0;
    simState.dieTempC = 65.0;
    simState.autoThermalLock = true;
    simState.photonicCores = 8;
    simState.matrixDimN = 2048;
    simState.demoTourActive = false;
    demoBtn.classList.remove('active');
    logMessage('Simulator reset to defaults.');
    recalculatePhysics();
    updateUI();
  });

  // Slider & Stepper Bindings
  const bindControl = (sliderId, decId, incId, step, key, isMult = false) => {
    const slider = document.getElementById(sliderId);
    slider.addEventListener('input', (e) => {
      simState[key] = parseFloat(e.target.value);
      recalculatePhysics();
      updateUI();
    });

    document.getElementById(decId).addEventListener('click', () => {
      if (isMult) simState[key] = Math.max(slider.min, simState[key] / 2);
      else simState[key] = Math.max(slider.min, simState[key] - step);
      recalculatePhysics();
      updateUI();
    });

    document.getElementById(incId).addEventListener('click', () => {
      if (isMult) simState[key] = Math.min(slider.max, simState[key] * 2);
      else simState[key] = Math.min(slider.max, simState[key] + step);
      recalculatePhysics();
      updateUI();
    });
  };

  bindControl('slider-laser-power', 'btn-laser-dec', 'btn-laser-inc', 5, 'laserPowerMw');
  bindControl('slider-wdm-channels', 'btn-wdm-dec', 'btn-wdm-inc', 1, 'wdmChannels', true);
  bindControl('slider-baud-rate', 'btn-baud-dec', 'btn-baud-inc', 14, 'baudRateGbps');
  bindControl('slider-temp', 'btn-temp-dec', 'btn-temp-inc', 5, 'dieTempC');
  bindControl('slider-cores', 'btn-cores-dec', 'btn-cores-inc', 2, 'photonicCores');
  bindControl('slider-matrix-dim', 'btn-matrix-dec', 'btn-matrix-inc', 512, 'matrixDimN', true);
  bindControl('slider-wg-loss', 'btn-loss-dec', 'btn-loss-inc', 0.25, 'waveguideLossDbPerCm');

  // Toggle Thermal Lock Button
  document.getElementById('btn-toggle-lock').addEventListener('click', () => {
    simState.autoThermalLock = !simState.autoThermalLock;
    logMessage(`Closed-Loop Thermal Lock: ${simState.autoThermalLock ? 'ON' : 'OFF'}`);
    recalculatePhysics();
    updateUI();
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === '1') { simState.mode = 0; recalculatePhysics(); updateUI(); }
    if (e.key === '2') { simState.mode = 1; recalculatePhysics(); updateUI(); }
    if (e.key === '3') { simState.mode = 2; recalculatePhysics(); updateUI(); }
    if (e.key.toLowerCase() === 'h') {
      simState.autoThermalLock = !simState.autoThermalLock;
      recalculatePhysics();
      updateUI();
    }
  });

  // Check URL params for auto demo
  if (window.location.search.includes('demo=true')) {
    demoBtn.click();
  }
}

// ─── Initialization ───
recalculatePhysics();
setupEvents();
updateUI();
render3D();
