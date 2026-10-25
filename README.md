# Metatopia Non-Euclidean Game Engine

A game engine written in Rust that treats space as a manifold, enabling seamless navigation through curved geometries, portals, and impossible spaces. Unlike traditional engines that assume flat Euclidean geometry, Metatopia allows creators to build worlds where space itself bends, loops, and stitches together in coherent yet mind-bending ways.

## 🌌 Key Features

### Manifold-Based World Representation
- **Local Charts**: Spaces are represented as manifolds with local coordinate charts
- **Multiple Geometries**: Support for Euclidean, Spherical, Hyperbolic, and Custom geometries
- **Seamless Transitions**: Automatic coordinate transformations between different geometric regions

### Portal System
- **Zero-Length Tunnels**: Connect distant spaces instantaneously
- **Geometry Preservation**: Portals maintain proper orientation through parallel transport
- **Bidirectional Connections**: Two-way portals with consistent physics

### GPU Rendering
- **Physically Based Rendering (PBR)**: Cook-Torrance BRDF with metallic/roughness workflow
- **Ray Marching**: Fullscreen signed-distance-field rendering for fractals and theater environments
- **Geometry-Specific Shaders**: Specialized WGSL shaders per geometry type (`non_euclidean.wgsl`, `pest_control.wgsl`, `fractal_explorer.wgsl`, `vr_theater.wgsl`)
- **Post-Processing**: Film grain, vignette, ACES tonemapping, procedural noise
- **IQ Cosine Palettes**: Multiple mathematically-generated color palettes

### Procedural Audio
- **Geometry-Aware Drones**: Ambient sine-wave layers that shift with space type (Euclidean, Hyperbolic, Spherical)
- **Event-Driven Sound Effects**: Bounce thuds, orb chimes, portal sweeps, quantum zaps
- **Rate-Limited Playback**: Prevents audio saturation during rapid events
- **Powered by rodio**: Cross-platform audio via `cpal` backend

### Non-Euclidean Physics
- **Curved Space Navigation**: Movement follows geodesics
- **Geometry-Dependent Parameters**: Gravity, restitution, drag, and push force vary by space type
- **Non-Deterministic Outcomes**: Quantum kicks with pseudo-random impulse injection
- **Sphere ↔ Wall, Floor, Player, Orb**: Full collision system with space-aware restitution

### Entity Component System
- **hecs-based ECS**: Lightweight archetypal ECS for game entities
- **Manifold-Aware Transforms**: Position and orientation tied to chart coordinates
- **System Pipeline**: Logic that operates across component sets

### Zero-Boilerplate Quickstart
- **GameApp Trait**: Implement one trait to create a full 3D game — no wgpu knowledge needed
- **Built-in FPS Camera**: WASD + mouse look with configurable speed and sensitivity
- **Collision Detection**: AABB, Sphere, and Ray colliders with spatial queries
- **Audio Engine**: Fire-and-forget SFX, looping music, preloading, volume control
- **Mouse + Crosshair**: Raw mouse delta, aim direction, built-in crosshair overlay
- **Mesh Builders**: Cube, sphere, and floor primitives with PBR vertex format

### Scoring & HUD
- **ScoreTracker**: Combo multiplier with decay timer, high score tracking
- **HudData**: GPU-uploadable struct for procedural 7-segment HUD overlays

## 🎮 Examples

### 1. Non-Euclidean Physics Demo (`basic_game`)
Interactive physics sandbox across three connected geometries with GPU-rendered PBR environment, procedural audio, and real-time portal transitions.

- **3 Geometries**: Euclidean (flat), Hyperbolic (K<0, divergent), Spherical (K>0, convergent)
- **Physics**: Gravity, bounce, geodesic curvature forces, quantum kicks
- **Objects**: Pushable sphere + 4 collectible orbs per space
- **Audio**: Ambient drones, bounce thuds, chimes, portal whooshes, quantum zaps

```bash
cargo run --release --example basic_game
```

**Controls**: WASD Move · Mouse Look · Space/Shift Up/Down · R Reset · ESC Quit

---

### 2. Pest Control Simulator (`pest_control_sim`)
First-person pest extermination with PBR-rendered environments, 5 pest types with AI behaviors, combo scoring, and 6 themed locations.

- **Pests**: Cockroach, Ant, Spider (leaps), Rat (ambush), Wasp (dive-bombs)
- **Tools**: Spray Bottle (fast, close range) · Vacuum Gun (powerful, limited ammo)
- **Scoring**: Combo multiplier up to x5 for rapid kills
- **Locations**: Kitchen, Bathroom, Basement, Garage, Attic, Garden

```bash
cargo run --release --example pest_control_sim
```

**Controls**: WASD Move · Mouse Aim · Left Click Fire · 1/2 Switch Tool · R Reload

---

### 3. CYBERSHOCK: Neon Grid Arena (`cyber_arena_fps`)
Fast-paced retro-futuristic 3D Cyberpunk FPS with 3 distinct sci-fi weapons, intelligent enemy swarms, quantum boss fight, procedural sound synthesis, and real-time GPU neon lighting.

- **3 Weapons**: Plasma Rifle (rapid laser bursts), Scatter Cannon (neon shotgun), Heavy Railgun (instant piercing beam)
- **Enemy AI**: Neon Drones (swarming), Cyber Tanks (armored bruisers), Phantom Snipers (laser snipers), Quantum Hex-Core Boss
- **Combat Mechanics**: Boost Dash (<kbd>Shift</kbd>), Jump (<kbd>Space</kbd>), Hitmarkers, Health/Shield gauges, Score combo chains, Powerups (Medkits, Shields, Ammo, Overcharge 2x Damage)
- **Visuals & Audio**: PBR cyber grid floor with player distance ripple, holographic HUD, procedural synth sound synthesis (10 synthesized SFX via Rodio)

```bash
cargo run --release --example cyber_arena_fps
```

**Controls**: WASD Move · Mouse Aim · Left Click Fire · 1/2/3/Q Switch Weapon · R Reload · Shift Dash · Space Jump · Enter Restart

---

### 4. Metatopia Studio — 3D Game Editor & Rhai Scripting Suite (`metatopia_editor`)
Full-featured 3D scene editor with embedded **Rhai scripting engine**, live entity inspector, infinite grid viewport, and real-time Play/Edit simulation modes.

- **Embedded Rhai Scripting**: Pure Rust dynamic scripting engine with live hot-reload (`R`), script presets (Pulsing Glow, Hover & Bob, Color Shift, Orbit), and console logging
- **Scene Outliner & Inspector**: Inspect and manipulate transforms ($X, Y, Z, \text{Yaw}, \text{Pitch}$), PBR materials, metallic/roughness, and emissive glow
- **Procedural Shapes**: Instant 3D primitive spawning (+Cube, +Sphere, +Cylinder, +Torus, +Capsule, +Particles)
- **Play / Edit Modes**: Toggle simulation with <kbd>Space</kbd> to test script logic and physics live!

```bash
cargo run --release --example metatopia_editor
```

**Controls**: WASD + Mouse Fly · Space Play/Edit · Tab Cycle Selection · 1–5 Spawn Objects · G Script Presets · R Hot-Reload · Arrow Keys Move · Q/E Rotate · X Delete

---

### 5. Mandelbulb Fractal Explorer (`fractal_explorer`)
Real-time 3D fractal rendered via GPU ray marching with orbit camera, 5 color palettes, adjustable power parameter, and audio-reactive pulsation.

- **Ray Marching**: 120-step march through Mandelbulb SDF
- **Lighting**: PBR with soft shadows, AO from march steps, Fresnel rim
- **Palettes**: Fire 🔥 · Ocean 🌊 · Nebula 🌌 · Earth 🌍 · Monochrome ⬛
- **Audio-Reactive**: Fractal power pulses with sine modulation

```bash
cargo run --release --example fractal_explorer
```

**Controls**: Mouse Drag Orbit · Scroll Zoom · 1–5 Palette · P/O Power · A Audio-Reactive

---

### 6. VR Netflix — Hyperbolic Cinema (`vr_netflix_hyperbolic`)
GPU-rendered non-Euclidean movie theater with 5 viewing spaces, procedural movie screens, and space-specific audio drones.

- **Hyperbolic Lobby** 🟣: 8 screens on Poincaré disk with hyperbolic 7-fold tiling floor
- **Spherical Dome** ⭐: Starfield cinema with nebula background
- **Escher Theater** 🔄: Impossible geometry with staircase-like screen arrangement
- **Personal Pocket** 🏠: Cozy curved room with main + side screen
- **Social Hub** 👥: Shared viewing space with 5 screens

```bash
cargo run --release --example vr_netflix_hyperbolic
```

---

### 7. Metatopia VR — Meta Quest 3S & PCVR Stereoscopic 3D (`metatopia_vr_quest`)
Full 360° Stereoscopic 3D Virtual Reality experience designed for the **Meta Quest 3S** (and PCVR headsets via Quest Link / Virtual Desktop).

- **Dual-Eye SBS 3D**: Side-by-Side Left & Right eye parallax with GPU split-viewport mapping
- **IPD Tuning**: In-game millimeter adjustment (55mm to 72mm) matching your physical Quest 3S lenses
- **Non-Euclidean VR**: Walk through glowing portals connecting Euclidean, Hyperbolic, and Spherical curved spaces
- **6-DoF Rig**: Real-time head tracking and 3D spatial particle bursts

```bash
cargo run --release --example metatopia_vr_quest
```

**Controls**: WASD + Mouse Fly · V Toggle VR Side-by-Side 3D · [ / ] Adjust IPD · F Cycle FOV (90° / 100° / 110°)

---

### 8. Metatopia Sandbox — Next-Gen Systems Master Showcase (`metatopia_sandbox`)
Interactive sandbox running all 7 next-generation engine subsystems in real-time.

- **Dynamic Atmosphere**: Moving Sun/Moon, Rayleigh/Mie color temperature shifts, and height fog
- **Destructible Voxel Terrain**: Mine craters (<kbd>Left Click</kbd>) and build crystal mountains (<kbd>Right Click</kbd>)
- **Raycast Cyber Rover**: 4-wheel spring-damper suspension physics vehicle with steering and drift
- **Two-Bone IK**: Animated procedural robotic arm with analytical inverse kinematics
- **Shader Graph**: Visual node-based procedural material generation in WGSL
- **Scene Persistence**: Real-time JSON scene saving (<kbd>F5</kbd>) and loading (<kbd>F9</kbd>)

```bash
cargo run --release --example metatopia_sandbox
```

**Controls**: C Drive Rover / Fly Camera · WASD Drive/Fly · Space Handbrake · Left/Right Click Voxel Terraforming · T Day/Night · F5/F9 Save/Load

---

### 9. Non-Euclidean Ray-Marched Portal, Klein Bottle $\mathbb{K}^3$ & Hyperbolic $\mathbb{H}^3$ Benchmark (`hyperbolic_portal_benchmark`)
Real-time WGSL ray-marched Riemannian manifold and multi-bounce portal benchmark featuring continuous curvature $K \in [-1.35, +1.35]$, exact 3D Poincaré ball gyrometric geodesics (`Metric::gyrometric_geodesic_point_3d`), 4th-order Runge-Kutta (`RK4`) Christoffel geodesic integration, volumetric geodesic particle streams, and a built-in 9-movement 128 BPM procedural stereo techno synthesizer (*"GEODESIC HORIZON"*).

- **5 Non-Euclidean Scenes**:
  1. **Impossible Non-Euclidean Corridor & 4-Chart Loop**: Seamless multi-chart portal loop (`Chart 0 -> 1 -> 2 -> 3 -> 0`) with local tangent-space Möbius transport
  2. **Hyperbolic $\mathbb{H}^3$ `{4,3,5}` Conformal Honeycomb Cathedral**: $C^1$-continuous hyperbolic space folding with conformal metric scaling $\lambda(x) = \frac{2}{1 + K\|x\|^2}$
  3. **Multi-Chart Curvature Bridge ($\mathbb{E}^3 \to \mathbb{H}^3 \to \mathbb{S}^3$)**: Continuous transition from flat Euclidean space to negatively curved hyperbolic space and positively curved spherical $\mathbb{S}^3$ star-vaults
  4. **16-Bounce Recursive Infinity Portal Tunnel**: High-depth recursive portal mirror stress test with exponential LOD border attenuation and full-3D neon spectrum overdrive
  5. **Non-Orientable Klein Bottle $\mathbb{K}^3$ & $\mathbb{T}^3$ Torus Chiral Labyrinth**: Orientation-reversing portal identification ($\det J = -1$, $x \mapsto -x$) that flips handedness (Electric Cyan $\leftrightarrow$ Solar Gold chiral double-helix pillars) on every portal bounce
- **Volumetric Geodesic Particle Swarm (`GeodesicParticleSwarm`)**: 1,024 CPU RK4-advected particles (`28.6M particle-steps/s`) paired with GPU volumetric geodesic orb trails (`sd_geodesic_swarm`)
- **Zero-Install WebGL2 + WebAudio Visualizer**: Open [`web_simulator/non_euclidean_portal.html`](web_simulator/non_euclidean_portal.html) directly in any modern browser
- **3 Native Execution Modes**: Interactive 60+ FPS WGPU Explorer, Headless CPU+GPU Benchmark Suite (`--bench`), and 3-Minute 128 BPM Audiovisual Videoclip Recorder (`--record <output.mp4>`)

```bash
# Interactive 60+ FPS Explorer
cargo run --release --example hyperbolic_portal_benchmark

# Headless CPU + GPU Geodesic, Klein Bottle & Multi-Bounce Benchmark Suite
cargo run --release --example hyperbolic_portal_benchmark -- --bench

# Record Full 3:00 (96-Bar @ 128 BPM) Techno Videoclip (Original Master + <20 MB Compressed)
cargo run --release --example hyperbolic_portal_benchmark -- --record hyperbolic_portal_gameplay.mp4
```

**Controls**: WASD + Mouse Fly · 1–5 Switch Scene (`5` = Klein Bottle $\mathbb{K}^3$) · K/J Adjust Curvature $\kappa$ · P/O Portal Bounce Depth (0–16) · G Toggle Geodesic Bending · H Cycle PBR / Step-Cost Heatmap / Full-3D Neon Overdrive · Tab Toggle Auto-Tour

---

### 10. GPU-to-Silicon Photonics Optical Interconnect Simulator (`photonics_sim`)
Interactive 3D simulator for WDM microring resonators, Mach-Zehnder interferometers (MZI), waveguide propagation loss, thermal tuning, and eye-diagram BER analysis.

```bash
cargo run --release --example photonics_sim
```

---

### 11. Basic Graphics (`basic_graphics`)
Minimal WGPU rendering setup demonstrating the engine's shader pipeline.

```bash
cargo run --release --example basic_graphics
```

### 12. Simple Demo (`simple_demo`)
Console-based demonstration of the manifold, ECS, and portal systems without GPU rendering.

```bash
cargo run --release --example simple_demo
```

### 13. My First Game — Quickstart Template (`my_first_game`)
Minimal ~130-line game demonstrating the zero-boilerplate `quickstart` module. Walk over colored cubes to collect them.

```bash
cargo run --release --example my_first_game
```

**Controls**: WASD Move · Mouse Look · Walk over cubes to collect · ESC Quit

## 🏗️ Architecture

### Source Tree

```
src/
├── lib.rs              # Prelude and module exports
├── core/mod.rs         # Engine, GameState, EngineConfig
├── quickstart.rs       # Zero-boilerplate game launcher (GameApp trait)
├── collision.rs        # AABB, Sphere, Ray, CollisionWorld
├── audio.rs            # Audio engine (SFX, music, preloading)
├── scoring/mod.rs      # ScoreTracker, combos, HudData
├── manifold/
│   ├── mod.rs          # Manifold container & multi-chart topology
│   ├── chart.rs        # Chart, ChartId, GeometryType
│   ├── portal.rs       # Portal connections, tangent-space transport & traversal
│   ├── metric.rs       # Riemannian metric tensor, Poincaré ball, Möbius addition, Christoffel symbols
│   ├── geodesic.rs     # Shortest-path & RK4 geodesic integration
│   └── raycast.rs      # Non-Euclidean geodesic ray-marching & multi-bounce portal benchmark
├── photonics/          # Silicon photonics WDM, microring, MZI & link budget engine
├── graphics/
│   ├── mod.rs          # Renderer
│   ├── camera.rs       # Non-Euclidean camera
│   ├── mesh.rs         # Vertex data
│   ├── shader.rs       # Shader management
│   └── texture.rs      # Texture loading
├── ecs/mod.rs          # World, Entity, components, systems
├── input/mod.rs        # Keyboard, mouse, gamepad input
├── time/mod.rs         # Frame timing, fixed timestep
├── math/mod.rs         # Mat4, vector math utilities
├── resources/mod.rs    # Asset management
└── window/mod.rs       # Window creation, event loop

shaders/
├── hyperbolic_portal_benchmark.wgsl # 16-bounce Non-Euclidean portal & H3 {4,3,5} honeycomb ray marcher
├── template_game.wgsl  # Starter shader for quickstart games
├── non_euclidean.wgsl  # PBR + portal rooms + orbs + noise
├── pest_control.wgsl   # FPS environment + PBR + film grain
├── fractal_explorer.wgsl  # Ray-marched Mandelbulb + 5 palettes
├── vr_theater.wgsl     # Hyperbolic theater + Poincaré disk
└── basic.wgsl          # Minimal vertex/fragment shader

examples/
├── hyperbolic_portal_benchmark.rs # Non-Euclidean ray-marched portal benchmark & 128 BPM audiovisual synth
├── photonics_sim.rs    # 3D GPU-to-silicon photonics optical interconnect simulator
├── metatopia_sandbox.rs # 7 next-gen engine subsystems showcase
├── metatopia_editor.rs # 3D studio editor with live Rhai scripting
├── cyber_arena_fps.rs  # Cyberpunk 3D arena FPS
├── metatopia_vr_quest.rs # Stereoscopic 3D VR experience for Meta Quest 3S
├── my_first_game.rs    # Quickstart template (~130 lines)
├── basic_game.rs       # Non-Euclidean physics sandbox
├── pest_control_sim.rs # FPS pest control game
├── fractal_explorer.rs # 3D fractal explorer
├── vr_netflix_hyperbolic.rs  # Hyperbolic cinema
├── basic_graphics.rs   # Minimal WGPU demo
└── simple_demo.rs      # Console manifold demo
```

### Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `wgpu` | 0.19 | GPU rendering (Vulkan/Metal/DX12) |
| `winit` | 0.29 | Window management and input events |
| `rodio` | 0.17 | Cross-platform audio playback |
| `cgmath` | 0.18 | Linear algebra (vectors, matrices, quaternions) |
| `hecs` | 0.10 | Archetypal Entity Component System |
| `bytemuck` | 1.14 | Safe GPU uniform casting |
| `pollster` | 0.3 | Async runtime for WGPU initialization |
| `gilrs` | 0.10 | Gamepad support |
| `image` | 0.24 | Texture loading |
| `rand` | 0.8 | Random number generation |
| `serde` + `ron` | 1.0 / 0.8 | Configuration serialization |
| `log` + `env_logger` | 0.4 / 0.11 | Logging |

## 🚀 Getting Started

### Prerequisites
- Rust 1.75 or later
- GPU with Vulkan, Metal, or DX12 support

### Build & Run

```bash
# Clone
git clone https://github.com/Metatopia-Dev/metatopia-game-engine.git
cd metatopia-game-engine

# Build
cargo build --release

# Run the quickstart template
cargo run --release --example my_first_game

# Or run the non-Euclidean demo
cargo run --release --example basic_game
```

## 📖 Usage

### Quickstart (Recommended for New Developers)

See [GAME_DEV_GUIDE.md](GAME_DEV_GUIDE.md) for the complete guide. Minimal example:

```rust
use metatopia_engine::quickstart::*;

struct MyGame { score: u32 }

impl GameApp for MyGame {
    fn update(&mut self, ctx: &mut UpdateCtx) {
        ctx.default_camera_movement();
        if ctx.key_pressed(VirtualKey::Escape) { ctx.quit(); }

        if ctx.mouse_pressed(winit::event::MouseButton::Left) {
            if let Some(hit) = ctx.raycast(50.0) {
                self.score += 10;
                ctx.audio.play_sfx("sounds/hit.wav");
            }
        }
    }
}

fn main() { run_game(MyGame { score: 0 }); }
```

### Advanced: Creating a Non-Euclidean World

```rust
use metatopia_engine::prelude::*;

// Create a manifold with different geometries
let mut manifold = Manifold::new();

// Add a hyperbolic space
let hyperbolic = manifold.add_chart(GeometryType::Hyperbolic);

// Add a spherical space
let spherical = manifold.add_chart(GeometryType::Spherical);

// Connect them with a portal
manifold.create_portal(
    ChartId(0),  // From Euclidean (default chart)
    hyperbolic,  // To Hyperbolic
    Point3::new(5.0, 0.0, 0.0),  // Entry point
    Point3::new(0.0, 0.0, 0.0),  // Exit point
    Mat4::from_scale(1.0),        // Transformation
).unwrap();
```

### Implementing a Game State

```rust
struct MyGame {
    manifold: Manifold,
    camera: Camera,
    world: World,
}

impl GameState for MyGame {
    fn on_init(&mut self, engine: &mut Engine) {
        // Initialize your game
    }

    fn on_update(&mut self, engine: &mut Engine, dt: f32) {
        self.camera.update(&self.manifold);
        self.world.update(dt);
    }

    fn on_render(&mut self, _engine: &mut Engine, renderer: &mut Renderer) {
        renderer.clear(0.1, 0.1, 0.2, 1.0);
    }

    fn on_cleanup(&mut self, _engine: &mut Engine) {
        // Cleanup
    }
}
```

## 🎯 Use Cases

### Game Development
- **Puzzle Games**: Impossible spaces and mind-bending navigation
- **Horror Games**: Disorienting non-Euclidean corridors
- **Exploration**: Infinite worlds that loop back on themselves
- **FPS**: Geometry-aware combat in curved arenas

### VR/AR Applications
- **Virtual Theaters**: Infinite screens on hyperbolic surfaces
- **Architectural Visualization**: Buildings bigger on the inside
- **Education**: Interactive non-Euclidean geometry lessons

### Research & Visualization
- **Physics Simulations**: Behavior in curved spacetime
- **Fractal Exploration**: Real-time 3D fractal rendering
- **Data Visualization**: High-dimensional data in navigable 3D

## 🔧 Configuration

### Engine Configuration
```rust
let config = EngineConfig {
    title: "My Non-Euclidean Game".to_string(),
    width: 1920,
    height: 1080,
    vsync: true,
    target_fps: Some(60),
    resizable: true,
};
```

### Geometry Types
| Type | Curvature | Behavior |
|---|---|---|
| `Euclidean` | $\kappa = 0$ | Standard flat geometry |
| `Spherical` | $\kappa > 0$ | Convergent geodesics, objects pulled inward |
| `Hyperbolic` | $\kappa < 0$ | Divergent geodesics, exact 3D Poincaré & Beltrami-Klein gyrometric geodesics |
| `Torus` | $\kappa = 0$ (compact $\mathbb{T}^3$) | Periodic boundary wrapping across $(x, y, z) \in [-L/2, L/2]^3$ |
| `KleinBottle` | $\kappa = 0$ (non-orientable $\mathbb{K}^3$) | Orientation-reversing boundary ($z \pm L/2 \implies x \mapsto -x$, $\det J = -1$) |
| `Custom` | User-defined | Custom metric tensor & 4th-order Runge-Kutta (`RK4`) Christoffel geodesic solver |

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Higher-genus hyperbolic surfaces ($\Sigma_g$) and Seifert fibered spaces
- Performance optimizations
- More example games and demos
- Documentation improvements
- Bug fixes

## 📚 References

### Non-Euclidean Geometry
- "Experiencing Hyperbolic Space" — Vi Hart and Henry Segerman
- "Non-Euclidean Geometry in Games" — Various GDC talks
- "Curved Spaces" — Jeffrey Weeks

### Technical Papers
- "Portal Rendering and Visibility"
- "Geodesic Ray Tracing in Curved Spaces"
- "Parallel Transport in Computer Graphics"

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- The Rust gamedev community
- WGPU team for the excellent graphics API
- rodio/cpal teams for cross-platform audio
- Researchers in non-Euclidean rendering
- Games like Antichamber and Hyperbolica for inspiration

## 🚧 Roadmap

- [x] Procedural audio with geometry-aware drones
- [x] PBR rendering with Cook-Torrance BRDF
- [x] Ray marching (Mandelbulb fractal explorer & 16-bounce Non-Euclidean portals)
- [x] Non-deterministic physics (quantum kicks)
- [x] Multiple themed environments per demo
- [x] Zero-boilerplate quickstart system
- [x] Collision detection (AABB, sphere, ray, spatial queries)
- [x] Audio engine integration in quickstart
- [x] Scoring system with combo mechanics
- [x] Mouse delta tracking + FPS crosshair
- [x] 103+ automated unit tests
- [x] Torus ($\mathbb{T}^3$) and Non-Orientable Klein Bottle ($\mathbb{K}^3$) manifolds with chirality inversion
- [x] Exact 3D Poincaré Ball gyrometric geodesics & RK4 Christoffel ODE integration
- [x] 3D Geodesic Particle Swarm (`GeodesicParticleSwarm`) + volumetric GPU particle trails
- [x] Standalone zero-install WebGL2 + WebAudio browser visualizer (`web_simulator/non_euclidean_portal.html`)
- [ ] Dynamic mesh updates (instanced drawing)
- [ ] Networked multiplayer support
- [ ] Visual scripting for portal creation
- [ ] Integration plugins for other engines

---

**Ready to bend reality?** Start building impossible worlds with Metatopia today! 🌀