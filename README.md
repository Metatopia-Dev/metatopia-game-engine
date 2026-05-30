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

### 3. Mandelbulb Fractal Explorer (`fractal_explorer`)
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

### 4. VR Netflix — Hyperbolic Cinema (`vr_netflix_hyperbolic`)
GPU-rendered non-Euclidean movie theater with 5 viewing spaces, procedural movie screens, and space-specific audio drones.

- **Hyperbolic Lobby** 🟣: 8 screens on Poincaré disk with hyperbolic 7-fold tiling floor
- **Spherical Dome** ⭐: Starfield cinema with nebula background
- **Escher Theater** 🔄: Impossible geometry with staircase-like screen arrangement
- **Personal Pocket** 🏠: Cozy curved room with main + side screen
- **Social Hub** 👥: Shared viewing space with 5 screens

```bash
cargo run --release --example vr_netflix_hyperbolic
```

**Controls**: WASD Move · Mouse Look · 1–5 Switch Space · Tab Cycle Screen · Click Play/Pause

---

### 5. Basic Graphics (`basic_graphics`)
Minimal WGPU rendering setup demonstrating the engine's shader pipeline.

```bash
cargo run --release --example basic_graphics
```

### 6. Simple Demo (`simple_demo`)
Console-based demonstration of the manifold, ECS, and portal systems without GPU rendering.

```bash
cargo run --release --example simple_demo
```

## 🏗️ Architecture

### Source Tree

```
src/
├── lib.rs              # Prelude and module exports
├── core/mod.rs         # Engine, GameState, EngineConfig
├── manifold/
│   ├── mod.rs          # Manifold container
│   ├── chart.rs        # Chart, ChartId, GeometryType
│   ├── portal.rs       # Portal connections + ray intersection
│   ├── metric.rs       # Metric tensor definitions
│   └── geodesic.rs     # Shortest-path computation
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
├── non_euclidean.wgsl  # PBR + portal rooms + orbs + noise
├── pest_control.wgsl   # FPS environment + PBR + film grain
├── fractal_explorer.wgsl  # Ray-marched Mandelbulb + 5 palettes
├── vr_theater.wgsl     # Hyperbolic theater + Poincaré disk
└── basic.wgsl          # Minimal vertex/fragment shader

examples/
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

# Run an example
cargo run --release --example basic_game
```

## 📖 Usage

### Creating a Non-Euclidean World

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
| `Euclidean` | κ = 0 | Standard flat geometry |
| `Spherical` | κ > 0 | Convergent geodesics, objects pulled inward |
| `Hyperbolic` | κ < 0 | Divergent geodesics, objects pushed outward |
| `Custom` | User-defined | Custom metric tensor |

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional geometry types (Torus, Klein bottle)
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
- [x] Ray marching (Mandelbulb fractal explorer)
- [x] Non-deterministic physics (quantum kicks)
- [x] Multiple themed environments per demo
- [ ] Networked multiplayer support
- [ ] More geometry types (Torus, Klein bottle)
- [ ] Visual scripting for portal creation
- [ ] Performance optimizations for mobile/web
- [ ] Integration plugins for other engines

---

**Ready to bend reality?** Start building impossible worlds with Metatopia today! 🌀