# Metatopia Non-Euclidean Game Engine

A groundbreaking game engine written in Rust that treats space as a manifold, enabling seamless navigation through curved geometries, portals, and impossible spaces. Unlike traditional engines that assume flat Euclidean geometry, Metatopia allows creators to build worlds where space itself bends, loops, and stitches together in coherent yet mind-bending ways.

## 🌌 Key Features

### Manifold-Based World Representation
- **Local Charts**: Spaces are represented as manifolds with local coordinate charts
- **Multiple Geometries**: Support for Euclidean, Spherical, Hyperbolic, and Custom geometries
- **Seamless Transitions**: Automatic coordinate transformations between different geometric regions

### Portal System
- **Zero-Length Tunnels**: Connect distant spaces instantaneously
- **Geometry Preservation**: Portals maintain proper orientation through parallel transport
- **Bidirectional Connections**: Two-way portals with consistent physics

### Metric-Aware Rendering
- **Geodesic Ray Casting**: Light and vision follow the curvature of space
- **Geometry-Specific Shaders**: Specialized shaders for each type of geometry
- **Adaptive Projection**: Camera automatically adjusts FOV based on local geometry

### Non-Euclidean Physics
- **Curved Space Navigation**: Movement follows geodesics
- **Parallel Transport**: Orientation preservation when moving through curved space
- **Metric-Aware Collision**: Collision detection that respects the local geometry

## 🎮 Example Applications

### 1. Basic Non-Euclidean Demo
Explore connected spaces with different geometries:
- Euclidean rooms
- Hyperbolic spaces (Poincaré disk)
- Spherical spaces
- Seamless portal transitions

```bash
cargo run --example basic_game
```

### 2. Pest Control Simulator
A practical game in Euclidean space showcasing traditional game mechanics:
- First-person pest extermination gameplay
- Multiple pest types with AI behaviors
- Various tools and strategies
- Level progression system

```bash
cargo run --example pest_control_sim
```

### 3. VR Netflix in Hyperbolic Space
Experience infinite movie theaters without overlap:
- **Hyperbolic Lobby**: Infinite screens in Poincaré disk
- **Spherical Dome**: 360° viewing experience
- **Escher Theater**: Impossible geometry viewing spaces
- **Personal Pocket**: Your own curved dimension
- **Social Hub**: Watch parties in non-Euclidean space

```bash
cargo run --example vr_netflix_hyperbolic
```

## 🏗️ Architecture

### Core Components

#### Manifold System (`src/manifold/`)
- `Chart`: Local coordinate patches with specific geometries
- `Portal`: Connections between charts with transformations
- `Metric`: Defines the geometry and curvature of space
- `Geodesic`: Shortest paths through curved space

#### Graphics (`src/graphics/`)
- `Renderer`: WGPU-based rendering with metric awareness
- `Shader`: Geometry-specific shader programs
- `Camera`: Non-Euclidean camera with parallel transport
- `Mesh`: Vertex data for curved space rendering

#### Entity Component System (`src/ecs/`)
- `World`: Container for all entities and components
- `Transform`: Position and orientation in manifold space
- `System`: Logic that operates on components

#### Input & Time (`src/input/`, `src/time/`)
- Comprehensive input handling for keyboard, mouse, and gamepad
- Fixed timestep for physics
- Frame timing and interpolation

## 🚀 Getting Started

### Prerequisites
- Rust 1.70 or later
- GPU with Vulkan, Metal, or DX12 support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/metatopia-game-engine.git
cd metatopia-game-engine
```

2. Build the engine:
```bash
cargo build --release
```

3. Run an example:
```bash
cargo run --example basic_game --release
```

## 📖 Usage

### Creating a Simple Non-Euclidean World

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
    ChartId(0),  // From Euclidean
    hyperbolic,  // To Hyperbolic
    Point3::new(5.0, 0.0, 0.0),  // Entry point
    Point3::new(0.0, 0.0, 0.0),  // Exit point
    Mat4::identity(),  // Transformation
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
        // Update game logic
        self.camera.update(&self.manifold);
        self.world.update(dt);
    }
    
    fn on_render(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        // Render your world
        renderer.clear(0.1, 0.1, 0.2, 1.0);
    }
    
    fn on_cleanup(&mut self, engine: &mut Engine) {
        // Cleanup
    }
}
```

## 🎯 Use Cases

### Game Development
- **Puzzle Games**: Impossible spaces and mind-bending navigation puzzles
- **Horror Games**: Disorienting non-Euclidean corridors
- **Exploration**: Infinite worlds that loop back on themselves
- **Strategy**: Hyperbolic maps with exponentially more space

### VR/AR Applications
- **Virtual Theaters**: Infinite screens without overlap
- **Architectural Visualization**: Buildings that are bigger on the inside
- **Education**: Interactive lessons on non-Euclidean geometry
- **Social Spaces**: Meeting rooms that can hold unlimited people

### Research & Simulation
- **Physics Simulations**: Studying behavior in curved spacetime
- **Navigation Algorithms**: Testing pathfinding in non-Euclidean spaces
- **Visualization**: Representing high-dimensional data in navigable 3D

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
- `Euclidean`: Standard flat geometry (κ = 0)
- `Spherical`: Positive curvature (κ > 0)
- `Hyperbolic`: Negative curvature (κ < 0)
- `Custom`: User-defined metric tensor

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional geometry types
- Performance optimizations
- More example games
- Documentation improvements
- Bug fixes

Please read our contributing guidelines before submitting PRs.

## 📚 References

### Non-Euclidean Geometry
- "Experiencing Hyperbolic Space" - Vi Hart and Henry Segerman
- "Non-Euclidean Geometry in Games" - Various GDC talks
- "Curved Spaces" - Jeffrey Weeks

### Technical Papers
- "Portal Rendering and Visibility"
- "Geodesic Ray Tracing in Curved Spaces"
- "Parallel Transport in Computer Graphics"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The Rust gamedev community
- WGPU team for the excellent graphics API
- Researchers in non-Euclidean rendering
- Games like Antichamber and Hyperbolica for inspiration

## 🚧 Roadmap

- [ ] Audio propagation in curved spaces
- [ ] Networked multiplayer support
- [ ] More geometry types (Torus, Klein bottle)
- [ ] Visual scripting for portal creation
- [ ] Performance optimizations for mobile/web
- [ ] Unity/Unreal integration plugins

## 💬 Contact

For questions, suggestions, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/metatopia-game-engine/issues)
- Discord: [Join our server](https://discord.gg/metatopia)
- Email: contact@metatopia-engine.dev

---

**Ready to bend reality?** Start building impossible worlds with Metatopia today! 🌀