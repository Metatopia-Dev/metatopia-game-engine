# 🎮 Game Developer's Guide — Metatopia Engine

Create your own game in **under 5 minutes**. No wgpu knowledge required.

---

## Quick Start

### 1. Create your game file

Create `examples/my_game.rs`:

```rust
use metatopia_engine::quickstart::*;

struct MyGame {
    score: u32,
}

impl GameApp for MyGame {
    fn title(&self) -> &str { "My Game" }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // Built-in WASD + mouse camera
        ctx.default_camera_movement();

        // Quit on ESC
        if ctx.key_pressed(VirtualKey::Escape) { ctx.quit(); }

        // Your game logic here!
        if ctx.mouse_pressed(winit::event::MouseButton::Left) {
            self.score += 10;
            println!("Score: {}", self.score);
        }
    }
}

fn main() {
    run_game(MyGame { score: 0 });
}
```

### 2. Run it

```bash
cargo run --release --example my_game
```

That's it! You get a window with a lit 3D scene, FPS camera, and input handling.

---

## Architecture

```
Your Game (examples/my_game.rs)
    │
    ├── implements GameApp trait (your code)
    │       ├── title()        → window title
    │       ├── init()         → startup logic
    │       ├── update(ctx)    → game logic every frame
    │       ├── build_mesh()   → 3D geometry
    │       └── shader_source()→ custom WGSL shader
    │
    └── calls run_game(my_game)
            │
            └── quickstart module handles everything:
                    ├── Window creation
                    ├── GPU device setup (wgpu)
                    ├── Render pipeline
                    ├── Depth buffer
                    ├── Camera uniforms
                    ├── Input tracking
                    └── Event loop
```

---

## The `UpdateCtx` — Your Main Interface

Every frame, your `update()` receives an `UpdateCtx` with everything you need:

### Camera

```rust
// Move camera with WASD (built-in)
ctx.default_camera_movement();

// Or control it manually
ctx.camera.position = Vector3::new(0.0, 5.0, 10.0);
ctx.camera.yaw = 1.57;  // radians
ctx.camera.move_speed = 10.0;
```

### Input

```rust
// Keyboard
if ctx.key_held(VirtualKey::KeyW) { /* held down */ }
if ctx.key_pressed(VirtualKey::KeyR) { /* just pressed this frame */ }

// Mouse
if ctx.mouse_pressed(winit::event::MouseButton::Left) { /* clicked */ }
```

### Timing

```rust
let elapsed = ctx.time;  // seconds since start
let delta = ctx.dt;       // seconds since last frame
```

### Scene Data → Shader

```rust
// Send any data to your WGSL shader via scene uniform
ctx.scene.game_data = [score as f32, level as f32, health, 0.0];
ctx.scene.extra0 = [enemy_x, enemy_y, enemy_z, alive as f32];

// Lighting (modify defaults)
ctx.scene.sun_direction = [0.5, 0.8, 0.3, 3.0]; // xyz=dir, w=intensity
ctx.scene.light0_pos = [x, y, z, range];
ctx.scene.light0_color = [r, g, b, intensity];
```

---

## Custom Meshes

Override `build_mesh()` to create your own geometry:

```rust
fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idxs = Vec::new();

    // Use built-in mesh builders
    let (fv, fi) = MeshBuilder::floor(20.0, [0.3, 0.3, 0.35]);
    verts.extend(fv); idxs.extend(fi);

    let (cv, ci) = MeshBuilder::cube(1.0, [0.8, 0.2, 0.3]);
    let offset = verts.len() as u32;
    verts.extend(cv);
    for i in ci { idxs.push(i + offset); }

    let (sv, si) = MeshBuilder::sphere(0.5, 16, 12, [0.2, 0.6, 1.0]);
    let offset = verts.len() as u32;
    verts.extend(sv);
    for i in si { idxs.push(i + offset); }

    (verts, idxs)
}
```

### Available Mesh Builders

| Builder | Description |
|---------|-------------|
| `MeshBuilder::floor(size, color)` | Flat grid on XZ plane at y=0 |
| `MeshBuilder::cube(size, color)` | Axis-aligned cube centered at origin |
| `MeshBuilder::sphere(radius, segments, rings, color)` | UV sphere |

### Custom Vertices

```rust
GameVertex::colored(
    [x, y, z],           // position
    [nx, ny, nz],        // normal
    [r, g, b],           // color
)
```

Or the full constructor:
```rust
GameVertex {
    position: [0.0, 1.0, 0.0],
    normal: [0.0, 1.0, 0.0],
    uv: [0.5, 0.5],
    color: [1.0, 0.0, 0.0],
    pbr: [metallic, roughness, emission, custom],
}
```

---

## Custom Shaders

Override `shader_source()` to use your own WGSL shader:

```rust
fn shader_source(&self) -> String {
    std::fs::read_to_string("shaders/my_shader.wgsl").unwrap()
}
```

Start by copying `shaders/template_game.wgsl` and modifying the fragment shader.

### Shader Uniform Layout

In your WGSL shader, these uniforms are available:

```wgsl
// @group(0) @binding(0)
camera.view_proj       // mat4x4<f32> — view-projection matrix
camera.view_position   // vec4<f32>   — camera world position

// @group(0) @binding(1)
scene.sun_direction    // vec4 — xyz=dir, w=intensity
scene.sun_color        // vec4 — rgb=color
scene.light0_pos       // vec4 — xyz=pos, w=range
scene.light0_color     // vec4 — rgb=color, w=intensity
scene.params           // vec4 — x=time (auto), y/z/w=yours
scene.game_data        // vec4 — all yours!
scene.extra0..extra3   // vec4 × 4 — all yours!
scene.hud_info         // vec4 — x=res_x, y=res_y (auto), z/w=yours
```

---

## Example Games

| Example | Complexity | Demonstrates |
|---------|-----------|-------------|
| `my_first_game` | ⭐ Beginner | Camera, collection, scoring |
| `basic_game` | ⭐⭐ Intermediate | Non-Euclidean physics, portals, audio |
| `pest_control_sim` | ⭐⭐⭐ Advanced | FPS combat, AI, procedural levels, HUD |

Run any example:
```bash
cargo run --release --example my_first_game
cargo run --release --example basic_game
cargo run --release --example pest_control_sim
```

---

## Tips

1. **Start with `my_first_game.rs`** — copy it and rename
2. **Use `ctx.scene.game_data`** to pass values to your shader
3. **Print to console** for debugging — `println!` works fine
4. **The shader runs per-pixel** — it's where visual magic happens
5. **Press R** to reset (implement in your `update()`)
6. **Performance**: use `--release` flag for 60+ fps

---

## File Structure for a New Game

```
metatopia-game-engine/
├── examples/
│   └── my_game.rs          ← Your game logic (~100 lines)
├── shaders/
│   └── my_shader.wgsl      ← Your shader (optional, template works)
└── src/
    └── quickstart.rs        ← Engine handles everything else
```

Happy building! 🚀
