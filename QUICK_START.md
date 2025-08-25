# Quick Start Guide - Metatopia Game Engine

## Prerequisites

### 1. Install Rust
If you don't have Rust installed, get it from [rustup.rs](https://rustup.rs/):

**macOS/Linux:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows:**
Download and run the installer from [rustup.rs](https://rustup.rs/)

After installation, restart your terminal and verify:
```bash
rustc --version
cargo --version
```

### 2. Install Build Dependencies

**macOS:**
```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config libx11-dev libxi-dev libgl1-mesa-dev
```

**Windows:**
- Install Visual Studio 2019 or later with C++ build tools
- Or install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)

## Running the Examples

### 1. Clone and Enter the Project
```bash
cd /Users/iliaschrysovergis/GitLab/metatopia-game-engine
```

### 2. Build the Project
First, build all dependencies and the engine:
```bash
cargo build --release
```

Note: The first build will take a few minutes as it downloads and compiles all dependencies.

### 3. Run the Examples

#### Option A: Basic Non-Euclidean Demo
Experience portals and different geometries:
```bash
cargo run --example basic_game --release
```

**Controls:**
- WASD - Move around
- Mouse - Look around  
- Space - Move up
- Shift - Move down
- Walk through glowing portals to transition between Euclidean, Hyperbolic, and Spherical spaces
- ESC - Quit

#### Option B: Pest Control Simulator
Play as an exterminator in Euclidean space:
```bash
cargo run --example pest_control_sim --release
```

**Controls:**
- WASD - Move
- Mouse - Look/Aim
- Left Click - Use tool
- 1-3 - Switch tools
- R - Reload
- ESC - Quit

#### Option C: VR Netflix in Hyperbolic Space
Infinite movie theaters without overlap:
```bash
cargo run --example vr_netflix_hyperbolic --release
```

**Controls:**
- Mouse - Look around to select screens
- 1 - Hyperbolic Lobby
- 2 - Spherical Dome
- 3 - Escher Theater
- 4 - Personal Pocket
- 5 - Social Hub
- Space - Play/Pause selected screen
- ESC - Quit

## Troubleshooting

### If you get "command not found: cargo"
Make sure Rust is in your PATH:
```bash
source $HOME/.cargo/env
```

### If you get graphics/GPU errors
The engine requires a GPU with Vulkan, Metal (macOS), or DirectX 12 (Windows) support. Update your graphics drivers:
- **NVIDIA**: [nvidia.com/drivers](https://www.nvidia.com/drivers)
- **AMD**: [amd.com/support](https://www.amd.com/support)
- **Intel**: [intel.com/content/www/us/en/support](https://www.intel.com/content/www/us/en/support.html)

### If the build fails with "package not found"
Update your Cargo index:
```bash
cargo update
```

### If you get linking errors on Linux
Install additional libraries:
```bash
sudo apt-get install libudev-dev libwayland-dev libxkbcommon-dev
```

### Performance Issues
If the examples run slowly:
1. Make sure you're using `--release` flag (optimized build)
2. Close other GPU-intensive applications
3. Reduce window size in the EngineConfig

## Development Mode

For faster compilation during development (but slower runtime):
```bash
cargo run --example basic_game
```

To see detailed logging:
```bash
RUST_LOG=debug cargo run --example basic_game
```

## Next Steps

1. **Explore the code**: Check out the examples in `examples/` directory
2. **Read the docs**: See README.md for architecture details
3. **Create your own**: Copy an example and start building your non-Euclidean experience!

## System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **CPU**: Dual-core 2.5GHz or better
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Vulkan 1.2, Metal, or DirectX 12 compatible
- **Disk**: 2GB free space for build artifacts

## Quick Test

To quickly verify everything works, run this minimal test:
```bash
cargo test
```

This will run the engine's unit tests without launching graphics.

---

**Need help?** Check the README.md for more details or create an issue on GitHub!