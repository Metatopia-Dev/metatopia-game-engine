//! My First Game — Starter template using the quickstart module
//!
//! Run with: cargo run --release --example my_first_game
//!
//! This is a minimal working game that demonstrates:
//!   - FPS camera movement (WASD + mouse)
//!   - A floor + spinning cube scene
//!   - Custom game logic (scoring, spawning objects)
//!   - Shader communication via scene.game_data
//!
//! Modify this file to build your own game!

use metatopia_engine::quickstart::*;
use cgmath::Vector3;

// ─── Your Game State ───────────────────────────────────────────────────────
// Put all your game variables here.

struct MyGame {
    score: u32,
    cubes_collected: u32,
    cube_positions: Vec<Vector3<f32>>,
    cube_spin: f32,
}

impl MyGame {
    fn new() -> Self {
        Self {
            score: 0,
            cubes_collected: 0,
            cube_positions: vec![
                Vector3::new(3.0, 0.5, -2.0),
                Vector3::new(-4.0, 0.5, 1.0),
                Vector3::new(0.0, 0.5, -6.0),
            ],
            cube_spin: 0.0,
        }
    }
}

// ─── Implement GameApp ─────────────────────────────────────────────────────
// This is the only trait you need to implement!

impl GameApp for MyGame {
    fn title(&self) -> &str { "My First Game — Collect the Cubes!" }

    fn init(&mut self) {
        println!("🎮 Welcome! Walk over cubes to collect them.");
        println!("   Score: {}", self.score);
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // ── Built-in camera movement (WASD + mouse) ──────────────
        ctx.default_camera_movement();

        // ── Quit on ESC ──────────────────────────────────────────
        if ctx.key_pressed(VirtualKey::Escape) {
            println!("Final Score: {}", self.score);
            ctx.quit();
            return;
        }

        // ── Spin the cubes ───────────────────────────────────────
        self.cube_spin += ctx.dt * 2.0;

        // ── Check collection (walk within 2 units of a cube) ─────
        let player_pos = ctx.camera.position;
        let mut collected_idx = None;
        for (i, cube_pos) in self.cube_positions.iter().enumerate() {
            let dist = ((player_pos.x - cube_pos.x).powi(2)
                      + (player_pos.z - cube_pos.z).powi(2)).sqrt();
            if dist < 2.0 {
                collected_idx = Some(i);
                break;
            }
        }
        if let Some(idx) = collected_idx {
            self.cube_positions.remove(idx);
            self.cubes_collected += 1;
            self.score += 100;
            println!("✨ Cube collected! Score: {} ({}/3)", self.score, self.cubes_collected);
            if self.cube_positions.is_empty() {
                println!("🎉 All cubes collected! You win!");
            }
        }

        // ── Send data to shader ──────────────────────────────────
        // game_data is available in WGSL as scene.game_data
        ctx.scene.game_data = [
            self.score as f32,
            self.cubes_collected as f32,
            self.cube_spin,
            self.cube_positions.len() as f32,
        ];

        // ── Send cube positions to shader via extra slots ────────
        for (i, pos) in self.cube_positions.iter().enumerate() {
            let slot = [pos.x, pos.y, pos.z, 1.0]; // w=1 means active
            match i {
                0 => ctx.scene.extra0 = slot,
                1 => ctx.scene.extra1 = slot,
                2 => ctx.scene.extra2 = slot,
                _ => {}
            }
        }
        // Zero out unused slots
        for i in self.cube_positions.len()..3 {
            match i {
                0 => ctx.scene.extra0 = [0.0; 4],
                1 => ctx.scene.extra1 = [0.0; 4],
                2 => ctx.scene.extra2 = [0.0; 4],
                _ => {}
            }
        }
    }

    fn build_mesh(&self) -> (Vec<GameVertex>, Vec<u32>) {
        // Build a floor
        let (mut verts, mut idxs) = MeshBuilder::floor(30.0, [0.25, 0.28, 0.3]);

        // Add 3 collectible cubes
        let cube_colors = [
            [0.9, 0.2, 0.3],  // Red
            [0.2, 0.8, 0.3],  // Green
            [0.3, 0.4, 0.9],  // Blue
        ];
        for (i, pos) in self.cube_positions.iter().enumerate() {
            let (cv, ci) = MeshBuilder::cube(0.8, cube_colors[i % 3]);
            let offset = verts.len() as u32;
            for mut v in cv {
                v.position[0] += pos.x;
                v.position[1] += pos.y;
                v.position[2] += pos.z;
                v.pbr[2] = 0.5; // slight emission so they glow
                verts.push(v);
            }
            for idx in ci { idxs.push(idx + offset); }
        }

        (verts, idxs)
    }
}

// ─── Entry Point ───────────────────────────────────────────────────────────

fn main() {
    run_game(MyGame::new());
}
