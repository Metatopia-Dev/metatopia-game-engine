//! Pest Control Simulator - Euclidean Space Game
//! 
//! Play as an exterminator dealing with various pests in homes and buildings.
//! Uses standard Euclidean geometry for realistic physics and movement.

use metatopia_engine::prelude::*;
use cgmath::{Point3, Vector3, Quaternion, Rad};
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
enum PestType {
    Cockroach,
    Ant,
    Spider,
    Rat,
    Wasp,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ToolType {
    SprayBottle,
    BaitStation,
    Trap,
    VacuumGun,
    Fumigator,
}

#[derive(Component, Clone)]
struct Pest {
    pest_type: PestType,
    health: f32,
    speed: f32,
    ai_state: PestAIState,
    detection_radius: f32,
}

impl Component for Pest {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[derive(Clone, Copy, Debug)]
enum PestAIState {
    Wandering,
    Fleeing,
    Hiding,
    Attacking,
    Dead,
}

#[derive(Component, Clone)]
struct Tool {
    tool_type: ToolType,
    ammo: u32,
    max_ammo: u32,
    range: f32,
    damage: f32,
    cooldown: f32,
    last_used: f32,
}

impl Component for Tool {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[derive(Component, Clone)]
struct Infestation {
    severity: f32,  // 0.0 to 1.0
    location_type: LocationType,
    pests_remaining: u32,
    pests_eliminated: u32,
}

impl Component for Infestation {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[derive(Clone, Copy, Debug)]
enum LocationType {
    Kitchen,
    Bathroom,
    Basement,
    Attic,
    Garden,
    Restaurant,
}

struct PestControlSimulator {
    world: World,
    player: Entity,
    current_tool: ToolType,
    score: u32,
    level: u32,
    time_remaining: f32,
    active_pests: Vec<Entity>,
    camera: Camera,
    camera_controller: FPSCameraController,
    current_location: LocationType,
    infestation_level: f32,
    tools_inventory: HashMap<ToolType, Tool>,
}

impl PestControlSimulator {
    fn new() -> Self {
        let mut world = World::new();
        
        // Create player entity
        let player = world.create_entity();
        world.add_component(
            player,
            Transform::new(ChartId(0), Point3::new(0.0, 1.7, 0.0)),
        );
        
        // Initialize tools inventory
        let mut tools_inventory = HashMap::new();
        
        tools_inventory.insert(
            ToolType::SprayBottle,
            Tool {
                tool_type: ToolType::SprayBottle,
                ammo: 100,
                max_ammo: 100,
                range: 3.0,
                damage: 25.0,
                cooldown: 0.2,
                last_used: 0.0,
            },
        );
        
        tools_inventory.insert(
            ToolType::VacuumGun,
            Tool {
                tool_type: ToolType::VacuumGun,
                ammo: 50,
                max_ammo: 50,
                range: 5.0,
                damage: 100.0,
                cooldown: 0.5,
                last_used: 0.0,
            },
        );
        
        tools_inventory.insert(
            ToolType::BaitStation,
            Tool {
                tool_type: ToolType::BaitStation,
                ammo: 10,
                max_ammo: 10,
                range: 1.0,
                damage: 0.0, // Bait works over time
                cooldown: 1.0,
                last_used: 0.0,
            },
        );
        
        // Create camera
        let camera = Camera::new(
            ChartId(0),
            Point3::new(0.0, 1.7, 0.0),
            Point3::new(0.0, 1.7, 1.0),
            1280.0 / 720.0,
        );
        
        let camera_controller = FPSCameraController::new();
        
        Self {
            world,
            player,
            current_tool: ToolType::SprayBottle,
            score: 0,
            level: 1,
            time_remaining: 300.0, // 5 minutes per level
            active_pests: Vec::new(),
            camera,
            camera_controller,
            current_location: LocationType::Kitchen,
            infestation_level: 0.3,
            tools_inventory,
        }
    }
    
    fn spawn_pest(&mut self, pest_type: PestType, position: Point3<f32>) {
        let pest_entity = self.world.create_entity();
        
        let (health, speed, detection_radius) = match pest_type {
            PestType::Cockroach => (50.0, 3.0, 5.0),
            PestType::Ant => (10.0, 1.0, 3.0),
            PestType::Spider => (30.0, 2.0, 4.0),
            PestType::Rat => (100.0, 4.0, 8.0),
            PestType::Wasp => (40.0, 5.0, 10.0),
        };
        
        self.world.add_component(
            pest_entity,
            Transform::new(ChartId(0), position),
        );
        
        self.world.add_component(
            pest_entity,
            Pest {
                pest_type,
                health,
                speed,
                ai_state: PestAIState::Wandering,
                detection_radius,
            },
        );
        
        self.world.add_component(
            pest_entity,
            Velocity {
                linear: Vector3::new(0.0, 0.0, 0.0),
                angular: Vector3::new(0.0, 0.0, 0.0),
            },
        );
        
        self.world.add_component(
            pest_entity,
            Renderable {
                mesh_id: format!("pest_{:?}", pest_type),
                shader_id: "pest_shader".to_string(),
                visible: true,
            },
        );
        
        self.active_pests.push(pest_entity);
    }
    
    fn spawn_infestation(&mut self, location: LocationType) {
        self.current_location = location;
        
        // Clear existing pests
        for pest in self.active_pests.clone() {
            self.world.destroy_entity(pest);
        }
        self.active_pests.clear();
        
        // Spawn pests based on location and level
        let mut rng = rand::thread_rng();
        let pest_count = (5 + self.level * 2).min(20);
        
        for _ in 0..pest_count {
            let pest_type = match location {
                LocationType::Kitchen => {
                    match rng.gen_range(0..3) {
                        0 => PestType::Cockroach,
                        1 => PestType::Ant,
                        _ => PestType::Rat,
                    }
                }
                LocationType::Bathroom => {
                    match rng.gen_range(0..2) {
                        0 => PestType::Cockroach,
                        _ => PestType::Spider,
                    }
                }
                LocationType::Basement => PestType::Rat,
                LocationType::Attic => {
                    match rng.gen_range(0..2) {
                        0 => PestType::Spider,
                        _ => PestType::Wasp,
                    }
                }
                LocationType::Garden => PestType::Wasp,
                LocationType::Restaurant => PestType::Cockroach,
            };
            
            let x = rng.gen_range(-10.0..10.0);
            let z = rng.gen_range(-10.0..10.0);
            let y = match pest_type {
                PestType::Wasp => rng.gen_range(1.0..3.0),
                _ => 0.1,
            };
            
            self.spawn_pest(pest_type, Point3::new(x, y, z));
        }
        
        // Create infestation entity
        let infestation = self.world.create_entity();
        self.world.add_component(
            infestation,
            Infestation {
                severity: self.infestation_level,
                location_type: location,
                pests_remaining: pest_count,
                pests_eliminated: 0,
            },
        );
    }
    
    fn use_tool(&mut self) {
        if let Some(tool) = self.tools_inventory.get_mut(&self.current_tool) {
            if tool.ammo > 0 && self.time_remaining - tool.last_used > tool.cooldown {
                tool.ammo -= 1;
                tool.last_used = self.time_remaining;
                
                // Apply tool effect
                match tool.tool_type {
                    ToolType::SprayBottle | ToolType::VacuumGun => {
                        self.spray_area(tool.range, tool.damage);
                    }
                    ToolType::BaitStation => {
                        self.place_bait();
                    }
                    ToolType::Trap => {
                        self.place_trap();
                    }
                    ToolType::Fumigator => {
                        self.fumigate_area();
                    }
                }
            }
        }
    }
    
    fn spray_area(&mut self, range: f32, damage: f32) {
        let camera_pos = self.camera.position.local.to_point();
        let camera_forward = self.camera.forward();
        
        // Check for pest hits
        for pest_entity in self.active_pests.clone() {
            if let Some(pest_transform) = self.world.get_component::<Transform>(*pest_entity) {
                let pest_pos = pest_transform.position.local.to_point();
                let to_pest = pest_pos - camera_pos;
                let distance = to_pest.magnitude();
                
                if distance <= range {
                    // Check if pest is in front of player
                    let dot = to_pest.normalize().dot(camera_forward);
                    if dot > 0.7 {  // ~45 degree cone
                        if let Some(pest) = self.world.get_component_mut::<Pest>(*pest_entity) {
                            pest.health -= damage;
                            if pest.health <= 0.0 {
                                pest.ai_state = PestAIState::Dead;
                                self.eliminate_pest(*pest_entity);
                            } else {
                                pest.ai_state = PestAIState::Fleeing;
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn place_bait(&mut self) {
        // Bait attracts pests then eliminates them over time
        println!("Bait station placed!");
    }
    
    fn place_trap(&mut self) {
        // Trap catches pests that walk over it
        println!("Trap placed!");
    }
    
    fn fumigate_area(&mut self) {
        // Fumigation affects entire room
        println!("Fumigating area!");
        for pest_entity in self.active_pests.clone() {
            self.eliminate_pest(pest_entity);
        }
    }
    
    fn eliminate_pest(&mut self, pest: Entity) {
        self.world.destroy_entity(pest);
        if let Some(pos) = self.active_pests.iter().position(|&e| e == pest) {
            self.active_pests.remove(pos);
        }
        self.score += 10;
        
        // Check if level complete
        if self.active_pests.is_empty() {
            self.complete_level();
        }
    }
    
    fn complete_level(&mut self) {
        println!("Level {} Complete! Score: {}", self.level, self.score);
        self.level += 1;
        self.infestation_level = (self.infestation_level + 0.1).min(1.0);
        self.time_remaining = 300.0;
        
        // Spawn next infestation
        let locations = [
            LocationType::Kitchen,
            LocationType::Bathroom,
            LocationType::Basement,
            LocationType::Attic,
            LocationType::Garden,
            LocationType::Restaurant,
        ];
        let mut rng = rand::thread_rng();
        let next_location = locations[rng.gen_range(0..locations.len())];
        self.spawn_infestation(next_location);
    }
    
    fn update_pest_ai(&mut self, dt: f32) {
        let player_pos = self.camera.position.local.to_point();
        
        for pest_entity in self.active_pests.clone() {
            if let Some(pest) = self.world.get_component_mut::<Pest>(pest_entity) {
                if let Some(transform) = self.world.get_component::<Transform>(pest_entity) {
                    let pest_pos = transform.position.local.to_point();
                    let distance_to_player = (player_pos - pest_pos).magnitude();
                    
                    // Update AI state based on player proximity
                    match pest.ai_state {
                        PestAIState::Wandering => {
                            if distance_to_player < pest.detection_radius {
                                pest.ai_state = PestAIState::Fleeing;
                            }
                        }
                        PestAIState::Fleeing => {
                            if distance_to_player > pest.detection_radius * 2.0 {
                                pest.ai_state = PestAIState::Hiding;
                            }
                        }
                        PestAIState::Hiding => {
                            if distance_to_player > pest.detection_radius * 3.0 {
                                pest.ai_state = PestAIState::Wandering;
                            }
                        }
                        _ => {}
                    }
                }
                
                // Update velocity based on AI state
                if let Some(velocity) = self.world.get_component_mut::<Velocity>(pest_entity) {
                    match pest.ai_state {
                        PestAIState::Wandering => {
                            // Random movement
                            let mut rng = rand::thread_rng();
                            velocity.linear = Vector3::new(
                                rng.gen_range(-1.0..1.0),
                                0.0,
                                rng.gen_range(-1.0..1.0),
                            ).normalize() * pest.speed;
                        }
                        PestAIState::Fleeing => {
                            // Move away from player
                            let flee_dir = (pest_pos - player_pos).normalize();
                            velocity.linear = Vector3::new(
                                flee_dir.x * pest.speed * 1.5,
                                0.0,
                                flee_dir.z * pest.speed * 1.5,
                            );
                        }
                        PestAIState::Hiding => {
                            velocity.linear = Vector3::new(0.0, 0.0, 0.0);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

impl GameState for PestControlSimulator {
    fn on_init(&mut self, engine: &mut Engine) {
        println!("=== PEST CONTROL SIMULATOR ===");
        println!("Eliminate all pests to complete each level!");
        println!("\nControls:");
        println!("  WASD - Move");
        println!("  Mouse - Look around");
        println!("  Left Click - Use current tool");
        println!("  1-5 - Switch tools");
        println!("  R - Reload/Refill");
        println!("  ESC - Quit");
        
        // Start first level
        self.spawn_infestation(LocationType::Kitchen);
    }
    
    fn on_update(&mut self, engine: &mut Engine, dt: f32) {
        // Update camera
        self.camera_controller.update(&mut self.camera, &engine.input, dt);
        
        // Update timer
        self.time_remaining -= dt;
        if self.time_remaining <= 0.0 {
            println!("Time's up! Game Over. Final Score: {}", self.score);
            engine.quit();
        }
        
        // Tool switching
        use KeyCode::*;
        if engine.input.is_key_pressed(Num1) {
            self.current_tool = ToolType::SprayBottle;
        } else if engine.input.is_key_pressed(Num2) {
            self.current_tool = ToolType::VacuumGun;
        } else if engine.input.is_key_pressed(Num3) {
            self.current_tool = ToolType::BaitStation;
        }
        
        // Use tool
        if engine.input.is_mouse_button_pressed(MouseButton::Left) {
            self.use_tool();
        }
        
        // Update pest AI
        self.update_pest_ai(dt);
        
        // Update world systems
        self.world.update(dt);
        
        // Update HUD info
        if engine.time.frame_count() % 60 == 0 {
            println!("Level: {} | Score: {} | Time: {:.0}s | Pests: {} | Tool: {:?} ({})", 
                self.level, 
                self.score, 
                self.time_remaining,
                self.active_pests.len(),
                self.current_tool,
                self.tools_inventory.get(&self.current_tool).map_or(0, |t| t.ammo)
            );
        }
        
        // Quit on ESC
        if engine.input.is_key_pressed(KeyCode::Escape) {
            engine.quit();
        }
    }
    
    fn on_render(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        renderer.clear(0.8, 0.8, 0.7, 1.0); // Light interior color
        
        // Render room based on location type
        match self.current_location {
            LocationType::Kitchen => {
                // Render kitchen environment
            }
            LocationType::Bathroom => {
                // Render bathroom environment
            }
            _ => {
                // Render generic room
            }
        }
        
        // Render pests
        for pest_entity in &self.active_pests {
            // Render pest models
        }
        
        // Render tool in hand
        // Render HUD
    }
    
    fn on_cleanup(&mut self, _engine: &mut Engine) {
        println!("Thanks for playing Pest Control Simulator!");
        println!("Final Score: {}", self.score);
    }
}

fn main() {
    pollster::block_on(async {
        let config = EngineConfig {
            title: "Pest Control Simulator".to_string(),
            width: 1280,
            height: 720,
            vsync: true,
            target_fps: Some(60),
            resizable: false,
        };
        
        let engine = Engine::new(config).await.expect("Failed to create engine");
        let game = PestControlSimulator::new();
        
        engine.run(game).expect("Failed to run game");
    });
}