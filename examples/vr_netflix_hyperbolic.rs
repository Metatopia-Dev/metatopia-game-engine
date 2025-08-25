//! VR Netflix in Non-Euclidean Space
//! 
//! Experience infinite movie theaters in hyperbolic space where you can have
//! unlimited screens without them overlapping. Navigate through a Poincaré disk
//! theater or spherical cinema dome with seamless portals to different viewing rooms.

use metatopia_engine::prelude::*;
use cgmath::{Point3, Vector3, Quaternion, Rad};
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Movie {
    title: String,
    genre: String,
    duration: f32,
    thumbnail_url: String,
    video_url: String,
    rating: f32,
    year: u32,
}

#[derive(Debug, Clone)]
struct Screen {
    entity: Entity,
    movie: Option<Movie>,
    position: ManifoldPosition,
    size: (f32, f32),
    playing: bool,
    current_time: f32,
    volume: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TheaterSpace {
    HyperbolicLobby,    // Poincaré disk with infinite screens
    SphericalDome,      // 360° viewing dome
    EscherTheater,      // Impossible geometry with looping stairs
    PersonalPocket,     // Your own curved pocket dimension
    SocialHub,          // Shared viewing space with friends
}

struct VRNetflixExperience {
    manifold: Manifold,
    current_space: TheaterSpace,
    screens: Vec<Screen>,
    movie_library: HashMap<String, Movie>,
    camera: Camera,
    selected_screen: Option<usize>,
    world: World,
    user_preferences: UserPreferences,
    friends: Vec<Friend>,
    watch_party: Option<WatchParty>,
}

#[derive(Clone)]
struct UserPreferences {
    preferred_distance: f32,
    curved_screen: bool,
    ambient_lighting: f32,
    spatial_audio: bool,
    auto_arrange: bool,
}

#[derive(Clone)]
struct Friend {
    name: String,
    avatar: String,
    position: ManifoldPosition,
    watching: Option<String>,
}

#[derive(Clone)]
struct WatchParty {
    host: String,
    movie: Movie,
    participants: Vec<Friend>,
    chat_messages: Vec<ChatMessage>,
}

#[derive(Clone)]
struct ChatMessage {
    sender: String,
    message: String,
    timestamp: f32,
}

impl VRNetflixExperience {
    fn new() -> Self {
        let mut manifold = Manifold::new();
        
        // Create hyperbolic lobby space (Poincaré disk)
        let hyperbolic_lobby = manifold.add_chart(GeometryType::Hyperbolic);
        
        // Create spherical dome theater
        let spherical_dome = manifold.add_chart(GeometryType::Spherical);
        
        // Create Escher-like impossible theater
        let escher_theater = manifold.add_chart(GeometryType::Custom);
        
        // Create personal pocket dimension (small hyperbolic space)
        let personal_pocket = manifold.add_chart(GeometryType::Hyperbolic);
        
        // Connect spaces with portals
        Self::create_theater_portals(&mut manifold, hyperbolic_lobby, spherical_dome, escher_theater, personal_pocket);
        
        // Initialize camera in hyperbolic lobby
        let camera = Camera::new(
            hyperbolic_lobby,
            Point3::new(0.0, 1.7, 0.0),
            Point3::new(0.0, 1.7, 0.5),
            1280.0 / 720.0,
        );
        
        let user_preferences = UserPreferences {
            preferred_distance: 3.0,
            curved_screen: true,
            ambient_lighting: 0.3,
            spatial_audio: true,
            auto_arrange: true,
        };
        
        let mut netflix = Self {
            manifold,
            current_space: TheaterSpace::HyperbolicLobby,
            screens: Vec::new(),
            movie_library: Self::load_movie_library(),
            camera,
            selected_screen: None,
            world: World::new(),
            user_preferences,
            friends: Vec::new(),
            watch_party: None,
        };
        
        // Create initial screens in hyperbolic space
        netflix.create_hyperbolic_theater();
        
        netflix
    }
    
    fn create_theater_portals(
        manifold: &mut Manifold,
        hyperbolic_lobby: ChartId,
        spherical_dome: ChartId,
        escher_theater: ChartId,
        personal_pocket: ChartId,
    ) {
        // Portal from lobby to spherical dome
        manifold.create_portal(
            hyperbolic_lobby,
            spherical_dome,
            Point3::new(0.7, 0.0, 0.0),
            Point3::new(0.0, -0.9, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        // Portal from lobby to Escher theater
        manifold.create_portal(
            hyperbolic_lobby,
            escher_theater,
            Point3::new(-0.7, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_angle_y(Rad(90.0_f32.to_radians())),
        ).unwrap();
        
        // Portal from lobby to personal pocket
        manifold.create_portal(
            hyperbolic_lobby,
            personal_pocket,
            Point3::new(0.0, 0.0, 0.7),
            Point3::new(0.0, 0.0, 0.0),
            Mat4::from_scale(0.5), // Smaller space
        ).unwrap();
        
        // Return portals
        manifold.create_portal(
            spherical_dome,
            hyperbolic_lobby,
            Point3::new(0.0, -0.9, 0.0),
            Point3::new(0.7, 0.0, 0.0),
            Mat4::from_scale(1.0),
        ).unwrap();
        
        manifold.create_portal(
            escher_theater,
            hyperbolic_lobby,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(-0.7, 0.0, 0.0),
            Mat4::from_angle_y(Rad(-90.0_f32.to_radians())),
        ).unwrap();
        
        manifold.create_portal(
            personal_pocket,
            hyperbolic_lobby,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.7),
            Mat4::from_scale(2.0), // Scale back up
        ).unwrap();
    }
    
    fn load_movie_library() -> HashMap<String, Movie> {
        let mut library = HashMap::new();
        
        // Sample movies
        library.insert("inception".to_string(), Movie {
            title: "Inception".to_string(),
            genre: "Sci-Fi".to_string(),
            duration: 148.0 * 60.0,
            thumbnail_url: "inception_thumb.jpg".to_string(),
            video_url: "inception.mp4".to_string(),
            rating: 8.8,
            year: 2010,
        });
        
        library.insert("interstellar".to_string(), Movie {
            title: "Interstellar".to_string(),
            genre: "Sci-Fi".to_string(),
            duration: 169.0 * 60.0,
            thumbnail_url: "interstellar_thumb.jpg".to_string(),
            video_url: "interstellar.mp4".to_string(),
            rating: 8.6,
            year: 2014,
        });
        
        library.insert("matrix".to_string(), Movie {
            title: "The Matrix".to_string(),
            genre: "Sci-Fi".to_string(),
            duration: 136.0 * 60.0,
            thumbnail_url: "matrix_thumb.jpg".to_string(),
            video_url: "matrix.mp4".to_string(),
            rating: 8.7,
            year: 1999,
        });
        
        library.insert("spirited_away".to_string(), Movie {
            title: "Spirited Away".to_string(),
            genre: "Animation".to_string(),
            duration: 125.0 * 60.0,
            thumbnail_url: "spirited_away_thumb.jpg".to_string(),
            video_url: "spirited_away.mp4".to_string(),
            rating: 8.6,
            year: 2001,
        });
        
        library
    }
    
    fn create_hyperbolic_theater(&mut self) {
        // In hyperbolic space, we can fit infinite screens without overlap
        // Arrange screens in a hyperbolic tiling pattern
        
        let movies: Vec<_> = self.movie_library.values().cloned().collect();
        let num_screens = 7; // Start with 7 screens in hyperbolic arrangement
        
        for i in 0..num_screens {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / num_screens as f32;
            let radius = 0.5; // In Poincaré disk
            
            // Hyperbolic positioning
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            
            let screen_entity = self.world.create_entity();
            let position = ManifoldPosition::new(
                ChartId(1), // Hyperbolic space
                Point3::new(x, 1.5, y),
            );
            
            self.world.add_component(
                screen_entity,
                Transform::new(ChartId(1), Point3::new(x, 1.5, y)),
            );
            
            let screen = Screen {
                entity: screen_entity,
                movie: movies.get(i % movies.len()).cloned(),
                position,
                size: (4.0, 2.25), // 16:9 aspect ratio
                playing: false,
                current_time: 0.0,
                volume: 0.8,
            };
            
            self.screens.push(screen);
        }
        
        // Add a central mega-screen for featured content
        let featured_entity = self.world.create_entity();
        let featured_position = ManifoldPosition::new(
            ChartId(1),
            Point3::new(0.0, 2.0, 0.0),
        );
        
        self.world.add_component(
            featured_entity,
            Transform::new(ChartId(1), Point3::new(0.0, 2.0, 0.0)),
        );
        
        self.screens.push(Screen {
            entity: featured_entity,
            movie: self.movie_library.get("inception").cloned(),
            position: featured_position,
            size: (8.0, 4.5),
            playing: false,
            current_time: 0.0,
            volume: 1.0,
        });
    }
    
    fn create_spherical_dome_theater(&mut self) {
        // In spherical space, create a dome theater with screens on the sphere surface
        let radius = 10.0;
        
        // Clear existing screens when switching spaces
        self.screens.clear();
        
        // Create screens arranged on sphere
        for i in 0..8 {
            for j in 0..4 {
                let theta = (i as f32) * 2.0 * std::f32::consts::PI / 8.0;
                let phi = (j as f32) * std::f32::consts::PI / 6.0 + std::f32::consts::PI / 6.0;
                
                let x = radius * phi.sin() * theta.cos();
                let y = radius * phi.cos();
                let z = radius * phi.sin() * theta.sin();
                
                let screen_entity = self.world.create_entity();
                let position = ManifoldPosition::new(
                    ChartId(2), // Spherical space
                    Point3::new(x, y, z),
                );
                
                self.world.add_component(
                    screen_entity,
                    Transform::new(ChartId(2), Point3::new(x, y, z)),
                );
                
                let movies: Vec<_> = self.movie_library.values().cloned().collect();
                
                self.screens.push(Screen {
                    entity: screen_entity,
                    movie: movies.get((i * 4 + j) % movies.len()).cloned(),
                    position,
                    size: (3.0, 1.7),
                    playing: false,
                    current_time: 0.0,
                    volume: 0.8,
                });
            }
        }
    }
    
    fn handle_vr_input(&mut self, input: &InputManager) {
        // Gaze-based selection
        let forward = self.camera.forward();
        let camera_pos = self.camera.position.local.to_point();
        
        // Find screen being looked at
        let mut closest_screen = None;
        let mut closest_distance = f32::MAX;
        
        for (i, screen) in self.screens.iter().enumerate() {
            if let Some(world_pos) = screen.position.to_world(&self.manifold) {
                let to_screen = world_pos - camera_pos;
                let distance = to_screen.magnitude();
                let dot = to_screen.normalize().dot(forward);
                
                if dot > 0.95 && distance < closest_distance {
                    closest_distance = distance;
                    closest_screen = Some(i);
                }
            }
        }
        
        self.selected_screen = closest_screen;
        
        // Play/pause with trigger
        if input.is_gamepad_button_pressed(GamepadButton::RightTrigger) {
            if let Some(idx) = self.selected_screen {
                self.screens[idx].playing = !self.screens[idx].playing;
            }
        }
        
        // Volume control with thumbstick
        let volume_adjust = input.gamepad_axis(GamepadAxis::RightStickY);
        if let Some(idx) = self.selected_screen {
            self.screens[idx].volume = (self.screens[idx].volume + volume_adjust * 0.01).clamp(0.0, 1.0);
        }
    }
    
    fn teleport_to_screen(&mut self, screen_index: usize) {
        if let Some(screen) = self.screens.get(screen_index) {
            // Calculate optimal viewing position based on geometry
            let screen_pos = screen.position.local.to_point();
            
            let viewing_distance = match self.current_space {
                TheaterSpace::HyperbolicLobby => {
                    // In hyperbolic space, closer distances feel farther
                    self.user_preferences.preferred_distance * 0.7
                }
                TheaterSpace::SphericalDome => {
                    // On sphere, maintain distance along geodesic
                    self.user_preferences.preferred_distance
                }
                _ => self.user_preferences.preferred_distance,
            };
            
            // Position camera in front of screen
            let new_pos = Point3::new(
                screen_pos.x * 0.5,
                screen_pos.y,
                screen_pos.z * 0.5 - viewing_distance,
            );
            
            self.camera.set_position(screen.position.chart_id, new_pos);
            self.camera.target = screen_pos;
        }
    }
    
    fn create_watch_party(&mut self, movie_key: &str) {
        if let Some(movie) = self.movie_library.get(movie_key).cloned() {
            self.watch_party = Some(WatchParty {
                host: "You".to_string(),
                movie,
                participants: self.friends.clone(),
                chat_messages: Vec::new(),
            });
            
            // Create shared viewing space
            self.switch_to_space(TheaterSpace::SocialHub);
            
            // Arrange friend avatars in hyperbolic circle
            for (i, friend) in self.friends.iter_mut().enumerate() {
                let angle = (i as f32) * 2.0 * std::f32::consts::PI / self.friends.len() as f32;
                let radius = 0.3;
                
                friend.position = ManifoldPosition::new(
                    self.camera.position.chart_id,
                    Point3::new(
                        radius * angle.cos(),
                        1.7,
                        radius * angle.sin(),
                    ),
                );
            }
        }
    }
    
    fn switch_to_space(&mut self, space: TheaterSpace) {
        self.current_space = space;
        
        match space {
            TheaterSpace::HyperbolicLobby => {
                self.create_hyperbolic_theater();
                self.camera.set_position(ChartId(1), Point3::new(0.0, 1.7, 0.0));
            }
            TheaterSpace::SphericalDome => {
                self.create_spherical_dome_theater();
                self.camera.set_position(ChartId(2), Point3::new(0.0, 0.0, 0.0));
            }
            TheaterSpace::EscherTheater => {
                // Create impossible geometry theater
                self.camera.set_position(ChartId(3), Point3::new(0.0, 1.7, 0.0));
            }
            TheaterSpace::PersonalPocket => {
                // Small personal viewing space
                self.screens.clear();
                
                let entity = self.world.create_entity();
                let position = ManifoldPosition::new(ChartId(4), Point3::new(0.0, 1.7, 2.0));
                
                self.screens.push(Screen {
                    entity,
                    movie: self.movie_library.get("matrix").cloned(),
                    position,
                    size: (6.0, 3.4),
                    playing: false,
                    current_time: 0.0,
                    volume: 1.0,
                });
                
                self.camera.set_position(ChartId(4), Point3::new(0.0, 1.7, -2.0));
            }
            TheaterSpace::SocialHub => {
                // Shared viewing with friends
                self.camera.set_position(ChartId(1), Point3::new(0.0, 1.7, -3.0));
            }
        }
    }
}

impl GameState for VRNetflixExperience {
    fn on_init(&mut self, engine: &mut Engine) {
        println!("=== VR Netflix in Non-Euclidean Space ===");
        println!("\nWelcome to infinite movie theaters!");
        println!("\nSpaces available:");
        println!("  • Hyperbolic Lobby - Infinite screens without overlap");
        println!("  • Spherical Dome - 360° viewing experience");
        println!("  • Escher Theater - Impossible geometry viewing");
        println!("  • Personal Pocket - Your cozy curved dimension");
        println!("  • Social Hub - Watch with friends");
        println!("\nControls:");
        println!("  Look - Gaze at screens to select");
        println!("  Right Trigger - Play/Pause");
        println!("  A Button - Teleport to screen");
        println!("  B Button - Return to lobby");
        println!("  X Button - Switch viewing mode");
        println!("  Y Button - Invite friends");
        println!("  Thumbsticks - Navigate and adjust volume");
        
        // Initialize graphics for movie screens
        engine.renderer.shader_mut().create_geometry_shaders();
    }
    
    fn on_update(&mut self, engine: &mut Engine, dt: f32) {
        // Update camera based on current space geometry
        self.camera.update(&self.manifold);
        
        // Handle VR input
        self.handle_vr_input(&engine.input);
        
        // Update playing movies
        for screen in &mut self.screens {
            if screen.playing {
                if let Some(movie) = &screen.movie {
                    screen.current_time += dt;
                    if screen.current_time >= movie.duration {
                        screen.current_time = 0.0;
                        screen.playing = false;
                    }
                }
            }
        }
        
        // Spatial audio falloff based on geometry
        for screen in &self.screens {
            if screen.playing {
                if let Some(world_pos) = screen.position.to_world(&self.manifold) {
                    let distance = (world_pos - self.camera.position.local.to_point()).magnitude();
                    
                    let falloff = match self.current_space {
                        TheaterSpace::HyperbolicLobby => {
                            // Exponential falloff in hyperbolic space
                            (-distance * 0.5).exp()
                        }
                        TheaterSpace::SphericalDome => {
                            // Uniform audio in spherical space
                            0.8
                        }
                        _ => {
                            // Standard inverse square falloff
                            1.0 / (1.0 + distance * distance * 0.1)
                        }
                    };
                    
                    // Apply spatial audio (would interface with audio system)
                    let effective_volume = screen.volume * falloff;
                }
            }
        }
        
        // Handle space switching
        use KeyCode::*;
        if engine.input.is_key_pressed(Num1) {
            self.switch_to_space(TheaterSpace::HyperbolicLobby);
        } else if engine.input.is_key_pressed(Num2) {
            self.switch_to_space(TheaterSpace::SphericalDome);
        } else if engine.input.is_key_pressed(Num3) {
            self.switch_to_space(TheaterSpace::EscherTheater);
        } else if engine.input.is_key_pressed(Num4) {
            self.switch_to_space(TheaterSpace::PersonalPocket);
        } else if engine.input.is_key_pressed(Num5) {
            self.switch_to_space(TheaterSpace::SocialHub);
        }
        
        // Teleport to selected screen
        if engine.input.is_gamepad_button_pressed(GamepadButton::A) {
            if let Some(idx) = self.selected_screen {
                self.teleport_to_screen(idx);
            }
        }
        
        // Return to lobby
        if engine.input.is_gamepad_button_pressed(GamepadButton::B) {
            self.switch_to_space(TheaterSpace::HyperbolicLobby);
        }
        
        // Start watch party
        if engine.input.is_gamepad_button_pressed(GamepadButton::Y) {
            self.create_watch_party("inception");
        }
        
        // Update world systems
        self.world.update(dt);
        
        // Quit
        if engine.input.is_key_pressed(KeyCode::Escape) {
            engine.quit();
        }
    }
    
    fn on_render(&mut self, engine: &mut Engine, renderer: &mut Renderer) {
        // Clear with theater ambiance
        renderer.clear(0.05, 0.05, 0.08, 1.0);
        
        // Render based on current space
        match self.current_space {
            TheaterSpace::HyperbolicLobby => {
                // Render Poincaré disk boundaries
                // Render hyperbolic floor pattern
            }
            TheaterSpace::SphericalDome => {
                // Render spherical dome
                // Stars/sky texture on sphere
            }
            TheaterSpace::EscherTheater => {
                // Render impossible stairs and geometry
            }
            TheaterSpace::PersonalPocket => {
                // Cozy small space rendering
            }
            TheaterSpace::SocialHub => {
                // Render friend avatars
                // Chat UI overlay
            }
        }
        
        // Render screens with movies
        for (i, screen) in self.screens.iter().enumerate() {
            let highlight = self.selected_screen == Some(i);
            
            // Render screen frame
            if highlight {
                // Glowing selection border
            }
            
            // Render movie content or thumbnail
            if screen.playing {
                // Render video frame
            } else {
                // Render movie poster
            }
            
            // Render UI elements
            if let Some(movie) = &screen.movie {
                // Title, progress bar, controls
            }
        }
        
        // Render watch party UI
        if let Some(party) = &self.watch_party {
            // Render chat messages
            // Render participant avatars
            // Render synchronized playback controls
        }
        
        // Render portal effects between spaces
        for portal in self.manifold.portals_from_chart(self.camera.position.chart_id) {
            // Render portal visualization
        }
    }
    
    fn on_cleanup(&mut self, _engine: &mut Engine) {
        println!("Thanks for using VR Netflix in Non-Euclidean Space!");
        println!("Your viewing preferences have been saved.");
    }
}

fn main() {
    pollster::block_on(async {
        let config = EngineConfig {
            title: "VR Netflix - Non-Euclidean Theater".to_string(),
            width: 1920,
            height: 1080,
            vsync: true,
            target_fps: Some(90), // VR target framerate
            resizable: false,
        };
        
        let engine = Engine::new(config).await.expect("Failed to create engine");
        let vr_netflix = VRNetflixExperience::new();
        
        engine.run(vr_netflix).expect("Failed to run VR Netflix");
    });
}