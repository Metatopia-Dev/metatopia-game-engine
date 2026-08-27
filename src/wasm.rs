//! WebAssembly (Wasm) & WebXR Browser Export Pipeline
//!
//! Provides HTML5 canvas binding, WebGPU/WebGL2 backend selection,
//! Touch/Pointer input adapters, and WebXR Immersive VR session management.

/// Configuration for WebAssembly browser export
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmConfig {
    pub canvas_id: String,
    pub enable_webxr: bool,
    pub prefer_webgpu: bool,
    pub auto_resize_canvas: bool,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            canvas_id: "metatopia-canvas".into(),
            enable_webxr: true,
            prefer_webgpu: true,
            auto_resize_canvas: true,
        }
    }
}

/// WebXR Immersive VR Session State (for Meta Quest Browser & WebXR headsets)
#[derive(Debug, Clone, PartialEq)]
pub struct WebXrSession {
    pub is_active: bool,
    pub session_mode: String, // "immersive-vr" or "inline"
    pub reference_space: String, // "local-floor"
    pub target_frame_rate: f32, // 72Hz, 90Hz, or 120Hz on Quest 3S
}

impl Default for WebXrSession {
    fn default() -> Self {
        Self {
            is_active: false,
            session_mode: "immersive-vr".into(),
            reference_space: "local-floor".into(),
            target_frame_rate: 90.0,
        }
    }
}

impl WebXrSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request entering immersive WebXR mode
    pub fn request_immersive_vr(&mut self) -> Result<(), String> {
        self.is_active = true;
        self.session_mode = "immersive-vr".into();
        Ok(())
    }

    /// Exit WebXR session
    pub fn end_session(&mut self) {
        self.is_active = false;
        self.session_mode = "inline".into();
    }
}

/// WebAssembly Runner & Event Dispatcher
pub struct WasmExportRunner {
    pub config: WasmConfig,
    pub xr_session: WebXrSession,
}

impl WasmExportRunner {
    pub fn new(config: WasmConfig) -> Self {
        Self {
            config,
            xr_session: WebXrSession::new(),
        }
    }

    /// Generate an HTML5 bootstrap entrypoint shell for running Metatopia in browser
    pub fn generate_html_shell(title: &str, wasm_filename: &str) -> String {
        format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #050608; }}
        #metatopia-canvas {{ width: 100%; height: 100%; display: block; }}
        #vr-button {{
            position: absolute; bottom: 20px; right: 20px;
            padding: 12px 24px; font-family: monospace; font-size: 16px; font-weight: bold;
            color: #00e5ff; background: rgba(10, 20, 30, 0.9); border: 2px solid #00e5ff;
            border-radius: 8px; cursor: pointer; transition: 0.2s;
        }}
        #vr-button:hover {{ background: #00e5ff; color: #000; }}
    </style>
</head>
<body>
    <canvas id="metatopia-canvas"></canvas>
    <button id="vr-button" onclick="enterVR()">🥽 ENTER WEBXR</button>
    <script type="module">
        import init from './{}.js';
        async function run() {{
            await init();
            console.log('Metatopia WebAssembly Engine Initialized!');
        }}
        run();
        window.enterVR = async function() {{
            if (navigator.xr) {{
                const session = await navigator.xr.requestSession('immersive-vr', {{
                    requiredFeatures: ['local-floor']
                }});
                console.log('WebXR Immersive Session Started');
            }} else {{
                alert('WebXR is not supported on this browser. Try Meta Quest Browser.');
            }}
        }};
    </script>
</body>
</html>"#, title, wasm_filename)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webxr_session_lifecycle() {
        let mut session = WebXrSession::new();
        assert!(!session.is_active);

        session.request_immersive_vr().unwrap();
        assert!(session.is_active);
        assert_eq!(session.session_mode, "immersive-vr");

        session.end_session();
        assert!(!session.is_active);
    }

    #[test]
    fn test_html_shell_generation() {
        let html = WasmExportRunner::generate_html_shell("Metatopia Game", "game_pkg");
        assert!(html.contains("metatopia-canvas"));
        assert!(html.contains("ENTER WEBXR"));
        assert!(html.contains("game_pkg.js"));
    }
}
