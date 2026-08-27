//! 2D and 3D Immediate-Mode UI & HUD Canvas
//!
//! Provides layout anchors (TopLeft, Center, BottomRight), health/progress bars,
//! floating 3D world damage popups, crosshairs, and minimap radar blip batches.

use crate::quickstart::GameVertex;

/// Screen layout anchor points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// 2D Screen-space axis-aligned rectangle
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

/// Floating 3D world damage or notification text/billboard
#[derive(Debug, Clone)]
pub struct DamagePopup {
    pub world_pos: [f32; 3],
    pub color: [f32; 4],
    pub lifetime: f32,
    pub max_life: f32,
    pub scale: f32,
}

/// Colored 2D UI Quad
#[derive(Debug, Clone)]
pub struct UiQuad {
    pub rect: Rect,
    pub color: [f32; 4],
}

/// Immediate-mode HUD and UI Canvas
#[derive(Debug, Clone)]
pub struct UiCanvas {
    pub width: f32,
    pub height: f32,
    pub quads: Vec<UiQuad>,
    pub popups: Vec<DamagePopup>,
}

impl UiCanvas {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            quads: Vec::new(),
            popups: Vec::new(),
        }
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    /// Clear all draw commands for the new frame
    pub fn clear(&mut self) {
        self.quads.clear();
    }

    /// Convert anchor + offset into absolute screen coordinates
    pub fn anchor_pos(&self, anchor: Anchor, offset_x: f32, offset_y: f32) -> (f32, f32) {
        match anchor {
            Anchor::TopLeft => (offset_x, offset_y),
            Anchor::TopCenter => (self.width * 0.5 + offset_x, offset_y),
            Anchor::TopRight => (self.width - offset_x, offset_y),
            Anchor::CenterLeft => (offset_x, self.height * 0.5 + offset_y),
            Anchor::Center => (self.width * 0.5 + offset_x, self.height * 0.5 + offset_y),
            Anchor::CenterRight => (self.width - offset_x, self.height * 0.5 + offset_y),
            Anchor::BottomLeft => (offset_x, self.height - offset_y),
            Anchor::BottomCenter => (self.width * 0.5 + offset_x, self.height - offset_y),
            Anchor::BottomRight => (self.width - offset_x, self.height - offset_y),
        }
    }

    /// Draw a colored solid 2D rectangle with anchor
    pub fn draw_rect(&mut self, anchor: Anchor, offset: (f32, f32), size: (f32, f32), color: [f32; 4]) -> Rect {
        let (ax, ay) = self.anchor_pos(anchor, offset.0, offset.1);
        let rect = match anchor {
            Anchor::TopLeft => Rect::new(ax, ay, size.0, size.1),
            Anchor::TopCenter | Anchor::BottomCenter | Anchor::Center => {
                Rect::new(ax - size.0 * 0.5, ay - size.1 * 0.5, size.0, size.1)
            }
            Anchor::TopRight | Anchor::BottomRight | Anchor::CenterRight => {
                Rect::new(ax - size.0, ay, size.0, size.1)
            }
            Anchor::BottomLeft | Anchor::CenterLeft => {
                Rect::new(ax, ay - size.1, size.0, size.1)
            }
        };

        self.quads.push(UiQuad { rect, color });
        rect
    }

    /// Draw an animated progress / health bar with background and fill
    pub fn draw_progress_bar(
        &mut self,
        anchor: Anchor,
        offset: (f32, f32),
        size: (f32, f32),
        fill_ratio: f32,
        fill_color: [f32; 4],
        bg_color: [f32; 4],
    ) {
        let bg_rect = self.draw_rect(anchor, offset, size, bg_color);
        let ratio = fill_ratio.clamp(0.0, 1.0);
        if ratio > 0.001 {
            let fill_rect = Rect::new(bg_rect.x, bg_rect.y, bg_rect.w * ratio, bg_rect.h);
            self.quads.push(UiQuad { rect: fill_rect, color: fill_color });
        }
    }

    /// Draw a crosshair at center
    pub fn draw_crosshair(&mut self, size: f32, thickness: f32, color: [f32; 4]) {
        let cx = self.width * 0.5;
        let cy = self.height * 0.5;

        // Horizontal bar
        self.quads.push(UiQuad {
            rect: Rect::new(cx - size * 0.5, cy - thickness * 0.5, size, thickness),
            color,
        });

        // Vertical bar
        self.quads.push(UiQuad {
            rect: Rect::new(cx - thickness * 0.5, cy - size * 0.5, thickness, size),
            color,
        });
    }

    /// Spawn a floating damage number in 3D world space
    pub fn add_damage_popup(&mut self, world_pos: [f32; 3], color: [f32; 4], lifetime: f32) {
        self.popups.push(DamagePopup {
            world_pos,
            color,
            lifetime: 0.0,
            max_life: lifetime.max(0.1),
            scale: 0.3,
        });
    }

    /// Step simulation for floating popups
    pub fn update(&mut self, dt: f32) {
        for popup in &mut self.popups {
            popup.lifetime += dt;
            popup.world_pos[1] += 1.5 * dt; // Float upwards
        }
        self.popups.retain(|p| p.lifetime < p.max_life);
    }

    /// Build billboard meshes for 3D world damage popups
    pub fn build_3d_popups_mesh(&self, camera_pos: [f32; 3]) -> (Vec<GameVertex>, Vec<u32>) {
        let mut verts = Vec::with_capacity(self.popups.len() * 4);
        let mut idxs = Vec::with_capacity(self.popups.len() * 6);

        for popup in &self.popups {
            let t = popup.lifetime / popup.max_life;
            let alpha = (popup.color[3] * (1.0 - t)).max(0.0);
            let s = popup.scale * (1.0 + t * 0.2);

            let p = popup.world_pos;
            let rgb = [popup.color[0], popup.color[1], popup.color[2]];
            let pbr = [0.0, 0.0, alpha * 4.0, 0.0];

            let dx = p[0] - camera_pos[0];
            let dz = p[2] - camera_pos[2];
            let len = (dx*dx + dz*dz).sqrt().max(0.001);
            let right = [-dz / len * s, 0.0, dx / len * s];
            let up = [0.0, s, 0.0];

            let p0 = [p[0] - right[0], p[1] - up[1], p[2] - right[2]];
            let p1 = [p[0] + right[0], p[1] - up[1], p[2] + right[2]];
            let p2 = [p[0] + right[0], p[1] + up[1], p[2] + right[2]];
            let p3 = [p[0] - right[0], p[1] + up[1], p[2] - right[2]];

            let base = verts.len() as u32;
            let norm = [0.0, 1.0, 0.0];

            verts.push(GameVertex::new(p0, norm, rgb, pbr));
            verts.push(GameVertex::new(p1, norm, rgb, pbr));
            verts.push(GameVertex::new(p2, norm, rgb, pbr));
            verts.push(GameVertex::new(p3, norm, rgb, pbr));

            idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        (verts, idxs)
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_anchor_positioning() {
        let canvas = UiCanvas::new(1920.0, 1080.0);
        let (cx, cy) = canvas.anchor_pos(Anchor::Center, 0.0, 0.0);
        assert!((cx - 960.0).abs() < 1e-4);
        assert!((cy - 540.0).abs() < 1e-4);

        let (br_x, br_y) = canvas.anchor_pos(Anchor::BottomRight, 20.0, 20.0);
        assert!((br_x - 1900.0).abs() < 1e-4);
        assert!((br_y - 1060.0).abs() < 1e-4);
    }

    #[test]
    fn test_progress_bar_drawing() {
        let mut canvas = UiCanvas::new(800.0, 600.0);
        canvas.draw_progress_bar(
            Anchor::BottomCenter,
            (0.0, 40.0),
            (200.0, 20.0),
            0.5,
            [1.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2, 1.0],
        );
        assert_eq!(canvas.quads.len(), 2);
        assert_eq!(canvas.quads[1].rect.w, 100.0); // 50% of 200.0
    }
}
