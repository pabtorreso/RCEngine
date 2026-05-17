/// Draw system — handles mouse input for interactive scene painting.
/// Left mouse button draws lights or occluders, right button erases.
const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const SceneTexture = @import("scene_texture.zig").SceneTexture;

pub const DrawSystem = struct {
    /// 0 = draw light, 1 = draw occluder
    draw_mode: i32 = 0,

    /// RGB color for light sources (0.0–1.0)
    light_color: [3]f32 = .{ 1.0, 0.9, 0.7 },

    /// Brush radius in pixels
    brush_size: f32 = 12.0,

    /// Previous cursor position for smooth line drawing
    prev_x: f64 = 0,
    prev_y: f64 = 0,
    was_drawing: bool = false,

    pub fn init() DrawSystem {
        return .{};
    }

    pub fn update(
        self: *DrawSystem,
        window: *zglfw.Window,
        scene: *SceneTexture,
        gctx: *zgpu.GraphicsContext,
        scene_width: u32,
        scene_height: u32,
    ) void {
        // Don't draw if ImGui wants the mouse
        const zgui = @import("zgui");
        if (zgui.io.getWantCaptureMouse()) {
            self.was_drawing = false;
            return;
        }

        const cursor = window.getCursorPos();
        const fb_size = window.getFramebufferSize();

        // Map window coordinates to scene coordinates
        const scene_x = cursor[0] * @as(f64, @floatFromInt(scene_width)) / @as(f64, @floatFromInt(fb_size[0]));
        const scene_y = cursor[1] * @as(f64, @floatFromInt(scene_height)) / @as(f64, @floatFromInt(fb_size[1]));

        const left_pressed = window.getMouseButton(.left) == .press;
        const right_pressed = window.getMouseButton(.right) == .press;

        if (left_pressed) {
            // Draw mode
            if (self.was_drawing) {
                // Interpolate from previous position for smooth strokes
                self.drawLine(scene, self.prev_x, self.prev_y, scene_x, scene_y);
            } else {
                self.drawAt(scene, scene_x, scene_y);
            }
            self.was_drawing = true;
            scene.upload(gctx);
        } else if (right_pressed) {
            // Erase mode
            if (self.was_drawing) {
                self.eraseLine(scene, self.prev_x, self.prev_y, scene_x, scene_y);
            } else {
                scene.eraseCircle(@floatCast(scene_x), @floatCast(scene_y), self.brush_size);
            }
            self.was_drawing = true;
            scene.upload(gctx);
        } else {
            self.was_drawing = false;
        }

        self.prev_x = scene_x;
        self.prev_y = scene_y;
    }

    fn drawAt(self: *const DrawSystem, scene: *SceneTexture, x: f64, y: f64) void {
        const fx: f32 = @floatCast(x);
        const fy: f32 = @floatCast(y);

        if (self.draw_mode == 0) {
            // Light: emissive color with full opacity
            scene.drawCircle(
                fx,
                fy,
                self.brush_size,
                @intFromFloat(self.light_color[0] * 255.0),
                @intFromFloat(self.light_color[1] * 255.0),
                @intFromFloat(self.light_color[2] * 255.0),
                255,
            );
        } else {
            // Occluder: black with full opacity
            scene.drawCircle(fx, fy, self.brush_size, 10, 10, 10, 255);
        }
    }

    fn drawLine(self: *const DrawSystem, scene: *SceneTexture, x0: f64, y0: f64, x1: f64, y1: f64) void {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const dist = @sqrt(dx * dx + dy * dy);
        const steps: u32 = @intFromFloat(@max(1.0, dist / (@as(f64, self.brush_size) * 0.5)));

        var i: u32 = 0;
        while (i <= steps) : (i += 1) {
            const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
            const x = x0 + dx * t;
            const y = y0 + dy * t;
            self.drawAt(scene, x, y);
        }
    }

    fn eraseLine(self: *const DrawSystem, scene: *SceneTexture, x0: f64, y0: f64, x1: f64, y1: f64) void {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const dist = @sqrt(dx * dx + dy * dy);
        const steps: u32 = @intFromFloat(@max(1.0, dist / (@as(f64, self.brush_size) * 0.5)));

        var i: u32 = 0;
        while (i <= steps) : (i += 1) {
            const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
            const x = x0 + dx * t;
            const y = y0 + dy * t;
            scene.eraseCircle(@floatCast(x), @floatCast(y), self.brush_size);
        }
    }
};
