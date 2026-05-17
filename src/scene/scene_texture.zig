/// Scene texture — holds the scene data as a GPU texture.
/// RGBA format where:
///   RGB = emissive color (light sources glow, occluders are dark)
///   A   = opacity (1.0 = solid surface or light, 0.0 = empty space)
///
/// The CPU keeps a staging copy for draw operations, then uploads to GPU.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;

pub const SceneTexture = struct {
    width: u32,
    height: u32,

    /// CPU-side pixel data (RGBA8 Unorm, row-major)
    pixels: []u8,
    allocator: std.mem.Allocator,

    /// GPU texture (used as input to SDF + RC)
    texture: zgpu.TextureHandle,
    texture_view: zgpu.TextureViewHandle,

    /// Tracks whether CPU data has been modified and needs re-upload.
    dirty: bool,

    pub fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext, width: u32, height: u32) SceneTexture {
        const pixel_count = width * height * 4;
        const pixels = allocator.alloc(u8, pixel_count) catch unreachable;
        @memset(pixels, 0); // Start with empty (transparent black) scene

        const texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .copy_dst = true, .storage_binding = true },
            .dimension = .tdim_2d,
            .size = .{
                .width = width,
                .height = height,
                .depth_or_array_layers = 1,
            },
            .format = .rgba8_unorm,
            .mip_level_count = 1,
            .sample_count = 1,
        });
        const texture_view = gctx.createTextureView(texture, .{});

        var self = SceneTexture{
            .width = width,
            .height = height,
            .pixels = pixels,
            .allocator = allocator,
            .texture = texture,
            .texture_view = texture_view,
            .dirty = true,
        };

        // Upload initial empty scene
        self.upload(gctx);

        return self;
    }

    pub fn deinit(self: *SceneTexture, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.texture_view);
        gctx.destroyResource(self.texture);
        self.allocator.free(self.pixels);
        self.* = undefined;
    }

    /// Set a pixel at (x, y) to the given RGBA color.
    pub fn setPixel(self: *SceneTexture, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) void {
        if (x >= self.width or y >= self.height) return;
        const idx = (y * self.width + x) * 4;
        self.pixels[idx + 0] = r;
        self.pixels[idx + 1] = g;
        self.pixels[idx + 2] = b;
        self.pixels[idx + 3] = a;
        self.dirty = true;
    }

    /// Draw a filled circle at (cx, cy) with given radius and color.
    pub fn drawCircle(self: *SceneTexture, cx: f32, cy: f32, radius: f32, r: u8, g: u8, b: u8, a: u8) void {
        const r2 = radius * radius;
        const min_x = @as(u32, @intFromFloat(@max(0.0, cx - radius)));
        const max_x = @as(u32, @intFromFloat(@min(@as(f32, @floatFromInt(self.width - 1)), cx + radius)));
        const min_y = @as(u32, @intFromFloat(@max(0.0, cy - radius)));
        const max_y = @as(u32, @intFromFloat(@min(@as(f32, @floatFromInt(self.height - 1)), cy + radius)));

        var py = min_y;
        while (py <= max_y) : (py += 1) {
            var px = min_x;
            while (px <= max_x) : (px += 1) {
                const dx = @as(f32, @floatFromInt(px)) - cx;
                const dy = @as(f32, @floatFromInt(py)) - cy;
                if (dx * dx + dy * dy <= r2) {
                    self.setPixel(px, py, r, g, b, a);
                }
            }
        }
    }

    /// Erase a filled circle (set to transparent black).
    pub fn eraseCircle(self: *SceneTexture, cx: f32, cy: f32, radius: f32) void {
        self.drawCircle(cx, cy, radius, 0, 0, 0, 0);
    }

    /// Clear the entire scene to empty.
    pub fn clear(self: *SceneTexture, gctx: *zgpu.GraphicsContext) void {
        @memset(self.pixels, 0);
        self.dirty = true;
        self.upload(gctx);
    }

    /// Upload CPU pixel data to GPU if dirty.
    pub fn upload(self: *SceneTexture, gctx: *zgpu.GraphicsContext) void {
        if (!self.dirty) return;

        const gpu_tex = gctx.lookupResource(self.texture) orelse return;
        gctx.queue.writeTexture(
            .{ .texture = gpu_tex },
            .{
                .bytes_per_row = self.width * 4,
                .rows_per_image = self.height,
            },
            .{ .width = self.width, .height = self.height },
            u8,
            self.pixels,
        );
        self.dirty = false;
    }
};
