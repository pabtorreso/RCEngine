/// Thin wrapper for creating the zgpu.GraphicsContext from a GLFW window.
/// Extracted from the zig-gamedev sample pattern to avoid copy-pasting boilerplate.
const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

pub fn createGraphicsContext(
    allocator: std.mem.Allocator,
    window: *zglfw.Window,
) !*zgpu.GraphicsContext {
    return try zgpu.GraphicsContext.create(
        allocator,
        .{
            .window = window,
            .fn_getTime = @ptrCast(&zglfw.getTime),
            .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),
            .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
            .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
            .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
            .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
            .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
            .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
        },
        .{},
    );
}
