const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "rc_engine",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // --- zglfw ---
    const zglfw = b.dependency("zglfw", .{
        .target = target,
    });
    exe.root_module.addImport("zglfw", zglfw.module("root"));
    exe.linkLibrary(zglfw.artifact("glfw"));

    // --- zgpu (Dawn/WebGPU) ---
    @import("zgpu").addLibraryPathsTo(exe);
    const zgpu = b.dependency("zgpu", .{
        .target = target,
        // Required for the `timestamp-query` feature on Dawn/D3D12: the
        // feature is gated behind Dawn's "allow unsafe APIs" toggle.
        .dawn_allow_unsafe_apis = true,
    });
    exe.root_module.addImport("zgpu", zgpu.module("root"));
    exe.linkLibrary(zgpu.artifact("zdawn"));

    // --- zgui (ImGui with GLFW+wgpu backend) ---
    const zgui = b.dependency("zgui", .{
        .target = target,
        .backend = .glfw_wgpu,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    // --- zmath ---
    const zmath = b.dependency("zmath", .{
        .target = target,
    });
    exe.root_module.addImport("zmath", zmath.module("root"));

    // --- zstbi ---
    const zstbi = b.dependency("zstbi", .{
        .target = target,
    });
    exe.root_module.addImport("zstbi", zstbi.module("root"));

    // --- zmesh ---
    const zmesh = b.dependency("zmesh", .{
        .target = target,
    });
    exe.root_module.addImport("zmesh", zmesh.module("root"));

    // --- znoise ---
    const znoise = b.dependency("znoise", .{
        .target = target,
    });
    exe.root_module.addImport("znoise", znoise.module("root"));

    // --- Build options ---
    const exe_options = b.addOptions();
    exe.root_module.addOptions("build_options", exe_options);
    exe_options.addOption([]const u8, "content_dir", "content/");

    // --- Install content directory ---
    const install_content = b.addInstallDirectory(.{
        .source_dir = b.path("content"),
        .install_dir = .{ .custom = "" },
        .install_subdir = b.pathJoin(&.{ "bin", "content" }),
    });
    exe.step.dependOn(&install_content.step);

    // --- Platform SDK paths ---
    if (target.result.os.tag == .macos) {
        if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
            exe.addLibraryPath(system_sdk.path("macos12/usr/lib"));
            exe.addSystemFrameworkPath(system_sdk.path("macos12/System/Library/Frameworks"));
        }
    } else if (target.result.os.tag == .linux) {
        if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
            exe.addLibraryPath(system_sdk.path("linux/lib/x86_64-linux-gnu"));
        }
    }

    // --- LTO workaround for Windows ---
    if (target.result.os.tag == .windows) {
        exe.want_lto = false;
    }
    if (exe.root_module.optimize != .Debug) {
        exe.root_module.strip = true;
    }

    // --- Install + run steps ---
    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&install_exe.step);
    b.step("run", "Build and run RCEngine").dependOn(&run_cmd.step);
}
