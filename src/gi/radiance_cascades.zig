/// Radiance Cascades — CPU-side orchestrator for the RC global illumination algorithm.
/// Manages cascade textures, compute dispatches, and parameter state.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const SceneTexture = @import("../scene/scene_texture.zig").SceneTexture;
const SdfGenerator = @import("../scene/sdf_generator.zig").SdfGenerator;

const rc_raymarch_wgsl = @embedFile("rc_raymarch.wgsl");
const rc_finalize_wgsl = @embedFile("rc_finalize.wgsl");

const RCPassParams = extern struct {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    _pad0: i32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

const FinalizeParams = extern struct {
    srgb_correction: i32,
    _pad0: i32 = 0,
    _pad1: i32 = 0,
    _pad2: i32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

pub const RadianceCascades = struct {
    width: u32,
    height: u32,

    // Output texture (what gets displayed)
    output_texture: zgpu.TextureHandle,
    output_view: zgpu.TextureViewHandle,

    // Cascade ping-pong textures
    cascade_textures: [2]zgpu.TextureHandle,
    cascade_views: [2]zgpu.TextureViewHandle,
    
    // A dummy empty texture to use as 'upper cascade' for the highest level
    empty_texture: zgpu.TextureHandle,
    empty_view: zgpu.TextureViewHandle,

    // Parameters (tweakable via ImGui)
    base_ray_count_log: i32 = 2, // log2: 1=2rays, 2=4rays, 3=8rays, 4=16rays
    srgb_correction: bool = true,
    sun_intensity: f32 = 1.0,
    cascade_count: u32 = 0,

    // Debug
    debug_show_sdf: bool = false,
    debug_show_cascades: bool = false,
    debug_cascade_index: i32 = 0,

    // Pipelines
    raymarch_pipeline: zgpu.ComputePipelineHandle,
    finalize_pipeline: zgpu.ComputePipelineHandle,

    // Bind group layouts
    raymarch_bgl: zgpu.BindGroupLayoutHandle,
    finalize_bgl: zgpu.BindGroupLayoutHandle,

    // Buffers
    rc_params_buffer: zgpu.BufferHandle,
    finalize_params_buffer: zgpu.BufferHandle,

    pub fn init(
        allocator: std.mem.Allocator,
        gctx: *zgpu.GraphicsContext,
        width: u32,
        height: u32,
    ) RadianceCascades {
        _ = allocator;

        const output_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true, .render_attachment = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const output_view = gctx.createTextureView(output_texture, .{});

        var cascade_textures: [2]zgpu.TextureHandle = undefined;
        var cascade_views: [2]zgpu.TextureViewHandle = undefined;
        for (0..2) |i| {
            cascade_textures[i] = gctx.createTexture(.{
                .usage = .{ .texture_binding = true, .storage_binding = true },
                .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
                .format = .rgba16_float,
                .mip_level_count = 1,
            });
            cascade_views[i] = gctx.createTextureView(cascade_textures[i], .{});
        }
        
        // Empty texture (1x1)
        const empty_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true },
            .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const empty_view = gctx.createTextureView(empty_texture, .{});

        const diagonal = @sqrt(@as(f32, @floatFromInt(width * width + height * height)));
        const base: f32 = 4.0;
        const count: u32 = @intFromFloat(@ceil(std.math.log(f32, base, diagonal)) + 1.0);

        const raymarch_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.textureEntry(1, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.textureEntry(2, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.storageTextureEntry(3, .{ .compute = true }, .write_only, .rgba16_float, .tvdim_2d),
            zgpu.bufferEntry(4, .{ .compute = true }, .uniform, false, @sizeOf(RCPassParams)),
        });

        const finalize_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.textureEntry(1, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.storageTextureEntry(2, .{ .compute = true }, .write_only, .rgba16_float, .tvdim_2d),
            zgpu.bufferEntry(3, .{ .compute = true }, .uniform, false, @sizeOf(FinalizeParams)),
        });

        const raymarch_pipeline = createComputePipeline(gctx, raymarch_bgl, rc_raymarch_wgsl, "rc_raymarch");
        const finalize_pipeline = createComputePipeline(gctx, finalize_bgl, rc_finalize_wgsl, "rc_finalize");

        const rc_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(RCPassParams) * count,
        });

        const finalize_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(FinalizeParams),
        });

        return .{
            .width = width,
            .height = height,
            .output_texture = output_texture,
            .output_view = output_view,
            .cascade_textures = cascade_textures,
            .cascade_views = cascade_views,
            .empty_texture = empty_texture,
            .empty_view = empty_view,
            .cascade_count = count,
            .raymarch_pipeline = raymarch_pipeline,
            .finalize_pipeline = finalize_pipeline,
            .raymarch_bgl = raymarch_bgl,
            .finalize_bgl = finalize_bgl,
            .rc_params_buffer = rc_params_buffer,
            .finalize_params_buffer = finalize_params_buffer,
        };
    }

    pub fn deinit(self: *RadianceCascades, allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext) void {
        _ = allocator;
        gctx.releaseResource(self.finalize_params_buffer);
        gctx.releaseResource(self.rc_params_buffer);
        gctx.releaseResource(self.finalize_bgl);
        gctx.releaseResource(self.raymarch_bgl);
        gctx.releaseResource(self.finalize_pipeline);
        gctx.releaseResource(self.raymarch_pipeline);
        gctx.releaseResource(self.empty_view);
        gctx.destroyResource(self.empty_texture);

        for (0..2) |i| {
            gctx.releaseResource(self.cascade_views[i]);
            gctx.destroyResource(self.cascade_textures[i]);
        }
        gctx.releaseResource(self.output_view);
        gctx.destroyResource(self.output_texture);
        self.* = undefined;
    }

    pub fn execute(
        self: *RadianceCascades,
        gctx: *zgpu.GraphicsContext,
        encoder: wgpu.CommandEncoder,
        scene: *SceneTexture,
        sdf: *SdfGenerator,
    ) void {
        const workgroups_x = (self.width + 7) / 8;
        const workgroups_y = (self.height + 7) / 8;

        const base_rays = @as(i32, 1) << @as(u5, @intCast(self.base_ray_count_log));

        var current_cascade: i32 = @as(i32, @intCast(self.cascade_count)) - 1;
        var upper_layer_view: zgpu.TextureViewHandle = self.empty_view;
        
        var write_layer: usize = 0;

        while (current_cascade >= 0) : (current_cascade -= 1) {
            const cascade_idx = @as(u32, @intCast(current_cascade));
            
            const params = RCPassParams{
                .cascade_index = current_cascade,
                .base_ray_count = base_rays,
                .sun_intensity = self.sun_intensity,
            };
            gctx.queue.writeBuffer(
                gctx.lookupResource(self.rc_params_buffer).?,
                cascade_idx * @sizeOf(RCPassParams),
                RCPassParams,
                &[_]RCPassParams{params},
            );

            {
                const pass = encoder.beginComputePass(null);
                const raymarch_bg = gctx.createBindGroup(self.raymarch_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = sdf.sdf_view },
                    .{ .binding = 1, .texture_view_handle = scene.texture_view },
                    .{ .binding = 2, .texture_view_handle = upper_layer_view },
                    .{ .binding = 3, .texture_view_handle = self.cascade_views[write_layer] },
                    .{ .binding = 4, .buffer_handle = self.rc_params_buffer, .offset = cascade_idx * @sizeOf(RCPassParams), .size = @sizeOf(RCPassParams) },
                });

                pass.setPipeline(gctx.lookupResource(self.raymarch_pipeline).?);
                pass.setBindGroup(0, gctx.lookupResource(raymarch_bg).?, &.{});
                pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
                pass.end();
                pass.release();
                gctx.releaseResource(raymarch_bg);
            }

            upper_layer_view = self.cascade_views[write_layer];
            write_layer = 1 - write_layer;
        }

        // Finalize
        var gi_input = upper_layer_view;
        if (self.debug_show_sdf) {
            gi_input = sdf.sdf_view;
        }

        const fin_p = FinalizeParams{
            .srgb_correction = if (self.srgb_correction) 1 else 0,
        };
        gctx.queue.writeBuffer(
            gctx.lookupResource(self.finalize_params_buffer).?,
            0,
            FinalizeParams,
            &[_]FinalizeParams{fin_p},
        );

        {
            const pass = encoder.beginComputePass(null);
            const finalize_bg = gctx.createBindGroup(self.finalize_bgl, &.{
                .{ .binding = 0, .texture_view_handle = gi_input },
                .{ .binding = 1, .texture_view_handle = scene.texture_view },
                .{ .binding = 2, .texture_view_handle = self.output_view },
                .{ .binding = 3, .buffer_handle = self.finalize_params_buffer, .offset = 0, .size = @sizeOf(FinalizeParams) },
            });

            pass.setPipeline(gctx.lookupResource(self.finalize_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(finalize_bg).?, &.{});
            pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(finalize_bg);
        }
    }
};

fn createComputePipeline(
    gctx: *zgpu.GraphicsContext,
    bgl: zgpu.BindGroupLayoutHandle,
    wgsl_source: [*:0]const u8,
    label: ?[*:0]const u8,
) zgpu.ComputePipelineHandle {
    const pl = gctx.createPipelineLayout(&.{bgl});
    defer gctx.releaseResource(pl);

    const cs_mod = zgpu.createWgslShaderModule(gctx.device, wgsl_source, label);
    defer cs_mod.release();

    return gctx.createComputePipeline(pl, .{
        .compute = .{
            .module = cs_mod,
            .entry_point = "main",
        },
    });
}
