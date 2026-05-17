/// Radiance Cascades — CPU-side orchestrator for the screen-space RC
/// global illumination pass. It owns the cascade textures and dispatches
/// three compute passes per frame:
///   1. raymarch  — sphere-traces the screen-space SDF and samples
///                  emissive radiance at every hit (one pass per cascade).
///   2. merge     — combines the current cascade with the already-merged
///                  parent cascade (one pass per cascade, top-down).
///   3. finalize  — produces the GI texture consumed by the deferred
///                  fragment shader.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;

const rc_raymarch_wgsl = @embedFile("rc_raymarch.wgsl");
const rc_merge_wgsl = @embedFile("rc_merge.wgsl");
const rc_finalize_wgsl = @embedFile("rc_finalize.wgsl");

const RCPassParams = extern struct {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    _pad0: i32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

const MergeParams = extern struct {
    cascade_index: i32,
    has_upper: i32,
    _pad0: i32 = 0,
    _pad1: i32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

const FinalizeParams = extern struct {
    intensity: f32,
    _pad0: f32 = 0,
    _pad1: f32 = 0,
    _pad2: f32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

pub const RadianceCascades = struct {
    width: u32,
    height: u32,

    // Final GI output sampled by deferred_fs.
    output_texture: zgpu.TextureHandle,
    output_view: zgpu.TextureViewHandle,

    // Raymarch destination (local radiance for the current cascade only).
    raymarch_texture: zgpu.TextureHandle,
    raymarch_view: zgpu.TextureViewHandle,

    // Ping-pong textures used by the merge pass.
    cascade_textures: [2]zgpu.TextureHandle,
    cascade_views: [2]zgpu.TextureViewHandle,

    // 1x1 dummy used as "upper cascade" for the highest level.
    empty_texture: zgpu.TextureHandle,
    empty_view: zgpu.TextureViewHandle,

    // Tweakables.
    base_ray_count_log: i32 = 2, // log2: 2=4rays at cascade 0
    sun_intensity: f32 = 1.0,
    gi_intensity: f32 = 1.0,
    cascade_count: u32 = 0,

    // Pipelines.
    raymarch_pipeline: zgpu.ComputePipelineHandle,
    merge_pipeline: zgpu.ComputePipelineHandle,
    finalize_pipeline: zgpu.ComputePipelineHandle,

    // Bind group layouts.
    raymarch_bgl: zgpu.BindGroupLayoutHandle,
    merge_bgl: zgpu.BindGroupLayoutHandle,
    finalize_bgl: zgpu.BindGroupLayoutHandle,

    // Uniform buffers (one slot per cascade for raymarch + merge).
    rc_params_buffer: zgpu.BufferHandle,
    merge_params_buffer: zgpu.BufferHandle,
    finalize_params_buffer: zgpu.BufferHandle,

    pub fn init(
        gctx: *zgpu.GraphicsContext,
        width: u32,
        height: u32,
    ) RadianceCascades {
        const output_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const output_view = gctx.createTextureView(output_texture, .{});

        const raymarch_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const raymarch_view = gctx.createTextureView(raymarch_texture, .{});

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
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false), // g_sdf
            zgpu.textureEntry(1, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false), // g_emissive
            zgpu.textureEntry(2, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false), // g_normal
            zgpu.storageTextureEntry(3, .{ .compute = true }, .write_only, .rgba16_float, .tvdim_2d),
            zgpu.bufferEntry(4, .{ .compute = true }, .uniform, false, @sizeOf(RCPassParams)),
        });

        const merge_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false), // current
            zgpu.textureEntry(1, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false), // upper
            zgpu.storageTextureEntry(2, .{ .compute = true }, .write_only, .rgba16_float, .tvdim_2d),
            zgpu.bufferEntry(3, .{ .compute = true }, .uniform, false, @sizeOf(MergeParams)),
        });

        const finalize_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .rgba16_float, .tvdim_2d),
            zgpu.bufferEntry(2, .{ .compute = true }, .uniform, false, @sizeOf(FinalizeParams)),
        });

        const raymarch_pipeline = createComputePipeline(gctx, raymarch_bgl, rc_raymarch_wgsl, "rc_raymarch");
        const merge_pipeline = createComputePipeline(gctx, merge_bgl, rc_merge_wgsl, "rc_merge");
        const finalize_pipeline = createComputePipeline(gctx, finalize_bgl, rc_finalize_wgsl, "rc_finalize");

        const rc_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(RCPassParams) * count,
        });
        const merge_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(MergeParams) * count,
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
            .raymarch_texture = raymarch_texture,
            .raymarch_view = raymarch_view,
            .cascade_textures = cascade_textures,
            .cascade_views = cascade_views,
            .empty_texture = empty_texture,
            .empty_view = empty_view,
            .cascade_count = count,
            .raymarch_pipeline = raymarch_pipeline,
            .merge_pipeline = merge_pipeline,
            .finalize_pipeline = finalize_pipeline,
            .raymarch_bgl = raymarch_bgl,
            .merge_bgl = merge_bgl,
            .finalize_bgl = finalize_bgl,
            .rc_params_buffer = rc_params_buffer,
            .merge_params_buffer = merge_params_buffer,
            .finalize_params_buffer = finalize_params_buffer,
        };
    }

    pub fn deinit(self: *RadianceCascades, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.finalize_params_buffer);
        gctx.releaseResource(self.merge_params_buffer);
        gctx.releaseResource(self.rc_params_buffer);
        gctx.releaseResource(self.finalize_bgl);
        gctx.releaseResource(self.merge_bgl);
        gctx.releaseResource(self.raymarch_bgl);
        gctx.releaseResource(self.finalize_pipeline);
        gctx.releaseResource(self.merge_pipeline);
        gctx.releaseResource(self.raymarch_pipeline);
        gctx.releaseResource(self.empty_view);
        gctx.destroyResource(self.empty_texture);

        for (0..2) |i| {
            gctx.releaseResource(self.cascade_views[i]);
            gctx.destroyResource(self.cascade_textures[i]);
        }
        gctx.releaseResource(self.raymarch_view);
        gctx.destroyResource(self.raymarch_texture);
        gctx.releaseResource(self.output_view);
        gctx.destroyResource(self.output_texture);
        self.* = undefined;
    }

    /// Execute the full RC pipeline: raymarch every cascade top-down,
    /// merge each one against the previous, then finalize the lowest
    /// cascade into the GI output texture.
    pub fn execute(
        self: *RadianceCascades,
        gctx: *zgpu.GraphicsContext,
        encoder: wgpu.CommandEncoder,
        g_sdf_view: zgpu.TextureViewHandle,
        g_emissive_view: zgpu.TextureViewHandle,
        g_normal_view: zgpu.TextureViewHandle,
    ) void {
        const workgroups_x = (self.width + 7) / 8;
        const workgroups_y = (self.height + 7) / 8;

        const base_rays = @as(i32, 1) << @as(u5, @intCast(self.base_ray_count_log));

        var current_cascade: i32 = @as(i32, @intCast(self.cascade_count)) - 1;
        var upper_layer_view: zgpu.TextureViewHandle = self.empty_view;
        var has_upper: i32 = 0;
        var write_layer: usize = 0;

        while (current_cascade >= 0) : (current_cascade -= 1) {
            const cascade_idx = @as(u32, @intCast(current_cascade));

            // --- Raymarch pass ---
            {
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

                const pass = encoder.beginComputePass(null);
                const bg = gctx.createBindGroup(self.raymarch_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = g_sdf_view },
                    .{ .binding = 1, .texture_view_handle = g_emissive_view },
                    .{ .binding = 2, .texture_view_handle = g_normal_view },
                    .{ .binding = 3, .texture_view_handle = self.raymarch_view },
                    .{ .binding = 4, .buffer_handle = self.rc_params_buffer, .offset = cascade_idx * @sizeOf(RCPassParams), .size = @sizeOf(RCPassParams) },
                });

                pass.setPipeline(gctx.lookupResource(self.raymarch_pipeline).?);
                pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
                pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
                pass.end();
                pass.release();
                gctx.releaseResource(bg);
            }

            // --- Merge pass ---
            {
                const m_params = MergeParams{
                    .cascade_index = current_cascade,
                    .has_upper = has_upper,
                };
                gctx.queue.writeBuffer(
                    gctx.lookupResource(self.merge_params_buffer).?,
                    cascade_idx * @sizeOf(MergeParams),
                    MergeParams,
                    &[_]MergeParams{m_params},
                );

                const pass = encoder.beginComputePass(null);
                const bg = gctx.createBindGroup(self.merge_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = self.raymarch_view },
                    .{ .binding = 1, .texture_view_handle = upper_layer_view },
                    .{ .binding = 2, .texture_view_handle = self.cascade_views[write_layer] },
                    .{ .binding = 3, .buffer_handle = self.merge_params_buffer, .offset = cascade_idx * @sizeOf(MergeParams), .size = @sizeOf(MergeParams) },
                });

                pass.setPipeline(gctx.lookupResource(self.merge_pipeline).?);
                pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
                pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
                pass.end();
                pass.release();
                gctx.releaseResource(bg);
            }

            upper_layer_view = self.cascade_views[write_layer];
            has_upper = 1;
            write_layer = 1 - write_layer;
        }

        // --- Finalize pass: project the fully merged cascade-0 into the
        // GI texture that the deferred fragment shader will sample.
        {
            const fin_p = FinalizeParams{ .intensity = self.gi_intensity };
            gctx.queue.writeBuffer(
                gctx.lookupResource(self.finalize_params_buffer).?,
                0,
                FinalizeParams,
                &[_]FinalizeParams{fin_p},
            );

            const pass = encoder.beginComputePass(null);
            const bg = gctx.createBindGroup(self.finalize_bgl, &.{
                .{ .binding = 0, .texture_view_handle = upper_layer_view },
                .{ .binding = 1, .texture_view_handle = self.output_view },
                .{ .binding = 2, .buffer_handle = self.finalize_params_buffer, .offset = 0, .size = @sizeOf(FinalizeParams) },
            });

            pass.setPipeline(gctx.lookupResource(self.finalize_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
            pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(bg);
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
