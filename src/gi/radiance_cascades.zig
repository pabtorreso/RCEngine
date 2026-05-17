/// Radiance Cascades — CPU-side orchestrator for the screen-space RC
/// global illumination pass.
///
/// Implementation notes:
///   - Cascade `N` lives at resolution `width/2^N` × `height/2^N`
///     ("hierarchical cascades", à la Sannikov). Cascade 0 is full-res.
///     This trades a small loss of high-frequency detail for ~4×–5×
///     less raymarch work.
///   - Raymarch reads from the full-resolution G-Buffer (`g_sdf`,
///     `g_emissive`, `g_normal`) and writes to a per-cascade output
///     texture. The probe position in screen space is reconstructed
///     inside the shader from the cascade coords.
///   - Merge bilinearly upsamples the parent cascade onto the current
///     one, with four manual `textureLoad`s.
///   - Finalize just copies cascade 0 into the GI output, applying the
///     user-configurable intensity factor.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;

const rc_raymarch_wgsl = @embedFile("rc_raymarch.wgsl");
const rc_merge_wgsl = @embedFile("rc_merge.wgsl");
const rc_finalize_wgsl = @embedFile("rc_finalize.wgsl");

const MAX_CASCADES: u32 = 6;

const RCPassParams = extern struct {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    cascade_width: i32,
    cascade_height: i32,
    screen_width: i32,
    screen_height: i32,
    _pad0: i32 = 0,
    _padding: [56]u32 = [_]u32{0} ** 56,
};

const MergeParams = extern struct {
    cascade_index: i32,
    has_upper: i32,
    cascade_width: i32,
    cascade_height: i32,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

const FinalizeParams = extern struct {
    intensity: f32,
    _pad0: f32 = 0,
    _pad1: f32 = 0,
    _pad2: f32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60,
};

fn cascadeDim(full_dim: u32, n: u32) u32 {
    const shifted = full_dim >> @as(u5, @intCast(n));
    return @max(1, shifted);
}

pub const RadianceCascades = struct {
    width: u32,
    height: u32,
    cascade_count: u32,

    // Final GI output sampled by deferred_fs (always full-res).
    output_texture: zgpu.TextureHandle,
    output_view: zgpu.TextureViewHandle,

    // Per-cascade textures (cascade N at width/2^N × height/2^N).
    cascade_textures: [MAX_CASCADES]zgpu.TextureHandle,
    cascade_views: [MAX_CASCADES]zgpu.TextureViewHandle,
    raymarch_textures: [MAX_CASCADES]zgpu.TextureHandle,
    raymarch_views: [MAX_CASCADES]zgpu.TextureViewHandle,

    // 1×1 dummy bound as "upper cascade" for the highest level.
    empty_texture: zgpu.TextureHandle,
    empty_view: zgpu.TextureViewHandle,

    // Tweakables.
    base_ray_count_log: i32 = 2,
    sun_intensity: f32 = 1.0,
    gi_intensity: f32 = 1.0,

    // Pipelines + bind group layouts.
    raymarch_pipeline: zgpu.ComputePipelineHandle,
    merge_pipeline: zgpu.ComputePipelineHandle,
    finalize_pipeline: zgpu.ComputePipelineHandle,
    raymarch_bgl: zgpu.BindGroupLayoutHandle,
    merge_bgl: zgpu.BindGroupLayoutHandle,
    finalize_bgl: zgpu.BindGroupLayoutHandle,

    // Uniform buffers (one slot per cascade).
    rc_params_buffer: zgpu.BufferHandle,
    merge_params_buffer: zgpu.BufferHandle,
    finalize_params_buffer: zgpu.BufferHandle,

    pub fn init(
        gctx: *zgpu.GraphicsContext,
        width: u32,
        height: u32,
    ) RadianceCascades {
        const diagonal = @sqrt(@as(f32, @floatFromInt(width * width + height * height)));
        const base: f32 = 4.0;
        const natural_count: u32 = @intFromFloat(@ceil(std.math.log(f32, base, diagonal)));
        const cascade_count = @min(@max(natural_count, 1), MAX_CASCADES);

        const output_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const output_view = gctx.createTextureView(output_texture, .{});

        var cascade_textures: [MAX_CASCADES]zgpu.TextureHandle = undefined;
        var cascade_views: [MAX_CASCADES]zgpu.TextureViewHandle = undefined;
        var raymarch_textures: [MAX_CASCADES]zgpu.TextureHandle = undefined;
        var raymarch_views: [MAX_CASCADES]zgpu.TextureViewHandle = undefined;

        for (0..MAX_CASCADES) |i| {
            const n: u32 = @intCast(i);
            // Allocate textures even for unused slots so deinit can free
            // them uniformly; pick 1×1 for the slots past `cascade_count`.
            const w = if (n < cascade_count) cascadeDim(width, n) else 1;
            const h = if (n < cascade_count) cascadeDim(height, n) else 1;

            cascade_textures[i] = gctx.createTexture(.{
                .usage = .{ .texture_binding = true, .storage_binding = true },
                .size = .{ .width = w, .height = h, .depth_or_array_layers = 1 },
                .format = .rgba16_float,
                .mip_level_count = 1,
            });
            cascade_views[i] = gctx.createTextureView(cascade_textures[i], .{});

            raymarch_textures[i] = gctx.createTexture(.{
                .usage = .{ .texture_binding = true, .storage_binding = true },
                .size = .{ .width = w, .height = h, .depth_or_array_layers = 1 },
                .format = .rgba16_float,
                .mip_level_count = 1,
            });
            raymarch_views[i] = gctx.createTextureView(raymarch_textures[i], .{});
        }

        const empty_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true },
            .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
        });
        const empty_view = gctx.createTextureView(empty_texture, .{});

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
            .size = @sizeOf(RCPassParams) * MAX_CASCADES,
        });
        const merge_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(MergeParams) * MAX_CASCADES,
        });
        const finalize_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(FinalizeParams),
        });

        return .{
            .width = width,
            .height = height,
            .cascade_count = cascade_count,
            .output_texture = output_texture,
            .output_view = output_view,
            .cascade_textures = cascade_textures,
            .cascade_views = cascade_views,
            .raymarch_textures = raymarch_textures,
            .raymarch_views = raymarch_views,
            .empty_texture = empty_texture,
            .empty_view = empty_view,
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
        for (0..MAX_CASCADES) |i| {
            gctx.releaseResource(self.raymarch_views[i]);
            gctx.destroyResource(self.raymarch_textures[i]);
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
        g_sdf_view: zgpu.TextureViewHandle,
        g_emissive_view: zgpu.TextureViewHandle,
        g_normal_view: zgpu.TextureViewHandle,
    ) void {
        const base_rays = @as(i32, 1) << @as(u5, @intCast(self.base_ray_count_log));

        var current_cascade: i32 = @as(i32, @intCast(self.cascade_count)) - 1;
        var has_upper: i32 = 0;
        var upper_view: zgpu.TextureViewHandle = self.empty_view;

        while (current_cascade >= 0) : (current_cascade -= 1) {
            const idx: u32 = @intCast(current_cascade);
            const cw = cascadeDim(self.width, idx);
            const ch = cascadeDim(self.height, idx);
            const wg_x = (cw + 7) / 8;
            const wg_y = (ch + 7) / 8;

            // --- Raymarch pass for this cascade ---
            {
                const params = RCPassParams{
                    .cascade_index = current_cascade,
                    .base_ray_count = base_rays,
                    .sun_intensity = self.sun_intensity,
                    .cascade_width = @intCast(cw),
                    .cascade_height = @intCast(ch),
                    .screen_width = @intCast(self.width),
                    .screen_height = @intCast(self.height),
                };
                gctx.queue.writeBuffer(
                    gctx.lookupResource(self.rc_params_buffer).?,
                    idx * @sizeOf(RCPassParams),
                    RCPassParams,
                    &[_]RCPassParams{params},
                );

                const pass = encoder.beginComputePass(null);
                const bg = gctx.createBindGroup(self.raymarch_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = g_sdf_view },
                    .{ .binding = 1, .texture_view_handle = g_emissive_view },
                    .{ .binding = 2, .texture_view_handle = g_normal_view },
                    .{ .binding = 3, .texture_view_handle = self.raymarch_views[idx] },
                    .{ .binding = 4, .buffer_handle = self.rc_params_buffer, .offset = idx * @sizeOf(RCPassParams), .size = @sizeOf(RCPassParams) },
                });

                pass.setPipeline(gctx.lookupResource(self.raymarch_pipeline).?);
                pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
                pass.dispatchWorkgroups(wg_x, wg_y, 1);
                pass.end();
                pass.release();
                gctx.releaseResource(bg);
            }

            // --- Merge pass for this cascade ---
            {
                const m_params = MergeParams{
                    .cascade_index = current_cascade,
                    .has_upper = has_upper,
                    .cascade_width = @intCast(cw),
                    .cascade_height = @intCast(ch),
                };
                gctx.queue.writeBuffer(
                    gctx.lookupResource(self.merge_params_buffer).?,
                    idx * @sizeOf(MergeParams),
                    MergeParams,
                    &[_]MergeParams{m_params},
                );

                const pass = encoder.beginComputePass(null);
                const bg = gctx.createBindGroup(self.merge_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = self.raymarch_views[idx] },
                    .{ .binding = 1, .texture_view_handle = upper_view },
                    .{ .binding = 2, .texture_view_handle = self.cascade_views[idx] },
                    .{ .binding = 3, .buffer_handle = self.merge_params_buffer, .offset = idx * @sizeOf(MergeParams), .size = @sizeOf(MergeParams) },
                });

                pass.setPipeline(gctx.lookupResource(self.merge_pipeline).?);
                pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
                pass.dispatchWorkgroups(wg_x, wg_y, 1);
                pass.end();
                pass.release();
                gctx.releaseResource(bg);
            }

            upper_view = self.cascade_views[idx];
            has_upper = 1;
        }

        // --- Finalize: cascade 0 is full-res, just scale & copy. ---
        {
            const wg_x = (self.width + 7) / 8;
            const wg_y = (self.height + 7) / 8;

            const fin_p = FinalizeParams{ .intensity = self.gi_intensity };
            gctx.queue.writeBuffer(
                gctx.lookupResource(self.finalize_params_buffer).?,
                0,
                FinalizeParams,
                &[_]FinalizeParams{fin_p},
            );

            const pass = encoder.beginComputePass(null);
            const bg = gctx.createBindGroup(self.finalize_bgl, &.{
                .{ .binding = 0, .texture_view_handle = self.cascade_views[0] },
                .{ .binding = 1, .texture_view_handle = self.output_view },
                .{ .binding = 2, .buffer_handle = self.finalize_params_buffer, .offset = 0, .size = @sizeOf(FinalizeParams) },
            });

            pass.setPipeline(gctx.lookupResource(self.finalize_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
            pass.dispatchWorkgroups(wg_x, wg_y, 1);
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
