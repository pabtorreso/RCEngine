/// SDF Generator — creates a distance field from the scene texture using
/// the Jump Flood Algorithm (JFA) in compute shaders.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const SceneTexture = @import("scene_texture.zig").SceneTexture;
const jfa_wgsl = @embedFile("../gi/jfa.wgsl");

const StepParams = extern struct {
    step_size: i32,
    _pad0: i32 = 0,
    _pad1: i32 = 0,
    _pad2: i32 = 0,
    _padding: [60]u32 = [_]u32{0} ** 60, // 64 * 4 = 256 bytes total
};

pub const MAX_SDF_QUALITY_LEVEL: u32 = 2; // 0=Ultra, 1=High (half), 2=Performance (quarter)

pub const SdfGenerator = struct {
    width: u32,  // effective SDF width (after quality shift)
    height: u32,
    quality_level: u32,
    sdf_texture: zgpu.TextureHandle,
    sdf_view: zgpu.TextureViewHandle,
    jfa_textures: [2]zgpu.TextureHandle,
    jfa_views: [2]zgpu.TextureViewHandle,
    step_pass_count: u32,

    // Pipelines
    init_pipeline: zgpu.ComputePipelineHandle,
    step_pipeline: zgpu.ComputePipelineHandle,
    distance_pipeline: zgpu.ComputePipelineHandle,

    // Bind groups
    init_bgl: zgpu.BindGroupLayoutHandle,
    step_bgl: zgpu.BindGroupLayoutHandle,
    distance_bgl: zgpu.BindGroupLayoutHandle,

    // Uniforms
    step_params_buffer: zgpu.BufferHandle,

    pub fn init(gctx: *zgpu.GraphicsContext, screen_width: u32, screen_height: u32, quality_level: u32) SdfGenerator {
        const q = @min(quality_level, MAX_SDF_QUALITY_LEVEL);
        const width = @max(1, screen_width >> @as(u5, @intCast(q)));
        const height = @max(1, screen_height >> @as(u5, @intCast(q)));
        var jfa_textures: [2]zgpu.TextureHandle = undefined;
        var jfa_views: [2]zgpu.TextureViewHandle = undefined;
        for (0..2) |i| {
            jfa_textures[i] = gctx.createTexture(.{
                .usage = .{ .texture_binding = true, .storage_binding = true },
                .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
                .format = .rg32_float,
                .mip_level_count = 1,
            });
            jfa_views[i] = gctx.createTextureView(jfa_textures[i], .{});
        }

        const sdf_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .r32_float,
            .mip_level_count = 1,
        });
        const sdf_view = gctx.createTextureView(sdf_texture, .{});

        const max_dim = @max(width, height);
        var pass_count: u32 = 0;
        var s = max_dim;
        while (s >= 1) : (s /= 2) {
            pass_count += 1;
        }

        // Create bind group layouts
        const init_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .depth, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .rg32_float, .tvdim_2d),
        });

        const step_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .rg32_float, .tvdim_2d),
            zgpu.bufferEntry(2, .{ .compute = true }, .uniform, false, @sizeOf(StepParams)),
        });

        const distance_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .unfilterable_float, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_float, .tvdim_2d),
        });

        // Create pipelines
        const init_pipeline = createComputePipeline(gctx, init_bgl, jfa_wgsl, "init_main", "jfa_init");
        const step_pipeline = createComputePipeline(gctx, step_bgl, jfa_wgsl, "step_main", "jfa_step");
        const distance_pipeline = createComputePipeline(gctx, distance_bgl, jfa_wgsl, "distance_main", "jfa_dist");

        // Uniform buffer for step params
        const step_params_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(StepParams) * pass_count,
        });

        return .{
            .width = width,
            .height = height,
            .quality_level = q,
            .jfa_textures = jfa_textures,
            .jfa_views = jfa_views,
            .sdf_texture = sdf_texture,
            .sdf_view = sdf_view,
            .step_pass_count = pass_count,
            .init_pipeline = init_pipeline,
            .step_pipeline = step_pipeline,
            .distance_pipeline = distance_pipeline,
            .init_bgl = init_bgl,
            .step_bgl = step_bgl,
            .distance_bgl = distance_bgl,
            .step_params_buffer = step_params_buffer,
        };
    }

    pub fn deinit(self: *SdfGenerator, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.step_params_buffer);
        gctx.releaseResource(self.distance_bgl);
        gctx.releaseResource(self.step_bgl);
        gctx.releaseResource(self.init_bgl);
        gctx.releaseResource(self.distance_pipeline);
        gctx.releaseResource(self.step_pipeline);
        gctx.releaseResource(self.init_pipeline);
        gctx.releaseResource(self.sdf_view);
        gctx.destroyResource(self.sdf_texture);
        for (0..2) |i| {
            gctx.releaseResource(self.jfa_views[i]);
            gctx.destroyResource(self.jfa_textures[i]);
        }
        self.* = undefined;
    }

    /// Generate SDF from scene texture. Called every frame.
    pub fn generate(self: *SdfGenerator, gctx: *zgpu.GraphicsContext, encoder: wgpu.CommandEncoder, depth_texv: zgpu.TextureViewHandle) void {
        const workgroups_x = (self.width + 7) / 8;
        const workgroups_y = (self.height + 7) / 8;

        // 1. Init pass
        {
            const pass = encoder.beginComputePass(null);
            const init_bg = gctx.createBindGroup(self.init_bgl, &.{
                .{ .binding = 0, .texture_view_handle = depth_texv },
                .{ .binding = 1, .texture_view_handle = self.jfa_views[0] },
            });

            const pipeline = gctx.lookupResource(self.init_pipeline) orelse return;
            const bind_group = gctx.lookupResource(init_bg) orelse return;

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind_group, &.{});
            pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(init_bg);
        }

        // 2. Step passes
        var current_read: usize = 0;
        var step_size: i32 = @divTrunc(@as(i32, @intCast(@max(self.width, self.height))), 2);
        var step_idx: u32 = 0;

        while (step_size >= 1) : ({
            step_size = @divTrunc(step_size, 2);
            step_idx += 1;
        }) {
            const current_write = 1 - current_read;

            // Write uniform
            const params = StepParams{ .step_size = step_size };
            gctx.queue.writeBuffer(
                gctx.lookupResource(self.step_params_buffer).?,
                step_idx * @sizeOf(StepParams),
                StepParams,
                &[_]StepParams{params},
            );

            const pass = encoder.beginComputePass(null);
            const step_bg = gctx.createBindGroup(self.step_bgl, &.{
                .{ .binding = 0, .texture_view_handle = self.jfa_views[current_read] },
                .{ .binding = 1, .texture_view_handle = self.jfa_views[current_write] },
                .{ .binding = 2, .buffer_handle = self.step_params_buffer, .offset = step_idx * @sizeOf(StepParams), .size = @sizeOf(StepParams) },
            });

            const pipeline = gctx.lookupResource(self.step_pipeline) orelse return;
            const bind_group = gctx.lookupResource(step_bg) orelse return;

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind_group, &.{});
            pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(step_bg);

            current_read = current_write;
        }

        // 3. Distance pass
        {
            const pass = encoder.beginComputePass(null);
            const dist_bg = gctx.createBindGroup(self.distance_bgl, &.{
                .{ .binding = 0, .texture_view_handle = self.jfa_views[current_read] },
                .{ .binding = 1, .texture_view_handle = self.sdf_view },
            });

            const pipeline = gctx.lookupResource(self.distance_pipeline) orelse return;
            const bind_group = gctx.lookupResource(dist_bg) orelse return;

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind_group, &.{});
            pass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(dist_bg);
        }
    }
};

fn createComputePipeline(
    gctx: *zgpu.GraphicsContext,
    bgl: zgpu.BindGroupLayoutHandle,
    wgsl_source: [*:0]const u8,
    entry_point: [*:0]const u8,
    label: ?[*:0]const u8,
) zgpu.ComputePipelineHandle {
    const pl = gctx.createPipelineLayout(&.{bgl});
    defer gctx.releaseResource(pl);

    const cs_mod = zgpu.createWgslShaderModule(gctx.device, wgsl_source, label);
    defer cs_mod.release();

    return gctx.createComputePipeline(pl, .{
        .compute = .{
            .module = cs_mod,
            .entry_point = entry_point,
        },
    });
}
