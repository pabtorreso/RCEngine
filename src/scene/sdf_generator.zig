/// SDF Generator — exact 2D Euclidean Distance Transform via PBA.
///
/// Implementation follows Cao/Tang/Tan "Parallel Banding Algorithm"
/// (2010) for the separable structure and Felzenszwalb & Huttenlocher
/// (2012) for the 1D lower-envelope sweep. Three compute dispatches
/// total (init, column scan, row envelope) — replaces the JFA's
/// 1 + log2(N) + 1 dispatches. The output is mathematically exact:
/// no `* 0.95` safety factor would be needed downstream (the raymarch
/// keeps it for paranoia).
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const pba_wgsl = @embedFile("../gi/pba.wgsl");

pub const MAX_SDF_QUALITY_LEVEL: u32 = 2; // 0=Ultra, 1=High (half), 2=Performance (quarter)

pub const SdfGenerator = struct {
    width: u32,
    height: u32,
    quality_level: u32,

    // Final r32_float SDF; sampled by the radiance-cascades raymarch.
    sdf_texture: zgpu.TextureHandle,
    sdf_view: zgpu.TextureViewHandle,

    // Two scratch r32_uint textures used by phases 1 and 2. They hold,
    // respectively, the seed marker (y of pixel if occupied) and the
    // per-column nearest seed-y.
    seed_texture: zgpu.TextureHandle,
    seed_view: zgpu.TextureViewHandle,
    col_texture: zgpu.TextureHandle,
    col_view: zgpu.TextureViewHandle,

    init_pipeline: zgpu.ComputePipelineHandle,
    column_pipeline: zgpu.ComputePipelineHandle,
    row_pipeline: zgpu.ComputePipelineHandle,

    init_bgl: zgpu.BindGroupLayoutHandle,
    column_bgl: zgpu.BindGroupLayoutHandle,
    row_bgl: zgpu.BindGroupLayoutHandle,

    pub fn init(gctx: *zgpu.GraphicsContext, screen_width: u32, screen_height: u32, quality_level: u32) SdfGenerator {
        const q = @min(quality_level, MAX_SDF_QUALITY_LEVEL);
        const width = @max(1, screen_width >> @as(u5, @intCast(q)));
        const height = @max(1, screen_height >> @as(u5, @intCast(q)));

        const seed_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .r32_uint,
            .mip_level_count = 1,
        });
        const seed_view = gctx.createTextureView(seed_texture, .{});

        const col_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .r32_uint,
            .mip_level_count = 1,
        });
        const col_view = gctx.createTextureView(col_texture, .{});

        const sdf_texture = gctx.createTexture(.{
            .usage = .{ .texture_binding = true, .storage_binding = true },
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .r32_float,
            .mip_level_count = 1,
        });
        const sdf_view = gctx.createTextureView(sdf_texture, .{});

        const init_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .depth, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_uint, .tvdim_2d),
        });
        const column_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .uint, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_uint, .tvdim_2d),
        });
        const row_bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .compute = true }, .uint, .tvdim_2d, false),
            zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_float, .tvdim_2d),
        });

        const init_pipeline = createComputePipeline(gctx, init_bgl, pba_wgsl, "init_main", "pba_init");
        const column_pipeline = createComputePipeline(gctx, column_bgl, pba_wgsl, "column_main", "pba_column");
        const row_pipeline = createComputePipeline(gctx, row_bgl, pba_wgsl, "row_main", "pba_row");

        return .{
            .width = width,
            .height = height,
            .quality_level = q,
            .seed_texture = seed_texture,
            .seed_view = seed_view,
            .col_texture = col_texture,
            .col_view = col_view,
            .sdf_texture = sdf_texture,
            .sdf_view = sdf_view,
            .init_pipeline = init_pipeline,
            .column_pipeline = column_pipeline,
            .row_pipeline = row_pipeline,
            .init_bgl = init_bgl,
            .column_bgl = column_bgl,
            .row_bgl = row_bgl,
        };
    }

    pub fn deinit(self: *SdfGenerator, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.row_bgl);
        gctx.releaseResource(self.column_bgl);
        gctx.releaseResource(self.init_bgl);
        gctx.releaseResource(self.row_pipeline);
        gctx.releaseResource(self.column_pipeline);
        gctx.releaseResource(self.init_pipeline);
        gctx.releaseResource(self.sdf_view);
        gctx.destroyResource(self.sdf_texture);
        gctx.releaseResource(self.col_view);
        gctx.destroyResource(self.col_texture);
        gctx.releaseResource(self.seed_view);
        gctx.destroyResource(self.seed_texture);
        self.* = undefined;
    }

    pub fn generate(self: *SdfGenerator, gctx: *zgpu.GraphicsContext, encoder: wgpu.CommandEncoder, depth_texv: zgpu.TextureViewHandle) void {
        // Phase 0: init — depth → seed marker
        {
            const wg_x = (self.width + 7) / 8;
            const wg_y = (self.height + 7) / 8;
            const pass = encoder.beginComputePass(null);
            const bg = gctx.createBindGroup(self.init_bgl, &.{
                .{ .binding = 0, .texture_view_handle = depth_texv },
                .{ .binding = 1, .texture_view_handle = self.seed_view },
            });

            pass.setPipeline(gctx.lookupResource(self.init_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
            pass.dispatchWorkgroups(wg_x, wg_y, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(bg);
        }

        // Phase 1: banded column scan — one workgroup per column,
        // 32 threads per workgroup cooperating across vertical bands
        // via workgroup shared memory + private per-band scratch.
        {
            const pass = encoder.beginComputePass(null);
            const bg = gctx.createBindGroup(self.column_bgl, &.{
                .{ .binding = 0, .texture_view_handle = self.seed_view },
                .{ .binding = 1, .texture_view_handle = self.col_view },
            });

            pass.setPipeline(gctx.lookupResource(self.column_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
            pass.dispatchWorkgroups(self.width, 1, 1);
            pass.end();
            pass.release();
            gctx.releaseResource(bg);
        }

        // Phase 2: row envelope — one thread per row, private v[] up to MAX_DIM.
        {
            const wg_y = (self.height + 63) / 64;
            const pass = encoder.beginComputePass(null);
            const bg = gctx.createBindGroup(self.row_bgl, &.{
                .{ .binding = 0, .texture_view_handle = self.col_view },
                .{ .binding = 1, .texture_view_handle = self.sdf_view },
            });

            pass.setPipeline(gctx.lookupResource(self.row_pipeline).?);
            pass.setBindGroup(0, gctx.lookupResource(bg).?, &.{});
            pass.dispatchWorkgroups(1, wg_y, 1);
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
