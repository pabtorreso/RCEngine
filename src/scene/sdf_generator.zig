/// SDF Generator — exact 2D Euclidean Distance Transform via PBA.
///
/// Implementation follows Cao/Tang/Tan "Parallel Banding Algorithm"
/// (2010) for the separable structure and Felzenszwalb & Huttenlocher
/// (2012) for the 1D lower-envelope sweep. Three compute dispatches
/// total (init, column scan, row envelope) — replaces the JFA's
/// 1 + log2(N) + 1 dispatches. The output is mathematically exact:
/// no `* 0.95` safety factor would be needed downstream (the raymarch
/// keeps it for paranoia).
///
/// Dual-path row pipeline:
///   - FAST path  (width <= ROW_FAST_PATH_MAX_W): row_main with a private
///                array v[] of size MAX_DIM=2000. Hot register access,
///                used for 1080p widescreen and below.
///   - BUF  path  (width >  ROW_FAST_PATH_MAX_W): row_main_buf with a
///                W*H*sizeof(i32) storage-buffer scratch. ~10x slower per
///                scratch access vs registers, but the only path with
///                unbounded capacity. Used for 1440p widescreen / 4K Ultra.
/// The path is picked once at init() based on the actual SDF width, so the
/// rest of the engine is unaware of which one is running.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const pba_wgsl = @embedFile("../gi/pba.wgsl");

pub const MAX_SDF_QUALITY_LEVEL: u32 = 2; // 0=Ultra, 1=High (half), 2=Performance (quarter)

// Must match MAX_DIM in pba.wgsl (private v[] capacity in row_main).
const ROW_FAST_PATH_MAX_W: u32 = 2000;

pub const SdfGenerator = struct {
    width: u32,
    height: u32,
    quality_level: u32,
    use_buf_path: bool,

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

    // BUF-path storage-buffer scratch. Only allocated when use_buf_path
    // is true (width > ROW_FAST_PATH_MAX_W). Size = W*H*sizeof(i32).
    scratch_buffer: zgpu.BufferHandle,
    scratch_size: u32,

    init_pipeline: zgpu.ComputePipelineHandle,
    column_pipeline: zgpu.ComputePipelineHandle,
    // Points to row_main (fast) or row_main_buf (buf) depending on path.
    row_pipeline: zgpu.ComputePipelineHandle,

    init_bgl: zgpu.BindGroupLayoutHandle,
    column_bgl: zgpu.BindGroupLayoutHandle,
    // 2-entry layout for fast path, 3-entry for buf path.
    row_bgl: zgpu.BindGroupLayoutHandle,

    pub fn init(gctx: *zgpu.GraphicsContext, screen_width: u32, screen_height: u32, quality_level: u32) SdfGenerator {
        const q = @min(quality_level, MAX_SDF_QUALITY_LEVEL);
        const width = @max(1, screen_width >> @as(u5, @intCast(q)));
        const height = @max(1, screen_height >> @as(u5, @intCast(q)));
        const use_buf_path = width > ROW_FAST_PATH_MAX_W;

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

        var scratch_size: u32 = 0;
        var scratch_buffer: zgpu.BufferHandle = .{};
        var row_bgl: zgpu.BindGroupLayoutHandle = undefined;
        var row_pipeline: zgpu.ComputePipelineHandle = undefined;

        if (use_buf_path) {
            scratch_size = width * height * @sizeOf(i32);
            scratch_buffer = gctx.createBuffer(.{
                .usage = .{ .storage = true },
                .size = scratch_size,
            });
            row_bgl = gctx.createBindGroupLayout(&.{
                zgpu.textureEntry(0, .{ .compute = true }, .uint, .tvdim_2d, false),
                zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_float, .tvdim_2d),
                zgpu.bufferEntry(2, .{ .compute = true }, .storage, false, 0),
            });
            row_pipeline = createComputePipeline(gctx, row_bgl, pba_wgsl, "row_main_buf", "pba_row_buf");
        } else {
            row_bgl = gctx.createBindGroupLayout(&.{
                zgpu.textureEntry(0, .{ .compute = true }, .uint, .tvdim_2d, false),
                zgpu.storageTextureEntry(1, .{ .compute = true }, .write_only, .r32_float, .tvdim_2d),
            });
            row_pipeline = createComputePipeline(gctx, row_bgl, pba_wgsl, "row_main", "pba_row");
        }

        const init_pipeline = createComputePipeline(gctx, init_bgl, pba_wgsl, "init_main", "pba_init");
        const column_pipeline = createComputePipeline(gctx, column_bgl, pba_wgsl, "column_main", "pba_column");

        return .{
            .width = width,
            .height = height,
            .quality_level = q,
            .use_buf_path = use_buf_path,
            .seed_texture = seed_texture,
            .seed_view = seed_view,
            .col_texture = col_texture,
            .col_view = col_view,
            .sdf_texture = sdf_texture,
            .sdf_view = sdf_view,
            .scratch_buffer = scratch_buffer,
            .scratch_size = scratch_size,
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
        if (self.use_buf_path) {
            gctx.releaseResource(self.scratch_buffer);
        }
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

        // Phase 2: row envelope — one thread per row. Uses the fast or
        // buf pipeline depending on the path picked at init().
        {
            const wg_y = (self.height + 63) / 64;
            const pass = encoder.beginComputePass(null);
            const bg = if (self.use_buf_path)
                gctx.createBindGroup(self.row_bgl, &.{
                    .{ .binding = 0, .texture_view_handle = self.col_view },
                    .{ .binding = 1, .texture_view_handle = self.sdf_view },
                    .{ .binding = 2, .buffer_handle = self.scratch_buffer, .offset = 0, .size = self.scratch_size },
                })
            else
                gctx.createBindGroup(self.row_bgl, &.{
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
