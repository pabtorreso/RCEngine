/// GpuProfiler — lightweight wrapper around WebGPU timestamp queries.
///
/// Usage pattern per frame:
///     profiler.beginFrame();
///     profiler.beginScope(encoder, "SDF");
///     sdf.generate(...);
///     profiler.endScope(encoder);
///     profiler.beginScope(encoder, "RC GI");
///     rc.execute(...);
///     profiler.endScope(encoder);
///     ...
///     profiler.finishFrame(encoder);
///     gctx.submit(...);
///     profiler.pollResults(device);
///
/// `beginScope` / `endScope` call `encoder.writeTimestamp` between
/// passes, which only needs the base `timestamp-query` feature (no
/// `timestamp-query-inside-passes` / Dawn unsafe APIs required).
///
/// Results live in `last_times_ms[0..scope_count]` and `last_total_ms`;
/// `scope_names[i]` matches `last_times_ms[i]`.
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;

pub const MAX_SCOPES: u32 = 16;
const SLOTS_PER_SCOPE: u32 = 2;
const TOTAL_SLOTS: u32 = MAX_SCOPES * SLOTS_PER_SCOPE;

const MapState = enum {
    idle,
    in_flight,
    ready,
};

pub const GpuProfiler = struct {
    enabled: bool = false,
    query_set: wgpu.QuerySet = undefined,
    resolve_buffer: wgpu.Buffer = undefined,
    readback_buffer: wgpu.Buffer = undefined,

    scope_names: [MAX_SCOPES][]const u8 = .{""} ** MAX_SCOPES,
    scope_count: u32 = 0,
    open_scope: bool = false,

    last_times_ms: [MAX_SCOPES]f32 = .{0.0} ** MAX_SCOPES,
    last_scope_count: u32 = 0,
    last_total_ms: f32 = 0,

    map_state: MapState = .idle,
    pending_scope_count: u32 = 0,

    pub fn init(device: wgpu.Device) GpuProfiler {
        if (!device.hasFeature(.timestamp_query)) {
            std.log.warn("[gpu_profiler] timestamp-query feature not available, profiling disabled", .{});
            return .{};
        }

        const byte_size: u64 = @as(u64, TOTAL_SLOTS) * @sizeOf(u64);

        const query_set = device.createQuerySet(.{
            .label = "gpu_profiler_qset",
            .query_type = .timestamp,
            .count = TOTAL_SLOTS,
            .pipeline_statistics = null,
            .pipeline_statistics_count = 0,
        });
        const resolve_buffer = device.createBuffer(.{
            .label = "gpu_profiler_resolve",
            .usage = .{ .query_resolve = true, .copy_src = true },
            .size = byte_size,
        });
        const readback_buffer = device.createBuffer(.{
            .label = "gpu_profiler_readback",
            .usage = .{ .map_read = true, .copy_dst = true },
            .size = byte_size,
        });

        return .{
            .enabled = true,
            .query_set = query_set,
            .resolve_buffer = resolve_buffer,
            .readback_buffer = readback_buffer,
        };
    }

    pub fn deinit(self: *GpuProfiler) void {
        if (!self.enabled) return;
        self.readback_buffer.release();
        self.resolve_buffer.release();
        self.query_set.release();
        self.* = .{};
    }

    pub fn beginFrame(self: *GpuProfiler) void {
        self.scope_count = 0;
        self.open_scope = false;
    }

    pub fn beginScope(self: *GpuProfiler, encoder: wgpu.CommandEncoder, name: []const u8) void {
        if (!self.enabled) return;
        if (self.open_scope) return; // forgot to call endScope — drop silently
        if (self.scope_count >= MAX_SCOPES) return;
        const begin_slot = self.scope_count * SLOTS_PER_SCOPE;
        encoder.writeTimestamp(self.query_set, begin_slot);
        self.scope_names[self.scope_count] = name;
        self.open_scope = true;
    }

    pub fn endScope(self: *GpuProfiler, encoder: wgpu.CommandEncoder) void {
        if (!self.enabled or !self.open_scope) return;
        const end_slot = self.scope_count * SLOTS_PER_SCOPE + 1;
        encoder.writeTimestamp(self.query_set, end_slot);
        self.scope_count += 1;
        self.open_scope = false;
    }

    /// Resolve + (if no readback in flight) copy + mapAsync.
    pub fn finishFrame(self: *GpuProfiler, encoder: wgpu.CommandEncoder) void {
        if (!self.enabled or self.scope_count == 0) return;

        const slot_count = self.scope_count * SLOTS_PER_SCOPE;
        const byte_size: u64 = @as(u64, slot_count) * @sizeOf(u64);
        encoder.resolveQuerySet(self.query_set, 0, slot_count, self.resolve_buffer, 0);

        if (self.map_state == .idle) {
            encoder.copyBufferToBuffer(self.resolve_buffer, 0, self.readback_buffer, 0, byte_size);
            self.pending_scope_count = self.scope_count;
            self.map_state = .in_flight;
            self.readback_buffer.mapAsync(
                .{ .read = true },
                0,
                byte_size,
                mapCallback,
                @ptrCast(self),
            );
        }
    }

    /// Advance the device and, if a mapping completed, read the timings out.
    pub fn pollResults(self: *GpuProfiler, device: wgpu.Device) void {
        if (!self.enabled) return;

        // Process any pending async callbacks (Dawn requires explicit polling).
        device.tick();

        if (self.map_state != .ready) return;

        const slot_count = self.pending_scope_count * SLOTS_PER_SCOPE;
        const data = self.readback_buffer.getConstMappedRange(u64, 0, slot_count) orelse {
            self.readback_buffer.unmap();
            self.map_state = .idle;
            return;
        };

        var total: u64 = 0;
        var i: u32 = 0;
        while (i < self.pending_scope_count) : (i += 1) {
            const begin = data[i * 2];
            const end = data[i * 2 + 1];
            const diff_ns: u64 = if (end >= begin) end - begin else 0;
            self.last_times_ms[i] = @as(f32, @floatFromInt(diff_ns)) / 1_000_000.0;
            total += diff_ns;
        }
        self.last_scope_count = self.pending_scope_count;
        self.last_total_ms = @as(f32, @floatFromInt(total)) / 1_000_000.0;

        self.readback_buffer.unmap();
        self.map_state = .idle;
    }
};

fn mapCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.c) void {
    const self: *GpuProfiler = @ptrCast(@alignCast(userdata.?));
    if (status == .success) {
        self.map_state = .ready;
    } else {
        // Map failed (device lost, etc.) — recover so we can try again next frame.
        self.map_state = .idle;
    }
}
