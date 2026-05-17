/// Fullscreen quad renderer.
/// Renders a fullscreen triangle using the final GI output texture.
/// Uses vertex-shader-generated UVs (no vertex buffer needed).
const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const RadianceCascades = @import("../gi/radiance_cascades.zig").RadianceCascades;

const fullscreen_vs =
    \\  struct VertexOut {
    \\      @builtin(position) position: vec4<f32>,
    \\      @location(0) uv: vec2<f32>,
    \\  }
    \\  @vertex fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    \\      var out: VertexOut;
    \\      // Generate a fullscreen triangle from vertex index.
    \\      let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    \\      let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    \\      out.position = vec4<f32>(x, y, 0.0, 1.0);
    \\      out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    \\      return out;
    \\  }
;

const fullscreen_fs =
    \\  @group(0) @binding(0) var output_texture: texture_2d<f32>;
    \\  @group(0) @binding(1) var output_sampler: sampler;
    \\  @fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    \\      return textureSample(output_texture, output_sampler, uv);
    \\  }
;

pub const FullscreenQuad = struct {
    pipeline: zgpu.RenderPipelineHandle,
    bind_group: zgpu.BindGroupHandle,
    sampler: zgpu.SamplerHandle,

    pub fn init(gctx: *zgpu.GraphicsContext, rc: *RadianceCascades) FullscreenQuad {
        // Bind group layout: texture + sampler
        const bgl = gctx.createBindGroupLayout(&.{
            zgpu.textureEntry(0, .{ .fragment = true }, .float, .tvdim_2d, false),
            zgpu.samplerEntry(1, .{ .fragment = true }, .filtering),
        });
        defer gctx.releaseResource(bgl);

        const pipeline_layout = gctx.createPipelineLayout(&.{bgl});
        defer gctx.releaseResource(pipeline_layout);

        // Shaders
        const vs_module = zgpu.createWgslShaderModule(gctx.device, fullscreen_vs, "fsq_vs");
        defer vs_module.release();

        const fs_module = zgpu.createWgslShaderModule(gctx.device, fullscreen_fs, "fsq_fs");
        defer fs_module.release();

        const color_targets = [_]wgpu.ColorTargetState{.{
            .format = zgpu.GraphicsContext.swapchain_format,
        }};

        const pipeline = gctx.createRenderPipeline(pipeline_layout, .{
            .vertex = .{
                .module = vs_module,
                .entry_point = "main",
            },
            .primitive = .{
                .topology = .triangle_list,
            },
            .fragment = &.{
                .module = fs_module,
                .entry_point = "main",
                .target_count = color_targets.len,
                .targets = &color_targets,
            },
        });

        // Sampler with bilinear filtering (important for smooth GI upscaling)
        const sampler_handle = gctx.createSampler(.{
            .mag_filter = .linear,
            .min_filter = .linear,
        });

        // Bind the RC output texture
        const bind_group = gctx.createBindGroup(bgl, &.{
            .{ .binding = 0, .texture_view_handle = rc.output_view },
            .{ .binding = 1, .sampler_handle = sampler_handle },
        });

        return .{
            .pipeline = pipeline,
            .bind_group = bind_group,
            .sampler = sampler_handle,
        };
    }

    pub fn deinit(self: *FullscreenQuad, gctx: *zgpu.GraphicsContext) void {
        gctx.releaseResource(self.sampler);
        gctx.releaseResource(self.bind_group);
        gctx.releaseResource(self.pipeline);
        self.* = undefined;
    }

    pub fn draw(self: *const FullscreenQuad, gctx: *zgpu.GraphicsContext, pass: wgpu.RenderPassEncoder) void {
        const pipeline = gctx.lookupResource(self.pipeline) orelse return;
        const bind_group = gctx.lookupResource(self.bind_group) orelse return;

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind_group, &.{});
        pass.draw(3, 1, 0, 0);
    }
};
