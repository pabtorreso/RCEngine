// rc_finalize.wgsl — Radiance Cascades: Finalize Pass.
//
// Reads from cascade 0 (which may be at full, half or quarter screen
// resolution depending on the user-chosen GI quality level) and writes
// to the full-resolution GI texture sampled by `deferred_fs`.
// Always uses a bilinear sample so the shader transparently handles
// both the 1:1 case and the upsampling case.

struct FinalizeParams {
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var gi_input: texture_2d<f32>;
@group(0) @binding(1) var gi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: FinalizeParams;

fn fetchClamped(p: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let q = clamp(p, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(gi_input, q, 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(gi_output);
    if (gid.x >= out_dims.x || gid.y >= out_dims.y) { return; }

    let in_dims = vec2<i32>(textureDimensions(gi_input));
    let in_dims_f = vec2<f32>(in_dims);
    let out_dims_f = vec2<f32>(out_dims);

    // Continuous UV at the center of the output pixel, then map into
    // input texel space. When `in_dims == out_dims`, this lands exactly
    // on the input pixel center (frac = 0) and the bilinear collapses
    // to a single sample, so there's no quality cost in the Ultra
    // (full-res cascade 0) path.
    let uv = (vec2<f32>(gid.xy) + 0.5) / out_dims_f;
    let in_coord = uv * in_dims_f - 0.5;
    let base = floor(in_coord);
    let frac = in_coord - base;
    let base_i = vec2<i32>(base);

    let c00 = fetchClamped(base_i + vec2<i32>(0, 0), in_dims);
    let c10 = fetchClamped(base_i + vec2<i32>(1, 0), in_dims);
    let c01 = fetchClamped(base_i + vec2<i32>(0, 1), in_dims);
    let c11 = fetchClamped(base_i + vec2<i32>(1, 1), in_dims);

    let cx0 = mix(c00, c10, frac.x);
    let cx1 = mix(c01, c11, frac.x);
    var gi = mix(cx0, cx1, frac.y);

    gi = max(gi, vec3<f32>(0.0));
    gi = gi * params.intensity;

    textureStore(gi_output, vec2<i32>(gid.xy), vec4<f32>(gi, 1.0));
}
