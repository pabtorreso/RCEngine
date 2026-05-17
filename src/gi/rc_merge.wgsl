// rc_merge.wgsl — Hierarchical Radiance Cascades merge pass.
//
// The current cascade lives at resolution `(cascade_width, cascade_height)`
// and the parent cascade at half of each. For each pixel in the current
// cascade we sample the parent bilinearly (manually, with four
// `textureLoad`s) and add it to the local raymarch result.

struct MergeParams {
    cascade_index: i32,
    has_upper: i32,
    cascade_width: i32,
    cascade_height: i32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var current_cascade: texture_2d<f32>;
@group(0) @binding(1) var upper_cascade: texture_2d<f32>;
@group(0) @binding(2) var merged_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: MergeParams;

fn fetchUpper(p: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let q = clamp(p, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(upper_cascade, q, 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = u32(params.cascade_width);
    let ch = u32(params.cascade_height);
    if (gid.x >= cw || gid.y >= ch) { return; }

    let pos = vec2<i32>(gid.xy);
    let local = textureLoad(current_cascade, pos, 0).rgb;

    var upper = vec3<f32>(0.0);
    if (params.has_upper == 1) {
        let udims = vec2<i32>(textureDimensions(upper_cascade));

        // Map cascade-N pixel center → cascade-(N+1) coords.
        // Each parent pixel covers a 2×2 block of current-cascade pixels,
        // so `upper_coord = pos * 0.5 - 0.25` (continuous coords).
        let upper_coord = vec2<f32>(pos) * 0.5 - 0.25;
        let base = floor(upper_coord);
        let frac = upper_coord - base;
        let base_i = vec2<i32>(base);

        let c00 = fetchUpper(base_i + vec2<i32>(0, 0), udims);
        let c10 = fetchUpper(base_i + vec2<i32>(1, 0), udims);
        let c01 = fetchUpper(base_i + vec2<i32>(0, 1), udims);
        let c11 = fetchUpper(base_i + vec2<i32>(1, 1), udims);

        let cx0 = mix(c00, c10, frac.x);
        let cx1 = mix(c01, c11, frac.x);
        upper = mix(cx0, cx1, frac.y);
    }

    // If the current cascade already returned a bright ray it likely
    // hit something nearby; weight the long-range parent contribution
    // by approximate "still-open" visibility (luminance complement).
    let lum = dot(local, vec3<f32>(0.299, 0.587, 0.114));
    let visibility = clamp(1.0 - lum, 0.0, 1.0);

    let merged = local + upper * visibility;
    textureStore(merged_output, pos, vec4<f32>(merged, 1.0));
}
