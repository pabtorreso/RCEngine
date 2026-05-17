// rc_merge.wgsl — Radiance Cascades: Merge Pass.
//
// Combines the radiance for the current cascade (which only covers
// distances [t_0, t_1)) with the already-merged radiance from the upper
// cascade (which covers everything farther out). Conceptually the upper
// cascade is what each ray in the current cascade "sees" after its own
// march budget is exhausted.
//
// Because our prototype keeps every cascade at full screen resolution
// the merge is mostly a per-pixel sum — but we apply a small bilateral
// blur on the upper cascade to compensate for the lower effective
// angular resolution of the longer rays. Cascades stay temporally
// coherent without any extra spatial cost.

struct MergeParams {
    cascade_index: i32,
    has_upper: i32,
    _pad0: i32,
    _pad1: i32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var current_cascade: texture_2d<f32>;
@group(0) @binding(1) var upper_cascade: texture_2d<f32>;
@group(0) @binding(2) var merged_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: MergeParams;

fn sampleUpper(p: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    let q = clamp(p, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(upper_cascade, q, 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(current_cascade);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pos = vec2<i32>(gid.xy);
    let local = textureLoad(current_cascade, pos, 0).rgb;

    var upper = vec3<f32>(0.0);
    if (params.has_upper == 1) {
        let udims = vec2<i32>(textureDimensions(upper_cascade));
        // 3x3 box blur with bilateral-ish weight: angular resolution
        // is lower in the parent cascade, so blurring smooths out
        // discrete ray banding without smearing real contrast much.
        var sum = vec3<f32>(0.0);
        var w_total = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let w = select(0.5, 1.0, dx == 0 && dy == 0);
                sum = sum + sampleUpper(pos + vec2<i32>(dx, dy), udims) * w;
                w_total = w_total + w;
            }
        }
        upper = sum / w_total;
    }

    // The current cascade is a fresh estimate for the [t_0, t_1)
    // interval — combine it with the parent which already accounts for
    // [t_1, inf). When the current ray actually hit something the
    // upper estimate is effectively occluded, so weight by (1 - local
    // visibility). We approximate that visibility from luminance.
    let visibility = clamp(1.0 - dot(local, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 1.0);
    let merged = local + upper * visibility;

    textureStore(merged_output, pos, vec4<f32>(merged, 1.0));
}
