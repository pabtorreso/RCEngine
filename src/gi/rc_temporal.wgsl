// rc_temporal.wgsl — Temporal accumulation for the SSRC GI.
//
// Exponential moving average between the denoised GI of the current
// frame and the GI texture written by the previous frame. Reduces
// noise dramatically when the camera is static and lets us lower the
// per-frame ray budget without losing apparent quality.
//
// Limitation: there's no reprojection here yet. Under fast camera
// movement, ghosting will appear; raising `alpha` (weight on the
// current frame) reduces ghosting at the cost of more visible noise.

struct TemporalParams {
    enabled: i32,
    alpha: f32, // weight of the current frame (0.05 = smooth/laggy, 1.0 = no temporal)
    _pad0: f32,
    _pad1: f32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var gi_current: texture_2d<f32>;
@group(0) @binding(1) var gi_history: texture_2d<f32>;
@group(0) @binding(2) var gi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: TemporalParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gi_output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let pos = vec2<i32>(gid.xy);

    let current = textureLoad(gi_current, pos, 0).rgb;

    if (params.enabled == 0) {
        textureStore(gi_output, pos, vec4<f32>(current, 1.0));
        return;
    }

    // Variance clamp: keep the history within the 3x3 min/max of the
    // current frame. Cheap heuristic that kills most of the ghosting
    // that appears when the camera moves quickly.
    var c_min = current;
    var c_max = current;
    let dims_i = vec2<i32>(dims);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let p = clamp(pos + vec2<i32>(dx, dy), vec2<i32>(0), dims_i - vec2<i32>(1));
            let c = textureLoad(gi_current, p, 0).rgb;
            c_min = min(c_min, c);
            c_max = max(c_max, c);
        }
    }

    let history = textureLoad(gi_history, pos, 0).rgb;
    let clamped = clamp(history, c_min, c_max);
    let alpha = clamp(params.alpha, 0.02, 1.0);
    let result = mix(clamped, current, alpha);

    textureStore(gi_output, pos, vec4<f32>(result, 1.0));
}
