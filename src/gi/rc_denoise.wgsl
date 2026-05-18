// rc_denoise.wgsl — Spatial bilateral denoise pass for the SSRC GI.
//
// Reads the raw, possibly noisy GI map from the finalize pass and
// outputs a smoothed version. The filter is a 3x3 bilateral guided by
// the world-space normal: neighbours with a similar normal are weighted
// strongly, neighbours on a different surface are weighted weakly so
// edges between surfaces stay sharp.
//
// When `enabled == 0` the shader degenerates to a plain copy so the
// caller can toggle the denoise from the GUI without changing wiring.

struct DenoiseParams {
    enabled: i32,
    sigma_normal: f32,
    _pad0: f32,
    _pad1: f32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var gi_input: texture_2d<f32>;
@group(0) @binding(1) var g_normal: texture_2d<f32>;
@group(0) @binding(2) var gi_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: DenoiseParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gi_output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pos = vec2<i32>(gid.xy);

    if (params.enabled == 0) {
        let c = textureLoad(gi_input, pos, 0).rgb;
        textureStore(gi_output, pos, vec4<f32>(c, 1.0));
        return;
    }

    // Map output pixel → normal-buffer pixel (they live at the same
    // resolution in the current pipeline but keep this generic).
    let nrm_dims = vec2<i32>(textureDimensions(g_normal));
    let dims_i = vec2<i32>(dims);
    let nrm_scale = vec2<f32>(f32(nrm_dims.x), f32(nrm_dims.y)) /
                    vec2<f32>(f32(dims_i.x), f32(dims_i.y));

    let center_normal = normalize(
        textureLoad(g_normal, vec2<i32>(vec2<f32>(pos) * nrm_scale), 0).xyz + vec3<f32>(1e-5),
    );

    let sigma = max(params.sigma_normal, 0.5);
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    var sum = vec3<f32>(0.0);
    var w_total = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let p = clamp(pos + vec2<i32>(dx, dy), vec2<i32>(0), dims_i - vec2<i32>(1));

            let n = normalize(
                textureLoad(g_normal, vec2<i32>(vec2<f32>(p) * nrm_scale), 0).xyz + vec3<f32>(1e-5),
            );
            let cos_d = clamp(dot(n, center_normal), -1.0, 1.0);
            let d = 1.0 - cos_d;
            let w_n = exp(-d * d * inv_two_sigma2);

            let dist2 = f32(dx * dx + dy * dy);
            let w_s = exp(-dist2 * 0.5);

            let w = w_s * w_n;
            sum = sum + textureLoad(gi_input, p, 0).rgb * w;
            w_total = w_total + w;
        }
    }

    let denoised = sum / max(w_total, 1e-4);
    textureStore(gi_output, pos, vec4<f32>(denoised, 1.0));
}
