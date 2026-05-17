// rc_finalize.wgsl — Radiance Cascades: Finalize Pass.
//
// Takes the fully merged cascade pyramid (the cascade-0 output) and
// produces the global-illumination texture sampled by `deferred_fs`.
// The composite with albedo / emissive lives in the PBR shader, so
// here we only sanitize the values (clamp NaNs, optional intensity
// scaling) and skip gamma — the deferred pass works in linear space.

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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gi_input);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pos = vec2<i32>(gid.xy);
    var gi = textureLoad(gi_input, pos, 0).rgb;

    // Replace NaN / negative values that may slip in from storage
    // textures, then apply the user-tunable intensity scaling.
    gi = max(gi, vec3<f32>(0.0));
    gi = gi * params.intensity;

    textureStore(gi_output, pos, vec4<f32>(gi, 1.0));
}
