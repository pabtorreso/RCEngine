// rc_merge.wgsl — Radiance Cascades Merge Pass
//
// Merges the current cascade (which we just raymarched) with the
// accumulated results from the higher cascades.
// This is the core "magic" of RC: interpolating the higher cascade
// (which has longer rays but lower angular resolution) and adding
// the local detail from the current cascade.

@group(0) @binding(0) var current_cascade: texture_2d<f32>;
@group(0) @binding(1) var upper_cascade: texture_2d<f32>;
@group(0) @binding(2) var merged_output: texture_storage_2d<rgba16float, write>;

// Optional params if needed
struct MergeParams {
    cascade_index: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}
@group(0) @binding(3) var<uniform> params: MergeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(current_cascade);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let pos = vec2<i32>(gid.xy);
    
    // Read the raymarched result for this cascade level
    let local_radiance = textureLoad(current_cascade, pos, 0).rgb;
    
    // Read the upper cascade (if it exists).
    // In a full implementation, we'd do bilinear interpolation and 
    // angular interpolation here. For this simplified MVP, we just add them,
    // assuming the upper cascade already represents the "background" light
    // that reached the end of the rays of the current cascade.
    
    let upper_dims = textureDimensions(upper_cascade);
    var upper_radiance = vec3<f32>(0.0);
    
    if (upper_dims.x > 0 && upper_dims.y > 0) {
        // Simplified fallback: just sample the same spatial location.
        // The real algorithm maps angles between cascade N and N+1.
        upper_radiance = textureLoad(upper_cascade, pos, 0).rgb;
    }
    
    let total_radiance = local_radiance + upper_radiance;
    
    textureStore(merged_output, pos, vec4<f32>(total_radiance, 1.0));
}
