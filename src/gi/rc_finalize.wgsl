// rc_finalize.wgsl — Composites GI and the Scene
@group(0) @binding(0) var gi_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;

struct FinalizeParams {
    srgb_correction: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
    _padding: array<vec4<u32>, 15>,
}
@group(0) @binding(3) var<uniform> params: FinalizeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gi_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let pos = vec2<i32>(gid.xy);
    let gi = textureLoad(gi_tex, pos, 0).rgb;
    let scene = textureLoad(scene_tex, pos, 0).rgba;
    
    // Add GI and emissive scene together
    var final_color = gi + scene.rgb * scene.a;
    
    if (params.srgb_correction == 1) {
        // Simple gamma correction
        final_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    }
    
    textureStore(output_tex, pos, vec4<f32>(final_color, 1.0));
}
