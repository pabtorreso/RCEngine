// jfa.wgsl - Jump Flood Algorithm for SDF generation
//
// Generates a distance field from an RGBA scene texture where
// opacity (alpha) determines if a pixel is occupied.

// --- Init Pass ---
// Reads the scene texture and outputs the initial seed coordinates.
@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var output_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8)
fn init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let depth = textureLoad(depth_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), 0);
    var seed: vec2<f32>;
    
    // Depth < 1.0 indicates solid geometry
    if (depth < 1.0) {
        // This pixel is occupied — seed with its own position
        seed = vec2<f32>(f32(gid.x), f32(gid.y));
    } else {
        // Empty pixel — mark as "no seed" with a far-away value
        seed = vec2<f32>(-1.0, -1.0);
    }
    
    textureStore(output_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), vec4<f32>(seed, 0.0, 0.0));
}

// --- Step Pass ---
// Performs one step of the Jump Flood Algorithm.
struct StepParams {
    step_size: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
    _padding: array<vec4<u32>, 15>,
}
@group(0) @binding(0) var step_input_tex: texture_2d<f32>;
@group(0) @binding(1) var step_output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: StepParams;

@compute @workgroup_size(8, 8)
fn step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(step_input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    var best_seed = textureLoad(step_input_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), 0).xy;
    var best_dist = 1e20;
    
    if (best_seed.x >= 0.0) {
        let d = pos - best_seed;
        best_dist = dot(d, d);
    }

    let step = params.step_size;
    
    // Check all 8 neighbors at 'step' distance
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sx = i32(gid.x) + dx * step;
            let sy = i32(gid.y) + dy * step;
            
            if (sx < 0 || sx >= i32(dims.x) || sy < 0 || sy >= i32(dims.y)) { continue; }

            let neighbor = textureLoad(step_input_tex, vec2<i32>(sx, sy), 0).xy;
            if (neighbor.x < 0.0) { continue; }

            let d = pos - neighbor;
            let dist = dot(d, d);
            if (dist < best_dist) {
                best_dist = dist;
                best_seed = neighbor;
            }
        }
    }

    textureStore(step_output_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), vec4<f32>(best_seed, 0.0, 0.0));
}

// --- Distance Pass ---
// Converts the final seed coordinates into normalized distances [0, 1].
@group(0) @binding(0) var dist_seed_tex: texture_2d<f32>;
@group(0) @binding(1) var dist_output_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn distance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dist_seed_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let seed = textureLoad(dist_seed_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), 0).xy;

    var dist: f32 = 0.0;
    if (seed.x >= 0.0) {
        let d = pos - seed;
        dist = sqrt(dot(d, d));
    } else {
        // No seed found — maximum distance
        dist = sqrt(f32(dims.x * dims.x + dims.y * dims.y));
    }

    // Normalize to [0, 1] range based on screen diagonal
    let diagonal = sqrt(f32(dims.x * dims.x + dims.y * dims.y));
    textureStore(dist_output_tex, vec2<i32>(vec2<u32>(gid.x, gid.y)), vec4<f32>(dist / diagonal, 0.0, 0.0, 0.0));
}
