// rc_raymarch.wgsl — Radiance Cascades Raymarch + Merge Pass
struct RCPassParams {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    _pad0: i32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var sdf_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var upper_cascade: texture_2d<f32>;
@group(0) @binding(3) var cascade_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: RCPassParams;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(sdf_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let cascade_idx = f32(params.cascade_index);
    let base_rays = f32(params.base_ray_count);
    let ray_count = base_rays * pow(2.0, cascade_idx);
    let t_0 = pow(4.0, cascade_idx);
    let t_1 = pow(4.0, cascade_idx + 1.0);
    
    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let diagonal = sqrt(f32(dims.x * dims.x + dims.y * dims.y));
    
    var local_radiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    let angular_step = (2.0 * PI) / ray_count;
    
    for (var i = 0; i < i32(ray_count); i = i + 1) {
        let angle = f32(i) * angular_step;
        let dir = vec2<f32>(cos(angle), sin(angle));
        
        var t = t_0;
        var hit_radiance = vec3<f32>(0.0);
        var hit_alpha = 0.0;
        
        for (var step = 0; step < 16; step = step + 1) {
            if (t >= t_1) { break; }
            
            let p = pos + dir * t;
            if (p.x < 0.0 || p.x >= f32(dims.x) || p.y < 0.0 || p.y >= f32(dims.y)) {
                if (dir.y > 0.0) {
                    hit_radiance = vec3<f32>(0.2, 0.4, 0.8) * params.sun_intensity * dir.y;
                    hit_alpha = 1.0;
                }
                break;
            }
            
            let dist_norm = textureLoad(sdf_tex, vec2<i32>(p), 0).r;
            let dist_px = dist_norm * diagonal;
            
            if (dist_px < 1.0) {
                let color = textureLoad(scene_tex, vec2<i32>(p), 0);
                hit_radiance = color.rgb; 
                hit_alpha = color.a;
                break;
            }
            
            t += dist_px;
        }
        
        local_radiance += hit_radiance;
        total_weight += 1.0;
    }
    
    if (total_weight > 0.0) {
        local_radiance /= total_weight;
    }
    
    // --- Merge ---
    let upper_dims = textureDimensions(upper_cascade);
    var upper_radiance = vec3<f32>(0.0);
    
    // Only merge if the upper cascade exists (meaning it's not the highest level, or we bound something valid)
    if (upper_dims.x > 1) {
        upper_radiance = textureLoad(upper_cascade, vec2<i32>(gid.xy), 0).rgb;
    }
    
    let total_radiance = local_radiance + upper_radiance;
    
    textureStore(cascade_output, vec2<i32>(gid.xy), vec4<f32>(total_radiance, 1.0));
}
