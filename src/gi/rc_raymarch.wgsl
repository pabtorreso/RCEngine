// rc_raymarch.wgsl — Screen-Space Radiance Cascades: Raymarch Pass.
//
// For each pixel and each cascade level N, we cast `ray_count` rays
// in equally-spaced angular directions and march each ray over screen
// space using sphere tracing on the 2D screen-space SDF (`g_sdf`).
//
// When a ray hits a surface (sdf <= 1px) we sample `g_emissive` at the
// hit position to recover the light emitted by that surface. If the
// ray escapes the screen we treat that as the sky and return a soft
// directional sky color (good enough until a real sky probe is wired in).
//
// The output stored in `raymarch_output` is the *local* radiance for the
// interval [t_0, t_1) of cascade N. The merge with cascade N+1 happens in
// a separate pass (rc_merge.wgsl).

struct RCPassParams {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    _pad0: i32,
    _padding: array<vec4<u32>, 15>,
}

@group(0) @binding(0) var g_sdf: texture_2d<f32>;
@group(0) @binding(1) var g_emissive: texture_2d<f32>;
@group(0) @binding(2) var g_normal: texture_2d<f32>;
@group(0) @binding(3) var raymarch_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: RCPassParams;

const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 64;
const HIT_EPSILON_PX: f32 = 1.0;

fn skyRadiance(dir: vec2<f32>, intensity: f32) -> vec3<f32> {
    // Soft sky: warm near horizon, cool overhead. `dir.y > 0` means the
    // ray points downward in screen space (y grows downward), so we
    // invert it to obtain an "up" factor.
    let up = clamp(-dir.y, 0.0, 1.0);
    let zenith = vec3<f32>(0.35, 0.55, 0.95);
    let horizon = vec3<f32>(0.85, 0.75, 0.55);
    return mix(horizon, zenith, up) * intensity;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_sdf);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let cascade_idx = f32(params.cascade_index);
    let base_rays = f32(params.base_ray_count);
    let ray_count = base_rays * pow(2.0, cascade_idx);
    let t_0 = pow(4.0, cascade_idx);
    let t_1 = pow(4.0, cascade_idx + 1.0);

    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let diagonal = sqrt(f32(dims.x * dims.x + dims.y * dims.y));

    // Skip the GI computation entirely for pixels that have no geometry
    // (background sky); their final color comes from the IBL sample.
    let center_sdf = textureLoad(g_sdf, vec2<i32>(gid.xy), 0).r * diagonal;
    if (center_sdf > diagonal * 0.5) {
        textureStore(raymarch_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    var accumulated = vec3<f32>(0.0);
    let angular_step = (2.0 * PI) / ray_count;

    for (var i: i32 = 0; i < i32(ray_count); i = i + 1) {
        // Stratified angle: half-step offset stops the ray bundle from
        // aligning with screen-axes on cascade 0.
        let angle = (f32(i) + 0.5) * angular_step;
        let dir = vec2<f32>(cos(angle), sin(angle));

        var t = t_0;
        var hit = vec3<f32>(0.0);

        for (var step: i32 = 0; step < MAX_STEPS; step = step + 1) {
            if (t >= t_1) { break; }

            let p = pos + dir * t;
            if (p.x < 0.0 || p.x >= f32(dims.x) || p.y < 0.0 || p.y >= f32(dims.y)) {
                hit = skyRadiance(dir, params.sun_intensity);
                break;
            }

            let dist_px = textureLoad(g_sdf, vec2<i32>(p), 0).r * diagonal;
            if (dist_px < HIT_EPSILON_PX) {
                // Surface hit — sample the radiance emitted by that
                // pixel and modulate it by the surface normal facing
                // the ray (basic Lambertian receive on the emitter).
                let emissive = textureLoad(g_emissive, vec2<i32>(p), 0).rgb;
                let n3 = textureLoad(g_normal, vec2<i32>(p), 0).xyz;
                let n2 = normalize(vec2<f32>(n3.x, -n3.y) + vec2<f32>(1e-5));
                let facing = clamp(dot(n2, -dir), 0.0, 1.0);
                hit = emissive * (0.25 + 0.75 * facing);
                break;
            }

            t = t + max(dist_px, 0.5);
        }

        // If the loop fell through without setting `hit` (ray exhausted
        // its budget inside the screen) we leave it as zero so the
        // upper cascade can supply the longer-range radiance in the
        // merge pass.
        accumulated = accumulated + hit;
    }

    let radiance = accumulated / max(ray_count, 1.0);
    textureStore(raymarch_output, vec2<i32>(gid.xy), vec4<f32>(radiance, 1.0));
}
