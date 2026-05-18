// rc_raymarch.wgsl — Hierarchical Screen-Space Radiance Cascades raymarch.
//
// Cascade `N` runs at resolution `width/2^N × height/2^N`. The cascade
// pixel `(gx, gy)` represents the *probe* that lives at the screen
// position `((gx + 0.5) * 2^N, (gy + 0.5) * 2^N)`. We sphere-trace from
// that screen position over the full-resolution SDF, sampling the
// emissive G-Buffer at every hit and returning a soft directional sky
// when a ray escapes the screen.
//
// Output is written at cascade resolution. The merge pass bilinearly
// upsamples the parent cascade onto the current one.

struct RCPassParams {
    cascade_index: i32,
    base_ray_count: i32,
    sun_intensity: f32,
    cascade_width: i32,
    cascade_height: i32,
    screen_width: i32,
    screen_height: i32,
    _pad0: i32,
    _padding: array<vec4<u32>, 14>,
}

@group(0) @binding(0) var g_sdf: texture_2d<f32>;
@group(0) @binding(1) var g_emissive: texture_2d<f32>;
@group(0) @binding(2) var g_normal: texture_2d<f32>;
@group(0) @binding(3) var raymarch_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: RCPassParams;

const PI: f32 = 3.14159265359;
const HIT_EPSILON_PX: f32 = 1.0;
const MAX_RAYS: f32 = 64.0;

fn skyRadiance(dir: vec2<f32>, intensity: f32) -> vec3<f32> {
    let up = clamp(-dir.y, 0.0, 1.0);
    let zenith = vec3<f32>(0.35, 0.55, 0.95);
    let horizon = vec3<f32>(0.85, 0.75, 0.55);
    return mix(horizon, zenith, up) * intensity;
}

// Hash a 2D integer coordinate into a `[0, 1)` float. Deterministic per
// pixel, so it gives spatial-jitter without introducing temporal flicker
// (which would happen if we mixed in the frame index).
fn hash2D(p: vec2<u32>) -> f32 {
    var x: u32 = p.x * 73856093u;
    x = x ^ (p.y * 19349663u);
    x = x ^ (x >> 16u);
    x = x * 2654435761u;
    x = x ^ (x >> 15u);
    return f32(x & 0xffffffu) / f32(0x1000000u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = u32(params.cascade_width);
    let ch = u32(params.cascade_height);
    if (gid.x >= cw || gid.y >= ch) { return; }

    let sw = f32(params.screen_width);
    let sh = f32(params.screen_height);
    let cascade_idx = f32(params.cascade_index);
    let base_rays = f32(params.base_ray_count);
    // Cap the ray count: cascades 4+ would otherwise march 64-128 rays
    // over very few pixels with diminishing angular benefit.
    let ray_count = min(base_rays * pow(2.0, cascade_idx), MAX_RAYS);
    let t_0 = pow(4.0, cascade_idx);
    let t_1 = pow(4.0, cascade_idx + 1.0);

    // Map cascade pixel → screen pixel. Derive the scale from real
    // dimensions so this also works when the user picks a coarser GI
    // quality level (cascade 0 becomes half-res, etc.).
    let scale = vec2<f32>(sw / f32(cw), sh / f32(ch));

    // Per-pixel rotation of the ray bundle. Breaks the angular banding
    // that appears when neighbouring pixels share the exact same set of
    // ray directions, at zero extra GPU cost.
    let jitter = hash2D(gid.xy);

    // Probe position in screen-space coordinates.
    let pos = (vec2<f32>(gid.xy) + 0.5) * scale;
    let diagonal = sqrt(sw * sw + sh * sh);

    // The SDF texture may be at a coarser resolution than the screen
    // (when the user picks SDF quality High / Performance). Convert
    // any screen-space coord into SDF-space before sampling.
    let sdf_dims = vec2<f32>(textureDimensions(g_sdf));
    let sdf_to_screen = sdf_dims / vec2<f32>(sw, sh);

    // Cap the per-ray step count so distant cascades don't burn cycles
    // on rays that have already escaped. The longer the interval the
    // fewer SDF samples it can afford anyway.
    let max_steps_f = clamp(64.0 / (1.0 + cascade_idx), 8.0, 32.0);
    let max_steps = i32(max_steps_f);

    let pos_screen_i = vec2<i32>(clamp(pos, vec2<f32>(0.0), vec2<f32>(sw - 1.0, sh - 1.0)));
    let center_sdf_coord = vec2<i32>(vec2<f32>(pos_screen_i) * sdf_to_screen);
    let center_sdf_px = textureLoad(g_sdf, center_sdf_coord, 0).r * diagonal;

    // Background pixels (very far from any geometry) contribute almost
    // nothing — skip the entire ray bundle for them.
    if (center_sdf_px > diagonal * 0.5) {
        textureStore(raymarch_output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    var accumulated = vec3<f32>(0.0);
    let angular_step = (2.0 * PI) / ray_count;

    for (var i: i32 = 0; i < i32(ray_count); i = i + 1) {
        let angle = (f32(i) + jitter) * angular_step;
        let dir = vec2<f32>(cos(angle), sin(angle));

        var t = t_0;
        var hit = vec3<f32>(0.0);

        for (var step: i32 = 0; step < max_steps; step = step + 1) {
            if (t >= t_1) { break; }

            let p = pos + dir * t;
            if (p.x < 0.0 || p.x >= sw || p.y < 0.0 || p.y >= sh) {
                hit = skyRadiance(dir, params.sun_intensity);
                break;
            }

            let dist_px = textureLoad(g_sdf, vec2<i32>(p * sdf_to_screen), 0).r * diagonal;
            if (dist_px < HIT_EPSILON_PX) {
                let emissive = textureLoad(g_emissive, vec2<i32>(p), 0).rgb;
                let n3 = textureLoad(g_normal, vec2<i32>(p), 0).xyz;
                let n2 = normalize(vec2<f32>(n3.x, -n3.y) + vec2<f32>(1e-5));
                let facing = clamp(dot(n2, -dir), 0.0, 1.0);
                hit = emissive * (0.25 + 0.75 * facing);
                break;
            }

            t = t + max(dist_px, 0.5);
        }

        accumulated = accumulated + hit;
    }

    let radiance = accumulated / max(ray_count, 1.0);
    textureStore(raymarch_output, vec2<i32>(gid.xy), vec4<f32>(radiance, 1.0));
}
