// pba.wgsl — Parallel-Banding-style exact 2D distance transform.
//
// Replaces the JFA cascade (1 init + log2(N) step passes + 1 distance =
// 13 dispatches at 1080p) with three passes that together do O(W*H) work
// and produce an EXACT Euclidean distance transform. The math follows
// Cao/Tang/Tan "PBA" (2010) for the high-level structure and uses
// Felzenszwalb & Huttenlocher's 1D parabolic envelope (2012) for the
// final row sweep. The implementation is a simplified WGSL port:
//
//   init_main   — marks seed pixels (value = y for occupied, SENTINEL otherwise).
//   column_main — per column, two sequential sweeps (top-down, bottom-up)
//                 settle the nearest seed Y for every pixel in that column.
//   row_main    — per row, runs the 1D lower envelope using col_y as the
//                 sampled function f(x) = (y - col_y[x])². The envelope's
//                 closest site gives the EXACT 2D Euclidean distance.
//
// MAX_DIM bounds the private arrays; this implementation supports SDFs
// up to 2048x2048 (covers 1080p and 1440p ultra). Larger needs tiling.

const SENTINEL: u32 = 0xFFFFFFFFu;
const MAX_DIM: u32 = 2048u;
const MAX_DIM_I: i32 = 2048;
const INF: f32 = 1e20;

// =====================================================================
// init_main — depth → r32uint seed marker (y if occupied, SENTINEL if not)
// =====================================================================
@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var seed_out: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8)
fn init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(seed_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let d_dims = textureDimensions(depth_tex);
    let scale = vec2<f32>(f32(d_dims.x), f32(d_dims.y)) /
                vec2<f32>(f32(dims.x), f32(dims.y));
    let d_coord = vec2<i32>((vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) * scale);
    let depth = textureLoad(depth_tex, d_coord, 0);

    var value: u32 = SENTINEL;
    if (depth < 1.0) { value = gid.y; }
    textureStore(seed_out, vec2<i32>(vec2<u32>(gid.x, gid.y)), vec4<u32>(value, 0u, 0u, 0u));
}

// =====================================================================
// column_main — per column, find nearest seed Y for each pixel.
//
// One thread = one column. Two sweeps (top-down then bottom-up) settle
// the candidate from each direction; we keep the closer one. The
// top-down result is stashed in a private array so we don't need a
// separate intermediate texture.
// =====================================================================
@group(0) @binding(0) var col_in: texture_2d<u32>;
@group(0) @binding(1) var col_out: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(64, 1, 1)
fn column_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(col_in);
    let x = i32(gid.x);
    if (x >= i32(dims.x)) { return; }
    let H = i32(dims.y);

    // Forward sweep stashed here. ~8KB private per thread.
    var fwd: array<u32, MAX_DIM>;

    var last: u32 = SENTINEL;
    for (var y: i32 = 0; y < H; y = y + 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL) { last = u32(y); }
        fwd[y] = last;
    }

    last = SENTINEL;
    for (var y: i32 = H - 1; y >= 0; y = y - 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL) { last = u32(y); }

        let f = fwd[y];
        var nearest: u32 = SENTINEL;
        if (f != SENTINEL && last != SENTINEL) {
            let d_f = abs(y - i32(f));
            let d_b = abs(i32(last) - y);
            if (d_f <= d_b) { nearest = f; } else { nearest = last; }
        } else if (f != SENTINEL) {
            nearest = f;
        } else if (last != SENTINEL) {
            nearest = last;
        }
        textureStore(col_out, vec2<i32>(x, y), vec4<u32>(nearest, 0u, 0u, 0u));
    }
}

// =====================================================================
// row_main — per row, 1D lower-envelope DT over columns.
//
// One thread = one row. f(x) = (y - col_y[x, y])² if col_y is valid,
// else +inf. Builds the parabolic lower envelope (sites v[], breakpoints
// z[]) in two phases:
//   1. Forward sweep building the envelope stack.
//   2. Linear scan reading the envelope back into the SDF, advancing
//      the pointer through z[] as x grows.
//
// Result is the EXACT 2D Euclidean distance — no JFA approximation. The
// output is normalised by the screen diagonal so the consumer keeps the
// same units it had with JFA.
// =====================================================================
@group(0) @binding(0) var row_in: texture_2d<u32>;
@group(0) @binding(1) var sdf_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(1, 64, 1)
fn row_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(row_in);
    let y = i32(gid.y);
    if (y >= i32(dims.y)) { return; }
    let W = i32(dims.x);

    // Envelope state. Each ~8KB private; total 16KB this shader.
    var v: array<i32, MAX_DIM>;   // envelope site indices
    var z: array<f32, MAX_DIM>;   // intersections; z[k+1] is the right edge of slot k

    let diag = sqrt(f32(dims.x * dims.x + dims.y * dims.y));

    // Locate the first column that actually has a column-nearest seed.
    var first: i32 = -1;
    for (var x: i32 = 0; x < W; x = x + 1) {
        if (textureLoad(row_in, vec2<i32>(x, y), 0).r != SENTINEL) {
            first = x;
            break;
        }
    }
    if (first < 0) {
        // No seed anywhere in any column at this row — write max distance.
        for (var x: i32 = 0; x < W; x = x + 1) {
            textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
        return;
    }

    var k: i32 = 0;
    v[0] = first;
    z[0] = -INF;
    z[1] = INF;

    // Build envelope.
    for (var q: i32 = first + 1; q < W; q = q + 1) {
        let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
        if (cy_q == SENTINEL) { continue; }
        let dy_q = f32(y - i32(cy_q));
        let f_q = dy_q * dy_q;

        // Pop sites whose dominance region is fully covered by q.
        loop {
            let p = v[k];
            let cy_p = textureLoad(row_in, vec2<i32>(p, y), 0).r;
            let dy_p = f32(y - i32(cy_p));
            let f_p = dy_p * dy_p;

            let s = ((f_q + f32(q) * f32(q)) - (f_p + f32(p) * f32(p))) /
                    (2.0 * f32(q - p));

            if (s > z[k]) {
                k = k + 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = INF;
                break;
            }
            if (k == 0) {
                v[0] = q;
                z[0] = -INF;
                break;
            }
            k = k - 1;
        }
    }

    // Read off — single pass through x, advancing the envelope pointer.
    var kk: i32 = 0;
    for (var x: i32 = 0; x < W; x = x + 1) {
        loop {
            if (kk >= MAX_DIM_I - 1) { break; }
            if (z[kk + 1] >= f32(x)) { break; }
            kk = kk + 1;
        }
        let p = v[kk];
        let cy_p = textureLoad(row_in, vec2<i32>(p, y), 0).r;
        let dy_p = f32(y - i32(cy_p));
        let dx = f32(x - p);
        let dist2 = dx * dx + dy_p * dy_p;
        textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(sqrt(dist2) / diag, 0.0, 0.0, 0.0));
    }
}
