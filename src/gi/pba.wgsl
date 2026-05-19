// pba.wgsl — Parallel-Banding-style exact 2D distance transform.
//
// Three compute passes that together do O(W*H) work and produce an
// EXACT Euclidean distance transform. Math follows Cao/Tang/Tan "PBA"
// (2010) for the separable structure and Felzenszwalb & Huttenlocher
// (2012) for the 1D parabolic lower-envelope.
//
//   init_main   — depth → r32uint seed marker (y if occupied, SENTINEL otherwise).
//   column_main — per column, forward+backward sweep settles nearest seed Y.
//   row_main    — per row, 1D lower envelope over column results → final SDF.
//
// Both column_main and row_main need O(H) and O(W) scratch respectively
// per thread; that overflows the HLSL/D3D12 per-thread temp-register
// limit (4096 regs ≈ 16KB) at any resolution above ~1080p. To remove
// the cap we keep the scratch in a single shared storage buffer — both
// passes alias the same allocation (they run sequentially, so the
// reuse is safe). Allocation size is W*H i32 entries, ~8MB at 1080p
// and ~33MB at 4K. The buffer also drops the MAX_DIM=2048 limit that
// the previous private-array implementation had.

const SENTINEL_U: u32 = 0xFFFFFFFFu;
const SENTINEL_I: i32 = -1;

// =====================================================================
// init_main — depth → r32uint seed marker
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

    var value: u32 = SENTINEL_U;
    if (depth < 1.0) { value = gid.y; }
    textureStore(seed_out, vec2<i32>(vec2<u32>(gid.x, gid.y)), vec4<u32>(value, 0u, 0u, 0u));
}

// =====================================================================
// column_main — per column, find nearest seed Y for each pixel.
//
// PBA+-style banded scan. Each column is split into BAND_COUNT vertical
// bands; one thread handles one band, and BAND_COUNT threads cooperate
// per column via workgroup shared memory. The dispatch launches one
// workgroup per column (workgroup_size = (1, BAND_COUNT, 1)), so the
// effective parallelism is W * BAND_COUNT — for 1080p that's
// 1920 * 32 ≈ 61k threads vs. 1920 in the non-banded version.
//
// Algorithm per band:
//   A. Intra-band: forward sweep within the band. Record the band's
//      first and last seed Y in shared memory.
//   B. workgroupBarrier.
//   C. Each thread independently determines:
//        above_seed = bottom-seed of the closest band above with any seed.
//        below_seed = top-seed of the closest band below with any seed.
//   D. Backward sweep within the band, combining the within-band forward
//      stash (or the propagated above_seed if the band had no seed yet
//      at this y) with the within-band/propagated below candidate.
// =====================================================================
@group(0) @binding(0) var col_in: texture_2d<u32>;
@group(0) @binding(1) var col_out: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<storage, read_write> scratch: array<i32>;

const BAND_COUNT: i32 = 32;

var<workgroup> band_top_seed: array<i32, 32>;  // smallest seed-y in each band (or SENTINEL_I)
var<workgroup> band_bot_seed: array<i32, 32>;  // largest seed-y in each band  (or SENTINEL_I)

@compute @workgroup_size(1, 32, 1)
fn column_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dims = textureDimensions(col_in);
    let x = i32(wgid.x);
    if (x >= i32(dims.x)) { return; }
    let H = i32(dims.y);
    let base = x * H;

    let band_idx = i32(lid.y);
    let band_h = (H + BAND_COUNT - 1) / BAND_COUNT;
    let y_start = band_idx * band_h;
    let y_end = min(y_start + band_h, H);

    // --- A. Intra-band forward sweep. ---
    var first: i32 = SENTINEL_I;
    var last: i32 = SENTINEL_I;
    for (var y: i32 = y_start; y < y_end; y = y + 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL_U) {
            if (first == SENTINEL_I) { first = y; }
            last = y;
        }
        scratch[base + y] = last;  // SENTINEL_I until first seed within band is hit
    }
    band_top_seed[band_idx] = first;
    band_bot_seed[band_idx] = last;

    workgroupBarrier();

    // --- C. Propagate cross-band seed candidates. ---
    var above_seed: i32 = SENTINEL_I;
    for (var b: i32 = 0; b < band_idx; b = b + 1) {
        if (band_bot_seed[b] != SENTINEL_I) { above_seed = band_bot_seed[b]; }
    }
    var below_seed: i32 = SENTINEL_I;
    for (var b: i32 = band_idx + 1; b < BAND_COUNT; b = b + 1) {
        if (band_top_seed[b] != SENTINEL_I) { below_seed = band_top_seed[b]; break; }
    }

    // --- D. Backward sweep within band + combine. ---
    var last_below: i32 = below_seed;
    for (var y: i32 = y_end - 1; y >= y_start; y = y - 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL_U) { last_below = y; }

        var f: i32 = scratch[base + y];
        if (f == SENTINEL_I) { f = above_seed; }

        var nearest: u32 = SENTINEL_U;
        if (f != SENTINEL_I && last_below != SENTINEL_I) {
            let d_f = abs(y - f);
            let d_b = abs(last_below - y);
            if (d_f <= d_b) { nearest = u32(f); } else { nearest = u32(last_below); }
        } else if (f != SENTINEL_I) {
            nearest = u32(f);
        } else if (last_below != SENTINEL_I) {
            nearest = u32(last_below);
        }
        textureStore(col_out, vec2<i32>(x, y), vec4<u32>(nearest, 0u, 0u, 0u));
    }
}

// =====================================================================
// row_main — per row, 1D lower-envelope DT over columns.
//
// f(x) = (y - col_y[x, y])² if col_y is valid, else +inf. Builds the
// parabolic lower envelope (sites v[k] = x indices) and reads it off
// to produce the exact 2D Euclidean distance. The envelope's z[]
// breakpoints are NOT stored; we recompute them on the fly from v[]
// to halve scratch usage.
// =====================================================================
@group(0) @binding(0) var row_in: texture_2d<u32>;
@group(0) @binding(1) var sdf_out: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read_write> row_scratch: array<i32>;

@compute @workgroup_size(1, 64, 1)
fn row_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(row_in);
    let y = i32(gid.y);
    if (y >= i32(dims.y)) { return; }
    let W = i32(dims.x);
    let base = y * W;

    var num: i32 = 0;
    let diag = sqrt(f32(dims.x * dims.x + dims.y * dims.y));

    // Build envelope.
    for (var q: i32 = 0; q < W; q = q + 1) {
        let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
        if (cy_q == SENTINEL_U) { continue; }
        let dy_q = f32(y - i32(cy_q));
        let f_q = dy_q * dy_q;

        loop {
            if (num == 0) { break; }
            let p = row_scratch[base + num - 1];
            let cy_p = textureLoad(row_in, vec2<i32>(p, y), 0).r;
            let dy_p = f32(y - i32(cy_p));
            let f_p = dy_p * dy_p;

            // Intersection between parabolas at p and q.
            let s_pq = ((f_q + f32(q) * f32(q)) - (f_p + f32(p) * f32(p))) /
                       (2.0 * f32(q - p));

            // Previous-site breakpoint (recomputed). Sentinel = -inf
            // when there's no v[num-2].
            var prev_z: f32 = -1e20;
            if (num >= 2) {
                let pp = row_scratch[base + num - 2];
                let cy_pp = textureLoad(row_in, vec2<i32>(pp, y), 0).r;
                let dy_pp = f32(y - i32(cy_pp));
                let f_pp = dy_pp * dy_pp;
                prev_z = ((f_p + f32(p) * f32(p)) - (f_pp + f32(pp) * f32(pp))) /
                         (2.0 * f32(p - pp));
            }

            if (s_pq > prev_z) { break; }
            num = num - 1;
        }
        row_scratch[base + num] = q;
        num = num + 1;
    }

    if (num == 0) {
        for (var x: i32 = 0; x < W; x = x + 1) {
            textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
        return;
    }

    // Read off.
    var k: i32 = 0;
    for (var x: i32 = 0; x < W; x = x + 1) {
        loop {
            if (k + 1 >= num) { break; }
            let p = row_scratch[base + k];
            let q = row_scratch[base + k + 1];
            let cy_p = textureLoad(row_in, vec2<i32>(p, y), 0).r;
            let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
            let dy_p = f32(y - i32(cy_p));
            let dy_q = f32(y - i32(cy_q));
            let f_p = dy_p * dy_p;
            let f_q = dy_q * dy_q;
            let z = ((f_q + f32(q) * f32(q)) - (f_p + f32(p) * f32(p))) /
                    (2.0 * f32(q - p));
            if (f32(x) < z) { break; }
            k = k + 1;
        }
        let p = row_scratch[base + k];
        let cy_p = textureLoad(row_in, vec2<i32>(p, y), 0).r;
        let dy_p = f32(y - i32(cy_p));
        let dx = f32(x - p);
        let dist2 = dx * dx + dy_p * dy_p;
        textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(sqrt(dist2) / diag, 0.0, 0.0, 0.0));
    }
}
