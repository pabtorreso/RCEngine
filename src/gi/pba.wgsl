// pba.wgsl — Parallel-Banding-style exact 2D distance transform.
//
// Three compute passes that together do O(W*H) work and produce an
// EXACT Euclidean distance transform. Math follows Cao/Tang/Tan "PBA"
// (2010) for the separable structure and Felzenszwalb & Huttenlocher
// (2012) for the 1D parabolic lower-envelope.
//
//   init_main   — depth → r32uint seed marker (y if occupied, SENTINEL otherwise).
//   column_main — banded per-column scan: one workgroup per column, 32 threads
//                 cooperating across vertical bands via workgroup shared memory.
//                 Per-band scratch lives in PRIVATE memory (not storage buffer)
//                 because each band is small (~H/32 entries) and registers are
//                 ~10x faster than global memory reads.
//   row_main    — per row, 1D lower envelope over column-nearest-seeds.
//                 Private array v[] holds envelope sites; z[] breakpoints are
//                 dropped and recomputed lazily to fit under D3D12's 4096-reg
//                 per-thread limit (MAX_DIM=2048 entries = ~8KB = 2048 regs).
//
// Capacity: width up to MAX_DIM=2048 in row_main; height up to BAND_COUNT *
// MAX_BAND_H = 32 * 128 = 4096 in column_main. Covers 1080p and most of
// 1440p Ultra; only the row_main width cap would bite at 1440p widescreen
// or above. A storage-buffer fallback for that case can be added later.

const SENTINEL_U: u32 = 0xFFFFFFFFu;
const SENTINEL_I: i32 = -1;
const MAX_DIM: u32 = 2048u;
const MAX_DIM_I: i32 = 2048;

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
// column_main — banded per-column nearest-seed-Y.
//
// One workgroup per column, 32 threads collaborating across bands.
// Each thread handles ~H/32 rows in private memory (~512 bytes), shares
// only its band's first/last seed in workgroup shared memory.
// =====================================================================
@group(0) @binding(0) var col_in: texture_2d<u32>;
@group(0) @binding(1) var col_out: texture_storage_2d<r32uint, write>;

const BAND_COUNT: i32 = 32;
const MAX_BAND_H: u32 = 128u;   // up to 4096 image height ÷ 32 bands

var<workgroup> band_top_seed: array<i32, 32>;  // smallest seed-y in each band
var<workgroup> band_bot_seed: array<i32, 32>;  // largest seed-y in each band

@compute @workgroup_size(1, 32, 1)
fn column_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dims = textureDimensions(col_in);
    let x = i32(wgid.x);
    if (x >= i32(dims.x)) { return; }
    let H = i32(dims.y);

    let band_idx = i32(lid.y);
    let band_h = (H + BAND_COUNT - 1) / BAND_COUNT;
    let y_start = band_idx * band_h;
    let y_end = min(y_start + band_h, H);

    // Per-thread private scratch — one slot per row in this band.
    var band_scratch: array<i32, MAX_BAND_H>;

    // --- A. Intra-band forward sweep + collect band-boundary seeds. ---
    var first: i32 = SENTINEL_I;
    var last: i32 = SENTINEL_I;
    for (var y: i32 = y_start; y < y_end; y = y + 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL_U) {
            if (first == SENTINEL_I) { first = y; }
            last = y;
        }
        band_scratch[y - y_start] = last;
    }
    band_top_seed[band_idx] = first;
    band_bot_seed[band_idx] = last;

    workgroupBarrier();

    // --- B. Cross-band propagation. ---
    var above_seed: i32 = SENTINEL_I;
    for (var b: i32 = 0; b < band_idx; b = b + 1) {
        if (band_bot_seed[b] != SENTINEL_I) { above_seed = band_bot_seed[b]; }
    }
    var below_seed: i32 = SENTINEL_I;
    for (var b: i32 = band_idx + 1; b < BAND_COUNT; b = b + 1) {
        if (band_top_seed[b] != SENTINEL_I) { below_seed = band_top_seed[b]; break; }
    }

    // --- C. Backward sweep within band + combine. ---
    var last_below: i32 = below_seed;
    for (var y: i32 = y_end - 1; y >= y_start; y = y - 1) {
        let v = textureLoad(col_in, vec2<i32>(x, y), 0).r;
        if (v != SENTINEL_U) { last_below = y; }

        var f: i32 = band_scratch[y - y_start];
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
// One thread per row. f(x) = (y - col_y[x, y])² if col_y is valid, else
// +inf. v[k] holds the envelope sites; the breakpoints z[k] are NOT
// stored — they're recomputed lazily from v[] during both the build and
// the read-off scan. That halves private memory at the cost of ~3x more
// texture reads, but keeps us comfortably under D3D12's 4096-temp-reg
// per-thread limit.
// =====================================================================
@group(0) @binding(0) var row_in: texture_2d<u32>;
@group(0) @binding(1) var sdf_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(1, 64, 1)
fn row_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(row_in);
    let y = i32(gid.y);
    if (y >= i32(dims.y)) { return; }
    let W = i32(dims.x);

    var v: array<i32, MAX_DIM>;
    var num: i32 = 0;
    let diag = sqrt(f32(dims.x * dims.x + dims.y * dims.y));

    // Build envelope. Top-of-stack and second-from-top are cached in
    // registers (top_*, sec_*) so the pop comparison reads ZERO from
    // the row_in texture; only when a pop demotes the second do we
    // refetch the next site below. This eliminates ~2 reads per pop
    // (≈2M reads/frame at 1080p).
    var top_p: i32 = 0;
    var top_dy: f32 = 0.0;
    var top_f: f32 = 0.0;
    var top_p_sq: f32 = 0.0;

    var sec_p: i32 = 0;
    var sec_dy: f32 = 0.0;
    var sec_f: f32 = 0.0;
    var sec_p_sq: f32 = 0.0;

    for (var q: i32 = 0; q < W; q = q + 1) {
        let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
        if (cy_q == SENTINEL_U) { continue; }
        let dy_q = f32(y - i32(cy_q));
        let f_q = dy_q * dy_q;
        let q_sq = f32(q) * f32(q);

        loop {
            if (num == 0) { break; }
            // Pop comparison: uses cached top_* and sec_*, no texture reads.
            let s_pq = ((f_q + q_sq) - (top_f + top_p_sq)) /
                       (2.0 * f32(q - top_p));
            var prev_z: f32 = -1e20;
            if (num >= 2) {
                prev_z = ((top_f + top_p_sq) - (sec_f + sec_p_sq)) /
                         (2.0 * f32(top_p - sec_p));
            }
            if (s_pq > prev_z) { break; }

            // Pop: cached second becomes top, refetch new second (if any).
            num = num - 1;
            if (num >= 1) {
                top_p = sec_p;
                top_dy = sec_dy;
                top_f = sec_f;
                top_p_sq = sec_p_sq;
                if (num >= 2) {
                    sec_p = v[num - 2];
                    let scy = textureLoad(row_in, vec2<i32>(sec_p, y), 0).r;
                    sec_dy = f32(y - i32(scy));
                    sec_f = sec_dy * sec_dy;
                    sec_p_sq = f32(sec_p) * f32(sec_p);
                }
            }
        }
        if (num < MAX_DIM_I) {
            // Push q. Old top demotes to second.
            if (num >= 1) {
                sec_p = top_p;
                sec_dy = top_dy;
                sec_f = top_f;
                sec_p_sq = top_p_sq;
            }
            v[num] = q;
            top_p = q;
            top_dy = dy_q;
            top_f = f_q;
            top_p_sq = q_sq;
            num = num + 1;
        }
    }

    if (num == 0) {
        for (var x: i32 = 0; x < W; x = x + 1) {
            textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
        return;
    }

    // Read off — cache the current envelope site (and the next one we'd
    // advance into) in registers so the per-pixel inner loop touches the
    // row_in texture ONLY when k advances. Previously each x re-loaded
    // cy_p, which added ~W redundant texture reads per row (~2M at 1080p).
    var k: i32 = 0;
    var cur_p: i32 = v[0];
    let cur_cy_load = textureLoad(row_in, vec2<i32>(cur_p, y), 0).r;
    var cur_dy: f32 = f32(y - i32(cur_cy_load));
    var cur_f: f32 = cur_dy * cur_dy;

    var has_next: bool = (num > 1);
    var next_p: i32 = 0;
    var next_dy: f32 = 0.0;
    var next_f: f32 = 0.0;
    if (has_next) {
        next_p = v[1];
        let next_cy_load = textureLoad(row_in, vec2<i32>(next_p, y), 0).r;
        next_dy = f32(y - i32(next_cy_load));
        next_f = next_dy * next_dy;
    }

    for (var x: i32 = 0; x < W; x = x + 1) {
        // Advance k while x is past the breakpoint between cur and next.
        loop {
            if (!has_next) { break; }
            let z = ((next_f + f32(next_p) * f32(next_p)) - (cur_f + f32(cur_p) * f32(cur_p))) /
                    (2.0 * f32(next_p - cur_p));
            if (f32(x) < z) { break; }
            // Advance: next slides into cur, fetch new next.
            cur_p = next_p;
            cur_dy = next_dy;
            cur_f = next_f;
            k = k + 1;
            has_next = (k + 1 < num);
            if (has_next) {
                next_p = v[k + 1];
                let nc = textureLoad(row_in, vec2<i32>(next_p, y), 0).r;
                next_dy = f32(y - i32(nc));
                next_f = next_dy * next_dy;
            }
        }
        let dx = f32(x - cur_p);
        let dist2 = dx * dx + cur_dy * cur_dy;
        textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(sqrt(dist2) / diag, 0.0, 0.0, 0.0));
    }
}
