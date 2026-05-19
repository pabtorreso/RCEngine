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
// MAX_DIM caps the row envelope. 2000 (not 2048) keeps row_main's
// workgroup-shared layout (band_v 8KB + band_num 128B + global_v +
// global_num) just under the WebGPU 16384-byte workgroup-storage
// minimum. Width is still effectively capped at 2048 by other paths;
// the envelope rarely approaches even W on real scenes, so trimming
// 48 entries off the cap is invisible in practice.
const MAX_DIM: u32 = 2000u;
const MAX_DIM_I: i32 = 2000;

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
// row_main — PBA-paper-style banded 1D envelope per row.
//
// One WORKGROUP per row, ROW_BAND_COUNT threads cooperating:
//   Phase A — each band thread builds the local envelope for its slice
//             of columns into workgroup shared memory.
//   Phase B — thread 0 merges all band envelopes into a single global
//             envelope (also in shared memory). This is the only
//             sequential part; the rest is fully parallel.
//   Phase C — every thread reads off the global envelope for its band's
//             pixels and writes the SDF. cur/next sites are kept in
//             registers so the inner loop is texture-read-free on stay.
//
// Workgroup-memory budget (~16KB, the WebGPU spec minimum):
//   band_v[32][64] i32 = 8KB   (per-band envelope sites)
//   band_num[32]   i32 = 128B  (per-band site counts)
//   global_v[2048] i32 = 8KB   (merged envelope sites)
//   global_num     i32 = 4B
// =====================================================================
@group(0) @binding(0) var row_in: texture_2d<u32>;
@group(0) @binding(1) var sdf_out: texture_storage_2d<r32float, write>;

const ROW_BAND_COUNT: i32 = 32;
const MAX_SITES_PER_BAND: u32 = 64u;
const MAX_SITES_PER_BAND_I: i32 = 64;

var<workgroup> band_v: array<array<i32, MAX_SITES_PER_BAND>, 32>;
var<workgroup> band_num: array<i32, 32>;
var<workgroup> global_v: array<i32, MAX_DIM>;
var<workgroup> global_num: i32;

@compute @workgroup_size(32, 1, 1)
fn row_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dims = textureDimensions(row_in);
    let y = i32(wgid.x);
    if (y >= i32(dims.y)) { return; }
    let W = i32(dims.x);
    let band_idx = i32(lid.x);
    let band_w = (W + ROW_BAND_COUNT - 1) / ROW_BAND_COUNT;
    let x_start = band_idx * band_w;
    let x_end = min(x_start + band_w, W);
    let diag = sqrt(f32(dims.x * dims.x + dims.y * dims.y));

    // =================================================================
    // Phase A — local envelope build for this band.
    // Same top/sec register-cached pattern as the single-threaded
    // version, but operating on only this band's column range.
    // =================================================================
    var local_num: i32 = 0;
    var top_p: i32 = 0;
    var top_f: f32 = 0.0;
    var top_p_sq: f32 = 0.0;
    var sec_p: i32 = 0;
    var sec_f: f32 = 0.0;
    var sec_p_sq: f32 = 0.0;

    for (var q: i32 = x_start; q < x_end; q = q + 1) {
        let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
        if (cy_q == SENTINEL_U) { continue; }
        let dy_q = f32(y - i32(cy_q));
        let f_q = dy_q * dy_q;
        let q_sq = f32(q) * f32(q);

        loop {
            if (local_num == 0) { break; }
            let s_pq = ((f_q + q_sq) - (top_f + top_p_sq)) /
                       (2.0 * f32(q - top_p));
            var prev_z: f32 = -1e20;
            if (local_num >= 2) {
                prev_z = ((top_f + top_p_sq) - (sec_f + sec_p_sq)) /
                         (2.0 * f32(top_p - sec_p));
            }
            if (s_pq > prev_z) { break; }

            local_num = local_num - 1;
            if (local_num >= 1) {
                top_p = sec_p;
                top_f = sec_f;
                top_p_sq = sec_p_sq;
                if (local_num >= 2) {
                    sec_p = band_v[band_idx][local_num - 2];
                    let scy = textureLoad(row_in, vec2<i32>(sec_p, y), 0).r;
                    let sdy = f32(y - i32(scy));
                    sec_f = sdy * sdy;
                    sec_p_sq = f32(sec_p) * f32(sec_p);
                }
            }
        }
        if (local_num < MAX_SITES_PER_BAND_I) {
            if (local_num >= 1) {
                sec_p = top_p;
                sec_f = top_f;
                sec_p_sq = top_p_sq;
            }
            band_v[band_idx][local_num] = q;
            top_p = q;
            top_f = f_q;
            top_p_sq = q_sq;
            local_num = local_num + 1;
        }
    }
    band_num[band_idx] = local_num;

    workgroupBarrier();

    // =================================================================
    // Phase B — thread 0 merges all band envelopes into global_v.
    // Iterates each band's sites in order, applying the same envelope
    // construction (top/sec cached) as Phase A. Site positions are
    // monotonically increasing across bands so the merge is just
    // another linear envelope build over the union.
    // =================================================================
    if (band_idx == 0) {
        var gnum: i32 = 0;
        var g_top_p: i32 = 0;
        var g_top_f: f32 = 0.0;
        var g_top_p_sq: f32 = 0.0;
        var g_sec_p: i32 = 0;
        var g_sec_f: f32 = 0.0;
        var g_sec_p_sq: f32 = 0.0;

        for (var b: i32 = 0; b < ROW_BAND_COUNT; b = b + 1) {
            let blen = band_num[b];
            for (var i: i32 = 0; i < blen; i = i + 1) {
                let q = band_v[b][i];
                let cy_q = textureLoad(row_in, vec2<i32>(q, y), 0).r;
                let dy_q = f32(y - i32(cy_q));
                let f_q = dy_q * dy_q;
                let q_sq = f32(q) * f32(q);

                loop {
                    if (gnum == 0) { break; }
                    let s_pq = ((f_q + q_sq) - (g_top_f + g_top_p_sq)) /
                               (2.0 * f32(q - g_top_p));
                    var prev_z: f32 = -1e20;
                    if (gnum >= 2) {
                        prev_z = ((g_top_f + g_top_p_sq) - (g_sec_f + g_sec_p_sq)) /
                                 (2.0 * f32(g_top_p - g_sec_p));
                    }
                    if (s_pq > prev_z) { break; }

                    gnum = gnum - 1;
                    if (gnum >= 1) {
                        g_top_p = g_sec_p;
                        g_top_f = g_sec_f;
                        g_top_p_sq = g_sec_p_sq;
                        if (gnum >= 2) {
                            g_sec_p = global_v[gnum - 2];
                            let scy = textureLoad(row_in, vec2<i32>(g_sec_p, y), 0).r;
                            let sdy = f32(y - i32(scy));
                            g_sec_f = sdy * sdy;
                            g_sec_p_sq = f32(g_sec_p) * f32(g_sec_p);
                        }
                    }
                }
                if (gnum < MAX_DIM_I) {
                    if (gnum >= 1) {
                        g_sec_p = g_top_p;
                        g_sec_f = g_top_f;
                        g_sec_p_sq = g_top_p_sq;
                    }
                    global_v[gnum] = q;
                    g_top_p = q;
                    g_top_f = f_q;
                    g_top_p_sq = q_sq;
                    gnum = gnum + 1;
                }
            }
        }
        global_num = gnum;
    }

    workgroupBarrier();

    // =================================================================
    // Phase C — each thread reads off its band's pixels using the
    // merged global envelope. Same cur/next register caching as the
    // single-threaded read-off.
    // =================================================================
    let gnum_local = global_num;
    if (gnum_local == 0) {
        for (var x: i32 = x_start; x < x_end; x = x + 1) {
            textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        }
        return;
    }

    var k: i32 = 0;
    var cur_p: i32 = global_v[0];
    let cur_cy = textureLoad(row_in, vec2<i32>(cur_p, y), 0).r;
    var cur_dy: f32 = f32(y - i32(cur_cy));
    var cur_f: f32 = cur_dy * cur_dy;

    var has_next: bool = (gnum_local > 1);
    var next_p: i32 = 0;
    var next_dy: f32 = 0.0;
    var next_f: f32 = 0.0;
    if (has_next) {
        next_p = global_v[1];
        let ncy = textureLoad(row_in, vec2<i32>(next_p, y), 0).r;
        next_dy = f32(y - i32(ncy));
        next_f = next_dy * next_dy;
    }

    // Advance k from 0 up to where x_start sits in the envelope.
    loop {
        if (!has_next) { break; }
        let z = ((next_f + f32(next_p) * f32(next_p)) - (cur_f + f32(cur_p) * f32(cur_p))) /
                (2.0 * f32(next_p - cur_p));
        if (f32(x_start) < z) { break; }
        cur_p = next_p;
        cur_dy = next_dy;
        cur_f = next_f;
        k = k + 1;
        has_next = (k + 1 < gnum_local);
        if (has_next) {
            next_p = global_v[k + 1];
            let ncy = textureLoad(row_in, vec2<i32>(next_p, y), 0).r;
            next_dy = f32(y - i32(ncy));
            next_f = next_dy * next_dy;
        }
    }

    for (var x: i32 = x_start; x < x_end; x = x + 1) {
        loop {
            if (!has_next) { break; }
            let z = ((next_f + f32(next_p) * f32(next_p)) - (cur_f + f32(cur_p) * f32(cur_p))) /
                    (2.0 * f32(next_p - cur_p));
            if (f32(x) < z) { break; }
            cur_p = next_p;
            cur_dy = next_dy;
            cur_f = next_f;
            k = k + 1;
            has_next = (k + 1 < gnum_local);
            if (has_next) {
                next_p = global_v[k + 1];
                let ncy = textureLoad(row_in, vec2<i32>(next_p, y), 0).r;
                next_dy = f32(y - i32(ncy));
                next_f = next_dy * next_dy;
            }
        }
        let dx = f32(x - cur_p);
        let dist2 = dx * dx + cur_dy * cur_dy;
        textureStore(sdf_out, vec2<i32>(x, y), vec4<f32>(sqrt(dist2) / diag, 0.0, 0.0, 0.0));
    }
}
