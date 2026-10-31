// Ray grid (renderer/src/raygrid.js): a hashed grid over Gaussians for ray-vs-Gaussian candidate narrowing,
// rebuilt every frame and shared by every system that queries rays in that frame (experiments/ray_intersect).
// Producers write items into slot ranges raygrid.js hands out: (centre, reach) in one frame of reference (view space
// for now) and a key = tag << 23 | index (the tags below; the index is the producer's own, < 2^23). The reach R is
// the distance from the centre within which the producer's queries can hit the item; an unused slot has R < 0.
// Nothing is cached, so LOD cuts, proxies and deforming Gaussians work the same. Scale agnostic: an item goes to the
// level whose cell size HG_HMIN 2^(L/2) is the smallest one >= R (half octaves keep cells under 1.41 R), into the
// <= 27 cells its reach ball touches. All levels hash into one table (its size, a power of two, in head word 8, set
// by raygrid.js from the slots in use); collisions only add candidates.
//
// Build, after the producers (raygrid.js encodeBuild): hg_count, hg_scan_blocks / top / add, hg_fill; hg_clear runs before the
// producers. Counting sort: the scan leaves each slot's end, hg_fill counts down from it, so afterwards slot s holds
// its start and the next slot's word its end. Queries walk one level at a time with hg_begin / hg_range / hg_next;
// each hg_refs entry is key | level << 27, and a hit counts only when the entry's level is the walk's and the hit
// point lies in the walk's cell (hg_owns), which drops hash collisions and dedupes the copies.

@group(1) @binding(0) var<storage, read_write> hg_items: array<vec4f>;   // per slot: centre, reach (< 0: unused)
@group(1) @binding(1) var<storage, read_write> hg_keys: array<u32>;
@group(1) @binding(2) var<storage, read_write> hg: array<atomic<u32>>;   // head, then the table
@group(1) @binding(3) var<storage, read_write> hg_refs: array<u32>;      // <= 27 per slot

const HG_HEAD: u32 = 16u;                          // [0,3) box min, [3,6) box max (ordered), [6] level mask, [8] table size
const HG_SLOT: u32 = HG_HEAD;                      // per slot: count -> end (scan) -> start (fill); [table] = total
const HG_LEVELS: u32 = 32u;                        // the level mask is one u32
const HG_HMIN: f32 = 0.0009765625;                 // 1/1024 m; level 31 is 45 m
const HG_KEY_BITS: u32 = 23u;                      // index bits of a key; the tag above them, the level above that
const HG_TAG_GLASS: u32 = 1u;                      // refraction exit trace: Gaussian index | lens << 20 (exitmap.wgsl)
const HG_TAG_INSIDE: u32 = 2u;                     // refraction gather: inside entry | lens << 20 (exitmap.wgsl prep_inner)

fn hg_key(tag: u32, index: u32) -> u32 { return (tag << HG_KEY_BITS) | index; }
fn hg_ord(x: f32) -> u32 { let b = bitcast<u32>(x); return select(b ^ 0x80000000u, ~b, (b & 0x80000000u) != 0u); }
fn hg_unord(u: u32) -> f32 { return bitcast<f32>(select(~u, u ^ 0x80000000u, (u & 0x80000000u) != 0u)); }
fn hg_size(L: u32) -> f32 { return HG_HMIN * exp2(0.5 * f32(L)); }
fn hg_level(R: f32) -> u32 {
  var L = u32(clamp(ceil(2.0 * log2(max(R, 1e-9) / HG_HMIN)), 0.0, f32(HG_LEVELS - 1u)));
  if (hg_size(L) < R && L + 1u < HG_LEVELS) { L += 1u; }     // rounding: the cell must hold the reach (<= 27 cells)
  return L;
}
fn hg_table() -> u32 { return atomicLoad(&hg[8]); }   // 1024 .. 262144, a power of two (raygrid.js)
fn hg_hash(L: u32, c: vec3i) -> u32 {
  let u = bitcast<vec3u>(c);
  return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u) ^ (L * 2654435761u)) & (hg_table() - 1u);
}
fn hg_cell_of(L: u32, x: vec3f) -> vec3i { return vec3i(floor(x / hg_size(L))); }
fn hg_owns(L: u32, c: vec3i, x: vec3f) -> bool { return all(hg_cell_of(L, x) == c); }

// ---- build ----

// the table and box, and every item slot marked unused (a producer that skips a frame leaves nothing behind)
@compute @workgroup_size(256)
fn hg_clear(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i != 8u && i <= HG_SLOT + hg_table()) { atomicStore(&hg[i], select(0u, 0xffffffffu, i < 3u)); }
  if (i < arrayLength(&hg_items)) { hg_items[i].w = -1.0; }
}

// the cells of slot i's reach ball at its level; count = true counts them, else files the item
fn hg_cells(i: u32, count: bool) {
  let it = hg_items[i];
  let p = it.xyz;
  let R = it.w;
  let L = hg_level(R);
  let h = hg_size(L);
  let a = hg_cell_of(L, p - R);
  let b = hg_cell_of(L, p + R);
  let key = hg_keys[i] | (L << 27u);
  for (var x = a.x; x <= b.x; x++) {
    for (var y = a.y; y <= b.y; y++) {
      for (var z = a.z; z <= b.z; z++) {
        let c = vec3i(x, y, z);
        let lo = vec3f(c) * h;
        let e = max(max(lo - p, p - lo - h), vec3f(0.0));       // centre to cell box
        if (dot(e, e) > R * R) { continue; }
        let s = HG_SLOT + hg_hash(L, c);
        if (count) { atomicAdd(&hg[s], 1u); }
        else {
          let at = atomicSub(&hg[s], 1u) - 1u;
          if (at < arrayLength(&hg_refs)) { hg_refs[at] = key; }
        }
      }
    }
  }
}

var<workgroup> hg_wbox: array<atomic<u32>, 7>;    // box min, max (ordered), level mask

// per slot: count its cells; box and level mask reduced per workgroup, then one global atomic each
@compute @workgroup_size(64)
fn hg_count(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < 7u) { atomicStore(&hg_wbox[li], select(0u, 0xffffffffu, li < 3u)); }
  workgroupBarrier();
  let i = gid.x;
  if (i < arrayLength(&hg_keys) && hg_items[i].w >= 0.0) {
    let it = hg_items[i];
    atomicOr(&hg_wbox[6], 1u << hg_level(it.w));
    for (var k = 0u; k < 3u; k++) {
      atomicMin(&hg_wbox[k], hg_ord(it[k] - it.w));
      atomicMax(&hg_wbox[3u + k], hg_ord(it[k] + it.w));
    }
    hg_cells(i, true);
  }
  workgroupBarrier();
  if (li < 3u) { atomicMin(&hg[li], atomicLoad(&hg_wbox[li])); }
  else if (li < 6u) { atomicMax(&hg[li], atomicLoad(&hg_wbox[li])); }
  else if (li == 6u) { atomicOr(&hg[6], atomicLoad(&hg_wbox[6])); }
}

@compute @workgroup_size(64)
fn hg_fill(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < arrayLength(&hg_keys) && hg_items[i].w >= 0.0) { hg_cells(i, false); }
}

var<workgroup> hg_wsum: array<u32, 256>;

// exclusive scan of 256 values in hg_wsum (Hillis-Steele), returns this thread's exclusive prefix
fn hg_wscan(li: u32, v: u32) -> u32 {
  hg_wsum[li] = v;
  workgroupBarrier();
  for (var d = 1u; d < 256u; d <<= 1u) {
    var add = 0u;
    if (li >= d) { add = hg_wsum[li - d]; }
    workgroupBarrier();
    hg_wsum[li] += add;
    workgroupBarrier();
  }
  return hg_wsum[li] - v;
}

// inclusive scan of the table in three dispatches: hg_scan_blocks (table / 1024 groups) scans each block of 1024
// slots and keeps its total, hg_scan_top (1 group) scans the <= 256 totals, hg_scan_add (table / 1024 groups) adds
// each block's offset. (One workgroup over the whole table cost a fixed 0.59 ms on the M4.)
const HG_SUMS: u32 = HG_SLOT + 262144u + 1u;       // per block total, after the largest table

@compute @workgroup_size(256)
fn hg_scan_blocks(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let base = HG_SLOT + wg.x * 1024u + 4u * li;
  let v = vec4u(atomicLoad(&hg[base]), atomicLoad(&hg[base + 1u]), atomicLoad(&hg[base + 2u]), atomicLoad(&hg[base + 3u]));
  let pre = hg_wscan(li, v.x + v.y + v.z + v.w);
  atomicStore(&hg[base], pre + v.x);
  atomicStore(&hg[base + 1u], pre + v.x + v.y);
  atomicStore(&hg[base + 2u], pre + v.x + v.y + v.z);
  atomicStore(&hg[base + 3u], pre + v.x + v.y + v.z + v.w);
  if (li == 255u) { atomicStore(&hg[HG_SUMS + wg.x], pre + v.x + v.y + v.z + v.w); }
}

@compute @workgroup_size(256)
fn hg_scan_top(@builtin(local_invocation_index) li: u32) {
  let nb = hg_table() / 1024u;
  let v = select(0u, atomicLoad(&hg[HG_SUMS + min(li, nb - 1u)]), li < nb);
  let pre = hg_wscan(li, v);
  if (li < nb) { atomicStore(&hg[HG_SUMS + li], pre); }
  if (li == nb - 1u) { atomicStore(&hg[HG_SLOT + hg_table()], pre + v); }   // the total: the last slot's end
}

@compute @workgroup_size(256)
fn hg_scan_add(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let off = atomicLoad(&hg[HG_SUMS + wg.x]);
  if (off == 0u) { return; }
  let base = HG_SLOT + wg.x * 1024u + 4u * li;
  for (var k = 0u; k < 4u; k++) { atomicStore(&hg[base + k], atomicLoad(&hg[base + k]) + off); }   // one writer per slot
}

// ---- queries ----

struct HgWalk { c: vec3i, dc: vec3i, tmax: vec3f, tdel: vec3f, se: f32, tout: f32 }

fn hg_mask() -> u32 { return atomicLoad(&hg[6]); }

// level L along o + s t (t unit), clipped to the grid's box: se > tout means nothing to walk
fn hg_begin(L: u32, o: vec3f, t: vec3f) -> HgWalk {
  let lo = vec3f(hg_unord(atomicLoad(&hg[0])), hg_unord(atomicLoad(&hg[1])), hg_unord(atomicLoad(&hg[2])));
  let hi = vec3f(hg_unord(atomicLoad(&hg[3])), hg_unord(atomicLoad(&hg[4])), hg_unord(atomicLoad(&hg[5])));
  let ts = select(t, vec3f(1e-9), abs(t) < vec3f(1e-9));
  let inv = 1.0 / ts;
  let ta = (lo - o) * inv;
  let tb = (hi - o) * inv;
  var w: HgWalk;
  w.se = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
  w.tout = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  let h = hg_size(L);
  w.c = hg_cell_of(L, o + ts * w.se);
  w.dc = select(vec3i(-1), vec3i(1), ts > vec3f(0.0));
  w.tmax = (vec3f(w.c + max(w.dc, vec3i(0))) * h - o) * inv;
  w.tdel = abs(h * inv);
  return w;
}

// the entries filed under the walk's current cell: hg_refs[x .. y), each key | level << 27
fn hg_range(L: u32, w: HgWalk) -> vec2u {
  let s = HG_SLOT + hg_hash(L, w.c);
  return vec2u(atomicLoad(&hg[s]), atomicLoad(&hg[s + 1u]));
}

// step to the next cell; false once the ray leaves the box
fn hg_next(w: ptr<function, HgWalk>) -> bool {
  if ((*w).tmax.x < (*w).tmax.y && (*w).tmax.x < (*w).tmax.z) { (*w).se = (*w).tmax.x; (*w).c.x += (*w).dc.x; (*w).tmax.x += (*w).tdel.x; }
  else if ((*w).tmax.y < (*w).tmax.z) { (*w).se = (*w).tmax.y; (*w).c.y += (*w).dc.y; (*w).tmax.y += (*w).tdel.y; }
  else { (*w).se = (*w).tmax.z; (*w).c.z += (*w).dc.z; (*w).tmax.z += (*w).tdel.z; }
  return (*w).se <= (*w).tout;
}
