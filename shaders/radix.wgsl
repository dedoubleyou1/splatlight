// Stable sort (option, off by default): an LSD radix sort of project's 16-bit
// keys in two 8-bit passes (shift 0, then shift 8), values = Gaussian index.
// It replaces scatter.wgsl's atomicAdd placement, which puts splats sharing a
// key in a different order every frame: on a flat wall thousands of Gaussians
// share a key and the over-blend of dark and lit neighbours flickers. Here ties
// keep Gaussian index order, so the draw order is the same every frame.
//
// Each pass is three dispatches over tiles of TILE elements (one workgroup per
// tile): `histogram` counts digits per tile into hist[digit * num_tiles + tile],
// `scan` turns that into exclusive offsets (digit-major, so each tile's slot for
// a digit follows every smaller digit and every earlier tile), and `scatter`
// walks the tile in index order and writes each element at its digit's offset
// plus its rank among earlier same-digit elements of the tile.
// INVALID_KEY needs no special case: its digit is 0xFF in both passes and
// project.wgsl clamps valid keys to 0xFFFE, so invalid entries land after every
// valid one, outside the instance count scan.wgsl writes.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var<uniform> params: RadixParams;
@group(0) @binding(2) var<storage, read> keys_in: array<u32>;
@group(0) @binding(3) var<storage, read> vals_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> hist: array<u32>;   // 256 x num_tiles counts, then offsets

struct RadixParams {
  shift: u32,       // 0 or 8
  first: u32,       // 1: values are the element index (vals_in unused)
  last: u32,        // 1: keys_out is not written (only the order is needed)
  num_tiles: u32,   // host-fixed from the buffer size; tiles past the count are empty
}

const THREADS: u32 = 256u;
const BATCHES: u32 = 16u;
const TILE: u32 = THREADS * BATCHES;   // 4096 elements per workgroup
const WORDS: u32 = THREADS / 32u;      // 8 mask words per digit per batch

// Scene Gaussians, the unused slot at `count`, then the emissive Gaussians: the same range scatter.wgsl sorts.
fn element_count() -> u32 {
  // with screen-space LOD (frame.lod) the tree's internal nodes follow at lod.y (project.wgsl)
  if (frame.lod.x != 0u) { return frame.lod.y + frame.lod.z; }
  return frame.count + 1u + u32(frame.palette[3].y);
}

fn digit(key: u32) -> u32 {
  return (key >> params.shift) & 0xffu;
}

var<workgroup> counts: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = element_count();
  atomicStore(&counts[t], 0u);
  workgroupBarrier();
  let base = wg.x * TILE;
  for (var b = 0u; b < BATCHES; b++) {
    let i = base + b * THREADS + t;
    if (i < n) { atomicAdd(&counts[digit(keys_in[i])], 1u); }
  }
  workgroupBarrier();
  hist[t * params.num_tiles + wg.x] = atomicLoad(&counts[t]);
}

var<workgroup> partial: array<u32, 256>;

// One workgroup: thread t owns digit t's num_tiles entries (like scan.wgsl's
// per-thread runs), then a Hillis-Steele scan over the 256 run sums.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) t: u32) {
  let base = t * params.num_tiles;
  var sum = 0u;
  for (var k = 0u; k < params.num_tiles; k++) { sum += hist[base + k]; }
  partial[t] = sum;
  workgroupBarrier();
  for (var off = 1u; off < THREADS; off <<= 1u) {
    var v = partial[t];
    if (t >= off) { v += partial[t - off]; }
    workgroupBarrier();
    partial[t] = v;
    workgroupBarrier();
  }
  var running = partial[t] - sum;
  for (var k = 0u; k < params.num_tiles; k++) {
    let h = hist[base + k];
    hist[base + k] = running;
    running += h;
  }
}

// Per batch of 256 elements: bit masks of which threads hold each digit give
// every element its in-batch rank (popcount of lower lanes); `next` carries
// each digit's write position from batch to batch, so ranks follow index order.
var<workgroup> masks: array<atomic<u32>, 2048>;   // 256 digits x WORDS
var<workgroup> next: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = element_count();
  next[t] = hist[t * params.num_tiles + wg.x];
  for (var k = 0u; k < WORDS; k++) { atomicStore(&masks[t * WORDS + k], 0u); }
  workgroupBarrier();
  let base = wg.x * TILE;
  let w = t / 32u;
  let lane = t % 32u;
  for (var b = 0u; b < BATCHES; b++) {
    let i = base + b * THREADS + t;
    let valid = i < n;
    var key = 0u;
    var d = 0u;
    if (valid) {
      key = keys_in[i];
      d = digit(key);
      atomicOr(&masks[d * WORDS + w], 1u << lane);
    }
    workgroupBarrier();
    var rank = 0u;
    var is_last = false;
    if (valid) {
      let mine = atomicLoad(&masks[d * WORDS + w]);
      rank = countOneBits(mine & ((1u << lane) - 1u));
      for (var k = 0u; k < w; k++) { rank += countOneBits(atomicLoad(&masks[d * WORDS + k])); }
      var above = (mine >> lane) >> 1u;
      for (var k = w + 1u; k < WORDS; k++) { above |= atomicLoad(&masks[d * WORDS + k]); }
      is_last = above == 0u;
      let dst = next[d] + rank;
      var val = i;
      if (params.first == 0u) { val = vals_in[i]; }
      if (params.last == 0u) { keys_out[dst] = key; }
      vals_out[dst] = val;
    }
    workgroupBarrier();   // every thread has read next[] and the masks
    if (is_last) { next[d] += rank + 1u; }
    for (var k = 0u; k < WORDS; k++) { atomicStore(&masks[t * WORDS + k], 0u); }
    workgroupBarrier();
  }
}
