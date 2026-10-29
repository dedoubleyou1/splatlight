// Pass 2 of the counting sort on the 16-bit depth key: one workgroup turns
// the 65536-bucket histogram into exclusive offsets and writes the indirect
// draw's instance count. Pass 3 (scatter.wgsl) places each visible splat.
// Splats sharing a key land in arbitrary order; at 16-bit log depth between
// near and far that is a sub-millimetre tie at room scale, but the order changes
// every frame. The stable-sort option (radix.wgsl) replaces pass 3 and keeps
// ties in Gaussian index order; this pass still writes the instance count.

@group(0) @binding(0) var<storage, read_write> histogram: array<u32>;   // counts in, offsets out
@group(0) @binding(1) var<storage, read_write> draw_args: array<u32, 4>;

const SCAN_THREADS: u32 = 256u;
const PER_THREAD: u32 = NUM_BUCKETS / SCAN_THREADS;
var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  let base = t * PER_THREAD;
  var sum = 0u;
  for (var k = 0u; k < PER_THREAD; k++) { sum += histogram[base + k]; }
  partial[t] = sum;
  workgroupBarrier();
  // Hillis-Steele inclusive scan over the 256 per-thread sums.
  for (var off = 1u; off < SCAN_THREADS; off <<= 1u) {
    var v = partial[t];
    if (t >= off) { v += partial[t - off]; }
    workgroupBarrier();
    partial[t] = v;
    workgroupBarrier();
  }
  var running = partial[t] - sum;
  for (var k = 0u; k < PER_THREAD; k++) {
    let h = histogram[base + k];
    histogram[base + k] = running;
    running += h;
  }
  if (t == SCAN_THREADS - 1u) {
    draw_args[0] = 4u;              // vertices per quad (triangle strip)
    draw_args[1] = partial[t];      // visible splats
    draw_args[2] = 0u;
    draw_args[3] = 0u;
  }
}
