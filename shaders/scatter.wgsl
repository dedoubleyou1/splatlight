// Pass 3: scatter visible splats into back-to-front order using the offsets
// the scan pass left in the histogram buffer.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> order: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  var last = select(frame.count + u32(frame.palette[3].y), frame.lod.y + frame.lod.z - 1u, frame.lod.x != 0u);   // + LOD nodes
  if (frame.particles.y != 0u) { last = frame.particles.x + frame.particles.y - 1u; }   // + Gaussian particles
  if (i > last) { return; }
  let key = keys[i];
  if (key == INVALID_KEY) { return; }
  order[atomicAdd(&offsets[key], 1u)] = i;
}
