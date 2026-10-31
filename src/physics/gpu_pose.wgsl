// Pose handoff (gpu_client.js), appended to gpu_kernels.wgsl + gpu_step.wgsl: after the frame's physics steps, each
// dynamic body's world_from_rest rows go straight into the renderer's part records (renderer.partBuf, asset Part: xf
// 12 f32 at 0, xf_prev 12 at 12, flags u32 at 31), with no readback. xf_prev is this body's previous pose as the
// physics last wrote it (pose_last), not what the CPU wrote this frame; the moved flag (bit 1) is set when the pose
// changed. world.js worldFromRest: R = R(q) R0^T, t = p - R com0.

// per body 4 vec4: (asset part index, 1 if handed off else 0, -, -) | R0 row 0, com0.x | R0 row 1, com0.y | R0 row 2, com0.z
@group(0) @binding(64) var<storage, read> pose_map: array<vec4f>;
@group(0) @binding(65) var<storage, read_write> parts_out: array<u32>;   // renderer.partBuf: 32 u32 per part
@group(0) @binding(66) var<storage, read_write> pose_last: array<f32>;   // per body 16: last written xf (12), valid (12)

@compute @workgroup_size(64)
fn pose_out(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody) { return; }
  let m = pose_map[4u * b];
  if (m.y == 0.0) { return; }
  let part = u32(m.x);
  let R = quat_mat(body_quat(b));
  let x = body_pos(b);
  let r0 = pose_map[4u * b + 1u]; let r1 = pose_map[4u * b + 2u]; let r2 = pose_map[4u * b + 3u];
  let com = vec3f(r0.w, r1.w, r2.w);
  var xf: array<f32, 12>;
  for (var r = 0u; r < 3u; r++) {
    let row = vec3f(R[0][r], R[1][r], R[2][r]);   // row r of R (mat3x3f is column-major)
    let a = vec3f(dot(row, r0.xyz), dot(row, r1.xyz), dot(row, r2.xyz));   // (R R0^T)[r][k]
    xf[4u * r] = a.x; xf[4u * r + 1u] = a.y; xf[4u * r + 2u] = a.z;
    xf[4u * r + 3u] = x[r] - dot(a, com);
  }
  let o = part * 32u; let l = b * 16u;
  let valid = pose_last[l + 12u] != 0.0;
  var moved = false;
  for (var k = 0u; k < 12u; k++) {
    let prev = select(xf[k], pose_last[l + k], valid);
    moved = moved || prev != xf[k];
    parts_out[o + 12u + k] = bitcast<u32>(prev);
    parts_out[o + k] = bitcast<u32>(xf[k]);
    pose_last[l + k] = xf[k];
  }
  pose_last[l + 12u] = 1.0;
  // flags: only OR in bit 1 (moved this frame) for a changed pose; the host's writeParts writes the flags each frame
  // (bit 0 static, bit 1 its own moved), so never clear a bit here
  if (moved) { parts_out[o + 31u] = parts_out[o + 31u] | 2u; }
}
