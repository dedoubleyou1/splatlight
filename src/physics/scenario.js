// Scenarios and pose helpers for the live physics (world.js). The Drop scenario mirrors
// experiments/physics_proto/scene_demo.py drop_poses().

export function qmul(a, b) {
  const [w1, x1, y1, z1] = a, [w2, x2, y2, z2] = b;
  return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2];
}

export function axisAngle(axis, ang) {
  const n = Math.hypot(...axis), s = Math.sin(ang / 2);
  return [Math.cos(ang / 2), ...axis.map(a => s * a / n)];
}

const bodyOf = (world, name) => world.meta.parts.findIndex(p => p.name === name);

// The Drop scenario. drop(world) is the fixed one (scene_demo.py drop_poses, the parity test): tall box lifted 0.4 m
// and tilted 25 deg about y and x (about its centre), short box thrown at the left wall at 3 m/s with spin.
// drop(world, seed) randomises it per seed (the Drop button passes a fresh one): tall box lifted 0.3-0.6 m, tilted
// 10-35 deg about a random axis, shifted up to 0.3 m in x / y; short box thrown at 2-4 m/s, usually at the left wall
// (+-30 deg), sometimes at the back or right wall, spinning 0-6 rad/s about a random axis. Starts are redrawn until
// both boxes are inside the room and apart. Returns the start actually used.
export function drop(world, seed) {
  world.reset();
  const bt = bodyOf(world, 'tall_box'), bs = bodyOf(world, 'short_box');
  const start = seed === undefined ? FIXED : randomStart(world, bt, bs, seed);
  if (bt >= 0) {
    const p = world.meta.parts[bt];
    world.setPose(bt, [p.com0[0] + start.tallXY[0], p.com0[1] + start.tallXY[1], p.com0[2] + start.lift],
      qmul(start.tiltQ ?? axisAngle(start.tiltAxis, start.tiltDeg * Math.PI / 180), p.rest_quat));
  }
  if (bs >= 0) {
    const p = world.meta.parts[bs];
    world.setPose(bs, [p.com0[0], p.com0[1], p.com0[2] + 0.02], p.rest_quat);
    world.vel.set(start.vel, 3 * bs);
    world.om.set(start.om, 3 * bs);
  }
  for (let b = 0; b < world.B; b++) if (world.invM[b] > 0) world.wake(b);
  return start;
}

// the fixed start (tiltQ exactly as scene_demo.py builds it; axis / angle for reports)
const FIXED = (() => {
  const q = qmul(axisAngle([0, 1, 0], 25 * Math.PI / 180), axisAngle([1, 0, 0], 25 * Math.PI / 180));
  const ang = 2 * Math.acos(q[0]), s = Math.sin(ang / 2);
  return { lift: 0.4, tiltQ: q, tiltAxis: [q[1] / s, q[2] / s, q[3] / s], tiltDeg: ang * 180 / Math.PI, tallXY: [0, 0],
    vel: [-3.0, 0.3, 0.8], om: [1.5, -1.0, 6.0], wall: 'left' };
})();

// mulberry32
function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ROOM = { lo: [-1.0, 0.0, 0.0], hi: [1.0, 2.01, 1.97] };

function randomStart(world, bt, bs, seed) {
  const r = rng(seed), u = (a, b) => a + (b - a) * r();
  const unit = () => { for (;;) { const v = [u(-1, 1), u(-1, 1), u(-1, 1)], l = Math.hypot(...v); if (l > 0.1 && l <= 1) return v.map(c => c / l); } };
  for (let tries = 0; tries < 200; tries++) {
    const pick = r(), wall = pick < 0.7 ? 'left' : pick < 0.85 ? 'back' : 'right';
    const heading = { left: Math.PI, back: Math.PI / 2, right: 0 }[wall] + u(-1, 1) * Math.PI / 6;
    const speed = u(2, 4), w = u(0, 6), axis = unit();
    const start = {
      lift: u(0.3, 0.6), tiltAxis: unit(), tiltDeg: u(10, 35), tallXY: [u(-0.3, 0.3), u(-0.3, 0.3)],
      vel: [speed * Math.cos(heading), speed * Math.sin(heading), u(0.3, 1.0)], om: axis.map(c => c * w), wall,
    };
    if (startOk(world, bt, bs, start)) return start;
  }
  return FIXED;
}

// both boxes (their face-aligned Gaussian-centre boxes + 2 cm) inside the room and not overlapping (separating axes)
function startOk(world, bt, bs, start) {
  const obb = (b, pos, q) => {
    const p = world.meta.parts[b], R = quatToMat(q), c = p.box_center;
    const ctr = [0, 1, 2].map(r => pos[r] + R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]);
    // world axes: R times each box axis (a row of box_axes, body frame)
    const ax = p.box_axes.map(a => [0, 1, 2].map(r => R[3 * r] * a[0] + R[3 * r + 1] * a[1] + R[3 * r + 2] * a[2]));
    return { ctr, h: p.box_half.map(v => v + 0.02), ax };
  };
  const pt = world.meta.parts[bt], ps = world.meta.parts[bs];
  const A = obb(bt, [pt.com0[0] + start.tallXY[0], pt.com0[1] + start.tallXY[1], pt.com0[2] + start.lift],
    qmul(axisAngle(start.tiltAxis, start.tiltDeg * Math.PI / 180), pt.rest_quat));
  const B = obb(bs, [ps.com0[0], ps.com0[1], ps.com0[2] + 0.02], ps.rest_quat);
  for (const O of [A, B]) for (let k = 0; k < 3; k++) {
    const ext = O.ax.reduce((s, a, i) => s + Math.abs(a[k]) * O.h[i], 0);
    if (O.ctr[k] - ext < ROOM.lo[k] - (k === 2 ? 0.025 : 0) || O.ctr[k] + ext > ROOM.hi[k]) return false;   // the short box starts 2 cm up
  }
  const d = [0, 1, 2].map(k => B.ctr[k] - A.ctr[k]);
  const axes = [...A.ax, ...B.ax];
  for (const a of A.ax) for (const b of B.ax) {
    const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], l = Math.hypot(...c);
    if (l > 1e-6) axes.push(c.map(v => v / l));
  }
  const dot = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  for (const n of axes) {
    const ra = A.ax.reduce((s, a, i) => s + Math.abs(dot(a, n)) * A.h[i], 0), rb = B.ax.reduce((s, b, i) => s + Math.abs(dot(b, n)) * B.h[i], 0);
    if (Math.abs(dot(d, n)) > ra + rb) return true;   // separated
  }
  return false;
}

function quatToMat([w, x, y, z]) {
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}

// physics pose from a world_from_rest transform given as a planar slider pose (partTransform: column-major 4x4 of a
// yaw about z plus a translation): R(q) = R_s R0, p = R_s com0 + t_s
export function poseFromYawXf(part, m, yawDeg) {
  const c = part.com0;
  const p = [0, 1, 2].map(r => m[r] * c[0] + m[4 + r] * c[1] + m[8 + r] * c[2] + m[12 + r]);
  return [p, qmul(axisAngle([0, 0, 1], yawDeg * Math.PI / 180), part.rest_quat)];
}

// loads physics.json + physics.bin of an asset folder (URL or fetch-able path); null when the asset has none
export async function loadPhysics(base, fetchFn = fetch) {
  const r = await fetchFn(`${base}/physics.json`).catch(() => null);
  if (!r?.ok) return null;
  const meta = await r.json();
  const bin = await fetchFn(`${base}/${meta.file}`).then(x => { if (!x.ok) throw new Error(`Could not load ${meta.file}`); return x.arrayBuffer(); });
  const n = meta.node_count * meta.stride, all = meta.dtype === '<f4' ? Float64Array.from(new Float32Array(bin)) : new Float64Array(bin);
  const sls = meta.slab_block?.stride ?? 4, es = meta.extent_block?.stride ?? 0;
  if (all.length !== n + (sls + es) * meta.node_count) throw new Error(`${meta.file}: wrong size`);
  const nodes = all.subarray(0, n);
  nodes.slab = all.subarray(n, n + sls * meta.node_count);   // per node: thin axis xyz, slab half-thickness, detail level tag (world.js)
  if (es) nodes.ext = all.subarray(n + sls * meta.node_count);   // per node: support along the bake's body-frame directions
  return { meta, nodes };
}
