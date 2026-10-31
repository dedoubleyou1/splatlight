// Physics worker: runs PhysicsWorld (world.js) off the render thread at a fixed dt of 1/60 s with an accumulator and
// posts the dynamic parts' world_from_rest rows after every step. A step can take far longer than 1/60 s (the
// prototype contact makes hundreds of thousands of node pairs while a box slides fast), so the clock drops time it
// cannot keep up with: the scene then runs in slow motion instead of freezing the page.
//
// in:  { type: 'init', base, budget } | { type: 'drop', seed } (no seed: the fixed Drop) | 'reset' | { type: 'poses', poses: [{ b, p, q }] } | { type: 'push', o, d, speed }
//      | { type: 'bodies', bodies: Float64Array(BS B) } (a hand-over from the GPU path: pos, quat, vel, om, asleep per body)
// out: { type: 'ready', meta } | 'none' | { type: 'state', xf: Float64Array(12 B), bodies: Float64Array(BS B), awake, ms, pairs, budget, incidents }
//      | { type: 'incident', record } (a box below the floor or out of the room: see afterStep)
import { PhysicsWorld } from './world.js';
import { drop, loadPhysics } from './scenario.js';

let world = null, running = false, acc = 0, last = 0, stepMs = 0;
const MAX_STEPS = 3, BS = 14;   // body record: pos 3, quat 4, vel 3, om 3, asleep 1
// the physics both paths run (the round-9 candidate, experiments/physics_proto/gpu_bench/GPU_PLAN.md) on a bake with
// materials; this solver stays Gauss-Seidel (cheaper on a CPU than the GPU form's Jacobi sweeps; validated 2026-10-30)
const CANDIDATE = { directedMargin: true, restitution: true, surfaceNormals: true, materials: true };

// Incident diagnostics: the last HIST steps (time, tick gap, steps in that tick, pairs, patches, swept hits, the
// dynamic bodies' pose and velocity) and UI messages. A dynamic body whose face-box centre goes below the floor or
// leaves the room through a wall or the ceiling is an incident: the record goes to the main thread (console +
// gsDebug.physics.lastIncident), then the body is put back at its last good pose, at rest, and counted. A body out
// through the open front (y < 0) is not: it falls freely and is parked (at rest, asleep) once below KILL_Z. It stays
// out (gone) while below the floor's level, so a fall that drifts back under the room is not an incident either.
const HIST = 120, KILL_Z = -2.0;
const hist = [], events = [];
let gone = null;
let lastGood = null, incidents = 0, tickGap = 0, tickSteps = 0;

function centreOf(b) {
  const p = world.meta.parts[b], c = p.box_center ?? [0, 0, 0], q = world.quat.subarray(4 * b, 4 * b + 4), [w, x, y, z] = q;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  return [0, 1, 2].map(r => world.pos[3 * b + r] + R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]);
}

function afterStep(t) {
  const dyn = [];
  for (let b = 0; b < world.B; b++) if (world.invM[b] > 0) dyn.push(b);
  hist.push({ t, gap: tickGap, steps: tickSteps, dt: world.dt, nsub: world.nsub, pairs: world.last.pairs, patches: world.last.patches,
    swept: world.stats[4], budget: world.budget > 0 ? { ...world.budgetStats } : null,
    bodies: dyn.map(b => ({ name: world.meta.parts[b].name, pos: Array.from(world.pos.subarray(3 * b, 3 * b + 3)), quat: Array.from(world.quat.subarray(4 * b, 4 * b + 4)),
      vel: Array.from(world.vel.subarray(3 * b, 3 * b + 3)), om: Array.from(world.om.subarray(3 * b, 3 * b + 3)), asleep: !!world.asleep[b] })) });
  if (hist.length > HIST) hist.shift();
  if (gone?.length !== world.B) gone = new Uint8Array(world.B);
  for (const b of dyn) {   // out through the open front, and still below the floor's level since
    const c = centreOf(b);
    gone[b] = c.every(Number.isFinite) && (c[1] < 0 || (gone[b] && c[2] < 0)) ? 1 : 0;
    if (gone[b] && c[2] < KILL_Z && !world.asleep[b]) { world.vel.fill(0, 3 * b, 3 * b + 3); world.om.fill(0, 3 * b, 3 * b + 3); world.asleep[b] = 1; }
  }
  const bad = dyn.filter(b => { const c = centreOf(b); return !gone[b] && (!c.every(Number.isFinite) || c[2] < 0 || c[2] > 2.0 || Math.abs(c[0]) > 1.05 || c[1] > 2.05); });
  if (!bad.length) {
    lastGood = { pos: Float64Array.from(world.pos), quat: Float64Array.from(world.quat) };
    return;
  }
  incidents++;
  const record = { at: t, bodies: bad.map(b => world.meta.parts[b].name), centres: bad.map(centreOf), steps: hist.slice(), events: events.slice(), incidents };
  postMessage({ type: 'incident', record });
  for (const b of bad) {
    if (lastGood) { world.pos.set(lastGood.pos.subarray(3 * b, 3 * b + 3), 3 * b); world.quat.set(lastGood.quat.subarray(4 * b, 4 * b + 4), 4 * b); }
    world.vel.fill(0, 3 * b, 3 * b + 3); world.om.fill(0, 3 * b, 3 * b + 3);
    world.cValid.fill(0);
  }
}

function post() {
  const xf = new Float64Array(12 * world.B), bodies = new Float64Array(BS * world.B);
  for (let b = 0; b < world.B; b++) {
    world.worldFromRest(b, xf, 12 * b);
    const o = BS * b;
    bodies.set(world.pos.subarray(3 * b, 3 * b + 3), o); bodies.set(world.quat.subarray(4 * b, 4 * b + 4), o + 3);
    bodies.set(world.vel.subarray(3 * b, 3 * b + 3), o + 7); bodies.set(world.om.subarray(3 * b, 3 * b + 3), o + 10); bodies[o + 13] = world.asleep[b];
  }
  postMessage({ type: 'state', xf, bodies, awake: world.anyAwake(), ms: stepMs, pairs: world.last.pairs, budget: { ...world.budgetStats }, incidents }, [xf.buffer, bodies.buffer]);
}

function start() {
  if (running) return;
  running = true; acc = 0; last = performance.now();
  setTimeout(tick, 0);
}

function tick() {
  const t = performance.now();
  tickGap = t - last;
  acc += Math.min(t - last, 100) / 1000;
  last = t;
  const dt = world.dt, cap = stepMs > 1000 * dt ? 1 : MAX_STEPS;
  let n = 0;
  tickSteps = Math.min(cap, Math.floor(acc / dt));
  while (acc >= dt - 1e-9 && n < cap) {   // 1e-9: float rounding of the accumulator never strands a due step
    const t0 = performance.now();
    world.step();
    stepMs = performance.now() - t0;
    acc -= dt; n++;
    afterStep(t0);
  }
  if (n === cap) acc = Math.min(acc, dt);   // behind real time: drop the rest (slow motion)
  if (n) post();
  running = world.anyAwake();
  if (running) setTimeout(tick, Math.max(1, 1000 * (dt - acc) - (performance.now() - t)));
  else { last = 0; post(); }
}

self.onmessage = async e => {
  const m = e.data;
  if (m.type === 'init') {
    // the scene's materials bake when there is one (<asset>_mat_extsb: the same parts and Gaussians), else its own
    const bakes = m.bake ? [m.bake] : [`${m.base}_mat_extsb`, m.base];
    let ph = null;
    for (const b of bakes) { ph = await loadPhysics(b).catch(() => null); if (ph) break; }
    if (!ph) return postMessage({ type: 'none' });
    const cand = (ph.meta.slab_block?.stride ?? 4) > 10 ? CANDIDATE : {};
    world = new PhysicsWorld(ph.meta, ph.nodes, { budget: m.budget ?? 0, ...cand, ...(m.worldOpts ?? {}) });   // worldOpts: tests (gpuForm)
    postMessage({ type: 'selfcheck', report: world.selfCheck() });
    postMessage({ type: 'ready', meta: ph.meta });
    return post();
  }
  if (!world) return;
  events.push({ t: performance.now(), type: m.type, ...(m.seed !== undefined ? { seed: m.seed } : {}), ...(m.type === 'push' ? { o: m.o, d: m.d, speed: m.speed } : {}) });
  if (events.length > 40) events.shift();
  if (m.type === 'drop') { drop(world, m.seed); start(); }
  else if (m.type === 'reset') { world.reset(); post(); }
  else if (m.type === 'poses') {   // placed parts (the sliders' poses), at rest and asleep
    world.reset();
    for (const { b, p, q } of m.poses) world.setPose(b, p, q);
    post();
  } else if (m.type === 'bodies') {   // a hand-over: the bodies as another path left them, stepping on from there
    world.reset();
    for (let b = 0; b < world.B; b++) {
      const o = BS * b;
      world.setPose(b, m.bodies.subarray(o, o + 3), m.bodies.subarray(o + 3, o + 7));
      world.vel.set(m.bodies.subarray(o + 7, o + 10), 3 * b); world.om.set(m.bodies.subarray(o + 10, o + 13), 3 * b);
      world.asleep[b] = m.bodies[o + 13] ? 1 : 0;
    }
    if (world.anyAwake()) start(); else post();
  } else if (m.type === 'push') {   // a click: velocity change along the view ray at the first box it hits
    const hit = world.pick(m.o, m.d);
    if (hit) { world.push(hit.body, hit.point, m.d, m.speed); start(); }
  }
};
