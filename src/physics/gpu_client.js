// GPU physics in the browser (?physgpu=1 forces it; the default ?physgpu=auto picks it or the worker, auto_client.js): the whole step of world.js's GPU
// form (gpu_kernels.wgsl + gpu_step.wgsl, as experiments/physics_proto/gpu_bench/gpu_world.py drives it: descent,
// overlap and patches, pjacobi sweeps, restitution, anchors, energy limiter, sleep) on the renderer's own GPUDevice,
// then the pose handoff (gpu_pose.wgsl) straight into the renderer's part records: no readback in the pose path.
// One small async readback at a time (the bodies' poses and sleep flags, the pair count) feeds the HUD, clicks, the
// moved mask main.js computes from xf (one frame late) and the frame loop's wake-ups.
//
// Same interface as PhysicsClient (client.js): xf, bodyOf, meta, ms, pairs, awake, budget, incidents, onState, drop,
// reset, place, attachClicks; plus writePoses(partBuf, enabled), called by main.js writeParts once per frame after it
// uploads the parts. Incidents (a box out of the room) follow the worker's rule on each readback (checkIncidents).
import { PhysicsWorld } from './world.js';
import { drop, loadPhysics } from './scenario.js';
import { PhysicsClient } from './client.js';

const BODY = 28, BX = 53, NS = 16, LEVEL_LOG = 64, NCOL = 8, MAX_STEPS = 3;
// the round-9 candidate (world.js options; experiments/physics_proto/gpu_bench/GPU_PLAN.md)
export const GPU_FORM = { gpuForm: true, surfaceNormals: true, materials: true, directedMargin: true, restitution: true };
const ENTRIES = ['bp_build', 'detect_prep', 'detect_ind2', 'pair_args', 'patch_insert', 'patch_scan', 'patch_fill', 'body_scan', 'body_fill',
  'patch_bp_fill', 'overlap_xf', 'patch_step', 'snap_bp', 'energy_mark', 'integrate2', 'body_apply_pj', 'velocity', 'damp_patch',
  'damp_body', 'energy_limit', 'sleep_step', 'step_record', 'damp_colors', 'stick_body', 'anchor_load', 'anchor_update', 'anchor_store', 'snap_patch', 'pose_out'];

// per compute entry point the bindings it statically uses (module variables reached through its calls): an auto layout
// holds exactly these (gpu_bench/wgsl_lint.py entry_bindings)
function entryBindings(src) {
  const code = src.replace(/\/\/[^\n]*/g, '');
  const vars = new Map();
  for (const m of code.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<[^>]*>)?\s+(\w+)/g)) vars.set(m[3], Number(m[2]));
  const fns = new Map();
  for (const m of code.matchAll(/\bfn\s+(\w+)\s*\(/g)) {
    let j = code.indexOf('{', m.index + m[0].length), depth = 0;
    for (; ; j++) { if (code[j] === '{') depth++; else if (code[j] === '}' && --depth === 0) break; }
    fns.set(m[1], code.slice(m.index, j + 1));
  }
  const out = new Map();
  for (const m of code.matchAll(/@compute[^\n]*\n\s*fn (\w+)/g)) {
    const seen = new Set(), todo = [m[1]], used = new Set();
    while (todo.length) {
      const f = todo.pop();
      if (seen.has(f)) continue;
      seen.add(f);
      for (const [name] of fns.get(f).matchAll(/\b\w+\b/g)) {
        if (vars.has(name)) used.add(vars.get(name));
        if (fns.has(name) && !seen.has(name)) todo.push(name);
      }
    }
    out.set(m[1], [...used].sort((a, b) => a - b));
  }
  return out;
}

// the bindings declared as storage buffers
function storageBindings(src) {
  const code = src.replace(/\/\/[^\n]*/g, '');
  return new Set([...code.matchAll(/@group\(\d+\)\s*@binding\((\d+)\)\s*var<storage/g)].map(m => Number(m[1])));
}

const KILL_Z = -2.0;   // out through the open front: parked below this (worker.js afterStep)

// a body's face-box centre in world space (worker.js centreOf)
function centreOf(W, b) {
  const p = W.meta.parts[b], c = p.box_center ?? [0, 0, 0], [w, x, y, z] = W.quat.subarray(4 * b, 4 * b + 4);
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  return [0, 1, 2].map(r => W.pos[3 * b + r] + R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]);
}

// depth (levels below; leaf 0) of every node (children after their parents: one reverse sweep)
function treeDepths(N) {
  const n = N.length / NS, depth = new Int32Array(n);
  for (let k = n - 1; k >= 0; k--) { const l = N[k * NS + 13]; if (l >= 0) depth[k] = 1 + Math.max(depth[l], depth[N[k * NS + 14]]); }
  return depth;
}

export class PhysicsGPU {
  // resolves to null when the bake is missing or the kernels fail to build (main.js then has no physics)
  // base: the physics bake folder (a materials bake of the scene's asset; same parts and Gaussians)
  // Never blocks the page: any missing bake, device limit, pipeline or validation error returns null with a warning
  // (main.js then starts the CPU worker instead).
  static async start(base, device, worldOpts = GPU_FORM) {
    try { return await PhysicsGPU.tryStart(base, device, worldOpts); } catch (e) { console.warn('GPU physics failed to start, using the CPU worker:', e?.message ?? e); return null; }
  }

  static async tryStart(base, device, worldOpts) {
    const url = new URL(base, location.href).href;
    const ph = await loadPhysics(url).catch(() => null);
    if (!ph) return null;
    const opts = { ...worldOpts };
    if ((ph.meta.slab_block?.stride ?? 4) <= 10) opts.materials = false;   // older bakes: no material slots
    const world = new PhysicsWorld(ph.meta, ph.nodes, opts);
    console.info('physics self-check (GPU form, CPU shadow world):', world.selfCheck().ok ? 'OK' : 'see report', '- bake', base);
    world.reset();
    const text = n => fetch(new URL(n, import.meta.url)).then(r => { if (!r.ok) throw new Error(`Could not load ${n}`); return r.text(); });
    const src = (await Promise.all(['gpu_kernels.wgsl', 'gpu_step.wgsl', 'gpu_pose.wgsl'].map(text))).join('\n');
    // the kernels' storage buffers per stage against the device's limit (browsers: 8 by default, adapters up to 10+)
    const uses = entryBindings(src), storage = storageBindings(src);
    const need = Math.max(...[...uses.values()].map(u => u.filter(x => storage.has(x)).length));
    if (need > device.limits.maxStorageBuffersPerShaderStage) {
      console.warn(`GPU physics needs ${need} storage buffers per shader stage, the device allows ${device.limits.maxStorageBuffersPerShaderStage}: using the CPU worker`);
      return null;
    }
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    let phys = null, thrown = null;
    try {
      const module = device.createShaderModule({ code: src, label: 'physics' });
      const pipes = {};
      for (const e of ENTRIES) pipes[e] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: e }, label: `physics ${e}` });
      phys = new PhysicsGPU(device, world, pipes, uses);   // buffers and bind groups inside the error scopes too
    } catch (e) { thrown = e; }
    const errV = await device.popErrorScope(), errM = await device.popErrorScope(), err = errV ?? errM;   // both scopes popped
    if (err || thrown) { console.warn('GPU physics: kernels or buffers failed, using the CPU worker:', (err ?? thrown)?.message ?? err ?? thrown); return null; }
    return phys;
  }

  constructor(device, world, pipes, uses) {
    Object.assign(this, { device, world, pipes, uses, meta: world.meta });
    this.bytes = 0;
    this.xf = null; this.ms = 0; this.pairs = 0; this.awake = false; this.budget = null; this.incidents = 0; this.lastIncident = null;
    this.onState = () => {};
    this.bodyOf = new Map(world.meta.parts.filter(p => !p.static).map(p => [p.asset_index, world.meta.parts.indexOf(p)]));
    const W = world, B = W.B, d = device, U = GPUBufferUsage;
    // every buffer and query set it makes, for release() (auto_client.js frees the GPU path while the CPU runs)
    const owned = this.owned = [], mk = desc => { const b = d.createBuffer(desc); owned.push(b); this.bytes += desc.size; return b; };
    const mkq = desc => { const q = d.createQuerySet(desc); owned.push(q); return q; };
    const buf = (size, usage = 0, label = 'physics') => mk({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | usage, label });
    const data = (arr, usage = 0) => { const b = buf(arr.byteLength, usage); d.queue.writeBuffer(b, 0, arr); return b; };
    // capacities: room for the 8-box piles' peaks (35k node pairs, 650 patches) several times over
    const pairCap = 1 << 18, cap = 1 << 19, T = 1 << 13, TA = 2 * T, nbpCap = Math.max(1, B * (B - 1) / 2);
    const depth = treeDepths(W.N), roots = Array.from(W.root);
    this.bound2 = Math.floor((2 * Math.max(...roots.map(r => depth[r])) + 2) / 2) + 2;
    Object.assign(this, { B, T, nbpCap, cap });
    const h = W.dt / W.nsub, sls = W.SLS;
    this.anchors = !!W.frictionAnchors;
    this.nsub = W.nsub; this.k = W.iters; this.dt = W.dt;
    // world settings (gpu_step.wgsl StepW) and per-dispatch params (gpu_kernels.wgsl Params)
    const sw = new Float32Array(32), swi = new Uint32Array(sw.buffer);
    sw.set([W.dt, W.mu, W.margin, Number.isFinite(W.marginCap) ? W.marginCap : 1e30, W.wakeV, W.sleepV, W.sleepT, W.sweptQ, W.sweptMin,
      Math.cos(W.snapDeg * Math.PI / 180), ...W.grav, W.energyLimit ? 1 : 0, W.islandSleep ? 1 : 0, W.snapDeg > 0 ? 1 : 0]);
    swi[16] = B; swi[17] = cap; swi[18] = nbpCap; swi[19] = sls > 4 ? 1 : 0; swi[20] = 3;   // colour damping
    sw[21] = W.stickV || 0; sw[22] = W.sleepDepth || 0; swi[23] = this.anchors ? 1 : 0;
    sw[24] = W.muS !== undefined && W.muK !== undefined ? W.muS : -1; sw[25] = W.muK || 0; sw[26] = W.stribeckV || 0.03; sw[27] = W.sleepDisp || 0;
    swi[28] = W.surfaceNormals ? 1 : 0; swi[29] = W.materials ? 1 : 0; swi[30] = W.restitution ? 1 : 0; swi[31] = W.sleepNet ? 1 : 0;
    const need = W.refineNeed && W.meta.slab_block?.fields?.[11] === 'extent_err' ? W.refineNeed : 0;
    const xflags = (W.directedMargin ? 16 : 0) | (W.oneSided === 'soft' ? 32 : 0) | (need > 0 ? 64 : 0);
    this.paramList = [];   // every dispatch's params, rewritten when the step length changes (setDt)
    const params = (count, flags = 0) => {
      const u = new Uint32Array(16), f = new Float32Array(u.buffer);
      u[0] = count; u[1] = B; u[2] = sls; u[3] = flags | xflags;
      f[4] = h; f[5] = W.kappa; f[6] = W.relax; f[7] = W.maxdep; f[8] = W.alpha / (h * h); f[9] = W.grav[2];
      f[10] = W.damp; f[11] = W.qcut; f[12] = W.patch; f[13] = 1.5 * W.dt; f[14] = W.N.length;   // the slab block's offset in nodes
      f[15] = need;   // refineNeed tolerance (m)
      const b = mk({ size: 64, usage: U.UNIFORM | U.COPY_DST, label: 'physics params' }); d.queue.writeBuffer(b, 0, u);
      this.paramList.push({ b, u }); return b;
    };
    const swb = mk({ size: 128, usage: U.UNIFORM | U.COPY_DST, label: 'physics world' }); d.queue.writeBuffer(swb, 0, sw);
    this.swb = swb; this.sw = sw; this.curDt = W.dt;
    // pose handoff: per body (asset part, handed off) and R0 rows with com0
    const pm = new Float32Array(16 * B);
    for (let b = 0; b < B; b++) {
      const p = W.meta.parts[b];
      if (p.static) continue;
      pm[16 * b] = p.asset_index; pm[16 * b + 1] = 1;
      for (let r = 0; r < 3; r++) { pm.set(p.R0.slice(3 * r, 3 * r + 3), 16 * b + 4 * (r + 1)); pm[16 * b + 4 * (r + 1) + 3] = p.com0[r]; }
    }
    this.bodies = data(new Float32Array(B * BODY)); this.bext = data(new Float32Array(B * BX));
    const leaves = buf(16 * pairCap), bpInfo = buf(16 * nbpCap);
    const by = {
      0: data(Float32Array.from([...W.N, ...W.SL])), 1: this.bodies, 17: bpInfo, 20: leaves, 22: buf(32 * cap),
      23: buf(4 * (8 + LEVEL_LOG)), 24: buf(16, U.INDIRECT), 27: leaves, 28: buf(16), 29: buf(48, U.INDIRECT), 30: buf(80 * pairCap),
      31: buf(4 * T), 32: buf(4 * T), 33: buf(4 * pairCap), 34: buf(4 * B), 35: buf(4 * (T + 1)), 36: buf(4 * pairCap), 37: buf(8 * T),
      38: buf(4 * (B + 1)), 39: buf(8 * T), 40: buf(4 * T), 10: buf(64 * T), 13: buf(64 * T), 43: buf(16 * nbpCap), 44: this.bext,
      46: buf(4 * T), 47: swb, 48: buf(16), 49: buf(4 * 9 * B), 50: data(new Uint32Array(nbpCap)), 52: buf(4 * nbpCap), 53: buf(16),
      54: buf(4 * nbpCap), 55: buf(4 * 6 * NCOL, U.INDIRECT), 56: buf(64 * T), 58: buf(4 * T), 63: buf(64 * T),
      64: data(pm), 66: buf(4 * 16 * B),
    };
    by[8] = by[35]; by[9] = by[36]; by[12] = by[37]; by[14] = by[38]; by[15] = by[39]; by[51] = by[17];
    this.atab = [[buf(4 * TA), buf(32 * TA)], [buf(4 * TA), buf(32 * TA)]];   // ping-pong anchor tables (keys, data)
    this.by = by;
    this.pr = { det: params(cap, 1 | (sls > 4 ? 2 : 0)), plain: params(0), reduce: params(0, 4), first: params(0, 8) };
    for (let c = 0; c < NCOL; c++) this.pr[`col${c}`] = params(c);
    const group = (entry, pkey = 'plain', parity = 0, extra = {}) => {
      const old = this.atab[parity], neu = this.atab[1 - parity];
      const res = { ...by, 2: this.pr[pkey], 59: old[0], 60: old[1], 61: neu[0], 62: neu[1], ...extra };
      return d.createBindGroup({ layout: pipes[entry].getBindGroupLayout(0), label: `physics ${entry}`,
        entries: this.uses.get(entry).map(x => ({ binding: x, resource: { buffer: res[x] } })) });
    };
    this.group = group;
    const G = {};
    for (const e of ENTRIES) if (e !== 'pose_out') G[e] = group(e);
    G.detect_ind2 = group('detect_ind2', 'det'); G.detect_prep = group('detect_prep', 'det');
    G.patch_step_reduce = group('patch_step', 'reduce'); G.patch_step_first = group('patch_step', 'first');
    for (const par of [0, 1]) { G[`anchor_load${par}`] = group('anchor_load', 'plain', par); G[`anchor_store${par}`] = group('anchor_store', 'plain', par); }
    for (let c = 0; c < NCOL; c++) G[`damp_patch${c}`] = group('damp_patch', `col${c}`);
    this.G = G;
    this.poseGroup = null; this.posePart = null;
    // readback: bodies (poses, velocities, sleep flags) and the solve's pair count
    this.rbBytes = 4 * BODY * B + 16;
    this.staging = mk({ size: this.rbBytes, usage: U.MAP_READ | U.COPY_DST, label: 'physics readback' });
    this.pending = false; this.stepCount = 0; this.acc = 0; this.last = 0; this.running = false;
    // the moved mask main.js gets back (writePoses): asset parts of awake bodies, held one frame past sleep
    this.dynMask = 0;
    for (const [ai] of this.bodyOf) this.dynMask |= 1 << ai;
    this.awakeMask = 0; this.prevMask = 0;
    // debug toggles (gsDebug.physics): maskMode 'awake' (the parts of awake bodies, every frame) or 'readback' (the
    // parts whose read-back pose changed, on the frame after each readback: the worker's cadence); maxStepsPerFrame (an
    // over-budget frame, longer than 1.5 dt, always gets 1)
    this.maskMode = 'awake'; this.maxStepsPerFrame = MAX_STEPS; this.readbackMoved = 0;
    // 'awake' counts only bodies that move visibly (2026-10-31): over a frame, |v| dt + |w| reach dt above moveTol (m;
    // 0.2 mm), from the last readback's velocities, or a read-back pose that has drifted more than moveTol (centre, or
    // rotation angle x reach) from the pose last marked. Awake but nearly still bodies (a settling pile) stop forcing
    // the lighting to re-splat their parts (glass marbles) every frame, and a slow creep is still caught
    this.moveTol = 2e-4; this.markPos = Float64Array.from(W.pos); this.markQuat = Float64Array.from(W.quat);
    this.reach = Float64Array.from({ length: B }, (_, b) => W.N[W.root[b] * NS + 10] * W.bs[b]);
    this.maxDt = 1 / 30;   // the longest step a slow frame takes (gsDebug.physics.maxDt; 1/60 keeps the old slow motion)
    // slow frames (2026-10-31): as many maxDt steps as cover the time owed, while their GPU time stays within
    // costFrac of the frame (stepGpuMs: measured with timestamp queries where the device has them, else this seed);
    // rtf: simulated over wall time (EMA), for auto_client.js
    this.costFrac = 0.1; this.stepGpuMs = 1; this.rtf = 1; this.simE = 0; this.wallE = 0;
    if (d.features?.has?.('timestamp-query')) {
      this.tq = mkq({ type: 'timestamp', count: 2 });
      this.tqRes = mk({ size: 16, usage: U.QUERY_RESOLVE | U.COPY_SRC });
      this.tqRead = mk({ size: 16, usage: U.MAP_READ | U.COPY_DST });
      this.tqPending = false;
    }
    this.upload();
  }

  // frees every GPU buffer and query set (auto_client.js, when the CPU path takes over); the object is unusable after
  release() {
    this.running = false; this.released = true;
    for (const o of this.owned) o.destroy();
    this.owned = [];
  }

  // the shadow world's state (all bodies) to the GPU, per-body extras as gpu_world.py: a placement, Drop or reset
  upload() {
    const W = this.world, B = this.B, bodies = new Float32Array(B * BODY), bext = new Float32Array(B * BX);
    for (let b = 0; b < B; b++) {
      const o = b * BODY, x = b * BX;
      bodies.set(W.pos.subarray(3 * b, 3 * b + 3), o); bodies[o + 3] = W.bs[b]; bodies.set(W.quat.subarray(4 * b, 4 * b + 4), o + 4);
      bodies.set(W.vel.subarray(3 * b, 3 * b + 3), o + 8); bodies[o + 11] = W.invM[b]; bodies.set(W.om.subarray(3 * b, 3 * b + 3), o + 12);
      bodies.set(W.Ib.subarray(3 * b, 3 * b + 3), o + 16); bodies[o + 19] = W.asleep[b];
      bodies.set(W.pos.subarray(3 * b, 3 * b + 3), o + 20); bodies.set(W.quat.subarray(4 * b, 4 * b + 4), o + 24);
      bext.set(W.baxes.subarray(9 * b, 9 * b + 9), x); bext.set(W.loff.subarray(12 * b, 12 * b + 12), x + 9);
      bext[x + 30] = -1e30; bext[x + 33] = W.still[b]; bext[x + 34] = W.sleepL[b]; bext[x + 36] = W.root[b];
    }
    this.device.queue.writeBuffer(this.bodies, 0, bodies);
    this.device.queue.writeBuffer(this.bext, 0, bext);
    this.xf = new Float64Array(12 * B);
    for (let b = 0; b < B; b++) W.worldFromRest(b, this.xf, 12 * b);
    this.awake = W.anyAwake();
    this.lastGood = null;   // incidents: the new scene's first good readback sets it
    this.dirty = true;   // the next frame writes these poses (pose_out) even with nothing stepping
  }

  // mask: the asset parts set in motion (a Drop: all; a push: the pushed one; the others keep their last read-back state)
  start(mask = this.dynMask) { this.running = true; this.awake = true; this.awakeMask |= mask; this.acc = 0; this.last = performance.now(); this.startStep = this.stepCount; this.onState(); }

  // the same scenarios as the worker (scenario.js drop, world.js reset / setPose / push) on the CPU shadow world, then
  // uploaded; the shadow world never steps
  drop(seed = Math.floor(Math.random() * 2 ** 32)) {
    console.info(`physics Drop (GPU), seed ${seed} (gsDebug.physics.drop(${seed}) replays it)`);
    drop(this.world, seed); this.upload(); this.awakeMask = 0; this.start();
  }
  reset() { this.world.reset(); this.upload(); this.running = false; this.onState(); }
  place(sliderXf) {
    PhysicsClient.prototype.place.call({ meta: this.meta, bodyOf: this.bodyOf, worker: { postMessage: m => {
      this.world.reset();
      for (const { b, p, q } of m.poses) this.world.setPose(b, p, q);
      this.upload(); this.running = false;
    } } }, sliderXf);
  }
  attachClicks(canvas, camera, enabled, speed = 2) {
    PhysicsClient.prototype.attachClicks.call({ worker: { postMessage: m => this.push(m.o, m.d, m.speed) } }, canvas, camera, enabled, speed);
  }
  // a click: the shadow world takes the last read-back poses, picks and pushes, and the pushed body's velocities and
  // wake flag go up (the GPU state otherwise stays as it is)
  push(o, dir, speed) {
    const W = this.world, s = this.lastState;
    if (s) for (let b = 0; b < this.B; b++) {
      W.pos.set(s.subarray(b * BODY, b * BODY + 3), 3 * b); W.quat.set(s.subarray(b * BODY + 4, b * BODY + 8), 4 * b);
      W.vel.set(s.subarray(b * BODY + 8, b * BODY + 11), 3 * b); W.om.set(s.subarray(b * BODY + 12, b * BODY + 15), 3 * b);
    }
    const hit = W.pick(o, dir);
    if (!hit) return;
    W.push(hit.body, hit.point, dir, speed);
    const b = hit.body, v = new Float32Array([...W.vel.subarray(3 * b, 3 * b + 3), W.invM[b], ...W.om.subarray(3 * b, 3 * b + 3), 0, 0, 0, 0, 0]);
    this.device.queue.writeBuffer(this.bodies, 4 * (b * BODY + 8), v);   // vel, invMass, om, pad, Ib (rewritten below), asleep 0
    this.device.queue.writeBuffer(this.bodies, 4 * (b * BODY + 16), new Float32Array([...W.Ib.subarray(3 * b, 3 * b + 3), 0]));
    this.device.queue.writeBuffer(this.bext, 4 * (b * BX + 33), new Float32Array([0]));   // not still
    this.start(1 << this.meta.parts[b].asset_index);
  }

  // one step's passes (gpu_world.py record_step)
  recordStep(enc) {
    const par = this.stepCount % 2, by = this.by, G = this.G, P = this.pipes;
    const disp = (cp, e, n, ind, gk) => {
      cp.setPipeline(P[e]); cp.setBindGroup(0, G[gk ?? e]);
      if (ind) cp.dispatchWorkgroupsIndirect(ind[0], ind[1]); else cp.dispatchWorkgroups(...n);
    };
    const wg = n => [Math.min(Math.max(1, Math.ceil(n / 64)), 65535), 1, 1];
    const sargs = by[29], dargs = by[24], B = this.B;
    for (const x of [31, 32, 34]) enc.clearBuffer(by[x]);
    if (this.anchors) enc.clearBuffer(this.atab[1 - par][0]);
    let cp = enc.beginComputePass({ label: 'physics detect' });
    disp(cp, 'bp_build', [1, 1, 1]);
    for (let l = 0; l < this.bound2; l++) { disp(cp, 'detect_prep', [1, 1, 1]); disp(cp, 'detect_ind2', null, [dargs, 0]); }
    cp.end();
    enc.copyBufferToBuffer(by[23], 4, by[28], 0, 4);   // detected pairs -> the solve's pair count
    cp = enc.beginComputePass({ label: 'physics solve' });
    disp(cp, 'pair_args', [1, 1, 1]); disp(cp, 'patch_insert', null, [sargs, 0]); disp(cp, 'patch_scan', [1, 1, 1]);
    disp(cp, 'patch_fill', null, [sargs, 0]); disp(cp, 'body_scan', [1, 1, 1]); disp(cp, 'body_fill', null, [sargs, 32]);
    disp(cp, 'patch_bp_fill', wg(this.T));
    if (this.anchors) disp(cp, 'anchor_load', null, [sargs, 32], `anchor_load${par}`);
    disp(cp, 'overlap_xf', null, [sargs, 0]); disp(cp, 'patch_step', null, [sargs, 16], 'patch_step_reduce');
    disp(cp, 'snap_bp', wg(this.nbpCap)); disp(cp, 'snap_patch', null, [sargs, 32]); disp(cp, 'energy_mark', wg(B));
    disp(cp, 'damp_colors', [1, 1, 1]);
    for (let s = 0; s < this.nsub; s++) {
      disp(cp, 'integrate2', wg(B));
      for (let it = 0; it < this.k; it++) {
        disp(cp, 'overlap_xf', null, [sargs, 0]);
        disp(cp, 'patch_step', null, [sargs, 16], it === 0 ? 'patch_step_first' : 'patch_step');
        disp(cp, 'body_apply_pj', wg(B));
      }
      if (this.anchors) disp(cp, 'anchor_update', null, [sargs, 32]);
      disp(cp, 'velocity', wg(B));
      for (let c = 0; c < NCOL; c++) { disp(cp, 'damp_patch', null, [by[55], 24 * c], `damp_patch${c}`); disp(cp, 'damp_body', null, [by[55], 24 * c + 12]); }
      if (this.world.stickV > 0) disp(cp, 'stick_body', wg(B));
    }
    disp(cp, 'energy_limit', wg(B));
    if (this.anchors) disp(cp, 'anchor_store', null, [sargs, 32], `anchor_store${par}`);
    disp(cp, 'sleep_step', [1, 1, 1]); disp(cp, 'step_record', [1, 1, 1]);
    cp.end();
    this.stepCount++;
  }

  // the worker's incident rule (worker.js afterStep), on each readback: a dynamic body whose face-box centre is below
  // the floor or out of the room (not through the open front: that one falls freely, stays out while below the floor's
  // level, and is parked at rest, asleep, once below KILL_Z) is put back at its last good read-back pose,
  // at rest, and only its rows go up; a record goes to the console and gsDebug.physics.lastIncident. No extra GPU sync:
  // the readback already in flight is what it sees, so a body may run a few frames further before it is put back
  checkIncidents(a) {
    const W = this.world, B = this.B, bad = [], centres = [];
    if (this.gone?.length !== B) this.gone = new Uint8Array(B);
    for (let b = 0; b < B; b++) {
      if (W.invM[b] === 0) continue;
      const c = centreOf(W, b);
      this.gone[b] = c.every(Number.isFinite) && (c[1] < 0 || (this.gone[b] && c[2] < 0)) ? 1 : 0;
      if (this.gone[b]) {   // out through the open front, and still below the floor's level since
        const o = b * BODY;
        if (c[2] < KILL_Z && a[o + 19] === 0) {
          // pose kept; vel 0, om 0, asleep; prev pos and quat = the pose
          const row = Float32Array.from(a.subarray(o, o + BODY));
          row.fill(0, 8, 11); row.fill(0, 12, 15); row[19] = 1; row.copyWithin(20, 0, 3); row.copyWithin(24, 4, 8);
          this.device.queue.writeBuffer(this.bodies, 4 * o, row);
          W.vel.fill(0, 3 * b, 3 * b + 3); W.om.fill(0, 3 * b, 3 * b + 3); W.asleep[b] = 1;
        }
        continue;
      }
      if (!c.every(Number.isFinite) || c[2] < 0 || c[2] > 2.0 || Math.abs(c[0]) > 1.05 || c[1] > 2.05) { bad.push(b); centres.push(c); }
    }
    if (!bad.length) { this.lastGood = Float32Array.from(a); return; }
    this.incidents++;
    this.lastIncident = { at: performance.now(), bodies: bad.map(b => W.meta.parts[b].name), centres, incidents: this.incidents, path: 'GPU' };
    console.warn(`physics incident ${this.incidents}: ${this.lastIncident.bodies.join(', ')} below the floor / out of the room; `
      + 'put back at its last good pose (gsDebug.physics.lastIncident)', this.lastIncident);
    for (const b of bad) {
      const g = this.lastGood ?? a, o = b * BODY;
      const pos = g.subarray(o, o + 3), quat = g.subarray(o + 4, o + 8);
      // pos, quat; vel 0, invMass, om 0; ...; prev pos and quat = the pose (no motion across the substep)
      const row = Float32Array.from(a.subarray(o, o + BODY));
      row.set(pos, 0); row.set(quat, 4); row.fill(0, 8, 11); row.fill(0, 12, 15); row.set(pos, 20); row.set(quat, 24);
      this.device.queue.writeBuffer(this.bodies, 4 * o, row);
      W.pos.set(pos, 3 * b); W.quat.set(quat, 4 * b); W.vel.fill(0, 3 * b, 3 * b + 3); W.om.fill(0, 3 * b, 3 * b + 3);
    }
  }

  // the step length of the next steps (all uniforms: the substep h, alpha / h^2, the motion horizon 1.5 dt, the world
  // dt): the same substeps over a longer step, when frames are slow (writePoses)
  setDt(dt) {
    if (dt === this.curDt) return;
    this.curDt = dt;
    const W = this.world, h = dt / this.nsub, q = this.device.queue;
    for (const { b, u } of this.paramList) {
      const f = new Float32Array(u.buffer);
      f[4] = h; f[8] = W.alpha / (h * h); f[13] = 1.5 * dt;
      q.writeBuffer(b, 0, u);
    }
    this.sw[0] = dt;
    q.writeBuffer(this.swb, 0, this.sw);
  }

  // hand-overs (auto_client.js), 14 per body: pos, quat, vel, om, asleep. Out: the last read-back state (a frame or
  // two old); in: the shadow world takes it and it goes up
  getBodies() {
    const s = this.lastState, W = this.world, out = new Float64Array(14 * this.B);
    for (let b = 0; b < this.B; b++) {
      const o = 14 * b;
      if (s) { out.set(s.subarray(b * BODY, b * BODY + 3), o); out.set(s.subarray(b * BODY + 4, b * BODY + 8), o + 3); out.set(s.subarray(b * BODY + 8, b * BODY + 11), o + 7); out.set(s.subarray(b * BODY + 12, b * BODY + 15), o + 10); out[o + 13] = s[b * BODY + 19]; }
      else { out.set(W.pos.subarray(3 * b, 3 * b + 3), o); out.set(W.quat.subarray(4 * b, 4 * b + 4), o + 3); out.set(W.vel.subarray(3 * b, 3 * b + 3), o + 7); out.set(W.om.subarray(3 * b, 3 * b + 3), o + 10); out[o + 13] = W.asleep[b]; }
    }
    return out;
  }
  setBodies(bodies) {
    const W = this.world;
    W.reset();
    for (let b = 0; b < this.B; b++) {
      const o = 14 * b;
      W.setPose(b, bodies.subarray(o, o + 3), bodies.subarray(o + 3, o + 7));
      W.vel.set(bodies.subarray(o + 7, o + 10), 3 * b); W.om.set(bodies.subarray(o + 10, o + 13), 3 * b); W.asleep[b] = bodies[o + 13] ? 1 : 0;
    }
    this.upload();
    if (W.anyAwake()) { this.awakeMask = 0; this.start(); } else { this.running = false; this.onState(); }
  }

  // inspection / tests: exactly n steps now (no clock), then the bodies' state read back (Float32Array, BODY per body)
  async debugSteps(n) {
    const d = this.device, enc = d.createCommandEncoder({ label: 'physics debug steps' });
    for (let s = 0; s < n; s++) this.recordStep(enc);
    const st = d.createBuffer({ size: 4 * BODY * this.B, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.bodies, 0, st, 0, 4 * BODY * this.B);
    d.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return a;
  }

  // a readback's state: the shadow world's poses, the awake and moved masks, incidents, stop when all sleep
  applyReadback(a, stepsAtRead) {
    const B = this.B;
    this.lastState = a;
    this.pairs = new Uint32Array(a.buffer, 4 * BODY * B, 1)[0];
    const W = this.world, xf = new Float64Array(12 * B);
    let awake = false, mask = 0;
    for (let b = 0; b < B; b++) {
      W.pos.set(a.subarray(b * BODY, b * BODY + 3), 3 * b); W.quat.set(a.subarray(b * BODY + 4, b * BODY + 8), 4 * b);
      W.worldFromRest(b, xf, 12 * b);
      if (W.invM[b] > 0 && a[b * BODY + 19] === 0) {
        awake = true;
        const o = b * BODY, fdt = (this.frameMs || 1000 * this.dt) / 1000;
        const v = Math.hypot(a[o + 8], a[o + 9], a[o + 10]), w = Math.hypot(a[o + 12], a[o + 13], a[o + 14]);
        const mp = this.markPos, mq = this.markQuat, R = this.reach[b];
        // rotation angle ~ 2 |q - q_marked| (the closer of q, -q): well conditioned near zero, unlike acos of the dot
        let dm = 0, dp = 0;
        for (let k = 0; k < 4; k++) { const x = a[o + 4 + k], y = mq[4 * b + k]; dm += (x - y) ** 2; dp += (x + y) ** 2; }
        const drift = Math.hypot(a[o] - mp[3 * b], a[o + 1] - mp[3 * b + 1], a[o + 2] - mp[3 * b + 2]) + 2 * Math.sqrt(Math.min(dm, dp)) * R;
        if ((v + w * R) * fdt > this.moveTol || drift > this.moveTol) {
          mask |= 1 << W.meta.parts[b].asset_index;
          for (let k = 0; k < 3; k++) mp[3 * b + k] = a[o + k];
          for (let k = 0; k < 4; k++) mq[4 * b + k] = a[o + 4 + k];
        }
      }
    }
    this.awakeMask = mask;
    this.checkIncidents(a);
    if (this.xf) for (let b = 0; b < B; b++) {   // maskMode 'readback': the parts whose read-back pose changed
      if (W.invM[b] === 0) continue;
      for (let k = 12 * b; k < 12 * b + 12; k++) if (xf[k] !== this.xf[k]) { this.readbackMoved |= 1 << W.meta.parts[b].asset_index; break; }
    }
    this.xf = xf;
    const wasRunning = this.running;
    if (this.running && !awake && stepsAtRead > (this.startStep ?? 0)) this.running = false;   // everything asleep after stepping: stop
    const changed = wasRunning || awake !== this.awake;
    this.awake = awake;
    if (changed) this.onState();   // only while stepping, or once when it stops: an idle readback asks for no frame
  }

  // the steps of one frame, [count, step length]; acc: the time owed (s), gap: this frame's length (ms). A normal
  // frame takes the due fixed-dt steps (at most maxStepsPerFrame, the rest dropped: slow motion). A slow frame (over
  // 1.5 dt) takes the time owed in equal steps of at most maxDt, as many as fit in costFrac of the frame at the
  // measured GPU cost per step (at least one); past that each is maxDt and the rest is dropped. The physics then keeps
  // real time at any render rate it can afford (David's marble pile at 110 ms frames: 3-4 steps, not one 1/30 s step)
  static plan(acc, gap, { dt, maxDt, maxStepsPerFrame, costFrac, stepGpuMs }) {
    if (acc < dt - 1e-9) return [0, dt];
    if (!(gap > 1500 * dt)) return [Math.min(Math.max(1, maxStepsPerFrame | 0), Math.floor((acc + 1e-9) / dt)), dt];
    const want = Math.ceil(acc / maxDt - 1e-9), afford = Math.max(1, Math.floor(costFrac * gap / Math.max(stepGpuMs, 1e-3)));
    const n = Math.min(want, afford);
    return [n, n === want ? acc / n : maxDt];
  }

  // main.js writeParts, once per frame after renderer.writeParts: the due steps (plan), then the physics parts' poses into partBuf; a readback if none is in flight.
  // Returns the asset parts it moves this frame, for lighting.movedMask: the parts of bodies awake and not known asleep
  // from the last readback (awake bodies are the ones moving; read-back poses reach main.js only every few frames),
  // held one more frame after a body falls asleep so the final pose gets its shadows. A stopgap; the renderer ORs in
  // the GPU-side moved bits (gpu_pose.wgsl) as they read back
  writePoses(partBuf, enabled) {
    if (!enabled || this.released) return 0;
    const d = this.device, t = performance.now();
    // idle (everything asleep, nothing placed or pushed since): no encode, no readback; main.js keeps drawing the poses
    // it read last (xf), which are the GPU's final ones. Without this every frame read back, and each readback's
    // onState asked for another frame: the page redrew continuously while all slept (David's lag, 2026-10-30)
    if (!this.running && !this.dirty && !this.pending) { const m = this.prevMask; this.prevMask = 0; this.frameT = t; return m; }
    this.dirty = false;
    if (this.posePart !== partBuf) { this.poseGroup = this.group('pose_out', 'plain', 0, { 65: partBuf }); this.posePart = partBuf; }
    let enc = d.createCommandEncoder({ label: 'physics' });
    let n = 0;
    if (this.running) {
      const wall = Math.min(t - this.last, 250) / 1000;
      this.acc += wall;
      this.last = t;
      const [cnt, dt] = PhysicsGPU.plan(this.acc, this.frameT ? t - this.frameT : 0, this);
      if (cnt) {
        this.setDt(dt);
        if (this.tq && !this.tqPending) { this.timedSteps = cnt; enc.beginComputePass({ label: 'physics t0', timestampWrites: { querySet: this.tq, beginningOfPassWriteIndex: 0 } }).end(); }
        // one command buffer per step (2026-10-31): a slow frame's several steps never make one huge command buffer
        // (David's M1 lost the device, Metal out of memory, on a Drop into the marble pile with multi-step frames)
        for (let s = 0; s < cnt; s++) {
          const e = s === 0 ? enc : d.createCommandEncoder({ label: 'physics step' });
          this.recordStep(e);
          if (cnt > 1) d.queue.submit([e.finish()]);
        }
        if (cnt > 1) enc = d.createCommandEncoder({ label: 'physics poses' });
        n = cnt;
        this.acc = Math.min(Math.max(0, this.acc - cnt * dt), this.dt);   // past what the frame took: dropped
      }
      this.simE += 0.1 * (n * dt - this.simE); this.wallE += 0.1 * (wall - this.wallE);
      if (this.wallE > 0) this.rtf = Math.min(1, this.simE / this.wallE);
    }
    if (this.frameT && t - this.frameT < 250) this.frameMs = this.frameMs ? 0.9 * this.frameMs + 0.1 * (t - this.frameT) : t - this.frameT;   // gsDebug.physics.frameMs
    this.frameT = t;
    this.stepsLast = n; this.stepDtMs = 1000 * this.curDt;   // inspection: steps and step length of the last frame
    const timed = this.tq && !this.tqPending && n > 0;
    const cp = enc.beginComputePass({ label: 'physics poses', ...(timed ? { timestampWrites: { querySet: this.tq, endOfPassWriteIndex: 1 } } : {}) });
    cp.setPipeline(this.pipes.pose_out); cp.setBindGroup(0, this.poseGroup); cp.dispatchWorkgroups(Math.ceil(this.B / 64));
    cp.end();
    const read = !this.pending;
    if (read) { enc.copyBufferToBuffer(this.bodies, 0, this.staging, 0, 4 * BODY * this.B); enc.copyBufferToBuffer(this.by[28], 0, this.staging, 4 * BODY * this.B, 16); }
    if (timed) { enc.resolveQuerySet(this.tq, 0, 2, this.tqRes, 0); enc.copyBufferToBuffer(this.tqRes, 0, this.tqRead, 0, 16); }
    d.queue.submit([enc.finish()]);
    if (timed) {   // the GPU time of this frame's steps (and the pose pass) per step, into stepGpuMs
      this.tqPending = true;
      const steps = this.timedSteps;
      this.tqRead.mapAsync(GPUMapMode.READ).then(() => {
        const ts = new BigUint64Array(this.tqRead.getMappedRange().slice(0));
        this.tqRead.unmap(); this.tqPending = false;
        const ms = Number(ts[1] - ts[0]) / 1e6 / steps;
        if (ms > 0 && ms < 1000) this.stepGpuMs += 0.2 * (ms - this.stepGpuMs);
      }).catch(() => { this.tqPending = false; });
    }
    this.ms = performance.now() - t;
    let moved;
    if (this.maskMode === 'readback') { moved = this.readbackMoved; this.readbackMoved = 0; }
    else { const cur = this.running ? this.awakeMask : 0; moved = cur | this.prevMask; this.prevMask = cur; }   // CPU time to record and submit (the GPU time is not measured here)
    if (read) {
      this.pending = true;
      const stepsAtRead = this.stepCount;
      this.staging.mapAsync(GPUMapMode.READ).then(() => {
        const a = new Float32Array(this.staging.getMappedRange().slice(0));
        this.staging.unmap();
        this.pending = false;
        this.applyReadback(a, stepsAtRead);
      }).catch(() => { this.pending = false; });
    }
    return moved;
  }
}
