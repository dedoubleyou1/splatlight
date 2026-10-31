// Default (or ?physgpu=auto): the physics runs on the GPU (gpu_client.js) or the CPU worker (client.js), chosen by workload with a
// per-machine cost model, and handed over between them with the bodies' full state (David, 2026-10-30: the choice
// follows the object count and packing, not the frame rate alone).
//
// Workload = the active contact node pairs (both paths report them). Cost, refined online from the running path
// (running means of the per-pair slope over a seeded intercept; the idle path keeps its last or seeded estimate):
//   CPU worker: step ms     ~ cpuA + cpuB pairs   (its own clock: real time while a step is under dt)
//   GPU:        GPU ms/step ~ gpuA + gpuB pairs   (timestamp queries where the device has them, else the frame with
//                                                  physics minus the recent frame without, per step)
// Each path is judged by the real-time factor it would reach at this frame rate (2026-10-31; an absolute frame budget
// rejected the GPU on any scene whose rendering alone is over it, e.g. the marble pile's refraction): the CPU's
// dt / step ms; the GPU's steps it can afford per frame (gpu_client.js plan: costFrac of the frame) over the steps
// the frame needs. The CPU is preferred while it keeps real time (it leaves the GPU to rendering). A switch needs the
// other path to reach a 25% higher factor (or real time), and at least 2 s since the last switch.
// gsDebug.physics.auto holds the estimates and the reason for the current choice.
//
// Same interface as PhysicsClient / PhysicsGPU. The GPU path is made only when a switch to it is wanted, and freed
// (every buffer destroyed) when the CPU takes over again (2026-10-31: it had held ~55 MB of GPU memory from the
// start on a scene that never left the CPU, next to David's out-of-memory loss on the M1). The worker sleeps when idle.
import { PhysicsClient } from './client.js';
import { PhysicsGPU } from './gpu_client.js';

const MARGIN = 1.25, DWELL_MS = 2000, EMA = 0.1;

export class PhysicsAuto {
  // null when neither path starts; with only one it runs that one
  static async start(base, bake, device, budget = 0) {
    const makeGpu = () => PhysicsGPU.start(bake, device);
    const cpu = await PhysicsClient.start(base, budget);
    const gpu = cpu ? null : await makeGpu();   // no worker: the GPU from the start
    if (!gpu && !cpu) return null;
    return new PhysicsAuto(gpu, cpu, makeGpu);
  }

  // makeGpu: async, a new PhysicsGPU or null (tests pass fakes)
  constructor(gpu, cpu, makeGpu = null) {
    this.gpu = gpu; this.cpu = cpu; this.makeGpu = makeGpu; this.gpuFailed = !gpu && !makeGpu;
    this.cur = cpu ?? gpu;   // the CPU first: preferred while it keeps real time
    this.onState = () => {};
    for (const c of [gpu, cpu]) if (c) this.hook(c);
    const dt = (gpu ?? cpu).dt ?? 1 / 60;
    // seeds: conservative (the MacBook M1 CPU about 1 us per pair; a GPU step about 1 ms plus 0.1 us per pair)
    this.auto = { path: this.cur === cpu ? 'cpu' : 'gpu', reason: 'start', cpuA: 1, cpuB: 1e-3, gpuA: 1, gpuB: 1e-4,
      baseFrameMs: null, pairs: 0, predicted: null, stepBudgetMs: 1000 * dt, switches: 0, lastSwitch: 0, gpu: !!gpu || !!makeGpu, cpu: !!cpu, gpuMade: gpu ? 1 : 0, gpuFreed: 0 };
  }

  hook(c) { c.onState = () => { if (c === this.cur) { this.sample(); this.onState(); } }; }

  // the GPU path, made on first need (async: the switch happens on a later frame, once it exists)
  wantGpu() {
    if (this.gpu || this.gpuFailed || this.gpuStarting) return;
    this.gpuStarting = true;
    this.makeGpu().then(g => {
      this.gpuStarting = false;
      if (!g) { this.gpuFailed = true; this.auto.gpu = false; return; }
      this.gpu = g; this.hook(g); this.auto.gpuMade++;
    }).catch(() => { this.gpuStarting = false; this.gpuFailed = true; this.auto.gpu = false; });
  }

  get meta() { return this.cur.meta; }
  get bodyOf() { return this.cur.bodyOf; }
  get xf() { return this.cur.xf; }
  get ms() { return this.cur.ms; }
  get pairs() { return this.cur.pairs; }
  get awake() { return this.cur.awake; }
  get budget() { return this.cur.budget; }
  get incidents() { return this.cur.incidents; }
  get lastIncident() { return this.cur.lastIncident; }

  drop(seed) { this.cur.drop(seed); }
  reset() { this.cur.reset(); }
  place(sliderXf) { this.cur.place(sliderXf); }
  attachClicks(canvas, camera, enabled, speed = 2) {
    PhysicsClient.prototype.attachClicks.call({ worker: { postMessage: m => {
      if (this.cur === this.gpu) this.gpu.push(m.o, m.d, m.speed); else this.cpu.worker.postMessage(m);
    } } }, canvas, camera, enabled, speed);
  }

  // the measurements of the running path into its cost estimate
  sample() {
    const a = this.auto, c = this.cur, pairs = c.pairs ?? 0;
    a.pairs = pairs;
    if (!c.awake || !pairs) return;
    if (c === this.cpu && c.ms > 0) a.cpuB += EMA * (Math.max(0, c.ms - a.cpuA) / pairs - a.cpuB);
    if (c === this.gpu) {
      const step = c.tq ? c.stepGpuMs : c.frameMs && a.baseFrameMs !== null ? (c.frameMs - a.baseFrameMs) / Math.max(1, c.stepsLast ?? 1) : null;
      if (step !== null) a.gpuB += EMA * (Math.max(0, step - a.gpuA) / pairs - a.gpuB);
    }
  }

  // main.js writeParts, every frame: the GPU path's steps and poses (gpu_client.js writePoses); the frame time without
  // physics while it idles (the GPU's baseline); the choice
  writePoses(partBuf, enabled) {
    const a = this.auto, t = performance.now();
    if (this.prevT && t - this.prevT < 250 && !(this.cur.awake && enabled)) {
      const f = t - this.prevT;
      a.baseFrameMs = a.baseFrameMs === null ? f : a.baseFrameMs + EMA * (f - a.baseFrameMs);
    }
    this.prevT = t;
    const moved = this.cur === this.gpu ? this.gpu.writePoses(partBuf, enabled) : 0;
    if (enabled && this.cpu && (this.gpu || !this.gpuFailed)) this.choose(t);
    return moved;
  }

  // the real-time factors each path would reach at P pairs and frame length F ms (inspection: a.predicted)
  predict(P, F) {
    const a = this.auto, g = this.gpu ?? {}, dt = 1000 * ((g.dt ?? this.cur.dt) || 1 / 60), maxDt = 1000 * (g.maxDt ?? 1 / 30);
    const cpuMs = a.cpuA + a.cpuB * P, gpuStep = a.gpuA + a.gpuB * P;
    const slow = F > 1.5 * dt, need = F / (slow ? maxDt : dt);
    const afford = slow ? Math.max(1, Math.floor((g.costFrac ?? 0.1) * F / gpuStep)) : (g.maxStepsPerFrame ?? 3);
    return { cpuStepMs: +cpuMs.toFixed(2), gpuStepMs: +gpuStep.toFixed(3), frameMs: +F.toFixed(1),
      cpuRtf: +Math.min(1, dt / cpuMs).toFixed(3), gpuRtf: +Math.min(1, afford / need).toFixed(3), gpuAddedMs: +(Math.min(need, afford) * gpuStep).toFixed(2) };
  }

  choose(t) {
    const a = this.auto, P = this.cur.pairs ?? 0, F = a.baseFrameMs ?? 16.7;
    const p = a.predicted = this.predict(P, F), rt = 0.999;
    if (!this.cur.awake || t - a.lastSwitch < DWELL_MS) return;
    let to = null, why = '';
    if (this.cur === this.cpu && p.cpuRtf < rt && p.gpuRtf >= Math.min(rt, MARGIN * p.cpuRtf)) { if (!this.gpu) { this.wantGpu(); return; } to = this.gpu; why = `CPU step ~${p.cpuStepMs} ms at ${P} pairs: ${p.cpuRtf}x real time; GPU ${p.gpuRtf}x at ${p.frameMs} ms frames (+${p.gpuAddedMs} ms)`; }
    else if (this.cur === this.gpu && p.cpuStepMs * MARGIN < a.stepBudgetMs) { to = this.cpu; why = `CPU step ~${p.cpuStepMs} ms keeps real time at ${P} pairs (preferred: leaves the GPU to rendering)`; }
    else if (this.cur === this.gpu && p.gpuRtf < rt && p.cpuRtf >= Math.min(rt, MARGIN * p.gpuRtf)) { to = this.cpu; why = `GPU ${p.gpuRtf}x real time at ${p.frameMs} ms frames; CPU ${p.cpuRtf}x`; }
    if (to) this.handOver(to, why, t);
  }

  handOver(to, why, t) {
    const from = this.cur, bodies = from === this.gpu ? this.gpu.getBodies() : this.cpu.bodies;
    if (!bodies) return;
    to.setBodies(bodies);
    // the path left behind stops stepping: the worker gets the same bodies asleep, the GPU client stops its clock
    if (from === this.cpu) { const rest = Float64Array.from(bodies); for (let k = 13; k < rest.length; k += 14) rest[k] = 1; this.cpu.setBodies(rest); }
    this.cur = to;
    if (from === this.gpu) { this.gpu.release(); this.gpu = null; this.auto.gpuFreed++; }   // remade if needed again
    Object.assign(this.auto, { path: to === this.gpu ? 'gpu' : 'cpu', reason: why, lastSwitch: t, switches: this.auto.switches + 1 });
    console.info(`physics: ${this.auto.path.toUpperCase()} path (${why})`);
    this.onState();
  }
}
