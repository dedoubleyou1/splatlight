// Deterministic one-frame one path (M45 E9 section 10; David 2026-10-28: converge in one render frame):
// ?lighting=onepath&det=1. Every frame: BOUNCES x (sender tree, gather_det over every receiver cluster with E2's
// light cut of DET_K entries, nothing sampled, copy) and the geometric tail for the bounces not run, then onepath's
// colour copy (display fill). The first bounce builds the cut and its shadow walks per cluster and later bounces reuse
// them. Starts each frame from the previous frame's B (the first frame from emission), so a still scene stays exactly
// put and a change is converged within the frame. Receivers are the room-facing Gaussians only (the asset's
// onepath_det.bin, bake_onepath.py --det; the hidden layers show their neighbours through the display fill). Sky and
// sun (?env=) come in as distant senders, gathered once per frame.

const DET_K = 128;               // light-cut entries per receiver cluster
const BOUNCES = 3;               // bounce passes per frame, then the tail
const GAIN = 0.45;               // per-bounce gain of the geometric tail (E1)
const DKMAX = 512, DREC = 1 + 2 * DKMAX;

export class OnePathDet {
  static async load(device, assetName, runtimeShaders, onepath, lighting) {
    const [op, det] = await Promise.all(['onepath', 'onepath_det'].map(n => fetch(`runtime/wgsl/${n}.wgsl`).then(r => r.text())));
    const base = `runtime/assets/${assetName}/onepath_det`;
    const dm = await fetch(`${base}.json`).then(r => { if (!r.ok) throw new Error(`${base}.json missing (bake_onepath.py --det)`); return r.json(); });
    const lists = await fetch(`${base}.bin`).then(r => r.arrayBuffer());
    // &detlamp=exact: the emitter part's nodes open to exact leaves (M48: no SH-lobe ring on the ceiling, half the
    // steady error vs leaf-exact, ~22% slower; off by default because with the Gaussian lamp the lobe lands closer to
    // the mesh reference). &detlamp=cos: exact leaves sending as Lambertian panels (onepath.wgsl lamp_area), the
    // closest to the mesh (FLIP 0.101 vs 0.158 for the default)
    const mode = new URLSearchParams(globalThis.location?.search ?? '').get('detlamp');
    const exact = mode === 'exact' || mode === 'cos';
    const emitterPart = exact ? await fetch(`runtime/assets/${assetName}/asset.json`).then(r => r.ok ? r.json() : null).then(h => h?.emitters?.part ?? null) : null;
    const epart = emitterPart == null ? 0 : (1 << emitterPart) >>> 0;
    return new OnePathDet(device, [runtimeShaders.common, op, det].join('\n'), onepath, lighting, dm, lists, epart, mode === 'cos');
  }

  constructor(device, src, onepath, lighting, dm, lists, epart = 0, lampCos = false) {
    this.device = device; this.op = onepath; this.dm = dm; this.epart = epart; this.lampCos = lampCos;
    const B = GPUBufferUsage, m = onepath.meta;
    // its own receiver lists (room-facing receivers, their clusters) and uniform slots; everything else is onepath's
    this.lists = device.createBuffer({ label: 'onepath det lists', size: lists.byteLength, usage: B.STORAGE | B.COPY_DST });
    device.queue.writeBuffer(this.lists, 0, lists);
    this.params = device.createBuffer({ label: 'onepath det params', size: onepath.slot * 10, usage: B.UNIFORM | B.COPY_DST });
    const lb = lighting.buffers;
    this.g0 = Array.from({ length: 10 }, (_, k) => device.createBindGroup({ layout: onepath.layout, entries: [
      { binding: 0, resource: { buffer: this.params, offset: k * onepath.slot, size: onepath.psize } },
      { binding: 1, resource: { buffer: lb.gaussians } }, { binding: 2, resource: { buffer: lb.nodes } },
      { binding: 3, resource: { buffer: lb.kids } }, { binding: 4, resource: { buffer: lb.vgrid } },
      { binding: 5, resource: { buffer: onepath.tree } }, { binding: 6, resource: { buffer: onepath.nsh } },
      { binding: 7, resource: { buffer: onepath.gext } }, { binding: 8, resource: { buffer: this.lists } }] }));
    const dcN = dm.ncl * DREC;
    this.offs = [0, dcN, dcN + 4];   // dbuf: cut cache | weight cache (unused here) | flags (unused) | distant light
    this.dsOff = dcN + 8;
    this.dbuf = device.createBuffer({ label: 'onepath det (cut cache, distant light)', size: (this.dsOff + dm.receivers * 3) * 4, usage: B.STORAGE });
    const layout = device.createBindGroupLayout({ label: 'onepath det', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [onepath.layout, layout] });
    const module = device.createShaderModule({ label: 'onepath det', code: src });
    this.pipes = Object.fromEntries(['gather_det', 'det_save', 'det_tail'].map(e =>
      [e, device.createComputePipeline({ label: e, layout: pl, compute: { module, entryPoint: e } })]));
    this.du = Array.from({ length: BOUNCES }, () => device.createBuffer({ size: 64, usage: B.UNIFORM | B.COPY_DST }));
    this.groups = this.du.map(b => device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: b } }, { binding: 1, resource: { buffer: this.dbuf } }] }));
    this.frames = 0;
    this.changed = true;
  }

  get converging() { return this.changed; }   // a still scene needs no further frames

  reset() { this.op.reset(); this.frames = 0; this.changed = true; }

  // onepath's Params with this mode's lists offsets
  pbytes(o) {
    const d = this.dm, u = this.op.paramBytes(o), f = new Float32Array(u.buffer);
    u[11] = d.lsh_off; u[13] = d.cl_off; u[14] = d.ncl; f[684] = 1;   // n_max 1: the copy replaces B
    u[699] = this.lampCos ? 1 : 0;                                   // envu.w bit 0: Lambertian lamp panels
    return u;
  }

  encode(encoder, partXf, environment = null) {
    const op = this.op, d = this.dm, dev = this.device, m = d;
    const [, moved] = partXf ? op.readParts(partXf) : [0, 0];
    const envChanged = op.setEnv(environment);
    this.changed = this.frames < 2 || moved !== 0 || envChanged;
    if (this.changed) {
      // uniform slots: 0 = receivers (gather / copy / tail), 1.. = the sender tree
      dev.queue.writeBuffer(this.params, 0, this.pbytes({ R: d.receivers }));
      dev.queue.writeBuffer(op.params, 0, op.paramBytes({ R: op.meta.receivers }));   // onepath's slot 0: the specular pass reads it
      const tree = [['nb_leaf', d.senders, [d.send_off, 0]], ...d.levels.map(([o, c]) => ['nb_up', c, [o, c]])];
      tree.forEach(([, , lvl], k) => dev.queue.writeBuffer(this.params, (k + 1) * op.slot, this.pbytes({ lvl })));
      // from emission (first frame) the first bounce's cut would hold the lamp alone: the second builds the stored one
      const modes = this.frames === 0 ? [0, 2, 1] : [2, 1, 1];
      const run = (pipe, g0, g1, n) => {
        const pass = encoder.beginComputePass({ label: 'onepath det' });
        pass.setPipeline(pipe); pass.setBindGroup(0, g0); if (g1) pass.setBindGroup(1, g1);
        pass.dispatchWorkgroups(n); pass.end();
      };
      for (let b = 0; b < BOUNCES; b++) {
        const u = new Uint32Array(16); u.set([DET_K, 1, modes[b], this.epart, ...this.offs, 0]); new Float32Array(u.buffer)[8] = GAIN;
        u[12] = this.dsOff;
        dev.queue.writeBuffer(this.du[b], 0, u);
        tree.forEach(([e, n], k) => run(op.pipes[e], this.g0[k + 1], null, Math.ceil(n / 64)));
        if (b === BOUNCES - 1) run(this.pipes.det_save, this.g0[0], this.groups[b], Math.ceil(d.receivers / 64));
        run(this.pipes.gather_det, this.g0[0], this.groups[b], d.ncl);
        run(op.pipes.history_x, this.g0[0], null, Math.ceil(d.receivers / 64));
      }
      run(this.pipes.det_tail, this.g0[0], this.groups[0], Math.ceil(d.receivers / 64));
      this.frames++;
    }
    const pass = encoder.beginComputePass({ label: 'onepath colour' });
    pass.setPipeline(op.cpipe); pass.setBindGroup(0, op.cgroup); pass.dispatchWorkgroups(Math.ceil(op.meta.G / 64)); pass.end();
  }
}
