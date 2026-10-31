// Gaussian particles (prototype, 2026-10-31): ?particles=sparks|smoke|dust|rain[&pcount=N][&porigin=x,y,z]
// [&psize=1][&pdens=1][&pgain=1][&pshutter=1][&pwind=x,y,z][&pturb=1][&pg=<HG g>]
// One compute pass a frame (shaders/particles.wgsl) steps the particles and writes them as world-space Gaussians into
// renderer.particleBuf; project.wgsl draws that range after the scene (RenderFrame.particles), so they are sorted and
// blended with the scene's own Gaussians. They are not parts: no part slot, no 32-part cap, no host transform.
// The lighting does not see them (no shadows from smoke, sparks do not light the room): see the design notes.

const PRESETS = { sparks: 0, smoke: 1, dust: 2, rain: 3 };
// per-preset defaults (the settings of renderer/results/particles/; docs/gaussian_particles.md)
const DEFAULTS = {
  sparks: { count: 3000, gain: 6, size: 1, density: 1, turb: 1 },
  smoke: { count: 12000, gain: 3, size: 0.35, density: 2.5, turb: 2 },
  dust: { count: 4000, gain: 5, size: 2.5, density: 3, turb: 1 },
  rain: { count: 6000, gain: 3, size: 1.5, density: 10, turb: 1 },
};
const U_BYTES = 10 * 16;
const GRID = 32;   // particles.wgsl GRID

export function particleConfig(params, header) {
  const kind = params.get('particles');
  if (!kind || !(kind in PRESETS)) return null;
  const vec = (k, d) => (params.get(k) ?? '').split(',').map(Number).length === 3 && params.get(k) ? params.get(k).split(',').map(Number) : d;
  const room = header.room ?? { x: [-1, 1], y: [0, 2], z: [0, 2] };
  const lo = [room.x[0], room.y[0], room.z[0]], hi = [room.x[1], room.y[1], room.z[1]];
  // default emitter: the middle of the floor, a little toward the open side (in front of the boxes)
  const origin = vec('porigin', [0.5 * (lo[0] + hi[0]), lo[1] + 0.35 * (hi[1] - lo[1]), lo[2] + 0.02]);
  return {
    kind, preset: PRESETS[kind],
    count: Math.max(1, Math.min(1 << 20, Number(params.get('pcount') ?? DEFAULTS[kind].count))),
    origin, lo, hi,
    size: Number(params.get('psize') ?? DEFAULTS[kind].size), density: Number(params.get('pdens') ?? DEFAULTS[kind].density),
    gain: Number(params.get('pgain') ?? DEFAULTS[kind].gain),
    shutter: Number(params.get('pshutter') ?? 1) / 60,   // the smear length: a 1/60 s exposure by default
    wind: vec('pwind', [0, 0, 0]), turb: Number(params.get('pturb') ?? DEFAULTS[kind].turb),
    g: params.has('pg') ? Number(params.get('pg')) : -2,
    selfShadow: params.get('pshadow') !== '0',   // ?pshadow=0: no particle shadow grid march
  };
}

export class Particles {
  constructor(device, shaders, cfg, outBuf, noProbes) {
    this.device = device;
    this.cfg = cfg;
    this.count = cfg.count;
    this.t = 0;
    const B = GPUBufferUsage, C = GPUShaderStage.COMPUTE;
    this.uniform = device.createBuffer({ label: 'particle uniform', size: U_BYTES, usage: B.UNIFORM | B.COPY_DST });
    this.state = device.createBuffer({ label: 'particle state', size: this.count * 32, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
    // staggered births: slot i is born at -i / rate, so emission is steady from the first frame (dust and rain start
    // already in flight: their first ages are spread over a life instead)
    const s = new Float32Array(this.count * 8);
    const meanLife = { 0: 1.5, 1: 6.5, 2: 9, 3: 0.45 }[cfg.preset];
    for (let i = 0; i < this.count; i++) {
      s[8 * i + 3] = cfg.preset >= 2 ? 0 : -meanLife * (i / this.count);   // age: dust and rain are born at once, mid-life (particles.wgsl)
      s[8 * i + 7] = 1e-6;   // life: a slot respawns on the first step that sees age >= life
    }
    device.queue.writeBuffer(this.state, 0, s);
    this.outBuf = outBuf;
    this.layout = device.createBindGroupLayout({ label: 'particles', entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: { type: 'storage' } },
      { binding: 2, visibility: C, buffer: { type: 'storage' } },
      { binding: 3, visibility: C, buffer: { type: 'uniform' } },
      { binding: 4, visibility: C, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: C, buffer: { type: 'storage' } },
    ] });
    this.grid = device.createBuffer({ label: 'particle shadow grid', size: GRID ** 3 * 4, usage: B.STORAGE | B.COPY_DST });
    const module = device.createShaderModule({ label: 'particles', code: shaders.common + '\n' + shaders.particles });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.pipelines = ['step', 'shade'].map(entryPoint => device.createComputePipeline({ label: `particles ${entryPoint}`, layout, compute: { module, entryPoint } }));
    this.bindProbes(noProbes);
  }

  // the lighting's SH2 probes (lighting.buffers.probe_params / probe_sh); without them a dim constant
  bindProbes({ params, sh }) {
    this.group = this.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.state } },
      { binding: 2, resource: this.outBuf }, { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: sh } }, { binding: 5, resource: { buffer: this.grid } }] });
  }

  // advance by dt seconds (0 = frozen, but still re-shaded for the current eye)
  // lamp: { center, area, radiance } (the asset's rectangular lamp) or null
  update(dt, eye, lamp = this.lamp) {
    this.lamp = lamp;
    const c = this.cfg;
    dt = Math.min(Math.max(dt, 0), 1 / 20);
    this.t += dt;
    const u = new ArrayBuffer(U_BYTES), f = new Float32Array(u), w = new Uint32Array(u);
    f[0] = this.t; f[1] = dt; w[2] = this.count; w[3] = c.preset;
    f.set([...c.origin, 1], 4);
    f.set([...c.lo, c.shutter], 8);
    f.set([...c.hi, c.gain], 12);
    f.set([...c.wind, c.turb], 16);
    f.set([c.size, c.density, c.g, 0], 20);
    f.set([...eye, 0], 24);
    if (lamp) { f.set([...lamp.center, lamp.area], 28); f.set([...lamp.radiance, c.selfShadow ? 1 : 0], 32); }
    this.device.queue.writeBuffer(this.uniform, 0, u);
  }

  // run `seconds` of simulation now, one submit per step (captures: start mid-effect instead of at the first spark)
  warm(seconds, eye, step = 1 / 60) {
    for (let k = 0; k < Math.round(seconds / step); k++) {
      this.update(step, eye);
      const e = this.device.createCommandEncoder();
      this.encode(e);
      this.device.queue.submit([e.finish()]);
    }
  }

  encode(encoder) {
    encoder.clearBuffer(this.grid);
    const pass = encoder.beginComputePass({ label: 'particles' });
    pass.setBindGroup(0, this.group);
    for (const p of this.pipelines) { pass.setPipeline(p); pass.dispatchWorkgroups(Math.ceil(this.count / 64)); }
    pass.end();
  }
}
