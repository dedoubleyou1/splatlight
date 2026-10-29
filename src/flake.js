// Camera-only SGGX diffuse flakes (runtime/wgsl/flake_common.wgsl, experiments/M40_flake_camera), the View panel's
// "SGGX flakes (camera)" box, off by default. Reads the lighting's buffers (lighting.buffers) and writes `cam`, one
// colour per Gaussian that project.wgsl shows instead of colour + gain x eye bounce; nothing the lighting reads changes.
// Passes, after the lighting and before project: flake_base, flake_direct, then flake_bounce with SH2 probes; and
// flake_expo once. Emissive light (shade_emit) only: with the analytic lamp main.js leaves the toggle off.

const WG = 64;
const PARAM_BYTES = 32;

export class FlakeCamera {
  constructor(device, sources, lighting) {
    this.device = device;
    this.count = lighting.count;
    this.materialCount = lighting.materialCount;
    const b = lighting.buffers, B = GPUBufferUsage;
    this.cam = device.createBuffer({ label: 'flake camera colour', size: this.count * 16, usage: B.STORAGE | B.COPY_SRC });
    this.expo = device.createBuffer({ label: 'flake exposure SH2', size: this.count * 48, usage: B.STORAGE | B.COPY_SRC });
    this.params = device.createBuffer({ label: 'flake params', size: PARAM_BYTES, usage: B.UNIFORM | B.COPY_DST });
    this.flatR = 0.1;       // sx / sy below which flakes are flat (closed form): ~70 % of the SDF room's Gaussians
    this.hFloor = 0.1;      // the lobe never exceeds 10 x the M39 colour
    this.samples = 4;       // VNDF samples for rounder Gaussians
    const tail = sources.flake_common;
    const direct = device.createShaderModule({ label: 'flake direct',
      code: [sources.common, sources.material_ext, sources.emissive, tail, sources.flake_direct].join('\n') });
    const bounce = device.createShaderModule({ label: 'flake bounce',
      code: [sources.common, sources.material_ext, sources.bounce, tail, sources.flake_bounce].join('\n') });
    const res = {
      0: b.uniform, 1: b.gaussians, 2: b.nodes, 3: b.vgrid, 4: b.parts, 6: b.lamp_state, 7: b.materials, 8: b.colour,
      10: b.probe_params, 11: b.probe_data, 14: b.lo_b, 18: b.emit_nodes, 19: b.emit_params, 25: b.probe_sh,
      30: this.cam, 31: this.params, 32: this.expo,
    };
    const binds = {
      flake_expo: [direct, [1, 2, 3, 31, 32]],
      flake_base: [direct, [1, 7, 8, 14, 30, 31]],
      flake_direct: [direct, [0, 1, 2, 3, 4, 6, 18, 19, 30, 31, 32]],
      flake_bounce: [bounce, [0, 1, 2, 3, 4, 10, 11, 25, 30, 31, 32]],
    };
    this.pipes = {};
    for (const [entry, [module, list]] of Object.entries(binds)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      const group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: list.map(k => ({ binding: k, resource: { buffer: res[k] } })) });
      this.pipes[entry] = { pipe, group };
    }
    this.expoDone = false;
  }

  writeParams(gain, shBounce) {
    const p = new ArrayBuffer(PARAM_BYTES), f = new Float32Array(p), u = new Uint32Array(p);
    f[0] = gain; f[1] = this.flatR; f[2] = this.hFloor;
    u[3] = this.samples; u[4] = shBounce ? 1 : 0; u[5] = this.count; u[6] = this.materialCount;
    this.device.queue.writeBuffer(this.params, 0, p);
  }

  // gain = the bounce gain (0 when the bounce is off); shBounce = the probe bounce ran this frame with SH2 probes
  encode(encoder, timers, gain, shBounce) {
    this.writeParams(gain, shBounce);
    const run = entry => {
      const { pipe, group } = this.pipes[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: timers.names?.includes(entry) ? timers.writes(entry) : undefined });
      pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(this.count / WG)); pass.end();
    };
    if (!this.expoDone) { run('flake_expo'); this.expoDone = true; }
    run('flake_base');
    run('flake_direct');
    if (shBounce) run('flake_bounce');
  }
}
