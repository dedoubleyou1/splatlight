// One-path specular (M45 E9 milestone 7; experiments/M45_unified/E9_gpu README section 7): with ?lighting=onepath,
// shiny Gaussians (a material with the specular flakes, the per-box shininess sliders) take every node of a 512-entry
// light cut as an ASG lobe toward the eye (E6). spec_cut stores the cut and its T per receiver cluster when the lighting,
// a part or a material changes; spec_eval runs every frame after the colour copy and adds the view-dependent change
// k (alb iso S - (B - Le)) to onepath's colour. Needs the asset's onepath_spec.bin (bake_spec.py) and 10 storage
// buffers per shader stage (main.js requests the adapter's limit); the materials are copied into its own buffer.

const SKMAX = 512;
const SREC = 2 + 2 * SKMAX;      // onepath_spec.wgsl: words per cluster record
const MATW = 32 * 8;
const CUT_EVERY = 32;            // frames between cut rebuilds while the one path converges             // material slots (lighting.js MAX_MATERIALS) x 8 words

export class OnePathSpec {
  static async load(device, assetName, runtimeShaders, onepath, lighting) {
    if (device.limits.maxStorageBuffersPerShaderStage < 10) {
      console.warn('one-path specular off: needs 10 storage buffers per stage, device has', device.limits.maxStorageBuffersPerShaderStage);
      return null;
    }
    const r = await fetch(`runtime/assets/${assetName}/onepath_spec.bin`);
    if (!r.ok) { console.warn('one-path specular off: onepath_spec.bin missing (bake_spec.py)'); return null; }
    const c6 = new Uint32Array(await r.arrayBuffer());
    const [op, sp] = await Promise.all(['onepath', 'onepath_spec'].map(n => fetch(`runtime/wgsl/${n}.wgsl`).then(x => x.text())));
    const src = [runtimeShaders.common, runtimeShaders.material_ext, op, sp].join('\n');
    return new OnePathSpec(device, src, c6, onepath, lighting);
  }

  constructor(device, src, c6, onepath, lighting) {
    this.device = device; this.onepath = onepath; this.lighting = lighting;
    const B = GPUBufferUsage, m = onepath.meta;
    this.ncl = m.ncl;
    this.recOff = MATW + c6.length;
    this.sbuf = device.createBuffer({ label: 'onepath spec (materials, covariances, cuts)', size: (this.recOff + m.ncl * SREC) * 4,
      usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
    device.queue.writeBuffer(this.sbuf, MATW * 4, c6);
    this.u = device.createBuffer({ label: 'onepath spec uniform', size: 48, usage: B.UNIFORM | B.COPY_DST });
    const BT = t => ({ type: t });
    const layout = device.createBindGroupLayout({ label: 'onepath spec', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: BT('uniform') },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: BT('storage') },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: BT('storage') }] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [onepath.layout, layout] });
    const module = device.createShaderModule({ label: 'onepath spec', code: src });
    this.pipes = Object.fromEntries(['spec_cut', 'spec_eval'].map(e =>
      [e, device.createComputePipeline({ label: e, layout: pl, compute: { module, entryPoint: e } })]));
    this.group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: this.u } }, { binding: 1, resource: { buffer: this.sbuf } },
      { binding: 2, resource: { buffer: onepath.colour } }] });
    this.matKey = null;
    this.cutFrames = 0;   // spec_cut runs while this is > 0 (a material change, every CUT_EVERY converging frames, settling)
    this.wasConverging = false;
  }

  // after onepath.encode (which wrote its params slot 0 and the colour copy this frame): the cuts if anything changed,
  // then the specular toward the eye
  encode(encoder, eye) {
    const L = this.lighting;
    const mu = new Uint32Array(L.materialBytes, 0, L.materialCount * 8);
    // nothing shiny (no material with the specular flakes and a shine above 0): no pass at all
    let shiny = false;
    for (let m = 0; m < L.materialCount; m++) if ((mu[m * 8 + 5] & 8) && (mu[m * 8 + 6] & 0x7fff)) shiny = true;
    if (!shiny) { this.wasConverging = false; return; }
    const mk = Array.from(mu).join();
    if (mk !== this.matKey) {
      this.matKey = mk; this.cutFrames = 1;
      encoder.copyBufferToBuffer(L.buffers.materials, 0, this.sbuf, 0, Math.min(L.materialCount * 32, MATW * 4));
    }
    // the cut follows B: while the one path converges, rebuild it every CUT_EVERY frames (every frame made those frames
    // ~15x slower in the M1 browser) and once more when it settles
    const conv = this.onepath.converging;
    if (conv && this.onepath.frame % CUT_EVERY === 0) this.cutFrames = 1;
    if (!conv && this.wasConverging) this.cutFrames = 1;
    this.wasConverging = conv;
    const u = new Uint32Array(12); new Float32Array(u.buffer).set(eye, 0);
    u.set([SKMAX, Math.min(L.materialCount, MATW / 8), this.ncl, this.recOff, MATW, L.specSplit ? 1 : 0], 4);
    this.device.queue.writeBuffer(this.u, 0, u);
    const run = e => {
      const pass = encoder.beginComputePass({ label: `onepath ${e}` });
      pass.setPipeline(this.pipes[e]); pass.setBindGroup(0, this.onepath.groups[0]); pass.setBindGroup(1, this.group);
      pass.dispatchWorkgroups(this.ncl); pass.end();
    };
    if (this.cutFrames > 0) { run('spec_cut'); this.cutFrames--; }
    run('spec_eval');
  }
}
