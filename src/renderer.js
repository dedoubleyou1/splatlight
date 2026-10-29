// GPU resources and per-frame encoding for the splat path:
//   project (compute) -> scan (compute) -> scatter (compute), or with the stable
//   sort on, radix (compute: 2 x histogram, scan, scatter) in scatter's place
//   -> splat (instanced quads into rgba16float) -> display (to the swapchain)
// Bind group layouts are explicit so the native wgpu shell can mirror them.

export const PASSES = ['walk_lamp', 'ema', 'shade', 'lp_shade', 'lp_transfer', 'near_vis', 'iso_xfer', 'direct_dirs', 'bounce_dirs', 'bounce_sh', 'node_leaf', 'probe_gather', 'probe_sh2',
  'flake_direct', 'flake_bounce', 'project', 'scan', 'scatter', 'radix', 'splat', 'display'];
export const FRAME_BYTES = 736;   // RenderFrame (shaders/common.wgsl): 400 + mat_view + part_material_ext + lod
const SPLAT_BYTES = 48;
const NUM_BUCKETS = 65536;
const RADIX_TILE = 4096;   // radix.wgsl TILE
const HDR_FORMAT = 'rgba16float';
export const MAX_EMITTERS = 16384;

const S = GPUShaderStage;
const B = GPUBufferUsage;

function layout(device, label, entries) {
  return device.createBindGroupLayout({
    label,
    entries: entries.map(([visibility, type], binding) => (
      type === 'texture' ? { binding, visibility, texture: { sampleType: 'unfilterable-float' } }
        : { binding, visibility, buffer: { type } })),
  });
}

export class SplatRenderer {
  // lodNodes: lod.js nodeGaussians() result (?lod=): its internal nodes follow the leaves in the Gaussian buffer and
  // get splat slots after the emitters (project.wgsl, frame.lod)
  constructor(device, canvasFormat, shaders, asset, lodNodes = null) {
    this.device = device;
    this.count = asset.count;
    this.partCount = asset.partCount;
    this.nodeBase = asset.count + 1 + MAX_EMITTERS;
    this.lodNodeCount = lodNodes ? lodNodes.N - lodNodes.G : 0;
    const n = this.nodeBase + this.lodNodeCount;  // + the analytic lamp's marker + emissive Gaussians (+ LOD nodes)
    this.splatCount = n;
    const buf = (label, size, usage) => device.createBuffer({ label, size, usage });

    this.frameBuf = buf('frame', FRAME_BYTES, B.UNIFORM | B.COPY_DST);
    this.gaussBuf = buf('gaussians', asset.gaussians.byteLength + (lodNodes ? lodNodes.records.byteLength : 0), B.STORAGE | B.COPY_DST);
    device.queue.writeBuffer(this.gaussBuf, 0, asset.gaussians);
    if (lodNodes) device.queue.writeBuffer(this.gaussBuf, asset.gaussians.byteLength, lodNodes.records);
    this.lodBuf = buf('lod flags', Math.max(16, (lodNodes ? lodNodes.N : 0) * 4), B.STORAGE | B.COPY_SRC);   // lod.wgsl lod_select
    this.partBuf = buf('parts', asset.parts.byteLength, B.STORAGE | B.COPY_DST);
    device.queue.writeBuffer(this.partBuf, 0, asset.parts);
    this.splatBuf = buf('splats', n * SPLAT_BYTES, B.STORAGE);
    this.keyBuf = buf('keys', n * 4, B.STORAGE);
    this.histBuf = buf('histogram', NUM_BUCKETS * 4, B.STORAGE | B.COPY_DST);
    this.orderBuf = buf('order', n * 4, B.STORAGE);
    this.drawBuf = buf('draw args', 16, B.STORAGE | B.INDIRECT | B.COPY_SRC);
    this.envBuf = buf('environment (none)', 416, B.UNIFORM | B.COPY_DST);   // SPEC 2.9; bindEnv swaps in lighting's

    const C = S.COMPUTE, VF = S.VERTEX | S.FRAGMENT;
    const L = {
      project: layout(device, 'project', [[C, 'uniform'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage'], [C, 'storage'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'read-only-storage']]),
      scan: layout(device, 'scan', [[C, 'storage'], [C, 'storage']]),
      scatter: layout(device, 'scatter', [[C, 'uniform'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage']]),
      splat: layout(device, 'splat', [[VF, 'uniform'], [S.VERTEX, 'read-only-storage'], [S.VERTEX, 'read-only-storage']]),
      display: layout(device, 'display', [[S.FRAGMENT, 'uniform'], [S.FRAGMENT, 'texture'], [S.FRAGMENT, 'uniform']]),   // 2: the environment (sky background)
      radix: layout(device, 'radix', [[C, 'uniform'], [C, 'uniform'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage'], [C, 'storage']]),
    };
    this.layouts = L;
    this.shaders = shaders;
    this.stableSort = true;   // on by default; main.js sets it from the View panel each frame; the radix resources are built on the first stable encode
    const module = name => device.createShaderModule({ label: name, code: shaders.common + '\n' + shaders[name] });
    const compute = name => device.createComputePipeline({
      label: name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [L[name]] }),
      compute: { module: module(name), entryPoint: 'main' },
    });
    this.pipelines = { project: compute('project'), scan: compute('scan'), scatter: compute('scatter') };

    const splatModule = module('splat');
    const splatPipe = (blend) => device.createRenderPipeline({
      label: 'splat',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.splat] }),
      vertex: { module: splatModule, entryPoint: 'vs' },
      fragment: { module: splatModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT, blend }] },
      primitive: { topology: 'triangle-strip' },
    });
    const over = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' };
    const add = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
    this.pipelines.splatOver = splatPipe({ color: over, alpha: over });
    this.pipelines.splatAdd = splatPipe({ color: add, alpha: add });
    const displayModule = module('display');
    this.pipelines.display = device.createRenderPipeline({
      label: 'display',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.display] }),
      vertex: { module: displayModule, entryPoint: 'vs' },
      fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });

    const group = (l, resources) => device.createBindGroup({
      layout: l, entries: resources.map((r, binding) => ({ binding, resource: r.createView ? r.createView() : { buffer: r } })),
    });
    this.group = group;
    this.module = module;
    this.groups = {
      scan: group(L.scan, [this.histBuf, this.drawBuf]),
      scatter: group(L.scatter, [this.frameBuf, this.keyBuf, this.histBuf, this.orderBuf]),
      splat: group(L.splat, [this.frameBuf, this.splatBuf, this.orderBuf]),
    };
  }

  // Stable sort resources: a temp key and index buffer (one u32 each per splat slot, ~8 MB at 1M), the tile
  // histogram (256 x tiles) and one small uniform per pass. Pass 1 reads `keys` in index order (the tie-break) and
  // writes the temps; pass 2 writes the indices into `order`, so splat.wgsl is unchanged.
  buildRadix() {
    const device = this.device, L = this.layouts;
    const tiles = Math.ceil(this.splatCount / RADIX_TILE);
    const buf = (label, size, usage) => device.createBuffer({ label, size, usage });
    this.radixKeys = buf('radix keys', this.splatCount * 4, B.STORAGE);
    this.radixVals = buf('radix values', this.splatCount * 4, B.STORAGE);
    this.radixHist = buf('radix histogram', 256 * tiles * 4, B.STORAGE);
    const params = [[0, 1, 0, tiles], [8, 0, 1, tiles]].map((p, k) => {
      const b = buf(`radix pass ${k + 1}`, 16, B.UNIFORM | B.COPY_DST);
      device.queue.writeBuffer(b, 0, new Uint32Array(p));
      return b;
    });
    const module = this.module('radix');
    const layout = device.createPipelineLayout({ bindGroupLayouts: [L.radix] });
    this.radixTiles = tiles;
    this.radixPipes = ['histogram', 'scan', 'scatter'].map(entryPoint => device.createComputePipeline({
      label: `radix ${entryPoint}`, layout, compute: { module, entryPoint },
    }));
    // `keys` is bound as keys_out in pass 2 but never written there (params.last); `order` as vals_in in pass 1, unread.
    this.radixGroups = [
      this.group(L.radix, [this.frameBuf, params[0], this.keyBuf, this.orderBuf, this.radixKeys, this.radixVals, this.radixHist]),
      this.group(L.radix, [this.frameBuf, params[1], this.radixKeys, this.radixVals, this.keyBuf, this.orderBuf, this.radixHist]),
    ];
  }

  // The lit colour (direct toward the eye) and the bounce come from the lighting kernels.
  bindColour(colourBuf, bounceBuf, emitterBuf) {
    this.groups.project = this.group(this.layouts.project,
      [this.frameBuf, this.gaussBuf, this.partBuf, this.splatBuf, this.keyBuf, this.histBuf, colourBuf, bounceBuf, emitterBuf, this.lodBuf]);
    this.groupsLit = { colour: this.groups.project };
  }

  // Camera-only flake colour (flake.js): project reads `cam` in colour's slot; RenderFrame lamp.w = 1 drops the bounce
  // add (cam already holds it). use(true / false) picks the group for the next encode.
  bindFlake(camBuf, bounceBuf, emitterBuf) {
    this.groupsLit.flake = this.group(this.layouts.project,
      [this.frameBuf, this.gaussBuf, this.partBuf, this.splatBuf, this.keyBuf, this.histBuf, camBuf, bounceBuf, emitterBuf, this.lodBuf]);
  }

  useFlake(on) {
    this.groups.project = on && this.groupsLit.flake ? this.groupsLit.flake : this.groupsLit.colour;
  }

  resize(width, height) {
    this.hdr?.destroy();
    this.hdr = this.device.createTexture({
      size: [width, height], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.groups.display = this.group(this.layouts.display, [this.frameBuf, this.hdr, this.envBuf]);
  }

  // The environment uniform (lighting.buffers.env, SPEC 2.9) for the sky background behind the splats.
  bindEnv(envBuf) {
    this.envBuf = envBuf;
    if (this.hdr) this.groups.display = this.group(this.layouts.display, [this.frameBuf, this.hdr, this.envBuf]);
  }

  // Per-frame part records (SPEC §2.4): the whole buffer is a few hundred bytes.
  writeParts(partBytes) {
    this.device.queue.writeBuffer(this.partBuf, 0, partBytes);
  }

  encode(encoder, target, frameBytes, viewMode, timers) {
    this.device.queue.writeBuffer(this.frameBuf, 0, frameBytes);
    encoder.clearBuffer(this.histBuf);
    const groups = Math.ceil(this.splatCount / 256);
    const computePass = (name, x) => {
      const pass = encoder.beginComputePass({ label: name, timestampWrites: timers.writes(name) });
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, this.groups[name]);
      pass.dispatchWorkgroups(x);
      pass.end();
    };
    computePass('project', groups);
    computePass('scan', 1);
    if (this.stableSort) {
      if (!this.radixPipes) this.buildRadix();
      // One timed pass; WebGPU orders the six dispatches' storage accesses within it.
      const pass = encoder.beginComputePass({ label: 'radix', timestampWrites: timers.writes('radix') });
      for (const g of this.radixGroups) {
        pass.setBindGroup(0, g);
        [this.radixTiles, 1, this.radixTiles].forEach((x, k) => { pass.setPipeline(this.radixPipes[k]); pass.dispatchWorkgroups(x); });
      }
      pass.end();
    } else {
      computePass('scatter', groups);
    }
    if (timers.ms) timers.ms[this.stableSort ? 'scatter' : 'radix'] = 0;   // the pass not run this frame reads 0 in the HUD

    let pass = encoder.beginRenderPass({
      label: 'splat',
      colorAttachments: [{ view: this.hdr.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' }],
      timestampWrites: timers.writes('splat'),
    });
    pass.setPipeline(viewMode === 2 ? this.pipelines.splatAdd : this.pipelines.splatOver);
    pass.setBindGroup(0, this.groups.splat);
    pass.drawIndirect(this.drawBuf, 0);
    pass.end();

    pass = encoder.beginRenderPass({
      label: 'display',
      colorAttachments: [{ view: target, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
      timestampWrites: timers.writes('display'),
    });
    pass.setPipeline(this.pipelines.display);
    pass.setBindGroup(0, this.groups.display);
    pass.draw(3);
    pass.end();
  }
}
