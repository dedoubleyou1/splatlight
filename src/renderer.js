// GPU resources and per-frame encoding for the splat path:
//   project (compute) -> scan (compute) -> scatter (compute), or with the stable
//   sort on, radix (compute: 2 x histogram, scan, scatter) in scatter's place
//   -> splat (instanced quads into rgba16float) -> display (to the swapchain)
// Bind group layouts are explicit so the native wgpu shell can mirror them.

export const PASSES = ['shadow', 'shade', 'near_vis', 'iso_xfer', 'direct_dirs', 'bounce_dirs', 'bounce_sh', 'node_leaf', 'probe_gather', 'probe_sh2',
  'flake_direct', 'flake_bounce', 'project', 'scan', 'scatter', 'radix', 'lensmap', 'hgbuild', 'exittrace', 'splat', 'exitmap', 'refract', 'display'];
export const FRAME_BYTES = 1200;   // RenderFrame (shaders/common.wgsl): 400 + mat_view + part_material_ext + lod + explode + mat_view_ext + tone (u32 260) + part_fade (f32 264-295) + particles (u32 296-299)
const SPLAT_BYTES = 48;
const NUM_BUCKETS = 65536;
const RADIX_TILE = 4096;   // radix.wgsl TILE
const HDR_FORMAT = 'rgba16float';
const LAYER_DEPTH_FORMAT = 'r16float';
const GRAD_FORMAT = 'rg16float';   // compact B: the summed glass gradient
const LENSMAP_FORMAT = 'rgba32float';   // per-pixel nearest lens (t_near, t_far, index + 1)   // ?refract=1: alpha-weighted view depth of the inside and behind layers
export const MAX_EMITTERS = 16384;

import { LENS_BYTES } from './refract.js';
import { RayGrid } from './raygrid.js';

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
  // particleCount: Gaussian particles (src/particles.js) get the last splat slots, after the LOD nodes
  constructor(device, canvasFormat, shaders, asset, lodNodes = null, particleCount = 0) {
    this.device = device;
    this.count = asset.count;
    this.partCount = asset.partCount;
    this.nodeBase = asset.count + 1 + MAX_EMITTERS;
    this.lodNodeCount = lodNodes ? lodNodes.N - lodNodes.G : 0;
    this.particleCount = particleCount;
    this.particleBase = particleCount > 0 ? this.nodeBase + this.lodNodeCount : 0;
    const n = this.nodeBase + this.lodNodeCount + particleCount;  // + the analytic lamp's marker + emissive Gaussians (+ LOD nodes) (+ particles)
    this.splatCount = n;
    const buf = (label, size, usage) => device.createBuffer({ label, size, usage });

    this.frameBuf = buf('frame', FRAME_BYTES, B.UNIFORM | B.COPY_DST);
    // Gaussian particles live in the same buffer, after the LOD node records at a 256 B aligned offset (particles.wgsl
    // writes them there, project.wgsl reads them as gaussians[frame.particles.z + k]): no extra binding in project,
    // which is at the 10 storage buffers per stage some GPUs allow
    const gaussBytes = asset.gaussians.byteLength + (lodNodes ? lodNodes.records.byteLength : 0);
    this.particleOffset = Math.ceil(gaussBytes / 256) * 256;
    this.gaussBuf = buf('gaussians', this.particleOffset + particleCount * 64, B.STORAGE | B.COPY_DST | B.COPY_SRC);
    device.queue.writeBuffer(this.gaussBuf, 0, asset.gaussians);
    if (lodNodes) device.queue.writeBuffer(this.gaussBuf, asset.gaussians.byteLength, lodNodes.records);
    this.lodBuf = buf('lod flags', Math.max(16, (lodNodes ? lodNodes.N : 0) * 4), B.STORAGE | B.COPY_SRC);   // lod.wgsl lod_select
    this.partBuf = buf('parts', asset.parts.byteLength, B.STORAGE | B.COPY_DST);
    this.particles = null;   // main.js: a Particles system steps them at the start of encode
    device.queue.writeBuffer(this.partBuf, 0, asset.parts);
    this.splatBuf = buf('splats', n * SPLAT_BYTES, B.STORAGE);
    this.keyBuf = buf('keys', n * 4, B.STORAGE);
    this.histBuf = buf('histogram', NUM_BUCKETS * 4, B.STORAGE | B.COPY_DST);
    this.orderBuf = buf('order', n * 4, B.STORAGE);
    this.drawBuf = buf('draw args', 16, B.STORAGE | B.INDIRECT | B.COPY_SRC);
    this.envBuf = buf('environment (none)', 416, B.UNIFORM | B.COPY_DST);   // SPEC 2.9; bindEnv swaps in lighting's
    this.skyBgBuf = buf('sky background (none)', (81 + 2 + 8 + 2 + 2) * 16, B.UNIFORM | B.COPY_DST);   // display.wgsl SkyBg; bindEnv swaps in the environment's
    this.lensBuf = buf('lenses', LENS_BYTES, B.UNIFORM | B.COPY_DST);   // ?refract=1 (refract.js)
    this.exitBuf = null;   // E: per Gaussian exit point / direction (exitmap.wgsl), with skinBuf allocated on E's first frame (ensureExit)
    this.glassCount = 0;   // E: setGlass builds the glass skin list
    this.inrayBuf = null;   // G: per Gaussian inside legs (exitmap.wgsl trace), 48 B each, on G's first frame (ensureExit)
    this.innerCount = 0;
    this.innerMaskBuf = buf('inside mask', Math.max(4, Math.ceil(asset.count / 32) * 4), B.STORAGE | B.COPY_DST);   // G: a bit per Gaussian inside a lens (project.wgsl leaves it undrawn)
    // E miss fallback: the lighting's probe params and SH2 (setGlass); until then a lattice of none (dims 0)
    this.noProbes = { params: device.createBuffer({ label: 'no probes', size: 5008, usage: B.UNIFORM }), sh: device.createBuffer({ label: 'no probe SH', size: 144, usage: B.STORAGE }) };
    this.refract = 0;   // main.js sets it each frame: refract.js MODES bits (1 A lens, 2 B gradient: layered splat + composite; 4 C: project only)

    const C = S.COMPUTE, VF = S.VERTEX | S.FRAGMENT;
    const L = {
      project: layout(device, 'project', [[C, 'uniform'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage'], [C, 'storage'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'uniform'], [C, 'read-only-storage']]),   // 10: lenses (C warp), 11: G inside mask
      scan: layout(device, 'scan', [[C, 'storage'], [C, 'storage']]),
      scatter: layout(device, 'scatter', [[C, 'uniform'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage']]),
      splat: layout(device, 'splat', [[VF, 'uniform'], [S.VERTEX, 'read-only-storage'], [S.VERTEX, 'read-only-storage']]),
      display: layout(device, 'display', [[S.FRAGMENT, 'uniform'], [S.FRAGMENT, 'texture'], [S.FRAGMENT, 'uniform'], [S.FRAGMENT, 'uniform']]),   // 2: the environment, 3: the background sky (SkyBg)
      splatLayers: layout(device, 'splat layers', [[VF, 'uniform'], [S.VERTEX, 'read-only-storage'], [S.VERTEX, 'read-only-storage'], [S.FRAGMENT, 'uniform'], [S.FRAGMENT, 'texture']]),
      lensmap: layout(device, 'lensmap', [[S.FRAGMENT, 'uniform'], [S.FRAGMENT, 'uniform']]),
      refract: device.createBindGroupLayout({
        label: 'refract',
        entries: [
          { binding: 0, visibility: S.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: S.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 2, visibility: S.FRAGMENT, sampler: { type: 'filtering' } },
          ...[3, 4, 5, 6, 7].map(binding => ({ binding, visibility: S.FRAGMENT, texture: { sampleType: 'float' } })),
          { binding: 8, visibility: S.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
          ...[9, 10].map(binding => ({ binding, visibility: S.FRAGMENT, texture: { sampleType: 'unfilterable-float' } })),   // E: exit map
          { binding: 11, visibility: S.FRAGMENT, buffer: { type: 'uniform' } },             // E miss fallback: probe lattice
          { binding: 12, visibility: S.FRAGMENT, buffer: { type: 'read-only-storage' } },   //   and the probes' SH2
          { binding: 13, visibility: S.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },   // E: miss confidence map
          { binding: 14, visibility: S.FRAGMENT, buffer: { type: 'read-only-storage' } },   // E: probe gain (refract_calib)
          ...[15, 16, 17].map(binding => ({ binding, visibility: S.FRAGMENT, texture: { sampleType: 'unfilterable-float' } })),   // G: inside ray map
          ...[18, 20].map(binding => ({ binding, visibility: S.FRAGMENT, buffer: { type: 'read-only-storage' } })),   // G: splats, posed inside Gaussians
        ],
      }),
      exittrace: layout(device, 'exit trace', [[C, 'uniform'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'uniform'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage'], [C, 'storage']]),   // 7: G inside rays
      exitinner: device.createBindGroupLayout({ label: 'exit inner', entries: [[0, 'uniform'], [1, 'read-only-storage'], [2, 'read-only-storage'], [8, 'read-only-storage'], [9, 'storage']].map(([binding, type]) => ({ binding, visibility: C, buffer: { type } })) }),   // G: prep_inner
      exitmap: device.createBindGroupLayout({
        label: 'exit map',
        entries: [
          { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
          { binding: 1, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 5, visibility: VF, buffer: { type: 'read-only-storage' } },
          { binding: 6, visibility: VF, buffer: { type: 'read-only-storage' } },   // G: inside rays
        ],
      }),
      radix: layout(device, 'radix', [[C, 'uniform'], [C, 'uniform'], [C, 'read-only-storage'], [C, 'read-only-storage'], [C, 'storage'], [C, 'storage'], [C, 'storage']]),
    };
    this.layouts = L;
    this.shaders = shaders;
    this.stableSort = true;   // on by default; main.js sets it from the View panel each frame; the radix resources are built on the first stable encode
    const module = name => device.createShaderModule({ label: name, code: shaders.common + '\n' + (name === 'project' ? shaders.lens + '\n' : '') + shaders[name] });
    const compute = name => device.createComputePipeline({
      label: name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [L[name]] }),
      compute: { module: module(name), entryPoint: 'main' },
    });
    this.pipelines = { project: compute('project'), scan: compute('scan'), scatter: compute('scatter') };

    // splat carries the lens helpers for fs_layers (?refract=1); fs ignores them
    const splatModule = device.createShaderModule({ label: 'splat', code: shaders.common + '\n' + shaders.lens + '\n' + shaders.splat });
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
    // A: three "over" layers; B: the same targets, the inside one summing the glass gradient instead
    const layersPipe = (insideBlend) => device.createRenderPipeline({
      label: 'splat layers',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.splatLayers] }),
      vertex: { module: splatModule, entryPoint: 'vs' },
      fragment: { module: splatModule, entryPoint: 'fs_layers', targets: [
        ...[over, insideBlend, over].map(b => ({ format: HDR_FORMAT, blend: { color: b, alpha: b } })),
        ...[0, 1].map(() => ({ format: LAYER_DEPTH_FORMAT, blend: { color: over, alpha: over } })),
      ] },
      primitive: { topology: 'triangle-strip' },
    });
    this.pipelines.splatLayers = layersPipe(over);
    this.pipelines.splatLayersGrad = layersPipe(add);
    const compactPipe = (auxFormat, auxBlend) => device.createRenderPipeline({
      label: 'splat layers (compact)',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.splatLayers] }),
      vertex: { module: splatModule, entryPoint: 'vs' },
      fragment: { module: splatModule, entryPoint: 'fs_layers2', targets: [
        { format: HDR_FORMAT, blend: { color: over, alpha: over } }, { format: HDR_FORMAT, blend: { color: over, alpha: over } },
        { format: auxFormat, blend: { color: auxBlend, alpha: auxBlend } }] },
      primitive: { topology: 'triangle-strip' },
    });
    this.pipelines.splatCompactLens = compactPipe(LAYER_DEPTH_FORMAT, over);
    this.pipelines.splatCompactGrad = compactPipe(GRAD_FORMAT, add);
    this.fastLayers = true;   // main.js: ?layers=full keeps the five-target path for A + C / B
    const gainDecl = rw => `@group(0) @binding(14) var<storage, ${rw}> probe_gain: array<vec4f>;\n`;
    const [refractMain] = shaders.refract.split('// --- refract_calib');
    // the frame's shared ray grid (raygrid.js): the exit trace and G's inside Gaussians fill it, the trace and the composite's gather walk it
    this.rayGrid = new RayGrid(device, shaders);
    const refractModule = device.createShaderModule({ label: 'refract', code: shaders.common + '\n' + shaders.lens + '\n' + shaders.hgrid + '\n' + gainDecl('read') + refractMain });
    // E miss fallback: refract_calib levels the probes to the screen once a frame (refract.wgsl)
    const calibModule = device.createShaderModule({ label: 'refract calib', code: shaders.common + '\n' + shaders.lens + '\n' + shaders.hgrid + '\n' + gainDecl('read_write') + shaders.refract });
    this.pipelines.refractCalib = device.createComputePipeline({ label: 'refract calib', layout: 'auto', compute: { module: calibModule, entryPoint: 'refract_calib' } });
    this.probeGainBuf = device.createBuffer({ label: 'probe gain (E miss fallback)', size: 16, usage: B.STORAGE });
    this.pipelines.refract = device.createRenderPipeline({
      label: 'refract',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.refract, this.rayGrid.layout] }),
      vertex: { module: refractModule, entryPoint: 'vs' },
      fragment: { module: refractModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    // E: per glass Gaussian exit trace (exitmap.wgsl prep + trace) and the additive exit map (splat.wgsl vs_exit / fs_exit)
    const exitModule = device.createShaderModule({ label: 'exitmap', code: shaders.common + '\n' + shaders.lens + '\n' + shaders.hgrid + '\n' + shaders.exitmap });
    const exitLayout = device.createPipelineLayout({ bindGroupLayouts: [L.exittrace, this.rayGrid.layout] });
    this.pipelines.exitPrep = device.createComputePipeline({ label: 'exit prep', layout: exitLayout, compute: { module: exitModule, entryPoint: 'prep' } });
    this.pipelines.exitTrace = device.createComputePipeline({ label: 'exit trace', layout: exitLayout, compute: { module: exitModule, entryPoint: 'trace' } });
    this.pipelines.exitInner = device.createComputePipeline({ label: 'exit inner', layout: device.createPipelineLayout({ bindGroupLayouts: [L.exitinner, this.rayGrid.layout] }), compute: { module: exitModule, entryPoint: 'prep_inner' } });
    this.pipelines.exitMap = device.createRenderPipeline({
      label: 'exit map',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.exitmap] }),
      vertex: { module: splatModule, entryPoint: 'vs_exit' },
      fragment: { module: splatModule, entryPoint: 'fs_exit', targets: [0, 1, 2].map(() => ({ format: HDR_FORMAT, blend: { color: add, alpha: add } })) },
      primitive: { topology: 'triangle-strip' },
    });
    this.pipelines.exitIn = device.createRenderPipeline({   // G: the inside ray map, same splats and weights as the exit map
      label: 'inside ray map',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.exitmap] }),
      vertex: { module: splatModule, entryPoint: 'vs_exit' },
      fragment: { module: splatModule, entryPoint: 'fs_exitin', targets: [0, 1, 2].map(() => ({ format: HDR_FORMAT, blend: { color: add, alpha: add } })) },
      primitive: { topology: 'triangle-strip' },
    });
    this.pipelines.lensmap = device.createRenderPipeline({
      label: 'lensmap',
      layout: device.createPipelineLayout({ bindGroupLayouts: [L.lensmap] }),
      vertex: { module: refractModule, entryPoint: 'vs' },
      fragment: { module: refractModule, entryPoint: 'fs_lensmap', targets: [{ format: LENSMAP_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    this.groups_lensmap = null;
    this.linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
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
      [this.frameBuf, this.gaussBuf, this.partBuf, this.splatBuf, this.keyBuf, this.histBuf, colourBuf, bounceBuf, emitterBuf, this.lodBuf, this.lensBuf, this.innerMaskBuf]);
    this.groupsLit = { colour: this.groups.project };
  }

  // Camera-only flake colour (flake.js): project reads `cam` in colour's slot; RenderFrame lamp.w = 1 drops the bounce
  // add (cam already holds it). use(true / false) picks the group for the next encode.
  bindFlake(camBuf, bounceBuf, emitterBuf) {
    this.groupsLit.flake = this.group(this.layouts.project,
      [this.frameBuf, this.gaussBuf, this.partBuf, this.splatBuf, this.keyBuf, this.histBuf, camBuf, bounceBuf, emitterBuf, this.lodBuf, this.lensBuf, this.innerMaskBuf]);
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
    this.groups.display = this.group(this.layouts.display, [this.frameBuf, this.hdr, this.envBuf, this.skyBgBuf]);
    if (this.layers) this.buildLayers();
  }

  // ?refract=1: the three colour layers and two depth layers at the hdr target's size, built on the first refract frame
  buildLayers() {
    for (const t of this.layers ?? []) t.destroy();
    const { width, height } = this.hdr;
    const tex = format => this.device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    // 0 front, 1 inside, 2 behind, 3 inside depth, 4 behind depth (compact A + C: its aux), 5 compact B gradient, 6 lens map
    this.layers = [tex(HDR_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT), tex(LAYER_DEPTH_FORMAT), tex(LAYER_DEPTH_FORMAT), tex(GRAD_FORMAT), tex(LENSMAP_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT), tex(HDR_FORMAT)];   // 7, 8: E exit map, 9: its miss confidence; 10-12: G inside ray map
    const refractGroup = slots => this.device.createBindGroup({
      layout: this.layouts.refract,
      entries: [{ binding: 0, resource: { buffer: this.frameBuf } }, { binding: 1, resource: { buffer: this.lensBuf } },
        { binding: 2, resource: this.linear }, ...slots.map((k, j) => ({ binding: 3 + j, resource: this.layers[k].createView() })),
        { binding: 11, resource: { buffer: this.probeBufs?.params ?? this.noProbes.params } }, { binding: 12, resource: { buffer: this.probeBufs?.sh ?? this.noProbes.sh } },
        { binding: 13, resource: this.layers[9].createView() }, { binding: 14, resource: { buffer: this.probeGainBuf } },
        ...[10, 11, 12].map((k, j) => ({ binding: 15 + j, resource: this.layers[k].createView() })),
        { binding: 18, resource: { buffer: this.splatBuf } }, { binding: 20, resource: { buffer: this.innerBuf ?? this.noProbes.sh } }],
    });
    // refract_calib: its auto layout holds only what it reads
    this.groups.refractCalib = this.device.createBindGroup({
      layout: this.pipelines.refractCalib.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.frameBuf } }, { binding: 1, resource: { buffer: this.lensBuf } }, { binding: 2, resource: this.linear },
        ...[[3, 0], [5, 2], [7, 4], [8, 6], [9, 7], [10, 8], [13, 9]].map(([binding, k]) => ({ binding, resource: this.layers[k].createView() })),
        { binding: 11, resource: { buffer: this.probeBufs?.params ?? this.noProbes.params } }, { binding: 12, resource: { buffer: this.probeBufs?.sh ?? this.noProbes.sh } },
        { binding: 14, resource: { buffer: this.probeGainBuf } }],
    });
    this.groups.refract = refractGroup([0, 1, 2, 3, 4, 6, 7, 8]);
    this.groups.refractGrad = refractGroup([0, 5, 2, 3, 4, 6, 7, 8]);
    this.groups.exitmap = null;   // ensureExit
    this.groups.splatLayers = this.group(this.layouts.splatLayers, [this.frameBuf, this.splatBuf, this.orderBuf, this.lensBuf, this.layers[6]]);
    this.groups.lensmap = this.group(this.layouts.lensmap, [this.frameBuf, this.lensBuf]);
  }

  // E: the glass skin Gaussians of each lens part, grouped by lens (refract.js fitLenses order): [count, (start,
  // count) x 8, part x 8, then Gaussian index | lens << 24] (exitmap.wgsl `glass`). The trace walks the asset tree
  // (tree: the lighting's nodes / kids buffers).
  // tree (the lighting buffers) is no longer read: the trace narrows its candidates with a per-frame grid (hgrid.wgsl)
  setGlass(lenses, asset, tree) {
    if (tree?.probe_params && tree?.probe_sh) {   // E miss fallback (refract.wgsl probe_radiance)
      this.probeBufs = { params: tree.probe_params, sh: tree.probe_sh };
      if (this.layers) this.buildLayers();
    }
    const glassMats = new Set((asset.header.materials ?? []).filter(m => m.coverage !== undefined).map(m => m.index));
    const u = new Uint32Array(asset.gaussians), f = new Float32Array(asset.gaussians);
    const head = new Uint32Array(27), list = [];
    lenses.forEach((L, k) => {
      const [lo, hi] = asset.header.parts[L.part].gaussians;
      head[1 + 2 * k] = list.length;
      for (let i = lo; i < hi; i++) if (glassMats.has(u[i * 16 + 15] & 0xffff)) list.push(i | (k << 24));
      head[2 + 2 * k] = list.length - head[1 + 2 * k];
      head[17 + k] = L.part;
    });
    head[0] = list.length;
    this.glassCount = list.length;
    // [25] the widest sigma_t (the widest beam is half of it), [26] the skin's slots in the ray grid
    head[25] = new Uint32Array(new Float32Array([Math.max(0, ...list.map(e => (e & 0xffffff) * 16).map(o => Math.sqrt(f[o + 9] * f[o + 10])))]).buffer)[0];
    if (this.glassSlots?.count !== list.length) this.glassSlots = { base: this.rayGrid.alloc(list.length), count: list.length };
    head[26] = this.glassSlots.base;
    const data = new Uint32Array(27 + Math.max(1, list.length)); data.set(head); data.set(list, 27);
    this.glassBuf = this.device.createBuffer({ label: 'glass skin', size: data.byteLength, usage: B.STORAGE | B.COPY_DST });
    this.device.queue.writeBuffer(this.glassBuf, 0, data);
    this.groups.exittrace = null;   // ensureExit
    // G: the Gaussians inside each lens part (not glass, not emitting), [count, (start, count) x 8, ray grid base slot,
    // then Gaussian | lens << 24] (exitmap.wgsl inner_list)
    const emitMats = new Set((asset.header.materials ?? []).filter(m => m.emission).map(m => m.index));
    const ihead = new Uint32Array(18), ilist = [];
    lenses.forEach((L, k) => {
      const [lo, hi] = asset.header.parts[L.part].gaussians;
      ihead[1 + 2 * k] = ilist.length;
      for (let i = lo; i < hi; i++) { const m = u[i * 16 + 15] & 0xffff; if (!glassMats.has(m) && !emitMats.has(m)) ilist.push(i | (k << 24)); }
      ihead[2 + 2 * k] = ilist.length - ihead[1 + 2 * k];
    });
    ihead[0] = ilist.length;
    this.innerCount = ilist.length;
    if (this.innerSlots?.count !== ilist.length) this.innerSlots = { base: this.rayGrid.alloc(ilist.length), count: ilist.length };
    ihead[17] = this.innerSlots.base;
    const mask = new Uint32Array(Math.ceil(asset.count / 32));
    for (const e of ilist) { const i = e & 0xffffff; mask[i >> 5] |= 1 << (i & 31); }
    this.device.queue.writeBuffer(this.innerMaskBuf, 0, mask);
    const idata = new Uint32Array(18 + Math.max(1, ilist.length)); idata.set(ihead); idata.set(ilist, 18);
    this.innerListBuf = this.device.createBuffer({ label: 'inside list', size: idata.byteLength, usage: B.STORAGE | B.COPY_DST });
    this.device.queue.writeBuffer(this.innerListBuf, 0, idata);
    this.innerBuf = this.device.createBuffer({ label: 'inside Gaussians (view)', size: Math.max(48, ilist.length * 48), usage: B.STORAGE });
    this.groups.exitinner = this.device.createBindGroup({ layout: this.layouts.exitinner, entries: [[0, this.frameBuf], [1, this.gaussBuf], [2, this.partBuf], [8, this.innerListBuf], [9, this.innerBuf]].map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
    if (this.layers) this.buildLayers();   // the composite's group holds innerBuf
  }

  // E's per Gaussian buffers (exits and the glass skin in view space, 32 B each per Gaussian: ~72 MB on hill_bin) and
  // the groups that use them, on the first frame E runs rather than at load
  // G's inside legs (48 B per Gaussian) likewise on G's first frame; until then the trace writes none (REFR_GATHER off)
  ensureExit(gather) {
    if (!this.exitBuf) {
      const n = Math.max(32, this.count * 32);
      this.exitBuf = this.device.createBuffer({ label: 'exits', size: n, usage: B.STORAGE });
      this.skinBuf = this.device.createBuffer({ label: 'glass skin (view)', size: n, usage: B.STORAGE });   // per Gaussian, zero when not glass
      this.noInray = this.device.createBuffer({ label: 'inside rays (none)', size: 48, usage: B.STORAGE });
    }
    if (gather && !this.inrayBuf) {
      this.inrayBuf = this.device.createBuffer({ label: 'inside rays', size: Math.max(48, this.count * 48), usage: B.STORAGE });
      this.groups.exittrace = this.groups.exitmap = null;
    }
    const inray = this.inrayBuf ?? this.noInray;
    if (!this.groups.exittrace) this.groups.exittrace = this.group(this.layouts.exittrace, [this.frameBuf, this.gaussBuf, this.partBuf, this.lensBuf, this.glassBuf, this.skinBuf, this.exitBuf, inray]);
    if (!this.groups.exitmap) this.groups.exitmap = this.device.createBindGroup({ layout: this.layouts.exitmap, entries: [[0, this.frameBuf], [1, this.splatBuf], [2, this.orderBuf], [5, this.exitBuf], [6, inray]].map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  }

  writeLenses(bytes) {
    this.device.queue.writeBuffer(this.lensBuf, 0, bytes);
  }

  // The environment uniform (lighting.buffers.env, SPEC 2.9) and the background sky (environment.js bgBuf) behind the splats.
  bindEnv(envBuf, skyBgBuf = null) {
    this.envBuf = envBuf;
    if (skyBgBuf) this.skyBgBuf = skyBgBuf;
    if (this.hdr) this.groups.display = this.group(this.layouts.display, [this.frameBuf, this.hdr, this.envBuf, this.skyBgBuf]);
  }

  // Per-frame part records (SPEC §2.4): the whole buffer is a few hundred bytes.
  writeParts(partBytes) {
    this.device.queue.writeBuffer(this.partBuf, 0, partBytes);
  }

  encode(encoder, target, frameBytes, viewMode, timers) {
    this.device.queue.writeBuffer(this.frameBuf, 0, frameBytes);
    encoder.clearBuffer(this.histBuf);
    this.particles?.encode(encoder);
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

    const layered = (this.refract & 3) !== 0 && viewMode !== 2;
    const lens = (this.refract & 1) !== 0;
    const compact = layered && this.fastLayers && this.refract !== 1;   // A alone keeps its inside layer
    if (layered && !this.layers) this.buildLayers();
    const clear = view => ({ view: view.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' });
    let pass;
    if (layered && lens) {
      pass = encoder.beginRenderPass({ label: 'lensmap', colorAttachments: [clear(this.layers[6])], timestampWrites: timers.writes('lensmap') });
      pass.setPipeline(this.pipelines.lensmap);
      pass.setBindGroup(0, this.groups.lensmap);
      pass.draw(3);
      pass.end();
    } else if (timers.ms) timers.ms.lensmap = 0;
    const exit = layered && lens && (this.refract & 8) !== 0 && this.glassCount > 0;
    const gather = exit && (this.refract & 16) !== 0;
    if (exit) {
      this.ensureExit(gather);
      // timed apart: the per-frame grid build (shared by any later consumer) and the trace that walks it
      pass = encoder.beginComputePass({ label: 'hgrid build', timestampWrites: timers.writes('hgbuild') });
      const n64 = Math.ceil(this.glassCount / 64), grid = this.rayGrid;
      grid.encodeClear(pass);
      pass.setBindGroup(0, this.groups.exittrace);
      pass.setPipeline(this.pipelines.exitPrep); pass.dispatchWorkgroups(n64);   // the skin into its grid slots
      if (gather && this.innerCount > 0) {   // G: the inside Gaussians posed and into their grid slots
        pass.setBindGroup(0, this.groups.exitinner);
        pass.setPipeline(this.pipelines.exitInner); pass.dispatchWorkgroups(Math.ceil(this.innerCount / 64));
      }
      grid.encodeBuild(pass);
      pass.end();
      pass = encoder.beginComputePass({ label: 'exit trace', timestampWrites: timers.writes('exittrace') });
      pass.setBindGroup(0, this.groups.exittrace);
      pass.setBindGroup(1, grid.group);
      pass.setPipeline(this.pipelines.exitTrace); pass.dispatchWorkgroups(n64);
      pass.end();
    } else if (timers.ms) { timers.ms.exittrace = 0; timers.ms.hgbuild = 0; }
    const targets = !layered ? [this.hdr] : !compact ? this.layers.slice(0, 5) : [0, 2, lens ? 4 : 5].map(k => this.layers[k]);
    pass = encoder.beginRenderPass({
      label: 'splat',
      colorAttachments: targets.map(clear),
      timestampWrites: timers.writes('splat'),
    });
    pass.setPipeline(!layered ? (viewMode === 2 ? this.pipelines.splatAdd : this.pipelines.splatOver)
      : compact ? (lens ? this.pipelines.splatCompactLens : this.pipelines.splatCompactGrad)
      : lens ? this.pipelines.splatLayers : this.pipelines.splatLayersGrad);
    pass.setBindGroup(0, layered ? this.groups.splatLayers : this.groups.splat);
    pass.drawIndirect(this.drawBuf, 0);
    pass.end();
    if (exit) {
      pass = encoder.beginRenderPass({ label: 'exit map', colorAttachments: [clear(this.layers[7]), clear(this.layers[8]), clear(this.layers[9])], timestampWrites: timers.writes('exitmap') });
      pass.setPipeline(this.pipelines.exitMap);
      pass.setBindGroup(0, this.groups.exitmap);
      pass.drawIndirect(this.drawBuf, 0);
      pass.end();
      const cp = encoder.beginComputePass({ label: 'refract calib' });
      cp.setPipeline(this.pipelines.refractCalib);
      cp.setBindGroup(0, this.groups.refractCalib);
      cp.dispatchWorkgroups(1);
      cp.end();
      if (gather) {
        pass = encoder.beginRenderPass({ label: 'inside ray map', colorAttachments: [10, 11, 12].map(k => clear(this.layers[k])) });
        pass.setPipeline(this.pipelines.exitIn);
        pass.setBindGroup(0, this.groups.exitmap);
        pass.drawIndirect(this.drawBuf, 0);
        pass.end();
      }
    } else if (timers.ms) timers.ms.exitmap = 0;
    if (layered) {
      pass = encoder.beginRenderPass({ label: 'refract', colorAttachments: [clear(this.hdr)], timestampWrites: timers.writes('refract') });
      pass.setPipeline(this.pipelines.refract);
      pass.setBindGroup(0, compact && !lens ? this.groups.refractGrad : this.groups.refract);
      if (!this.rayGrid.group) this.rayGrid.alloc(0);
      pass.setBindGroup(1, this.rayGrid.group);   // G: the gather walks it
      pass.draw(3);
      pass.end();
    } else if (timers.ms) timers.ms.refract = 0;

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
