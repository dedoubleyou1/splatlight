// Ray grid: one hashed grid over Gaussians per frame, shared by every system that intersects rays with Gaussians in
// that frame (renderer/shaders/hgrid.wgsl; experiments/ray_intersect). Producers take slot ranges (alloc) at setup,
// write their items in their own compute passes between encodeClear and encodeBuild, and consumers prepend
// hgrid.wgsl, bind `group` at @group(1) (read it each frame: alloc can reallocate) and filter candidates by tag.
// Nothing is cached between frames.

const B = GPUBufferUsage;
const MAX_TABLE = 262144;   // hg_scan_top scans one total per 1024 slots in a single 256-thread workgroup

export class RayGrid {
  constructor(device, shaders) {
    this.device = device;
    this.used = 0;
    this.capacity = 0;
    this.layout = device.createBindGroupLayout({
      label: 'ray grid',
      entries: [0, 1, 2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: 'storage' } })),   // fragment: the refraction composite's gather (refract.wgsl)
    });
    const module = device.createShaderModule({ label: 'ray grid', code: shaders.hgrid });
    const empty = device.createBindGroupLayout({ label: 'empty', entries: [] });
    this.empty = device.createBindGroup({ label: 'empty', layout: empty, entries: [] });   // @group(0) of the build pipelines
    const layout = device.createPipelineLayout({ bindGroupLayouts: [empty, this.layout] });
    this.pipes = Object.fromEntries(['hg_clear', 'hg_count', 'hg_scan_blocks', 'hg_scan_top', 'hg_scan_add', 'hg_fill'].map(e => [e,
      device.createComputePipeline({ label: e, layout, compute: { module, entryPoint: e } })]));
    this.group = null;
  }

  // a producer's slots: items[base .. base + count). The hash table gets about 16 slots per item in use: an item fills
  // about 14 cells (glass skin: 192k entries for 14k items), and a walk pays for every same-level entry another cell
  // hashed into its slot, empty cells included (experiments/ray_intersect: 65k slots put ~3 strangers in every slot).
  alloc(count) {
    const base = this.used;
    this.used += count;
    if (this.used > this.capacity || !this.hg) this.resize(Math.max(this.used, 2 * this.capacity, 1));   // alloc(0) too: no glass
    this.table = Math.min(MAX_TABLE, Math.max(1024, 2 ** Math.ceil(Math.log2(16 * this.used))));
    this.device.queue.writeBuffer(this.hg, 8 * 4, new Uint32Array([this.table]));
    return base;
  }

  resize(capacity) {
    for (const b of [this.items, this.keys, this.hg, this.refs]) b?.destroy();
    this.capacity = capacity;
    const buf = (label, size) => this.device.createBuffer({ label, size: Math.max(16, size), usage: B.STORAGE | B.COPY_DST });
    this.items = buf('ray grid items', capacity * 16);
    this.keys = buf('ray grid keys', capacity * 4);
    this.hg = buf('ray grid table', (16 + MAX_TABLE + 1 + MAX_TABLE / 1024) * 4);   // head, table + total, block sums
    this.refs = buf('ray grid refs', capacity * 27 * 4);   // <= 27 cells per item
    this.group = this.device.createBindGroup({ label: 'ray grid', layout: this.layout,
      entries: [this.items, this.keys, this.hg, this.refs].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  // in a compute pass, before the producers (both leave @group(0) empty: set yours again after)
  encodeClear(pass) {
    pass.setBindGroup(0, this.empty);
    pass.setBindGroup(1, this.group);
    pass.setPipeline(this.pipes.hg_clear);
    pass.dispatchWorkgroups(Math.ceil(Math.max(16 + this.table + 1, this.used) / 256));
  }

  // in a compute pass, after the producers: count, scan (three dispatches), fill
  encodeBuild(pass) {
    const n = Math.ceil(this.used / 64);
    pass.setBindGroup(0, this.empty);
    pass.setBindGroup(1, this.group);
    pass.setPipeline(this.pipes.hg_count); pass.dispatchWorkgroups(n);
    const blocks = this.table / 1024;
    pass.setPipeline(this.pipes.hg_scan_blocks); pass.dispatchWorkgroups(blocks);
    pass.setPipeline(this.pipes.hg_scan_top); pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipes.hg_scan_add); pass.dispatchWorkgroups(blocks);
    pass.setPipeline(this.pipes.hg_fill); pass.dispatchWorkgroups(n);
  }
}
