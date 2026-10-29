// The environment (runtime/SPEC.md 2.9): an SH sky plus distant lights (the sun) as ASGs at infinity.
//
// The asset's asset.json `environment` block, or ?env=<name> (runtime/assets/env/<name>.json), describes it. This
// writes the Env uniform (lighting.buffers.env) that the probes (sky on rays that leave the scene), the display pass
// (background) and runtime/wgsl/environment.wgsl (the distant lights' direct light and shadows) read.
//   env_shadow  only when a distant light or a part moved: a deterministic walk per Gaussian toward each light, in
//               slices of SHADOW_CHUNK Gaussians, one slice a frame (one dispatch over hill's 1.17M Gaussians froze
//               the Mac); after a movable part moves, static receivers re-walk only the movable parts (SPEC 2.4)
//   env_direct  every frame after shade_emit: the distant lights' direct light into colour
//   env_dirs    after direct_dirs(_emit): the same into lo_dir, so the sun bounces through the probes

import { SunMap } from './sun_map.js';

const ENV_BYTES = 416;
const ESH_BYTES = 80;   // environment.wgsl EnvShadow: first, count, mode, pad, static_mask[4] (vec4u)
// env_shadow Gaussians per frame (?envchunk=<n> overrides); a pass over hill (1.17M) then takes ~36 frames
const SHADOW_CHUNK = Math.max(1024, Number(new URLSearchParams(globalThis.location?.search ?? '').get('envchunk')) || 32768);
// rays per receiver and light (?envrays=1 the old centre ray): a bundle from points over the receiver Gaussian toward
// directions over the light's angular size (M45 E5: one centre ray speckles 25 cm soil Gaussians under grass blades;
// 16 fixes it). A slice then holds SHADOW_CHUNK x 4 / rays Gaussians, so a slice costs about the same as before.
const SHADOW_RAYS = Math.max(1, Math.min(32, Number(new URLSearchParams(globalThis.location?.search ?? '').get('envrays')) || 16));
const DEG = Math.PI / 180;

// asset.json environment -> { sh: [16][3], lights: [{ dir, irradiance, std, shadows }] }
export function parseEnvironment(env) {
  if (!env) return null;
  const sh = env.sky?.sh_rgb ?? [];
  const lights = (env.lights ?? []).slice(0, 4).map(L => ({
    dir: normalize(L.dir), irradiance: L.irradiance_rgb ?? [1, 1, 1],
    std: (L.angular_std_deg ?? 0.27) * DEG, shadows: L.casts_shadows ?? true,
  }));
  return { sh, lights };
}

export async function loadEnvironment(asset, params) {
  const name = (params.get('env') ?? '').replace(/[^A-Za-z0-9_-]/g, '');
  if (name === 'none') return null;
  if (name) {
    const r = await fetch(`runtime/assets/env/${name}.json`);
    if (!r.ok) throw new Error(`environment ${name}: HTTP ${r.status}`);
    return parseEnvironment(await r.json());
  }
  return parseEnvironment(asset.header.environment);
}

function normalize(v) { const l = Math.hypot(...v) || 1; return v.map(x => x / l); }

// direction toward the light from azimuth (degrees from +y toward -x, as M44's sky.py) and elevation
export function sunDir(az, el) {
  const a = az * DEG, e = el * DEG;
  return [-Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), Math.sin(e)];
}
export function sunAngles(d) {
  return [Math.atan2(-d[0], d[1]) / DEG, Math.asin(Math.max(-1, Math.min(1, d[2]))) / DEG];
}

export class Environment {
  // parts: asset.json parts[] (their `mobility`, SPEC 2.4); absent = every part movable
  constructor(device, sources, lighting, env, parts = [], asset = null) {
    this.device = device;
    this.lighting = lighting;
    this.base = env;
    this.skyGain = 1;
    this.lightGain = 1;
    this.sun = env.lights[0] ? sunAngles(env.lights[0].dir) : [0, 45];
    this.on = true;
    this.key = null;
    const module = device.createShaderModule({ label: 'environment',
      code: sources.common + '\n' + sources.material_ext + '\n' + sources.environment });
    const b = lighting.buffers;
    this.esh = device.createBuffer({ label: 'env_shadow slice', size: ESH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.eshBytes = new ArrayBuffer(ESH_BYTES);
    const mask = new Uint32Array(this.eshBytes, 16, 16);
    parts.forEach((p, i) => { if (p.mobility === 'static' && i < 512) mask[i >> 5] |= 1 << (i & 31); });
    this.anyStatic = parts.some(p => p.mobility === 'static');
    this.next = lighting.count;   // next Gaussian of the running shadow pass (count = none running)
    this.mode = 0;                // the running pass: 0 every occluder, 1 movable occluders for static receivers
    this.pendingMove = false;     // a movable part moved during the running pass: another mode 1 pass follows it
    const res = { 0: b.uniform, 1: b.gaussians, 2: b.nodes, 3: b.vgrid, 4: b.parts, 7: b.materials, 8: b.colour,
      9: b.kids, 13: b.lo_dir, 40: b.env, 41: b.env_T, 42: this.esh, 61: lighting.lodFlagBuf, 62: lighting.lodList };
    // motion-blur shadows from a smear-sized cut (lighting.js blurCut; ?blurcut=0 off): (shutter, k), rest nodes
    const bce = device.createBuffer({ label: 'env blur cut', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bce, 0, new Float32Array([lighting.blur ? lighting.blurShutter : 0, lighting.blurCut ?? 0, 0, 0]));
    Object.assign(res, { 55: bce, 56: b.nodes });
    const binds = { env_shadow: [0, 1, 2, 4, 9, 40, 41, 42, 55, 56], env_direct: [0, 1, 2, 3, 4, 7, 8, 40, 41],
      env_dirs: [0, 1, 2, 3, 4, 7, 13, 40, 41] };
    // screen-space LOD lighting (lighting.js lodLight / lodCut): drawn nodes, and with the cut the drawn leaves only
    if (lighting.lodLight) binds.env_direct_nodes = [0, 1, 2, 3, 4, 7, 8, 40, 41, 61];
    if (lighting.lodCut) Object.assign(binds, { env_direct_list: [0, 1, 2, 3, 4, 7, 8, 40, 41, 62],
      env_direct_nodes_list: [0, 1, 2, 3, 4, 7, 8, 40, 41, 62] });
    this.k = {};
    for (const [entry, list] of Object.entries(binds)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      // motion blur (lighting.js blur_records): the sun-shadow walk sees moving occluders' stretched nodes; the
      // receiver keeps its rest record (gaussians), as the lamp's walk_emit does
      const nodesFor = entry === 'env_shadow' && b.nodes_blur ? b.nodes_blur : res[2];
      this.k[entry] = { pipe, group: device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: list.map(n => ({ binding: n, resource: { buffer: n === 2 ? nodesFor : res[n] } })) }) };
    }
    // one-frame shadows from movable parts (sun_map.js); the walk then covers static occluders only (ESH.mode 2), and
    // a moving part's own receivers re-walk them each frame it moves (one env_shadow over its Gaussians)
    this.sunMap = asset ? new SunMap(device, sources, lighting, this, asset) : null;
    if (this.sunMap && !this.sunMap.ok) this.sunMap = null;
    if (this.sunMap) {
      this.partWalks = this.sunMap.movable.map(p => {
        const [lo, hi] = parts[p].gaussians;
        const bytes = this.eshBytes.slice(0);
        new Uint32Array(bytes, 0, 4).set([lo, hi - lo, 2, SHADOW_RAYS]);
        const eb = device.createBuffer({ label: `env_shadow ${parts[p].name}`, size: ESH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(eb, 0, bytes);
        const k = this.k.env_shadow;
        return { p, count: hi - lo, group: device.createBindGroup({ layout: k.pipe.getBindGroupLayout(0),
          entries: binds.env_shadow.map(n => ({ binding: n, resource: { buffer: n === 42 ? eb : n === 2 ? (b.nodes_blur ?? res[2]) : res[n] } })) }) };
      });
    }
    this.stale = true;
    this.write();
    lighting.env = this;
  }

  // UI: sun azimuth / elevation (degrees) move the first distant light; gains scale the sky and the lights
  // returns true when anything changed (the frame loop then treats the light as changed)
  set({ on = this.on, sunAz = this.sun[0], sunEl = this.sun[1], skyGain = this.skyGain, lightGain = this.lightGain }) {
    this.on = on; this.sun = [sunAz, sunEl]; this.skyGain = skyGain; this.lightGain = lightGain;
    return this.write();
  }

  write() {
    const key = [this.on, ...this.sun, this.skyGain, this.lightGain].join();
    if (key === this.key) return false;
    const dirKey = [this.on, ...this.sun].join();
    if (dirKey !== this.dirKey) { this.stale = true; this.dirKey = dirKey; }
    this.key = key;
    const ab = new ArrayBuffer(ENV_BYTES), f = new Float32Array(ab), u = new Uint32Array(ab);
    const { sh, lights } = this.base;
    if (this.on) {
      sh.forEach((c, k) => f.set(c.map(v => v * this.skyGain), 4 * k));
      lights.forEach((L, k) => {
        const dir = k === 0 ? sunDir(...this.sun) : L.dir;
        f.set([...dir, L.std], 64 + 8 * k);
        f.set([...L.irradiance.map(v => v * this.lightGain), L.shadows ? 1 : 0], 68 + 8 * k);
      });
      u[96] = lights.length; u[97] = sh.length ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.lighting.buffers.env, 0, ab);
    this.lighting.firstGather = true;   // the probes re-gather the new sky
    return true;
  }

  // the first distant light's direction and radiance peak, for the display (sun disc)
  lights() {
    if (!this.on) return [];
    return this.base.lights.map((L, k) => ({ dir: k === 0 ? sunDir(...this.sun) : L.dir, std: L.std,
      irradiance: L.irradiance.map(v => v * this.lightGain) }));
  }

  // threads: a count, or a byte offset into lighting.lodList's indirect sizes ({ indirect })
  run(encoder, entry, threads) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.k[entry].pipe); pass.setBindGroup(0, this.k[entry].group);
    if (threads.indirect !== undefined) pass.dispatchWorkgroupsIndirect(this.lighting.lodList, threads.indirect);
    else pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / 64)));
    pass.end();
  }

  // drawn LOD nodes, after emissive.wgsl shade_emit_nodes (lighting.js encodeNodeDirect)
  encodeNodes(encoder) {
    if (!this.on || !this.base.lights.length) return;
    if (this.lighting.lodCut) this.run(encoder, 'env_direct_nodes_list', { indirect: 32 });
    else if (this.k.env_direct_nodes) this.run(encoder, 'env_direct_nodes', this.lighting.lodN);
  }

  // true while a shadow pass is still running (main.js keeps rendering frames until it ends)
  get busy() { return this.on && this.base.lights.length > 0 && (this.stale || this.next < this.lighting.count); }

  // after shade_emit: one slice of shadows while a pass runs, then the direct light. The sun moving restarts a full
  // pass; a part moving (lighting.movedMask) queues a pass over the movable occluders. Until a slice reaches a
  // Gaussian it keeps its last T (1 on load: unshadowed), so shadows fill in over a few frames.
  encodeDirect(encoder, timers, changed) {
    if (!this.on || !this.base.lights.length) return;
    if (this.sunMap) return this.encodeDirectMap(encoder);
    const n = this.lighting.count;
    const moved = changed && (this.lighting.movedMask ?? 1) !== 0;
    if (this.stale) { this.next = 0; this.mode = 0; this.pendingMove = false; this.stale = false; }
    else if (moved) {
      if (this.next < n) this.pendingMove = true;
      else { this.next = 0; this.mode = this.anyStatic ? 1 : 0; }
    }
    if (this.next < n) {
      const count = Math.min(Math.max(1024, Math.floor(SHADOW_CHUNK * 4 / SHADOW_RAYS)), n - this.next);
      new Uint32Array(this.eshBytes, 0, 4).set([this.next, count, this.mode, SHADOW_RAYS]);
      this.device.queue.writeBuffer(this.esh, 0, this.eshBytes);
      this.run(encoder, 'env_shadow', count);
      this.next += count;
      if (this.next >= n && this.pendingMove) { this.next = 0; this.mode = this.anyStatic ? 1 : 0; this.pendingMove = false; }
    }
    if (this.lighting.lodCut) this.run(encoder, 'env_direct_list', { indirect: 16 });   // ?lodlight=2: the cut
    else this.run(encoder, 'env_direct', n);
  }

  // with sun maps: the static walk in slices only after the light moves; each frame a movable part moved, its own
  // receivers re-walk the static occluders and the maps refresh; then env_T = static T x the maps' T
  encodeDirectMap(encoder) {
    const n = this.lighting.count, moved = (this.lighting.movedMask ?? 0) >>> 0;
    if (this.stale) { this.next = 0; this.stale = false; }
    let staticChanged = false;
    if (this.next < n) {
      const count = Math.min(Math.max(1024, Math.floor(SHADOW_CHUNK * 4 / SHADOW_RAYS)), n - this.next);
      new Uint32Array(this.eshBytes, 0, 4).set([this.next, count, 2, SHADOW_RAYS]);
      this.device.queue.writeBuffer(this.esh, 0, this.eshBytes);
      this.run(encoder, 'env_shadow', count);
      this.next += count;
      staticChanged = true;
    }
    for (const w of this.partWalks) {
      if (w.p > 31 || !((moved >>> w.p) & 1)) continue;
      const pass = encoder.beginComputePass({ label: 'env_shadow (moving part)' });
      pass.setPipeline(this.k.env_shadow.pipe); pass.setBindGroup(0, w.group);
      pass.dispatchWorkgroups(Math.ceil(w.count / 64)); pass.end();
      staticChanged = true;
    }
    this.sunMap.encode(encoder, moved, staticChanged);
    if (this.lighting.lodCut) this.run(encoder, 'env_direct_list', { indirect: 16 });   // ?lodlight=2: the cut
    else this.run(encoder, 'env_direct', n);
  }

  encodeDirs(encoder) {
    if (!this.on || !this.base.lights.length) return;
    const sh = this.lighting.iso && this.lighting.shProbes;
    this.run(encoder, 'env_dirs', sh ? Math.ceil(this.lighting.count / 4) : this.lighting.count);
  }
}
