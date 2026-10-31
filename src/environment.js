// The environment (runtime/SPEC.md 2.9): an SH sky plus distant lights (the sun) as ASGs at infinity.
//
// The asset's asset.json `environment` block, or ?env=<name> (runtime/assets/env/<name>.json), describes it. This
// writes the Env uniform (lighting.buffers.env) that the probes (sky on rays that leave the scene), the display pass
// (background) and runtime/wgsl/environment.wgsl (the distant lights' direct light and shadows) read.
//   shadows     sun_map.js: opacity maps per part and distant light, re-splatted for what moved, then a lookup per
//               Gaussian (the per-receiver walk, env_shadow, was archived 2026-10-29; its WGSL stays)
//   env_direct  every frame after shade_emit: the distant lights' direct light into colour
//   env_dirs    after direct_dirs(_emit): the same into lo_dir, so the sun bounces through the probes

import { SunMap } from './sun_map.js';

const ENV_BYTES = 416;
// display only (display.wgsl SkyBg): the background sky in up to 81 SH coefficients (bands 0-8), its coefficient count
// and horizon lobe count, per light the visible disc's angular std (0 = the lighting's), the horizon ring's 8 lobe
// colours, the ground colour + blend, the ring's widths, and the glow round the sun (irradiance + std, direction);
// 81 + 2 + 8 + 2 + 2 vec4
export const SKY_BG_BYTES = (81 + 2 + 8 + 2 + 2) * 16;
const DEG = Math.PI / 180;

// asset.json environment -> { sh: [16][3], bg: [n][3] | null, lights: [{ dir, irradiance, std, displayStd, shadows,
// followSun }] }
export function parseEnvironment(env) {
  if (!env) return null;
  const sh = env.sky?.sh_rgb ?? [];
  const bg = env.sky?.background_sh_rgb?.slice(0, 81) ?? null;
  const hz = env.sky?.horizon ?? null;
  const horizon = hz ? { rgb: hz.rgb.slice(0, 8), along: (hz.along_deg ?? 25) * DEG, across: (hz.across_deg ?? 8) * DEG } : null;
  const ground = env.sky?.ground_rgb ?? null;
  const g = env.sky?.glow ?? null;
  const glow = g ? { irradiance: g.irradiance_rgb, std: (g.angular_std_deg ?? 12) * DEG, followSun: g.follows_sun ?? true,
    dir: g.dir ? normalize(g.dir) : null } : null;
  const lights = (env.lights ?? []).slice(0, 4).map(L => ({
    dir: normalize(L.dir), irradiance: L.irradiance_rgb ?? [1, 1, 1],
    std: (L.angular_std_deg ?? 0.27) * DEG, displayStd: (L.display_std_deg ?? 0) * DEG, shadows: L.casts_shadows ?? true,
    followSun: L.follows_sun ?? false,
  }));
  return { sh, bg, horizon, ground, glow, lights };
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
  constructor(device, sources, lighting, env, asset = null) {
    this.device = device;
    this.lighting = lighting;
    this.base = env;
    this.skyGain = 1;
    this.lightGain = 1;
    this.sun = env.lights[0] ? sunAngles(env.lights[0].dir) : [0, 45];
    this.on = true;
    this.key = null;
    this.bgBuf = device.createBuffer({ label: 'sky background', size: SKY_BG_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });   // display.wgsl SkyBg (renderer.bindEnv)
    const module = device.createShaderModule({ label: 'environment',
      code: sources.common + '\n' + sources.material_ext + '\n' + sources.environment });
    const b = lighting.buffers;
    const res = { 0: b.uniform, 1: b.gaussians, 2: b.nodes, 3: b.vgrid, 4: b.parts, 7: b.materials, 8: b.colour,
      9: b.kids, 13: b.lo_dir, 40: b.env, 41: b.env_T, 61: lighting.lodFlagBuf, 62: lighting.lodList,
      81: lighting.scList, 83: b.sc_X };
    const binds = { env_direct: [0, 1, 2, 3, 4, 7, 8, 40, 41],
      env_dirs: [0, 1, 2, 3, 4, 7, 13, 40, 41] };
    // screen-space LOD lighting (lighting.js lodLight / lodCut): drawn nodes, and with the cut the drawn leaves only
    if (lighting.lodLight) binds.env_direct_nodes = [0, 1, 2, 3, 4, 7, 8, 40, 41, 61];
    if (lighting.lodCut) Object.assign(binds, { env_direct_list: [0, 1, 2, 3, 4, 7, 8, 40, 41, 62],
      env_direct_nodes_list: [0, 1, 2, 3, 4, 7, 8, 40, 41, 62] });
    // sender LOD (lighting.js senderCut): the cut's leaves and nodes toward the probes
    if (lighting.senderCut) Object.assign(binds, { env_dirs_sc: [0, 1, 2, 3, 4, 7, 13, 40, 41, 81],
      env_dirs_nodes_sc: [0, 1, 2, 3, 4, 7, 40, 41, 81, 83] });
    this.k = {};
    for (const [entry, list] of Object.entries(binds)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.k[entry] = { pipe, group: device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: list.map(n => ({ binding: n, resource: { buffer: res[n] } })) }) };
    }
    // one-frame shadows from every part (sun_map.js; 2026-10-29): opacity maps per part and light, cascades for large
    // parts (null without distant lights)
    this.sunMap = asset ? new SunMap(device, sources, lighting, this, asset) : null;
    if (this.sunMap && !this.sunMap.ok) this.sunMap = null;
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
    this.key = key;
    const ab = new ArrayBuffer(ENV_BYTES), f = new Float32Array(ab), u = new Uint32Array(ab);
    const { sh, lights } = this.base;
    if (this.on) {
      sh.forEach((c, k) => f.set(c.map(v => v * this.skyGain), 4 * k));
      lights.forEach((L, k) => {
        const dir = k === 0 || L.followSun ? sunDir(...this.sun) : L.dir;
        f.set([...dir, L.std], 64 + 8 * k);
        f.set([...L.irradiance.map(v => v * this.lightGain), L.shadows ? 1 : 0], 68 + 8 * k);
      });
      u[96] = lights.length; u[97] = sh.length ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.lighting.buffers.env, 0, ab);
    // the background: its own SH when the environment has one (display only; it stays put when the sun moves, as the
    // lighting sky does), and the visible discs' sizes
    const bb = new ArrayBuffer(SKY_BG_BYTES), bf = new Float32Array(bb), bu = new Uint32Array(bb);
    const bg = this.base.bg;
    if (this.on && bg?.length) {
      bg.forEach((c, k) => bf.set(c.map(v => v * this.skyGain), 4 * k));
      bu[4 * 81] = bg.length;
    }
    lights.forEach((L, k) => { bf[4 * 82 + k] = L.displayStd; });
    // the horizon ring and the ground below it (SPEC 2.9 sky.horizon / ground_rgb; with background_sh_rgb only)
    const { horizon, ground } = this.base;
    if (this.on && bg?.length && horizon && ground) {
      horizon.rgb.forEach((c, k) => bf.set(c.map(v => v * this.skyGain), 4 * (83 + k)));
      bu[4 * 81 + 1] = horizon.rgb.length;
      bf.set([...ground.map(v => v * this.skyGain), Math.sin(1 * DEG)], 4 * 91);   // 1 deg blend
      bf.set([1 / (2 * horizon.along ** 2), 1 / (2 * horizon.across ** 2)], 4 * 92);
    }
    // the glow round the sun, drawn only (its light is in the lighting sky, sh_rgb)
    const glow = this.base.glow;
    if (this.on && bg?.length && glow) {
      bf.set([...glow.irradiance.map(v => v * this.skyGain), glow.std], 4 * 93);
      bf.set(glow.followSun || !glow.dir ? sunDir(...this.sun) : glow.dir, 4 * 94);
    }
    this.device.queue.writeBuffer(this.bgBuf, 0, bb);
    this.lighting.firstGather = true;   // the probes re-gather the new sky
    return true;
  }

  // the first distant light's direction and radiance peak, for the display (sun disc)
  lights() {
    if (!this.on) return [];
    return this.base.lights.map((L, k) => ({ dir: k === 0 || L.followSun ? sunDir(...this.sun) : L.dir, std: L.std,
      irradiance: L.irradiance.map(v => v * this.lightGain) }));
  }

  // threads: a count, or a byte offset into lighting.lodList's indirect sizes ({ indirect })
  run(encoder, entry, threads) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.k[entry].pipe); pass.setBindGroup(0, this.k[entry].group);
    if (threads.indirect !== undefined) pass.dispatchWorkgroupsIndirect(threads.buf ?? this.lighting.lodList, threads.indirect);
    else pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / 64)));
    pass.end();
  }

  // drawn LOD nodes, after emissive.wgsl shade_emit_nodes (lighting.js encodeNodeDirect)
  encodeNodes(encoder) {
    if (!this.on || !this.base.lights.length) return;
    if (this.lighting.lodCut) this.run(encoder, 'env_direct_nodes_list', { indirect: 32 });
    else if (this.k.env_direct_nodes) this.run(encoder, 'env_direct_nodes', this.lighting.lodN);
  }

  // after shade_emit: the sun maps refresh what changed (a moved part, the light, a cascade) and write env_T, then
  // the direct light
  // still (lighting.js encode): nothing moved for a while, so the sun map holds
  encodeDirect(encoder, timers, changed, still = false) {
    if (!this.on || !this.base.lights.length) return;
    // with motion blur a part that stopped this frame was stretched last frame, so its layer re-splats once more, sharp
    // (lighting.js blurMask, as lights_map / light_shadow get it; 2026-10-31: the sun kept the last smeared shadow)
    const L = this.lighting;
    if (!still) this.sunMap?.encode(encoder, ((L.blur ? L.blurMask : L.movedMask) ?? 0) >>> 0);
    if (this.lighting.lodCut) this.run(encoder, 'env_direct_list', { indirect: 16 });   // ?lodlight=2: the cut
    else this.run(encoder, 'env_direct', this.lighting.count);
  }

  encodeDirs(encoder) {
    if (!this.on || !this.base.lights.length) return;
    this.run(encoder, 'env_dirs', Math.ceil(this.lighting.count / 4));   // the Gaussians direct_dirs_emit refreshed
  }

  // sender LOD (lighting.js senderCut): after direct_dirs_emit_sc / direct_dirs_nodes_sc
  encodeDirsSC(encoder) {
    if (!this.on || !this.base.lights.length) return;
    this.run(encoder, 'env_dirs_sc', { indirect: 8, buf: this.lighting.scList });
    this.run(encoder, 'env_dirs_nodes_sc', { indirect: 20, buf: this.lighting.scList });
  }
}
