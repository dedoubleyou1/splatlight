import { loadAsset } from './asset.js';
import { OrbitCamera } from './camera.js';
import { acquireGpuSession } from './gpu-session.js';
import { partTransform, perspective, srgbToLinear, linearToSrgb } from './math.js';
import { SplatRenderer, PASSES, FRAME_BYTES } from './renderer.js';
import { Lighting, KERNELS, nodeProbeSource } from './lighting.js';
import { nodeGaussians, Lod } from './lod.js';
import { FlakeCamera } from './flake.js';
import { LightShadow } from './light_shadow.js';
import { Environment, loadEnvironment } from './environment.js';
import { GpuTimers } from './timers.js';
import { buildPanel } from './ui.js';

const MAX_PIXELS = 2560 * 1600;   // containment: bounded render size
const SHADERS = ['common', 'project', 'scan', 'scatter', 'radix', 'splat', 'display'];
const RUNTIME = ['common', ...KERNELS.map(k => k[0]), 'bounce', 'emissive', 'material_ext', 'progressive', 'iso_init', 'shadow_cull', 'flake_common', 'flake_direct', 'flake_bounce', 'light_shadow', 'environment', 'fill', 'lod', 'motion_blur', 'sun_map'];
const EMISSIVE = 0;   // light model: 0 emissive Gaussians, 1 the analytic rectangle lamp
const LIT = 3;
const SETTLE_FRAMES = 96;   // lit view: keep rendering after a change while the lamp EMA and the probe bounce converge
const $ = id => document.getElementById(id);
const coneCount = s => (String(s).startsWith('6') ? 6 : String(s).startsWith('8') ? 8 : 0);   // near-field cones select

function fail(message) {
  $('status').textContent = message;
  $('status').className = 'error';
}

const hex = rgb => '#' + rgb.map(c => Math.round(linearToSrgb(Math.min(1, c)) * 255).toString(16).padStart(2, '0')).join('');
const unhex = h => [1, 3, 5].map(i => srgbToLinear(parseInt(h.slice(i, i + 2), 16) / 255));

async function main() {
  if (!navigator.gpu) return fail('WebGPU is not available in this browser.');
  const params = new URLSearchParams(location.search);
  // ?pane=1: the asset's glass-pane variant (runtime/add_glass_pane.py, reflection-coverage tests), ?pane=tall the
  // 1.7 m one (_glasstall); off by default
  const pane = (params.get('pane') ?? '').replace(/[^a-z0-9]/g, '');
  // default room (David, 2026-10-28): true-density V + the display fill (fill.wgsl), replacing cornell_sdf_room's 0.3x skin;
  // its binary moment-matched tree (_bin, experiments/M37_sdf_splats/bintree.py: 15% faster frames, same look) since
  // 2026-10-29. The glass-pane variants exist for the octree room only
  const assetName = (params.get('asset') ?? (pane ? 'cornell_sdf_room_plain' : 'cornell_sdf_room_plain_bin')).replace(/[^A-Za-z0-9_-]/g, '') + (pane === '1' ? '_glass' : pane ? `_glass${pane}` : '');

  await acquireGpuSession();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter.');
  const features = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  // Large assets (1M Gaussians) need storage bindings past the 128 MiB default: ask for what the adapter offers.
  const requiredLimits = { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage };   // one-path specular binds 10
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });

  // Containment: stop on the first device loss or GPU error; never retry.
  let stopped = false;
  device.lost.then(info => { stopped = true; fail(`GPU device lost (${info.reason}): ${info.message}. Reload manually once the cause is understood.`); });
  device.addEventListener('uncapturederror', e => { stopped = true; fail(`GPU error, rendering stopped: ${e.error.message}`); });

  const text = url => fetch(url).then(r => { if (!r.ok) throw new Error(`Could not load ${url}`); return r.text(); });
  const [asset, ...sources] = await Promise.all([
    loadAsset(`runtime/assets/${assetName}`),
    ...SHADERS.map(n => text(`shaders/${n}.wgsl`)),
    ...RUNTIME.map(n => text(`runtime/wgsl/${n}.wgsl`)),
  ]);
  const shaders = Object.fromEntries(SHADERS.map((n, i) => [n, sources[i]]));
  const runtimeShaders = Object.fromEntries(RUNTIME.map((n, i) => [n, sources[SHADERS.length + i]]));
  runtimeShaders.bounce = nodeProbeSource(runtimeShaders.bounce);   // ?nodeprobes=1 (lighting.js)
  const { header } = asset;
  // Animated clip (asset.json `clips`, SPEC 2.8): f32 [frame][part][12] world_from_rest rows, played in place.
  const clip = header.clips?.[0];
  const clipXf = clip ? new Float32Array(await fetch(`runtime/assets/${assetName}/${clip.file}`)
    .then(r => { if (!r.ok) throw new Error(`Could not load ${clip.file}`); return r.arrayBuffer(); })) : null;
  if (clip && clipXf.length !== clip.frame_count * clip.parts.length * 12) throw new Error(`${clip.file}: wrong size`);
  const clipSlot = new Map((clip?.parts ?? []).map((p, k) => [p, k]));

  const canvas = $('view');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  // screen-space LOD (?lod=<px>: a node is drawn in place of its subtree under <px> on screen; lod.js)
  const lodPx = Math.max(0, Number(params.get('lod')) || 0);
  const lodNodes = lodPx > 0 ? nodeGaussians(asset) : null;
  const renderer = new SplatRenderer(device, format, shaders, asset, lodNodes);
  if (lodNodes) { asset.lodNodes = lodNodes; asset.lodFlagBuf = renderer.lodBuf; }   // lighting.js: room for lit nodes
  const lighting = new Lighting(device, runtimeShaders, asset, renderer.gaussBuf, renderer.partBuf);
  // light-side shadows A / B (light_shadow.js, off by default: the walk stays the shadow method)
  lighting.lightShadow = new LightShadow(device, runtimeShaders, lighting, asset);
  await lighting.loadFill(assetName, runtimeShaders);   // display fill for true-density assets (fill.wgsl)
  const lod = lodNodes ? new Lod(device, runtimeShaders, asset, lighting, renderer, lodNodes) : null;
  globalThis.__lod = lod;   // inspection (renderer/results/lod/README.md)
  // projection-field shadows (prototype): ?lights=unified&shadow=pf (sampled) or pfd (direct) loads the asset's pfield.bin
  // without a usable field (missing or stale for this asset) it warns once and keeps the sampled shadows
  let fieldWarning = null;
  if (lighting.unified && ['pf', 'pfd'].includes(params.get('shadow'))) {
    fieldWarning = await lighting.loadField(`runtime/assets/${assetName}/pfield.bin`, params.get('shadow') === 'pfd', asset, assetName);
    if (fieldWarning) console.warn(fieldWarning);
  }
  // occluder-proxy shadows (prototype): ?lights=unified&shadow=rcd (radius-cut walk, ?rmin=) or pxd (the asset's proxy.bin);
  // rcc / pxc: the same once per receiver cluster (?rcl=)
  if (lighting.unified && ['rcd', 'pxd', 'rcc', 'pxc'].includes(params.get('shadow'))) {
    fieldWarning = await lighting.loadProxyShadow(params.get('shadow'), asset, assetName);
    if (fieldWarning) console.warn(fieldWarning);
  }
  renderer.bindColour(lighting.buffers.colour, lighting.buffers.lo_b, lighting.buffers.emitters);
  // SPEC 2.9 environment: the asset's sky + distant lights, or ?env=<name> (runtime/assets/env/<name>.json), ?env=none off
  const envDesc = await loadEnvironment(asset, params);
  const environment = envDesc ? new Environment(device, runtimeShaders, lighting, envDesc, header.parts, asset) : null;
  renderer.bindEnv(lighting.buffers.env);
  // camera-only SGGX flakes (View panel, off by default): its own colour buffer, read by project instead of colour
  const flake = new FlakeCamera(device, runtimeShaders, lighting);
  renderer.bindFlake(flake.cam, lighting.buffers.lo_b, lighting.buffers.emitters);
  // Inspection hook for debugging from the console (reads buffers with readBuffer below).
  window.gsDebug = { device, lighting, renderer, flake, asset, readBuffer: async (buf, size = buf.size) => {
    const tmp = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, tmp, 0, size);
    device.queue.submit([enc.finish()]);
    await tmp.mapAsync(GPUMapMode.READ);
    const out = tmp.getMappedRange().slice(0);
    tmp.destroy();
    return out;
  } };
  const timers = new GpuTimers(device, PASSES);
  const camera = new OrbitCamera(header.camera);
  window.gsDebug.camera = camera;   // inspection / scripted views (target, distance, yaw, pitch)

  const lamp = asset.lamps[0];
  const lampHome = lamp?.center ?? [0, 1, 1.9];
  // Flat-colour palette defaults: each material's albedo as stored in the asset.
  const gview = new DataView(asset.gaussians);
  const ARM_SLOT = 4;   // the clip's parts get their own palette slot (RenderFrame.palette_ext[0])
  const partMaterial = header.parts.map((p, i) => (clipSlot.has(i) ? ARM_SLOT : /red/.test(p.name) ? 1 : /green/.test(p.name) ? 2 : 0));
  const albedo = { white: null, red: null, green: null };
  header.parts.forEach((p, i) => {
    if (partMaterial[i] === ARM_SLOT) return;
    const key = ['white', 'red', 'green'][partMaterial[i]];
    if (!albedo[key]) albedo[key] = [0, 1, 2].map(k => gview.getFloat32(p.gaussians[0] * 64 + 48 + 4 * k, true));
  });
  for (const k in albedo) albedo[k] ??= [0.7, 0.7, 0.7];
  const defaults = {
    lampX: lampHome[0], lampY: lampHome[1], lampZ: header.emitters?.home === 'rest' ? lampHome[2] : 1.97, lampSize: 1,   // z 1.97 matches the reference scene's lamp (David, 2026-10-24); a lamp built in place (emitters.home 'rest', M47 flush lamp) stays where it is
    shortX: 0, shortY: 0, shortYaw: 0, tallX: 0, tallY: 0, tallYaw: 0,
    white: hex(albedo.white), red: hex(albedo.red), green: hex(albedo.green),
    albedoScale: 1, roughness: 0.01, nearField: false, nearRefl: false, emitBounceFix: true, nearCones: '8 cones', progressive: false, iso: true, shProbes: true, flakeCam: false, lpBudget: 1, lpPx: 4,   // emitBounceFix on by default (David, 2026-10-28; M47: emitters out of the probes' bounce); own-part visibility is the base (David, 2026-10-23); cross-part near-field is opt-in and uses 8 cones (P2 moving-part check, 2026-10-24), the walk stays selectable; iso: M39 direction-independent lighting is the default (David, 2026-10-24)
    bounce: clip ? 0 : 1,   // the probe bounce's near_vis is ~0.5 s per frame on a moving Armadillo (M1)
    lightModel: EMISSIVE, emitterCount: 0, lightCut: true, shortShine: 0, shortRough: 0.3, tallShine: 0, tallRough: 0.3, floorShine: 0, floorRough: 0.1, oneSided: lighting.emitOneSided, edgeSoft: lighting.emitEdgeSoft, armPlay: true, armSpeed: 1, armFrame: 0, armX: 0, armY: 0, armYaw: 0, armTint: '#ffffff', armBright: 1, viewMode: params.get('view') === 'flat' ? 0 : LIT,   // ?view=flat starts without lighting (large assets)
    exposure: 1, lowpass: 0.3, continuous: false,
    blurShutter: params.get('blur') !== '0' ? Number(params.get('shutter') ?? 0.5) : 0,   // world-space motion blur (motion blur thread), on by default since 2026-10-29 (David); ?blur=0 off, ?shutter= fraction of the frame (default 0.5)
    blurCov: params.get('blurcov') !== '0',   // motion blur: keep a stretched splat's coverage, not its density; on by default, ?blurcov=0 turns it off (project.wgsl)
    blurLod: Number(params.get('blurlod') ?? 0),   // motion blur variant D: moving parts draw a cut k x their smear (needs ?lod)
    stableSort: !['counting', 'atomic'].includes(params.get('sort')),   // stable sort on by default (David, 2026-10-24): ties in Gaussian index order (radix.wgsl), no per-frame reshuffle; ?sort=counting (or atomic) opts out
    transmit: lighting.assetFeatures.transmit, shortTransl: 0, shortTint: 0, tallTransl: 0, tallTint: 0,   // M41 transmission: on when the asset's own materials transmit (flags bit 4), else off
    asgSpec: true, specCut: false,   // P3: lights and probe cells as shapes in shiny highlights; on by default (David, 2026-10-25)
    reflCoverage: params.has('coverage') ? params.get('coverage') === '1' : lighting.assetFeatures.coverage, reflFresnel: params.has('fresnel') ? params.get('fresnel') === '1' : lighting.assetFeatures.fresnel, glassF0: Number(params.get('f0') ?? 0.04),   // glassF0: Fresnel F0 of coverage materials (?f0=0.2 a stylized pane)   // reflection coverage / view-only Fresnel + probe sheen (experiments/reflection_coverage); on when the asset has coverage materials (flags bit 5) or asset.json features, else off; ?coverage=0/1, ?fresnel=0/1 override
    hierShadow: params.get('hier') !== '0', hierAmortize: params.get('amort') === '1',   // hierarchical shadow walk (shadow_cull.wgsl) on by default (David, 2026-10-25), ?hier=0 off; still Gaussians walking every 4th frame stays opt-in (it settles 4x slower, noisier meanwhile), ?amort=1 on
    shadowMethod: { walk: 0, map: 1, lobe: 2, slice: 3 }[params.get('shadow')] ?? 1,   // light-side shadows (shadow light-map thread): 1 A opacity map, the default (David, 2026-10-25); 0 the sampled walk, 2 B lobe tiles, 3 C light-space slices; ?shadow=walk / map / lobe / slice
    envOn: !!environment, sunAz: environment?.sun[0] ?? 0, sunEl: environment?.sun[1] ?? 45, skyGain: 1, sunGain: 1,   // SPEC 2.9 environment
    smHistory: true, smGolden: false, smJitter: false, smProg4: false,   // N5 shadow smoothing: count-rule history on by default (David, 2026-10-23); the rest opt-in
  };
  // unified lights (?lights=unified, lights.js): per light whether it lights the scene and its gain
  for (const L of lighting.lightSet?.lights ?? []) { defaults[`light${L.id}On`] = L.on; defaults[`light${L.id}Gain`] = L.gain; }
  // asset materials marked `ui: true` (asset.json materials[]): live shine / roughness per material
  for (const m of lighting.uiMaterials) {
    defaults[`mat${m.index}Shine`] = m.shine; defaults[`mat${m.index}Rough`] = m.rough;
    defaults[`mat${m.index}Transl`] = m.transl; defaults[`mat${m.index}Tint`] = m.tint;
  }
  const state = { ...defaults };
  let dirty = true, settle = (state.hierAmortize ? 4 : 1) * SETTLE_FRAMES;
  let clipTime = 0;   // seconds into the clip
  const touch = key => {
    if (key === 'armFrame' && clip) clipTime = state.armFrame / clip.fps;
    // progressive lighting refines over more frames; amortized shadow walks take one sample per stride frames
    dirty = true; settle = (state.progressive ? 4 : state.hierAmortize ? 4 : 1) * SETTLE_FRAMES; schedule();
  };

  // the shadow method select: switching restarts the shadow history (the walk's count rule would otherwise start from
  // the maps' values) and the cull
  function syncShadowMethod() {
    const m = [null, 'map', 'lobe', 'slice'][state.shadowMethod] ?? null;
    if (m !== lighting.lightShadow.method) {
      lighting.lightShadow.setMethod(m);
      lighting.resetHistory();
      lighting.hsDirty = true;
    }
  }
  const groups = [
    { title: 'Lamp', controls: [
      { key: 'lightModel', label: 'light', type: 'select', options: ['emissive Gaussians', 'analytic rectangle'] },
      { key: 'emitterCount', label: 'emitters', type: 'select', options: ['300', '1,200', '4,800', '16,000'] },
      { key: 'lightCut', label: 'light cut (tree)', type: 'check' },
      { key: 'oneSided', label: 'one-sided emission', type: 'check' },
      { key: 'edgeSoft', label: 'one-sided edge softness', type: 'range', min: 0, max: 0.5, step: 0.01 },
      { key: 'lampX', label: 'x', type: 'range', min: -1, max: 1, step: 0.01 },
      { key: 'lampY', label: 'y', type: 'range', min: 0, max: 2, step: 0.01 },
      { key: 'lampZ', label: 'z', type: 'range', min: 0, max: Math.max(1.97, lampHome[2]), step: 0.01 },
      { key: 'lampSize', label: 'size ×', type: 'range', min: 0.1, max: 3, step: 0.05 },
    ] },
    ...(lighting.lightSet ? [{ title: 'Lights', controls: lighting.lightSet.lights.flatMap(L => [
      { key: `light${L.id}On`, label: `${L.name.replace(/^sdf_/, '')} lights the scene`, type: 'check' },
      { key: `light${L.id}Gain`, label: `${L.name.replace(/^sdf_/, '')} brightness ×`, type: 'range', min: 0, max: 8, step: 0.05 },
    ]) }] : []),
    { title: 'Short box', controls: [
      { key: 'shortX', label: 'x offset', type: 'range', min: -0.6, max: 0.6, step: 0.01 },
      { key: 'shortY', label: 'y offset', type: 'range', min: -0.6, max: 0.9, step: 0.01 },
      { key: 'shortYaw', label: 'yaw °', type: 'range', min: -180, max: 180, step: 1, digits: 0 },
      { key: 'shortShine', label: 'shininess', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'shortRough', label: 'roughness', type: 'range', min: 0.02, max: 1, step: 0.01 },
      { key: 'shortTransl', label: 'translucency (M41)', type: 'range', min: 0, max: 0.95, step: 0.01 },
      { key: 'shortTint', label: 'tint (M41)', type: 'range', min: 0, max: 1, step: 0.01 },
    ] },
    { title: 'Tall box', controls: [
      { key: 'tallX', label: 'x offset', type: 'range', min: -0.6, max: 0.6, step: 0.01 },
      { key: 'tallY', label: 'y offset', type: 'range', min: -0.9, max: 0.7, step: 0.01 },
      { key: 'tallYaw', label: 'yaw °', type: 'range', min: -180, max: 180, step: 1, digits: 0 },
      { key: 'tallShine', label: 'shininess', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'tallRough', label: 'roughness', type: 'range', min: 0.02, max: 1, step: 0.01 },
      { key: 'tallTransl', label: 'translucency (M41)', type: 'range', min: 0, max: 0.95, step: 0.01 },
      { key: 'tallTint', label: 'tint (M41)', type: 'range', min: 0, max: 1, step: 0.01 },
    ] },
    ...(lighting.boxMaterial.floor !== undefined ? [{ title: 'Floor', controls: [
      { key: 'floorShine', label: 'shininess', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'floorRough', label: 'roughness', type: 'range', min: 0.02, max: 1, step: 0.01 },
    ] }] : []),
    ...(clip ? [{ title: 'Armadillo', controls: [
      { key: 'armPlay', label: 'play', type: 'check' },
      { key: 'armSpeed', label: 'speed ×', type: 'range', min: 0.05, max: 2, step: 0.05 },
      { key: 'armFrame', label: 'frame (paused)', type: 'range', min: 0, max: clip.frame_count - 1, step: 1, digits: 0 },
      { key: 'armX', label: 'x offset', type: 'range', min: -0.5, max: 1.3, step: 0.01 },
      { key: 'armY', label: 'y offset', type: 'range', min: -0.4, max: 1.3, step: 0.01 },
      { key: 'armYaw', label: 'yaw °', type: 'range', min: -180, max: 180, step: 1, digits: 0 },
      { key: 'armTint', label: 'tint', type: 'color' },
      { key: 'armBright', label: 'albedo ×', type: 'range', min: 0, max: 6, step: 0.05 },
    ] }] : []),
    ...lighting.uiMaterials.map(m => ({ title: m.name, controls: [
      { key: `mat${m.index}Shine`, label: 'shininess', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: `mat${m.index}Rough`, label: 'roughness', type: 'range', min: 0.02, max: 1, step: 0.01 },
      { key: `mat${m.index}Transl`, label: 'translucency (M41)', type: 'range', min: 0, max: 0.95, step: 0.01 },
      { key: `mat${m.index}Tint`, label: 'tint (M41)', type: 'range', min: 0, max: 1, step: 0.01 },
    ] })),
    { title: 'Material', controls: [
      { key: 'white', label: 'white albedo', type: 'color' },
      { key: 'red', label: 'red albedo', type: 'color' },
      { key: 'green', label: 'green albedo', type: 'color' },
      { key: 'albedoScale', label: 'albedo scale', type: 'range', min: 0, max: 1.5, step: 0.01 },
      { key: 'roughness', label: 'wall roughness', type: 'range', min: 0.01, max: 1, step: 0.01 },
    ] },
    ...(environment ? [{ title: 'Sky & sun', controls: [
      { key: 'envOn', label: 'sky + sun (environment)', type: 'check' },
      { key: 'sunAz', label: 'sun azimuth °', type: 'range', min: -180, max: 180, step: 1, digits: 0 },
      { key: 'sunEl', label: 'sun elevation °', type: 'range', min: -10, max: 90, step: 0.5, digits: 1 },
      { key: 'sunGain', label: 'sun ×', type: 'range', min: 0, max: 4, step: 0.05 },
      { key: 'skyGain', label: 'sky ×', type: 'range', min: 0, max: 4, step: 0.05 },
    ] }] : []),
    { title: 'Shadow smoothing (N5)', controls: [
      { key: 'smHistory', label: 'count-rule history', type: 'check' },
      { key: 'smGolden', label: 'golden pick sequence', type: 'check' },
      { key: 'smJitter', label: 'emitter-shaped jitter', type: 'check' },
      { key: 'smProg4', label: 'progressive: 4 samples + ease', type: 'check' },
      { key: 'hierShadow', label: 'hierarchical shadow walk', type: 'check' },
      { key: 'hierAmortize', label: 'still Gaussians walk every 4th frame', type: 'check' },
      { key: 'shadowMethod', label: 'shadows', type: 'select', options: ['walk (sampled)', 'A: opacity map from the light (default)', 'B: Gaussian lobes from the light', 'C: light-space lobes by distance slice'] },
    ] },
    { title: 'View', controls: [
      { key: 'viewMode', label: 'mode', type: 'select', options: ['flat albedo', 'depth key', 'overdraw (fragments > 1/255)', 'lit (M36 shading)'] },
      { key: 'iso', label: 'direction-independent (M39)', type: 'check' },
      { key: 'shProbes', label: 'SH2 probes (M39 stage 2)', type: 'check' },
      { key: 'flakeCam', label: 'SGGX flakes (camera; M39, emissive light)', type: 'check' },
      { key: 'asgSpec', label: 'ASG specular: lights as shapes (M39, emissive light)', type: 'check' },
      { key: 'specCut', label: 'ASG: open nodes wider than the lobe', type: 'check' },
      { key: 'transmit', label: 'transmission: translucency + tint (M41)', type: 'check' },
      { key: 'reflCoverage', label: 'reflection coverage (specular by coverage, not alpha)', type: 'check' },
      { key: 'reflFresnel', label: 'view-only Fresnel + probe sheen', type: 'check' },
      { key: 'bounce', label: 'bounce gain', type: 'range', min: 0, max: 3, step: 0.05 },
      { key: 'nearField', label: 'near-field visibility', type: 'check' },
      { key: 'nearRefl', label: 'near-field reflection', type: 'check' },
      { key: 'emitBounceFix', label: 'emitters out of bounce (M47)', type: 'check' },
      { key: 'nearCones', label: 'near-field method', type: 'select', options: ['8 cones', '6 cones', 'walk (128 cells)'] },
      { key: 'progressive', label: 'progressive tree lighting', type: 'check' },
      { key: 'lpBudget', label: 'nodes per frame', type: 'select', options: ['1,000', '4,000', '16,000', '64,000'] },
      { key: 'lpPx', label: 'refine to (px)', type: 'range', min: 1, max: 32, step: 1, digits: 0 },
      { key: 'exposure', label: 'exposure', type: 'range', min: 0.1, max: 4, step: 0.01 },
      { key: 'lowpass', label: 'low-pass px²', type: 'range', min: 0, max: 1, step: 0.01 },
      { key: 'blurCov', label: 'blur: keep coverage', type: 'check' },
      { key: 'stableSort', label: 'stable sort (ties in a fixed order)', type: 'check' },
      { key: 'continuous', label: 'render every frame', type: 'check' },
      { type: 'button', text: 'Reset camera', onClick: () => { camera.reset(); touch(); } },
      { type: 'button', text: 'Reset scene', onClick: () => {
        Object.assign(state, defaults);
        for (const [k, el] of Object.entries(inputs)) {
          if (el.type === 'checkbox') el.checked = state[k]; else { el.value = state[k]; el.dispatchEvent(new Event('input')); }
        }
        touch();
      } },
    ] },
  ];
  // The public demo shows these controls; ?debug=1 or the panel's debug toggle shows the rest and the pass timers.
  const DEMO = new Set(['lampX', 'lampY', 'lampZ', 'lampSize', 'shortX', 'shortY', 'shortYaw', 'shortShine', 'shortRough',
    'tallX', 'tallY', 'tallYaw', 'tallShine', 'tallRough', 'floorShine', 'floorRough', 'white', 'red', 'green',
    'envOn', 'sunAz', 'sunEl', 'sunGain', 'skyGain', 'bounce', 'exposure', 'Reset camera', 'Reset scene']);
  const inputs = buildPanel($('panel'), groups, state, touch, DEMO);
  const debugToggle = $('debug');
  const setDebug = on => { document.body.classList.toggle('debug-on', on); debugToggle.checked = on; };
  debugToggle.oninput = () => setDebug(debugToggle.checked);
  setDebug(params.get('debug') === '1');
  camera.attach(canvas, touch);

  // Render uniform (layout: shaders/common.wgsl `RenderFrame`).
  const frameBytes = new ArrayBuffer(FRAME_BYTES);
  window.gsDebug.frameBytes = frameBytes;   // inspection: the current render uniform
  // inspection: the UI state, so captures can set toggles (e.g. gsDebug.state.nearField = true; gsDebug.state.nearRefl
  // = true) without the panel; the frame loop and captureLinear sync it into the lighting
  window.gsDebug.state = state;
  state.lodPx = lodPx;   // ?lod=<px>: the LOD threshold, changeable at run time (0 = draw every leaf)
  window.gsDebug.timers = timers;   // inspection: timers.ms holds each pass's averaged GPU time
  window.gsDebug.touch = () => touch();   // inspection: wake the render-on-demand loop after setting state
  const f32 = new Float32Array(frameBytes);
  const u32 = new Uint32Array(frameBytes);
  if (asset.partCount > 32) throw new Error(`${asset.partCount} parts; the renderer handles 32`);
  partMaterial.forEach((m, i) => { u32[(i < 16 ? 48 : 164 - 16) + i] = m; });   // part_material, part_material_ext
  // palette slots 4-7 (RenderFrame.palette_ext, floats 84-99): neutral until something maps a part to them
  for (let k = 0; k < 4; k++) f32.set([1, 1, 1, 0], 84 + 4 * k);
  // camera-only flakes apply to the per-leaf M39 path with emissive Gaussian light (flake.js)
  const flakeOn = () => state.flakeCam && state.viewMode === LIT && state.iso && !state.progressive && state.lightModel === EMISSIVE;
  // reflection coverage / Fresnel: the per-leaf M39 path with emissive light (shade_emit writes the specular layer)
  const reflFlags = () => (state.viewMode === LIT && state.iso && !state.progressive && state.lightModel === EMISSIVE && !flakeOn()
    ? (state.reflCoverage ? 1 : 0) | (state.reflFresnel ? 2 : 0) : 0);
  // far plane: 50 m, or past the scene's bounds (outdoor assets such as hill reach ~90 m)
  const sceneFar = header.bounds ? Math.max(50, 1.5 * Math.hypot(...header.bounds.hi.map((v, k) => v - header.bounds.lo[k]))) : 50;
  // the camera's view-projection (column-major), for the one-path scheduler (?budget=N)
  const viewProj = () => {
    const V = f32.subarray(0, 16), Pm = f32.subarray(16, 32), o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let a = 0; for (let k = 0; k < 4; k++) a += Pm[k * 4 + r] * V[c * 4 + k]; o[c * 4 + r] = a;
    }
    return o;
  };
  let lodView = { focal: 1, near: 0.02 };
  function writeFrame(width, height) {
    const near = 0.02, far = sceneFar;
    f32.set(camera.view(), 0);
    f32.set(perspective(camera.fovy, width / height, near, far), 16);
    ['white', 'red', 'green'].forEach((m, i) => {
      const c = unhex(state[m]);
      // Lit view multiplies the shaded colour by chosen / asset albedo.
      f32.set([...(state.viewMode === LIT ? c.map((v, k) => v / Math.max(albedo[m][k], 1e-4)) : c), 1], 32 + 4 * i);
    });
    f32.set([state.bounce, emitterCount, state.blurShutter, state.blurCov ? 1 : 0], 44);   // z: motion blur shutter (project.wgsl), 0 = off; w = coverage cap (on by default, ?blurcov=0 turns it off)
    f32.set([state.lampX, state.lampY, state.lampZ, flakeOn() ? 1 : 0], 64);   // w: project reads the flake colour
    const rad = lamp?.radiance ?? [1, 1, 1], peak = Math.max(...rad);
    f32.set(rad.map(c => c / peak * 4), 68);
    const refl = reflFlags();
    f32[71] = refl;                                     // lamp_color.w: reflection flags (project.wgsl, splat.wgsl)
    f32.set(lighting.materialView(0.04, state.glassF0), 100);   // mat_view
    lighting.specSplit = refl !== 0;                    // the lighting keeps the specular layer apart
    lighting.sheen = (refl & 2) !== 0 && state.shProbes;
    const focal = height / (2 * Math.tan(camera.fovy / 2));
    f32.set([width, height, focal, focal, near, far], 72);
    lodView = { focal, near };
    f32[180] = lodNodes ? state.lodPx : 0; u32[181] = renderer.nodeBase; u32[182] = renderer.lodNodeCount;   // frame.lod (project.wgsl)
    u32[78] = asset.count;
    u32[79] = state.viewMode;
    f32.set([state.albedoScale, state.roughness, state.exposure, state.lowpass], 80);
    // armadillo slot: multiplies the shaded colour (lit) or the asset albedo (flat; w = 0 keeps the asset's own)
    if (clip) f32.set([...unhex(state.armTint).map(v => v * state.armBright), 0], 84 + 4 * (ARM_SLOT - 4));
  }

  // Part records (SPEC §2.4): the host writes xf (rows [R | t]), xf_prev and
  // flags bit 1 (moved this frame) for every part each frame.
  const partBytes = asset.parts.slice(0);
  const pf = new Float32Array(partBytes), pu = new Uint32Array(partBytes);
  lighting.partXf = pf;   // unified lights follow their parts (lights.js)
  // the Lights panel into the light set (unified lights; a change rebuilds the light tree)
  function syncLights() {
    for (const L of lighting.lightSet?.lights ?? []) { L.on = state[`light${L.id}On`]; L.gain = state[`light${L.id}Gain`]; }
  }
  const rigOf = { short_box: ['shortX', 'shortY', 'shortYaw'], tall_box: ['tallX', 'tallY', 'tallYaw'] };
  // Clip pose at clipTime: rows [R | t] lerped between the two nearest frames, R re-orthonormalised (the clip moves at
  // most ~7 degrees per frame), then the placement (x, y offset, yaw about the torso pivot) on top.
  const torsoPivot = clip ? header.parts[clip.parts[0]].pivot : null;
  function clipRows(k) {
    const F = clip.frame_count, P = clip.parts.length;
    const x = ((clipTime * clip.fps) % F + F) % F, f0 = Math.floor(x), f1 = (f0 + 1) % F, a = x - f0;
    const A = clipXf.subarray((f0 * P + k) * 12, (f0 * P + k + 1) * 12), B = clipXf.subarray((f1 * P + k) * 12, (f1 * P + k + 1) * 12);
    const m = Array.from(A, (v, j) => v + a * (B[j] - v));
    const r0 = [m[0], m[4], m[8]], r1 = [m[1], m[5], m[9]];        // columns 0 and 1 of R
    const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const unit = u => { const l = Math.hypot(...u); return u.map(c => c / l); };
    const c0 = unit(r0), c1 = unit(r1.map((c, j) => c - dot(r1, c0) * c0[j]));
    const c2 = [c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0]];
    return [c0[0], c1[0], c2[0], m[3], c0[1], c1[1], c2[1], m[7], c0[2], c1[2], c2[2], m[11]];
  }
  const staticWarned = new Set();
  function writeParts() {
    let anyMoved = false;
    let movedMask = 0;
    header.parts.forEach((p, i) => {
      const o = i * 32;
      const prev = pf.slice(o, o + 12);
      const keys = rigOf[p.rig_name ?? p.name];
      if (clipSlot.has(i)) {
        const c = clipRows(clipSlot.get(i));
        const m = partTransform(torsoPivot, state.armX, state.armY, state.armYaw);   // column-major 4x4
        for (let r = 0; r < 3; r++) {
          const row = [0, 1, 2, 3].map(j => m[r] * c[j] + m[4 + r] * c[4 + j] + m[8 + r] * c[8 + j] + (j === 3 ? m[12 + r] : 0));
          pf.set(row, o + 4 * r);
        }
      } else if (keys) {
        const m = partTransform(p.pivot ?? [0, 0, 0], ...keys.map(k => state[k]));
        for (let r = 0; r < 3; r++) pf.set([m[r], m[4 + r], m[8 + r], m[12 + r]], o + 4 * r);
      } else if (window.gsDebug?.partOffset?.[i]) {
        // debug / scripted tests: translate a part with no UI rig (e.g. a hill cloud) by gsDebug.partOffset[i] (m)
        const t = window.gsDebug.partOffset[i];
        for (let r = 0; r < 3; r++) pf.set([+(r === 0), +(r === 1), +(r === 2), t[r]], o + 4 * r);
      } else if (i === header.emitters?.part) {
        // the emissive lamp part follows the lamp sliders: translated so its rest centre lands on the lamp position
        const t = [state.lampX, state.lampY, state.lampZ].map((v, k) => v - lampHome[k]);
        for (let r = 0; r < 3; r++) pf.set([+(r === 0), +(r === 1), +(r === 2), t[r]], o + 4 * r);
      }
      // SPEC 2.4 mobility: a static part never moves (its light may be baked); refuse and say so once
      if (p.mobility === 'static' && pf.slice(o, o + 12).some((v, k) => v !== prev[k])) {
        pf.set(prev, o);
        if (!staticWarned.has(i)) { staticWarned.add(i); console.warn(`part ${p.name} is static (asset.json mobility); not moving it`); }
      }
      pf.set(prev, o + 12);
      const moved = pf.slice(o, o + 12).some((v, k) => v !== prev[k]);
      pu[o + 31] = (pu[o + 31] & ~2) | (moved ? 2 : 0);
      anyMoved ||= moved;
      if (moved) movedMask |= 1 << i;
    });
    renderer.writeParts(partBytes);
    lighting.movedMask = movedMask;   // light_shadow.js re-splats only these parts' layers
    return anyMoved;
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    let h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    const s = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h)));
    w = Math.floor(w * s); h = Math.floor(h * s);
    if (!renderer.hdr || canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      renderer.resize(w, h);
      dirty = true;
    }
  }
  new ResizeObserver(touch).observe(canvas);

  const hud = $('timers');
  let lastFrame = performance.now(), cpuMs = 0, frames = 0;
  function showTimers() {
    const lines = timers.enabled
      ? PASSES.map(n => `${n.padEnd(9)} ${timers.ms[n].toFixed(3)} ms`)
      : ['timestamp-query unavailable'];
    if (timers.enabled) lines.push(`${'gpu sum'.padEnd(9)} ${PASSES.reduce((s, n) => s + timers.ms[n], 0).toFixed(3)} ms`);
    lines.push(`frame    ${cpuMs.toFixed(2)} ms (rAF interval)`);
    lines.push(`splats   ${timers.extra[1].toLocaleString()} visible / ${asset.count.toLocaleString()}`);
    lines.push(`target   ${canvas.width}x${canvas.height}`);
    hud.textContent = lines.join('\n');
  }

  let pending = false;
  // ?raf=timeout keeps frames coming in a hidden tab, where requestAnimationFrame stops (benchmarks)
  const nextFrame = params.get('raf') === 'timeout' ? cb => setTimeout(() => cb(performance.now()), 0) : requestAnimationFrame;
  function schedule() {
    if (!pending && !stopped) { pending = true; nextFrame(frame); }
  }
  function frame(now) {
    pending = false;
    if (stopped || window.gsDebug.harness) return;   // debug: a capture script owns the frame (inert unless set)
    resize();
    const playing = clip && state.armPlay;
    if (!dirty && !state.continuous && !playing) return;
    dirty = false;
    // Only back-to-back frames say anything about frame rate; skip idle gaps.
    const dt = now - lastFrame;
    if (playing && dt < 250) clipTime += dt / 1000 * state.armSpeed;
    if (dt < 250) cpuMs = cpuMs ? cpuMs * 0.9 + dt * 0.1 : dt;
    lastFrame = now;
    const emissive = state.lightModel === EMISSIVE;
    lighting.emitterTarget = [300, 1200, 4800, 16000][state.emitterCount];
    lighting.emitMode = state.lightCut ? 1 : 0;
    lighting.emitSmooth = (state.smGolden ? 1 : 0) | (state.smJitter ? 2 : 0) | (state.smProg4 ? 4 : 0);
    lighting.asgSpec = state.asgSpec ? (state.specCut ? 3 : 1) : 0;
    lighting.historyMode = state.smHistory ? 1 : 0;
    if (state.hierShadow !== lighting.hierShadow || (state.hierAmortize ? 4 : 1) !== lighting.hierStride) lighting.setHierShadow(state.hierShadow, state.hierAmortize ? 4 : 1);
    syncShadowMethod();
    lighting.emitEdgeSoft = state.edgeSoft;
    if (state.oneSided !== lighting.emitOneSided) lighting.setOneSided(state.oneSided);
    // The emitter panel is the lamp's visual in both light models (it only lights the scene in the emissive one).
    syncLights();
    const emitKeyBefore = lighting.emitKey;
    emitterCount = lighting.setEmitters([state.lampX, state.lampY, state.lampZ], state.lampSize);
    writeFrame(canvas.width, canvas.height);
    const partsMoved = writeParts();
    const envChanged = environment?.set({ on: state.envOn, sunAz: state.sunAz, sunEl: state.sunEl, skyGain: state.skyGain, lightGain: state.sunGain }) ?? false;
    const lightChanged = partsMoved || envChanged || lighting.emitKey !== emitKeyBefore;
    timers.begin();
    const encoder = device.createCommandEncoder();
    if (state.viewMode === LIT) {
      const center = [state.lampX, state.lampY, state.lampZ];
      lighting.setLamp(0, center, state.lampSize, lampPrev ?? center, emissive);
      lampPrev = center;
      if (state.nearField !== lighting.nearField) lighting.setNearField(state.nearField);
      if (state.nearRefl !== lighting.nearRefl) lighting.setNearReflection(state.nearRefl);
    if (state.emitBounceFix !== lighting.emitterBounceFix) lighting.setEmitterBounceFix(state.emitBounceFix);
      if (coneCount(state.nearCones) !== lighting.nearCones) lighting.setNearCones(coneCount(state.nearCones));
      if (state.iso !== lighting.iso) lighting.setIso(state.iso);
      if (state.shProbes !== lighting.shProbes) lighting.setShProbes(state.shProbes);
      if (state.roughness !== roughnessSet) { lighting.setRoughness(state.roughness); roughnessSet = state.roughness; }
      const boxKey = [state.shortShine, state.shortRough, state.tallShine, state.tallRough, state.floorShine, state.floorRough,
        ...lighting.uiMaterials.flatMap(m => [state[`mat${m.index}Shine`], state[`mat${m.index}Rough`]])].join();
      if (boxKey !== boxSet) {
        lighting.setBoxMaterial('short_box', state.shortShine, state.shortRough);
        lighting.setBoxMaterial('tall_box', state.tallShine, state.tallRough);
        // the floor keeps the walls' material until it is made shiny once
        if (state.floorShine > 0 || lighting.floorCustom) { lighting.setBoxMaterial('floor', state.floorShine, state.floorRough); lighting.floorCustom = true; }
        // asset materials only once their sliders move, so the asset's own flakes survive loading
        for (const m of lighting.uiMaterials) {
          const v = [state[`mat${m.index}Shine`], state[`mat${m.index}Rough`]];
          if (v[0] !== m.shine || v[1] !== m.rough) { lighting.setMaterialShine(m.index, ...v, true); m.shine = v[0]; m.rough = v[1]; }
        }
        boxSet = boxKey;
      }
      // M41 transmission: per-material translucency and tint (only shown with the View toggle on)
      lighting.transmit = state.transmit;
      const trKey = [state.shortTransl, state.shortTint, state.tallTransl, state.tallTint,
        ...lighting.uiMaterials.flatMap(m => [state[`mat${m.index}Transl`], state[`mat${m.index}Tint`]])].join();
      if (trKey !== transmitSet) {
        lighting.setBoxTransmit('short_box', state.shortTransl, state.shortTint);
        lighting.setBoxTransmit('tall_box', state.tallTransl, state.tallTint);
        for (const m of lighting.uiMaterials) {
          const v = [state[`mat${m.index}Transl`], state[`mat${m.index}Tint`]];
          if (v[0] !== m.transl || v[1] !== m.tint) { lighting.setMaterialTransmit(m.index, ...v); m.transl = v[0]; m.tint = v[1]; }
        }
        transmitSet = trKey;
      }
      if (state.progressive && emissive) {
        // progressive tree lighting: a change (lamp, emitters or a part moved) coarsens and refines again
        lighting.encodeProgressive(encoder, camera.eye(), timers, {
          budget: [1000, 4000, 16000, 64000][state.lpBudget], minAng: state.lpPx * camera.fovy / canvas.height,
          refresh: 16, changed: lightChanged || !lpStarted, resetDepth: 3 });
        lpStarted = true;
      } else {
        if (lod?.cut) lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);   // ?lodlight=2
        lighting.encode(encoder, camera.eye(), timers, emissive, lightChanged);
        if (state.bounce > 0) lighting.encodeBounce(encoder, timers, emissive);
        lighting.encodeFill(encoder);
        lod?.encode(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);
      }
      if (flakeOn()) flake.encode(encoder, timers, state.bounce, state.bounce > 0 && state.shProbes);
    }
    renderer.useFlake(flakeOn());
    renderer.stableSort = state.stableSort;
    renderer.encode(encoder, context.getCurrentTexture().createView(), frameBytes, state.viewMode, timers);
    const readback = timers.end(encoder, renderer.drawBuf);
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    // Inspection: submit-to-done time per frame (GPU work + queue latency), not limited by vsync.
    if (window.gsDebug?.gpuTimes) device.queue.onSubmittedWorkDone().then(() => window.gsDebug.gpuTimes?.push(performance.now() - t0));
    timers.collect(readback, showTimers);
    if (++frames === 1) $('status').textContent = `${header.format} ${header.version} · ${assetName} · ${asset.count.toLocaleString()} Gaussians, ${asset.partCount} parts`
      + (fieldWarning ? ` · ${fieldWarning}` : '');
    if (state.viewMode === LIT && settle > 0) { settle--; dirty = true; }
    if (state.viewMode === LIT && environment?.busy) dirty = true;   // the sun shadows run in slices (environment.js)
    if (state.continuous || dirty || playing) schedule();
  }
  let lampPrev = null, roughnessSet = null, boxSet = null, transmitSet = null, emitterCount = 0, lpStarted = false;

  // Inspection: render the current lighting from the asset's camera into a size x size linear HDR image and post it
  // to the server (renderer/results/captures/<name>.f16, rgba half floats, rows top-down) for scoring against
  // references offline. `frames` lighting frames run at the new eye first (the per-leaf bounce refreshes 1/4 a frame).
  // `view` = { eye, at, fov } (degrees) renders from that camera instead of the asset's (M40 flake views).
  // `syncEvery` waits for the GPU every that many frames instead of every frame (a hidden tab throttles each wait; at
  // most that many frames are ever queued).
  window.gsDebug.captureLinear = async (name, size = 256, frames = 8, view = null, syncEvery = 1) => {
    const W = canvas.width, H = canvas.height;
    renderer.resize(size, size);
    if (view) camera.setView(view.eye, view.at, view.fov); else camera.reset();
    if (clip && !state.armPlay) clipTime = state.armFrame / clip.fps;   // a paused clip at the state's frame (set without the panel)
    const emissive = state.lightModel === EMISSIVE;
    // the lighting settings the frame loop syncs before it encodes; without this a capture on a page whose loop never
    // ran (hidden tab) keeps the constructor's nearField = true and walks near_vis for every Gaussian in one submit
    if (state.nearField !== lighting.nearField) lighting.setNearField(state.nearField);
    if (state.nearRefl !== lighting.nearRefl) lighting.setNearReflection(state.nearRefl);
    if (state.emitBounceFix !== lighting.emitterBounceFix) lighting.setEmitterBounceFix(state.emitBounceFix);
    if (coneCount(state.nearCones) !== lighting.nearCones) lighting.setNearCones(coneCount(state.nearCones));
    if (state.iso !== lighting.iso) lighting.setIso(state.iso);
    if (state.shProbes !== lighting.shProbes) lighting.setShProbes(state.shProbes);
    if (state.hierShadow !== lighting.hierShadow || (state.hierAmortize ? 4 : 1) !== lighting.hierStride) lighting.setHierShadow(state.hierShadow, state.hierAmortize ? 4 : 1);
    syncShadowMethod();
    syncLights();
    lighting.setEmitters([state.lampX, state.lampY, state.lampZ], state.lampSize);
    for (let f = 0; f < frames; f++) {
      writeFrame(size, size);
      writeParts();
      const encoder = device.createCommandEncoder();
      const noTimers = { writes: () => undefined };
      if (state.viewMode === LIT) {
        if (state.progressive && emissive) {
          lighting.encodeProgressive(encoder, camera.eye(), noTimers, {
            budget: [1000, 4000, 16000, 64000][state.lpBudget], minAng: state.lpPx * camera.fovy / size,
            refresh: 16, changed: false, resetDepth: 3 });
        } else {
          if (lod?.cut) lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);   // ?lodlight=2
          if (!window.gsDebug.freezeLight) {   // debug: keep the settled lighting, redo only the LOD cut (inert unless set)
            lighting.encode(encoder, camera.eye(), noTimers, emissive);
            if (state.bounce > 0) lighting.encodeBounce(encoder, noTimers, emissive);
            lighting.encodeFill(encoder);
          }
          lod?.encode(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);
        }
        if (flakeOn()) flake.encode(encoder, noTimers, state.bounce, state.bounce > 0 && state.shProbes);
      }
      renderer.useFlake(flakeOn());
      const out = device.createTexture({ size: [size, size], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      renderer.stableSort = state.stableSort;
      renderer.encode(encoder, out.createView(), frameBytes, state.viewMode, noTimers);
      device.queue.submit([encoder.finish()]);
      if ((f + 1) % syncEvery === 0 || f === frames - 1) await device.queue.onSubmittedWorkDone();
      out.destroy();
    }
    const buf = device.createBuffer({ size: size * size * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: renderer.hdr }, { buffer: buf, bytesPerRow: size * 8 }, [size, size]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const bytes = buf.getMappedRange().slice(0);
    buf.destroy();
    renderer.resize(W, H);
    touch();
    const res = await fetch(`/captures/${name}.f16`, { method: 'POST', body: bytes });
    return res.status;
  };
  schedule();
}

main().catch(e => fail(e.message ?? String(e)));
