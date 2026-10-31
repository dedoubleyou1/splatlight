import { loadAsset } from './asset.js';
import { OrbitCamera } from './camera.js';
import { acquireGpuSession } from './gpu-session.js';
import { partTransform, perspective, srgbToLinear, linearToSrgb } from './math.js';
import { SplatRenderer, PASSES, FRAME_BYTES } from './renderer.js';
import { Lighting, KERNELS } from './lighting.js';
import { nodeGaussians, Lod } from './lod.js';
import { FlakeCamera } from './flake.js';
import { LightShadow } from './light_shadow.js';
import { Environment, loadEnvironment } from './environment.js';
import { GpuTimers } from './timers.js';
import { buildPanel } from './ui.js';
import { PhysicsClient } from './physics/client.js';
import { PhysicsGPU } from './physics/gpu_client.js';
import { PhysicsAuto } from './physics/auto_client.js';
import { fitLenses, writeLenses, LENS_BYTES, MODES } from './refract.js';
import { particleConfig, Particles } from './particles.js';

const MAX_PIXELS = 2560 * 1600;   // containment: bounded render size
const SHADERS = ['common', 'project', 'scan', 'scatter', 'radix', 'splat', 'display', 'lens', 'refract', 'exitmap', 'hgrid', 'particles'];
const RUNTIME = ['common', ...KERNELS.map(k => k[0]), 'bounce', 'emissive', 'material_ext', 'iso_init', 'flake_common', 'flake_direct', 'flake_bounce', 'light_shadow', 'environment', 'fill', 'lod', 'motion_blur', 'sun_map', 'lights_map'];
const LIT = 3;
const SETTLE_FRAMES = 16;   // lit view: keep rendering after a change while the lighting settles (one frame for most of it since
                             // 2026-10-29; the quarter-a-frame direct light toward the probes and later bounces take a few more). Was 96.
const $ = id => document.getElementById(id);
const coneCount = s => (String(s).startsWith('6') ? 6 : 8);   // near-field cones select

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
  // Splatlight's scenes (?scene=; ?asset= still loads any asset folder): the Cornell box, and the hill outdoors under
  // the physical clear sky with drifting clouds (hill_lite: the full hill_bin's files are too big for GitHub)
  const SCENES = { cornell: { label: 'Cornell box', asset: 'cornell_sdf_room_plain_bin' }, hill: { label: 'Hill (sky, sun, clouds)', asset: 'hill_lite' } };
  const sceneKey = SCENES[params.get('scene')] ? params.get('scene') : 'cornell';
  // default room (David, 2026-10-28): true-density V + the display fill (fill.wgsl), replacing cornell_sdf_room's 0.3x skin;
  // its binary moment-matched tree (_bin, experiments/M37_sdf_splats/bintree.py: 15% faster frames, same look) since
  // 2026-10-29. The glass-pane variants exist for the octree room only
  const assetRaw = (params.get('asset') ?? (pane ? 'cornell_sdf_room_plain' : SCENES[sceneKey].asset)).replace(/[^A-Za-z0-9_-]/g, '') + (pane === '1' ? '_glass' : pane ? `_glass${pane}` : '');
  // the hills load their binary-tree builds (runtime/bintree.py; faster, David 2026-10-29); ?tree=octree the originals
  const BINARY_ALIAS = { hill: 'hill_bin', hill_lite: 'hill_lite_bin' };
  // ?spares=1: the asset's spare-object pool build (<asset>_spares, runtime/add_spares.py; Spares panel)
  const assetName = (params.get('tree') === 'octree' ? assetRaw : (BINARY_ALIAS[assetRaw] ?? assetRaw)) + (params.get('spares') === '1' ? '_spares' : '');

  await acquireGpuSession();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter.');
  const features = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  // Large assets (1M Gaussians) need storage bindings past the 128 MiB default: ask for what the adapter offers.
  const requiredLimits = { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage };   // one-path specular binds 10
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
  // inspection (?gpumem=1): every buffer and texture this page allocates, by label and size (gsDebug.gpuAlloc; GPU
  // memory checks; it keeps a reference to each, so it stays off by default)
  const gpuAlloc = [];
  if (params.get('gpumem') === '1') for (const [fn, size] of [['createBuffer', d => d.size], ['createTexture', d => {
    const s = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1];
    return s.reduce((a, b) => a * (b ?? 1), 1) * ({ rgba16float: 8, rgba32float: 16, r32float: 4, rg32float: 8, depth32float: 4 }[d.format] ?? 4);
  }]]) {
    const orig = device[fn].bind(device);
    device[fn] = d => { const o = orig(d); gpuAlloc.push({ label: d.label ?? fn, bytes: size(d), obj: o }); return o; };
  }

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
  const { header } = asset;
  // Animated clip (asset.json `clips`, SPEC 2.8): f32 [frame][part][12] world_from_rest rows, played in place.
  const clip = header.clips?.[0];
  const clipXf = clip ? new Float32Array(await fetch(`runtime/assets/${assetName}/${clip.file}`)
    .then(r => { if (!r.ok) throw new Error(`Could not load ${clip.file}`); return r.arrayBuffer(); })) : null;
  if (clip && clipXf.length !== clip.frame_count * clip.parts.length * 12) throw new Error(`${clip.file}: wrong size`);
  const clipSlot = new Map((clip?.parts ?? []).map((p, k) => [p, k]));
  // live rigid-body physics (renderer/src/physics/): when the asset has a physics bake (physics.json + physics.bin from
  // experiments/physics_proto/bake_physics.py) a worker steps its boxes; while the Physics toggle is on it moves them
  // instead of the box sliders
  // ?physbudget=N: node pairs per physics step for the adaptive precision (coarser calibrated levels past it); off by
  // default (David, 2026-10-29: no coarsening until we see it is needed), every pair at full detail
  // default (David, 2026-10-31; or ?physgpu=auto): GPU or CPU worker by workload (physics/auto_client.js), the GPU on the
  // scene's materials bake (?physbake=<folder>, default <asset>_mat_extsb), its poses straight into renderer.partBuf
  // ?physgpu=1: always the GPU physics (physics/gpu_client.js); ?physgpu=0 or none: always the CPU worker
  // (a GPU start that fails, e.g. on a device limit, warns and falls back to the worker; it never blocks the page)
  const physBake = `runtime/assets/${params.get('physbake') ?? `${assetName}_mat_extsb`}`, physMode = params.get('physgpu') ?? 'auto';
  const physics = (physMode === '1' ? await PhysicsGPU.start(physBake, device)
    : physMode === 'auto' ? await PhysicsAuto.start(`runtime/assets/${assetName}`, physBake, device, Number(params.get('physbudget') ?? 0)) : null)
    ?? await PhysicsClient.start(`runtime/assets/${assetName}`, Number(params.get('physbudget') ?? 0));

  const canvas = $('view');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  // screen-space LOD (?lod=<px>: a node is drawn in place of its subtree under <px> on screen; lod.js)
  const lodPx = Math.max(0, Number(params.get('lod') ?? 2) || 0);   // screen-space LOD on by default at 2 px (David, 2026-10-29); ?lod=0 off
  // the tree's node records serve both screen LOD and the sender cut (lighting.js senderCut): ?lod=0 keeps the sender
  // cut (2026-10-31: it used to switch it off too, which made LOD-off look 60% slower lit)
  const lodLightQ = ['1', '2'].includes(params.get('lodlight')), senderQ = params.get('sendercut') !== '0';
  const lodNodes = lodPx > 0 || senderQ || lodLightQ ? nodeGaussians(asset, lodLightQ || senderQ) : null;   // node V grids for lit nodes and sender LOD
  // scene sizes the renderer can't hold: say so instead of rendering wrong (2026-10-31). Parts past 32 fall out of
  // every per-part bit mask (moved, blur, shadow-map layers, LOD open parts); materials past 32 only lose their own
  // view settings (mat_view), so they warn
  if (asset.partCount > 32) throw new Error(`${assetName} has ${asset.partCount} parts; the renderer supports at most 32`);
  if (header.buffers.materials.count > 32) console.warn(`${assetName} has ${header.buffers.materials.count} materials: materials past 32 share material 31's view settings (coverage, Fresnel)`);
  const pcfg = particleConfig(params, header);   // ?particles=sparks|smoke|dust|rain (src/particles.js)
  const renderer = new SplatRenderer(device, format, shaders, asset, lodNodes, pcfg?.count ?? 0);
  // faked refraction (experiments/refraction_fake): one ellipsoid lens per glass part (coverage materials)
  const lenses = fitLenses(asset);
  const lensBytes = new ArrayBuffer(LENS_BYTES);
  if (lodNodes) { asset.lodNodes = lodNodes; asset.lodFlagBuf = renderer.lodBuf; }   // lighting.js: room for lit nodes
  const lighting = new Lighting(device, runtimeShaders, asset, renderer.gaussBuf, renderer.partBuf);
  // buffers past the device's binding limit would fail validation (or bind short) on smaller GPUs: name them
  const tooBig = Object.entries(lighting.buffers).filter(([, b]) => b?.size > device.limits.maxStorageBufferBindingSize && (b.usage & GPUBufferUsage.STORAGE));
  if (tooBig.length) throw new Error(`${assetName} needs storage buffers larger than this GPU binds (${device.limits.maxStorageBufferBindingSize} B): `
    + tooBig.map(([k, b]) => `${k} ${(b.size / 2 ** 20).toFixed(0)} MB`).join(', '));
  renderer.setGlass(lenses, asset, lighting.buffers);   // E: the glass skin per lens; the exit trace walks the asset tree
  if (pcfg) {
    renderer.particles = new Particles(device, shaders, pcfg, { buffer: renderer.gaussBuf, offset: renderer.particleOffset, size: pcfg.count * 64 }, renderer.noProbes);
    if (lighting.buffers.probe_params && lighting.buffers.probe_sh) renderer.particles.bindProbes({ params: lighting.buffers.probe_params, sh: lighting.buffers.probe_sh });
    if (params.has('pwarm')) renderer.particles.pendingWarm = Number(params.get('pwarm'));   // ?pwarm=<s>: start that far into the effect
  }
  // the emitter light's shadows: its opacity map (light_shadow.js; unified lights have one per light, lights_map.js)
  if (!lighting.unified && !lighting.lampOff) lighting.lightShadow = new LightShadow(device, runtimeShaders, lighting, asset);   // none without a lamp (~60 MB of maps on hill_bin)
  await lighting.loadFill(assetName, runtimeShaders);   // display fill for true-density assets (fill.wgsl)
  const lod = lodNodes ? new Lod(device, runtimeShaders, asset, lighting, renderer, lodNodes) : null;
  // the refraction exit map (E / EC) traces the glass skin's leaves: glass parts always open to their leaves meanwhile
  const glassMask = lenses.reduce((m, l) => (l.part < 32 ? m | (1 << l.part) : m), 0) >>> 0;
  if (lod) lod.openParts = () => (state.refractMode >= 6 ? glassMask : 0);
  globalThis.__lod = lod;   // inspection (renderer/results/lod/README.md)
  renderer.bindColour(lighting.buffers.colour, lighting.buffers.lo_b, lighting.buffers.emitters);
  // SPEC 2.9 environment: the asset's sky + distant lights, or ?env=<name> (runtime/assets/env/<name>.json), ?env=none off
  const envDesc = await loadEnvironment(asset, params);
  const environment = envDesc ? new Environment(device, runtimeShaders, lighting, envDesc, asset) : null;
  renderer.bindEnv(lighting.buffers.env, environment?.bgBuf);
  // camera-only SGGX flakes (View panel, off by default): its own colour buffer, read by project instead of colour
  // camera-only flakes: built the first time they are switched on (~70 MB on hill_bin, off by default; 2026-10-29)
  let flakeCam = null;
  const flake = () => {
    if (!flakeCam) {
      flakeCam = new FlakeCamera(device, runtimeShaders, lighting);
      renderer.bindFlake(flakeCam.cam, lighting.buffers.lo_b, lighting.buffers.emitters);
    }
    return flakeCam;
  };
  // Inspection hook for debugging from the console (reads buffers with readBuffer below).
  window.gsDebug = { device, gpuAlloc, lighting, renderer, flake, asset, readBuffer: async (buf, size = buf.size) => {
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
    lampX: lampHome[0], lampY: lampHome[1], lampZ: header.emitters?.home === 'rest' ? lampHome[2] : 1.97,   // z 1.97 matches the reference scene's lamp (David, 2026-10-24); a lamp built in place (emitters.home 'rest', M47 flush lamp) stays where it is
    shortX: 0, shortY: 0, shortYaw: 0, tallX: 0, tallY: 0, tallYaw: 0,
    white: hex(albedo.white), red: hex(albedo.red), green: hex(albedo.green),
    albedoScale: 1, roughness: 0.01, nearField: false, nearRefl: false, emitBounceFix: true, nearCones: '8 cones', flakeCam: false,   // emitBounceFix on by default (David, 2026-10-28; M47: emitters out of the probes' bounce); own-part visibility is the base (David, 2026-10-23); cross-part near-field is opt-in and uses 8 cones (P2 moving-part check, 2026-10-24), the walk stays selectable; iso: M39 direction-independent lighting is the default (David, 2026-10-24)
    bounce: clip ? 0 : 1,   // the probe bounce's near_vis is ~0.5 s per frame on a moving Armadillo (M1)
    lightCut: true, shortShine: 0, shortRough: 0.3, tallShine: 0, tallRough: 0.3, floorShine: 0, floorRough: 0.1, oneSided: lighting.emitOneSided, edgeSoft: lighting.emitEdgeSoft, armPlay: true, armSpeed: 1, armFrame: 0, armX: 0, armY: 0, armYaw: 0, armTint: '#ffffff', armBright: 1, viewMode: params.get('view') === 'flat' ? 0 : LIT,   // ?view=flat starts without lighting (large assets)
    exposure: 1, tone: Math.max(0, ['clamp', 'agx', 'aces', 'reinhard', 'neutral', 'filmic'].indexOf(params.get('tone') ?? 'agx')),   // display tone curve (display.wgsl tone_map): AgX by default (David, 2026-10-31), ?tone=clamp|aces|reinhard|neutral|filmic
    lowpass: 0.3, continuous: false, physics: false, explodeSpeed: 3,
    blurShutter: params.get('blur') !== '0' ? Number(params.get('shutter') ?? 0.5) : 0,   // world-space motion blur (motion blur thread), on by default since 2026-10-29 (David); ?blur=0 off, ?shutter= fraction of the frame (default 0.5)
    blurCov: params.get('blurcov') !== '0',   // motion blur: keep a stretched splat's coverage, not its density; on by default, ?blurcov=0 turns it off (project.wgsl)
    blurLod: Number(params.get('blurlod') ?? 0),   // motion blur variant D: moving parts draw a cut k x their smear (needs ?lod)
    refractMode: Math.max(0, ['off', 'A', 'B', 'C', 'AC', 'BC', 'E', 'EC', 'EG'].indexOf(params.get('refract') === '1' ? 'A' : (params.get('refract') ?? 'off'))), bend: Number(params.get('bend') ?? -4), fastLayers: params.get('layers') !== 'full', ior: Number(params.get('ior') ?? 1.5), dispersion: Number(params.get('disp') ?? 0), absorb: Number(params.get('absorb') ?? 0.5),   // faked refraction through each glass part's ellipsoid lens (refract.js), off by default; ?refract=1 on, ?ior= ?disp= (ior spread per channel) ?absorb= (1/m)
    stableSort: !['counting', 'atomic'].includes(params.get('sort')),   // stable sort on by default (David, 2026-10-24): ties in Gaussian index order (radix.wgsl), no per-frame reshuffle; ?sort=counting (or atomic) opts out
    transmit: lighting.assetFeatures.transmit, shortTransl: 0, shortTint: 0, tallTransl: 0, tallTint: 0,   // M41 transmission: on when the asset's own materials transmit (flags bit 4), else off
    asgSpec: true, specCut: false,   // P3: lights and probe cells as shapes in shiny highlights; on by default (David, 2026-10-25)
    reflCoverage: params.has('coverage') ? params.get('coverage') === '1' : lighting.assetFeatures.coverage, reflFresnel: params.has('fresnel') ? params.get('fresnel') === '1' : lighting.assetFeatures.fresnel, glassF0: Number(params.get('f0') ?? 0.04),   // glassF0: Fresnel F0 of coverage materials (?f0=0.2 a stylized pane)   // reflection coverage / view-only Fresnel + probe sheen (experiments/reflection_coverage); on when the asset has coverage materials (flags bit 5) or asset.json features, else off; ?coverage=0/1, ?fresnel=0/1 override
    envOn: !!environment, sunAz: environment?.sun[0] ?? 0, sunEl: environment?.sun[1] ?? 45, skyGain: 1, sunGain: 1,   // SPEC 2.9 environment
    cloudWind: Number(params.get('wind') ?? 1), cloudDir: Number(params.get('winddir') ?? 15),   // drifting clouds (movable parts named cloud*): m/s (?wind=0 holds them still) toward azimuth ° from +x
  };
  // unified lights (?lights=unified, lights.js): per light whether it lights the scene and its gain
  for (const L of lighting.lightSet?.lights ?? []) { defaults[`light${L.id}On`] = L.on; defaults[`light${L.id}Gain`] = L.gain; }
  // asset materials marked `ui: true` (asset.json materials[]): live shine / roughness per material
  for (const m of lighting.uiMaterials) {
    defaults[`mat${m.index}Shine`] = m.shine; defaults[`mat${m.index}Rough`] = m.rough;
    defaults[`mat${m.index}Transl`] = m.transl; defaults[`mat${m.index}Tint`] = m.tint;
  }
  const state = { ...defaults };
  state.stillSkip = params.get('stillskip') !== '0';   // ?stillskip=0: every lighting pass every frame (frame(): unchanged inputs)
  let dirty = true, settle = SETTLE_FRAMES;
  let clipTime = 0;   // seconds into the clip
  const touch = key => {
    if (key === 'armFrame' && clip) clipTime = state.armFrame / clip.fps;
    if (key === 'physics' && state.physics) physics.place(sliderXf);   // physics starts where the sliders put the boxes
    dirty = true; settle = SETTLE_FRAMES; schedule();
  };

  // The spare-object pool (asset.json spares, runtime/add_spares.py; ?spares=1 loads <asset>_spares): hidden copies of
  // parts, parked far below the scene and flagged hidden (part flags bit 2, project.wgsl) until shown. A shown spare
  // stands with its pivot at (x, y) on the floor, turned by yaw, like a box slider pose; showing, moving or hiding one
  // is a part move for the lighting. Spares are not physics bodies (the physics bake has the source's bodies only).
  const spares = (header.spares?.parts ?? []).map(s => ({ ...s, shown: false, x: 0, y: 0, yaw: 0 }));
  const sparePart = new Map(spares.map(s => [s.part, s]));
  const pivotOf = i => header.parts[i].pivot ?? [0, 0, 0];
  // each part's floor footprint: an oriented rectangle in its rest frame (centre, unit axis u from the Gaussians'
  // horizontal covariance, half-extents along u and its normal, incl. 2 sigma)
  const footprint = header.parts.map((p, i) => {
    if (!spares.length || p.static) return null;
    const g = new Float32Array(asset.gaussians), [lo, hi] = p.gaussians, n = hi - lo;
    let cx = 0, cy = 0, xx = 0, xy = 0, yy = 0;
    for (let k = lo; k < hi; k++) { cx += g[16 * k]; cy += g[16 * k + 1]; }
    cx /= n; cy /= n;
    for (let k = lo; k < hi; k++) { const x = g[16 * k] - cx, y = g[16 * k + 1] - cy; xx += x * x; xy += x * y; yy += y * y; }
    const th = 0.5 * Math.atan2(2 * xy, xx - yy), u = [Math.cos(th), Math.sin(th)], h = [0, 0];
    for (let k = lo; k < hi; k++) {
      const x = g[16 * k] - cx, y = g[16 * k + 1] - cy, m = 2 * g[16 * k + 10];
      h[0] = Math.max(h[0], Math.abs(x * u[0] + y * u[1]) + m); h[1] = Math.max(h[1], Math.abs(-x * u[1] + y * u[0]) + m);
    }
    return { c: [cx, cy], u, h };
  });
  // a footprint placed by a transform's rows [R | t] (yaw only): world centre, axes, half-extents
  const placed = (f, xf) => {
    const R = (x, y) => [xf[0] * x + xf[1] * y, xf[4] * x + xf[5] * y];
    const c = R(...f.c);
    return { c: [c[0] + xf[3], c[1] + xf[7]], a: [R(...f.u), R(-f.u[1], f.u[0])], h: f.h };
  };
  // separating axes: two placed rectangles overlap unless one of their four axes separates them
  const overlap = (A, B) => [...A.a, ...B.a].every(ax => {
    const d = Math.abs((A.c[0] - B.c[0]) * ax[0] + (A.c[1] - B.c[1]) * ax[1]);
    const r = Q => Q.h.reduce((sum, h, k) => sum + h * Math.abs(Q.a[k][0] * ax[0] + Q.a[k][1] * ax[1]), 0);
    return d < r(A) + r(B);
  });
  const spareOf = q => (typeof q === 'number' ? spares[q] : spares.find(s => s.name === q));
  function showSpare(q, x, y, yaw = 0) {
    const s = spareOf(q);
    if (!s) throw new Error(`no spare ${q} (${spares.map(t => t.name).join(', ') || 'this asset has none'})`);
    Object.assign(s, { shown: true, x, y, yaw });
    touch();
    return s.name;
  }
  function hideSpare(q) { const s = spareOf(q); if (s) { s.shown = false; touch(); } }
  const spareXf = (i, x, y, yaw) => { const m = partTransform(pivotOf(i), x - pivotOf(i)[0], y - pivotOf(i)[1], yaw); return [0, 1, 2].flatMap(r => [m[r], m[4 + r], m[8 + r], m[12 + r]]); };
  // Spawn: the next hidden spare (of a given source part, or any) at a random free spot on the floor: random pivots and
  // yaws, kept when its footprint is inside the room and clear of every shown movable part's (sliders, physics,
  // spares); after 400 random misses, a grid search. A spare with no free spot stays hidden and the next kind is tried;
  // none: null.
  function spawnSpare(of) {
    for (const s of spares.filter(t => !t.shown && (!of || t.of === of))) { const name = trySpawn(s); if (name) return name; }
    return null;
  }
  function trySpawn(s) {
    const room = header.room ?? { x: [-1, 1], y: [0, 2] };
    const others = header.parts.map((p, i) => i).filter(i => i !== s.part && footprint[i] && (sparePart.get(i)?.shown ?? true))
      .map(i => placed(footprint[i], pf.subarray(i * 32, i * 32 + 12)));
    const fits = (x, y, yaw) => {
      const F = placed(footprint[s.part], spareXf(s.part, x, y, yaw));
      const inside = [-1, 1].every(i => [-1, 1].every(j => {
        const px = F.c[0] + i * F.h[0] * F.a[0][0] + j * F.h[1] * F.a[1][0], py = F.c[1] + i * F.h[0] * F.a[0][1] + j * F.h[1] * F.a[1][1];
        return px > room.x[0] && px < room.x[1] && py > room.y[0] && py < room.y[1];
      }));
      return inside && !others.some(O => overlap(F, O));
    };
    for (let t = 0; t < 400; t++) {
      const x = room.x[0] + Math.random() * (room.x[1] - room.x[0]), y = room.y[0] + Math.random() * (room.y[1] - room.y[0]);
      const yaw = Math.round(Math.random() * 360 - 180);
      if (fits(x, y, yaw)) return showSpare(spares.indexOf(s), x, y, yaw);
    }
    // random tries missed: every free spot on a 10 cm x 15 degree grid, one of them at random (null only when none fits)
    const free = [];
    for (let x = room.x[0]; x <= room.x[1]; x += 0.1) for (let y = room.y[0]; y <= room.y[1]; y += 0.1)
      for (let yaw = -180; yaw < 180; yaw += 15) if (fits(x, y, yaw)) free.push([x, y, yaw]);
    if (free.length) return showSpare(spares.indexOf(s), ...free[Math.floor(Math.random() * free.length)]);
    console.info(`no free spot on the floor for ${s.name}`);
    return null;
  }
  // inspection / scripted scenes: gsDebug.spares.show('short_box_spare1', x, y, yaw), .spawn('tall_box'), .hide(name), .clear()
  window.gsDebug.spares = { list: spares, show: showSpare, hide: hideSpare, spawn: spawnSpare,
    clear: () => { spares.forEach(s => { s.shown = false; }); touch(); } };

  const groups = [
    { title: 'Lamp', controls: [
      { key: 'lightCut', label: 'light cut (tree)', type: 'check' },
      { key: 'oneSided', label: 'one-sided emission', type: 'check' },
      { key: 'edgeSoft', label: 'one-sided edge softness', type: 'range', min: 0, max: 0.5, step: 0.01 },
      { key: 'lampX', label: 'x', type: 'range', min: -1, max: 1, step: 0.01 },
      { key: 'lampY', label: 'y', type: 'range', min: 0, max: 2, step: 0.01 },
      { key: 'lampZ', label: 'z', type: 'range', min: 0, max: Math.max(1.97, lampHome[2]), step: 0.01 },
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
    ...(spares.length ? [{ title: `Spares (${spares.length} hidden copies)`, controls: [
      { label: 'show the next spare at a free spot on the floor', type: 'button', text: 'Spawn', onClick: () => spawnSpare() },
      { label: 'hide them all again', type: 'button', text: 'Clear', onClick: () => window.gsDebug.spares.clear() },
    ] }] : []),
    ...(physics ? [{ title: 'Physics', controls: [
      { key: 'physics', label: 'physics (click a box: push)', type: 'check' },
      { label: 'tall box dropped, short one thrown (random)', type: 'button', text: 'Drop', onClick: () => { setPhysics(); physics.drop(); } },
      { label: 'boxes back at rest', type: 'button', text: 'Reset', onClick: () => { setPhysics(); physics.reset(); } },
    ] }] : []),
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
    ...(header.parts.some(p => /^cloud/.test(p.name) && p.mobility !== 'static') ? [{ title: 'Clouds', controls: [
      { key: 'cloudWind', label: 'cloud wind m/s', type: 'range', min: 0, max: 6, step: 0.1, digits: 1 },
      { key: 'cloudDir', label: 'wind toward °', type: 'range', min: -180, max: 180, step: 1, digits: 0 },
    ] }] : []),
    ...(lenses.length ? [{ title: `Glass: faked refraction (${lenses.length} lenses)`, controls: [
      { key: 'refractMode', label: 'method', type: 'select', options: ['off', 'A: ellipsoid lens', 'B: density gradient', 'C: warp inside Gaussians', 'A + C', 'B + C', 'E: exit map', 'E + C', 'E + G: gather inside'] },
      { key: 'bend', label: 'B: bend', type: 'range', min: -5, max: 5, step: 0.05 },
      { key: 'ior', label: 'index of refraction', type: 'range', min: 1, max: 2.4, step: 0.01 },
      { key: 'dispersion', label: 'dispersion (ior spread)', type: 'range', min: 0, max: 0.15, step: 0.005, digits: 3 },
      { key: 'absorb', label: 'absorption 1/m', type: 'range', min: 0, max: 10, step: 0.1, digits: 1 },
    ] }] : []),
    { title: 'Explode', controls: [
      { label: 'everything is Gaussians: scatter, fall, reassemble', type: 'button', text: 'Explode', onClick: () => startExplode() },
      { key: 'explodeSpeed', label: 'burst speed m/s', type: 'range', min: 0.5, max: 8, step: 0.1 },
    ] },
    { title: 'View', controls: [
      { key: 'viewMode', label: 'mode', type: 'select', options: ['flat albedo', 'depth key', 'overdraw (fragments > 1/255)', 'lit (M36 shading)'] },
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
      { key: 'nearCones', label: 'near-field method', type: 'select', options: ['8 cones', '6 cones'] },
      { key: 'exposure', label: 'exposure', type: 'range', min: 0.1, max: 4, step: 0.01 },
      { key: 'tone', label: 'tone curve', type: 'select', options: ['clamp', 'AgX', 'ACES', 'Reinhard (luminance)', 'Khronos PBR Neutral', 'Filmic (Hable)'] },
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
  // The public demo shows these controls; ?debug=1 (no visible toggle) shows the rest and the pass timers.
  // The room keeps its lamp, box, material and physics controls; the hill shows the sky, sun and clouds.
  const room = header.parts.some(p => p.name === 'short_box');
  const DEMO = new Set([...(room ? ['lampX', 'lampY', 'lampZ', 'oneSided', 'shortX', 'shortY', 'shortYaw', 'shortShine', 'shortRough',
    'tallX', 'tallY', 'tallYaw', 'tallShine', 'tallRough', 'floorShine', 'floorRough', 'white', 'red', 'green',
    'physics', 'Drop', 'Reset'] : []),
    'envOn', 'sunAz', 'sunEl', 'sunGain', 'skyGain', 'cloudWind', 'cloudDir', 'Explode', 'bounce', 'exposure', 'Reset camera', 'Reset scene']);
  // Scene picker: each scene is its own page load (?scene=)
  const scenePick = document.createElement('fieldset');
  scenePick.innerHTML = '<legend>Scene</legend>';
  const sceneRow = document.createElement('label');
  sceneRow.className = 'row';
  sceneRow.innerHTML = '<span>scene</span>';
  const sceneSel = document.createElement('select');
  for (const [k, v] of Object.entries(SCENES)) sceneSel.add(new Option(v.label, k));
  sceneSel.value = sceneKey;
  sceneSel.oninput = () => { const u = new URL(location.href); u.searchParams.set('scene', sceneSel.value); u.searchParams.delete('asset'); location.href = u.href; };
  sceneRow.append(sceneSel);
  scenePick.append(sceneRow);
  $('panel').append(scenePick);
  const inputs = buildPanel($('panel'), groups, state, touch, DEMO);
  document.body.classList.toggle('debug-on', params.get('debug') === '1');
  camera.attach(canvas, touch);
  function setPhysics() { state.physics = true; inputs.physics.checked = true; }
  if (physics) {
    window.gsDebug.physics = physics;   // inspection: gsDebug.physics.drop(seed) replays a Drop
    physics.onState = () => { if (state.physics) touch(); };
    physics.attachClicks(canvas, camera, () => state.physics);
  }

  // Render uniform (layout: shaders/common.wgsl `RenderFrame`).
  const frameBytes = new ArrayBuffer(FRAME_BYTES);
  window.gsDebug.frameBytes = frameBytes;   // inspection: the current render uniform
  window.gsDebug.lenses = lenses;   // inspection: the fitted glass lenses (refract.js)
  // inspection: the UI state, so captures can set toggles (e.g. gsDebug.state.nearField = true; gsDebug.state.nearRefl
  // = true) without the panel; the frame loop and captureLinear sync it into the lighting
  window.gsDebug.state = state;
  window.gsDebug.lod = lod;   // inspection (smoke_renderer.mjs lod_colour_cut)
  state.lodPx = lodPx;   // ?lod=<px>: the LOD threshold, changeable at run time (0 = draw every leaf)
  window.gsDebug.timers = timers;   // inspection: timers.ms holds each pass's averaged GPU time
  window.gsDebug.touch = () => touch();   // inspection: wake the render-on-demand loop after setting state
  lighting.requestFrame = () => touch();   // asynchronous work that lands later (lighting.js growSlots) asks for frames
  const f32 = new Float32Array(frameBytes);
  const u32 = new Uint32Array(frameBytes);
  if (asset.partCount > 32) throw new Error(`${asset.partCount} parts; the renderer handles 32`);
  partMaterial.forEach((m, i) => { u32[(i < 16 ? 48 : 164 - 16) + i] = m; });   // part_material, part_material_ext
  // palette slots 4-7 (RenderFrame.palette_ext, floats 84-99): neutral until something maps a part to them
  for (let k = 0; k < 4; k++) f32.set([1, 1, 1, 0], 84 + 4 * k);
  // camera-only flakes apply to the per-leaf M39 path with emissive Gaussian light (flake.js)
  // the camera's eye and forward axis for the lighting (sun_map.js places its cascades ahead of the camera)
  const setCamForLighting = () => {
    const v = camera.view();
    lighting.camEye = camera.eye();
    lighting.camFwd = [-v[2], -v[6], -v[10]];
  };
  const flakeOn = () => state.flakeCam && state.viewMode === LIT;
  // reflection coverage / Fresnel: the per-leaf M39 path with emissive light (shade_emit writes the specular layer)
  const reflFlags = () => (state.viewMode === LIT && !flakeOn()
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
  // Explode button: each Gaussian keeps the lit colour it has now (the lighting is paused and never sees the move) and
  // project.wgsl flies it off from the scene centre, tumbling and bouncing on the floor, then eases it back home.
  // Real seconds: FLY of flight (played at SLOWMO), HOLD on the floor, BACK to reassemble.
  const EXPLODE = { FLY: 3.2, SLOWMO: 0.75, HOLD: 0.6, BACK: 2.4 };
  let explode = null;   // { start ms, prev [t, a], centre, floor }
  function startExplode() {
    if (!explode?.centre) {
      // rest-pose bounds of the scene Gaussians (z up): burst from the middle, land on the lowest point
      const g = new Float32Array(asset.gaussians, 0, asset.count * 16);
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < asset.count; i++) for (let k = 0; k < 3; k++) { const v = g[16 * i + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
      explode = { centre: lo.map((v, k) => 0.5 * (v + hi[k])), floor: lo[2] };
    }
    explode.start = performance.now();
    explode.prev = [0, 0];
    touch();
  }
  window.gsDebug.explode = startExplode;
  const exploding = () => explode?.start !== undefined;
  function writeExplode() {
    if (!exploding()) { f32.fill(0, 184, 196); return; }
    const s = (performance.now() - explode.start) / 1000, { FLY, SLOWMO, HOLD, BACK } = EXPLODE;
    const t = Math.min(s, FLY) * SLOWMO, a = Math.min(Math.max((s - FLY - HOLD) / BACK, 0), 1);
    f32.set([t, a, ...explode.prev, ...explode.centre, explode.floor, state.explodeSpeed, 9.8, 1, 0], 184);
    explode.prev = [t, a];
    if (a >= 1) { delete explode.start; explode.ended = true; }   // this last frame draws everything home
  }

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
    const mv = lighting.materialView(0.04, state.glassF0);
    f32.set(mv.subarray(0, 64), 100); f32.set(mv.subarray(64), 196);   // mat_view (materials 0-15), mat_view_ext (16-31)
    lighting.specSplit = refl !== 0;                    // the lighting keeps the specular layer apart
    lighting.sheen = (refl & 2) !== 0;
    const focal = height / (2 * Math.tan(camera.fovy / 2));
    f32.set([width, height, focal, focal, near, far], 72);
    lodView = { focal, near };
    f32[180] = lodNodes ? state.lodPx : 0; u32[181] = renderer.nodeBase; u32[182] = renderer.lodNodeCount;   // frame.lod (project.wgsl)
    writeExplode();
    u32[78] = asset.count;
    u32[79] = state.viewMode;
    f32.set([state.albedoScale, state.roughness, state.exposure, state.lowpass], 80);
    u32[260] = state.tone;   // frame.tone.x (display.wgsl tone_map)
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
  // Drifting clouds (David, 2026-10-31; experiments/M50_drifting_clouds): movable parts named cloud* drift along the
  // wind (state.cloudWind m/s toward azimuth state.cloudDir degrees from +x), each part a little faster or slower than
  // the rest so they overtake one another and their puffs overlap, merge and part (hill_bin: 13 parts of 2-3 puffs,
  // replace_clouds.py). Past CLOUD_FIELD m from the hill along the wind a part fades out (draw time only: frame.part_fade,
  // project.wgsl; its shadow then falls off the hill), is hidden for the jump and wraps round to fade in upwind.
  // Translation only. The pose is a function of the distance travelled (cloudDist), so the wind slider changes speed
  // without a jump and wind 0 freezes them. gsDebug.cloudDist reads / sets it (captures).
  const CLOUD_FIELD = 75, CLOUD_FADE = 22;   // half length of the wrap (m) and the fade band inside its ends
  const cloudParts = header.parts.map((p, i) => ({ p, i })).filter(({ p, i }) => /^cloud/.test(p.name) && p.mobility !== 'static' && i < 32)
    .map(({ i }, k) => {
      const b = new Float32Array(asset.parts, i * 128 + 96, 4);   // rest bound: centre, radius
      return { i, c: [b[0], b[1], b[2]], rate: 1 + 0.22 * Math.sin(2.3 * k + 1), ph: 1.7 * k };
    });
  const cloudOf = new Map(cloudParts.map(c => [c.i, c]));
  let cloudDist = 0;
  // -> { t: translation, fade: 1 shown .. 0 gone }
  function cloudPose(c) {
    const a = state.cloudDir * Math.PI / 180, u = [Math.cos(a), Math.sin(a)];
    const a0 = c.c[0] * u[0] + c.c[1] * u[1], L = CLOUD_FIELD;
    const along = ((a0 + cloudDist * c.rate + L) % (2 * L) + 2 * L) % (2 * L) - L;
    const x = Math.min(1, Math.max(0, (L - Math.abs(along)) / CLOUD_FADE)), fade = x * x * (3 - 2 * x);
    const d = along - a0;
    return { t: [d * u[0], d * u[1], 0.6 * Math.sin(cloudDist * c.rate / 23 + 2 * c.ph)], fade };
  }
  const partFade = new Float32Array(frameBytes, 1056, 32);   // frame.part_fade (common.wgsl): 0 shown, 1 gone
  if (window.gsDebug && cloudParts.length) Object.defineProperty(window.gsDebug, 'cloudDist',
    { get: () => cloudDist, set: v => { cloudDist = v; touch(); }, configurable: true });
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
  // a box's slider pose (partTransform matrix and yaw), for placing the physics bodies
  function sliderXf(i) {
    const p = header.parts[i], keys = rigOf[p.rig_name ?? p.name];
    return keys ? { m: partTransform(p.pivot ?? [0, 0, 0], ...keys.map(k => state[k])), yaw: state[keys[2]] } : null;
  }
  const staticWarned = new Set();
  function writeParts() {
    let anyMoved = false;
    let movedMask = 0;
    header.parts.forEach((p, i) => {
      const o = i * 32;
      const prev = pf.slice(o, o + 12);
      const keys = rigOf[p.rig_name ?? p.name];
      const body = state.physics && physics?.xf ? physics.bodyOf.get(i) : undefined;
      if (body !== undefined) {
        for (let r = 0; r < 3; r++) pf.set(physics.xf.subarray(12 * body + 4 * r, 12 * body + 4 * r + 4), o + 4 * r);
      } else if (clipSlot.has(i)) {
        const c = clipRows(clipSlot.get(i));
        const m = partTransform(torsoPivot, state.armX, state.armY, state.armYaw);   // column-major 4x4
        for (let r = 0; r < 3; r++) {
          const row = [0, 1, 2, 3].map(j => m[r] * c[j] + m[4 + r] * c[4 + j] + m[8 + r] * c[8 + j] + (j === 3 ? m[12 + r] : 0));
          pf.set(row, o + 4 * r);
        }
      } else if (keys) {
        const m = partTransform(p.pivot ?? [0, 0, 0], ...keys.map(k => state[k]));
        for (let r = 0; r < 3; r++) pf.set([m[r], m[4 + r], m[8 + r], m[12 + r]], o + 4 * r);
      } else if (sparePart.has(i)) {
        // a spare: parked (its file transform) while hidden, else at its pivot pose
        const s = sparePart.get(i);
        pf.set(s.shown ? spareXf(i, s.x, s.y, s.yaw) : new Float32Array(asset.parts, i * 128, 12), o);
        pu[o + 31] = (pu[o + 31] & ~4) | (s.shown ? 0 : 4);
      } else if (window.gsDebug?.partOffset?.[i]) {
        // debug / scripted tests: translate a part with no UI rig (e.g. a hill cloud) by gsDebug.partOffset[i] (m)
        const t = window.gsDebug.partOffset[i];
        for (let r = 0; r < 3; r++) pf.set([+(r === 0), +(r === 1), +(r === 2), t[r]], o + 4 * r);
      } else if (cloudOf.has(i)) {
        const { t, fade } = cloudPose(cloudOf.get(i));
        for (let r = 0; r < 3; r++) pf.set([+(r === 0), +(r === 1), +(r === 2), t[r]], o + 4 * r);
        partFade[i] = 1 - fade;
        pu[o + 31] = (pu[o + 31] & ~4) | (fade <= 0 ? 4 : 0);   // hidden while it wraps round (not drawn, no shadow work)
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
    // GPU physics: its steps and poses into partBuf; the parts it moves (its awake bodies) join the moved mask
    const gpuMoved = physics?.writePoses?.(renderer.partBuf, state.physics) ?? 0;
    movedMask |= gpuMoved;
    lighting.gpuMovedMask = gpuMoved;   // poses written on the GPU: lighting.js isoMoved cannot see their rotation
    anyMoved ||= movedMask !== 0;
    lighting.movedMask = movedMask;   // light_shadow.js re-splats only these parts' layers
    return anyMoved;
  }

  // the lenses in view space for this frame's camera and part poses (refract.js); off without glass parts
  function syncLenses(width, height) {
    const name = ['off', 'A', 'B', 'C', 'AC', 'BC', 'E', 'EC', 'EG'][state.refractMode];
    const mode = lenses.length ? (MODES[name] ?? 0) : 0;
    renderer.refract = mode;
    renderer.fastLayers = state.fastLayers;
    const focal = height / (2 * Math.tan(camera.fovy / 2));
    writeLenses(lensBytes, lenses, pf, camera.view(), [width, height], [focal, focal],
      { ior: state.ior, dispersion: state.dispersion, absorb: state.absorb, frontMargin: 0.15, mode, bend: state.bend, compact: state.fastLayers && mode !== 0 && mode !== 1 && mode !== 4 });
    renderer.writeLenses(lensBytes);
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
  // FPS ticker: frames drawn per half second. The page renders on demand, so it reads "idle" once nothing is changing.
  const fpsEl = $('fps');
  let fpsFrames = 0, fpsT0 = performance.now();
  setInterval(() => {
    const t = performance.now();
    fpsEl.textContent = fpsFrames ? `${Math.round(fpsFrames * 1000 / (t - fpsT0))} fps` : 'idle';
    fpsFrames = 0; fpsT0 = t;
  }, 500);
  function showTimers() {
    const lines = timers.enabled
      ? PASSES.map(n => `${n.padEnd(9)} ${timers.ms[n].toFixed(3)} ms`)
      : ['timestamp-query unavailable'];
    if (timers.enabled) lines.push(`${'gpu sum'.padEnd(9)} ${PASSES.reduce((s, n) => s + timers.ms[n], 0).toFixed(3)} ms`);
    lines.push(`frame    ${cpuMs.toFixed(2)} ms (rAF interval)`);
    if (physics && state.physics) {
      lines.push(`physics  ${physics.ms.toFixed(1)} ms/step, ${physics.pairs.toLocaleString()} pairs${physics.awake ? '' : ', asleep'}${physics.incidents ? `, ${physics.incidents} incident(s): gsDebug.physics.lastIncident` : ''}`);
      const b = physics.budget;
      if (b?.budget > 0) lines.push(`budget   ${b.used.toLocaleString()} / ${b.budget.toLocaleString()} pairs, coarsest level L${b.coarsest}${b.coarseImpacts ? `, ${b.coarseImpacts} impact(s) coarse` : ''}${b.overrun ? ', over' : ''}`);
    }
    lines.push(`splats   ${timers.extra[1].toLocaleString()} visible / ${asset.count.toLocaleString()}`);
    lines.push(`target   ${canvas.width}x${canvas.height}`);
    hud.textContent = lines.join('\n');
  }

  let pending = false;
  // frames submitted and not finished on the GPU: at most MAX_IN_FLIGHT, so an overloaded GPU doesn't queue up seconds
  // of stale frames; a frame that finds the queue full waits for one to finish and then renders the newest state
  // (2026-10-31; latency, not throughput)
  const MAX_IN_FLIGHT = 2;
  let inFlight = 0, waitingGpu = false;
  // ?raf=timeout keeps frames coming in a hidden tab, where requestAnimationFrame stops (benchmarks)
  const nextFrame = params.get('raf') === 'timeout' ? cb => setTimeout(() => cb(performance.now()), 0) : requestAnimationFrame;
  function schedule() {
    if (!pending && !stopped) { pending = true; nextFrame(frame); }
  }
  function frame(now) {
    pending = false;
    if (stopped || window.gsDebug.harness) return;   // debug: a capture script owns the frame (inert unless set)
    if (inFlight >= MAX_IN_FLIGHT) { waitingGpu = true; return; }
    resize();
    const playing = clip && state.armPlay;
    const drifting = (cloudParts.length > 0 && state.cloudWind > 0) || (renderer.particles && !renderer.particles.paused);   // clouds and particles keep the frames coming
    if (!dirty && !state.continuous && !playing && !drifting) return;
    dirty = false;
    // Only back-to-back frames say anything about frame rate; skip idle gaps.
    const dt = now - lastFrame;
    if (playing && dt < 250) clipTime += dt / 1000 * state.armSpeed;
    if (drifting && dt < 250) cloudDist += dt / 1000 * state.cloudWind;
    if (renderer.particles) {
      // a fixed step (?pstep=<s>, captures) or the frame time
      const pstep = params.has('pstep') ? Number(params.get('pstep')) : (dt < 250 ? dt / 1000 : 0);
      if (renderer.particles.pendingWarm) { renderer.particles.warm(renderer.particles.pendingWarm, camera.eye(), 1 / 60); renderer.particles.pendingWarm = 0; }
      const L = asset.lamps[0], lampOn = L && !lighting.lampOff;
      const plamp = lampOn ? { center: [state.lampX, state.lampY, state.lampZ], area: 4 * Math.hypot(...L.e1) * Math.hypot(...L.e2), radiance: L.radiance } : null;
      renderer.particles.update(renderer.particles.paused ? 0 : pstep, camera.eye(), plamp);
      u32.set([renderer.particleBase, renderer.particleCount, renderer.particleOffset / 64], 296);
    }
    if (dt < 250) cpuMs = cpuMs ? cpuMs * 0.9 + dt * 0.1 : dt;
    lastFrame = now;
    fpsFrames++;
    lighting.emitMode = state.lightCut ? 1 : 0;
    lighting.asgSpec = state.asgSpec ? (state.specCut ? 3 : 1) : 0;
    lighting.emitEdgeSoft = state.edgeSoft;
    if (state.oneSided !== lighting.emitOneSided) lighting.setOneSided(state.oneSided);
    syncLights();
    const emitKeyBefore = lighting.emitKey;
    emitterCount = lighting.setEmitters([state.lampX, state.lampY, state.lampZ]);
    writeFrame(canvas.width, canvas.height);
    const partsMoved = writeParts();
    const envChanged = environment?.set({ on: state.envOn, sunAz: state.sunAz, sunEl: state.sunEl, skyGain: state.skyGain, lightGain: state.sunGain }) ?? false;
    const lightChanged = partsMoved || envChanged || lighting.emitKey !== emitKeyBefore || !!explode?.ended;
    if (explode?.ended && !exploding()) { explode.ended = false; touch(); }   // the lighting resumes and settles
    timers.begin();
    const encoder = device.createCommandEncoder();
    if (state.viewMode === LIT && !exploding()) {   // Explode: the lit colours stay as they were
      const center = [state.lampX, state.lampY, state.lampZ];
      lighting.setLamp(0, center, lampPrev ?? center);
      lampPrev = center;
      if (state.nearField !== lighting.nearField) lighting.setNearField(state.nearField);
      if (state.nearRefl !== lighting.nearRefl) lighting.setNearReflection(state.nearRefl);
    if (state.emitBounceFix !== lighting.emitterBounceFix) lighting.setEmitterBounceFix(state.emitBounceFix);
      if (coneCount(state.nearCones) !== lighting.nearCones) lighting.setNearCones(coneCount(state.nearCones));
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
      {
        if (lod?.cut) lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);   // ?lodlight=2
        // Unchanged inputs, unchanged lighting (2026-10-31): after SETTLE_FRAMES frames with no lighting input changed
        // (settings, parts, lights, sky) the view-independent passes stop (light toward the probes, tree, probes,
        // shadow maps); once the camera has also held still that long, the lit colours are kept as they are. Any
        // change runs every pass again in that frame, so one-frame convergence is unchanged.
        const litKey = JSON.stringify(state);
        lightQuiet = !lightChanged && litKey === litKeyPrev ? lightQuiet + 1 : 0;
        litKeyPrev = litKey;
        const viewKey = [...camera.view(), canvas.width, canvas.height].join();
        viewQuiet = viewKey === viewKeyPrev ? viewQuiet + 1 : 0;
        viewKeyPrev = viewKey;
        const lightStill = lightQuiet >= SETTLE_FRAMES && state.stillSkip !== false, allStill = lightStill && viewQuiet >= SETTLE_FRAMES;
        setCamForLighting();   // sun_map.js cascades sit ahead of the camera
        if (!allStill) {
          lighting.encode(encoder, camera.eye(), timers, lightChanged, lightStill);
          if (state.bounce > 0) lighting.encodeBounce(encoder, timers, lightStill);
          lighting.encodeFill(encoder);
          lod?.encode(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod, partsMoved);
        } else lod?.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod, partsMoved);
        lighting.stillFrames = allStill ? (lighting.stillFrames ?? 0) + 1 : 0;   // inspection (smoke_renderer.mjs still_skip)
      }
      if (flakeOn()) flake().encode(encoder, timers, state.bounce, state.bounce > 0);
    }
    // the LOD cut is the camera's, whatever the view: flat, depth and overdraw views, an explosion and one-path lighting
    // skip the lighting's own select (2026-10-31: flat mode drew the last lit view's cut)
    if (lod && (state.viewMode !== LIT || exploding())) {
      lod.colourAll = exploding();   // an explosion keeps the last lit colours while its cut follows the camera: colour every node
      lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod, partsMoved);
      if (exploding() && state.lodPx > 0) lod.encodeColour(encoder);
    }
    renderer.useFlake(flakeOn());
    renderer.stableSort = state.stableSort;
    syncLenses(canvas.width, canvas.height);
    renderer.encode(encoder, context.getCurrentTexture().createView(), frameBytes, state.viewMode, timers);
    const readback = timers.end(encoder, renderer.drawBuf);
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    inFlight++;
    device.queue.onSubmittedWorkDone().then(() => { inFlight--; if (waitingGpu) { waitingGpu = false; schedule(); } });
    // Inspection: submit-to-done time per frame (GPU work + queue latency), not limited by vsync.
    if (window.gsDebug?.gpuTimes) device.queue.onSubmittedWorkDone().then(() => window.gsDebug.gpuTimes?.push(performance.now() - t0));
    timers.collect(readback, showTimers);
    if (++frames === 1) $('status').textContent = `${header.format} ${header.version} · ${assetName} · ${asset.count.toLocaleString()} Gaussians, ${asset.partCount} parts`;
    if (state.viewMode === LIT && settle > 0) { settle--; dirty = true; }
    if (exploding()) dirty = true;
    if (state.continuous || dirty || playing || drifting) schedule();
  }
  let lampPrev = null, roughnessSet = null, boxSet = null, transmitSet = null, emitterCount = 0;
  let lightQuiet = 0, viewQuiet = 0, litKeyPrev = null, viewKeyPrev = null;   // frame(): unchanged lighting inputs / camera

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
    // the lighting settings the frame loop syncs before it encodes; without this a capture on a page whose loop never
    // ran (hidden tab) keeps the constructor's nearField = true and walks near_vis for every Gaussian in one submit
    if (state.nearField !== lighting.nearField) lighting.setNearField(state.nearField);
    if (state.nearRefl !== lighting.nearRefl) lighting.setNearReflection(state.nearRefl);
    if (state.emitBounceFix !== lighting.emitterBounceFix) lighting.setEmitterBounceFix(state.emitBounceFix);
    if (coneCount(state.nearCones) !== lighting.nearCones) lighting.setNearCones(coneCount(state.nearCones));
    syncLights();
    lighting.setEmitters([state.lampX, state.lampY, state.lampZ]);
    for (let f = 0; f < frames; f++) {
      writeFrame(size, size);
      writeParts();
      const encoder = device.createCommandEncoder();
      const noTimers = { writes: () => undefined };
      if (state.viewMode === LIT) {
        if (lod?.cut) lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);   // ?lodlight=2
        if (!window.gsDebug.freezeLight) {   // debug: keep the settled lighting, redo only the LOD cut (inert unless set)
          setCamForLighting();
          lighting.encode(encoder, camera.eye(), noTimers);
          if (state.bounce > 0) lighting.encodeBounce(encoder, noTimers);
          lighting.encodeFill(encoder);
        }
        lod?.encode(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);
        if (flakeOn()) flake().encode(encoder, noTimers, state.bounce, state.bounce > 0);
      } else if (lod) lod.encodeSelect(encoder, camera.view(), state.lodPx, lodView.focal, lodView.near, state.bounce, state.blurShutter, state.blurLod);   // other views: the cut only
      renderer.useFlake(flakeOn());
      const out = device.createTexture({ size: [size, size], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      renderer.stableSort = state.stableSort;
      syncLenses(size, size);
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
