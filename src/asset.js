// Asset loader for the shared runtime format (runtime/SPEC.md v0.1): asset.json
// plus raw little-endian .bin buffers that upload to the GPU unchanged. The
// renderer draws from gaussians and parts; the lighting kernels (runtime/wgsl)
// also read nodes, kids, vgrid, lamps and materials.

const MAJOR = '0';
// The fields the renderer's WGSL structs assume (shaders/common.wgsl).
const EXPECT = {
  gaussians: { stride: 64, fields: { mu: 0, kappa: 12, quat: 16, scale: 32, part_surface: 44, albedo: 48 } },
  parts: { stride: 128, fields: { xf: 0, xf_prev: 48, bound: 96, root: 112, g_lo: 116, g_hi: 120, flags: 124 } },
  lamps: { stride: 96, fields: { center: 0, kind: 12, e1: 16, radius: 28, e2: 32, normal: 48, radiance: 64, center_prev: 80 } },
  materials: { stride: 32, fields: { flake: 0, phase_samples: 12, vfloor: 16, flags: 20 } },
  nodes: { stride: 96, fields: {} },
  kids: { stride: 4, fields: {} },
  vgrid: { stride: 128, fields: {} },
};

export function checkHeader(header) {
  if (header.format !== 'gs-runtime') throw new Error(`Unknown asset format ${header.format}`);
  if (String(header.version).split('.')[0] !== MAJOR) throw new Error(`Unsupported asset version ${header.version}`);
  for (const [name, want] of Object.entries(EXPECT)) {
    const b = header.buffers?.[name];
    if (!b) throw new Error(`Asset has no ${name} buffer`);
    if (b.stride !== want.stride) throw new Error(`${name} stride ${b.stride}, renderer expects ${want.stride}`);
    for (const [f, off] of Object.entries(want.fields)) {
      if (b.fields?.[f]?.offset !== off) throw new Error(`${name}.${f} at ${b.fields?.[f]?.offset}, renderer expects ${off}`);
    }
    if (!Number.isInteger(b.count) || b.count <= 0 || b.count > (1 << 24)) throw new Error(`Bad ${name} count ${b.count}`);
  }
}

export function checkBuffer(header, name, bytes) {
  const b = header.buffers[name];
  if (bytes.byteLength !== b.count * b.stride) {
    throw new Error(`${b.file} is ${bytes.byteLength} B, expected ${b.count} x ${b.stride} B`);
  }
  return bytes;
}

// Lamp records as plain objects, for the UI defaults.
export function readLamps(header, bytes) {
  const v = new DataView(bytes);
  const vec = o => [0, 1, 2].map(k => v.getFloat32(o + 4 * k, true));
  return Array.from({ length: header.buffers.lamps.count }, (_, i) => {
    const o = i * 96;
    return { center: vec(o), kind: v.getUint32(o + 12, true), e1: vec(o + 16), radius: v.getFloat32(o + 28, true),
      e2: vec(o + 32), normal: vec(o + 48), radiance: vec(o + 64) };
  });
}

export async function loadAsset(base) {
  const res = await fetch(`${base}/asset.json`);
  if (!res.ok) throw new Error(`Could not load ${base}/asset.json (${res.status})`);
  const header = await res.json();
  checkHeader(header);
  const load = async name => {
    const r = await fetch(`${base}/${header.buffers[name].file}`);
    if (!r.ok) throw new Error(`Could not load ${base}/${header.buffers[name].file} (${r.status})`);
    return checkBuffer(header, name, await r.arrayBuffer());
  };
  const names = ['gaussians', 'parts', 'lamps', 'materials', 'nodes', 'kids', 'vgrid'];
  const raw = Object.fromEntries(names.map((n, i) => [n, null]));
  (await Promise.all(names.map(load))).forEach((b, i) => { raw[names[i]] = b; });
  // optional: own-part blockers for near-field reflection (runtime/bake_blockers.py, 64 B per Gaussian)
  raw.blockers = null;
  const bk = header.blockers;
  if (bk?.file && bk.stride === 64 && bk.count === header.buffers.gaussians.count) {
    const r = await fetch(`${base}/${bk.file}`);
    if (r.ok) {
      const bytes = await r.arrayBuffer();
      if (bytes.byteLength === bk.count * 64) raw.blockers = bytes;
    }
  }
  return {
    header, raw, gaussians: raw.gaussians, parts: raw.parts, lamps: readLamps(header, raw.lamps),
    count: header.buffers.gaussians.count, partCount: header.buffers.parts.count, fingerprint: fingerprint(header, raw.gaussians),
  };
}

// Which version of the asset this is, for baked sidecars (pfield.json, experiments/M42_many_lights/pf_export.py
// fingerprint): Gaussian count, part ranges, gaussians.bin bytes and sum_i w_i * (i * 2654435761 + 1) mod 2^32 over
// its u32 words. Taken at load, before the renderer edits the Gaussian buffer in place.
function fingerprint(header, gaussians) {
  const w = new Uint32Array(gaussians);
  let h = 0;
  for (let i = 0; i < w.length; i++) h = (h + Math.imul(w[i], Math.imul(i, 2654435761) + 1)) >>> 0;
  return { count: header.buffers.gaussians.count, parts: header.parts.map(p => p.gaussians), bytes: gaussians.byteLength, hash: h };
}
