// Faked refraction (src/refract.js, ?refract=1): ellipsoid lenses in view space, shared by splat.wgsl (layer
// sorting) and refract.wgsl (the composite). u = n (x - c) maps a lens to the unit sphere.

struct Lens {
  c: vec4f,              // view-space centre, w = bounding radius
  n: array<vec4f, 3>,    // rows of N (xyz)
  rect: vec4f,           // pixel rect (x0, y0, x1, y1), y down
}
struct Lenses {
  info: vec4u,           // lens count, dispersion on, mode bits (REFR_*)
  params: vec4f,         // ior, dispersion (ior spread), absorption 1/m, front margin (fraction of the chord)
  params2: vec4f,        // B: bend (offset px = bend * grad tau * H^2 / 1000)
  lens: array<Lens, 8>,
}

const REFR_LENS: u32 = 1u;    // A: composite through each lens (three layers)
const REFR_GRAD: u32 = 2u;    // B: offset by the glass splats' optical-depth gradient (no shape)
const REFR_WARP: u32 = 4u;    // C: project.wgsl moves Gaussians inside a lens to where the lens shows them
const REFR_EXIT: u32 = 8u;    // E: bend by the exit map (exitmap.wgsl), traced per glass skin Gaussian
const REFR_GATHER: u32 = 16u; // E + G: what is inside the glass is gathered per pixel along the mapped inside ray (refract.wgsl gather_inside), not splatted
const SPLAT_GLASS: u32 = 1u;  // splat flags, the low bits of the depth word (project.wgsl)
const SPLAT_WARPED: u32 = 2u;

fn inv3(m: mat3x3f) -> mat3x3f {
  let a = m[0]; let b = m[1]; let c = m[2];
  let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
  return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(a, r0));
}

// N as a matrix (rows n[0..2])
fn lens_mat(l: Lens) -> mat3x3f { return transpose(mat3x3f(l.n[0].xyz, l.n[1].xyz, l.n[2].xyz)); }

fn lens_n(l: Lens, v: vec3f) -> vec3f { return vec3f(dot(l.n[0].xyz, v), dot(l.n[1].xyz, v), dot(l.n[2].xyz, v)); }
// N^T v: the view-space gradient direction of |u|^2 (outward normal, unnormalised)
fn lens_nt(l: Lens, v: vec3f) -> vec3f { return l.n[0].xyz * v.x + l.n[1].xyz * v.y + l.n[2].xyz * v.z; }

// Ray o + t d (view space) against the lens: (t_near, t_far), or t_near > t_far for a miss.
// C: where a view-space point inside lens l appears through the lens's near surface (one spherical interface in the
// lens's unit-sphere frame, paraxial: 1/s' = (n - 1) - n/s from the front vertex; lateral offset magnified and
// compressed, rho' = tanh(m rho), so images stay inside the silhouette). The warp fades to none over the outer
// WARP_BAND of the radius, so a Gaussian crossing the surface moves smoothly. minv = inv3(lens_mat(l)).
const WARP_BAND = 0.15;
fn lens_warp(l: Lens, minv: mat3x3f, x: vec3f, n: f32) -> vec3f {
  let u = lens_n(l, x - l.c.xyz);
  let r = length(u);
  if (r >= 1.0) { return x; }
  let zh = normalize(lens_n(l, -l.c.xyz));   // toward the eye
  let a = dot(u, zh);
  let s = clamp(1.0 - a, 1e-3, 2.0);
  let sp = 1.0 / (n - 1.0 - n / s);
  let m0 = clamp(-n * sp / s, 0.25, 4.0);
  let lat = u - a * zh;
  let rho = length(lat);
  let mag = select(m0, tanh(m0 * rho) / max(rho, 1e-6), rho > 1e-4);
  let u2 = (1.0 + sp) * zh + mag * lat;
  return l.c.xyz + minv * mix(u2, u, smoothstep(1.0 - WARP_BAND, 1.0, r));
}

fn lens_hit(l: Lens, o: vec3f, d: vec3f) -> vec2f {
  let ou = lens_n(l, o - l.c.xyz);
  let du = lens_n(l, d);
  let a = dot(du, du); let b = dot(ou, du); let c = dot(ou, ou) - 1.0;
  let disc = b * b - a * c;
  if (disc <= 0.0) { return vec2f(1.0, -1.0); }
  let s = sqrt(disc);
  return vec2f((-b - s) / a, (-b + s) / a);
}

fn in_rect(l: Lens, p: vec2f) -> bool { return all(p >= l.rect.xy) && all(p <= l.rect.zw); }

// The eye ray through pixel p (y down) in view space, z = -1 (so t is view depth).
fn eye_ray(p: vec2f) -> vec3f {
  return vec3f((p.x - 0.5 * frame.viewport.x) / frame.focal.x, -(p.y - 0.5 * frame.viewport.y) / frame.focal.y, -1.0);
}
