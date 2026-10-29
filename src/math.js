// Column-major 4x4 matrices in Float32Array(16), matching WGSL mat4x4f.
// World frame: right-handed, z up, metres. Views look down -z.

export const identity = () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function translation(x, y, z) {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

export function rotationZ(rad) {
  const m = identity();
  const c = Math.cos(rad), s = Math.sin(rad);
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
  return m;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = a => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

export function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

// WebGPU clip space (z in [0, 1]); kept for later passes, the splat path
// projects with the focal lengths directly.
export function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = far / (near - far); m[11] = -1;
  m[14] = near * far / (near - far);
  return m;
}

// Rigid transform about a floor pivot: T(pivot + offset) Rz(yaw) T(-pivot).
export function partTransform(pivot, dx, dy, yawDeg) {
  const t = translation(pivot[0] + dx, pivot[1] + dy, pivot[2]);
  return multiply(multiply(t, rotationZ(yawDeg * Math.PI / 180)), translation(-pivot[0], -pivot[1], -pivot[2]));
}

export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}
