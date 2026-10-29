import { lookAt } from './math.js';

// Orbit camera around a target, z up. Drag to orbit, shift- or right-drag to
// pan, wheel to dolly.
export class OrbitCamera {
  constructor(sceneCamera) {
    this.home = sceneCamera;
    this.reset();
  }

  // Start at the classic Cornell view: the scene's eye, orbiting the room centre.
  reset() {
    const { eye, look_at: at, fov_deg: fov } = this.home;
    this.target = [at[0], 1.0, at[2]];
    const d = [eye[0] - this.target[0], eye[1] - this.target[1], eye[2] - this.target[2]];
    this.distance = Math.hypot(...d);
    this.yaw = Math.atan2(d[1], d[0]);
    this.pitch = Math.asin(d[2] / this.distance);
    this.fovy = fov * Math.PI / 180;
  }

  // A fixed view (captures): orbit parameters that put the eye at `eye` looking at `at`.
  setView(eye, at, fovDeg = this.home.fov_deg) {
    this.target = [...at];
    const d = [eye[0] - at[0], eye[1] - at[1], eye[2] - at[2]];
    this.distance = Math.hypot(...d);
    this.yaw = Math.atan2(d[1], d[0]);
    this.pitch = Math.asin(d[2] / this.distance);
    this.fovy = fovDeg * Math.PI / 180;
  }

  eye() {
    const c = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * c * Math.cos(this.yaw),
      this.target[1] + this.distance * c * Math.sin(this.yaw),
      this.target[2] + this.distance * Math.sin(this.pitch),
    ];
  }

  view() {
    return lookAt(this.eye(), this.target, [0, 0, 1]);
  }

  attach(canvas, onChange) {
    let last = null, panning = false;
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      canvas.setPointerCapture(e.pointerId);
      last = [e.clientX, e.clientY];
      panning = e.button === 2 || e.shiftKey;
    });
    canvas.addEventListener('pointerup', () => { last = null; });
    canvas.addEventListener('pointermove', e => {
      if (!last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      if (panning) {
        const v = this.view();
        const s = this.distance * 0.0015;
        for (let k = 0; k < 3; k++) this.target[k] += (-dx * v[k * 4] + dy * v[k * 4 + 1]) * s;
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.005));
      }
      onChange();
    });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.distance = Math.max(0.05, Math.min(50, this.distance * Math.exp(e.deltaY * 0.001)));
      onChange();
    }, { passive: false });
  }
}
