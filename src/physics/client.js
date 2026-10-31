// Main-thread side of the live physics: starts worker.js for an asset folder with physics.json, keeps the latest
// world_from_rest rows of every physics body, and turns clicks on the canvas into pushes along the view ray.
import { poseFromYawXf } from './scenario.js';

export class PhysicsClient {
  // resolves to null when the asset has no physics bake (or workers fail)
  // budget: node pairs per frame for the adaptive precision (world.js assignLevels), 0 = full detail everywhere
  static async start(base, budget = 0) {
    let worker;
    try { worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' }); } catch { return null; }
    const meta = await new Promise(resolve => {
      worker.onmessage = e => {
        if (e.data.type === 'selfcheck') logSelfCheck(e.data.report);
        else if (e.data.type === 'ready') resolve(e.data.meta);
        else if (e.data.type === 'none') resolve(null);
      };
      worker.onerror = e => { console.warn('physics worker failed:', e.message); resolve(null); };
      worker.postMessage({ type: 'init', base: new URL(base, location.href).href, budget });
    });
    if (!meta) { worker.terminate(); return null; }
    return new PhysicsClient(worker, meta);
  }

  constructor(worker, meta) {
    this.worker = worker;
    this.meta = meta;
    this.xf = null;          // Float64Array(12 B): rows [R | t] per body, null until the worker has answered
    this.bodies = null;      // Float64Array(14 B): pos, quat, vel, om, asleep per body (hand-overs, physics/auto_client.js)
    this.ms = 0; this.pairs = 0; this.awake = false; this.budget = null; this.incidents = 0; this.lastIncident = null;
    this.onState = () => {};
    this.bodyOf = new Map(meta.parts.filter(p => !p.static).map(p => [p.asset_index, meta.parts.indexOf(p)]));   // asset part -> body
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'incident') {
        this.lastIncident = m.record;
        console.warn(`physics incident ${m.record.incidents}: ${m.record.bodies.join(', ')} below the floor / out of the room; `
          + 'last steps and UI events in gsDebug.physics.lastIncident (put back at its last good pose)', m.record);
        return;
      }
      if (m.type !== 'state') return;
      Object.assign(this, { xf: m.xf, bodies: m.bodies, ms: m.ms, pairs: m.pairs, awake: m.awake, budget: m.budget, incidents: m.incidents ?? 0 });
      this.onState();
    };
  }

  // a random Drop (scenario.js drop) unless a seed is given; the seed goes to the console so a start can be replayed
  drop(seed = Math.floor(Math.random() * 2 ** 32)) {
    console.info(`physics Drop, seed ${seed} (gsDebug.physics.drop(${seed}) replays it)`);
    this.worker.postMessage({ type: 'drop', seed });
  }
  reset() { this.worker.postMessage({ type: 'reset' }); }
  // a hand-over: the bodies (14 per body: pos, quat, vel, om, asleep) as another path left them
  setBodies(bodies) { this.worker.postMessage({ type: 'bodies', bodies }); }

  // place the bodies where the box sliders put them (partTransform matrices, yaw in degrees), at rest
  place(sliderXf) {
    this.xf = null;
    const poses = [];
    for (const [ai, b] of this.bodyOf) {
      const s = sliderXf(ai);
      if (!s) continue;
      const [p, q] = poseFromYawXf(this.meta.parts[b], s.m, s.yaw);
      poses.push({ b, p, q });
    }
    this.worker.postMessage({ type: 'poses', poses });
  }

  // a click (no drag) on the canvas while enabled() pushes the first box under the cursor at `speed` m/s along the ray
  attachClicks(canvas, camera, enabled, speed = 2) {
    let down = null;
    canvas.addEventListener('pointerdown', e => { down = e.button === 0 && !e.shiftKey ? [e.clientX, e.clientY] : null; });
    canvas.addEventListener('pointerup', e => {
      if (!down || !enabled() || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const r = canvas.getBoundingClientRect();
      const x = 2 * (e.clientX - r.left) / r.width - 1, y = 1 - 2 * (e.clientY - r.top) / r.height;
      const o = camera.eye(), t = camera.target;
      const f = norm([t[0] - o[0], t[1] - o[1], t[2] - o[2]]);
      const s = norm(cross(f, [0, 0, 1])), u = cross(s, f);
      const ty = Math.tan(camera.fovy / 2), tx = ty * r.width / r.height;
      const d = norm([0, 1, 2].map(k => f[k] + x * tx * s[k] + y * ty * u[k]));
      this.worker.postMessage({ type: 'push', o, d, speed });
    });
  }
}

// the worker's startup self-check (world.js selfCheck): a box standing on the floor must find contact pairs with it
function logSelfCheck(r) {
  const lines = r.dynamic.map(d => `${d.name}${d.onFloor ? ' (on the floor)' : ''}: ${Object.entries(d.pairs).map(([k, v]) => `${k} ${v}`).join(', ') || 'no pairs'}`);
  if (r.ok) console.info(`physics self-check OK: ${r.pairs} node pairs at the rest pose; ${lines.join('; ')}`, r);
  else console.error(`physics self-check FAILED: a box on the floor has no contact pairs with it, so it will fall through. ${lines.join('; ')}`, r);
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = v => { const l = Math.hypot(...v); return v.map(c => c / l); };
