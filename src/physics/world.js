// Live rigid-body physics on Gaussian parts: a port of the prototype's World (experiments/physics_proto/world.py and
// kernels.py: option A tree-overlap contacts, mode 3, pair-normal snap, swept time-of-impact contact, positional
// Coulomb friction, depenetration cap, relaxation, alternating Gauss-Seidel sweep, sleeping). Plain ES module, no
// dependencies, float64 throughout, allocation-free in step(). The collision trees come from the bake
// (experiments/physics_proto/bake_physics.py: physics.json + physics.bin), already cut at the collision level.
//
// Bodies are in their principal frame: body point = R0^T (x_rest - com0); pose (pos, quat wxyz).

const PI32 = (2 * Math.PI) ** 1.5;
const NS = 16;   // node stride (float64): mean 0-2, cov 3-8 (xx xy xz yy yz zz), mass 9, reach 10, size 11, thick 12, left 13, right 14, off 15
const STACK = 8192;
const REPS = 96;
const SEP_FRAMES = 5;   // lockLevels: frames apart before a pair's locked level is released
const ENERGY_V = 0.3, ENERGY_EPS = 1e-4;   // energy limiter: below this speed-equivalent kinetic energy; tolerance (J)   // compact contacts: representatives per body pair at most
const CALM_FRAMES = 5, CALM_V = 0.1, EXPAND_V = 0.15, EXPAND_G = 0.005, REFRESH = 30;

// scratch (module level: step() allocates nothing)
const OV = new Float64Array(8);   // overlap: O, grad xyz, contact point xyz, r
const A6 = new Float64Array(6), B6 = new Float64Array(6), OS = new Float64Array(6);
const WA = new Float64Array(4), WB = new Float64Array(4);
const T3 = new Float64Array(3);

function quatMat(q, qo, R, ro) {
  const w = q[qo], x = q[qo + 1], y = q[qo + 2], z = q[qo + 3];
  R[ro] = 1 - 2 * (y * y + z * z); R[ro + 1] = 2 * (x * y - w * z); R[ro + 2] = 2 * (x * z + w * y);
  R[ro + 3] = 2 * (x * y + w * z); R[ro + 4] = 1 - 2 * (x * x + z * z); R[ro + 5] = 2 * (y * z - w * x);
  R[ro + 6] = 2 * (x * z - w * y); R[ro + 7] = 2 * (y * z + w * x); R[ro + 8] = 1 - 2 * (x * x + y * y);
}

// R S R^T as (xx, xy, xz, yy, yz, zz), S = the node's covariance
function rotCov6(R, ro, N, nb, out, oo = 0) {
  const s00 = N[nb + 3], s01 = N[nb + 4], s02 = N[nb + 5], s11 = N[nb + 6], s12 = N[nb + 7], s22 = N[nb + 8];
  const r00 = R[ro], r01 = R[ro + 1], r02 = R[ro + 2], r10 = R[ro + 3], r11 = R[ro + 4], r12 = R[ro + 5];
  const r20 = R[ro + 6], r21 = R[ro + 7], r22 = R[ro + 8];
  const t00 = r00 * s00 + r01 * s01 + r02 * s02, t01 = r00 * s01 + r01 * s11 + r02 * s12, t02 = r00 * s02 + r01 * s12 + r02 * s22;
  const t10 = r10 * s00 + r11 * s01 + r12 * s02, t11 = r10 * s01 + r11 * s11 + r12 * s12, t12 = r10 * s02 + r11 * s12 + r12 * s22;
  const t20 = r20 * s00 + r21 * s01 + r22 * s02, t21 = r20 * s01 + r21 * s11 + r22 * s12, t22 = r20 * s02 + r21 * s12 + r22 * s22;
  out[oo + 0] = t00 * r00 + t01 * r01 + t02 * r02;
  out[oo + 1] = t00 * r10 + t01 * r11 + t02 * r12;
  out[oo + 2] = t00 * r20 + t01 * r21 + t02 * r22;
  out[oo + 3] = t10 * r10 + t11 * r11 + t12 * r12;
  out[oo + 4] = t10 * r20 + t11 * r21 + t12 * r22;
  out[oo + 5] = t20 * r20 + t21 * r21 + t22 * r22;
}

// overlap of two Gaussian densities (world space): OV = O, dO/dmu_a, contact point, r; returns O
function overlap(ma, ax, ay, az, a6, ao, mb, bx, by, bz, b6, bo) {
  const sxx = a6[ao] + b6[bo], sxy = a6[ao + 1] + b6[bo + 1], sxz = a6[ao + 2] + b6[bo + 2];
  const syy = a6[ao + 3] + b6[bo + 3], syz = a6[ao + 4] + b6[bo + 4], szz = a6[ao + 5] + b6[bo + 5];
  const c00 = syy * szz - syz * syz, c01 = sxz * syz - sxy * szz, c02 = sxy * syz - sxz * syy;
  const det = sxx * c00 + sxy * c01 + sxz * c02;
  const c11 = sxx * szz - sxz * sxz, c12 = sxy * sxz - sxx * syz, c22 = sxx * syy - sxy * sxy;
  const idet = 1 / det;
  const dx = ax - bx, dy = ay - by, dz = az - bz;
  const yx = (c00 * dx + c01 * dy + c02 * dz) * idet;
  const yy = (c01 * dx + c11 * dy + c12 * dz) * idet;
  const yz = (c02 * dx + c12 * dy + c22 * dz) * idet;
  const q = dx * yx + dy * yy + dz * yz;
  if (q > 40) { OV[0] = 0; return 0; }
  const r = Math.exp(-0.5 * q);
  const O = ma * mb * r / (PI32 * Math.sqrt(det));
  OV[0] = O; OV[1] = -O * yx; OV[2] = -O * yy; OV[3] = -O * yz;
  OV[4] = bx + b6[bo] * yx + b6[bo + 1] * yy + b6[bo + 2] * yz;
  OV[5] = by + b6[bo + 1] * yx + b6[bo + 3] * yy + b6[bo + 4] * yz;
  OV[6] = bz + b6[bo + 2] * yx + b6[bo + 4] * yy + b6[bo + 5] * yz;
  OV[7] = r;
  return O;
}

// distance from the origin to the segment d + t m, t in [0, 1] (mm = |m|^2; 0: the point d)
function segDist(dx, dy, dz, mx, my, mz, mm) {
  if (mm > 0) { const t = Math.min(1, Math.max(0, -(dx * mx + dy * my + dz * mz) / mm)); dx += t * mx; dy += t * my; dz += t * mz; }
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function maha(a6, ao, b6, bo, dx, dy, dz) {
  const sxx = a6[ao] + b6[bo], sxy = a6[ao + 1] + b6[bo + 1], sxz = a6[ao + 2] + b6[bo + 2];
  const syy = a6[ao + 3] + b6[bo + 3], syz = a6[ao + 4] + b6[bo + 4], szz = a6[ao + 5] + b6[bo + 5];
  const c00 = syy * szz - syz * syz, c01 = sxz * syz - sxy * szz, c02 = sxy * syz - sxz * syy;
  const det = sxx * c00 + sxy * c01 + sxz * c02;
  const c11 = sxx * szz - sxz * sxz, c12 = sxy * sxz - sxx * syz, c22 = sxx * syy - sxy * sxy;
  return (dx * (c00 * dx + c01 * dy + c02 * dz) + dy * (c01 * dx + c11 * dy + c12 * dz)
    + dz * (c02 * dx + c12 * dy + c22 * dz)) / det;
}

// q += 0.5 h [0, w] q, normalised
function qint(q, o, wx, wy, wz, h) {
  const q0 = q[o], q1 = q[o + 1], q2 = q[o + 2], q3 = q[o + 3];
  const d0 = -wx * q1 - wy * q2 - wz * q3;
  const d1 = wx * q0 + wy * q3 - wz * q2;
  const d2 = -wx * q3 + wy * q0 + wz * q1;
  const d3 = wx * q2 - wy * q1 + wz * q0;
  const a = q0 + 0.5 * h * d0, b = q1 + 0.5 * h * d1, c = q2 + 0.5 * h * d2, d = q3 + 0.5 * h * d3;
  const n = Math.sqrt(a * a + b * b + c * c + d * d);
  q[o] = a / n; q[o + 1] = b / n; q[o + 2] = c / n; q[o + 3] = d / n;
}

// in-place heapsort of the indices idx[0..n) by key[idx]
function heapsortBy(idx, key, n) {
  const sift = (i, m) => {
    for (;;) {
      let c = 2 * i + 1;
      if (c >= m) return;
      if (c + 1 < m && key[idx[c + 1]] > key[idx[c]]) c++;
      if (key[idx[c]] <= key[idx[i]]) return;
      const t = idx[i]; idx[i] = idx[c]; idx[c] = t; i = c;
    }
  };
  for (let i = (n >> 1) - 1; i >= 0; i--) sift(i, n);
  for (let m = n - 1; m > 0; m--) { const t = idx[0]; idx[0] = idx[m]; idx[m] = t; sift(0, m); }
}

// in-place heapsort of a[0..n)
function heapsort(a, n) {
  const sift = (i, m) => {
    for (;;) {
      let c = 2 * i + 1;
      if (c >= m) return;
      if (c + 1 < m && a[c + 1] > a[c]) c++;
      if (a[c] <= a[i]) return;
      const t = a[i]; a[i] = a[c]; a[c] = t; i = c;
    }
  };
  for (let i = (n >> 1) - 1; i >= 0; i--) sift(i, n);
  for (let m = n - 1; m > 0; m--) { const t = a[0]; a[0] = a[m]; a[m] = t; sift(0, m); }
}

export class PhysicsWorld {
  // meta: physics.json; nodes: Float64Array of physics.bin; opts override meta.world (World() defaults)
  constructor(meta, nodes, opts = {}) {
    // gpuForm: the step as gpu_bench/step.wgsl runs it (its CPU replica): pjacobi with 4 sweeps (2 sank piles), no pair
    // cache, no sequential hard limit, damping by body-pair colour (dampForm); friction as Stribeck static / kinetic
    // plus anchors (a leaning box crept on an 8-box pile without static friction; stickV, a 5 cm/s stick threshold, and
    // fricAfter stood in for it before); sleep needs the body slow and within 2 mm of where its still window began
    // (sleepDisp; a raw contact-error rule kept a jammed box awake forever)
    if (opts.gpuForm) opts = { solver: 'pjacobi', iters: 4, muS: 0.6, muK: 0.45, frictionAnchors: true, sleepDisp: 0.002, pair_tol: 0, hardLimit: 1e9, capacity: 1_000_000, ...opts };
    const W = { ...meta.world, ...opts };
    this.opts = opts;
    const rt = opts.rolling;   // per part mu_r (rolling)
    this.muR = Float64Array.from(meta.parts, p => !rt ? 0 : typeof rt === 'object' ? rt[p.material] ?? 0 : p.rolling ?? 0);
    this.rollOn = this.muR.some(v => v > 0);
    Object.assign(this, {
      sCol: W.s_col, sMax: W.s_max, tMax: W.t_max, patch: W.patch, margin: W.margin, nsub: W.nsub, dt: W.dt,
      alpha: W.alpha, kappa: W.kappa, qcut: W.qcut, mu: W.mu, damp: W.damp, sleepV: opts.sleepV ?? W.sleep_v, sleepT: opts.sleepT ?? W.sleep_t,
      grav: W.grav, snapDeg: W.snap_deg, maxdep: W.maxdep, relax: W.relax, sweptQ: W.swept_q, sweptMin: W.swept_min,
      wakeV: W.wake_v ?? 0.05, marginCap: W.margin_cap ?? Infinity, pairTol: W.pair_tol ?? 0, useSlab: !!W.slab,
      hardLimit: opts.hardLimit ?? W.hard_limit ?? 200_000,   // node pairs per step past which detection stops refining
      sleepStatic: opts.sleepStatic ?? W.sleep_static ?? true,   // sleeping bodies are static for the solver
      islandSleep: opts.islandSleep ?? W.island_sleep ?? true,   // contact islands sleep and wake together
      islandAvg: opts.islandAvg ?? W.island_avg ?? false,   // island sleep on the island's average speed (off: every body)
      lockLevels: opts.lockLevels ?? W.lock_levels ?? false,   // budget: a pair's level is chosen at contact onset and kept until it separates
      directedMargin: opts.directedMargin ?? false,   // detection's speculative reach along each pair's relative motion (no cap)
      // one-sided contact depth behind thin nodes with solid behind them (bake --solid-behind / --solid-boundary: slab
      // slot 5 = +-2): a partner centre past the sheet counts as on it plus its distance behind (the overlap vanished)
      oneSided: opts.oneSided ?? false,
      directionalExtent: opts.directionalExtent ?? false,
      contactRefine: opts.contactRefine ?? false,
      saturate: opts.saturate ?? false,
      surfaceOnly: opts.surfaceOnly ?? false,
      refineNeed: opts.refineNeed ?? 0,
      needMaxTag: opts.needMaxTag ?? 1,   // refineNeed stops only at nodes tagged this level or finer (L1: the per-level calibration holds there)
      sweptCutOnly: opts.sweptCutOnly ?? false,   // the swept (time-of-impact) test only on node pairs at the size cut (level 0)   // m: thin node pairs with extent errors under this stop descending (needStops counts them)   // 'ii' (or true): interior x interior node pairs skipped; 'all': any pair with an interior node   // detection stops at node pairs that already overlap within one sigma (satStops counts them)
      restitution: opts.restitution ?? false,
      energyPassive: opts.energyPassive ?? false,   // energy limiter without gates: the solver adds no energy in total (limitEnergyPassive)   // the materials' restitution sets each contact's normal velocity (replaces damp)   // in-contact pairs evaluated on the covered render Gaussians (needs setExactSupport)
      normalSide: opts.normalSide ?? false,   // surfaceNormals: flakes weighted by how squarely they face the pair (|n . g|)
      // feature-adaptive bakes only (slab slot 11 = extent error): split size-cut nodes whose error is above this (m)
      extentTol: meta.slab_block?.fields?.[11] === 'extent_err' ? (opts.extentTol ?? Infinity) : Infinity,
      depenSplit: opts.depenSplit ?? W.depen_split ?? false,   // depenetration moves bodies without giving them velocity
      energyLimit: opts.energyLimit ?? W.energy_limit ?? false,   // low-energy bodies in contact never gain energy in a step
      compactOn: opts.compact ?? W.compact ?? false,   // compact resting contacts (off by default: parity path)
      contactLevel: opts.contactLevel ?? W.contact_level ?? Number(globalThis.process?.env?.CONTACTL ?? 0),   // budget on: level of pairs in contact
      // GPU solve studies (gpu_bench): 'gs' sequential Gauss-Seidel (default); 'jacobi' every patch against the same
      // poses, corrections averaged per body over its patches (bench kernels.wgsl); 'color' Gauss-Seidel in the order
      // of a body-pair colouring (body pairs of one colour share no dynamic body: one GPU dispatch per colour);
      // 'cjacobi' the colours in turn, Jacobi within a colour (a body's patches in it all belong to one body pair);
      // 'pjacobi' Jacobi, but a body moves by the sum over its body pairs of each pair's mean patch correction (the
      // mean over all its patches let a box with many box patches sink into the floor).
      // iters: sweeps per substep
      solver: opts.solver ?? 'gs', iters: opts.iters ?? 1, gpuForm: !!opts.gpuForm,
      // gpuForm damping: 'color' (body pairs in colour order, a colour's pairs share no dynamic body; per pair its
      // patches' scaled mean impulse: Gauss-Seidel across body pairs, one GPU dispatch pair per colour), 'mean' (per
      // body the mean over its body pairs of each pair's scaled mean impulse), 'sum' (their sum), 'plain' (the sum of
      // unscaled means), 'gs' (sequential, as the default solver). Jacobi forms gave boxes energy: 8-box piles lost
      // boxes through the open front (2026-10-29)
      dampForm: opts.dampForm ?? 'color', dampIters: opts.dampIters ?? 1,   // gpuForm: Jacobi damping sweeps
      // creep studies (2026-10-30): f32State rounds poses and velocities to f32 each substep (the GPU's storage);
      // fricAfter lets a patch's friction see its own normal correction (pjacobi); stickV (m/s, 0 off): velocity-level
      // static friction in the damping pass, a contact sliding slower than stickV stops when mu times the substep's
      // normal impulse allows it
      f32State: !!opts.f32State, fricAfter: !!opts.fricAfter, stickV: opts.stickV ?? W.stick_v ?? 0,
      // sleepDepth (m, 0 off): a body is not still while a contact of it still pushes it out of more overlap than this
      // (the patch's constraint error before the speed cap). Depenetration after a deep impact is slower than sleep_v:
      // a box fell asleep 14 mm in the floor (Drop seed 13, gpuForm, 2026-10-30)
      sleepDepth: opts.sleepDepth ?? W.sleep_depth ?? 0,
      // sleepRecover (m per step, 0 off): a body is not still while it moves out of its contacts by more than this per
      // step (its displacement along its correction-weighted outward contact normal): recovering bodies stay awake,
      // jammed ones (no outward motion: a box held 11 mm deep in the floor, GPU Drop 2) can sleep. Replaces sleepDepth,
      // whose raw contact error is large at rest on the coarse floor nodes
      sleepRecover: opts.sleepRecover ?? W.sleep_recover ?? 0,
      // sleepDisp (m, 0 off): still also needs the body to have stayed within this distance of where its still window
      // began (kinetic energy via sleep_v, plus net displacement over the window): a slow crawl or a slow recovery
      // keeps a body awake, a jammed body (no net motion) sleeps. Replaces sleepDepth and sleepRecover
      sleepDisp: opts.sleepDisp ?? W.sleep_disp ?? 0,
      // sleepNet (2026-10-31): the sleep speed from the step's net pose change (|dp| / dt + rotation angle / dt x
      // sleepL), not the last substep's velocity. A resting contact that corrects only every few substeps leaves 1-2
      // substeps of free fall in the end-of-step velocity (20-45 mm/s on the marble pile, over sleepV: never slept)
      sleepNet: !!opts.sleepNet,
      // rolling (2026-10-31, opt-in): rolling resistance per patch, a torque opposing the bodies' relative rotation
      // about axes in the contact plane, bounded by mu_r |dl| r (the normal correction times the contact lever: the
      // smaller dynamic body's distance from its centre to the contact point); mu_r the geometric mean of the two
      // parts' (physics.json parts[].rolling, physics_materials.json mu_r). true: the bake's values; an object
      // { material: mu_r } supplies them for older bakes
      rolling: opts.rolling ?? false,
      // speculative (2026-10-31, opt-in): a patch separated by less than the margin (Cv <= 0, gap -Cv) gets no position
      // correction as before, but the velocity pass caps its approach at gap / h: the body arrives at contact instead
      // of falling freely into it. Without it a resting contact corrected every 2nd-3rd substep and gathered 36-52 mm/s
      // of approach, over the restitution rest threshold 2|g|h, so it bounced (glass e 0.5): the substep chatter
      speculative: !!opts.speculative,
      // fricSecond (pjacobi): friction as a second Jacobi pass in each sweep, from the poses after that sweep's normal
      // corrections (as a Gauss-Seidel patch's friction sees its own normal correction); without it friction shares the
      // normal pass's per-pair mean
      fricSecond: !!(opts.fricSecond ?? W.fric_second),
      // surfaceNormals: a patch whose overlap is mostly between thin nodes takes the O-weighted sum of both sides'
      // surface normals (the node's thin axis, signed outward by the bake's slab slot 5, or for two-sided nodes such
      // as open sheets facing the partner body) instead of the overlap gradient, which turns sideways or inward once a
      // body is past a surface's node centres (a box held 11 mm in the floor); replaces the 20 deg snap there
      surfaceNormals: !!(opts.surfaceNormals ?? W.surface_normals),
      // materials: per patch friction from the bake's per-node physics materials (slab slots 6-10: mu_s mu_k v_s
      // restitution compliance), each side O-weighted over the patch's node pairs, the two sides combined: friction
      // and v_s as geometric means (symmetric; ice on anything stays slippery, rubber on rubber grips), restitution
      // as the max (stored for the damping replacement)
      materials: !!(opts.materials ?? W.materials),
      // friction studies (2026-10-30). Stribeck (muS and muK set): mu(v) = muK + (muS - muK) exp(-(v / stribeckV)^2),
      // v the contact's tangential slip speed this substep (unset: mu everywhere). frictionAnchors: per patch (keyed by
      // its bodies and rest-frame cell, kept across substeps and frames; expires one frame unseen) the contact point on
      // both bodies; static friction pulls the anchors back together up to mu(0) |dl|, past it the contact slides at
      // mu(v) and re-anchors (a cache: a cold start or a new patch anchors where it is)
      muS: opts.muS ?? W.mu_s, muK: opts.muK ?? W.mu_k, stribeckV: opts.stribeckV ?? W.stribeck_v ?? 0.03,
      frictionAnchors: !!(opts.frictionAnchors ?? W.friction_anchors),
    });
    this.meta = meta;
    this.N = nodes;
    // per-node block (bake slab_block): thin axis xyz, slab half-thickness (slab pruning), detail level tag (0 = the
    // cut, L = the coarsest level L0-L3 at which the node is not split, 99 = always split)
    this.SLS = meta.slab_block?.stride ?? 4;
    this.SL = nodes.slab ?? new Float64Array(this.SLS * (nodes.length / NS));
    // directionalExtent (bake --directional-extent): per node its support along the body-frame directions extDirs; a
    // pair's contact offsets then follow the pair's direction (edges and corners) instead of each node's own thin axis
    this.EXT = nodes.ext ?? null;
    this.extDirs = meta.extent_block ? Float64Array.from(meta.extent_block.dirs.flat()) : null;
    this.extRef = Float64Array.from(meta.parts, p => p.extent_ref ?? NaN);   // per part's face reference (NaN: no table)
    const B = this.B = meta.parts.length;
    this.root = Int32Array.from(meta.parts, p => p.root);
    this.invM = Float64Array.from(meta.parts, p => (p.static ? 0 : 1 / p.mass));
    this.invS = new Float64Array(this.invM.length);
    this.islandPar = new Int32Array(B); this.islandCnt = new Int32Array(B * B);
    this.spB = new Float64Array(B); this.islandStill = new Float64Array(B);
    // split depenetration: per substep, each body's displacement and rotation (rotation vector) from depenetration,
    // removed from the velocity the substep derives; energy limiter: each body's mechanical energy at step start
    this.penP = new Float64Array(3 * B); this.penW = new Float64Array(3 * B);
    this.E0 = new Float64Array(B); this.inContact = new Uint8Array(B); this.impactB = new Uint8Array(B);
    this.popStats = { depenMoves: 0, depenSum: 0, limited: 0, limitedEnergy: 0 };   // the solver's inverse masses (sleeping bodies static: substeps)
    this.Ib = new Float64Array(3 * B);
    meta.parts.forEach((p, b) => { if (!p.static) this.Ib.set(p.inertia, 3 * b); });
    this.sleepL = Float64Array.from(meta.parts, p => p.sleep_L);
    this.pos = new Float64Array(3 * B); this.quat = new Float64Array(4 * B);
    this.vel = new Float64Array(3 * B); this.om = new Float64Array(3 * B);
    this.asleep = new Uint8Array(B); this.still = new Float64Array(B);
    this.Rs = new Float64Array(9 * B); this.Rp = new Float64Array(9 * B); this.Iinv = new Float64Array(9 * B);
    this.prevP = new Float64Array(3 * B); this.prevQ = new Float64Array(4 * B);
    // pair buffers: the hard limit plus the descent stack's slack (detection never keeps more pairs than that)
    const cap = this.cap = opts.capacity ?? this.hardLimit + 2 * STACK;
    this.oa = new Int32Array(cap); this.ob = new Int32Array(cap); this.op = new Int32Array(cap);
    this.patchOf = new Int32Array(cap); this.pairPatch = new Int32Array(cap);
    this.pba = new Int32Array(B * B); this.pbb = new Int32Array(B * B);
    this.stack = new Int32Array(2 * STACK);
    // pair cache (pair_tol > 0, world.py): per body pair (i B + j) the relative pose at the build, the build margin and
    // the pair's range in last frame's list
    this.cValid = new Uint8Array(B * B); this.cRel = new Float64Array(12 * B * B); this.cM = new Float64Array(B * B);
    this.cCap = new Int32Array(B * B);
    // per body: calibrated rest offsets per level along its box axes (loff[12 b + 3 level + axis]) and the box axes
    // (rows, body frame); zeros = no level offsets
    this.loff = new Float64Array(12 * B); this.baxes = new Float64Array(9 * B);
    meta.parts.forEach((p, b) => {
      this.baxes.set(p.box_axes?.length ? p.box_axes.flat() : [1, 0, 0, 0, 1, 0, 0, 0, 1], 9 * b);
      (p.level_offsets ?? []).forEach((o, l) => this.loff.set(o, 12 * b + 3 * l));
    });
    // adaptive precision (budget > 0, node pairs per frame): a detail level per body pair (bcap), chosen before
    // detection from last frame's per-pair counts; budget 0 = every pair at full detail (the parity path)
    this.budget = opts.budget ?? W.budget ?? 0;
    this.levels = Math.min(meta.levels?.count ?? 1, (opts.maxLevel ?? W.max_level ?? 3) + 1);   // coarsest level used: levels - 1
    this.bcap = new Int32Array(B * B); this.pairCnt = new Int32Array(B * B); this.pairLev = new Int32Array(B * B);
    this.cand = new Int32Array(B * B); this.candKey = new Float64Array(B * B); this.candCls = new Int8Array(B * B);
    this.candEst = new Float64Array(B * B); this.fixed = new Uint8Array(B * B);
    // lockLevels: per body pair the level locked at contact onset (-1 none) and frames since it last had node pairs
    this.lockLev = new Int8Array(B * B).fill(-1); this.sepCount = new Int32Array(B * B);
    this.budgetStats = { used: 0, budget: this.budget, coarsest: 0, coarseImpacts: 0, impacts: 0, overrun: false };
    this.cStart = new Int32Array(B * B); this.cLen = new Int32Array(B * B);
    this.prevA = new Int32Array(cap); this.prevB = new Int32Array(cap);
    // patches: open-addressing hash from patch key to patch id, then the ids ranked by key (Python: argsort of keys)
    this.HB = 18; this.H = 1 << this.HB;
    this.hKey = new Float64Array(this.H).fill(-1); this.hVal = new Int32Array(this.H);
    this.pcap = this.H >> 1;
    this.pKey = new Float64Array(this.pcap); this.pSlot = new Int32Array(this.pcap); this.pIdx = new Int32Array(this.pcap); this.pSorted = new Float64Array(this.pcap);
    this.pCount = new Int32Array(this.pcap); this.pRank = new Int32Array(this.pcap);
    this.starts = new Int32Array(this.pcap + 1); this.cursor = new Int32Array(this.pcap);
    this.pn = new Float64Array(3 * this.pcap); this.pxc = new Float64Array(3 * this.pcap); this.pl = new Float64Array(this.pcap);
    this.pgap = new Float64Array(this.pcap);   // speculative: per patch the gap this substep (Infinity: none)
    this.pairN = new Float64Array(3 * B * B);
    this.npairs = 0; this.nbp = 0; this.npatch = 0;
    this.stats = new Float64Array(6);
    this.cacheHits = 0; this.cacheBuilds = 0;   // pair cache counters (inspection)
    this.satStops = 0;   // saturate: node pairs the descent stopped at (inspection)
    this.needStops = 0;   // refineNeed: node pairs the descent stopped at (inspection)
    // Jacobi accumulators (solver 'jacobi'): per body summed linear and angular corrections and the patches giving them
    this.jac = false; this.jL = new Float64Array(3 * B); this.jA = new Float64Array(3 * B); this.jN = new Int32Array(B);
    this.jMark = new Int32Array(B).fill(-1); this.jPatch = 0;
    // pjacobi: per (body, body pair) sums and patch counts, and the touched slots
    this.pj = false; this.jBP = 0; this.pjL = new Float64Array(3 * B * B * B); this.pjA = new Float64Array(3 * B * B * B);
    this.pjN = new Int32Array(B * B * B); this.pjMark = new Int32Array(B * B * B).fill(-1); this.pjList = new Int32Array(B * B * B); this.pjn = 0;
    this.colorStats = { colors: 0, maxColors: 0 };   // solver 'color' (inspection)
    this.bpHit = new Uint8Array(B * B);   // per body pair of the last detect: 1 = its pairs came from the cache (inspection)
    // compact resting contacts (compactOn): per body pair a calm-frame counter, a compact flag and age, and up to REPS
    // representative contacts (one per full-detail patch): point on i and on j (body frames), normal (j's frame)
    this.calm = new Int32Array(B * B); this.compactF = new Uint8Array(B * B); this.compactAge = new Int32Array(B * B);
    this.repN = new Int32Array(B * B); this.rep = new Float64Array(B * B * REPS * 9); this.expandNext = new Uint8Array(B * B);
    this.compactStats = { pairs: 0, reps: 0, compactions: 0, expansions: 0 };
    // per-node caches of world means and covariances, one per side of a body pair (side 0 = body i, 1 = body j):
    // instances of one asset share its tree (meta parts with the same root, each with its own scale), so a node can
    // sit on both sides of a pair; the caches are valid for one body pair (detect) or one patch (solve)
    const nn = this.nn = nodes.length / NS;
    this.cW = new Float64Array(18 * nn); this.cP = new Float64Array(6 * nn);
    this.mStamp = new Int32Array(2 * nn); this.cStamp = new Int32Array(2 * nn); this.pStamp = new Int32Array(2 * nn);
    // per-body scale of its (possibly shared) tree: means x s, covariances x s^2, mass x s^3, lengths x s
    this.bs = Float64Array.from(meta.parts, p => p.scale ?? 1);
    this.stamp = 0; this.pstamp = 0;
    this.last = { detectMs: 0, solveMs: 0, pairs: 0, patches: 0 };
    this.reset();
  }

  // every part at its rest pose, at rest; dynamic parts asleep until something wakes them
  reset() {
    this.meta.parts.forEach((p, b) => this.setPose(b, p.rest_pos, p.rest_quat));
    this.vel.fill(0); this.om.fill(0); this.still.fill(0);
    for (let b = 0; b < this.B; b++) this.asleep[b] = this.invM[b] > 0 ? 1 : 0;
    this.cValid.fill(0);
    this.lockLev?.fill(-1); this.sepCount?.fill(0);
  }

  // Startup self-check: one detection with every part at its rest pose and the dynamic parts awake. Reports each
  // part's world root position and, per dynamic part, its node pairs against each other part. `ok` is false when a
  // dynamic part whose root sphere reaches the floor (its lowest point at or below z = 0) has no pairs at all, i.e.
  // nothing holds it up: it may rest on the floor or on other parts (a pile: 2026-10-31, the marbles' top layer sits on
  // the lower one), so the check holds on any scene. Also false when such parts exist and none of them touches the
  // floor (the pile itself must stand on something).
  // Leaves the world reset (rest pose, dynamic parts asleep).
  selfCheck() {
    this.reset();
    for (let b = 0; b < this.B; b++) if (this.invM[b] > 0) this.wake(b);
    this.detect();
    const parts = this.meta.parts, N = this.N;
    const rootWorld = b => {
      quatMat(this.quat, 4 * b, T9, 0);
      const n = parts[b].root * NS;
      const s_ = this.bs[b];
      return [0, 1, 2].map(r => (T9[3 * r] * N[n] + T9[3 * r + 1] * N[n + 1] + T9[3 * r + 2] * N[n + 2]) * s_ + this.pos[3 * b + r]);
    };
    const pairsWith = new Map();
    for (let t = 0; t < this.npairs; t++) {
      const bp = this.op[t], key = `${this.pba[bp]}-${this.pbb[bp]}`;
      pairsWith.set(key, (pairsWith.get(key) ?? 0) + 1);
    }
    const floor = parts.findIndex(p => p.name === 'floor');
    const report = { ok: true, pairs: this.npairs, parts: parts.map((p, b) => ({ name: p.name, static: p.static, root: rootWorld(b).map(v => +v.toFixed(4)), reach: +(N[p.root * NS + 10] * this.bs[b]).toFixed(3) })), dynamic: [] };
    for (let b = 0; b < this.B; b++) {
      if (this.invM[b] === 0) continue;
      const with_ = {};
      for (let s = 0; s < this.B; s++) if (s !== b) { const k = s < b ? `${s}-${b}` : `${b}-${s}`; const c = pairsWith.get(k) ?? 0; if (c) with_[parts[s].name] = c; }
      const onFloor = floor >= 0 && rootWorld(b)[2] - N[parts[b].root * NS + 10] * this.bs[b] <= 0;
      if (onFloor && !Object.keys(with_).length) report.ok = false;
      report.dynamic.push({ name: parts[b].name, onFloor, pairs: with_ });
    }
    const near = report.dynamic.filter(d => d.onFloor);
    if (near.length && !near.some(d => d.pairs[parts[floor]?.name])) report.ok = false;
    this.reset();
    return report;
  }

  setPose(b, p, q) { this.pos.set(p, 3 * b); this.quat.set(q, 4 * b); }
  wake(b) { this.asleep[b] = 0; this.still[b] = 0; }
  anyAwake() { for (let b = 0; b < this.B; b++) if (this.invM[b] > 0 && !this.asleep[b]) return true; return false; }

  // world_from_rest rows [R | t] (SPEC 2.4 xf): R = R(q) R0^T, t = p - R com0
  worldFromRest(b, out, o = 0) {
    const p = this.meta.parts[b], R0 = p.R0, c = p.com0;
    quatMat(this.quat, 4 * b, T9, 0);
    for (let r = 0; r < 3; r++) {
      let t = this.pos[3 * b + r];
      for (let k = 0; k < 3; k++) {
        const v = T9[3 * r] * R0[3 * k] + T9[3 * r + 1] * R0[3 * k + 1] + T9[3 * r + 2] * R0[3 * k + 2];   // (R R0^T)[r][k]
        out[o + 4 * r + k] = v;
        t -= v * c[k];
      }
      out[o + 4 * r + 3] = t;
    }
    return out;
  }

  step() {
    const t0 = now();
    if (this.budget > 0) this.assignLevels();
    this.detect();
    if (this.budget > 0) this.budgetStats.used = this.npairs;
    if (this.budget > 0 && this.lockLevels) this.updateLocks();
    if (this.npairs > 0) this.makePatches(); else this.npatch = 0;
    const t1 = now();
    this.stats.fill(0);
    (this.depthB ??= new Float64Array(this.B)).fill(0);   // sleepDepth: per body the largest contact error this step
    if (this.sleepRecover > 0 || this.sleepNet) (this.pos0 ??= new Float64Array(3 * this.B)).set(this.pos);   // sleepRecover, sleepNet: step-start poses
    if (this.sleepNet) (this.quat0 ??= new Float64Array(4 * this.B)).set(this.quat);
    this.frame = (this.frame ?? 0) + 1;
    if (this.frictionAnchors) {   // anchors unseen for a frame expire
      this.anchors ??= new Map();
      for (const [k, a] of this.anchors) if (a.frame < this.frame - 1) this.anchors.delete(k);
      if (this.opts?.anchorsOneFrame) this.anchors.clear();   // diagnostic: no anchor survives a step
    }
    if (this.compactOn) this.expandCompact();
    if (this.energyLimit) this.markEnergy();
    this.substeps();
    if (this.energyLimit) { if (this.energyPassive) this.limitEnergyPassive(); else this.limitEnergy(); }
    if (this.compactOn) this.updateCompact();
    const t2 = now();
    this.sleep();
    this.last.detectMs = t1 - t0; this.last.solveMs = t2 - t1; this.last.pairs = this.npairs; this.last.patches = this.npatch;
  }

  detect() {
    const { B, N, pos, quat, vel, om, asleep, invM, root, Rs, oa, ob, op, pba, pbb, stack, margin: margin0, dt, qcut } = this;
    const sCol = this.sCol, sMax = this.sMax, tMax = this.tMax, cap = this.cap, tol = this.pairTol;
    const { cValid, cRel, cM, cStart, cLen, prevA, prevB, SL, SLS, cCap, bcap, pairCnt, pairLev } = this, useSlab = this.useSlab;
    const hasLev = SLS > 4;   // bakes before the detail levels have no level tags (every pair at full detail)
    const hardLimit = this.hardLimit;
    const extTol = this.extentTol;
    const intOn = this.surfaceOnly && this.meta.slab_block?.fields?.[11] === 'interior';
    const needTol = this.refineNeed && this.meta.slab_block?.fields?.[11] === 'extent_err' ? this.refineNeed : 0;
    for (let b = 0; b < B; b++) quatMat(quat, 4 * b, Rs, 9 * b);
    const cW = this.cW, bs = this.bs;
    let npairs = 0, nbp = 0, visits = 0;
    // wake pass first (kernels.py detect): a sleeping body woken by a moving neighbour must also get its other contacts
    // (the floor) this step. Waking inside the pair loop skipped the pairs before it: the woken box fell one frame
    // through the floor and popped back out (2026-10-29)
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
      if (!(asleep[i] || asleep[j]) || (invM[i] === 0 && invM[j] === 0)) continue;
      if ((asleep[i] || invM[i] === 0) && (asleep[j] || invM[j] === 0)) continue;
      const si = bs[i], sj = bs[j];
      const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
      const spd = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) + Math.hypot(om[3 * i], om[3 * i + 1], om[3 * i + 2]) * N[root[i] * NS + 10] * si
        + Math.hypot(om[3 * j], om[3 * j + 1], om[3 * j + 2]) * N[root[j] * NS + 10] * sj;
      const margin = margin0 + (invM[i] === 0 || invM[j] === 0 ? 1.5 * spd * dt : Math.min(1.5 * spd * dt, this.marginCap));
      xfm(Rs, 9 * i, pos, 3 * i, N, root[i] * NS, si); const ax = X3[0], ay = X3[1], az = X3[2];
      xfm(Rs, 9 * j, pos, 3 * j, N, root[j] * NS, sj);
      if (Math.sqrt((ax - X3[0]) ** 2 + (ay - X3[1]) ** 2 + (az - X3[2]) ** 2) > N[root[i] * NS + 10] * si + N[root[j] * NS + 10] * sj + margin) continue;
      if (asleep[i] && Math.hypot(vel[3 * j], vel[3 * j + 1], vel[3 * j + 2]) > this.wakeV) asleep[i] = 0;
      if (asleep[j] && Math.hypot(vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]) > this.wakeV) asleep[j] = 0;
    }
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
      if (invM[i] === 0 && invM[j] === 0) continue;
      const ai = asleep[i] || invM[i] === 0, aj = asleep[j] || invM[j] === 0;
      if (ai && aj) { cValid[i * B + j] = 0; this.pairCnt[i * B + j] = 0; continue; }
      const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
      let spd = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
      const si = bs[i], sj = bs[j];
      spd += Math.hypot(om[3 * i], om[3 * i + 1], om[3 * i + 2]) * N[root[i] * NS + 10] * si
        + Math.hypot(om[3 * j], om[3 * j + 1], om[3 * j + 2]) * N[root[j] * NS + 10] * sj;
      // speculative margin from the pair's per-frame motion; capped between two dynamic bodies (both finely cut), not
      // against a static part (coarsely cut, cheap)
      let margin = margin0 + (invM[i] === 0 || invM[j] === 0 ? 1.5 * spd * dt : Math.min(1.5 * spd * dt, this.marginCap));
      // directedMargin: the speculative reach follows the pair's relative motion. Node pairs are tested against the
      // segment their offset sweeps over 1.5 frames (straight-line means), plus an isotropic part for rotation; no cap.
      // The isotropic margin had to be capped between dynamic bodies for cost, and the cap missed a box glancing off
      // another at 2 m/s (the fixed Drop, 2026-10-30)
      // Per node pair: each node centre's own velocity (v + w x r) gives the segment; spin adds only the node's own
      // reach (|w| times its radius) to the isotropic margin
      let Mx = 0, My = 0, Mz = 0, MM = 0, wTi = 0, wTj = 0;
      const dirM = this.directedMargin, T = 1.5 * dt;
      if (dirM) { wTi = Math.hypot(om[3 * i], om[3 * i + 1], om[3 * i + 2]) * T; wTj = Math.hypot(om[3 * j], om[3 * j + 1], om[3 * j + 2]) * T; margin = margin0; }
      // the relative motion of node centres a (of i) and b (of j) over 1.5 frames
      const motion = (ax, ay, az, bx, by, bz) => {
        const rax = ax - pos[3 * i], ray = ay - pos[3 * i + 1], raz = az - pos[3 * i + 2];
        const rbx = bx - pos[3 * j], rby = by - pos[3 * j + 1], rbz = bz - pos[3 * j + 2];
        const wi = 3 * i, wj = 3 * j;
        Mx = (vel[wi] + om[wi + 1] * raz - om[wi + 2] * ray - vel[wj] - (om[wj + 1] * rbz - om[wj + 2] * rby)) * T;
        My = (vel[wi + 1] + om[wi + 2] * rax - om[wi] * raz - vel[wj + 1] - (om[wj + 2] * rbx - om[wj] * rbz)) * T;
        Mz = (vel[wi + 2] + om[wi] * ray - om[wi + 1] * rax - vel[wj + 2] - (om[wj] * rby - om[wj + 1] * rbx)) * T;
        MM = Mx * Mx + My * My + Mz * Mz;
      };
      const ri = 9 * i, rj = 9 * j;
      const xi = 3 * i, xj = 3 * j;
      {
        const na = root[i] * NS, nb = root[j] * NS;
        const mx = N[na] * si, my = N[na + 1] * si, mz = N[na + 2] * si, nx_ = N[nb] * sj, ny_ = N[nb + 1] * sj, nz_ = N[nb + 2] * sj;
        const ax = Rs[ri] * mx + Rs[ri + 1] * my + Rs[ri + 2] * mz + pos[xi];
        const ay = Rs[ri + 3] * mx + Rs[ri + 4] * my + Rs[ri + 5] * mz + pos[xi + 1];
        const az = Rs[ri + 6] * mx + Rs[ri + 7] * my + Rs[ri + 8] * mz + pos[xi + 2];
        const bx = Rs[rj] * nx_ + Rs[rj + 1] * ny_ + Rs[rj + 2] * nz_ + pos[xj];
        const by = Rs[rj + 3] * nx_ + Rs[rj + 4] * ny_ + Rs[rj + 5] * nz_ + pos[xj + 1];
        const bz = Rs[rj + 6] * nx_ + Rs[rj + 7] * ny_ + Rs[rj + 8] * nz_ + pos[xj + 2];
        if (dirM) motion(ax, ay, az, bx, by, bz);
        if (segDist(ax - bx, ay - by, az - bz, Mx, My, Mz, MM) > N[na + 10] * si * (1 + wTi) + N[nb + 10] * sj * (1 + wTj) + margin) { cValid[i * B + j] = 0; this.pairCnt[i * B + j] = 0; continue; }
      }
      if (asleep[i] && Math.hypot(vel[xj], vel[xj + 1], vel[xj + 2]) > this.wakeV) asleep[i] = 0;
      if (asleep[j] && Math.hypot(vel[xi], vel[xi + 1], vel[xi + 2]) > this.wakeV) asleep[j] = 0;
      const bp = nbp++;
      pba[bp] = i; pbb[bp] = j; this.bpHit[bp] = 0;
      if (this.compactOn && this.compactF[i * B + j]) {   // compact resting contact: its representatives stand in
        cValid[i * B + j] = 0; cStart[i * B + j] = npairs; cLen[i * B + j] = 0; pairCnt[i * B + j] = Math.max(1, pairCnt[i * B + j]);
        continue;
      }
      this.stamp++;   // node caches are per side: valid for this body pair only
      const pid = i * B + j, start = npairs, lev = bcap[pid];   // this pair's detail level cap (0 = full detail)
      if (tol > 0) {
        // relative pose of i in j's frame: Rr = Rj^T Ri, tr = Rj^T (pi - pj)
        const ex = pos[xi] - pos[xj], ey = pos[xi + 1] - pos[xj + 1], ez = pos[xi + 2] - pos[xj + 2];
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) RR[3 * a + b] = Rs[rj + a] * Rs[ri + b] + Rs[rj + 3 + a] * Rs[ri + 3 + b] + Rs[rj + 6 + a] * Rs[ri + 6 + b];
          RR[9 + a] = Rs[rj + a] * ex + Rs[rj + 3 + a] * ey + Rs[rj + 6 + a] * ez;
        }
        const c = 12 * pid;
        if (cValid[pid] && cCap[pid] === lev) {
          // a point of i at radius r moved by at most |dt| + angle r since the build
          let tr = 0;
          for (let a = 0; a < 9; a++) tr += cRel[c + a] * RR[a];
          const ang = Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2)));
          const n0 = root[i] * NS, rad = (Math.hypot(N[n0], N[n0 + 1], N[n0 + 2]) + N[n0 + 10]) * si;
          const drift = Math.hypot(RR[9] - cRel[c + 9], RR[10] - cRel[c + 10], RR[11] - cRel[c + 11]) + ang * rad;
          if (drift + margin + (dirM ? Math.sqrt(MM) : 0) <= cM[pid]) {
            for (let t = cStart[pid], e = cStart[pid] + cLen[pid]; t < e; t++) {
              if (npairs < cap) { oa[npairs] = prevA[t]; ob[npairs] = prevB[t]; op[npairs] = bp; npairs++; }
            }
            cStart[pid] = start; cLen[pid] = npairs - start;
            pairCnt[pid] = npairs - start; pairLev[pid] = lev;
            this.cacheHits++; this.bpHit[bp] = 1;
            continue;
          }
        }
        margin += tol;
        this.cacheBuilds++;
        cRel.set(RR, c); cM[pid] = margin; cValid[pid] = 1; cCap[pid] = lev;
      }
      stack[0] = root[i]; stack[1] = root[j];
      let sp = 1;
      while (sp > 0) {
        sp--; visits++;
        const ka = stack[2 * sp], kb = stack[2 * sp + 1];
        const na = ka * NS, nb = kb * NS;
        const ca = this.mean(0, ka, Rs, ri, pos, xi, si), cb = this.mean(1, kb, Rs, rj, pos, xj, sj);
        let ddx = cW[ca] - cW[cb], ddy = cW[ca + 1] - cW[cb + 1], ddz = cW[ca + 2] - cW[cb + 2];
        if (this.oneSided) {
          // behind a solid-backed thin node is in contact range (one-sided depth): drop that part of the offset
          for (let side = 0; side < 2; side++) {
            const k = side === 0 ? ka : kb, sl = SLS * k;
            if (Math.abs(SL[sl + 5]) < 2 || !(SL[sl + 3] <= 0.25 * N[k * NS + 11])) continue;
            const r = side === 0 ? ri : rj, sg = Math.sign(SL[sl + 5]) * (side === 0 ? 1 : -1), ux = SL[sl], uy = SL[sl + 1], uz = SL[sl + 2];
            const nx_ = Rs[r] * ux + Rs[r + 1] * uy + Rs[r + 2] * uz, ny_ = Rs[r + 3] * ux + Rs[r + 4] * uy + Rs[r + 5] * uz, nz_ = Rs[r + 6] * ux + Rs[r + 7] * uy + Rs[r + 8] * uz;
            const c = sg * (nx_ * ddx + ny_ * ddy + nz_ * ddz);   // > 0: the partner is behind this node's sheet
            const lat2 = ddx * ddx + ddy * ddy + ddz * ddz - c * c, reach = N[k * NS + 10] * (side === 0 ? si : sj) + N[(side === 0 ? kb : ka) * NS + 10] * (side === 0 ? sj : si);
            let fw = 1;
            const ko = side === 0 ? kb : ka, so = SLS * ko;
            if (SL[so + 5] !== 0 && SL[so + 3] <= 0.25 * N[ko * NS + 11]) {   // the partner's surface must face this sheet
              const ro = side === 0 ? rj : ri, sgo = Math.sign(SL[so + 5]), vx = SL[so], vy = SL[so + 1], vz = SL[so + 2], sgs = Math.sign(SL[sl + 5]);
              fw = -sgo * sgs * ((Rs[ro] * vx + Rs[ro + 1] * vy + Rs[ro + 2] * vz) * nx_ + (Rs[ro + 3] * vx + Rs[ro + 4] * vy + Rs[ro + 5] * vz) * ny_ + (Rs[ro + 6] * vx + Rs[ro + 7] * vy + Rs[ro + 8] * vz) * nz_);
            }
            if (fw > (this.oneSided === 'soft' ? 0 : 0.5) && c > 0 && c <= (Math.abs(SL[sl + 5]) - 2) * (side === 0 ? si : sj) && lat2 <= reach * reach) { ddx -= sg * c * nx_; ddy -= sg * c * ny_; ddz -= sg * c * nz_; }
          }
        }
        const d0x = ddx, d0y = ddy, d0z = ddz;
        let mg = margin;
        if (dirM) {   // directedMargin: the closest offset on this node pair's swept segment stands in for the offset
          motion(cW[ca], cW[ca + 1], cW[ca + 2], cW[cb], cW[cb + 1], cW[cb + 2]);
          if (MM > 0) { const t = Math.min(1, Math.max(0, -(ddx * Mx + ddy * My + ddz * Mz) / MM)); ddx += t * Mx; ddy += t * My; ddz += t * Mz; }
          mg = margin + N[na + 10] * si * wTi + N[nb + 10] * sj * wTj;   // spin within each node
        }
        const dist = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
        if (dist > N[na + 10] * si + N[nb + 10] * sj + mg) continue;
        if (useSlab) {
          // slab bound: every leaf of a node lies within its slab half-thickness of its plane (thin axis)
          const sa4 = SLS * ka, sb4 = SLS * kb;
          const ux = SL[sa4], uy = SL[sa4 + 1], uz = SL[sa4 + 2], vx = SL[sb4], vy = SL[sb4 + 1], vz = SL[sb4 + 2];
          const pa = Math.abs((Rs[ri] * ux + Rs[ri + 1] * uy + Rs[ri + 2] * uz) * ddx + (Rs[ri + 3] * ux + Rs[ri + 4] * uy + Rs[ri + 5] * uz) * ddy
            + (Rs[ri + 6] * ux + Rs[ri + 7] * uy + Rs[ri + 8] * uz) * ddz);
          if (pa > SL[sa4 + 3] * si + N[nb + 10] * sj + mg) continue;
          const pb = Math.abs((Rs[rj] * vx + Rs[rj + 1] * vy + Rs[rj + 2] * vz) * ddx + (Rs[rj + 3] * vx + Rs[rj + 4] * vy + Rs[rj + 5] * vz) * ddy
            + (Rs[rj + 6] * vx + Rs[rj + 7] * vy + Rs[rj + 8] * vz) * ddz);
          if (pb > SL[sb4 + 3] * sj + N[na + 10] * si + mg) continue;
        }
        const sa = N[na + 11] * si, sb = N[nb + 11] * sj;
        // hard limit: past hardLimit node pairs the descent stops at the next calibrated level (L3 or finer, whose
        // rest offsets apply) instead of going finer; stopping at arbitrary coarser nodes launched boxes (16-box pile)
        const full = npairs + sp >= hardLimit;
        // extentTol (feature-adaptive bakes): at full detail a size-cut node whose moment misses its own extent by more
        // than extentTol (slab slot 11, scaled with the body) is split further, so edges and corners contact where they
        // render; coarser levels and the hard limit stop at the size cut as before
        const xa = extTol < Infinity && lev === 0 && SL[SLS * ka + 11] * si > extTol, xb = extTol < Infinity && lev === 0 && SL[SLS * kb + 11] * sj > extTol;
        let la = (full && (!hasLev || SL[SLS * ka + 4] <= 3)) || N[na + 13] < 0 || sa <= sCol || (sa <= sMax && N[na + 12] * si <= tMax) || (hasLev && SL[SLS * ka + 4] <= lev && !(xa && !full));
        let lb = (full && (!hasLev || SL[SLS * kb + 4] <= 3)) || N[nb + 13] < 0 || sb <= sCol || (sb <= sMax && N[nb + 12] * sj <= tMax) || (hasLev && SL[SLS * kb + 4] <= lev && !(xb && !full));
        // saturate (opt-in): two nodes whose centres lie within one standard deviation of each other (Mahalanobis^2 of
        // their offset under the summed covariances below 1) already overlap about as much as they can; their children
        // add no contact information (in a deep overlap of volumetric bodies they are interior pairs: an impact made
        // 3.3M node pairs in a 64-box pile), so the pair stops here and the two nodes' moments stand in for them
        // surfaceOnly (interior-flag bakes, slab slot 11 'interior'): contact is a surface phenomenon; two interior nodes
        // (solid around them by at least their reach) never pair, at any level ('ii'); 'all': an interior node pairs
        // with nothing. Mass and inertia keep every Gaussian; only contact generation changes (2026-10-30)
        if (intOn) {
          const ia = SL[SLS * ka + 11] > 0.5, ib = SL[SLS * kb + 11] > 0.5;
          if (this.surfaceOnly === 'all' ? ia || ib : ia && ib) continue;   // (the descent's continue: the next stack entry)
        }
        // refineNeed (bakes with every node's extent error in slab slot 11): a node pair stops descending when both nodes
        // are thin slabs whose moment already describes their surface (extent error under refineNeed, scaled with the
        // body): flat face-to-face contacts stay coarse, edges and corners refine. Rest heights hold at any level through
        // the per-node surface offsets (2026-10-30)
        let needStop = null;   // the pair's own leaf flags when refineNeed stopped it (a failed q test descends on)
        if (needTol > 0 && !(la && lb) && lev === 0 && !full && SL[SLS * ka + 4] <= this.needMaxTag && SL[SLS * kb + 4] <= this.needMaxTag) {
          const thinA = SL[SLS * ka + 3] <= 0.25 * N[na + 11], thinB = SL[SLS * kb + 3] <= 0.25 * N[nb + 11];
          if (thinA && thinB && SL[SLS * ka + 11] >= 0 && SL[SLS * ka + 11] * si <= needTol && SL[SLS * kb + 11] >= 0 && SL[SLS * kb + 11] * sj <= needTol) { needStop = [la, lb]; la = true; lb = true; }
        }
        let sat = false;
        if (this.saturate && !(la && lb)) {
          this.cov(0, ka, Rs, ri, si); this.cov(1, kb, Rs, rj, sj);
          sat = maha(cW, ca + 3, cW, cb + 3, d0x, d0y, d0z) < (this.saturate === true ? 1 : this.saturate);   // a number: that Mahalanobis^2
          if (sat) this.satStops++;
        }
        if ((la && lb) || sat) {
          this.cov(0, ka, Rs, ri, si); this.cov(1, kb, Rs, rj, sj);
          let q;
          if (dirM && MM > 0) {
            // the Mahalanobis-closest offset on the segment (q is quadratic in t; the Euclidean-closest point missed
            // a thin floor's pairs), then shrunk by the isotropic margin as below
            const qD = maha(cW, ca + 3, cW, cb + 3, d0x, d0y, d0z), qM = maha(cW, ca + 3, cW, cb + 3, Mx, My, Mz);
            const qb = 0.5 * (maha(cW, ca + 3, cW, cb + 3, d0x + Mx, d0y + My, d0z + Mz) - qD - qM);
            const t = qM > 0 ? Math.min(1, Math.max(0, -qb / qM)) : 0;
            const ex = d0x + t * Mx, ey = d0y + t * My, ez = d0z + t * Mz, el = Math.sqrt(ex * ex + ey * ey + ez * ez);
            const sc = Math.max(el - mg, 0) / Math.max(el, 1e-12);
            q = maha(cW, ca + 3, cW, cb + 3, ex * sc, ey * sc, ez * sc);
          } else {
            const sc = Math.max(dist - mg, 0) / Math.max(dist, 1e-12);
            q = maha(cW, ca + 3, cW, cb + 3, ddx * sc, ddy * sc, ddz * sc);
          }
          if (q < qcut && npairs < cap) { oa[npairs] = ka; ob[npairs] = kb; op[npairs] = bp; npairs++; if (needStop) this.needStops++; }
          else if (needStop) { la = needStop[0]; lb = needStop[1]; }   // a coarse pair the q test rejects: its children may touch
          else if (this.detLog) this.detLog.push([bp, ka, kb, q, dist, margin]);   // inspection (opt-in): leaf pairs the q test dropped
          if (!needStop || q < qcut) continue;
        }
        if (sp + 2 >= STACK) continue;
        if (!la && (lb || sa >= sb)) {
          stack[2 * sp] = N[na + 13]; stack[2 * sp + 1] = kb; sp++;
          stack[2 * sp] = N[na + 14]; stack[2 * sp + 1] = kb; sp++;
        } else {
          stack[2 * sp] = ka; stack[2 * sp + 1] = N[nb + 13]; sp++;
          stack[2 * sp] = ka; stack[2 * sp + 1] = N[nb + 14]; sp++;
        }
      }
      cStart[pid] = start; cLen[pid] = npairs - start;
      pairCnt[pid] = npairs - start; pairLev[pid] = lev;
    }
    if (tol > 0) for (let t = 0; t < npairs; t++) { prevA[t] = oa[t]; prevB[t] = ob[t]; }
    this.npairs = npairs; this.nbp = nbp; this.visits = visits;
  }

  // Adaptive precision: a detail level per candidate body pair. Estimates come from last frame's count at its level
  // (x4 per level: contact pairs scale with area / node size^2). The budget only governs pairs not yet touching:
  //   1. pairs in contact (resting or sliding) are always at full detail, outside the budget: no level ever changes
  //      while two bodies touch;
  //   2. impacts (approaching from apart, close enough to touch this frame) get at most L1, and L0 fastest first while
  //      the budget lasts (exceeded rather than coarser);
  //   3. the rest (apart, not about to touch) take what is left, breadth first.
  // Levels >= 1 use the bake's calibrated rest offsets (level_offsets, per box axis).
  assignLevels() {
    const { B, N, pos, vel, om, asleep, invM, root, bcap, pairCnt, pairLev, cand, candKey, candCls, candEst, fixed } = this;
    const L = this.levels - 1, budget = this.budget, st = this.budgetStats;
    bcap.fill(0);
    if (L <= 0) return;
    let n = 0;
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
      if (invM[i] === 0 && invM[j] === 0) continue;
      if ((asleep[i] || invM[i] === 0) && (asleep[j] || invM[j] === 0)) continue;
      const pid = i * B + j;
      const ni = root[i] * NS, nj = root[j] * NS, si = this.bs[i], sj = this.bs[j];
      quatMat(this.quat, 4 * i, T9, 0); xfm(T9, 0, pos, 3 * i, N, ni, si);
      const cx = X3[0], cy = X3[1], cz = X3[2];
      quatMat(this.quat, 4 * j, T9, 0); xfm(T9, 0, pos, 3 * j, N, nj, sj);
      const dx = X3[0] - cx, dy = X3[1] - cy, dz = X3[2] - cz;
      const dist = Math.hypot(dx, dy, dz);
      const rvx = vel[3 * i] - vel[3 * j], rvy = vel[3 * i + 1] - vel[3 * j + 1], rvz = vel[3 * i + 2] - vel[3 * j + 2];
      const spd = Math.hypot(rvx, rvy, rvz) + Math.hypot(om[3 * i], om[3 * i + 1], om[3 * i + 2]) * N[ni + 10] * si + Math.hypot(om[3 * j], om[3 * j + 1], om[3 * j + 2]) * N[nj + 10] * sj;
      const gap = dist - N[ni + 10] * si - N[nj + 10] * sj;
      if (gap > this.margin + 1.5 * spd * this.dt + this.pairTol) continue;   // detect will skip it too
      const closing = dist > 0 ? (rvx * dx + rvy * dy + rvz * dz) / dist : 0;   // > 0: i moves toward j
      const contact = pairCnt[pid] > 0;
      // impact: approaching from apart, close enough to touch within this frame's motion. A pair already in contact is
      // never re-classified: switching its level kicks it by the level's residual offset, and that kick read as a new
      // impact (a feedback loop that kept piles from ever sleeping)
      const impact = !contact && closing > 0.3 && gap <= this.margin + 1.5 * spd * this.dt;
      // class 1: in contact (resting or sliding): full detail (see above)
      const cls = impact ? 0 : contact ? 1 : 2;
      cand[n] = pid; candCls[n] = cls;
      candKey[n] = cls === 0 ? -closing : cls === 1 ? 0 : gap;
      // node pairs at the coarsest level L
      candEst[n] = contact ? Math.max(1, pairCnt[pid] * 4 ** (pairLev[pid] - L)) : 16;
      n++;
    }
    // sort candidates by (class, key): insertion sort, n is small (body pairs near each other)
    for (let a = 1; a < n; a++) {
      const p = cand[a], c = candCls[a], k = candKey[a], e = candEst[a];
      let b = a - 1;
      while (b >= 0 && (candCls[b] > c || (candCls[b] === c && candKey[b] > k))) {
        cand[b + 1] = cand[b]; candCls[b + 1] = candCls[b]; candKey[b + 1] = candKey[b]; candEst[b + 1] = candEst[b]; b--;
      }
      cand[b + 1] = p; candCls[b + 1] = c; candKey[b + 1] = k; candEst[b + 1] = e;
    }
    const est = (t, l) => candEst[t] * 4 ** (L - l);
    let total = 0;
    for (let t = 0; t < n; t++) { bcap[cand[t]] = L; total += est(t, L); fixed[t] = 0; }
    // pairs in contact (resting or sliding): full detail, always, and outside the budget. Coarse contact let leaning
    // stacks slide and never sleep, and any level switch while touching popped boxes (2026-10-29); sleeping makes
    // settled contact free
    const CL = Math.min(this.contactLevel, L);   // the fixed contact level (0 = full detail)
    for (let t = 0; t < n; t++) {
      const pid = cand[t];
      // lockLevels: a pair keeps the level it made contact at until it has been apart for SEP_FRAMES frames
      if (this.lockLevels && this.lockLev[pid] >= 0) {
        const l = Math.min(this.lockLev[pid], L);
        total += est(t, l) - est(t, L); bcap[pid] = l; fixed[t] = 1;
        continue;
      }
      if (candCls[t] !== 1) continue;
      total += est(t, CL) - est(t, L); bcap[pid] = CL; fixed[t] = 1;
    }
    // impacts (approaching, about to touch): never coarser than L1 (at budget 1000 impacts at L3 went 17-37 mm into
    // the floor), refined to L0 first, fastest first, while the budget lasts; exceeded rather than coarser
    const IMPACT_MAX = Math.min(1, L);
    for (let t = 0; t < n; t++) {
      if (candCls[t] !== 0 || fixed[t]) continue;
      total += est(t, IMPACT_MAX) - est(t, L); bcap[cand[t]] = IMPACT_MAX;
    }
    for (let t = 0; t < n; t++) {
      if (candCls[t] !== 0 || fixed[t] || bcap[cand[t]] === 0) continue;
      const d = est(t, 0) - est(t, IMPACT_MAX);
      if (total + d <= budget) { bcap[cand[t]] = 0; total += d; }
    }
    // the rest (apart, not about to touch) take what is left, breadth first
    for (let l = L - 1; l >= 0; l--) {
      for (let t = 0; t < n; t++) {
        if (fixed[t] || candCls[t] === 0 || bcap[cand[t]] !== l + 1) continue;
        const d = est(t, l) - est(t, l + 1);
        if (total + d <= budget) { bcap[cand[t]] = l; total += d; }
      }
    }
    let coarsest = 0, ci = 0, imp = 0;
    for (let t = 0; t < n; t++) {
      const l = bcap[cand[t]];
      if (l > coarsest) coarsest = l;
      if (candCls[t] === 0) { imp++; if (l > 0) ci++; }
    }
    st.coarsest = coarsest; st.coarseImpacts = ci; st.impacts = imp; st.overrun = total > budget; st.budget = budget;
  }

  // lockLevels: a pair with node pairs locks the level it found them at (contact onset); the lock is released once it
  // has had none for SEP_FRAMES frames
  updateLocks() {
    const { B, pairCnt, pairLev, lockLev, sepCount } = this;
    for (let pid = 0; pid < B * B; pid++) {
      if (pairCnt[pid] > 0) { if (lockLev[pid] < 0) lockLev[pid] = pairLev[pid]; sepCount[pid] = 0; }
      else if (lockLev[pid] >= 0 && ++sepCount[pid] > SEP_FRAMES) lockLev[pid] = -1;
    }
  }

  // group node pairs into patches (same body pair, same patch-sized cell of the lighter body's frame), patches in key
  // order as the Python argsort; pairs keep their detection order within a patch
  makePatches() {
    const { N, oa, ob, op, pba, pbb, invM, hKey, hVal, pKey, pSlot, pIdx, pCount, pRank, starts, cursor, patchOf, pairPatch } = this;
    const n = this.npairs, ps = this.patch, H = this.H, mask = H - 1, shift = 32 - this.HB;
    let P = 0;
    for (let k = 0; k < n; k++) {
      const bp = op[k], i = pba[bp], j = pbb[bp];
      const useB = invM[j] >= invM[i], m = (useB ? ob[k] : oa[k]) * NS, sc = this.bs[useB ? j : i];
      const ix = Math.floor(N[m] * sc / ps + 0.5) & 1023, iy = Math.floor(N[m + 1] * sc / ps + 0.5) & 1023, iz = Math.floor(N[m + 2] * sc / ps + 0.5) & 1023;
      const key = ((bp * 1024 + ix) * 1024 + iy) * 1024 + iz;
      if (!Number.isFinite(key)) { pairPatch[k] = -1; continue; }   // NaN guard: a non-finite pose never hangs the table
      let h = Math.imul(((ix << 20) | (iy << 10) | iz) ^ Math.imul(bp + 1, 0x9E3779B1), 0x85EBCA6B) >>> shift;
      while (hKey[h] !== -1 && hKey[h] !== key) h = (h + 1) & mask;
      let pid;
      if (hKey[h] === -1) {
        if (P >= this.pcap) { pairPatch[k] = -1; continue; }   // patch table full: drop the pair
        hKey[h] = key; hVal[h] = pid = P; pKey[P] = key; pSlot[P] = h; pCount[P] = 0; P++;
      } else pid = hVal[h];
      pairPatch[k] = pid; pCount[pid]++;
    }
    // rank the patches by key: sort patch ids by their keys (keys are unique). An earlier version packed key * 2^18 +
    // id into one float64, which is exact only for keys < 2^35, i.e. fewer than 32 body pairs; past that patches got
    // other patches' ids and mixed node pairs of different body pairs (piles of 8+ boxes exploded, 2026-10-29)
    for (let p = 0; p < P; p++) { pIdx[p] = p; hKey[pSlot[p]] = -1; }
    heapsortBy(pIdx, pKey, P);
    for (let r = 0; r < P; r++) pRank[pIdx[r]] = r;
    for (let p = 0; p < P; p++) cursor[pRank[p]] = pCount[p];   // counts in rank order
    let s = 0;
    for (let r = 0; r < P; r++) { const c = cursor[r]; starts[r] = s; cursor[r] = s; s += c; }
    starts[P] = s;
    for (let k = 0; k < n; k++) { const pid = pairPatch[k]; if (pid >= 0) patchOf[cursor[pRank[pid]]++] = k; }
    this.npatch = P;
  }

  // node k's world mean (and, by cov(), its world covariance) at the body's current pose, cached per stamp: offset in cW
  mean(side, k, R, ro, P, po, sc) {
    const o = 9 * (side * this.nn + k), st = side * this.nn + k;
    if (this.mStamp[st] !== this.stamp) {
      const N = this.N, n = k * NS, cW = this.cW, vx = N[n] * sc, vy = N[n + 1] * sc, vz = N[n + 2] * sc;
      cW[o] = R[ro] * vx + R[ro + 1] * vy + R[ro + 2] * vz + P[po];
      cW[o + 1] = R[ro + 3] * vx + R[ro + 4] * vy + R[ro + 5] * vz + P[po + 1];
      cW[o + 2] = R[ro + 6] * vx + R[ro + 7] * vy + R[ro + 8] * vz + P[po + 2];
      this.mStamp[st] = this.stamp;
    }
    return o;
  }

  cov(side, k, R, ro, sc) {
    const st = side * this.nn + k;
    if (this.cStamp[st] !== this.stamp) {
      const o = 9 * st + 3, cW = this.cW, s2 = sc * sc;
      rotCov6(R, ro, this.N, k * NS, cW, o);
      for (let c = 0; c < 6; c++) cW[o + c] *= s2;
      this.cStamp[st] = this.stamp;
    }
  }

  prevMean(side, k, R, ro, P, po, sc) {
    const o = 3 * (side * this.nn + k), st = side * this.nn + k;
    if (this.pStamp[st] !== this.pstamp) {
      const N = this.N, n = k * NS, cP = this.cP, vx = N[n] * sc, vy = N[n + 1] * sc, vz = N[n + 2] * sc;
      cP[o] = R[ro] * vx + R[ro + 1] * vy + R[ro + 2] * vz + P[po];
      cP[o + 1] = R[ro + 3] * vx + R[ro + 4] * vy + R[ro + 5] * vz + P[po + 1];
      cP[o + 2] = R[ro + 6] * vx + R[ro + 7] * vy + R[ro + 8] * vz + P[po + 2];
      this.pStamp[st] = this.pstamp;
    }
    return o;
  }

  // --- compact resting contacts -------------------------------------------------------------------------------------
  // A body pair calm for CALM_FRAMES frames (both bodies slower than CALM_V at their root spheres, in contact) is
  // compacted: each of its full-detail patches becomes one representative (the patch's contact point fixed on both
  // bodies and its normal, from the last full-detail solve), solved as a rigid non-penetration point with friction.
  // It re-expands when either body moves faster than EXPAND_V, a representative is violated by more than EXPAND_G,
  // or after REFRESH frames.
  relSpeed(i, j) {
    const { vel, om, N, root, bs } = this;
    const v = b => Math.hypot(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]) + Math.hypot(om[3 * b], om[3 * b + 1], om[3 * b + 2]) * N[root[b] * NS + 10] * bs[b];
    return Math.max(v(i), v(j));
  }

  updateCompact() {
    const { B, npatch, op, pba, pbb, patchOf, starts, pn, calm, compactF, compactAge, repN, rep, pos, quat, invM } = this;
    const st = this.compactStats;
    for (let bp = 0; bp < this.nbp; bp++) {
      const i = pba[bp], j = pbb[bp], pid = i * B + j;
      if (compactF[pid]) { compactAge[pid]++; continue; }
      if (invM[i] === 0 && invM[j] === 0) continue;
      calm[pid] = this.pairCnt[pid] > 0 && this.relSpeed(i, j) < CALM_V ? calm[pid] + 1 : 0;
    }
    // the pairs that just became calm: one representative per patch of the pair, its O-weighted contact point and
    // push direction over the patch's node pairs at the current poses (the patch solve only records patches it moved)
    const { N, oa, ob, bs } = this;
    for (let p = 0; p < npatch; p++) {
      const bp = op[patchOf[starts[p]]], i = pba[bp], j = pbb[bp], pid = i * B + j;
      if (compactF[pid] || calm[pid] < CALM_FRAMES || repN[pid] >= REPS) continue;
      const si = bs[i], sj = bs[j];
      quatMat(quat, 4 * i, this.Rs, 0); quatMat(quat, 4 * j, this.Rs, 9);
      let C = 0, gx_ = 0, gy_ = 0, gz_ = 0, sx = 0, sy = 0, sz = 0;
      for (let t = starts[p]; t < starts[p + 1]; t++) {
        const k = patchOf[t], na = oa[k] * NS, nb = ob[k] * NS;
        xfm(this.Rs, 0, pos, 3 * i, N, na, si); const ax = X3[0], ay = X3[1], az = X3[2];
        xfm(this.Rs, 9, pos, 3 * j, N, nb, sj);
        rotCov6(this.Rs, 0, N, na, A6); rotCov6(this.Rs, 9, N, nb, B6); scale6(A6, si * si); scale6(B6, sj * sj);
        const O = overlap(N[na + 9] * si * si * si, ax, ay, az, A6, 0, N[nb + 9] * sj * sj * sj, X3[0], X3[1], X3[2], B6, 0);
        if (O === 0) continue;
        const gl = Math.hypot(OV[1], OV[2], OV[3]);
        if (gl > 0) { gx_ += O * OV[1] / gl; gy_ += O * OV[2] / gl; gz_ += O * OV[3] / gl; }
        C += O; sx += O * OV[4]; sy += O * OV[5]; sz += O * OV[6];
      }
      const gn = Math.hypot(gx_, gy_, gz_);
      if (C <= 1e-30 || gn <= 0) continue;
      pn[3 * p] = -gx_ / gn; pn[3 * p + 1] = -gy_ / gn; pn[3 * p + 2] = -gz_ / gn;
      const cx = sx / C, cy = sy / C, cz = sz / C, o = (pid * REPS + repN[pid]) * 9;
      quatMat(quat, 4 * i, T9, 0);
      for (let k = 0; k < 3; k++) rep[o + k] = T9[k] * (cx - pos[3 * i]) + T9[3 + k] * (cy - pos[3 * i + 1]) + T9[6 + k] * (cz - pos[3 * i + 2]);
      quatMat(quat, 4 * j, T9, 0);
      for (let k = 0; k < 3; k++) {
        rep[o + 3 + k] = T9[k] * (cx - pos[3 * j]) + T9[3 + k] * (cy - pos[3 * j + 1]) + T9[6 + k] * (cz - pos[3 * j + 2]);
        rep[o + 6 + k] = T9[k] * pn[3 * p] + T9[3 + k] * pn[3 * p + 1] + T9[6 + k] * pn[3 * p + 2];
      }
      repN[pid]++;
    }
    let pairs = 0, reps = 0;
    for (let pid = 0; pid < B * B; pid++) {
      if (!compactF[pid] && calm[pid] >= CALM_FRAMES && repN[pid] > 0) { compactF[pid] = 1; compactAge[pid] = 0; calm[pid] = 0; st.compactions++; }
      else if (!compactF[pid]) repN[pid] = 0;
      if (compactF[pid]) { pairs++; reps += repN[pid]; }
    }
    st.pairs = pairs; st.reps = reps;
  }

  expandCompact() {
    const { B, compactF, compactAge, repN, calm, expandNext, cValid } = this;
    for (let pid = 0; pid < B * B; pid++) {
      if (!compactF[pid]) continue;
      const i = (pid / B) | 0, j = pid % B;
      if (expandNext[pid] || compactAge[pid] >= REFRESH || this.relSpeed(i, j) > EXPAND_V) {
        compactF[pid] = 0; repN[pid] = 0; calm[pid] = 0; expandNext[pid] = 0; cValid[pid] = 0; this.compactStats.expansions++;
      }
    }
  }

  // one Gauss-Seidel pass over the compact representatives (per substep): non-penetration along the stored normal,
  // then positional Coulomb friction at the point
  solveCompact(h, at) {
    const { B, compactF, repN, rep, pos, quat, prevQ, Rs, Rp, asleep, invS, expandNext } = this;
    for (let pid = 0; pid < B * B; pid++) {
      if (!compactF[pid]) continue;
      const i = (pid / B) | 0, j = pid % B, xi = 3 * i, xj = 3 * j;
      if ((asleep[i] || invS[i] === 0) && (asleep[j] || invS[j] === 0)) continue;
      for (let r = 0; r < repN[pid]; r++) {
        const o = (pid * REPS + r) * 9;
        quatMat(quat, 4 * i, Rs, 0); quatMat(quat, 4 * j, Rs, 9);
        const ax = Rs[0] * rep[o] + Rs[1] * rep[o + 1] + Rs[2] * rep[o + 2] + pos[xi];
        const ay = Rs[3] * rep[o] + Rs[4] * rep[o + 1] + Rs[5] * rep[o + 2] + pos[xi + 1];
        const az = Rs[6] * rep[o] + Rs[7] * rep[o + 1] + Rs[8] * rep[o + 2] + pos[xi + 2];
        const bx = Rs[9] * rep[o + 3] + Rs[10] * rep[o + 4] + Rs[11] * rep[o + 5] + pos[xj];
        const by = Rs[12] * rep[o + 3] + Rs[13] * rep[o + 4] + Rs[14] * rep[o + 5] + pos[xj + 1];
        const bz = Rs[15] * rep[o + 3] + Rs[16] * rep[o + 4] + Rs[17] * rep[o + 5] + pos[xj + 2];
        const nx = Rs[9] * rep[o + 6] + Rs[10] * rep[o + 7] + Rs[11] * rep[o + 8];
        const ny = Rs[12] * rep[o + 6] + Rs[13] * rep[o + 7] + Rs[14] * rep[o + 8];
        const nz = Rs[15] * rep[o + 6] + Rs[16] * rep[o + 7] + Rs[17] * rep[o + 8];
        const g = (ax - bx) * nx + (ay - by) * ny + (az - bz) * nz;   // A is pushed along -n: penetration grows with it
        if (g <= 0) continue;
        if (g > EXPAND_G) expandNext[pid] = 1;
        const cx = 0.5 * (ax + bx), cy = 0.5 * (ay + by), cz = 0.5 * (az + bz);
        const ln = Math.abs(this.contact(i, j, nx, ny, nz, cx, cy, cz, -g, at));
        if (this.mu > 0) {
          const rax = cx - pos[xi], ray = cy - pos[xi + 1], raz = cz - pos[xi + 2];
          const rbx = cx - pos[xj], rby = cy - pos[xj + 1], rbz = cz - pos[xj + 2];
          quatMat(quat, 4 * i, Rs, 0); quatMat(quat, 4 * j, Rs, 9);
          quatMat(prevQ, 4 * i, Rp, 0); quatMat(prevQ, 4 * j, Rp, 9);
          const dax = this.slip(Rs, Rp, 0, xi, rax, ray, raz), day = T3[1], daz = T3[2];
          const dbx = this.slip(Rs, Rp, 9, xj, rbx, rby, rbz), dby = T3[1], dbz = T3[2];
          const dpx = dax - dbx, dpy = day - dby, dpz = daz - dbz;
          const dn = dpx * nx + dpy * ny + dpz * nz;
          const tx = dpx - dn * nx, ty = dpy - dn * ny, tz = dpz - dn * nz;
          const m = Math.sqrt(tx * tx + ty * ty + tz * tz);
          if (m > 1e-12) {
            const ux = tx / m, uy = ty / m, uz = tz / m;
            const cax = ray * uz - raz * uy, cay = raz * ux - rax * uz, caz = rax * uy - ray * ux;
            const cbx = rby * uz - rbz * uy, cby = rbz * ux - rbx * uz, cbz = rbx * uy - rby * ux;
            const wt = invS[i] + this.wI(i, cax, cay, caz) + invS[j] + this.wI(j, cbx, cby, cbz);
            const dlt = -Math.min(m / wt, this.mu * ln);
            this.applyCorr(i, ux, uy, uz, cax, cay, caz, dlt);
            this.applyCorr(j, -ux, -uy, -uz, -cbx, -cby, -cbz, dlt);
          }
        }
      }
    }
  }

  // normal velocity damping at the compact representatives (as the patches get after each substep's velocity update)
  dampCompact(damp) {
    const { B, compactF, repN, rep, pos, quat, vel, om, invS, asleep, Iinv } = this;
    for (let pid = 0; pid < B * B; pid++) {
      if (!compactF[pid]) continue;
      const i = (pid / B) | 0, j = pid % B, xi = 3 * i, xj = 3 * j;
      if ((asleep[i] || invS[i] === 0) && (asleep[j] || invS[j] === 0)) continue;
      quatMat(quat, 4 * i, T9, 0);
      const Ri = Array.from(T9);
      quatMat(quat, 4 * j, T9, 0);
      for (let r = 0; r < repN[pid]; r++) {
        const o = (pid * REPS + r) * 9;
        const px = Ri[0] * rep[o] + Ri[1] * rep[o + 1] + Ri[2] * rep[o + 2] + pos[xi];
        const py = Ri[3] * rep[o] + Ri[4] * rep[o + 1] + Ri[5] * rep[o + 2] + pos[xi + 1];
        const pz = Ri[6] * rep[o] + Ri[7] * rep[o + 1] + Ri[8] * rep[o + 2] + pos[xi + 2];
        const nx = T9[0] * rep[o + 6] + T9[1] * rep[o + 7] + T9[2] * rep[o + 8];
        const ny = T9[3] * rep[o + 6] + T9[4] * rep[o + 7] + T9[5] * rep[o + 8];
        const nz = T9[6] * rep[o + 6] + T9[7] * rep[o + 7] + T9[8] * rep[o + 8];
        const rax = px - pos[xi], ray = py - pos[xi + 1], raz = pz - pos[xi + 2];
        const rbx = px - pos[xj], rby = py - pos[xj + 1], rbz = pz - pos[xj + 2];
        const vax = vel[xi] + om[xi + 1] * raz - om[xi + 2] * ray, vay = vel[xi + 1] + om[xi + 2] * rax - om[xi] * raz, vaz = vel[xi + 2] + om[xi] * ray - om[xi + 1] * rax;
        const vbx = vel[xj] + om[xj + 1] * rbz - om[xj + 2] * rby, vby = vel[xj + 1] + om[xj + 2] * rbx - om[xj] * rbz, vbz = vel[xj + 2] + om[xj] * rby - om[xj + 1] * rbx;
        const vn = (vax - vbx) * nx + (vay - vby) * ny + (vaz - vbz) * nz;
        const cax = ray * nz - raz * ny, cay = raz * nx - rax * nz, caz = rax * ny - ray * nx;
        const cbx = rby * nz - rbz * ny, cby = rbz * nx - rbx * nz, cbz = rbx * ny - rby * nx;
        const wn = invS[i] + this.wI(i, cax, cay, caz) + invS[j] + this.wI(j, cbx, cby, cbz);
        if (wn <= 0) continue;
        const Pm = -damp * vn / wn;
        if (invS[i] > 0 && !asleep[i]) {
          vel[xi] += invS[i] * Pm * nx; vel[xi + 1] += invS[i] * Pm * ny; vel[xi + 2] += invS[i] * Pm * nz;
          addIinv(Iinv, 9 * i, om, xi, cax * Pm, cay * Pm, caz * Pm, 1);
        }
        if (invS[j] > 0 && !asleep[j]) {
          vel[xj] -= invS[j] * Pm * nx; vel[xj + 1] -= invS[j] * Pm * ny; vel[xj + 2] -= invS[j] * Pm * nz;
          addIinv(Iinv, 9 * j, om, xj, cbx * Pm, cby * Pm, cbz * Pm, -1);
        }
      }
    }
  }

  // a body's correction: pos += inv_m lin dl, q integrated by Iinv ang dl
  applyCorr(b, lx, ly, lz, gx, gy, gz, dl) {
    const im = this.invS[b];
    if (im === 0) return;
    const pos = this.pos, I = this.Iinv, o = 9 * b;
    if (this.pj) {   // pjacobi: accumulate per (body, body pair)
      const sl = b * this.B * this.B + this.jBP, L = this.pjL, A = this.pjA;
      L[3 * sl] += im * lx * dl; L[3 * sl + 1] += im * ly * dl; L[3 * sl + 2] += im * lz * dl;
      A[3 * sl] += (I[o] * gx + I[o + 1] * gy + I[o + 2] * gz) * dl; A[3 * sl + 1] += (I[o + 3] * gx + I[o + 4] * gy + I[o + 5] * gz) * dl;
      A[3 * sl + 2] += (I[o + 6] * gx + I[o + 7] * gy + I[o + 8] * gz) * dl;
      if (this.pjMark[sl] !== this.jPatch) {
        if (this.pjN[sl] === 0) this.pjList[this.pjn++] = sl;
        this.pjMark[sl] = this.jPatch; this.pjN[sl]++;
      }
      return;
    }
    if (this.jac) {   // Jacobi: accumulate, applied averaged after the sweep (jacobiApply)
      const L = this.jL, A = this.jA;
      L[3 * b] += im * lx * dl; L[3 * b + 1] += im * ly * dl; L[3 * b + 2] += im * lz * dl;
      A[3 * b] += (I[o] * gx + I[o + 1] * gy + I[o + 2] * gz) * dl; A[3 * b + 1] += (I[o + 3] * gx + I[o + 4] * gy + I[o + 5] * gz) * dl;
      A[3 * b + 2] += (I[o + 6] * gx + I[o + 7] * gy + I[o + 8] * gz) * dl;
      if (this.jMark[b] !== this.jPatch) { this.jMark[b] = this.jPatch; this.jN[b]++; }
      return;
    }
    pos[3 * b] += im * lx * dl; pos[3 * b + 1] += im * ly * dl; pos[3 * b + 2] += im * lz * dl;
    qint(this.quat, 4 * b, (I[o] * gx + I[o + 1] * gy + I[o + 2] * gz) * dl, (I[o + 3] * gx + I[o + 4] * gy + I[o + 5] * gz) * dl,
      (I[o + 6] * gx + I[o + 7] * gy + I[o + 8] * gz) * dl, 1);
  }

  // Jacobi sweep end: every body moves by the mean of its patches' corrections
  jacobiApply() {
    const { pos, quat, jL, jA, jN } = this;
    for (let b = 0; b < this.B; b++) {
      const n = jN[b];
      if (n === 0) continue;
      pos[3 * b] += jL[3 * b] / n; pos[3 * b + 1] += jL[3 * b + 1] / n; pos[3 * b + 2] += jL[3 * b + 2] / n;
      qint(quat, 4 * b, jA[3 * b] / n, jA[3 * b + 1] / n, jA[3 * b + 2] / n, 1);
    }
    jL.fill(0); jA.fill(0); jN.fill(0);
  }

  // stickV: static-friction impulses are collected per patch during the damping pass and applied after it (per body the
  // mean over its sticking patches, so many patches of one contact do not add up)
  stickBegin() { (this.stk ??= { n: new Int32Array(this.B), dv: new Float64Array(3 * this.B), da: new Float64Array(3 * this.B) }); this.stk.n.fill(0); this.stk.dv.fill(0); this.stk.da.fill(0); }
  stickPatch(i, j, ux, uy, uz, sax, say, saz, sbx, sby, sbz, J) {
    const st = this.stk, invS = this.invS;
    if (invS[i] > 0 && !this.asleep[i]) { st.n[i]++; st.dv[3 * i] += invS[i] * J * ux; st.dv[3 * i + 1] += invS[i] * J * uy; st.dv[3 * i + 2] += invS[i] * J * uz; st.da[3 * i] += sax * J; st.da[3 * i + 1] += say * J; st.da[3 * i + 2] += saz * J; }
    if (invS[j] > 0 && !this.asleep[j]) { st.n[j]++; st.dv[3 * j] -= invS[j] * J * ux; st.dv[3 * j + 1] -= invS[j] * J * uy; st.dv[3 * j + 2] -= invS[j] * J * uz; st.da[3 * j] -= sbx * J; st.da[3 * j + 1] -= sby * J; st.da[3 * j + 2] -= sbz * J; }
  }
  stickApply() {
    const st = this.stk, I = this.Iinv, vel = this.vel, om = this.om;
    for (let b = 0; b < this.B; b++) {
      const n = st.n[b];
      if (!n) continue;
      const o = 9 * b, ax = st.da[3 * b] / n, ay = st.da[3 * b + 1] / n, az = st.da[3 * b + 2] / n;
      vel[3 * b] += st.dv[3 * b] / n; vel[3 * b + 1] += st.dv[3 * b + 1] / n; vel[3 * b + 2] += st.dv[3 * b + 2] / n;
      om[3 * b] += I[o] * ax + I[o + 1] * ay + I[o + 2] * az; om[3 * b + 1] += I[o + 3] * ax + I[o + 4] * ay + I[o + 5] * az;
      om[3 * b + 2] += I[o + 6] * ax + I[o + 7] * ay + I[o + 8] * az;
    }
  }

  // gpuForm damping: a body's velocity impulse (dv, and the torque arm x impulse for Iinv) per (body, body pair)
  dampAdd(b, vx, vy, vz, tx, ty, tz) {
    if (this.invS[b] === 0 || this.asleep[b]) return;
    const sl = b * this.B * this.B + this.jBP, L = this.pjL, A = this.pjA;
    L[3 * sl] += vx; L[3 * sl + 1] += vy; L[3 * sl + 2] += vz;
    A[3 * sl] += tx; A[3 * sl + 1] += ty; A[3 * sl + 2] += tz;
    if (this.pjMark[sl] !== this.jPatch) {
      if (this.pjN[sl] === 0) this.pjList[this.pjn++] = sl;
      this.pjMark[sl] = this.jPatch; this.pjN[sl]++;
    }
  }

  // gpuForm damping end: per body vel += the mean over its body pairs of each pair's mean dv, om += Iinv (the mean of
  // the mean arms). A body pair's n patches damp in turn under Gauss-Seidel, removing 1 - (1 - damp)^n of the normal
  // speed, not damp: each pair's mean is scaled to that (a plain mean left the bounce that pushed a box 30 mm into the
  // floor, seed 2). Across body pairs the mean, not the sum: summed full removals overshoot for a box between two
  // others (a pile sank 92 mm, a box left the room)
  dampApply() {
    const { pjL, pjA, pjN, pjList, vel, om, Iinv } = this, BB = this.B * this.B, d = this.restitution ? 1 : this.damp;   // restitution: a full target, the pair mean
    const per = this.dampPer ??= new Int32Array(this.B);
    per.fill(0);
    for (let k = 0; k < this.pjn; k++) per[Math.floor(pjList[k] / BB)]++;
    const form = this.dampForm === 'color' ? 'sum' : this.dampForm;   // colour: one body pair per body
    for (let k = 0; k < this.pjn; k++) {
      const sl = pjList[k], b = Math.floor(sl / BB), o = 9 * b;
      const n = form === 'plain' ? pjN[sl] : (form === 'mean' ? per[b] : 1) * pjN[sl] * d / (1 - (1 - d) ** pjN[sl]);
      vel[3 * b] += pjL[3 * sl] / n; vel[3 * b + 1] += pjL[3 * sl + 1] / n; vel[3 * b + 2] += pjL[3 * sl + 2] / n;
      const ax = pjA[3 * sl] / n, ay = pjA[3 * sl + 1] / n, az = pjA[3 * sl + 2] / n;
      om[3 * b] += Iinv[o] * ax + Iinv[o + 1] * ay + Iinv[o + 2] * az;
      om[3 * b + 1] += Iinv[o + 3] * ax + Iinv[o + 4] * ay + Iinv[o + 5] * az;
      om[3 * b + 2] += Iinv[o + 6] * ax + Iinv[o + 7] * ay + Iinv[o + 8] * az;
      for (let c = 0; c < 3; c++) { pjL[3 * sl + c] = 0; pjA[3 * sl + c] = 0; }
      pjN[sl] = 0;
    }
    this.pjn = 0;
  }

  // pjacobi sweep end: per body the sum over its body pairs of their mean correction
  pjacobiApply() {
    const { pjL, pjA, pjN, pjList, jL, jA, jN } = this, BB = this.B * this.B;
    for (let k = 0; k < this.pjn; k++) {
      const sl = pjList[k], b = Math.floor(sl / BB), n = pjN[sl];
      for (let c = 0; c < 3; c++) { jL[3 * b + c] += pjL[3 * sl + c] / n; jA[3 * b + c] += pjA[3 * sl + c] / n; pjL[3 * sl + c] = 0; pjA[3 * sl + c] = 0; }
      jN[b] = 1; pjN[sl] = 0;
    }
    this.pjn = 0;
    this.jacobiApply();
  }

  // solver 'color': patch order grouped by a greedy colouring of the body pairs (pairs sharing a dynamic body get
  // different colours), the patches of one body pair kept together; returns the order
  colorOrder(npatch) {
    const { op, pba, pbb, patchOf, starts, invS } = this, B = this.B;
    const bpOf = new Int32Array(npatch), seen = new Map();
    for (let p = 0; p < npatch; p++) { const bp = op[patchOf[starts[p]]]; bpOf[p] = bp; if (!seen.has(bp)) seen.set(bp, seen.size); }
    const bps = [...seen.keys()];
    const deg = b => (invS[b] > 0 ? 1 : 0);
    // larger body pairs first (more patches = longer sequential chain), then greedy smallest free colour
    const cnt = new Map(); for (let p = 0; p < npatch; p++) cnt.set(bpOf[p], (cnt.get(bpOf[p]) ?? 0) + 1);
    bps.sort((a, b) => cnt.get(b) - cnt.get(a));
    const used = Array.from({ length: B }, () => new Set()), col = new Map();
    let ncol = 0;
    for (const bp of bps) {
      const i = pba[bp], j = pbb[bp];
      let c = 0;
      while ((deg(i) && used[i].has(c)) || (deg(j) && used[j].has(c))) c++;
      col.set(bp, c); used[i].add(c); used[j].add(c); ncol = Math.max(ncol, c + 1);
    }
    this.colorStats.colors = ncol; this.colorStats.maxColors = Math.max(this.colorStats.maxColors, ncol);
    this.patchColor = Int32Array.from({ length: npatch }, (_, p) => col.get(bpOf[p]));
    return Array.from({ length: npatch }, (_, p) => p).sort((a, b) => col.get(bpOf[a]) - col.get(bpOf[b]) || a - b);
  }

  // u^T Iinv_b u
  wI(b, ux, uy, uz) {
    const I = this.Iinv, o = 9 * b;
    return ux * (I[o] * ux + I[o + 1] * uy + I[o + 2] * uz) + uy * (I[o + 3] * ux + I[o + 4] * uy + I[o + 5] * uz)
      + uz * (I[o + 6] * ux + I[o + 7] * uy + I[o + 8] * uz);
  }

  substeps() {
    // solver inverse masses: a sleeping body is static for the solver (it still wakes when hit faster than wake_v).
    // Sleeping bodies that took position corrections without integrating velocity let a box leaning on them creep
    // (~20 mm/s) forever
    const invS = this.invS;
    for (let b = 0; b < this.B; b++) invS[b] = this.sleepStatic && this.asleep[b] ? 0 : this.invM[b];
    const { B, N, pos, quat, vel, om, Ib, asleep, Iinv, Rs, Rp, prevP, prevQ, oa, ob, op, pba, pbb, starts, patchOf, pn, pxc, pl, pairN, stats } = this;
    const invM = invS;
    const nsub = this.nsub, h = this.dt / nsub, at = this.alpha / (h * h), npatch = this.npatch;
    const kappa = this.kappa, mu = this.mu, damp = this.damp, snapDeg = this.snapDeg, maxdep = this.maxdep, relax = this.relax;
    const sweptQ = this.sweptQ, sweptMin = this.sweptMin, g = this.grav;
    const SL = this.SL, SLS = this.SLS, hasLev = SLS > 4, bs = this.bs;
    let evals = 0;
    let nbp_ = 0;
    for (let p = 0; p < npatch; p++) nbp_ = Math.max(nbp_, op[patchOf[starts[p]]] + 1);
    pairN.fill(0, 0, 3 * Math.max(nbp_, 1));
    if (snapDeg > 0) {
      // one normal per body pair from the gradient of the pair's total overlap
      for (let p = 0; p < npatch; p++) {
        const bp = op[patchOf[starts[p]]], i = pba[bp], j = pbb[bp], si = bs[i], sj = bs[j];
        quatMat(quat, 4 * i, Rs, 0); quatMat(quat, 4 * j, Rs, 9);
        for (let t = starts[p]; t < starts[p + 1]; t++) {
          const k = patchOf[t], na = oa[k] * NS, nb = ob[k] * NS;
          xfm(Rs, 0, pos, 3 * i, N, na, si); const ax = X3[0], ay = X3[1], az = X3[2];
          xfm(Rs, 9, pos, 3 * j, N, nb, sj);
          rotCov6(Rs, 0, N, na, A6); rotCov6(Rs, 9, N, nb, B6); scale6(A6, si * si); scale6(B6, sj * sj);
          const O = overlap(N[na + 9] * si * si * si, ax, ay, az, A6, 0, N[nb + 9] * sj * sj * sj, X3[0], X3[1], X3[2], B6, 0);
          const gx = OV[1], gy = OV[2], gz = OV[3], gl = Math.sqrt(gx * gx + gy * gy + gz * gz);
          if (gl > 0) { pairN[3 * bp] -= O * gx / gl; pairN[3 * bp + 1] -= O * gy / gl; pairN[3 * bp + 2] -= O * gz / gl; }
        }
      }
      for (let bp = 0; bp < nbp_; bp++) {
        const nn = Math.hypot(pairN[3 * bp], pairN[3 * bp + 1], pairN[3 * bp + 2]);
        if (nn > 0) { pairN[3 * bp] /= nn; pairN[3 * bp + 1] /= nn; pairN[3 * bp + 2] /= nn; }
      }
    }
    const cosSnap = Math.cos(snapDeg * Math.PI / 180);
    const cW = this.cW, cP = this.cP;
    for (let s = 0; s < nsub; s++) {
      this.sub = (this.sub ?? 0) + 1;   // substep counter (friction anchors)
      this.pstamp++;   // previous poses are fixed for the substep
      for (let b = 0; b < B; b++) {
        prevP[3 * b] = pos[3 * b]; prevP[3 * b + 1] = pos[3 * b + 1]; prevP[3 * b + 2] = pos[3 * b + 2];
        if (this.depenSplit) { this.penP.fill(0, 3 * b, 3 * b + 3); this.penW.fill(0, 3 * b, 3 * b + 3); }
        for (let c = 0; c < 4; c++) prevQ[4 * b + c] = quat[4 * b + c];
        if (invM[b] === 0 || asleep[b]) continue;
        const x = 3 * b;
        vel[x] += g[0] * h; vel[x + 1] += g[1] * h; vel[x + 2] += g[2] * h;
        pos[x] += vel[x] * h; pos[x + 1] += vel[x + 1] * h; pos[x + 2] += vel[x + 2] * h;
        quatMat(quat, 4 * b, Rs, 0);
        // body-frame angular velocity, gyroscopic torque, back to world
        let wx = Rs[0] * om[x] + Rs[3] * om[x + 1] + Rs[6] * om[x + 2];
        let wy = Rs[1] * om[x] + Rs[4] * om[x + 1] + Rs[7] * om[x + 2];
        let wz = Rs[2] * om[x] + Rs[5] * om[x + 1] + Rs[8] * om[x + 2];
        const Ix = Ib[x] * wx, Iy = Ib[x + 1] * wy, Iz = Ib[x + 2] * wz;
        const tx = -(wy * Iz - wz * Iy), ty = -(wz * Ix - wx * Iz), tz = -(wx * Iy - wy * Ix);
        wx += h * tx / Ib[x]; wy += h * ty / Ib[x + 1]; wz += h * tz / Ib[x + 2];
        om[x] = Rs[0] * wx + Rs[1] * wy + Rs[2] * wz;
        om[x + 1] = Rs[3] * wx + Rs[4] * wy + Rs[5] * wz;
        om[x + 2] = Rs[6] * wx + Rs[7] * wy + Rs[8] * wz;
        qint(quat, 4 * b, om[x], om[x + 1], om[x + 2], h);
      }
      if (this.restitution) { (this.v0 ??= new Float64Array(3 * B)).set(vel); (this.w0 ??= new Float64Array(3 * B)).set(om); }   // pre-solve velocities
      for (let b = 0; b < B; b++) {
        if (invM[b] === 0) Iinv.fill(0, 9 * b, 9 * b + 9);   // static for the solver (as kernels.py: Iinv starts at 0)
        else {
          quatMat(quat, 4 * b, Rs, 0);
          const o = 9 * b, x = 3 * b;
          const d0 = Ib[x] === 0 ? 0 : 1 / Ib[x], d1 = Ib[x + 1] === 0 ? 0 : 1 / Ib[x + 1], d2 = Ib[x + 2] === 0 ? 0 : 1 / Ib[x + 2];
          for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++)
            Iinv[o + 3 * a + c] = Rs[3 * a] * d0 * Rs[3 * c] + Rs[3 * a + 1] * d1 * Rs[3 * c + 1] + Rs[3 * a + 2] * d2 * Rs[3 * c + 2];
        }
      }
      const cj = this.solver === 'cjacobi', pj = this.solver === 'pjacobi', jacobi = this.solver === 'jacobi' || cj;
      const order = (this.solver === 'color' || cj) && npatch ? this.colorOrder(npatch) : null;
      let curCol = -1;
      if (this.rollOn && pj && s === 0) {   // rolling under pjacobi: per body the body pairs with patches (each pair's
        // rolling correction targets the same rotation; summed over pairs it would overshoot, so each takes its share)
        const rc = this.rollPairs ??= new Float64Array(B), seen = new Set();
        rc.fill(0);
        for (let p = 0; p < npatch; p++) { const bp = op[patchOf[starts[p]]]; if (!seen.has(bp)) { seen.add(bp); rc[pba[bp]]++; rc[pbb[bp]]++; } }
      }
      for (let it = 0; it < this.iters; it++) {
      this.jac = jacobi; this.pj = pj;
      for (let pp = 0; pp < npatch; pp++) {
        // alternate the sweep direction every substep
        const q = s % 2 === 0 ? pp : npatch - 1 - pp, p = order ? order[q] : q;
        if (cj && this.patchColor[p] !== curCol) { this.jacobiApply(); curCol = this.patchColor[p]; }   // colour done
        this.jPatch++;
        const bp = op[patchOf[starts[p]]], i = pba[bp], j = pbb[bp], si = bs[i], sj = bs[j];
        const mi = si * si * si, mj = sj * sj * sj;   // node masses scale with the volume
        this.jBP = bp;
        if (it === 0) { pl[p] = 0; if (this.speculative) this.pgap[p] = Infinity; }
        if ((asleep[i] || invM[i] === 0) && (asleep[j] || invM[j] === 0)) continue;
        const xi = 3 * i, xj = 3 * j;
        quatMat(quat, 4 * i, Rs, 0); quatMat(quat, 4 * j, Rs, 9);   // Ri at 0, Rj at 9
        // swept (space-time) test only for fast pairs
        let doSw = false;
        if (sweptQ > 0) {
          const rdx = (pos[xi] - prevP[xi]) - (pos[xj] - prevP[xj]);
          const rdy = (pos[xi + 1] - prevP[xi + 1]) - (pos[xj + 1] - prevP[xj + 1]);
          const rdz = (pos[xi + 2] - prevP[xi + 2]) - (pos[xj + 2] - prevP[xj + 2]);
          const rot = (Math.hypot(om[xi], om[xi + 1], om[xi + 2]) + Math.hypot(om[xj], om[xj + 1], om[xj + 2])) * h * 0.1;
          doSw = Math.sqrt(rdx * rdx + rdy * rdy + rdz * rdz) + rot > sweptMin;
          if (doSw) { quatMat(prevQ, 4 * i, Rp, 0); quatMat(prevQ, 4 * j, Rp, 9); }
        }
        const Rpi = doSw ? Rp : Rs;
        this.stamp++; this.pstamp++;   // poses are fixed for this patch's pair loop; node caches are per side
        let tbest = 2, snx = 0, sny = 0, snz = 0, spen = 0, sxx = 0, sxy = 0, sxz = 0, swt = 0;
        WA.fill(0); WB.fill(0);   // overlap of the patch's coarse-level nodes per level (side i, side j)
        let C = 0, EO = 0, gux = 0, guy = 0, guz = 0, gAx = 0, gAy = 0, gAz = 0, xcx = 0, xcy = 0, xcz = 0;
        let unx = 0, uny = 0, unz = 0, sW = 0;
        const MS = this.matSum ??= new Float64Array(8); if (this.materials) MS.fill(0);   // per side O-weighted mu_s mu_k v_s e   // surfaceNormals: O-weighted outward normals of thin nodes, their weight
        OS.fill(0);
        for (let t = starts[p]; t < starts[p + 1]; t++) {
          const ka = oa[patchOf[t]], kb = ob[patchOf[t]], na = ka * NS, nb = kb * NS;
          const ca = this.mean(0, ka, Rs, 0, pos, xi, si), cb = this.mean(1, kb, Rs, 9, pos, xj, sj);
          this.cov(0, ka, Rs, 0, si); this.cov(1, kb, Rs, 9, sj);
          const ax = cW[ca], ay = cW[ca + 1], az = cW[ca + 2], bx = cW[cb], by = cW[cb + 1], bz = cW[cb + 2];
          if (doSw && !(this.sweptCutOnly && hasLev && (SL[SLS * ka + 4] > 0 || SL[SLS * kb + 4] > 0))) {
            // the two means move on straight lines over the substep: first t where q falls through swept_q
            // (sweptCutOnly: coarse pairs' means are not where their surface is; they keep the regular contact)
            const pa = this.prevMean(0, ka, Rpi, 0, prevP, xi, si), pb = this.prevMean(1, kb, Rpi, 9, prevP, xj, sj);
            const d0x = cP[pa] - cP[pb], d0y = cP[pa + 1] - cP[pb + 1], d0z = cP[pa + 2] - cP[pb + 2];
            const ddx = (ax - bx) - d0x, ddy = (ay - by) - d0y, ddz = (az - bz) - d0z;
            const Sxx = cW[ca + 3] + cW[cb + 3], Sxy = cW[ca + 4] + cW[cb + 4], Sxz = cW[ca + 5] + cW[cb + 5];
            const Syy = cW[ca + 6] + cW[cb + 6], Syz = cW[ca + 7] + cW[cb + 7], Szz = cW[ca + 8] + cW[cb + 8];
            const c00 = Syy * Szz - Syz * Syz, c01 = Sxz * Syz - Sxy * Szz, c02 = Sxy * Syz - Sxz * Syy;
            const c11 = Sxx * Szz - Sxz * Sxz, c12 = Sxy * Sxz - Sxx * Syz, c22 = Sxx * Syy - Sxy * Sxy;
            const idet = 1 / (Sxx * c00 + Sxy * c01 + Sxz * c02);
            const y0x = (c00 * d0x + c01 * d0y + c02 * d0z) * idet;
            const y0y = (c01 * d0x + c11 * d0y + c12 * d0z) * idet;
            const y0z = (c02 * d0x + c12 * d0y + c22 * d0z) * idet;
            const ydx = (c00 * ddx + c01 * ddy + c02 * ddz) * idet;
            const ydy = (c01 * ddx + c11 * ddy + c12 * ddz) * idet;
            const ydz = (c02 * ddx + c12 * ddy + c22 * ddz) * idet;
            const qa = ddx * ydx + ddy * ydy + ddz * ydz;
            const qb = ddx * y0x + ddy * y0y + ddz * y0z;
            const qc = d0x * y0x + d0y * y0y + d0z * y0z - sweptQ;
            if (qb < 0 && qa > 0) {
              const disc = qb * qb - qa * qc;
              let tc = 2;
              if (qc <= 0) tc = 0;
              else if (disc > 0) tc = (-qb - Math.sqrt(disc)) / qa;
              if (tc < 1) {
                const nx = y0x + tc * ydx, ny = y0y + tc * ydy, nz = y0z + tc * ydz;
                const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
                const pen = -(1 - tc) * (ddx * nx + ddy * ny + ddz * nz) / nl;
                if (pen > 0) {
                  if (tc < tbest) tbest = tc;
                  spen = Math.max(spen, pen);
                  snx += pen * nx / nl; sny += pen * ny / nl; snz += pen * nz / nl;
                  sxx += pen * 0.5 * (ax + bx); sxy += pen * 0.5 * (ay + by); sxz += pen * 0.5 * (az + bz);
                  swt += pen;
                }
              }
            }
          }
          let ox = ax, oy = ay, oz = az, behind = 0, bgx = 0, bgy = 0, bgz = 0, bw = 0;
          // oneSided 'soft': no facing cut; a pair behind a sheet counts as the blend w (one-sided: on the sheet, its
          // distance behind as depth, pushing along the sheet normal) + (1 - w) (its ordinary two-sided overlap), w the
          // facing weight max(0, -n . m): edge-on surfaces behave as before, face-on ones fully one-sided (scaling the
          // one-sided parts alone either pulled a box down through an inward gradient or lifted the boxes 0.2 m)
          const fCut = this.oneSided === 'soft' ? 0 : 0.5;
          // how squarely the partner's surface faces a sheet with outward normal (mx, my, mz): -n_partner . m; 1 for a
          // partner with no surface normal (a blob, or two-sided). Only surfaces facing within 60 degrees (> 0.5) are behind
          // each other: a box's side face at its bottom edge is not behind the floor (it pushed the box sideways)
          const facing = (k, ro, mx, my, mz) => {
            const sl = SLS * k;
            if (SL[sl + 5] === 0 || !(SL[sl + 3] <= 0.25 * N[k * NS + 11])) return 1;
            const sg = Math.sign(SL[sl + 5]), ux = SL[sl], uy = SL[sl + 1], uz = SL[sl + 2];
            return Math.max(0, -sg * ((Rs[ro] * ux + Rs[ro + 1] * uy + Rs[ro + 2] * uz) * mx + (Rs[ro + 3] * ux + Rs[ro + 4] * uy + Rs[ro + 5] * uz) * my + (Rs[ro + 6] * ux + Rs[ro + 7] * uy + Rs[ro + 8] * uz) * mz));
          };
          if (this.oneSided) {
            // a partner centre behind a solid-backed thin node's sheet: evaluated on the sheet, its distance behind added
            const sa_ = SLS * ka, sb_ = SLS * kb;
            // (slot 5 = sign (2 + D): D the solid held behind, scaled with the body; deeper than that is not behind it)
            if (Math.abs(SL[sa_ + 5]) >= 2 && SL[sa_ + 3] <= 0.25 * N[na + 11]) {
              const sg = Math.sign(SL[sa_ + 5]), ux = SL[sa_], uy = SL[sa_ + 1], uz = SL[sa_ + 2];
              const nx_ = sg * (Rs[0] * ux + Rs[1] * uy + Rs[2] * uz), ny_ = sg * (Rs[3] * ux + Rs[4] * uy + Rs[5] * uz), nz_ = sg * (Rs[6] * ux + Rs[7] * uy + Rs[8] * uz);
              const sd = nx_ * (bx - ax) + ny_ * (by - ay) + nz_ * (bz - az);
              const lat2 = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2 - sd * sd;   // within the sheet node's footprint
              const fw = sd < 0 ? facing(kb, 9, nx_, ny_, nz_) : 0;
              if (fw > fCut && -sd <= (Math.abs(SL[sa_ + 5]) - 2) * si && lat2 <= (N[na + 10] * si) ** 2) {   // the sheet moved onto b instead
                bw = fCut ? 1 : Math.min(1, fw);
                ox += sd * nx_; oy += sd * ny_; oz += sd * nz_; behind = -sd; bgx = nx_; bgy = ny_; bgz = nz_;
              }
            }
            if (!behind && Math.abs(SL[sb_ + 5]) >= 2 && SL[sb_ + 3] <= 0.25 * N[nb + 11]) {
              const sg = Math.sign(SL[sb_ + 5]), ux = SL[sb_], uy = SL[sb_ + 1], uz = SL[sb_ + 2];
              const nx_ = sg * (Rs[9] * ux + Rs[10] * uy + Rs[11] * uz), ny_ = sg * (Rs[12] * ux + Rs[13] * uy + Rs[14] * uz), nz_ = sg * (Rs[15] * ux + Rs[16] * uy + Rs[17] * uz);
              const sd = nx_ * (ax - bx) + ny_ * (ay - by) + nz_ * (az - bz);
              const lat2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2 - sd * sd;
              const fw = sd < 0 ? facing(ka, 0, nx_, ny_, nz_) : 0;
              if (fw > fCut && -sd <= (Math.abs(SL[sb_ + 5]) - 2) * sj && lat2 <= (N[nb + 10] * sj) ** 2) {   // a moved back onto the sheet
                bw = fCut ? 1 : Math.min(1, fw);
                ox -= sd * nx_; oy -= sd * ny_; oz -= sd * nz_; behind = -sd; bgx = -nx_; bgy = -ny_; bgz = -nz_;
              }
            }
          }
          // evaluations of this pair: one-sided (weight bw) and, for a soft partial facing, the ordinary one (1 - bw)
          // contactRefine: a pair in contact asks the Gaussians below. Each side whose cut node covers render Gaussians
          // (exactSupport data) contributes those Gaussians' own overlaps, gradients and spreads instead of the node's;
          // the node stays the unit for detection and the patch (2026-10-30)
          let nA = 1, nB = 1;
          if (this.contactRefine && this.xs) { nA = this.subItems(0, ka, i, 0, si); nB = this.subItems(1, kb, j, 9, sj); }
          const SA = this.subA, SB = this.subB;
          const nEv = behind > 0 && bw < 1 ? 2 : 1;
          for (let ev = 0; ev < nEv; ev++) {
          const one = behind > 0 && ev === 0, wt = behind > 0 ? (one ? bw : 1 - bw) : 1;
          const shx = one ? ox - ax : 0, shy = one ? oy - ay : 0, shz = one ? oz - az : 0;   // a's one-sided move
          for (let sa = 0; sa < Math.max(nA, 1); sa++) for (let sb = 0; sb < Math.max(nB, 1); sb++) {
          const ra = nA > 0, rb = nB > 0;   // refined sides (0: the node itself)
          const aA = ra && SA ? SA.c : cW, aO = ra && SA ? 6 * sa : ca + 3, bA = rb && SB ? SB.c : cW, bO = rb && SB ? 6 * sb : cb + 3;
          const amx = (ra && SA ? SA.x[3 * sa] : ax) + shx, amy = (ra && SA ? SA.x[3 * sa + 1] : ay) + shy, amz = (ra && SA ? SA.x[3 * sa + 2] : az) + shz;
          const bmx = rb && SB ? SB.x[3 * sb] : bx, bmy = rb && SB ? SB.x[3 * sb + 1] : by, bmz = rb && SB ? SB.x[3 * sb + 2] : bz;
          const O = overlap(ra && SA ? SA.m[sa] : N[na + 9] * mi, amx, amy, amz, aA, aO, rb && SB ? SB.m[sb] : N[nb + 9] * mj, bmx, bmy, bmz, bA, bO) * wt;
          evals++;
          if (O === 0) continue;
          C += O;
          let offA = N[na + 15] * si, offB = N[nb + 15] * sj;
          if ((this.directionalExtent && this.EXT) || (this.xs && !this.contactRefine)) {
            // each side's surface along the pair's direction (its overlap gradient, a toward b: the Mahalanobis
            // separation, across a thin floor not along it as the centre offset was; on a sheet, its outward normal):
            // the node's measured extent there minus its moment support, less the part's face reference
            let dx = OV[1], dy = OV[2], dz = OV[3];
            // after the first sweep: the patch's own normal (what the contact pushes along; per-pair gradients scatter
            // and the offsets with them: the boxes never settled). n pushes i away from j, so a toward b is -n
            if (it > 0 && (pn[3 * p] || pn[3 * p + 1] || pn[3 * p + 2])) { dx = -pn[3 * p]; dy = -pn[3 * p + 1]; dz = -pn[3 * p + 2]; }
            else if (one) { dx = bgx; dy = bgy; dz = bgz; }
            const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dl > 0) {
              dx /= dl; dy /= dl; dz /= dl;
              if (this.xs) {   // exactSupport: the covered render Gaussians' own support
                if (this.xs.ref[i] === this.xs.ref[i]) offA = (this.exactOffset(ka, i, Rs, 0, dx, dy, dz) - this.xs.ref[i]) * si;
                if (this.xs.ref[j] === this.xs.ref[j]) offB = (this.exactOffset(kb, j, Rs, 9, -dx, -dy, -dz) - this.xs.ref[j]) * sj;
              } else {
                if (!Number.isNaN(this.extRef[i])) offA = this.dirOffset(ka, Rs, 0, dx, dy, dz) * si - this.extRef[i] * si;
                if (!Number.isNaN(this.extRef[j])) offB = this.dirOffset(kb, Rs, 9, -dx, -dy, -dz) * sj - this.extRef[j] * sj;
              }
            }
          }
          EO += O * (offA + offB + (one ? behind : 0));
          if (hasLev) {
            const la_ = SL[SLS * ka + 4], lb_ = SL[SLS * kb + 4];
            if (la_ > 0 && la_ < 4) WA[la_] += O;
            if (lb_ > 0 && lb_ < 4) WB[lb_] += O;
          }
          let gx = OV[1] * wt, gy = OV[2] * wt, gz = OV[3] * wt;
          let gl = Math.sqrt(gx * gx + gy * gy + gz * gz);
          // oneSided, partner behind the sheet: evaluated on the sheet the overlap has no gradient across it; the pair
          // pushes along the sheet's outward normal, and its depth is the distance behind (EO), not the gradient's
          if (one) { gx = bgx; gy = bgy; gz = bgz; gl = 1; }
          if (gl > 0) { gux += O * gx / gl; guy += O * gy / gl; guz += O * gz / gl; }
          for (let c = 0; c < 6; c++) OS[c] += O * (aA[aO + c] + bA[bO + c]);
          if (!one) { gAx += gx; gAy += gy; gAz += gz; }
          xcx += O * OV[4]; xcy += O * OV[5]; xcz += O * OV[6];
          if (this.materials && SLS > 10) for (let c = 0; c < 4; c++) { MS[c] += O * SL[SLS * ka + 6 + c]; MS[4 + c] += O * SL[SLS * kb + 6 + c]; }
          if (this.surfaceNormals && SLS > 5) {
            // each thin node's outward normal (world); two-sided ones face the partner body. The patch normal n pushes
            // i away from j (contact(): i moves along +n): j's outward normals count as they are, i's negated
            for (let side = 0; side < 2; side++) {
              const k = side === 0 ? ka : kb, sl = SLS * k, b_ = side === 0 ? i : j, o_ = side === 0 ? j : i;
              if (!(SL[sl + 3] <= 0.25 * N[k * NS + 11])) continue;   // thick or volumetric: the gradient decides
              const ro = side === 0 ? 0 : 9, ux = SL[sl], uy = SL[sl + 1], uz = SL[sl + 2];
              let wx = Rs[ro] * ux + Rs[ro + 1] * uy + Rs[ro + 2] * uz, wy = Rs[ro + 3] * ux + Rs[ro + 4] * uy + Rs[ro + 5] * uz, wz = Rs[ro + 6] * ux + Rs[ro + 7] * uy + Rs[ro + 8] * uz;
              let sg = Math.sign(SL[sl + 5]);
              if (sg === 0) {   // two-sided: toward the partner's centre
                const mx = side === 0 ? ax : bx, my = side === 0 ? ay : by, mz = side === 0 ? az : bz;
                sg = (pos[3 * o_] - mx) * wx + (pos[3 * o_ + 1] - my) * wy + (pos[3 * o_ + 2] - mz) * wz >= 0 ? 1 : -1;
              }
              // normalSide: each flake weighted by how squarely it faces the pair's separation (|w . g|, g the pair's
              // overlap gradient: its projected area along the approach); an edge-on flake (a box's side face at its
              // bottom edge, on the floor) adds nothing (2026-10-30: the plain sum tilted edge patches 45 degrees)
              const face = this.normalSide && gl > 0 ? Math.abs(wx * gx + wy * gy + wz * gz) / gl : 1;
              const f = O * sg * (side === 0 ? -1 : 1) * face;
              unx += f * wx; uny += f * wy; unz += f * wz; sW += O;
              void b_;
            }
          }
          }
          }
        }
        if (this.materials && SLS > 10 && C > 0) {   // the patch's combined material (geometric means, max restitution)
          const pm = this.patchMat ??= new Float64Array(4 * this.pcap);
          for (let c = 0; c < 3; c++) pm[4 * p + c] = Math.sqrt((MS[c] / C) * (MS[4 + c] / C));
          pm[4 * p + 3] = Math.max(MS[3] / C, MS[7] / C);
        }
        let nx, ny, nz, cx, cy, cz, ln;
        if (tbest < 1) {
          // speculative contact from the earliest time of impact
          let l = Math.sqrt(snx * snx + sny * sny + snz * snz);
          nx = snx / l; ny = sny / l; nz = snz / l;
          cx = sxx / swt; cy = sxy / swt; cz = sxz / swt;
          // better normal: the overlap-weighted push direction of the patch's pairs at the time of impact
          let c2 = 0, g2x = 0, g2y = 0, g2z = 0, x2x = 0, x2y = 0, x2z = 0;
          for (let t = starts[p]; t < starts[p + 1]; t++) {
            const k = patchOf[t], na = oa[k] * NS, nb = ob[k] * NS;
            xfm(Rs, 0, pos, xi, N, na, si); let ax = X3[0], ay = X3[1], az = X3[2];
            xfm(Rs, 9, pos, xj, N, nb, sj); let bx = X3[0], by = X3[1], bz = X3[2];
            xfm(Rp, 0, prevP, xi, N, na, si); ax = X3[0] + tbest * (ax - X3[0]); ay = X3[1] + tbest * (ay - X3[1]); az = X3[2] + tbest * (az - X3[2]);
            xfm(Rp, 9, prevP, xj, N, nb, sj); bx = X3[0] + tbest * (bx - X3[0]); by = X3[1] + tbest * (by - X3[1]); bz = X3[2] + tbest * (bz - X3[2]);
            rotCov6(Rs, 0, N, na, A6); rotCov6(Rs, 9, N, nb, B6); scale6(A6, si * si); scale6(B6, sj * sj);
            const O = overlap(N[na + 9] * mi, ax, ay, az, A6, 0, N[nb + 9] * mj, bx, by, bz, B6, 0);
            const gx = OV[1], gy = OV[2], gz = OV[3], gl = Math.sqrt(gx * gx + gy * gy + gz * gz);
            if (O > 0 && gl > 0) {
              c2 += O;
              g2x -= O * gx / gl; g2y -= O * gy / gl; g2z -= O * gz / gl;
              x2x += O * (OV[4] + (1 - tbest) * (pos[xi] - prevP[xi]));
              x2y += O * (OV[5] + (1 - tbest) * (pos[xi + 1] - prevP[xi + 1]));
              x2z += O * (OV[6] + (1 - tbest) * (pos[xi + 2] - prevP[xi + 2]));
            }
          }
          l = Math.sqrt(g2x * g2x + g2y * g2y + g2z * g2z);
          if (c2 > 0 && l > 0) { nx = g2x / l; ny = g2y / l; nz = g2z / l; cx = x2x / c2; cy = x2y / c2; cz = x2z / c2; }
          const dl = this.contact(i, j, nx, ny, nz, cx, cy, cz, -spen, at);
          ln = Math.abs(dl);
          this.lastDl = dl;
          if (this.impactLog) {   // inspection (opt-in): the swept contact's normal and depth against the other normals
            const gn = Math.hypot(gux, guy, guz), sl_ = Math.hypot(unx, uny, unz), pl_ = Math.hypot(snx, sny, snz);
            this.impactLog.push({ sub: this.sub, it, p, bp, i, j, swept: 1, tc: tbest, depth: spen, ln, C, thin: C > 0 ? sW / C : 0,
              n: [nx, ny, nz], nPen: [snx / pl_, sny / pl_, snz / pl_], nGrad: gn > 0 ? [-gux / gn, -guy / gn, -guz / gn] : null,
              nSurf: sl_ > 0 ? [unx / sl_, uny / sl_, unz / sl_] : null, x: [cx, cy, cz] });
          }
          stats[4] += 1; stats[5] = Math.max(stats[5], spen);
        } else if (C <= 1e-30) {
          continue;
        } else {
          // overlap-derived separation along the patch normal (mode 3)
          const gn = Math.sqrt(gux * gux + guy * guy + guz * guz);
          if (gn <= 1e-300) continue;
          nx = -gux / gn; ny = -guy / gn; nz = -guz / gn;
          cx = xcx / C; cy = xcy / C; cz = xcz / C;
          const sl_ = Math.hypot(unx, uny, unz);
          if (this.surfaceNormals && sW >= C && sl_ > 0) {   // mostly thin nodes (sW counts both sides: >= half)
            nx = unx / sl_; ny = uny / sl_; nz = unz / sl_;   // (a consistency^4 side weighting fixed 2 Drops but cost piles: dropped)

          } else if (snapDeg > 0 && nx * pairN[3 * bp] + ny * pairN[3 * bp + 1] + nz * pairN[3 * bp + 2] > cosSnap) {
            nx = pairN[3 * bp]; ny = pairN[3 * bp + 1]; nz = pairN[3 * bp + 2];
          }
          const s2 = (nx * nx * OS[0] + ny * ny * OS[3] + nz * nz * OS[5] + 2 * (nx * ny * OS[1] + nx * nz * OS[2] + ny * nz * OS[4])) / C;
          const dsep = s2 * Math.max(-(gAx * nx + gAy * ny + gAz * nz), 0) / C;
          // detail levels: a coarse level's calibrated rest offset along the body's box axes, weighted by the contact
          // normal's components there (bake_physics.py calibrate, kernels.py)
          let lo = 0;
          for (let side = 0; side < 2; side++) {
            const W = side === 0 ? WA : WB, b_ = side === 0 ? i : j;
            if (W[1] + W[2] + W[3] === 0) continue;
            quatMat(quat, 4 * b_, T9, 0);
            const nbx = T9[0] * nx + T9[3] * ny + T9[6] * nz, nby = T9[1] * nx + T9[4] * ny + T9[7] * nz, nbz = T9[2] * nx + T9[5] * ny + T9[8] * nz;
            for (let l = 1; l < 4; l++) {
              if (W[l] === 0) continue;
              for (let k2 = 0; k2 < 3; k2++) {
                const o = 9 * b_ + 3 * k2, a = this.baxes[o] * nbx + this.baxes[o + 1] * nby + this.baxes[o + 2] * nbz;
                lo += W[l] * a * a * this.loff[12 * b_ + 3 * l + k2];
              }
            }
          }
          let Cv = kappa * Math.sqrt(s2) + (EO + lo) / C - dsep;   // node offsets (EO) and level offsets (lo)
          if (Cv <= 0) {
            if (this.speculative && -Cv < this.margin && -Cv < this.pgap[p]) {   // the velocity pass caps the approach
              this.pgap[p] = -Cv;
              pn[3 * p] = nx; pn[3 * p + 1] = ny; pn[3 * p + 2] = nz; pxc[3 * p] = cx; pxc[3 * p + 1] = cy; pxc[3 * p + 2] = cz;
            }
            continue;
          }
          if (this.sleepDepth > 0) { const dp = this.depthB; if (Cv > dp[i]) dp[i] = Cv; if (Cv > dp[j]) dp[j] = Cv; }
          // cap the depenetration speed, but always allow stopping the current approach
          const rix = cx - pos[xi], riy = cy - pos[xi + 1], riz = cz - pos[xi + 2];
          const rjx = cx - pos[xj], rjy = cy - pos[xj + 1], rjz = cz - pos[xj + 2];
          const vax = vel[xi] + om[xi + 1] * riz - om[xi + 2] * riy, vay = vel[xi + 1] + om[xi + 2] * rix - om[xi] * riz, vaz = vel[xi + 2] + om[xi] * riy - om[xi + 1] * rix;
          const vbx = vel[xj] + om[xj + 1] * rjz - om[xj + 2] * rjy, vby = vel[xj + 1] + om[xj + 2] * rjx - om[xj] * rjz, vbz = vel[xj + 2] + om[xj] * rjy - om[xj + 1] * rjx;
          const vapp = Math.max(-((vax - vbx) * nx + (vay - vby) * ny + (vaz - vbz) * nz), 0);
          Cv = Math.min(Cv, (maxdep + vapp) * h) * relax;
          // split depenetration: the part of the correction beyond stopping this substep's approach recovers existing
          // overlap; it moves the bodies but must not turn into separating velocity (a resting box popping)
          const penPart = Math.max(0, Cv - vapp * h * relax);
          this.popStats.depenSum += penPart;   // inspection: overlap recovered by depenetration (m, summed)
          const penShare = this.depenSplit && Cv > 0 ? penPart / Cv : 0;
          ln = Math.abs(this.contact(i, j, nx, ny, nz, cx, cy, cz, -Cv, at, penShare));
          this.lastDl = -ln;
          if (this.impactLog) this.impactLog.push({ sub: this.sub, it, p, bp, i, j, swept: 0, depth: Cv, ln, C, thin: sW / C, n: [nx, ny, nz],
            nGrad: [-gux / gn, -guy / gn, -guz / gn], nSurf: sl_ > 0 ? [unx / sl_, uny / sl_, unz / sl_] : null, x: [cx, cy, cz] });
        }
        pn[3 * p] = nx; pn[3 * p + 1] = ny; pn[3 * p + 2] = nz;
        pxc[3 * p] = cx; pxc[3 * p + 1] = cy; pxc[3 * p + 2] = cz; pl[p] = Math.max(pl[p], ln);
        stats[1] += ln; if (ln > stats[3]) stats[3] = ln;
        if (C > stats[0]) stats[0] = C;
        // positional Coulomb friction at the patch point (fricSecond, pjacobi: queued for a second pass after this
        // sweep's normal corrections are applied, as a Gauss-Seidel patch sees its own normal correction)
        if (mu > 0) {
          if (this.fricSecond && this.pj) { (this.fjobs ??= []).push([p, bp, i, j, nx, ny, nz, cx, cy, cz, ln]); }
          else this.frictionPatch(p, i, j, nx, ny, nz, cx, cy, cz, ln, h);
        }
        if (this.rollOn && ln > 0) this.rollingPatch(i, j, nx, ny, nz, cx, cy, cz, ln);
      }
      if (jacobi) { this.jac = false; this.jacobiApply(); curCol = -1; }
      if (pj) { this.pj = false; this.pjacobiApply(); }
      if (pj && this.fricSecond && this.fjobs?.length) {
        // the friction pass: from the poses after the normal corrections, the same per-(body, body pair) means
        this.pj = true;
        for (const [fp, fbp, fi, fj, fnx, fny, fnz, fcx, fcy, fcz, fln] of this.fjobs) {
          this.jPatch++; this.jBP = fbp;
          this.frictionPatch(fp, fi, fj, fnx, fny, fnz, fcx, fcy, fcz, fln, h);
        }
        this.fjobs.length = 0;
        this.pj = false; this.pjacobiApply();
      }
      }
      if (this.frictionAnchors) this.anchorUpdate();
      if (this.compactOn) this.solveCompact(h, at);
      // velocities from the positional update
      for (let b = 0; b < B; b++) {
        if (invM[b] === 0 || asleep[b]) continue;
        const x = 3 * b, o = 4 * b;
        vel[x] = (pos[x] - prevP[x]) / h; vel[x + 1] = (pos[x + 1] - prevP[x + 1]) / h; vel[x + 2] = (pos[x + 2] - prevP[x + 2]) / h;
        const pw = this.depenSplit;
        if (pw) { const P = this.penP; vel[x] -= P[x] / h; vel[x + 1] -= P[x + 1] / h; vel[x + 2] -= P[x + 2] / h; }
        const p0 = prevQ[o], p1 = prevQ[o + 1], p2 = prevQ[o + 2], p3 = prevQ[o + 3];
        const q0 = quat[o], q1 = quat[o + 1], q2 = quat[o + 2], q3 = quat[o + 3];
        const w0 = q0 * p0 + q1 * p1 + q2 * p2 + q3 * p3;
        const qx = -q0 * p1 + q1 * p0 - q2 * p3 + q3 * p2;
        const qy = -q0 * p2 + q1 * p3 + q2 * p0 - q3 * p1;
        const qz = -q0 * p3 - q1 * p2 + q2 * p1 + q3 * p0;
        const sgn = w0 >= 0 ? 1 : -1;
        om[x] = 2 * qx / h * sgn; om[x + 1] = 2 * qy / h * sgn; om[x + 2] = 2 * qz / h * sgn;
        if (pw) { const Wp = this.penW; om[x] -= Wp[x] / h; om[x + 1] -= Wp[x + 1] / h; om[x + 2] -= Wp[x + 2] / h; }
      }
      if (damp > 0 && this.compactOn) this.dampCompact(damp);
      if (this.stickV > 0) this.stickBegin();
      const dSweeps = this.gpuForm && this.dampForm !== 'gs' ? this.dampIters : 1;
      const dOrder = this.gpuForm && this.dampForm === 'color' && npatch ? this.colorOrder(npatch) : null;
      let dCol = -1;
      // restitution (in place of damp): each contact's normal velocity is set to -e times its approach speed before the
      // substep's solve (the patch's material restitution), or to 0 for approaches slower than 2 |g| h (resting: no
      // bounce from gravity's own step); XPBD's velocity pass (Mueller et al. 2020)
      const rest = this.restitution, vThr = 2 * Math.hypot(g[0], g[1], g[2]) * h, v0 = this.v0, w0 = this.w0;
      if (damp > 0 || rest) for (let ds = 0; ds < dSweeps; ds++) {
        for (let pp = 0; pp < npatch; pp++) {
          const p = dOrder ? dOrder[pp] : pp;
          if (dOrder && this.patchColor[p] !== dCol) { if (dCol >= 0) this.dampApply(); dCol = this.patchColor[p]; }   // colour done
          const specP = pl[p] <= 0;   // speculative: no correction this substep, inside the margin
          if (specP && !(this.speculative && this.pgap[p] < Infinity)) continue;
          const bp = op[patchOf[starts[p]]], i = pba[bp], j = pbb[bp], xi = 3 * i, xj = 3 * j;
          const nx = pn[3 * p], ny = pn[3 * p + 1], nz = pn[3 * p + 2];
          const rax = pxc[3 * p] - pos[xi], ray = pxc[3 * p + 1] - pos[xi + 1], raz = pxc[3 * p + 2] - pos[xi + 2];
          const rbx = pxc[3 * p] - pos[xj], rby = pxc[3 * p + 1] - pos[xj + 1], rbz = pxc[3 * p + 2] - pos[xj + 2];
          const vax = vel[xi] + om[xi + 1] * raz - om[xi + 2] * ray, vay = vel[xi + 1] + om[xi + 2] * rax - om[xi] * raz, vaz = vel[xi + 2] + om[xi] * ray - om[xi + 1] * rax;
          const vbx = vel[xj] + om[xj + 1] * rbz - om[xj + 2] * rby, vby = vel[xj + 1] + om[xj + 2] * rbx - om[xj] * rbz, vbz = vel[xj + 2] + om[xj] * rby - om[xj + 1] * rbx;
          const vn = (vax - vbx) * nx + (vay - vby) * ny + (vaz - vbz) * nz;
          const cax = ray * nz - raz * ny, cay = raz * nx - rax * nz, caz = rax * ny - ray * nx;
          const cbx = rby * nz - rbz * ny, cby = rbz * nx - rbx * nz, cbz = rbx * ny - rby * nx;
          const wn = invM[i] + this.wI(i, cax, cay, caz) + invM[j] + this.wI(j, cbx, cby, cbz);
          if (wn <= 0) continue;
          let Pm = -damp * vn / wn;
          if (specP) {   // approach capped at gap / h, nothing else (no restitution, no friction)
            const lim = -this.pgap[p] / h;
            if (vn >= lim) continue;
            Pm = (lim - vn) / wn;
          } else if (rest) {
            const pax = v0[xi] + w0[xi + 1] * raz - w0[xi + 2] * ray, pay = v0[xi + 1] + w0[xi + 2] * rax - w0[xi] * raz, paz = v0[xi + 2] + w0[xi] * ray - w0[xi + 1] * rax;
            const pbx = v0[xj] + w0[xj + 1] * rbz - w0[xj + 2] * rby, pby = v0[xj + 1] + w0[xj + 2] * rbx - w0[xj] * rbz, pbz = v0[xj + 2] + w0[xj] * rby - w0[xj + 1] * rbx;
            const vnPre = (pax - pbx) * nx + (pay - pby) * ny + (paz - pbz) * nz;   // < 0: approaching
            const e = this.materials && this.patchMat ? this.patchMat[4 * p + 3] : 0;
            Pm = ((vnPre < -vThr ? -e * vnPre : 0) - vn) / wn;
          }
          if (this.stickV > 0 && !this.inCompactDamp) {
            // velocity-level static friction: the tangential relative velocity at the patch point, removed when slow
            // and within mu times the substep's normal impulse (|dl| / h), collected per body (stickApply)
            const vtx = (vax - vbx) - vn * nx, vty = (vay - vby) - vn * ny, vtz = (vaz - vbz) - vn * nz, vt = Math.hypot(vtx, vty, vtz);
            if (vt > 1e-9 && vt < this.stickV) {
              const ux = vtx / vt, uy = vty / vt, uz = vtz / vt;
              const sax = ray * uz - raz * uy, say = raz * ux - rax * uz, saz = rax * uy - ray * ux;
              const sbx = rby * uz - rbz * uy, sby = rbz * ux - rbx * uz, sbz = rbx * uy - rby * ux;
              const wt = invM[i] + this.wI(i, sax, say, saz) + invM[j] + this.wI(j, sbx, sby, sbz);
              if (wt > 0 && vt / wt <= this.muAtP(p, vt) * pl[p] / h) this.stickPatch(i, j, ux, uy, uz, sax, say, saz, sbx, sby, sbz, -vt / wt);
            }
          }
          if (this.gpuForm && this.dampForm !== 'gs') {   // Jacobi: per (body, body pair) sums, applied as the sum of the pairs' means below
            this.jPatch++; this.jBP = bp; this.pj = true;
            this.dampAdd(i, invM[i] * Pm * nx, invM[i] * Pm * ny, invM[i] * Pm * nz, cax * Pm, cay * Pm, caz * Pm);
            this.dampAdd(j, -invM[j] * Pm * nx, -invM[j] * Pm * ny, -invM[j] * Pm * nz, -cbx * Pm, -cby * Pm, -cbz * Pm);
            this.pj = false;
            continue;
          }
          if (invM[i] > 0 && !asleep[i]) {
            vel[xi] += invM[i] * Pm * nx; vel[xi + 1] += invM[i] * Pm * ny; vel[xi + 2] += invM[i] * Pm * nz;
            addIinv(Iinv, 9 * i, om, xi, cax * Pm, cay * Pm, caz * Pm, 1);
          }
          if (invM[j] > 0 && !asleep[j]) {
            vel[xj] -= invM[j] * Pm * nx; vel[xj + 1] -= invM[j] * Pm * ny; vel[xj + 2] -= invM[j] * Pm * nz;
            addIinv(Iinv, 9 * j, om, xj, cbx * Pm, cby * Pm, cbz * Pm, -1);
          }
        }
        if (this.gpuForm && this.dampForm !== 'gs') { this.dampApply(); dCol = -1; }
      }
      if (this.stickV > 0) this.stickApply();
      if (this.f32State) {
        const fr = a => { for (let k = 0; k < a.length; k++) a[k] = Math.fround(a[k]); };
        fr(pos); fr(quat); fr(vel); fr(om);
      }
    }
    stats[2] += evals;
  }

  // one normal contact correction dl = C / (w + at) along n at point c (A pushed along -n, B along +n); returns dl
  // exactSupport (prototype, experiments/physics_proto/exact_support.mjs): the covered render Gaussians and per body
  // the face reference (the median offset of its outside flat full-detail leaves along their outward normals, so faces
  // rest where they did); NaN reference: no Gaussians (static parts keep their node offsets)
  setExactSupport(xs) {
    const B = this.B, SL = this.SL, SLS = this.SLS, N = this.N;
    xs.ref = new Float64Array(B).fill(NaN);
    this.xs = xs;
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (let b = 0; b < B; b++) {
      const p = this.meta.parts[b], gs = xs.byAsset.get(p.asset_index);
      if (!gs) continue;
      xs.ref[b] = 0;
      const vals = [], [lo, hi] = p.nodes;
      let top = null;
      for (let k = lo; k < hi; k++) {
        const sl = SLS * k, sg = Math.sign(SL[sl + 5]);
        if (SL[sl + 4] !== 0 || sg === 0 || !(SL[sl + 3] <= 0.25 * N[k * NS + 11]) || N[k * NS + 13] >= 0) continue;
        const ux = sg * SL[sl], uy = sg * SL[sl + 1], uz = sg * SL[sl + 2];
        // on the part's outside along u: the leaf's max reach there within 5 mm of the part's
        let mk = -Infinity, mp = -Infinity;
        for (let t = xs.start[k], e = t + xs.count[k]; t < e; t++) { const gi = gs.perm[t]; if (gi < gs.n) mk = Math.max(mk, ux * gs.M[3 * gi] + uy * gs.M[3 * gi + 1] + uz * gs.M[3 * gi + 2]); }
        for (let gi = 0; gi < gs.n; gi++) mp = Math.max(mp, ux * gs.M[3 * gi] + uy * gs.M[3 * gi + 1] + uz * gs.M[3 * gi + 2]);
        if (mk < mp - 0.005) continue;
        vals.push(this.exactOffset(k, b, I, 0, ux, uy, uz));
        if (vals.length > 400) break;   // a sample is enough for the median
      }
      void top;
      vals.sort((x, y) => x - y);
      xs.ref[b] = vals.length ? vals[vals.length >> 1] : 0;
    }
  }

  // contactRefine: side (0: a, 1: b) of a pair, node k of body b (rotation rows at Rs[ro], scale s): its covered render
  // Gaussians in world space (means, covariances, masses) in this.subA / subB; returns their count, 0 if the node covers
  // none (a static part: the node itself is used)
  subItems(side, k, b, ro, s) {
    const xs = this.xs, gs = xs.byAsset.get(this.meta.parts[b].asset_index);
    if (!gs) return 0;
    const n0 = xs.count[k];
    let S = side === 0 ? this.subA : this.subB;
    if (!S || S.m.length < n0) { S = { x: new Float64Array(3 * n0 + 48), c: new Float64Array(6 * n0 + 96), m: new Float64Array(n0 + 16) }; if (side === 0) this.subA = S; else this.subB = S; }
    const R = this.Rs, pos = this.pos, s2 = s * s, s3 = s2 * s;
    let n = 0;
    for (let t = xs.start[k], e = t + n0; t < e; t++) {
      const gi = gs.perm[t];
      if (gi >= gs.n) continue;   // core fill, not render
      const M = gs.M, C = gs.C, o = 6 * gi, mx = M[3 * gi] * s, my = M[3 * gi + 1] * s, mz = M[3 * gi + 2] * s;
      S.x[3 * n] = R[ro] * mx + R[ro + 1] * my + R[ro + 2] * mz + pos[3 * b];
      S.x[3 * n + 1] = R[ro + 3] * mx + R[ro + 4] * my + R[ro + 5] * mz + pos[3 * b + 1];
      S.x[3 * n + 2] = R[ro + 6] * mx + R[ro + 7] * my + R[ro + 8] * mz + pos[3 * b + 2];
      // R C R^T s^2
      const c = [C[o], C[o + 1], C[o + 2], C[o + 1], C[o + 3], C[o + 4], C[o + 2], C[o + 4], C[o + 5]];
      const T = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (let a = 0; a < 3; a++) for (let q = 0; q < 3; q++) T[3 * a + q] = R[ro + 3 * a] * c[q] + R[ro + 3 * a + 1] * c[3 + q] + R[ro + 3 * a + 2] * c[6 + q];
      const W = (a, q) => (T[3 * a] * R[ro + 3 * q] + T[3 * a + 1] * R[ro + 3 * q + 1] + T[3 * a + 2] * R[ro + 3 * q + 2]) * s2;
      S.c[6 * n] = W(0, 0); S.c[6 * n + 1] = W(0, 1); S.c[6 * n + 2] = W(0, 2); S.c[6 * n + 3] = W(1, 1); S.c[6 * n + 4] = W(1, 2); S.c[6 * n + 5] = W(2, 2);
      S.m[n] = gs.W[gi] * s3;
      n++;
    }
    return n;
  }

  // exactSupport: node k (of body b) along world direction (x, y, z), body units: the max over its covered render
  // Gaussians of u.(mu - c) + kappa sqrt(u^T S u) (each Gaussian's own surface) minus kappa sqrt(u^T C u) (the node's)
  exactOffset(k, b, R, ro, x, y, z) {
    const ux = R[ro] * x + R[ro + 3] * y + R[ro + 6] * z, uy = R[ro + 1] * x + R[ro + 4] * y + R[ro + 7] * z, uz = R[ro + 2] * x + R[ro + 5] * y + R[ro + 8] * z;
    const xs = this.xs, gs = xs.byAsset.get(this.meta.parts[b].asset_index), N = this.N, n = k * NS, kap = this.kappa;
    const cx = N[n], cy = N[n + 1], cz = N[n + 2];
    let h = -Infinity;
    for (let t = xs.start[k], e = t + xs.count[k]; t < e; t++) {
      const gi = gs.perm[t];
      if (gi >= gs.n) continue;   // core fill, not render
      const M = gs.M, C = gs.C, o = 6 * gi;
      const s2 = ux * ux * C[o] + uy * uy * C[o + 3] + uz * uz * C[o + 5] + 2 * (ux * uy * C[o + 1] + ux * uz * C[o + 2] + uy * uz * C[o + 4]);
      const v = ux * (M[3 * gi] - cx) + uy * (M[3 * gi + 1] - cy) + uz * (M[3 * gi + 2] - cz) + kap * Math.sqrt(Math.max(s2, 0));
      if (v > h) h = v;
    }
    if (h === -Infinity) return 0;
    const s2 = ux * ux * N[n + 3] + uy * uy * N[n + 6] + uz * uz * N[n + 8] + 2 * (ux * uy * N[n + 4] + ux * uz * N[n + 5] + uy * uz * N[n + 7]);
    return h - kap * Math.sqrt(Math.max(s2, 0));
  }

  // directionalExtent: node k's contact offset along world direction (x, y, z) (unit, outward from the node), in body
  // units: its support there (the bake's table read by an angular blend, weights max(0, n . d)^8) minus kappa times its
  // spread there (the overlap model's own surface)
  dirOffset(k, R, ro, x, y, z) {
    const ux = R[ro] * x + R[ro + 3] * y + R[ro + 6] * z, uy = R[ro + 1] * x + R[ro + 4] * y + R[ro + 7] * z, uz = R[ro + 2] * x + R[ro + 5] * y + R[ro + 8] * z;
    const D = this.extDirs, E = this.EXT, nd = D.length / 3, o = k * nd;
    let sw = 0, sh = 0;
    for (let d = 0; d < nd; d++) {
      let c = ux * D[3 * d] + uy * D[3 * d + 1] + uz * D[3 * d + 2];
      if (c <= 0) continue;
      c *= c; c *= c; c *= c;
      sw += c; sh += c * E[o + d];
    }
    const N = this.N, n = k * NS;
    const s2 = ux * ux * N[n + 3] + uy * uy * N[n + 6] + uz * uz * N[n + 8] + 2 * (ux * uy * N[n + 4] + ux * uz * N[n + 5] + uy * uz * N[n + 7]);
    return (sw > 0 ? sh / sw : 0) - this.kappa * Math.sqrt(Math.max(s2, 0));
  }

  contact(i, j, nx, ny, nz, cx, cy, cz, Cneg, at, penShare = 0) {
    const pos = this.pos, invM = this.invS;
    const rax = cx - pos[3 * i], ray = cy - pos[3 * i + 1], raz = cz - pos[3 * i + 2];
    const rbx = cx - pos[3 * j], rby = cy - pos[3 * j + 1], rbz = cz - pos[3 * j + 2];
    // ta = ra x (-n), tb = rb x n
    const tax = -(ray * nz - raz * ny), tay = -(raz * nx - rax * nz), taz = -(rax * ny - ray * nx);
    const tbx = rby * nz - rbz * ny, tby = rbz * nx - rbx * nz, tbz = rbx * ny - rby * nx;
    const w = invM[i] + this.wI(i, tax, tay, taz) + invM[j] + this.wI(j, tbx, tby, tbz);
    const dl = Cneg / (w + at);
    this.applyCorr(i, -nx, -ny, -nz, tax, tay, taz, dl);
    this.applyCorr(j, nx, ny, nz, tbx, tby, tbz, dl);
    if (penShare > 0) {   // record the depenetration share of the move (split depenetration)
      this.recordPen(i, -nx, -ny, -nz, tax, tay, taz, dl * penShare);
      this.recordPen(j, nx, ny, nz, tbx, tby, tbz, dl * penShare);
      this.popStats.depenMoves++;
    }
    return dl;
  }

  // rolling resistance at a patch (rolling): the bodies' relative rotation this substep (from prevQ) about axes in the
  // contact plane is taken out, up to mu_r |dl| r of angular correction (torque bound mu_r N r, positional form)
  rollingPatch(i, j, nx, ny, nz, cx, cy, cz, ln) {
    const muR = Math.sqrt(this.muR[i] * this.muR[j]);
    if (!(muR > 0)) return;
    const invS = this.invS, q = this.quat, p = this.prevQ, rot = (b, out) => {   // 2 vec(q conj(prevQ)), the short way
      const o = 4 * b, w1 = q[o], x1 = q[o + 1], y1 = q[o + 2], z1 = q[o + 3], w2 = p[o], x2 = -p[o + 1], y2 = -p[o + 2], z2 = -p[o + 3];
      const w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, s = w < 0 ? -2 : 2;
      out[0] = s * (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2); out[1] = s * (w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2); out[2] = s * (w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);
    };
    const ri = [0, 0, 0], rj = [0, 0, 0];
    if (invS[i] > 0) rot(i, ri);
    if (invS[j] > 0) rot(j, rj);
    let dx = ri[0] - rj[0], dy = ri[1] - rj[1], dz = ri[2] - rj[2];
    const dn = dx * nx + dy * ny + dz * nz;
    dx -= dn * nx; dy -= dn * ny; dz -= dn * nz;
    const mag = Math.hypot(dx, dy, dz);
    if (mag < 1e-12) return;
    const ux = dx / mag, uy = dy / mag, uz = dz / mag, pos = this.pos;
    let r = Infinity;
    for (const b of [i, j]) if (invS[b] > 0) r = Math.min(r, Math.hypot(cx - pos[3 * b], cy - pos[3 * b + 1], cz - pos[3 * b + 2]));
    const k = (invS[i] > 0 ? this.wI(i, ux, uy, uz) : 0) + (invS[j] > 0 ? this.wI(j, ux, uy, uz) : 0);
    if (!(k > 0) || !Number.isFinite(r)) return;
    const P = Math.min(mag / k, muR * Math.abs(ln) * r), rc = this.pj ? this.rollPairs : null;
    this.applyCorr(i, 0, 0, 0, -ux, -uy, -uz, rc ? P / Math.max(1, rc[i]) : P);
    this.applyCorr(j, 0, 0, 0, ux, uy, uz, rc ? P / Math.max(1, rc[j]) : P);
  }

  // the displacement and rotation vector applyCorr(b, lin, ang, dl) gives body b, added to its depenetration record
  recordPen(b, lx, ly, lz, gx, gy, gz, dl) {
    const im = this.invS[b];
    if (im === 0) return;
    const P = this.penP, Wp = this.penW, I = this.Iinv, o = 9 * b, x = 3 * b;
    P[x] += im * lx * dl; P[x + 1] += im * ly * dl; P[x + 2] += im * lz * dl;
    Wp[x] += (I[o] * gx + I[o + 1] * gy + I[o + 2] * gz) * dl;
    Wp[x + 1] += (I[o + 3] * gx + I[o + 4] * gy + I[o + 5] * gz) * dl;
    Wp[x + 2] += (I[o + 6] * gx + I[o + 7] * gy + I[o + 8] * gz) * dl;
  }

  // mechanical energy of body b: translational + rotational kinetic energy + gravitational potential (z up)
  energy(b) {
    const x = 3 * b, m = 1 / this.invM[b], vel = this.vel, om = this.om;
    quatMat(this.quat, 4 * b, T9, 0);
    let rot = 0;
    for (let k = 0; k < 3; k++) { const w = T9[k] * om[x] + T9[3 + k] * om[x + 1] + T9[6 + k] * om[x + 2]; rot += this.Ib[x + k] * w * w; }
    const ke = 0.5 * m * (vel[x] * vel[x] + vel[x + 1] * vel[x + 1] + vel[x + 2] * vel[x + 2]) + 0.5 * rot;
    return [ke, ke - m * (this.grav[0] * this.pos[x] + this.grav[1] * this.pos[x + 1] + this.grav[2] * this.pos[x + 2])];
  }

  // per body before the solve: in contact this step (any node pair or compact pair), in an impact (a pair closing
  // faster than 0.3 m/s), and its mechanical energy if its kinetic energy is low (else Infinity: not limited)
  markEnergy() {
    const { B, invM, vel, om, pos, op, pba, pbb, E0, inContact, impactB } = this;
    inContact.fill(0); impactB.fill(0);
    for (let t = 0; t < this.npairs; t++) { const bp = op[t]; inContact[pba[bp]] = 1; inContact[pbb[bp]] = 1; }
    for (let pid = 0; pid < B * B; pid++) if (this.compactF[pid]) { inContact[(pid / B) | 0] = 1; inContact[pid % B] = 1; }
    for (let bp = 0; bp < this.nbp; bp++) {
      const i = pba[bp], j = pbb[bp];
      const dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1], dz = pos[3 * j + 2] - pos[3 * i + 2], d = Math.hypot(dx, dy, dz) || 1;
      const closing = ((vel[3 * i] - vel[3 * j]) * dx + (vel[3 * i + 1] - vel[3 * j + 1]) * dy + (vel[3 * i + 2] - vel[3 * j + 2]) * dz) / d;
      if (closing > 0.3) { impactB[i] = 1; impactB[j] = 1; }
    }
    for (let b = 0; b < B; b++) {
      if (invM[b] === 0) continue;
      const [ke, e] = this.energy(b);
      E0[b] = this.energyPassive ? e : ke < 0.5 * ENERGY_V * ENERGY_V / invM[b] ? e : Infinity;
    }
    void om;
  }

  // energyPassive: the gate-free form. The solver as a whole may not add mechanical energy in a step: over the awake
  // bodies in contact, any excess of their summed energy over the step start is removed from their kinetic energy by
  // one common factor (energy moves between bodies in an impact; only the total is bounded). No speed or impact gate
  limitEnergyPassive() {
    const { B, invM, asleep, vel, om, E0, inContact } = this;
    let e1 = 0, e0 = 0, ke = 0;
    const set = [];
    for (let b = 0; b < B; b++) {
      if (invM[b] === 0 || asleep[b] || !inContact[b]) continue;
      const [k, e] = this.energy(b);
      e1 += e; e0 += E0[b]; ke += k; set.push(b);
    }
    const excess = e1 - e0 - ENERGY_EPS * set.length;
    if (excess <= 0 || ke <= 0) return;
    const f = Math.sqrt(Math.max(0, ke - excess) / ke);
    for (const b of set) for (let k = 0; k < 3; k++) { vel[3 * b + k] *= f; om[3 * b + k] *= f; }
    this.popStats.limited++; this.popStats.limitedEnergy += excess;
  }

  // energy limiter: a body in contact (not in an impact) that started the step with low kinetic energy may not end it
  // with more mechanical energy (kinetic + potential) than it started with: the solver cannot create energy. Pushes
  // and other bodies' impulses act between steps or through impacts, so they are not limited.
  limitEnergy() {
    const { B, invM, asleep, vel, om, E0, inContact, impactB } = this;
    for (let b = 0; b < B; b++) {
      if (invM[b] === 0 || asleep[b] || !inContact[b] || impactB[b] || E0[b] === Infinity) continue;
      const [ke, e] = this.energy(b);
      const excess = e - E0[b] - ENERGY_EPS;
      if (excess <= 0 || ke <= 0) continue;
      const f = Math.sqrt(Math.max(0, ke - excess) / ke);
      for (let k = 0; k < 3; k++) { vel[3 * b + k] *= f; om[3 * b + k] *= f; }
      this.popStats.limited++; this.popStats.limitedEnergy += excess;
    }
  }

  // a body point's move over the substep: (p + R l) - (p_prev + R_prev l), l = R^T r; result in T3 (x also returned)
  slip(R, Rp, ro, x, rx, ry, rz) {
    const pos = this.pos, prevP = this.prevP;
    const lx = R[ro] * rx + R[ro + 3] * ry + R[ro + 6] * rz;
    const ly = R[ro + 1] * rx + R[ro + 4] * ry + R[ro + 7] * rz;
    const lz = R[ro + 2] * rx + R[ro + 5] * ry + R[ro + 8] * rz;
    for (let r = 0; r < 3; r++) {
      T3[r] = (pos[x + r] + R[ro + 3 * r] * lx + R[ro + 3 * r + 1] * ly + R[ro + 3 * r + 2] * lz)
        - (prevP[x + r] + Rp[ro + 3 * r] * lx + Rp[ro + 3 * r + 1] * ly + Rp[ro + 3 * r + 2] * lz);
    }
    return T3[0];
  }

  sleep() {
    const { B, invM, asleep, vel, om, still, sleepL } = this, island = this.islandSleep;
    for (let b = 0; b < B; b++) {
      if (invM[b] === 0 || asleep[b]) continue;
      let sp;
      if (this.sleepNet) {   // net motion over the step: |dp| / dt + angle / dt x sleepL (angle ~ 2 |q - q0|, the closer sign)
        const p0 = this.pos0, q0 = this.quat0, q = this.quat;
        let dm = 0, dp = 0;
        for (let k = 0; k < 4; k++) { dm += (q[4 * b + k] - q0[4 * b + k]) ** 2; dp += (q[4 * b + k] + q0[4 * b + k]) ** 2; }
        sp = (Math.hypot(this.pos[3 * b] - p0[3 * b], this.pos[3 * b + 1] - p0[3 * b + 1], this.pos[3 * b + 2] - p0[3 * b + 2]) + 2 * Math.sqrt(Math.min(dm, dp)) * sleepL[b]) / this.dt;
      } else sp = Math.hypot(vel[3 * b], vel[3 * b + 1], vel[3 * b + 2]) + Math.hypot(om[3 * b], om[3 * b + 1], om[3 * b + 2]) * sleepL[b];
      this.spB[b] = sp;
      let moved = false;
      if (this.sleepDisp > 0) {   // net displacement since the still window began (its start: when still was last 0)
        const w0 = this.win0 ??= Float64Array.from(this.pos);
        if (still[b] === 0) w0.set(this.pos.subarray(3 * b, 3 * b + 3), 3 * b);
        moved = Math.hypot(this.pos[3 * b] - w0[3 * b], this.pos[3 * b + 1] - w0[3 * b + 1], this.pos[3 * b + 2] - w0[3 * b + 2]) > this.sleepDisp;
      }
      still[b] = sp < this.sleepV && !moved && !(this.sleepDepth > 0 && this.depthB[b] > this.sleepDepth) && !(this.sleepRecover > 0 && this.recovering(b)) ? still[b] + this.dt : 0;
      if (moved && this.sleepDisp > 0) this.win0.set(this.pos.subarray(3 * b, 3 * b + 3), 3 * b);   // a new window from here
      if (!island && still[b] > this.sleepT) {
        asleep[b] = 1;
        vel.fill(0, 3 * b, 3 * b + 3); om.fill(0, 3 * b, 3 * b + 3);
      }
    }
    if (!island) return;
    // island sleeping (world.py _sleep): dynamic bodies in contact (a body pair with node pairs this step) form an
    // island; it sleeps when every awake member has been still for sleepT, and an awake member wakes its sleeping ones
    const par = this.islandPar, cnt = this.islandCnt, nbp = this.nbp;
    for (let b = 0; b < B; b++) par[b] = b;
    const find = a => { while (par[a] !== a) { par[a] = par[par[a]]; a = par[a]; } return a; };
    cnt.fill(0, 0, Math.max(nbp, 1));
    for (let t = 0; t < this.npairs; t++) cnt[this.op[t]]++;
    for (let bp = 0; bp < nbp; bp++) {
      const i = this.pba[bp], j = this.pbb[bp];
      if ((cnt[bp] > 0 || this.compactF[i * B + j]) && invM[i] > 0 && invM[j] > 0) {
        const ri = find(i), rj = find(j);
        if (ri !== rj) par[Math.max(ri, rj)] = Math.min(ri, rj);
      }
    }
    if (this.islandAvg) {
      // island-average speed: an island's awake members share one stillness counter (the island's), kept in each member
      for (let r = 0; r < B; r++) {
        if (invM[r] === 0 || find(r) !== r) continue;
        let n = 0, sum = 0, st = Infinity;
        for (let b = 0; b < B; b++) if (invM[b] > 0 && find(b) === r && !asleep[b]) { n++; sum += this.spB[b]; st = Math.min(st, this.islandStill[b]); }
        if (!n) continue;
        const ist = sum / n < this.sleepV ? st + this.dt : 0;
        for (let b = 0; b < B; b++) if (invM[b] > 0 && find(b) === r && !asleep[b]) { this.islandStill[b] = ist; still[b] = ist; }
      }
    }
    for (let r = 0; r < B; r++) {
      if (invM[r] === 0 || find(r) !== r) continue;
      let anyAwake = false, allStill = true;
      for (let b = 0; b < B; b++) if (invM[b] > 0 && find(b) === r && !asleep[b]) { anyAwake = true; if (!(still[b] > this.sleepT)) allStill = false; }
      if (!anyAwake) continue;
      for (let b = 0; b < B; b++) {
        if (invM[b] === 0 || find(b) !== r) continue;
        if (allStill) { if (!asleep[b]) { asleep[b] = 1; vel.fill(0, 3 * b, 3 * b + 3); om.fill(0, 3 * b, 3 * b + 3); } }
        else if (asleep[b]) { asleep[b] = 0; still[b] = 0; this.islandStill[b] = 0; }
      }
    }
  }

  // positional Coulomb friction at patch p's point (normal n pushes i away from j, point c, the normal correction's
  // |dl|): slip since the substep start (or the anchored points' separation), bounded by mu(v) |dl|; Stribeck,
  // anchors, materials as configured
  frictionPatch(p, i, j, nx, ny, nz, cx, cy, cz, ln, h) {
    const { pos, quat, prevQ, Rs, Rp } = this, invM = this.invS, xi = 3 * i, xj = 3 * j;
      const rax = cx - pos[xi], ray = cy - pos[xi + 1], raz = cz - pos[xi + 2];
      const rbx = cx - pos[xj], rby = cy - pos[xj + 1], rbz = cz - pos[xj + 2];
      quatMat(quat, 4 * i, Rs, 0); quatMat(quat, 4 * j, Rs, 9);
      quatMat(prevQ, 4 * i, Rp, 0); quatMat(prevQ, 4 * j, Rp, 9);
      const dax = this.slip(Rs, Rp, 0, xi, rax, ray, raz), day = T3[1], daz = T3[2];
      const dbx = this.slip(Rs, Rp, 9, xj, rbx, rby, rbz), dby = T3[1], dbz = T3[2];
      let dpx = dax - dbx, dpy = day - dby, dpz = daz - dbz;
      if (this.fricAfter && this.pj) {
        // the patch's own normal correction, not yet applied under pjacobi: point moves dp + dtheta x r per body
        const dl = this.lastDl;
        const tax = -(ray * nz - raz * ny), tay = -(raz * nx - rax * nz), taz = -(rax * ny - ray * nx);
        const tbx = rby * nz - rbz * ny, tby = rbz * nx - rbx * nz, tbz = rbx * ny - rby * nx;
        const I = this.Iinv, oi = 9 * i, oj = 9 * j;
        const gix = (I[oi] * tax + I[oi + 1] * tay + I[oi + 2] * taz) * dl, giy = (I[oi + 3] * tax + I[oi + 4] * tay + I[oi + 5] * taz) * dl, giz = (I[oi + 6] * tax + I[oi + 7] * tay + I[oi + 8] * taz) * dl;
        const gjx = (I[oj] * tbx + I[oj + 1] * tby + I[oj + 2] * tbz) * dl, gjy = (I[oj + 3] * tbx + I[oj + 4] * tby + I[oj + 5] * tbz) * dl, gjz = (I[oj + 6] * tbx + I[oj + 7] * tby + I[oj + 8] * tbz) * dl;
        dpx += -invM[i] * nx * dl + (giy * raz - giz * ray) - (invM[j] * nx * dl + (gjy * rbz - gjz * rby));
        dpy += -invM[i] * ny * dl + (giz * rax - gix * raz) - (invM[j] * ny * dl + (gjz * rbx - gjx * rbz));
        dpz += -invM[i] * nz * dl + (gix * ray - giy * rax) - (invM[j] * nz * dl + (gjx * rby - gjy * rbx));
      }
      let dn = dpx * nx + dpy * ny + dpz * nz;
      let tx = dpx - dn * nx, ty = dpy - dn * ny, tz = dpz - dn * nz;
      const vslip = Math.sqrt(tx * tx + ty * ty + tz * tz) / h;   // this substep's slip speed (Stribeck)
      let anc = null;
      if (this.frictionAnchors) {
        // the anchored points' separation replaces this substep's slip once the patch has an anchor
        const key = (i * this.B + j) * 1073741824 + this.pKey[this.pIdx[p]] % 1073741824;
        anc = this.anchors.get(key);
        // a contact that broke releases its anchor: valid only through consecutive substeps of contact (a stale
        // anchor pulled a re-touching contact toward where it was substeps ago, up to 500x the slip and reversed)
        if (anc && !anc.fresh && anc.sub < this.sub - 1) { anc.fresh = true; }
        if (anc && !anc.fresh) {
          const ax = anc.la[0], ay = anc.la[1], az = anc.la[2], bx = anc.lb[0], by = anc.lb[1], bz = anc.lb[2];
          const pax = pos[xi] + Rs[0] * ax + Rs[1] * ay + Rs[2] * az, pay = pos[xi + 1] + Rs[3] * ax + Rs[4] * ay + Rs[5] * az, paz = pos[xi + 2] + Rs[6] * ax + Rs[7] * ay + Rs[8] * az;
          const pbx = pos[xj] + Rs[9] * bx + Rs[10] * by + Rs[11] * bz, pby = pos[xj + 1] + Rs[12] * bx + Rs[13] * by + Rs[14] * bz, pbz = pos[xj + 2] + Rs[15] * bx + Rs[16] * by + Rs[17] * bz;
          if (this.anchorStats) {   // inspection: the anchor gap against the slip it replaces (tangential parts)
            const gx = pax - pbx, gy = pay - pby, gz = paz - pbz, gn = gx * nx + gy * ny + gz * nz;
            const ax_ = gx - gn * nx, ay_ = gy - gn * ny, az_ = gz - gn * nz, la_ = Math.hypot(ax_, ay_, az_), ls_ = Math.hypot(tx, ty, tz);
            const st = this.anchorStats; st.cmp ??= [];
            if (la_ > 0 && ls_ > 0) st.cmp.push([la_ / ls_, Math.acos(Math.max(-1, Math.min(1, (ax_ * tx + ay_ * ty + az_ * tz) / (la_ * ls_)))) * 180 / Math.PI]);
          }
          dpx = pax - pbx; dpy = pay - pby; dpz = paz - pbz;
          dn = dpx * nx + dpy * ny + dpz * nz;
          tx = dpx - dn * nx; ty = dpy - dn * ny; tz = dpz - dn * nz;
        } else if (!anc) {
          anc = { la: [0, 0, 0], lb: [0, 0, 0], frame: 0, fresh: true, i, j, p: 0, reanchor: false, sub: 0 };
          this.anchors.set(key, anc);
        }
        anc.frame = this.frame; anc.p = p; anc.i = i; anc.j = j; anc.sub = this.sub;
      }
      const m = Math.sqrt(tx * tx + ty * ty + tz * tz);
      let slid = !anc || anc.fresh;
      if (m > 1e-12) {
        const ux = tx / m, uy = ty / m, uz = tz / m;
        const cax = ray * uz - raz * uy, cay = raz * ux - rax * uz, caz = rax * uy - ray * ux;
        const cbx = rby * uz - rbz * uy, cby = rbz * ux - rbx * uz, cbz = rbx * uy - rby * ux;
        const wt = invM[i] + this.wI(i, cax, cay, caz) + invM[j] + this.wI(j, cbx, cby, cbz);
        const stat = this.muAtP(p, 0) * ln, need = m / wt;
        // anchored and within static friction: hold; else slide at mu(v) (and re-anchor below)
        const dlt = anc && !anc.fresh && need <= stat ? -need : -Math.min(need, this.muAtP(p, vslip) * ln);
        if (anc) { const st = this.anchorStats ??= { held: 0, slid: 0, fresh: 0 }; if (anc.fresh) st.fresh++; else if (need <= stat) st.held++; else st.slid++; }   // inspection
        if (anc && !anc.fresh && need > stat) slid = true;
        this.applyCorr(i, ux, uy, uz, cax, cay, caz, dlt);
        this.applyCorr(j, -ux, -uy, -uz, -cbx, -cby, -cbz, dlt);
      }
      // (re-)anchor at the substep's end (anchorUpdate), from the final poses: re-anchoring here, at the poses
      // before this sweep's corrections, let the next sweep pull back this sweep's own sliding move (a jolt)
      if (anc && slid) anc.reanchor = true;
  }

  // frictionAnchors, substep end: new and slid contacts anchor at their patch point in both bodies' frames (l = R^T r)
  anchorUpdate() {
    const { pos, quat, pxc } = this;
    for (const a of this.anchors.values()) {
      if (!a.reanchor || a.frame !== this.frame) continue;
      const c = [pxc[3 * a.p], pxc[3 * a.p + 1], pxc[3 * a.p + 2]];
      for (const [b, out] of [[a.i, 'la'], [a.j, 'lb']]) {
        quatMat(quat, 4 * b, T9, 0);
        const rx = c[0] - pos[3 * b], ry = c[1] - pos[3 * b + 1], rz = c[2] - pos[3 * b + 2];
        a[out] = [T9[0] * rx + T9[3] * ry + T9[6] * rz, T9[1] * rx + T9[4] * ry + T9[7] * rz, T9[2] * rx + T9[5] * ry + T9[8] * rz];
      }
      a.reanchor = false; a.fresh = false;
    }
  }

  // patch p's friction coefficient at slip speed v: its combined material (materials) with the Stribeck blend, else muAt
  muAtP(p, v) {
    if (!this.materials || !this.patchMat || this.SLS <= 10) return this.muAt(v);
    const pm = this.patchMat, r = v / Math.max(pm[4 * p + 2], 1e-6);
    return pm[4 * p + 1] + (pm[4 * p] - pm[4 * p + 1]) * Math.exp(-r * r);
  }

  // friction coefficient at a tangential slip speed v (Stribeck when muS and muK are set, else mu)
  muAt(v) {
    if (this.muS === undefined || this.muK === undefined) return this.mu;
    const r = v / this.stribeckV;
    return this.muK + (this.muS - this.muK) * Math.exp(-r * r);
  }

  // sleepRecover: did body b move out of its contacts this step by more than sleepRecover? Outward normal: the patches'
  // normals (n pushes j out of i) oriented away from the partner, weighted by the patch's largest correction (pl)
  recovering(b) {
    const { npatch, op, patchOf, starts, pba, pbb, pn, pl, pos, pos0 } = this;
    let nx = 0, ny = 0, nz = 0;
    for (let p = 0; p < npatch; p++) {
      if (pl[p] <= 0) continue;
      const bp = op[patchOf[starts[p]]], sg = pba[bp] === b ? -1 : pbb[bp] === b ? 1 : 0;
      if (!sg) continue;
      nx += sg * pl[p] * pn[3 * p]; ny += sg * pl[p] * pn[3 * p + 1]; nz += sg * pl[p] * pn[3 * p + 2];
    }
    const l = Math.hypot(nx, ny, nz);
    if (l === 0) return false;
    const out = ((pos[3 * b] - pos0[3 * b]) * nx + (pos[3 * b + 1] - pos0[3 * b + 1]) * ny + (pos[3 * b + 2] - pos0[3 * b + 2]) * nz) / l;
    return out > this.sleepRecover;
  }

  // Iinv of body b in world space (current pose), into out[o..o+9)
  inverseInertia(b, out, o = 0) {
    quatMat(this.quat, 4 * b, T9, 0);
    const Ib = this.Ib, x = 3 * b;
    const d = [Ib[x] ? 1 / Ib[x] : 0, Ib[x + 1] ? 1 / Ib[x + 1] : 0, Ib[x + 2] ? 1 / Ib[x + 2] : 0];
    for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++)
      out[o + 3 * a + c] = T9[3 * a] * d[0] * T9[3 * c] + T9[3 * a + 1] * d[1] * T9[3 * c + 1] + T9[3 * a + 2] * d[2] * T9[3 * c + 2];
    return out;
  }

  // a velocity change of `speed` (m/s) at world point c along unit d on body b (an impulse); wakes it
  push(b, c, d, speed) {
    if (this.invM[b] === 0) return;
    const I = this.inverseInertia(b, new Float64Array(9));
    const x = 3 * b, pos = this.pos;
    const rx = c[0] - pos[x], ry = c[1] - pos[x + 1], rz = c[2] - pos[x + 2];
    const ux = ry * d[2] - rz * d[1], uy = rz * d[0] - rx * d[2], uz = rx * d[1] - ry * d[0];   // r x d
    const Iu = [I[0] * ux + I[1] * uy + I[2] * uz, I[3] * ux + I[4] * uy + I[5] * uz, I[6] * ux + I[7] * uy + I[8] * uz];
    const J = speed / (this.invM[b] + ux * Iu[0] + uy * Iu[1] + uz * Iu[2]);
    for (let k = 0; k < 3; k++) { this.vel[x + k] += this.invM[b] * J * d[k]; this.om[x + k] += J * Iu[k]; }
    this.wake(b);
  }

  // nearest hit of a world ray (o, unit d) with a dynamic body's face-aligned box (its Gaussian centres + 1 cm):
  // { body, t, point } or null
  pick(o, d) {
    let best = null;
    this.meta.parts.forEach((p, b) => {
      if (this.invM[b] === 0) return;
      quatMat(this.quat, 4 * b, T9, 0);
      const x = 3 * b;
      // body frame, then the box frame (rows of box_axes, centred on box_center)
      const bo = [0, 1, 2].map(k => T9[k] * (o[0] - this.pos[x]) + T9[3 + k] * (o[1] - this.pos[x + 1]) + T9[6 + k] * (o[2] - this.pos[x + 2]) - p.box_center[k]);
      const bd = [0, 1, 2].map(k => T9[k] * d[0] + T9[3 + k] * d[1] + T9[6 + k] * d[2]);
      const A = p.box_axes;
      const ro = A.map(a => a[0] * bo[0] + a[1] * bo[1] + a[2] * bo[2]), rd = A.map(a => a[0] * bd[0] + a[1] * bd[1] + a[2] * bd[2]);
      let t0 = 0, t1 = Infinity;
      for (let k = 0; k < 3; k++) {
        const lo = -p.box_half[k] - 0.01, hi = p.box_half[k] + 0.01;
        if (Math.abs(rd[k]) < 1e-12) { if (ro[k] < lo || ro[k] > hi) return; continue; }
        let a = (lo - ro[k]) / rd[k], c = (hi - ro[k]) / rd[k];
        if (a > c) [a, c] = [c, a];
        t0 = Math.max(t0, a); t1 = Math.min(t1, c);
      }
      if (t0 <= t1 && (!best || t0 < best.t)) best = { body: b, t: t0, point: [o[0] + t0 * d[0], o[1] + t0 * d[1], o[2] + t0 * d[2]] };
    });
    return best;
  }
}

const T9 = new Float64Array(9);
const RR = new Float64Array(12);
const X3 = new Float64Array(3);

// world position of node at N[n..] on a body (R at R[ro..], p at P[po..]) into X3; returns X3
function xfm(R, ro, P, po, N, n, sc = 1) {
  const vx = N[n] * sc, vy = N[n + 1] * sc, vz = N[n + 2] * sc;
  X3[0] = R[ro] * vx + R[ro + 1] * vy + R[ro + 2] * vz + P[po];
  X3[1] = R[ro + 3] * vx + R[ro + 4] * vy + R[ro + 5] * vz + P[po + 1];
  X3[2] = R[ro + 6] * vx + R[ro + 7] * vy + R[ro + 8] * vz + P[po + 2];
  return X3;
}

// a[0..6) *= f (a covariance scaled with its body)
function scale6(a, f) { for (let c = 0; c < 6; c++) a[c] *= f; }

// om[x..] += sign * Iinv u
function addIinv(I, o, om, x, ux, uy, uz, sign) {
  om[x] += sign * (I[o] * ux + I[o + 1] * uy + I[o + 2] * uz);
  om[x + 1] += sign * (I[o + 3] * ux + I[o + 4] * uy + I[o + 5] * uz);
  om[x + 2] += sign * (I[o + 6] * ux + I[o + 7] * uy + I[o + 8] * uz);
}

const now = () => (globalThis.performance ?? Date).now();
