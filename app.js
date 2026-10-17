import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ───────────────────────── Config ─────────────────────────
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_FULL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const MODEL_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const REF_FPS = 10;          // reference poses sampled per second of video
const VIS = 0.5;             // minimum landmark visibility to trust it
const MAX_CANVAS_H = 720;

// Segments that are scored (indices into the 33 MediaPipe landmarks + 33/34 midpoints).
const SEGMENTS = [
  { id: "torso", a: 34, b: 33, part: "Torso" },
  { id: "lUpperArm", a: 11, b: 13, part: "Left arm" },
  { id: "lForearm", a: 13, b: 15, part: "Left arm" },
  { id: "rUpperArm", a: 12, b: 14, part: "Right arm" },
  { id: "rForearm", a: 14, b: 16, part: "Right arm" },
  { id: "lThigh", a: 23, b: 25, part: "Left leg" },
  { id: "lShin", a: 25, b: 27, part: "Left leg" },
  { id: "rThigh", a: 24, b: 26, part: "Right leg" },
  { id: "rShin", a: 26, b: 28, part: "Right leg" },
];
const FRAME_BONES = [[11, 12], [23, 24], [11, 23], [12, 24]];
const SIDES = [
  { name: "left", sh: 11, el: 13, wr: 15, hip: 23, kn: 25, an: 27 },
  { name: "right", sh: 12, el: 14, wr: 16, hip: 24, kn: 26, an: 28 },
];
const SWAP = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
const COLORS = { good: "#3ddc97", ok: "#ffc93c", bad: "#ff5470", neutral: "#ffffff", ref: "#ff3d81", ghost: "rgba(180,160,255,0.85)" };

// ───────────────────────── DOM ─────────────────────────
const $ = (s) => document.querySelector(s);
const els = {
  modelStatus: $("#modelStatus"), fileInput: $("#fileInput"), dropzone: $("#dropzone"), changeVideoBtn: $("#changeVideoBtn"),
  refVideo: $("#refVideo"), refCanvas: $("#refCanvas"), analyzeOverlay: $("#analyzeOverlay"), analyzeBar: $("#analyzeBar"), analyzeText: $("#analyzeText"),
  scrub: $("#scrub"), timeLabel: $("#timeLabel"), durLabel: $("#durLabel"),
  camVideo: $("#camVideo"), camCanvas: $("#camCanvas"), camPlaceholder: $("#camPlaceholder"), camBtn: $("#camBtn"),
  scoreBadge: $("#scoreBadge"), scoreNum: $("#scoreNum"), cues: $("#cues"), countdown: $("#countdown"), timingPill: $("#timingPill"),
  startBtn: $("#startBtn"), speed: $("#speed"), mirrorToggle: $("#mirrorToggle"), ghostToggle: $("#ghostToggle"),
  refSkelToggle: $("#refSkelToggle"), voiceToggle: $("#voiceToggle"),
  summary: $("#summary"), sumScore: $("#sumScore"), sumGrade: $("#sumGrade"), sumSub: $("#sumSub"), sumChart: $("#sumChart"),
  sumParts: $("#sumParts"), sumMoments: $("#sumMoments"), downloadTake: $("#downloadTake"), closeSummary: $("#closeSummary"), againBtn: $("#againBtn"),
};
const refCtx = els.refCanvas.getContext("2d");
const camCtx = els.camCanvas.getContext("2d");

// ───────────────────────── State ─────────────────────────
const state = {
  refLandmarker: null, liveLandmarker: null,
  refFrames: [], refFramesMirrored: [], refAspect: 16 / 9, refReady: false, analyzing: false,
  camReady: false, camAspect: 4 / 3, lastCamTime: -1, userPose: null,
  running: false, countingDown: false, practice: null,
  scoreEma: null, segEma: {}, offsetEma: 0,
  cueMemory: new Map(), lastSpoken: { text: "", at: 0 },
  session: null, recorder: null, recordedChunks: [], takeUrl: null,
};

// ───────────────────────── Geometry helpers ─────────────────────────
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const mul = (a, k) => ({ x: a.x * k, y: a.y * k });
const len = (v) => Math.hypot(v.x, v.y);
const dist = (a, b) => len(sub(a, b));
const norm = (v) => { const l = len(v) || 1; return { x: v.x / l, y: v.y / l }; };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });
function angleBetween(u, v) {
  const d = (u.x * v.x + u.y * v.y) / ((len(u) * len(v)) || 1);
  return (Math.acos(clamp(d, -1, 1)) * 180) / Math.PI;
}
const jointAngle = (p, a, b, c) => angleBetween(sub(p[a], p[b]), sub(p[c], p[b]));
const visible = (p, ...idx) => idx.every((i) => p[i] && p[i].v >= VIS);
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// Convert MediaPipe normalized landmarks into aspect-corrected coords (x scaled by width/height)
// so that angles are true angles regardless of video shape. Adds shoulder (33) and hip (34) midpoints.
function toPose(landmarks, aspect) {
  const p = landmarks.map((l) => ({ x: l.x * aspect, y: l.y, v: l.visibility ?? 1 }));
  p[33] = mid(p[11], p[12]);
  p[34] = mid(p[23], p[24]);
  return p;
}
function mirrorPose(p) {
  const q = p.map((l) => ({ x: -l.x, y: l.y, v: l.v }));
  for (const [a, b] of SWAP) [q[a], q[b]] = [q[b], q[a]];
  return q;
}
function lerpPose(a, b, t) {
  if (!a) return b; if (!b) return a;
  return a.map((p, i) => ({ x: p.x + (b[i].x - p.x) * t, y: p.y + (b[i].y - p.y) * t, v: Math.min(p.v, b[i].v) }));
}

// ───────────────────────── Model loading ─────────────────────────
async function createLandmarker(modelPath) {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: modelPath, delegate }, runningMode: "VIDEO", numPoses: 1 });
  try { return await PoseLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { return await PoseLandmarker.createFromOptions(fileset, opts("CPU")); }
}

async function initModels() {
  try {
    [state.refLandmarker, state.liveLandmarker] = await Promise.all([createLandmarker(MODEL_FULL), createLandmarker(MODEL_LITE)]);
    els.modelStatus.textContent = "● Pose model ready";
    els.modelStatus.classList.add("ready");
    if (state.pendingFile) analyzeReference(state.pendingFile);
  } catch (err) {
    console.error(err);
    els.modelStatus.textContent = "Model failed to load — check your connection";
  }
}

// ───────────────────────── Reference video analysis ─────────────────────────
function seek(video, t) {
  return new Promise((resolve) => {
    const done = () => { video.removeEventListener("seeked", done); clearTimeout(timer); resolve(); };
    const timer = setTimeout(done, 1500);
    video.addEventListener("seeked", done);
    video.currentTime = t;
  });
}

function sizeCanvas(canvas, w, h) {
  const scale = Math.min(1, MAX_CANVAS_H / h);
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
}

async function analyzeReference(file) {
  if (!state.refLandmarker) { state.pendingFile = file; els.dropzone.querySelector("strong").textContent = "Waiting for pose model…"; return; }
  state.pendingFile = null;
  if (state.analyzing) return;
  stopSession(false);
  const v = els.refVideo;
  state.refReady = false; state.analyzing = true;
  els.startBtn.disabled = true;
  els.dropzone.classList.add("hidden");
  els.analyzeOverlay.classList.remove("hidden");
  els.changeVideoBtn.classList.add("hidden");

  if (v.src) URL.revokeObjectURL(v.src);
  v.src = URL.createObjectURL(file);
  v.muted = true;
  await new Promise((r) => v.addEventListener("loadeddata", r, { once: true }));
  state.refAspect = v.videoWidth / v.videoHeight;
  sizeCanvas(els.refCanvas, v.videoWidth, v.videoHeight);

  const duration = v.duration;
  const n = Math.floor(duration * REF_FPS) + 1;
  const frames = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    await seek(v, Math.min(i / REF_FPS, duration - 0.05));
    state.refTs = Math.max((state.refTs || 0) + 1, performance.now()); // must strictly increase
    const res = state.refLandmarker.detectForVideo(v, state.refTs);
    if (res.landmarks?.[0]) frames[i] = toPose(res.landmarks[0], state.refAspect);
    drawRefFrame(frames[i]);
    const pct = Math.round(((i + 1) / n) * 100);
    els.analyzeBar.style.width = pct + "%";
    els.analyzeText.textContent = `${pct}% · ${frames.filter(Boolean).length} poses captured`;
  }
  // fill gaps (frames where the dancer wasn't detected) with the nearest detected pose
  let last = null;
  for (let i = 0; i < n; i++) { if (frames[i]) last = frames[i]; else frames[i] = last; }
  last = null;
  for (let i = n - 1; i >= 0; i--) { if (frames[i]) last = frames[i]; else frames[i] = last; }

  if (!frames.some(Boolean)) {
    els.analyzeText.textContent = "Couldn't find a dancer in this video. Try one with a clear, full-body view.";
    state.analyzing = false;
    els.changeVideoBtn.classList.remove("hidden");
    return;
  }

  state.refFrames = frames;
  state.refFramesMirrored = frames.map((f) => f && mirrorPose(f));
  await seek(v, 0);
  v.muted = false;
  state.analyzing = false;
  state.refReady = true;
  els.analyzeOverlay.classList.add("hidden");
  els.changeVideoBtn.classList.remove("hidden");
  els.scrub.disabled = false;
  els.durLabel.textContent = fmtTime(duration);
  updateStartBtn();
}

function refPoseAt(t, mirrored = false) {
  const frames = mirrored ? state.refFramesMirrored : state.refFrames;
  if (!frames.length) return null;
  const f = clamp(t * REF_FPS, 0, frames.length - 1);
  const i = Math.floor(f);
  return lerpPose(frames[i], frames[Math.min(i + 1, frames.length - 1)], f - i);
}

// ───────────────────────── Comparison ─────────────────────────
function compare(u, r) {
  let total = 0, count = 0;
  const seg = {};
  for (const s of SEGMENTS) {
    if (!visible(u, s.a, s.b) || !visible(r, s.a, s.b)) continue;
    const diff = angleBetween(sub(u[s.b], u[s.a]), sub(r[s.b], r[s.a]));
    const sc = clamp(1 - (diff - 15) / 60, 0, 1); // ≤15° off = perfect, ≥75° off = zero
    seg[s.id] = sc; total += sc; count++;
  }
  return { score: count >= 4 ? (100 * total) / count : null, seg, count };
}

// The user naturally lags the video a little, so compare against a short window of reference
// frames and keep the best match. The winning offset also tells us if they're behind or rushing.
function bestMatch(u, t) {
  const frames = els.mirrorToggle.checked ? state.refFramesMirrored : state.refFrames;
  const rate = Number(els.speed.value);
  const center = Math.round(t * REF_FPS);
  const from = Math.max(0, center - Math.round(0.6 * rate * REF_FPS));
  const to = Math.min(frames.length - 1, center + Math.round(0.3 * rate * REF_FPS));
  let best = null;
  for (let j = from; j <= to; j++) {
    const r = frames[j];
    if (!r) continue;
    const res = compare(u, r);
    if (res.score == null) continue;
    const offset = (j - center) / REF_FPS / rate; // seconds in real time
    const adjusted = res.score - 3 * Math.abs(offset); // tie-break toward "on time"
    if (!best || adjusted > best.adjusted) best = { ...res, r, offset, adjusted };
  }
  return best;
}

// Build a "ghost" skeleton: the reference limb directions, anchored on the user's own body and
// scaled to the user's size, so they can see exactly where each limb should go.
function buildGhost(u, r) {
  if (!visible(u, 33, 34) || !visible(r, 33, 34)) return null;
  const uTorso = dist(u[33], u[34]), rTorso = dist(r[33], r[34]) || 1;
  const k = uTorso / rTorso;
  const g = {};
  const chain = (root, m, end) => {
    if (!visible(u, root) || !visible(r, root, m, end)) return;
    g[root] = u[root];
    g[m] = add(u[root], mul(sub(r[m], r[root]), k));
    g[end] = add(g[m], mul(sub(r[end], r[m]), k));
  };
  for (const s of SIDES) { chain(s.sh, s.el, s.wr); chain(s.hip, s.kn, s.an); }
  return { g, torso: uTorso };
}

// Turn pose differences into human coaching cues. All directions are from the dancer's own point
// of view (in the raw camera image, +x is the user's left side).
function generateCues(u, r, ghost) {
  const cues = [];
  const push = (text, severity, key) => cues.push({ text, severity, key: key || text });

  for (const s of SIDES) {
    let armCue = false;
    if (visible(u, s.sh, s.el, s.hip) && visible(r, s.sh, s.el, s.hip)) {
      const raise = (p) => angleBetween(sub(p[s.hip], p[s.sh]), sub(p[s.el], p[s.sh]));
      const d = raise(u) - raise(r);
      if (d < -30) { push(`Raise your ${s.name} arm higher`, -d); armCue = true; }
      else if (d > 30) { push(`Lower your ${s.name} arm`, d); armCue = true; }
    }
    if (visible(u, s.sh, s.el, s.wr) && visible(r, s.sh, s.el, s.wr)) {
      const d = jointAngle(u, s.sh, s.el, s.wr) - jointAngle(r, s.sh, s.el, s.wr);
      if (d < -35) { push(`Straighten your ${s.name} arm`, -d); armCue = true; }
      else if (d > 35) { push(`Bend your ${s.name} elbow more`, d); armCue = true; }
    }
    if (!armCue && ghost?.g[s.wr] && visible(u, s.wr)) {
      const delta = mul(sub(ghost.g[s.wr], u[s.wr]), 1 / ghost.torso);
      if (len(delta) > 0.45) {
        const parts = [];
        if (delta.y < -0.25) parts.push("up"); else if (delta.y > 0.25) parts.push("down");
        if (delta.x > 0.25) parts.push(`to your left`); else if (delta.x < -0.25) parts.push(`to your right`);
        if (parts.length) push(`Move your ${s.name} hand ${parts.join(" and ")}`, len(delta) * 60, `${s.name}-hand`);
      }
    }
    if (visible(u, s.sh, s.hip, s.kn) && visible(r, s.sh, s.hip, s.kn)) {
      const lift = (p) => 180 - jointAngle(p, s.sh, s.hip, s.kn);
      const d = lift(u) - lift(r);
      if (d < -25) push(`Lift your ${s.name} leg higher`, -d);
      else if (d > 25) push(`Lower your ${s.name} leg`, d);
    }
    if (visible(u, s.hip, s.kn, s.an) && visible(r, s.hip, s.kn, s.an)) {
      const d = jointAngle(u, s.hip, s.kn, s.an) - jointAngle(r, s.hip, s.kn, s.an);
      if (d > 25) push(`Bend your ${s.name} knee more`, d);
      else if (d < -25) push(`Straighten your ${s.name} leg`, -d);
    }
  }
  if (visible(u, 33, 34) && visible(r, 33, 34)) {
    const tilt = (p) => { const v = sub(p[33], p[34]); return (Math.atan2(v.x, -v.y) * 180) / Math.PI; };
    const d = tilt(r) - tilt(u);
    if (d > 15) push("Lean more to your left", d);
    else if (d < -15) push("Lean more to your right", -d);
  }
  return cues;
}

// Smooth cues over time so the text doesn't flicker: a cue must persist for a few frames to show.
function stableCues(raw) {
  const mem = state.cueMemory;
  const seen = new Set();
  for (const c of raw) {
    seen.add(c.key);
    const m = mem.get(c.key) || { p: 0 };
    m.p = m.p * 0.88 + 0.12; m.text = c.text; m.severity = c.severity;
    mem.set(c.key, m);
  }
  for (const [k, m] of mem) {
    if (!seen.has(k)) { m.p *= 0.88; if (m.p < 0.05) mem.delete(k); }
  }
  return [...mem.values()].filter((m) => m.p > 0.5).sort((a, b) => b.p * b.severity - a.p * a.severity).slice(0, 2);
}

// ───────────────────────── Drawing ─────────────────────────
function px(p, H) { return [p.x * H, p.y * H]; }

function drawBone(ctx, a, b, H, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(...px(a, H)); ctx.lineTo(...px(b, H)); ctx.stroke();
}

function drawSkeleton(ctx, pose, H, colorFor, width = 5) {
  if (!pose) return;
  ctx.lineCap = "round";
  for (const [a, b] of FRAME_BONES) if (visible(pose, a, b)) drawBone(ctx, pose[a], pose[b], H, "rgba(255,255,255,0.7)", width - 1);
  for (const s of SEGMENTS) {
    if (s.id === "torso" || !visible(pose, s.a, s.b)) continue;
    drawBone(ctx, pose[s.a], pose[s.b], H, colorFor(s.id), width);
  }
  ctx.setLineDash([]);
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if (!visible(pose, i)) continue;
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(...px(pose[i], H), width * 0.8, 0, Math.PI * 2); ctx.fill();
  }
  if (visible(pose, 0)) {
    ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(...px(pose[0], H), dist(pose[11], pose[12]) * H * 0.35, 0, Math.PI * 2); ctx.stroke();
  }
}

function drawGhost(ctx, ghost, H) {
  if (!ghost) return;
  const g = ghost.g;
  ctx.lineCap = "round";
  for (const s of SIDES) {
    for (const [a, b] of [[s.sh, s.el], [s.el, s.wr], [s.hip, s.kn], [s.kn, s.an]]) {
      if (g[a] && g[b]) drawBone(ctx, g[a], g[b], H, COLORS.ghost, 9, [4, 10]);
    }
    for (const end of [s.wr, s.an]) {
      if (!g[end]) continue;
      ctx.setLineDash([]); ctx.strokeStyle = COLORS.ghost; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(...px(g[end], H), 12, 0, Math.PI * 2); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
}

function segColor(score) {
  if (score == null) return COLORS.neutral;
  return score >= 0.75 ? COLORS.good : score >= 0.45 ? COLORS.ok : COLORS.bad;
}

function drawRefFrame(pose) {
  const { width: W, height: H } = els.refCanvas;
  refCtx.drawImage(els.refVideo, 0, 0, W, H);
  if (pose && els.refSkelToggle.checked) drawSkeleton(refCtx, pose, H, () => COLORS.ref, 4);
}

// ───────────────────────── Camera ─────────────────────────
async function enableCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    els.camVideo.srcObject = stream;
    await els.camVideo.play();
    state.camAspect = els.camVideo.videoWidth / els.camVideo.videoHeight;
    sizeCanvas(els.camCanvas, els.camVideo.videoWidth, els.camVideo.videoHeight);
    state.camReady = true;
    els.camPlaceholder.classList.add("hidden");
    updateStartBtn();
  } catch (err) {
    console.error(err);
    els.camPlaceholder.querySelector("strong").textContent = "Camera blocked — allow access in your browser and try again";
  }
}

// ───────────────────────── Main loop ─────────────────────────
function loop() {
  requestAnimationFrame(loop);
  const v = els.refVideo;
  const t = v.currentTime;

  if (state.refReady && !state.analyzing) {
    drawRefFrame(refPoseAt(t, false));
    if (!state.scrubbing) els.scrub.value = String(Math.round((t / (v.duration || 1)) * 1000));
    els.timeLabel.textContent = fmtTime(t);
    if (state.running && state.practice && t >= state.practice.end) finishSession();
  }
  if (!state.camReady) return;

  const cam = els.camVideo;
  if (cam.currentTime !== state.lastCamTime && state.liveLandmarker) {
    state.lastCamTime = cam.currentTime;
    const res = state.liveLandmarker.detectForVideo(cam, performance.now());
    state.userPose = res.landmarks?.[0] ? toPose(res.landmarks[0], state.camAspect) : null;
  }
  const u = state.userPose;
  const { width: W, height: H } = els.camCanvas;

  let match = null, ghost = null;
  if (u && state.refReady && !state.countingDown) {
    match = bestMatch(u, t);
    if (match) ghost = buildGhost(u, match.r);
  }

  // draw the camera mirrored, like a mirror in a dance studio
  camCtx.save();
  camCtx.translate(W, 0); camCtx.scale(-1, 1);
  camCtx.drawImage(cam, 0, 0, W, H);
  camCtx.fillStyle = "rgba(7,7,12,0.25)"; camCtx.fillRect(0, 0, W, H);
  if (els.ghostToggle.checked) drawGhost(camCtx, ghost, H);
  for (const s of SEGMENTS) {
    const sc = match?.seg[s.id];
    state.segEma[s.id] = sc == null ? null : state.segEma[s.id] == null ? sc : state.segEma[s.id] * 0.7 + sc * 0.3;
  }
  drawSkeleton(camCtx, u, H, (id) => segColor(state.segEma[id]), 6);
  camCtx.restore();

  updateHud(u, match, ghost, t);
}

function updateHud(u, match, ghost, t) {
  let cueList = [];
  if (!u) {
    cueList = [{ text: "I can't see you — step into the frame", cls: "info" }];
  } else if (state.refReady && !match) {
    cueList = [{ text: "Step back so your whole body is visible", cls: "info" }];
  }

  if (match) {
    state.scoreEma = state.scoreEma == null ? match.score : state.scoreEma * 0.85 + match.score * 0.15;
    state.offsetEma = state.offsetEma * 0.95 + match.offset * 0.05;
    const s = Math.round(state.scoreEma);
    els.scoreBadge.classList.remove("hidden", "good", "ok", "bad");
    els.scoreBadge.classList.add(s >= 75 ? "good" : s >= 50 ? "ok" : "bad");
    els.scoreNum.textContent = s;

    const stable = stableCues(generateCues(u, match.r, ghost));
    if (stable.length) cueList = stable.map((c) => ({ text: c.text }));
    else if (s >= 80) cueList = [{ text: "Perfect — keep it up!", cls: "praise" }];

    if (state.running) {
      recordSample(t, match, stable[0]?.text);
      speak(stable[0]?.text || (s >= 85 ? "Nice!" : ""));
      const off = state.offsetEma;
      els.timingPill.classList.remove("hidden", "good", "warn");
      if (off < -0.25) { els.timingPill.textContent = `⏱ ~${(-off).toFixed(1)}s behind`; els.timingPill.classList.add("warn"); }
      else if (off > 0.18) { els.timingPill.textContent = "⏱ Rushing — wait for the beat"; els.timingPill.classList.add("warn"); }
      else { els.timingPill.textContent = "⏱ On the beat"; els.timingPill.classList.add("good"); }
    }
  } else if (!state.refReady) {
    els.scoreBadge.classList.add("hidden");
    if (u) cueList = [{ text: "Looking good! Now add a reference dance on the left", cls: "info" }];
  }

  const html = cueList.map((c) => `<div class="cue ${c.cls || ""}">${c.text}</div>`).join("");
  if (els.cues.dataset.html !== html) { els.cues.innerHTML = html; els.cues.dataset.html = html; }
}

function speak(text) {
  if (!els.voiceToggle.checked || !text || !("speechSynthesis" in window)) return;
  const now = performance.now();
  if (speechSynthesis.speaking || now - state.lastSpoken.at < 3500) return;
  if (text === state.lastSpoken.text && now - state.lastSpoken.at < 7000) return;
  const u = new SpeechSynthesisUtterance(text.replace(/your /g, ""));
  u.rate = 1.15; u.volume = 0.9;
  speechSynthesis.speak(u);
  state.lastSpoken = { text, at: now };
}

// ───────────────────────── Sessions ─────────────────────────
function updateStartBtn() {
  els.startBtn.disabled = !(state.refReady && state.camReady && state.liveLandmarker);
}

async function startSession(practice = null) {
  if (state.running || state.countingDown) return;
  const v = els.refVideo;
  state.practice = practice;
  if (practice) els.speed.value = "0.5";
  v.playbackRate = Number(els.speed.value);
  await seek(v, practice ? practice.start : 0);

  state.countingDown = true;
  els.countdown.classList.remove("hidden");
  for (const n of ["3", "2", "1", "Go!"]) { els.countdown.textContent = n; await new Promise((r) => setTimeout(r, 800)); }
  els.countdown.classList.add("hidden");
  state.countingDown = false;

  state.session = { samples: [], partSum: {}, partCount: {}, start: v.currentTime };
  state.scoreEma = null; state.offsetEma = 0; state.cueMemory.clear();
  state.running = true;
  els.startBtn.textContent = "■ Stop"; els.startBtn.classList.add("stop");
  startRecording();
  v.play();
}

function recordSample(t, match, cue) {
  const s = state.session;
  if (!s) return;
  s.samples.push({ t, score: match.score, cue });
  for (const seg of SEGMENTS) {
    const sc = match.seg[seg.id];
    if (sc == null) continue;
    s.partSum[seg.part] = (s.partSum[seg.part] || 0) + sc;
    s.partCount[seg.part] = (s.partCount[seg.part] || 0) + 1;
  }
}

function stopSession(showSummary) {
  if (!state.running) return;
  state.running = false;
  els.refVideo.pause();
  els.startBtn.textContent = "▶ Start dancing"; els.startBtn.classList.remove("stop");
  els.timingPill.classList.add("hidden");
  speechSynthesis?.cancel?.();
  stopRecording().then(() => { if (showSummary) renderSummary(); });
}
const finishSession = () => stopSession(true);

function startRecording() {
  state.recordedChunks = [];
  if (!window.MediaRecorder || !els.camCanvas.captureStream) return;
  const type = ["video/webm;codecs=vp9", "video/webm", "video/mp4"].find((t) => MediaRecorder.isTypeSupported(t));
  if (!type) return;
  try {
    state.recorder = new MediaRecorder(els.camCanvas.captureStream(30), { mimeType: type });
    state.recorder.ondataavailable = (e) => e.data.size && state.recordedChunks.push(e.data);
    state.recorder.start(500);
  } catch (err) { console.warn("Recording unavailable", err); state.recorder = null; }
}
function stopRecording() {
  const rec = state.recorder;
  if (!rec || rec.state === "inactive") return Promise.resolve();
  return new Promise((resolve) => {
    rec.onstop = () => {
      if (state.takeUrl) URL.revokeObjectURL(state.takeUrl);
      const blob = new Blob(state.recordedChunks, { type: rec.mimeType });
      state.takeUrl = URL.createObjectURL(blob);
      els.downloadTake.href = state.takeUrl;
      els.downloadTake.download = rec.mimeType.includes("mp4") ? "my-dance-take.mp4" : "my-dance-take.webm";
      resolve();
    };
    rec.stop();
    state.recorder = null;
  });
}

// ───────────────────────── Summary ─────────────────────────
function renderSummary() {
  const s = state.session;
  if (!s || s.samples.length < 10) return;
  const scores = s.samples.map((x) => x.score);
  const overall = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);

  els.sumScore.textContent = overall;
  els.sumScore.parentElement.style.setProperty("--pct", overall + "%");
  const [grade, sub] =
    overall >= 85 ? ["Flawless 🔥", "You basically are the dancer now."] :
    overall >= 70 ? ["Great groove", "Clean it up with the tips below and you'll nail it."] :
    overall >= 55 ? ["Getting there", "Drill the toughest moments at half speed."] :
    ["Keep practicing", "Start with the toughest moments at 0.5× — slow is smooth, smooth is fast."];
  els.sumGrade.textContent = grade;
  els.sumSub.textContent = sub;

  // body parts
  els.sumParts.innerHTML = Object.keys(s.partSum)
    .map((p) => [p, Math.round((100 * s.partSum[p]) / s.partCount[p])])
    .sort((a, b) => a[1] - b[1])
    .map(([p, v]) => `<div class="part-row"><span>${p}</span><div class="bar"><div style="width:${v}%;background:${v >= 75 ? COLORS.good : v >= 50 ? COLORS.ok : COLORS.bad}"></div></div><b>${v}</b></div>`)
    .join("");

  // toughest moments: 2-second buckets with lowest average score
  const buckets = new Map();
  for (const x of s.samples) {
    const k = Math.floor(x.t / 2);
    const b = buckets.get(k) || { sum: 0, n: 0, cues: {} };
    b.sum += x.score; b.n++;
    if (x.cue) b.cues[x.cue] = (b.cues[x.cue] || 0) + 1;
    buckets.set(k, b);
  }
  const ranked = [...buckets.entries()].filter(([, b]) => b.n >= 3).map(([k, b]) => ({ k, avg: b.sum / b.n, cue: Object.entries(b.cues).sort((a, c) => c[1] - a[1])[0]?.[0] })).sort((a, b) => a.avg - b.avg);
  const picked = [];
  for (const m of ranked) { if (picked.length < 3 && m.avg < 80 && !picked.some((p) => Math.abs(p.k - m.k) <= 1)) picked.push(m); }
  picked.sort((a, b) => a.k - b.k);
  els.sumMoments.innerHTML = picked.length
    ? picked.map((m) => `<div class="moment"><span class="t">${fmtTime(m.k * 2)}</span><span class="tip"><b>${Math.round(m.avg)}% match.</b> ${m.cue ? m.cue + "." : "Watch the reference closely here."}</span><button class="ghost-btn" data-start="${Math.max(0, m.k * 2 - 1)}" data-end="${m.k * 2 + 3}">Drill 0.5×</button></div>`).join("")
    : `<div class="moment"><span class="tip">No weak spots found — try it at full speed!</span></div>`;

  drawChart(s.samples);
  els.downloadTake.classList.toggle("hidden", !state.takeUrl);
  els.summary.showModal();
}

function drawChart(samples) {
  const c = els.sumChart, ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 24;
  ctx.clearRect(0, 0, W, H);
  const t0 = samples[0].t, t1 = samples[samples.length - 1].t || t0 + 1;
  const X = (t) => pad + ((t - t0) / (t1 - t0 || 1)) * (W - pad * 2);
  const Y = (v) => H - pad - (v / 100) * (H - pad * 2);
  ctx.strokeStyle = "#2a2a3d"; ctx.lineWidth = 1; ctx.fillStyle = "#9a98ad"; ctx.font = "11px Inter";
  for (const v of [50, 75]) { ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W - pad, Y(v)); ctx.stroke(); ctx.fillText(v, 2, Y(v) + 4); }
  // smoothed line
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, "#ff3d81"); grad.addColorStop(1, "#7c4dff");
  ctx.strokeStyle = grad; ctx.lineWidth = 2.5; ctx.beginPath();
  let ema = samples[0].score;
  samples.forEach((x, i) => { ema = ema * 0.9 + x.score * 0.1; i ? ctx.lineTo(X(x.t), Y(ema)) : ctx.moveTo(X(x.t), Y(ema)); });
  ctx.stroke();
  ctx.fillText(fmtTime(t0), pad, H - 6);
  ctx.fillText(fmtTime(t1), W - pad - 24, H - 6);
}

// ───────────────────────── Events ─────────────────────────
function handleFile(file) { if (file && file.type.startsWith("video/")) analyzeReference(file); }

els.fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
els.changeVideoBtn.addEventListener("click", () => els.fileInput.click());
const refScreen = $("#refScreen");
refScreen.addEventListener("dragover", (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); });
refScreen.addEventListener("dragleave", () => els.dropzone.classList.remove("drag"));
refScreen.addEventListener("drop", (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); handleFile(e.dataTransfer.files[0]); });
els.camBtn.addEventListener("click", (e) => { e.stopPropagation(); enableCamera(); });
els.startBtn.addEventListener("click", () => (state.running ? finishSession() : startSession()));
els.speed.addEventListener("change", () => { els.refVideo.playbackRate = Number(els.speed.value); });
els.refVideo.addEventListener("ended", () => state.running && finishSession());

els.scrub.addEventListener("input", () => {
  state.scrubbing = true;
  if (state.running) stopSession(false);
  els.refVideo.currentTime = (Number(els.scrub.value) / 1000) * els.refVideo.duration;
});
els.scrub.addEventListener("change", () => { state.scrubbing = false; });
$("#refScreen").addEventListener("click", () => {
  if (!state.refReady || state.running) return;
  els.refVideo.paused ? els.refVideo.play() : els.refVideo.pause();
});

els.closeSummary.addEventListener("click", () => els.summary.close());
els.againBtn.addEventListener("click", () => { els.summary.close(); startSession(state.practice); });
els.sumMoments.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-start]");
  if (!btn) return;
  els.summary.close();
  startSession({ start: Number(btn.dataset.start), end: Number(btn.dataset.end) });
});
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !els.startBtn.disabled && e.target === document.body) { e.preventDefault(); els.startBtn.click(); }
});

initModels();
requestAnimationFrame(loop);
