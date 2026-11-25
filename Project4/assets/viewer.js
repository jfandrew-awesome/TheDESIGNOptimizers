import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const container = document.getElementById("bridge-viewer");
const metricsRoot = document.getElementById("metrics-panel");
const tooltip = document.getElementById("viewer-tooltip");
const deformSlider = document.getElementById("deform-scale");
const deformLabel = document.getElementById("deform-scale-label");
const thicknessSlider = document.getElementById("thickness-scale");
const thicknessLabel = document.getElementById("thickness-scale-label");
const animationSelect = document.getElementById("animation-mode");
const playButton = document.getElementById("animation-toggle");

let data;
try {
  const response = await fetch("assets/results.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`results.json returned HTTP ${response.status}`);
  data = await response.json();
} catch (err) {
  container.innerHTML = `<div class="viewer-error"><strong>3D viewer data could not be loaded.</strong><br>${String(err.message || err)}</div>`;
  throw err;
}

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.setClearColor(0xf3f6f8, 1);
container.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(34, 1, 0.05, 120);
camera.position.set(17, -20, 12);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(7.5, 0, 0.9);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 9;
controls.maxDistance = 50;

function makeScene() {
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6d7780, 1.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(4, -6, 10);
  scene.add(sun);
  const grid = new THREE.GridHelper(18, 18, 0xcbd3da, 0xe2e7eb);
  grid.position.set(7.5, 0, -0.04);
  scene.add(grid);
  return scene;
}

const scenes = [makeScene(), makeScene()];
const designData = [data.initial, data.optimized];
const buildState = [null, null];
const modeState = [null, null];
const modeCache = designData.map(d => ({
  vibration: normalizeMode(d, d.vibration_mode_dofs),
  buckling: normalizeMode(d, d.buckling_mode_dofs),
}));
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let playing = true;
let phase = 0;

function mulR(R, v) {
  return [
    R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
    R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
    R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
  ];
}
function mulRT(R, v) {
  return [
    R[0][0]*v[0] + R[1][0]*v[1] + R[2][0]*v[2],
    R[0][1]*v[0] + R[1][1]*v[1] + R[2][1]*v[2],
    R[0][2]*v[0] + R[1][2]*v[1] + R[2][2]*v[2],
  ];
}
function add3(a,b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function scale3(a,s) { return [a[0]*s, a[1]*s, a[2]*s]; }

class HermiteElementCurve extends THREE.Curve {
  constructor(design, element, dofs, deformationScale) {
    super();
    this.design = design;
    this.element = element;
    this.dofs = dofs;
    this.scale = deformationScale;
  }
  getPoint(t, target = new THREE.Vector3()) {
    const e = this.element;
    const [n1,n2] = e.nodes;
    const x1 = this.design.nodes_m[n1];
    const R = e.basis_global_to_local;
    const L = e.length_m;
    const d1 = this.dofs[n1], d2 = this.dofs[n2];
    const tr1 = mulR(R, d1.slice(0,3)), tr2 = mulR(R, d2.slice(0,3));
    const ro1 = mulR(R, d1.slice(3,6)), ro2 = mulR(R, d2.slice(3,6));
    const s=t, s2=s*s, s3=s2*s;
    const h1=1-3*s2+2*s3, h2=L*(s-2*s2+s3), h3=3*s2-2*s3, h4=L*(-s2+s3);
    const ul = (1-s)*tr1[0] + s*tr2[0];
    const vl = h1*tr1[1] + h2*ro1[2] + h3*tr2[1] + h4*ro2[2];
    const wl = h1*tr1[2] - h2*ro1[1] + h3*tr2[2] - h4*ro2[1];
    const original = mulRT(R, [s*L, 0, 0]);
    const def = mulRT(R, [ul, vl, wl]);
    const p = add3(add3(x1, original), scale3(def, this.scale));
    return target.set(p[0], p[1], p[2]);
  }
}

function stressColor(util) {
  const u = Math.max(0, Math.min(1.15, util));
  const light = 94 - 53 * Math.min(1, u);
  return new THREE.Color(`hsl(210, ${35 + 35*Math.min(1,u)}%, ${light}%)`);
}

function disposeGroup(group) {
  if (!group) return;
  group.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
  });
  group.parent?.remove(group);
}

function addDeck(scene, design) {
  const g = new THREE.BoxGeometry(design.parameters.span_m, design.parameters.width_m, design.parameters.deck_thickness_m);
  const m = new THREE.MeshStandardMaterial({ color: 0xb7bec5, transparent: true, opacity: 0.23, roughness: 0.9 });
  const deck = new THREE.Mesh(g,m);
  deck.position.set(design.parameters.span_m/2, 0, -design.parameters.deck_thickness_m/2 - 0.012);
  scene.add(deck);
}

function addSupports(scene, design) {
  const group = new THREE.Group();
  const geom = new THREE.ConeGeometry(0.16, 0.28, 4);
  geom.rotateX(Math.PI/2);
  for (const s of design.supports) {
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x2f353a, roughness: .75 }));
    mesh.position.set(s.position[0], s.position[1], -0.16);
    mesh.rotation.z = Math.PI/4;
    group.add(mesh);
  }
  scene.add(group);
}

function addLoads(scene, design) {
  const group = new THREE.Group();
  for (const a of design.load_arrows) {
    const origin = new THREE.Vector3(...a.position);
    const dir = new THREE.Vector3(...a.direction).normalize();
    group.add(new THREE.ArrowHelper(dir, origin, 0.42, 0xd47a24, 0.12, 0.065));
  }
  scene.add(group);
}

function normalizeMode(design, dofs) {
  const q = dofs.map(row => row.slice());
  let maxT = 0;
  for (const r of q) maxT = Math.max(maxT, Math.hypot(r[0],r[1],r[2]));
  if (maxT <= 0) return q;
  const f = 0.48/maxT;
  return q.map(r => r.map(v => v*f));
}

function rebuildDesign(index) {
  const scene = scenes[index], design = designData[index];
  disposeGroup(buildState[index]?.tubeGroup);
  disposeGroup(buildState[index]?.markerGroup);
  const tubeGroup = new THREE.Group();
  const markerGroup = new THREE.Group();
  const deformScale = Number(deformSlider.value);
  const thicknessScale = Number(thicknessSlider.value);
  const memberMeshes = [];

  for (const e of design.elements) {
    const m = design.members[e.member];
    const curve = new HermiteElementCurve(design, e, design.static_dofs, deformScale);
    const geom = new THREE.TubeGeometry(curve, 8, Math.max(0.004, m.radius_m*thicknessScale), 6, false);
    const mat = new THREE.MeshStandardMaterial({ color: stressColor(e.stress_utilization_fatigue), roughness: 0.62, metalness: 0.08 });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData = { element: e, member: m, designName: design.name };
    tubeGroup.add(mesh);
    memberMeshes.push(mesh);

    if (e.stress_utilization_fatigue > 1.0) {
      const p = curve.getPoint(0.5);
      const marker = new THREE.Mesh(
        new THREE.OctahedronGeometry(Math.max(0.075, m.radius_m*2.2), 0),
        new THREE.MeshBasicMaterial({ color: 0x111111, wireframe: true })
      );
      marker.position.copy(p);
      marker.userData = mesh.userData;
      markerGroup.add(marker);
    }
  }
  scene.add(tubeGroup, markerGroup);
  buildState[index] = { tubeGroup, markerGroup, memberMeshes };
}

function rebuildAll() { for (let i=0;i<2;i++) rebuildDesign(i); }

const MODE_SAMPLES = 8;
function createModeOverlay(index) {
  const design = designData[index];
  const segmentCount = design.elements.length * MODE_SAMPLES;
  const positions = new Float32Array(segmentCount * 2 * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x15191d, transparent: true, opacity: 0.72 });
  const lines = new THREE.LineSegments(geometry, material);
  lines.frustumCulled = false;
  scenes[index].add(lines);
  modeState[index] = { lines, geometry, positions };
}

function updateModeOverlay(index, timePhase) {
  const state = modeState[index];
  if (!state) return;
  const mode = animationSelect.value;
  state.lines.visible = mode !== 'none';
  if (mode === 'none') return;
  const design = designData[index];
  const dofs = modeCache[index][mode];
  const amp = mode === 'vibration' ? Math.sin(timePhase) : 0.5*(1-Math.cos(timePhase));
  let k = 0;
  const p0 = new THREE.Vector3(), p1 = new THREE.Vector3();
  for (const e of design.elements) {
    const curve = new HermiteElementCurve(design, e, dofs, amp);
    for (let j=0; j<MODE_SAMPLES; j++) {
      curve.getPoint(j/MODE_SAMPLES, p0);
      curve.getPoint((j+1)/MODE_SAMPLES, p1);
      state.positions[k++] = p0.x; state.positions[k++] = p0.y; state.positions[k++] = p0.z;
      state.positions[k++] = p1.x; state.positions[k++] = p1.y; state.positions[k++] = p1.z;
    }
  }
  state.geometry.attributes.position.needsUpdate = true;
  state.geometry.computeBoundingSphere();
}

function fmt(v, digits=2) { return Number(v).toLocaleString(undefined,{maximumFractionDigits:digits, minimumFractionDigits:digits}); }
function status(pass) { return `<span class="status ${pass?'pass':'fail'}">${pass?'PASS':'FAIL'}</span>`; }
function metricCard(d) {
  const m=d.metrics;
  const rows = [
    ['Total mass', `${fmt(m.mass_total_kg,1)} kg`, true],
    ['R1 max stress', `${fmt(m.max_stress_mpa,2)} MPa / ${fmt(m.static_strength_limit_mpa,0)}`, m.max_stress_mpa<=m.static_strength_limit_mpa],
    ['R2 fatigue FOS', `${fmt(m.fatigue_factor_of_safety,3)} / 2.000`, m.fatigue_factor_of_safety>=2],
    ['R3 deck deflection', `${fmt(m.max_deflection_mm,2)} mm / ${fmt(m.deflection_limit_mm,1)}`, m.max_deflection_mm<=m.deflection_limit_mm],
    ['R4 lowest frequency', `${fmt(m.lowest_frequency_hz,3)} Hz / ${fmt(m.frequency_limit_hz,1)}`, m.lowest_frequency_hz>=m.frequency_limit_hz],
    ['R5 buckling factor', `${fmt(m.buckling_factor,2)} / ${fmt(m.buckling_limit,1)}`, m.buckling_factor>=m.buckling_limit],
    ['R6 max slenderness', `${fmt(m.max_slenderness,2)} / ${fmt(m.slenderness_limit,0)}`, m.max_slenderness<=m.slenderness_limit],
    ['D2 deck strip stress', `${fmt(m.deck_plate_stress_mpa,2)} MPa / 53.3`, m.deck_plate_stress_mpa<=53.3],
  ];
  return `<section class="metrics-card"><h3>${d.name}</h3><table class="metrics-table">${rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]} ${status(r[2])}</td></tr>`).join('')}</table></section>`;
}
metricsRoot.innerHTML = designData.map(metricCard).join('');

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w,h,false);
  camera.aspect = Math.max(0.2, (w/2)/h);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function renderSplit() {
  const w = container.clientWidth, h=container.clientHeight;
  renderer.setScissorTest(true);
  for (let i=0;i<2;i++) {
    const x = i===0 ? 0 : Math.floor(w/2);
    const ww = i===0 ? Math.floor(w/2) : w-Math.floor(w/2);
    renderer.setViewport(x,0,ww,h);
    renderer.setScissor(x,0,ww,h);
    renderer.render(scenes[i], camera);
  }
  renderer.setScissorTest(false);
}

let lastModeFrame = 0;
function animate(ms) {
  requestAnimationFrame(animate);
  controls.update();
  if (playing && animationSelect.value !== 'none') {
    phase += 0.032;
    if (ms-lastModeFrame > 42) {
      updateModeOverlay(0, phase);
      updateModeOverlay(1, phase);
      lastModeFrame=ms;
    }
  }
  renderSplit();
}

function scheduleRebuild() {
  deformLabel.textContent = `×${deformSlider.value}`;
  thicknessLabel.textContent = `×${Number(thicknessSlider.value).toFixed(1)}`;
  rebuildAll();
}
deformSlider.addEventListener('change', scheduleRebuild);
thicknessSlider.addEventListener('change', scheduleRebuild);
animationSelect.addEventListener('change', () => {
  phase=0;
  updateModeOverlay(0,0);
  updateModeOverlay(1,0);
});
playButton.addEventListener('click', () => {
  playing=!playing;
  playButton.textContent = playing ? 'Pause animation' : 'Play animation';
});

renderer.domElement.addEventListener('pointermove', ev => {
  const rect = renderer.domElement.getBoundingClientRect();
  const xpx=ev.clientX-rect.left, ypx=ev.clientY-rect.top;
  const side = xpx < rect.width/2 ? 0 : 1;
  const localX = side===0 ? xpx : xpx-rect.width/2;
  pointer.x = 2*localX/(rect.width/2)-1;
  pointer.y = -2*ypx/rect.height+1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(buildState[side]?.memberMeshes || [], false);
  if (!hits.length) { tooltip.style.display='none'; return; }
  const u=hits[0].object.userData;
  const e=u.element, m=u.member;
  tooltip.style.display='block';
  tooltip.style.left=`${Math.min(rect.width-270, xpx+14)}px`;
  tooltip.style.top=`${Math.min(rect.height-120, ypx+14)}px`;
  tooltip.innerHTML = `<strong>${u.designName}</strong><br>${m.name} (${m.kind})<br>radius ${fmt(m.radius_m*1000,1)} mm · stress ${fmt(e.stress_pa/1e6,1)} MPa<br>fatigue utilization ${fmt(e.stress_utilization_fatigue,3)}${e.stress_utilization_fatigue>1?'<br><strong>OVER FATIGUE LIMIT</strong>':''}<br>slenderness ${fmt(m.slenderness,1)}`;
});
renderer.domElement.addEventListener('pointerleave',()=>tooltip.style.display='none');

for (let i=0;i<2;i++) {
  addDeck(scenes[i],designData[i]);
  addSupports(scenes[i],designData[i]);
  addLoads(scenes[i],designData[i]);
}
resize();
scheduleRebuild();
for (let i=0;i<2;i++) createModeOverlay(i);
updateModeOverlay(0,0);
updateModeOverlay(1,0);
requestAnimationFrame(animate);
