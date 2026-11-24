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

const data = await (await fetch("assets/results.json")).json();

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6d7780, 2.0));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
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
    // The FE element uses the right-handed w/ry sign convention encoded by S in frame_element.py.
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
  const m = new THREE.MeshStandardMaterial({ color: 0xb7bec5, transparent: true, opacity: 0.25, roughness: 0.9 });
  const deck = new THREE.Mesh(g,m);
  deck.position.set(design.parameters.span_m/2, 0, -design.parameters.deck_thickness_m/2 - 0.012);
  scene.add(deck);
}

function addSupports(scene, design) {
  const group = new THREE.Group();
  const geom = new THREE.ConeGeometry(0.16, 0.28, 4);
  geom.rotateX(Math.PI/2);
  for (const s of design.supports) {
    const mat = new THREE.MeshStandardMaterial({ color: 0x2f353a, roughness: .75 });
    const mesh = new THREE.Mesh(geom, mat);
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
    const arrow = new THREE.ArrowHelper(dir, origin, 0.42, 0xd47a24, 0.12, 0.065);
    group.add(arrow);
  }
  scene.add(group);
}

function normalizedModeDofs(design, key) {
  const q = design[key].map(row => row.slice());
  let maxT = 0;
  for (const r of q) maxT = Math.max(maxT, Math.hypot(r[0],r[1],r[2]));
  if (maxT <= 0) return q;
  const target = 0.48; // viewer-only normalized peak translation in metres
  const f = target/maxT;
  return q.map(r => r.map(v => v*f));
}

function rebuildDesign(index) {
  const scene = scenes[index], design = designData[index];
  disposeGroup(buildState[index]?.tubeGroup);
  disposeGroup(buildState[index]?.markerGroup);
  disposeGroup(buildState[index]?.modeGroup);
  const tubeGroup = new THREE.Group();
  const markerGroup = new THREE.Group();
  const modeGroup = new THREE.Group();
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
  scene.add(tubeGroup, markerGroup, modeGroup);
  buildState[index] = { tubeGroup, markerGroup, modeGroup, memberMeshes };
}

function rebuildAll() {
  for (let i=0;i<2;i++) rebuildDesign(i);
}

function updateModeOverlay(index, timePhase) {
  const state = buildState[index];
  if (!state) return;
  disposeGroup(state.modeGroup);
  const group = new THREE.Group();
  const mode = animationSelect.value;
  if (mode === 'none') { sceneAdd(index, group); state.modeGroup = group; return; }
  const design = designData[index];
  const key = mode === 'vibration' ? 'vibration_mode_dofs' : 'buckling_mode_dofs';
  const dofs = normalizedModeDofs(design, key);
  const amp = mode === 'vibration' ? Math.sin(timePhase) : 0.5*(1-Math.cos(timePhase));
  const mat = new THREE.LineBasicMaterial({ color: 0x15191d, transparent: true, opacity: 0.72 });
  for (const e of design.elements) {
    const curve = new HermiteElementCurve(design, e, dofs, amp);
    const pts = curve.getPoints(8);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.Line(geo, mat.clone()));
  }
  sceneAdd(index, group);
  state.modeGroup = group;
}
function sceneAdd(index, group) { scenes[index].add(group); }

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

function summaryTable() {
  const i=data.initial.metrics, o=data.optimized.metrics;
  const b=data.comparisons;
  const baseline = data.baseline_metrics || null;
  const m5Baseline = data.uniform_feasible_baseline || null;
  // Baseline values are exported below from results.json when available.
  const ub = data.baseline || {};
  const rows = [
    ['Total mass', `${fmt(i.mass_total_kg,0)} kg`, `${fmt(ub.mass_total_kg,0)} kg`, `${fmt(o.mass_total_kg,0)} kg`],
    ['Member steel mass', `${fmt(i.mass_members_kg,0)} kg`, `${fmt(ub.mass_members_kg,0)} kg`, `${fmt(o.mass_members_kg,0)} kg`],
    ['Truss depth H', `${fmt(data.initial.parameters.H_m,3)} m`, `${fmt(ub.H_m,3)} m`, `${fmt(data.optimized.parameters.H_m,3)} m`],
    ['Camber', `${fmt(data.initial.parameters.camber_m,3)} m`, `${fmt(ub.camber_m,3)} m`, `${fmt(data.optimized.parameters.camber_m,3)} m`],
    ['Max von Mises stress', `${fmt(i.max_stress_mpa,2)} MPa`, `${fmt(ub.max_stress_mpa,2)} MPa`, `${fmt(o.max_stress_mpa,2)} MPa`],
    ['Fatigue factor of safety', fmt(i.fatigue_factor_of_safety,3), fmt(ub.fatigue_factor_of_safety,3), fmt(o.fatigue_factor_of_safety,3)],
    ['Max deck deflection', `${fmt(i.max_deflection_mm,2)} mm`, `${fmt(ub.max_deflection_mm,2)} mm`, `${fmt(o.max_deflection_mm,2)} mm`],
    ['Lowest natural frequency', `${fmt(i.lowest_frequency_hz,3)} Hz`, `${fmt(ub.lowest_frequency_hz,3)} Hz`, `${fmt(o.lowest_frequency_hz,3)} Hz`],
    ['Buckling load factor', fmt(i.buckling_factor,2), fmt(ub.buckling_factor,2), fmt(o.buckling_factor,2)],
    ['Max member slenderness', fmt(i.max_slenderness,2), fmt(ub.max_slenderness,2), fmt(o.max_slenderness,2)],
  ];
  return `<table class="report"><thead><tr><th>Metric</th><th>Initial</th><th>Uniform feasible baseline</th><th>Optimized</th></tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

document.getElementById('headline-result').innerHTML = `<strong>Headline result.</strong> The optimized bridge uses <strong>${fmt(data.optimized.metrics.mass_members_kg,0)} kg of member steel</strong>, ${fmt(data.comparisons.member_mass_reduction_vs_uniform_feasible_pct,1)}% less than the lightest uniform feasible baseline in our model. Fatigue and member slenderness govern the final design; vibration and linear buckling remain inactive.`;
document.getElementById('result-summary-table').innerHTML = summaryTable();
document.getElementById('result-interpretation').innerHTML = `<p>The starting bridge is lighter than the optimum only because it is infeasible: it violates both static strength and fatigue. The meaningful comparison is against a feasible design. Relative to the uniform feasible baseline, optimization removes <strong>${fmt(data.comparisons.member_mass_reduction_vs_uniform_feasible_pct,1)}%</strong> of member steel while satisfying every requirement.</p>`;
document.getElementById('model-optimization-text').innerHTML = `<p>The structural model contains <strong>${data.model.physical_members}</strong> physical circular-hollow-section members and <strong>${data.model.finite_elements}</strong> 3D frame elements. Every physical member is split into two finite elements. The deck is treated with option D2: its pressure and mass are transferred to five longitudinal support lines, while the 8 mm plate itself is checked separately by the required simply-supported-strip calculation.</p><p>The design uses ${data.model.design_variables}. The optimization was staged with a fully stressed warm start, MMA, and a feasible sequential-linear refinement. All retained incumbents were independently reanalyzed with the complete FE/eigenvalue model.</p><p>The required S1 nonlinear/stability component is the state-dependent linear-buckling eigenproblem. The final design has <strong>α<sub>cr</sub> = ${fmt(data.optimized.metrics.buckling_factor,2)}</strong> > ${fmt(data.limits.R5_buckling_factor_min,1)}, so R5 is inactive. ${data.with_without_R5.conclusion}</p>`;

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w,h,false);
  camera.aspect = (w/2)/h;
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
    phase += 0.028;
    // Rebuild the lightweight mode centerlines at ~24 fps, not the tube meshes.
    if (ms-lastModeFrame > 42) {
      updateModeOverlay(0, phase); updateModeOverlay(1, phase); lastModeFrame=ms;
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
animationSelect.addEventListener('change', () => { phase=0; updateModeOverlay(0,0); updateModeOverlay(1,0); });
playButton.addEventListener('click', () => { playing=!playing; playButton.textContent = playing ? 'Pause animation' : 'Play animation'; });

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

for (let i=0;i<2;i++) { addDeck(scenes[i],designData[i]); addSupports(scenes[i],designData[i]); addLoads(scenes[i],designData[i]); }
resize();
scheduleRebuild();
updateModeOverlay(0,0); updateModeOverlay(1,0);
requestAnimationFrame(animate);
