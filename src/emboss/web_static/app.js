import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { ImageGallery, drawFootprint } from "/emboss-imagery/gallery.js";
const $ = (id) => document.getElementById(id);
let houses = [],
  results = [],
  selected = new Set(),
  renderer,
  scene,
  camera,
  controls,
  current;
let activeHouse = null, imageInfo = null, viewVersion = 0, processing = false;
let review = null, reviewVersion = 0, libraryBusy = false, undoToken = null, modelRadius = 1;
const gallery = new ImageGallery($("image-gallery"), $("gallery-status"), id => {
  if (!review) return;
  review.choices[review.house] = id;
  updateGalleryAction();
});
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok)
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  return data;
}
function tab(which) {
  $("selection").hidden = which !== "select";
  $("workspace").hidden = which !== "results";
  sessionStorage.setItem("emboss-view", which);
  $("select-tab").classList.toggle("active", which === "select");
  $("select-tab").setAttribute("aria-pressed", String(which === "select"));
  $("results-tab").setAttribute("aria-pressed", String(which === "results"));
  $("results-tab").classList.toggle("active", which === "results");
  if (which === "results") {
    load().catch((error) => showError(error));
    setTimeout(resize, 0);
  }
}
$("select-tab").onclick = () => tab("select");
$("results-tab").onclick = () => tab("results");
function showError(error) {
  $("progress").textContent = error.message;
  $("progress").className = "error";
}
window.addEventListener("message", (event) => {
  if (
    event.origin !== location.origin ||
    event.source !== $("picker").contentWindow ||
    event.data?.type !== "building-data-acquired"
  )
    return;
  selected = new Set(event.data.houses);
  openGallery(event.data.houses, "acquire").catch(showError);
});
async function load() {
  let removed;
  [houses, results, removed] = await Promise.all([
    api("/api/houses"),
    api("/api/results"),
    api("/api/removed-buildings"),
  ]);
  selected = new Set([...selected].filter(id => houses.some(house => house.id === id)));
  if (activeHouse && !houses.some(house => house.id === activeHouse)) clearResult();
  $("library-empty").hidden = houses.length > 0;
  $("building-count").textContent = houses.length;
  $("filter-buildings").hidden = houses.length === 0;
  $("select-all").disabled = houses.length === 0;
  renderRemoved(removed);
  const done = new Map(results.map((item) => [item.house_id, item]));
  $("houses").replaceChildren(
    ...houses.map((house) => {
      const row = document.createElement("div");
      row.className = "house";
      row.dataset.house = house.id;
      row.dataset.number = String(house.building_fid);
      row.classList.toggle("current", house.id === activeHouse);
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.setAttribute("aria-label", `Select building ${house.building_fid}`);
      checkbox.checked = selected.has(house.id);
      checkbox.onchange = () => {
        checkbox.checked ? selected.add(house.id) : selected.delete(house.id);
        updateActions();
      };
      const button = document.createElement("button");
      button.className = "open-building";
      button.textContent = `Building ${house.building_fid}`;
      const status = document.createElement("small");
      status.textContent = done.has(house.id)
        ? `Open model · ${done.get(house.id).solid_count} roof details`
        : "Choose imagery & model";
      button.append(status);
      button.onclick = () =>
        done.has(house.id) ? view(house.id).then(revealResult).catch(showError) : openGallery([house.id], "acquire").catch(showError);
      const remove = document.createElement("button");
      remove.className = "remove-building";
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", `Remove building ${house.building_fid}`);
      remove.title = "Remove this building and its saved model. You can undo this.";
      remove.onclick = () => removeBuilding(house);
      const selectionLabel = document.createElement("label");
      selectionLabel.className = "batch-select";
      selectionLabel.append(checkbox);
      row.append(selectionLabel, button, remove);
      return row;
    }),
  );
  filterBuildings();
  if (!activeHouse && results.length && !$("workspace").hidden) {
    const last = sessionStorage.getItem("emboss-building");
    const first = results.find(result => result.house_id === last) || results[0];
    await view(first.house_id);
  }
}
function updateActions() {
  const visible = [...$("houses").children].filter(row => !row.hidden);
  $("select-all").disabled = !visible.length;
  $("select-all").textContent = visible.length && visible.every(row => selected.has(row.dataset.house)) ? "Clear selection" : "Select all";
  $("run").disabled = processing || libraryBusy || !selected.size;
  $("run").textContent = processing ? "Modeling…" : selected.size ? `Model ${selected.size} ${selected.size === 1 ? "building" : "buildings"}` : "Model selected";
  $("selection-count").textContent = selected.size ? `${selected.size} selected` : "None selected";
  for (const button of document.querySelectorAll(".remove-building, .restore-building, #undo-remove")) {
    button.disabled = libraryBusy;
  }
  $("change-image").disabled = processing || libraryBusy;
  updateGalleryAction();
}
function filterBuildings() {
  const query = $("filter-buildings").value.trim().toLowerCase();
  for (const row of $("houses").children) row.hidden = !row.dataset.number.toLowerCase().includes(query);
  $("no-matches").hidden = !houses.length || [...$("houses").children].some(row => !row.hidden);
  updateActions();
}
$("filter-buildings").oninput = filterBuildings;
$("back-library").onclick = () => { $("library-title").scrollIntoView(); $("library-title").focus({preventScroll: true}); };
$("select-all").onclick = () => {
  const visible = [...$("houses").children].filter(row => !row.hidden);
  const allSelected = visible.every(row => selected.has(row.dataset.house));
  for (const row of visible) {
    allSelected ? selected.delete(row.dataset.house) : selected.add(row.dataset.house);
    row.querySelector("input").checked = !allSelected;
  }
  updateActions();
};
document.querySelector(".skip-link").onclick = () => { tab("results"); $("workspace").focus(); };
function revealResult() {
  if (matchMedia("(max-width: 800px)").matches) document.querySelector(".result").scrollIntoView({block: "start"});
  $("title").focus({preventScroll: true});
}
function clearResult() {
  viewVersion++;
  activeHouse = null;
  imageInfo = null;
  if (current) {
    scene.remove(current);
    current.traverse(object => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    current = null;
  }
  $("title").textContent = "Select a building";
  $("empty").hidden = false;
  $("model-help").hidden = true;
  $("view-status").hidden = true;
  $("retry-view").hidden = true;
  sessionStorage.removeItem("emboss-building");
  $("imagery-panel").hidden = true;
  $("download").hidden = true;
  $("summary").textContent = "";
  $("detail-list").textContent = "";
  for (const id of ["orthophoto", "segmentation"]) $(id).removeAttribute("src");
}
function renderRemoved(items) {
  $("removed-buildings").hidden = items.length === 0;
  $("removed-list").replaceChildren(...items.map(item => {
    const row = document.createElement("div");
    row.className = "removed-building";
    const label = document.createElement("span");
    label.textContent = `Building ${item.building_fid || item.house_id}`;
    const restore = document.createElement("button");
    restore.textContent = "Restore";
    restore.className = "restore-building";
    restore.setAttribute("aria-label", `Restore ${label.textContent.toLowerCase()}`);
    restore.onclick = () => restoreBuilding(item.undo_token);
    row.append(label, restore);
    return row;
  }));
}
function libraryError(error) {
  $("notice-text").textContent = error.message;
  $("undo-remove").hidden = true;
  $("library-notice").hidden = false;
}
async function removeBuilding(house) {
  if (libraryBusy) return;
  libraryBusy = true;
  updateActions();
  try {
    const removed = await api(`/api/houses/${encodeURIComponent(house.id)}`, {method: "DELETE"});
    selected.delete(house.id);
    if (activeHouse === house.id) clearResult();
    undoToken = removed.undo_token;
    const withModel = results.some(result => result.house_id === house.id);
    $("notice-text").textContent = `Building ${house.building_fid}${withModel ? " and its saved model" : ""} removed.`;
    $("undo-remove").hidden = false;
    $("library-notice").hidden = false;
    await load();
  } catch (error) { libraryError(error); }
  finally { libraryBusy = false; updateActions(); }
}
async function restoreBuilding(token) {
  if (libraryBusy) return;
  libraryBusy = true;
  updateActions();
  try {
    await api(`/api/removed-buildings/${encodeURIComponent(token)}/restore`, {method: "POST"});
    if (undoToken === token) undoToken = null;
    $("notice-text").textContent = "Building restored.";
    $("undo-remove").hidden = true;
    $("library-notice").hidden = false;
    await load();
  } catch (error) { libraryError(error); }
  finally { libraryBusy = false; updateActions(); }
}
$("undo-remove").onclick = () => { if (undoToken) restoreBuilding(undoToken); };
$("dismiss-notice").onclick = () => { $("library-notice").hidden = true; };
$("find-first").onclick = () => tab("select");

async function reconstruct(ids, options = {}) {
  if (processing || libraryBusy) return;
  processing = true;
  updateActions();
  try {
    $("progress").className = "";
    const job = await api("/api/reconstructions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        houses: ids,
        ...options,
      }),
    });
    localStorage.setItem("emboss-active-job", job.id);
    await follow(job.id);
  } catch (error) {
    showError(error);
  } finally {
    processing = false;
    updateActions();
  }
}
$("run").onclick = () => reconstruct([...selected], {
  force: $("force").checked,
});

function closeGallery() {
  reviewVersion++;
  gallery.clear();
  review = null;
  $("image-dialog").close();
}
$("close-gallery").onclick = closeGallery;
$("cancel-gallery").onclick = closeGallery;
$("image-dialog").addEventListener("cancel", event => {
  event.preventDefault();
  closeGallery();
});
$("gallery-footprint").onchange = () => {
  $("image-gallery").classList.toggle("hide-footprints", !$("gallery-footprint").checked);
};
function updateGalleryAction() {
  if (!review) return;
  $("confirm-gallery").disabled = processing || !review.info || !gallery.ready.has(review.choices[review.house]);
  $("confirm-gallery").textContent = review.mode === "change" ? "Use image & rerun" :
    review.ids.length === 1 ? "Confirm & model" : `Confirm & model ${review.ids.length} buildings`;
  const count = Object.keys(review.choices).length;
  $("gallery-note").textContent = review.mode === "change" ?
    "Rerunning updates the segmentation and 3D model." : review.ids.length > 1 ?
    `${count} of ${review.ids.length} reviewed. Unreviewed buildings use their existing or automatic choice.` :
    "Next: reconstruct the roof and segment its details.";
}
async function openGallery(ids, mode) {
  if (processing || libraryBusy || !ids.length) return;
  closeGallery();
  review = {ids, mode, choices: {}, cache: new Map(), house: null, info: null};
  $("cancel-gallery").textContent = "Cancel";
  const state = review;
  $("image-dialog").showModal();
  // House IDs remain a fallback while human-readable building numbers load.
  try { await load(); } catch (error) { showError(error); }
  if (review !== state) return;
  $("gallery-buildings").hidden = ids.length < 2;
  $("gallery-buildings").replaceChildren(...ids.map(id => {
    const button = document.createElement("button");
    const house = houses.find(item => item.id === id);
    button.textContent = `Building ${house?.building_fid || id}`;
    button.dataset.house = id;
    button.onclick = () => reviewHouse(id);
    return button;
  }));
  await reviewHouse(ids[0]);
}
async function reviewHouse(id) {
  const version = ++reviewVersion;
  const state = review;
  gallery.clear();
  state.house = id;
  state.info = null;
  $("retry-gallery").hidden = true;
  const house = houses.find(item => item.id === id);
  $("gallery-title").textContent = `Choose an orthophoto · Building ${house?.building_fid || id}`;
  for (const button of $("gallery-buildings").children) {
    button.classList.toggle("active", button.dataset.house === id);
    button.setAttribute("aria-pressed", String(button.dataset.house === id));
  }
  $("gallery-status").className = "";
  $("gallery-status").textContent = "Preparing available images… You can leave this open while they load.";
  updateGalleryAction();
  try {
    let info = state.cache.get(id);
    if (!info) {
      const job = await api(`/api/houses/${encodeURIComponent(id)}/imagery`, {method: "POST"});
      for (;;) {
        if (version !== reviewVersion) return;
        const status = await api(`/api/imagery-jobs/${job.id}`);
        if (version !== reviewVersion) return;
        if (status.status === "completed") { info = status.result; break; }
        if (["failed", "interrupted"].includes(status.status)) throw Error(status.error);
        $("gallery-status").textContent = status.messages.at(-1) || "Preparing available images…";
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      state.cache.set(id, info);
    }
    if (version !== reviewVersion) return;
    state.info = info;
    state.choices[id] ??= info.selected_id;
    gallery.show(id, info, state.choices[id]);
    updateGalleryAction();
  } catch (error) {
    if (version !== reviewVersion) return;
    $("gallery-status").textContent = error.message;
    $("gallery-status").className = "error";
    $("retry-gallery").hidden = false;
  }
}
$("retry-gallery").onclick = () => { if (review) reviewHouse(review.house); };
$("confirm-gallery").onclick = () => {
  if (!review?.info || processing) return;
  const {ids, choices, mode} = review;
  closeGallery();
  selected = new Set(ids);
  tab("results");
  reconstruct(ids, {image_choices: choices, force: mode === "change"});
};
$("change-image").onclick = () => {
  if (activeHouse) openGallery([activeHouse], "change").catch(showError);
};
$("show-segmentation").onchange = () => {
  $("segmentation").hidden = !$("show-segmentation").checked;
};
function showFootprint(footprint) {
  drawFootprint($("footprint"), footprint);
  $("footprint").toggleAttribute("hidden", !footprint || !$("show-footprint").checked);
  $("show-footprint").disabled = !footprint;
}
$("show-footprint").onchange = () => showFootprint(imageInfo?.footprint);

function showImagery(id, info, buildingFid) {
  imageInfo = info;
  showFootprint(info.footprint);
  const base = `/api/results/${encodeURIComponent(id)}`;
  const version = Date.now();
  $("orthophoto").alt = `Orthophoto of building ${buildingFid}`;
  $("orthophoto").src = `${base}/orthophoto.png?v=${version}`;
  $("segmentation").src = `${base}/segmentation.png?v=${version}`;
  $("segmentation").hidden = !$("show-segmentation").checked;
  $("open-orthophoto").href = `${base}/orthophoto.png`;
  $("download-segmentation").href = `${base}/artifacts/segmentation`;
  const source = info.candidates.find(item => item.id === info.selected_id);
  const label = source?.kind === "raw" ? "Uncorrected image" : `Flight strip ${info.selected_id || "unknown"}`;
  $("image-source").textContent = [label, info.source_year, info.override_id ? "Your choice" : "Automatic choice"].filter(Boolean).join(" · ");
  $("segmentation-legend").replaceChildren(...info.classes.map(item => {
    const label = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = item.color;
    label.append(swatch, document.createTextNode(item.label));
    return label;
  }));
  $("imagery-panel").hidden = false;
  updateActions();
}
async function follow(id) {
  for (;;) {
    const job = await api("/api/reconstructions/" + id);
    $("progress").textContent =
      job.messages.slice(-5).join("\n") || "Starting reconstruction…";
    if (job.status === "completed") {
      localStorage.removeItem("emboss-active-job");
      await load();
      const result = job.result.results[0];
      if (result) { await view(result.house_id); revealResult(); }
      if (job.result.failures.length)
        showError(
          Error(
            job.result.failures
              .map((f) => `${f.house_id}: ${f.error}`)
              .join("\n"),
          ),
        );
      return;
    }
    if (["failed", "interrupted"].includes(job.status)) {
      localStorage.removeItem("emboss-active-job");
      throw Error(job.error);
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
function setup() {
  const canvas = $("viewer");
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor("#eaf0eb");
  canvas.appendChild(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", "3D roof model. Drag to orbit; use arrow keys to pan.");
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000);
  camera.up.set(0, 0, 1);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.listenToKeyEvents(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x66725b, 2.5));
  const light = new THREE.DirectionalLight(0xffffff, 2.5);
  light.position.set(50, -30, 90);
  scene.add(light);
  new ResizeObserver(resize).observe(canvas);
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}
function resize() {
  if (!renderer) return;
  const target = $("viewer");
  renderer.setSize(target.clientWidth, target.clientHeight);
  camera.aspect = target.clientWidth / target.clientHeight;
  camera.updateProjectionMatrix();
}
async function view(id) {
  const version = ++viewVersion;
  $("view-status").hidden = false;
  $("view-status").className = "";
  $("view-status").textContent = "Loading model…";
  $("retry-view").hidden = true;
  let data, summary, imagery;
  try { [data, summary, imagery] = await Promise.all([
    api(`/api/results/${id}/mesh`),
    api(`/api/results/${id}`),
    api(`/api/results/${id}/imagery`),
  ]); } catch (error) {
    if (version === viewVersion) {
      $("view-status").className = "error";
      $("view-status").textContent = "Could not open this model. Try again.";
      $("retry-view").hidden = false;
      $("retry-view").onclick = () => view(id).then(revealResult).catch(showError);
    }
    throw error;
  }
  if (version !== viewVersion) return;
  $("view-status").hidden = true;
  activeHouse = id;
  sessionStorage.setItem("emboss-building", id);
  for (const row of $("houses").children) row.classList.toggle("current", row.dataset.house === id);
  if (!renderer) setup();
  if (current) {
    scene.remove(current);
    current.traverse((object) => {
      object.geometry?.dispose();
      if (object.material) object.material.dispose();
    });
  }
  const vertices = data.vertices;
  const min = [0, 1, 2].map((d) => Math.min(...vertices.map((v) => v[d]))),
    max = [0, 1, 2].map((d) => Math.max(...vertices.map((v) => v[d])));
  const center = min.map((v, i) => (v + max[i]) / 2),
    positions = [];
  for (const face of data.faces) {
    for (let i = 1; i < face.length - 1; i++) {
      for (const j of [face[0], face[i], face[i + 1]])
        positions.push(...vertices[j].map((v, d) => v - center[d]));
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.computeVertexNormals();
  current = new THREE.Group();
  current.add(
    new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0xc1ccaf,
        roughness: 0.8,
        side: THREE.DoubleSide,
      }),
    ),
  );
  current.add(
    new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 20),
      new THREE.LineBasicMaterial({
        color: 0x425e45,
        transparent: true,
        opacity: 0.55,
      }),
    ),
  );
  scene.add(current);
  const size = Math.max(...max.map((v, i) => v - min[i]), 5);
  modelRadius = Math.sqrt(max.reduce((sum, v, i) => sum + (v - min[i]) ** 2, 0)) / 2;
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  $("empty").hidden = true;
  $("model-help").hidden = false;
  $("title").textContent = `Building ${summary.provenance.input.building_fid}`;
  $("summary").textContent =
    `${summary.solid_count} reconstructed roof details`;
  $("download").hidden = false;
  $("download").href = `/api/results/${id}/download`;
  $("detail-list").textContent = "";
  const classes = {};
  for (const feature of data.details.features) {
    const label = feature.properties.class_label || "Roof detail";
    classes[label] = (classes[label] || 0) + 1;
  }
  $("detail-list").textContent = Object.entries(classes)
    .map(([name, count]) => `${count} ${name}`)
    .join(" · ");
  showImagery(id, imagery, summary.provenance.input.building_fid);
  resize();
  resetView();
}
function resetView() {
  if (!camera || !current) return;
  const halfFov = Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(camera.aspect, 1));
  const distance = Math.max(modelRadius, 1) / Math.sin(halfFov) * 1.12;
  camera.position.copy(new THREE.Vector3(1, -1, .8).normalize().multiplyScalar(distance));
  controls.target.set(0, 0, 0);
  controls.update();
}
$("reset-view").onclick = resetView;
const pending = localStorage.getItem("emboss-active-job");
if (pending) {
  processing = true;
  tab("results");
  follow(pending).catch(showError).finally(() => {
    processing = false;
    updateActions();
  });
} else if (sessionStorage.getItem("emboss-view") === "results") tab("results");
else load().catch(showError);
