const $ = (id) => document.getElementById(id);
const map = L.map("map").setView([46.95, 7.45], 12);
let baseTiles = null, activeDataset = null, coverageLayer = null, coverageBounds = null;
let datasetList = [], footprintSequence = 0;
const footprints = L.geoJSON(null, {
  style: { color: "#b37415", weight: 1.5, fillColor: "#efbc61", fillOpacity: .18 },
}).addTo(map);
// Keep reference labels above the imagery and below selection outlines/markers.
const labelsPane = map.createPane("labels");
labelsPane.style.zIndex = 350;
labelsPane.style.pointerEvents = "none";
const placeNames = L.tileLayer(
  "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
  {
    pane: "labels",
    maxZoom: 20,
    attribution:
      'Labels: <a href="https://www.arcgis.com/home/item.html?id=a842e359856a4365b1ddf8cc34fde079">Esri</a>, HERE, Garmin, © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, and the GIS user community',
  },
).addTo(map);
L.control.layers({}, { "Place names": placeNames }, { collapsed: false }).addTo(map);
let mode = "house",
  selection = null,
  layer = null,
  start = null,
  busy = false;
function clear() {
  if (layer) map.removeLayer(layer);
  layer = null;
  selection = null;
  start = null;
  $("acquire").disabled = true;
  $("clear-selection").hidden = true;
}
function setMode(value) {
  if (busy) return;
  mode = value;
  clear();
  $("house-mode").classList.toggle("selected", value === "house");
  $("area-mode").classList.toggle("selected", value === "area");
  $("house-mode").setAttribute("aria-pressed", String(value === "house"));
  $("area-mode").setAttribute("aria-pressed", String(value === "area"));
  $("hint").textContent =
    value === "house"
      ? "Click a roof, or pan the map and press Enter."
      : "Choose two opposite corners. Use clicks or pan and press Enter.";
  $("acquire").textContent =
    value === "house" ? "Prepare building" : "Prepare area";
}
$("house-mode").onclick = () => setMode("house");
$("area-mode").onclick = () => setMode("area");
function choosePoint(p) {
  if (busy || !activeDataset) return;
  if (coverageBounds && !coverageBounds.contains(p)) {
    clear();
    $("hint").textContent = "Choose a point inside the coverage outline.";
    return;
  }
  if (mode === "house") {
    clear();
    layer = L.marker(p).addTo(map);
    selection = { mode, provider: activeDataset.id, longitude: p.lng, latitude: p.lat };
    $("hint").textContent = `Selected ${p.lat.toFixed(5)}, ${p.lng.toFixed(5)}`;
  } else if (!start) {
    clear();
    start = p;
    layer = L.rectangle([p, p], { color: "#ecb951", weight: 2 }).addTo(map);
    $("hint").textContent = "Choose the opposite corner.";
    $("clear-selection").hidden = false;
    return;
  } else {
    const bounds = L.latLngBounds(start, p);
    layer.setBounds(bounds);
    if (bounds.getWest() === bounds.getEast() || bounds.getSouth() === bounds.getNorth()) {
      $("hint").textContent = "Move diagonally across the area, then choose the opposite corner.";
      $("acquire").disabled = true;
      return;
    }
    selection = {
      mode,
      provider: activeDataset.id,
      bbox: [
        bounds.getWest(),
        bounds.getSouth(),
        bounds.getEast(),
        bounds.getNorth(),
      ],
    };
    start = null;
    $("hint").textContent =
      "Every building intersecting the highlighted area will be prepared.";
  }
  $("acquire").disabled = false;
  $("clear-selection").hidden = false;
}
map.on("click", event => choosePoint(event.latlng));
$("map").addEventListener("keydown", event => {
  if (event.key === "Enter" && event.target === $("map")) {
    event.preventDefault();
    choosePoint(map.getCenter());
  }
});
$("clear-selection").onclick = () => { if (!busy) setMode(mode); };
new ResizeObserver(() => map.invalidateSize()).observe($("map"));
map.on("mousemove", (event) => {
  if (start && layer) layer.setBounds(L.latLngBounds(start, event.latlng));
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
function clearSuggestions() {
  $("suggestions").replaceChildren();
  $("search").setAttribute("aria-expanded", "false");
}
$("search").addEventListener("keydown", event => {
  if (event.key === "Escape") { searchSequence++; clearSuggestions(); }
  if (event.key === "ArrowDown") { event.preventDefault(); $("suggestions").querySelector("button")?.focus(); }
});
$("suggestions").addEventListener("keydown", event => {
  const buttons = [...$("suggestions").querySelectorAll("button")];
  const index = buttons.indexOf(document.activeElement);
  if (event.key === "Escape") { clearSuggestions(); $("search").focus(); }
  if (["ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault();
    buttons[(index + (event.key === "ArrowDown" ? 1 : buttons.length - 1)) % buttons.length]?.focus();
  }
});
let searchTimer,
  searchSequence = 0;
$("search").oninput = () => {
  clearTimeout(searchTimer);
  const seq = ++searchSequence;
  clearSuggestions();
  $("search-status").hidden = true;
  if ($("search").value.trim().length < 2) return;
  searchTimer = setTimeout(async () => {
    try {
      const found = await api(
        "/api/search?q=" + encodeURIComponent($("search").value) + "&provider=" + encodeURIComponent(activeDataset.id),
      );
      if (seq !== searchSequence) return;
      $("search").setAttribute("aria-expanded", String(found.length > 0));
      $("search-status").textContent = found.length ? "" : "No matching addresses. Try a town or select on the map.";
      $("search-status").hidden = found.length > 0;
      $("suggestions").replaceChildren(
        ...found.map((item) => {
          const button = document.createElement("button");
          const doc = new DOMParser().parseFromString(item.label, "text/html");
          button.textContent = doc.body.textContent;
          button.onclick = () => {
            map.setView([item.latitude, item.longitude], 19);
            $("search").value = button.textContent;
            clearSuggestions();
            $("map").focus();
          };
          return button;
        }),
      );
    } catch (error) {
      if (seq !== searchSequence) return;
      $("search-status").hidden = false;
      $("search-status").textContent = "Address search is unavailable. Select on the map or try again.";
    }
  }, 350);
};
function setBusy(value) {
  busy = value;
  for (const id of ["dataset", "house-mode", "area-mode", "search", "clear-selection"]) $(id).disabled = value;
  $("search").disabled = value || !activeDataset?.search;
  $("acquire").disabled = value || !selection;
  $("acquire").textContent = value ? "Preparing…" : mode === "house" ? "Prepare building" : "Prepare area";
  $("working").hidden = !value;
}
$("acquire").onclick = async () => {
  if (!selection || busy) return;
  setBusy(true);
  $("summary").textContent = "";
  $("status").className = "";
  try {
    const job = await api("/api/acquisition", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selection),
    });
    localStorage.setItem("building-data-active-job", job.id);
    await follow(job.id);
  } catch (error) {
    $("status").textContent = error.message;
    $("status").className = "error";
  } finally {
    setBusy(false);
  }
};
async function follow(id) {
  for (;;) {
    const state = await api("/api/acquisition/" + id);
    $("status").textContent =
      state.messages.at(-1) || "Preparing data…";
    if (state.status === "completed") {
      const result = state.result;
      const failures = result.failures || [];
      $("summary").textContent =
        `${result.houses.length} ${result.houses.length === 1 ? "building" : "buildings"} ready${failures.length ? `; ${failures.length} could not be prepared` : ""}.`;
      if (failures.length)
        $("status").textContent = failures
          .map((f) => `Building ${f.building_fid}: ${f.error}`)
          .join("\n");
      parent.postMessage(
        {
          type: "building-data-acquired",
          acquisition_id: result.id,
          houses: result.houses,
        },
        location.origin,
      );
      localStorage.removeItem("building-data-active-job");
      return;
    }
    if (["failed", "interrupted"].includes(state.status)) {
      localStorage.removeItem("building-data-active-job");
      throw Error(state.error);
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}
async function refreshFootprints() {
  const seq = ++footprintSequence;
  footprints.clearLayers();
  if (!activeDataset?.coverage || map.getZoom() < 16) return;
  const bounds = map.getBounds();
  try {
    const data = await api(`/api/datasets/${activeDataset.id}/footprints?bbox=${bounds.toBBoxString()}`);
    if (seq === footprintSequence) footprints.addData(data);
  } catch (error) {
    if (seq === footprintSequence) $("status").textContent = "Building outlines could not load. Try moving the map.";
  }
}
map.on("moveend", refreshFootprints);
function selectDataset(id) {
  const dataset = datasetList.find(item => item.id === id) || datasetList[0];
  activeDataset = dataset;
  map.setMinZoom(dataset.minZoom || 0);
  footprintSequence++;
  footprints.clearLayers();
  searchSequence++;
  clearTimeout(searchTimer);
  clearSuggestions();
  $("search").value = "";
  $("search").disabled = !dataset.search;
  $("search").closest(".search").hidden = !dataset.search;
  $("search").placeholder = dataset.search ? "Address or town" : "Use the map to select Cambridge buildings";
  $("search-status").hidden = true;
  $("dataset").value = dataset.id;
  $("dataset-description").textContent = dataset.description;
  $("source-credit").textContent = dataset.footprint_attribution || "";
  $("source-credit").hidden = !dataset.footprint_attribution;
  if (baseTiles) map.removeLayer(baseTiles);
  baseTiles = L.tileLayer(dataset.tiles, { minZoom: dataset.minZoom || 0, maxZoom: 20, attribution: dataset.attribution }).addTo(map);
  if (coverageLayer) map.removeLayer(coverageLayer);
  coverageLayer = dataset.coverage ? L.geoJSON(dataset.coverage, {
    interactive: false, style: { color: "#215b44", weight: 2, dashArray: "6 5", fillOpacity: .025 },
  }).addTo(map) : null;
  coverageBounds = coverageLayer?.getBounds() || null;
  setMode(mode);
  $("status").textContent = "";
  $("summary").textContent = "";
  if (coverageBounds) map.fitBounds(coverageBounds, { padding: [20, 20] });
  else map.setView(dataset.center, dataset.zoom);
  localStorage.setItem("building-data-dataset", dataset.id);
}
$("dataset").onchange = () => selectDataset($("dataset").value);
async function initialize() {
  datasetList = await api("/api/datasets");
  $("dataset").replaceChildren(...datasetList.map(dataset => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.label;
    return option;
  }));
  $("dataset").disabled = false;
  selectDataset(localStorage.getItem("building-data-dataset"));
  const pending = localStorage.getItem("building-data-active-job");
  if (pending) {
    setBusy(true);
    try { await follow(pending); }
    finally { setBusy(false); }
  }
}
initialize().catch(error => {
  $("status").textContent = error.message;
  $("status").className = "error";
});
