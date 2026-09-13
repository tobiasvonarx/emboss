// One image chooser for acquisition and existing reconstructions.
export function drawFootprint(svg, footprint) {
  svg.replaceChildren();
  if (!footprint) return;
  svg.setAttribute("viewBox", `0 0 ${footprint.width} ${footprint.height}`);
  const d = footprint.rings.map(ring => ring.map(([x, y], i) =>
    `${i ? "L" : "M"}${x},${y}`).join(" ") + " Z").join(" ");
  for (const style of ["footprint-halo", "footprint-line"]) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("class", style);
    svg.append(path);
  }
}
export class ImageGallery {
  constructor(container, status, onSelect) {
    this.container = container;
    this.status = status;
    this.onSelect = onSelect;
    this.urls = [];
    this.ready = new Set();
  }
  clear() {
    this.controller?.abort();
    this.urls.forEach(url => URL.revokeObjectURL(url));
    this.urls = [];
    this.ready = new Set();
    this.container.replaceChildren();
    this.status.textContent = "";
  }
  show(house, info, selected) {
    this.clear();
    this.controller = new AbortController();
    const {signal} = this.controller;
    this.selected = selected;
    const rows = [...info.candidates].sort((a, b) =>
      Number(b.id === info.selected_id) - Number(a.id === info.selected_id));
    let loaded = 0, failed = 0;
    const buttons = [];
    const update = () => {
      if (signal.aborted) return;
      this.status.textContent = `${loaded} of ${rows.length} images ready` +
        (failed ? ` · ${failed} unavailable — retry below` : loaded < rows.length ? " · Preparing the rest…" : "");
    };
    const choose = id => {
      this.selected = id;
      for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.candidate === id));
      this.onSelect(id);
    };
    const tasks = rows.map(item => {
      const card = document.createElement("div");
      card.className = "image-card";
      const button = document.createElement("button");
      button.className = "image-option";
      button.dataset.candidate = item.id;
      button.setAttribute("aria-pressed", String(item.id === selected));
      button.disabled = true;
      const label = item.kind === "raw" ? "Uncorrected image" : item.flight_date || String(item.year || "Aerial image");
      button.setAttribute("aria-label", `Choose ${label} (${item.id})`);
      const frame = document.createElement("div");
      frame.className = "image-stack";
      const image = document.createElement("img");
      image.alt = label;
      image.hidden = true;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "footprint-overlay");
      svg.setAttribute("aria-hidden", "true");
      drawFootprint(svg, info.footprint);
      svg.setAttribute("hidden", "");
      const waiting = document.createElement("span");
      waiting.className = "image-loading";
      waiting.textContent = "Preparing image…";
      frame.append(image, svg, waiting);
      const caption = document.createElement("div");
      caption.className = "image-caption";
      const name = document.createElement("strong");
      name.textContent = label;
      const badge = document.createElement("span");
      badge.className = "image-badge";
      badge.textContent = [
        item.id === info.current_id && info.has_reconstruction ? "Current" : "",
        item.id === (info.recommended_id || info.selected_id) ? "Automatic choice" : ""
      ].filter(Boolean).join(" · ");
      const detail = document.createElement("small");
      detail.textContent = item.kind === "raw" ? "Original aerial imagery" : `Corrected · ${item.id}`;
      caption.append(name, badge, detail);
      button.append(frame, caption);
      button.onclick = () => choose(item.id);
      buttons.push(button);
      const retry = document.createElement("button");
      retry.className = "retry-image";
      retry.textContent = "Retry image";
      retry.hidden = true;
      card.append(button, retry);
      this.container.append(card);
      const load = async () => {
        retry.hidden = true;
        waiting.textContent = "Preparing image…";
        try {
          const response = await fetch(`/api/houses/${encodeURIComponent(house)}/imagery/preview?candidate_id=${encodeURIComponent(item.id)}`, {signal});
          if (!response.ok) {
            const error = await response.json();
            throw Error(error.detail || "Image could not be prepared");
          }
          const blob = await response.blob();
          if (signal.aborted) return;
          const url = URL.createObjectURL(blob);
          this.urls.push(url);
          image.src = url;
          await image.decode();
          if (signal.aborted) return;
          image.hidden = false;
          svg.removeAttribute("hidden");
          waiting.hidden = true;
          button.disabled = false;
          loaded++;
          this.ready.add(item.id);
          if (this.selected === item.id) this.onSelect(item.id);
        } catch (error) {
          if (signal.aborted) return;
          failed++;
          waiting.textContent = error.message;
          retry.hidden = false;
        }
        update();
      };
      retry.onclick = () => { failed--; load(); };
      return load;
    });
    // Show all cards immediately; keep expensive correction requests bounded.
    const worker = async () => {
      while (tasks.length && !signal.aborted) await tasks.shift()();
    };
    update();
    void worker();
    void worker();
  }
}
