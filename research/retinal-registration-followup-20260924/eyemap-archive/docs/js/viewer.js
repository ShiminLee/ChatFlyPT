(async function () {
  const datasetSelect = document.getElementById("dataset-select");
  const sideSelect = document.getElementById("side-select");
  const modeTabs = document.getElementById("mode-tabs");
  const statusEl = document.getElementById("status");
  const plotEl = document.getElementById("plot");

  let manifest;
  let mode = "scatter3d";
  let currentPoints = null; // cache to avoid re-fetching on mode switch

  const params = new URLSearchParams(location.search);

  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifest = await res.json();
  } catch (err) {
    statusEl.textContent = `Could not load manifest.json (${err.message}).`;
    return;
  }

  if (!manifest.datasets || manifest.datasets.length === 0) {
    statusEl.textContent = "No datasets found.";
    return;
  }

  for (const ds of manifest.datasets) {
    const opt = document.createElement("option");
    opt.value = ds.id;
    opt.textContent = `${ds.title} (${ds.date})`;
    datasetSelect.appendChild(opt);
  }

  const requestedDataset = params.get("dataset");
  if (requestedDataset && manifest.datasets.some((d) => d.id === requestedDataset)) {
    datasetSelect.value = requestedDataset;
  }

  populateSides();
  const requestedSide = params.get("side");
  if (requestedSide && [...sideSelect.options].some((o) => o.value === requestedSide)) {
    sideSelect.value = requestedSide;
  }

  datasetSelect.addEventListener("change", () => {
    populateSides();
    loadAndRender();
  });
  sideSelect.addEventListener("change", loadAndRender);

  modeTabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-mode]");
    if (!btn) return;
    mode = btn.dataset.mode;
    for (const b of modeTabs.querySelectorAll("button")) {
      b.classList.toggle("active", b === btn);
    }
    if (currentPoints) render(currentPoints);
  });

  loadAndRender();

  function currentDataset() {
    return manifest.datasets.find((d) => d.id === datasetSelect.value);
  }

  function populateSides() {
    const ds = currentDataset();
    const prev = sideSelect.value;
    sideSelect.innerHTML = "";
    for (const side of Object.keys(ds.hemispheres)) {
      const opt = document.createElement("option");
      opt.value = side;
      opt.textContent = `${side} (${ds.hemispheres[side].n_points} pts)`;
      sideSelect.appendChild(opt);
    }
    if ([...sideSelect.options].some((o) => o.value === prev)) {
      sideSelect.value = prev;
    }
  }

  async function loadAndRender() {
    const ds = currentDataset();
    const side = sideSelect.value;
    const url = new URL(location.href);
    url.searchParams.set("dataset", ds.id);
    url.searchParams.set("side", side);
    history.replaceState(null, "", url);

    statusEl.textContent = "Loading…";
    try {
      const res = await fetch(ds.hemispheres[side].files.json);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      currentPoints = await res.json();
      statusEl.textContent = `${ds.title} — ${side} (${currentPoints.length} points)`;
      render(currentPoints);
    } catch (err) {
      statusEl.textContent = `Could not load data (${err.message}).`;
    }
  }

  function hoverText(d) {
    let t = `p=${d.p}, q=${d.q}<br>theta=${d.theta}°, phi=${d.phi}°`;
    if (d.rootid) t += `<br>rootid=${d.rootid}`;
    return t;
  }

  function render(points) {
    const v = points.map((d) => d.p + d.q);
    const text = points.map(hoverText);
    const commonMarker = {
      size: mode === "scatter3d" ? 3 : 5,
      color: v,
      colorscale: "Viridis",
      colorbar: { title: "p + q" },
    };

    if (mode === "scatter3d") {
      Plotly.newPlot(
        plotEl,
        [
          {
            type: "scatter3d",
            mode: "markers",
            x: points.map((d) => d.x),
            y: points.map((d) => d.y),
            z: points.map((d) => d.z),
            text,
            hoverinfo: "text",
            marker: commonMarker,
          },
        ],
        {
          paper_bgcolor: "transparent",
          font: { color: "#e6e8eb" },
          margin: { l: 0, r: 0, t: 20, b: 0 },
          scene: {
            xaxis: { title: "x (forward)" },
            yaxis: { title: "y (left)" },
            zaxis: { title: "z (up / dorsal)" },
          },
        },
        { responsive: true }
      );
      return;
    }

    const projectFn = mode === "mollweide" ? sph2Mollweide : sph2Mercator;
    const proj = points.map((d) => {
      const { theta, phi } = cart2sph(d.x, d.y, d.z);
      return projectFn(theta, phi);
    });

    const graticule = buildGraticule(projectFn).map((line) => ({
      type: "scatter",
      mode: "lines",
      x: line.x,
      y: line.y,
      line: { color: "#3a4048", width: 1 },
      hoverinfo: "skip",
      showlegend: false,
    }));

    const dataTrace = {
      type: "scatter",
      mode: "markers",
      x: proj.map((p) => p.x),
      y: proj.map((p) => p.y),
      text,
      hoverinfo: "text",
      marker: commonMarker,
      showlegend: false,
    };

    const xaxis =
      mode === "mollweide"
        ? { range: [-Math.PI, Math.PI], visible: false }
        : { range: [-Math.PI, Math.PI], visible: false };
    const yaxis =
      mode === "mollweide"
        ? { range: [-Math.PI / 2, Math.PI / 2], visible: false }
        : { visible: false };

    Plotly.newPlot(
      plotEl,
      [...graticule, dataTrace],
      {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#e6e8eb" },
        margin: { l: 10, r: 10, t: 20, b: 10 },
        xaxis: { ...xaxis, scaleanchor: "y", scaleratio: 1 },
        yaxis,
      },
      { responsive: true }
    );
  }
})();
