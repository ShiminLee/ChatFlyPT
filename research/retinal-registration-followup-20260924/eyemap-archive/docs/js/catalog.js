(async function () {
  const container = document.getElementById("cards");

  let manifest;
  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifest = await res.json();
  } catch (err) {
    container.innerHTML = `<p class="info-note">Could not load manifest.json (${err.message}).</p>`;
    return;
  }

  if (!manifest.datasets || manifest.datasets.length === 0) {
    container.innerHTML = `<p class="info-note">No datasets found.</p>`;
    return;
  }

  container.innerHTML = "";
  for (const ds of manifest.datasets) {
    container.appendChild(buildCard(ds));
  }

  function buildCard(ds) {
    const card = document.createElement("div");
    card.className = "card";

    const badges = [
      `<span class="badge">${ds.source}</span>`,
      ds.region ? `<span class="badge">${ds.region}</span>` : "",
      ds.has_rootid ? `<span class="badge">rootid</span>` : "",
    ].join("");

    const sides = Object.entries(ds.hemispheres || {});
    const sideRows = sides
      .map(([side, info]) => {
        const links = ["xlsx", "csv", "json"]
          .filter((fmt) => info.files[fmt])
          .map((fmt) => `<a class="btn" href="${info.files[fmt]}">${fmt}</a>`)
          .join(" ");
        return `<div class="btn-row">
          <span style="min-width:70px; color: var(--muted); font-size:0.85rem;">
            ${side} (${info.n_points})
          </span>
          ${links}
          <a class="btn primary" href="view.html?dataset=${encodeURIComponent(ds.id)}&side=${side}">View in 3D</a>
        </div>`;
      })
      .join("");

    const rdaLink = ds.rda_file
      ? `<div class="btn-row"><a class="btn" href="${ds.rda_file}">lens_med.rda</a></div>`
      : "";

    card.innerHTML = `
      <h3>${ds.title}</h3>
      <div class="meta"><span>${ds.date}</span><span>${ds.id}</span></div>
      <div>${badges}</div>
      ${sideRows}
      ${rdaLink}
    `;
    return card;
  }
})();
