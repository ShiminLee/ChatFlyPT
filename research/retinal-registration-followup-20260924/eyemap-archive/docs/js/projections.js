// Spherical projection math for eyemap viewing-direction vectors (x,y,z on
// the unit sphere). cart2sph/sph2Mollweide are a direct port of the Python
// functions in connectome_interpreter's external_map.py
// (plot_mollweide_projection), which already renders this exact dataset.
// Mercator has no prior art in that codebase and is implemented here from
// the standard formula.

// Cartesian unit vector -> spherical angles, theta from +z in [0, pi],
// phi (azimuth) in [0, 2*pi).
function cart2sph(x, y, z) {
  const theta = Math.acos(Math.max(-1, Math.min(1, z)));
  let phi = Math.atan2(y, x);
  if (phi < 0) phi += 2 * Math.PI;
  return { theta, phi };
}

// Equal-area Mollweide projection. theta/phi as returned by cart2sph.
function sph2Mollweide(theta, phi) {
  let azim = phi;
  if (azim > Math.PI) azim -= 2 * Math.PI; // longitude in [-pi, pi]
  const elev = Math.PI / 2 - theta; // latitude in [-pi/2, pi/2]

  let t;
  if (Math.abs(Math.abs(elev) - Math.PI / 2) < 1e-9) {
    t = Math.sign(elev) * (Math.PI / 2);
  } else {
    t = Math.asin((2 * elev) / Math.PI);
    let dtheta = Infinity;
    let iter = 0;
    while (dtheta > 1e-6 && iter < 100) {
      const tNew =
        t -
        (2 * t + Math.sin(2 * t) - Math.PI * Math.sin(elev)) /
          (2 + 2 * Math.cos(2 * t));
      dtheta = Math.abs(tNew - t);
      t = tNew;
      iter++;
    }
  }
  return {
    x: -((2 * Math.sqrt(2)) / Math.PI) * azim * Math.cos(t), // x-axis flipped, matching source
    y: Math.sqrt(2) * Math.sin(t),
  };
}

// Mercator projection. Latitude is clamped away from the poles since the
// formula diverges there (not a concern for this dataset's viewing-direction
// range, but keeps the math well-defined for arbitrary input).
function sph2Mercator(theta, phi) {
  let lon = phi;
  if (lon > Math.PI) lon -= 2 * Math.PI;
  const maxLat = (89.9 * Math.PI) / 180;
  const lat = Math.max(-maxLat, Math.min(maxLat, Math.PI / 2 - theta));
  return {
    x: lon,
    y: Math.log(Math.tan(Math.PI / 4 + lat / 2)),
  };
}

// Meridian/parallel guide lines every `stepDeg` degrees, projected with
// `projectFn(theta, phi) -> {x, y}`. Returns an array of {x: [...], y: [...]}
// polylines (NaN-separated within a single trace is left to the caller).
function buildGraticule(projectFn, stepDeg = 45) {
  const lines = [];
  const toRad = (d) => (d * Math.PI) / 180;

  // Meridians: constant longitude, varying latitude.
  for (let lonDeg = -180; lonDeg <= 180; lonDeg += stepDeg) {
    const xs = [];
    const ys = [];
    for (let latDeg = -90; latDeg <= 90; latDeg += 2) {
      const theta = Math.PI / 2 - toRad(latDeg);
      let phi = toRad(lonDeg);
      if (phi < 0) phi += 2 * Math.PI;
      const p = projectFn(theta, phi);
      xs.push(p.x);
      ys.push(p.y);
    }
    lines.push({ x: xs, y: ys });
  }

  // Parallels: constant latitude, varying longitude.
  for (let latDeg = -90; latDeg <= 90; latDeg += stepDeg) {
    const xs = [];
    const ys = [];
    for (let lonDeg = -180; lonDeg <= 180; lonDeg += 2) {
      const theta = Math.PI / 2 - toRad(latDeg);
      let phi = toRad(lonDeg);
      if (phi < 0) phi += 2 * Math.PI;
      const p = projectFn(theta, phi);
      xs.push(p.x);
      ys.push(p.y);
    }
    lines.push({ x: xs, y: ys });
  }

  return lines;
}
