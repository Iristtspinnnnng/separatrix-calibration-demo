/* fractal.js -- Newton basins of z^3 - 1, computed in the browser.
 *
 * Pure computation, no DOM: this file is loaded by index.html and can also be
 * required from Node for testing (it attaches itself to `window`, or to
 * `globalThis` when there is no window).
 *
 * The map is N(z) = z - (z^3 - 1)/(3 z^2); its attractors are the three cube
 * roots of unity, and their basins are separated by a fractal (the Julia set
 * of N).  Everything the page shows about the "truth" comes from here, at
 * whatever resolution the canvas has.
 */
(function (global) {
  "use strict";

  var SQRT3_2 = Math.sqrt(3) / 2;
  var ROOTS = [[1, 0], [-0.5, SQRT3_2], [-0.5, -SQRT3_2]];

  // return [label, steps]: label 1..3 for a basin, 0 if it escaped or stalled
  function labelAt(x, y, maxSteps) {
    maxSteps = maxSteps || 60;
    var zr = x, zi = y, it;
    for (it = 0; it < maxSteps; it++) {
      var zr2 = zr * zr, zi2 = zi * zi;
      // f(z) = z^3 - 1
      var fr = zr * (zr2 - 3 * zi2) - 1;
      var fi = zi * (3 * zr2 - zi2);
      // f'(z) = 3 z^2
      var dr = 3 * (zr2 - zi2);
      var di = 6 * zr * zi;
      var den = dr * dr + di * di;
      if (!isFinite(den) || den < 1e-300) { return [0, it]; }
      var nr = zr - (fr * dr + fi * di) / den;
      var ni = zi - (fi * dr - fr * di) / den;
      if (!isFinite(nr) || !isFinite(ni)) { return [0, it]; }
      zr = nr; zi = ni;
      var r2 = zr * zr + zi * zi;
      if (r2 > 1e6) { return [0, it]; }
      for (var k = 0; k < 3; k++) {
        var dx = zr - ROOTS[k][0], dy = zi - ROOTS[k][1];
        if (dx * dx + dy * dy < 1e-12) { return [k + 1, it]; }
      }
    }
    return [0, it];
  }

  /* Rasterise a window of the plane.
   * n     : pixels per axis
   * view  : { cx, cy, half }  (half = half width of the window)
   * out   : { label: Uint8Array(n*n), steps: Uint8Array(n*n) }
   * Pixel (col i, row j) is at x = cx - half + (i+0.5)*2*half/n,
   *                             y = cy + half - (j+0.5)*2*half/n  (y up).
   */
  function raster(n, view, out) {
    out = out || { label: new Uint8Array(n * n), steps: new Uint8Array(n * n) };
    var half = view.half, step = (2 * half) / n;
    var x0 = view.cx - half + step / 2;
    var y0 = view.cy + half - step / 2;
    var p = 0;
    for (var j = 0; j < n; j++) {
      var y = y0 - j * step;
      for (var i = 0; i < n; i++) {
        var res = labelAt(x0 + i * step, y);
        out.label[p] = res[0];
        out.steps[p] = res[1];
        p++;
      }
    }
    return out;
  }

  /* Basin colour with step shading, as [r,g,b] in 0..255.
   * Slow orbits (high step count) sit next to the fractal boundary: they are
   * brightened towards white so the structure reads at any zoom. */
  var BASE = [[59, 76, 192], [224, 176, 0], [192, 57, 43]];
  function shade(label, steps) {
    if (label < 1 || label > 3) { return [42, 46, 54]; }
    var b = BASE[label - 1];
    var t = Math.min(steps, 24) / 24;
    var lift = 0.18 + 0.82 * Math.sqrt(t);
    var mix = 0.55 * t * t;
    var clamp = function (v) { return v > 255 ? 255 : (v < 0 ? 0 : Math.round(v)); };
    return [
      clamp(b[0] * lift + 255 * mix * 0.55),
      clamp(b[1] * lift + 255 * mix * 0.55),
      clamp(b[2] * lift + 255 * mix * 0.55)
    ];
  }

  function fillImage(imageData, n, out) {
    var d = imageData.data, p, q;
    for (p = 0, q = 0; p < n * n; p++, q += 4) {
      var c = shade(out.label[p], out.steps[p]);
      d[q] = c[0]; d[q + 1] = c[1]; d[q + 2] = c[2]; d[q + 3] = 255;
    }
  }

  /* How many of the catalogued islands contain at least one point of a
   * uniform n x n training grid?  This is the reachability question, made
   * concrete.  `cell` is the fine-grid spacing and `lim` the half-width of the
   * plane window (1.5).
   *
   * A grid point counts as "inside" an island if the fine cell it falls in
   * belongs to that island's mask -- the same test the Python code uses.
   */
  function gridHits(islands, n, cell, lim) {
    lim = lim === undefined ? 1.5 : lim;
    if (n < 2) { return { total: 0, hits: 0, hitIds: [] }; }
    var h = (2 * lim) / (n - 1);
    var hits = 0, hitIds = [];
    for (var k = 0; k < islands.length; k++) {
      var isl = islands[k];
      var y0 = isl.origin[0], x0 = isl.origin[1];
      var mask = isl.mask, nr = mask.length, nc = mask[0].length;
      var found = false;
      for (var r = 0; r < nr && !found; r++) {
        for (var c = 0; c < nc; c++) {
          if (mask[r].charCodeAt(c) !== 49) { continue; }   // '1'
          var x = -lim + (x0 + c) * cell;
          var y = -lim + (y0 + r) * cell;
          var gi = Math.round((x + lim) / h);
          var gj = Math.round((y + lim) / h);
          if (gi < 0 || gi >= n || gj < 0 || gj >= n) { continue; }
          // where does that grid point actually fall on the fine raster?
          var gx = Math.round((-lim + gi * h + lim) / cell);
          var gy = Math.round((-lim + gj * h + lim) / cell);
          var py = gy - y0, px = gx - x0;
          if (py >= 0 && py < nr && px >= 0 && px < nc &&
              mask[py].charCodeAt(px) === 49) {
            found = true;
            break;
          }
        }
      }
      if (found) { hits++; hitIds.push(isl.id); }
    }
    return { total: islands.length, hits: hits, hitIds: hitIds };
  }

  /* Island under the cursor (for hover/click): nearest island centre within
   * 1.5 island widths. */
  function islandAt(islands, x, y) {
    var best = null, bestD = Infinity;
    for (var k = 0; k < islands.length; k++) {
      var isl = islands[k];
      var d = Math.hypot(x - isl.cx, y - isl.cy);
      if (d < 1.5 * isl.width_units + 0.004 && d < bestD) {
        bestD = d; best = isl;
      }
    }
    return best;
  }

  global.Fractal = {
    ROOTS: ROOTS,
    labelAt: labelAt,
    raster: raster,
    shade: shade,
    fillImage: fillImage,
    gridHits: gridHits,
    islandAt: islandAt
  };
})(typeof window !== "undefined" ? window : globalThis);
