// Stored outer-test scores only, independent of the selected fossil run and site.
(function (KR) {
  "use strict";
  const { $, D, Z, t, esc, fmt, merke, umschalter } = KR;
  const tx = (key, values = {}) => esc(Object.entries(values).reduce(
    (s, [name, value]) => s.replaceAll("{" + name + "}", value), t(key)));
  function zeige() {
    const data = D.artenvergleich;
    const content = $("arten-inhalt");
    const available = !!(data && data.available);
    $("arten-auswahl").hidden = !available;
    if (!available) {
      content.innerHTML = `<p class="warn" role="status">${tx("av_unavailable")}</p>`;
      return;
    }
    if (!data.blocks.includes(Z.artenBlock)) Z.artenBlock = data.blocks[0];
    const block = Z.artenBlock;
    if (!["summary", "folds"].includes(Z.artenEbene)) Z.artenEbene = "summary";
    umschalter("arten-ebene", Z.artenEbene, (value) => { Z.artenEbene = value; merke(); zeige(); });
    const foldMode = Z.artenEbene === "folds";
    const foldData = data.fold_scores;
    $("arten-taxon-feld").hidden = !foldMode || !foldData || !foldData.available;
    if (!Object.hasOwn(data.species, Z.artenTaxon)) Z.artenTaxon = Object.keys(data.species)[0];
    const select = $("arten-taxon");
    if (!select.options.length) select.innerHTML = Object.entries(data.species).map(([key, name]) =>
      `<option value="${esc(key)}">${esc(name)}</option>`).join("");
    select.value = Z.artenTaxon;
    select.onchange = () => { Z.artenTaxon = select.value; merke(); zeige(); };
    const switcher = $("arten-block");
    if (!switcher.children.length) switcher.innerHTML = data.blocks.map((b) =>
      `<button data-wert="${b}">${b}°</button>`).join("");
    umschalter("arten-block", String(block), (value) => { Z.artenBlock = Number(value); merke(); zeige(); });
    const extra = $("arten-extra");
    extra.checked = !!Z.artenExtra;
    extra.onchange = () => { Z.artenExtra = extra.checked; merke(); zeige(); };
    const methods = Z.artenExtra ? data.methods : data.methods.slice(0, 2);
    const metricCells = (r) => `<td>${r.balanced_accuracy === null ? tx("av_undefined") : fmt(100 * r.balanced_accuracy, 2) + " %"}</td>` +
      `<td>${fmt(r.brier, 5)}</td>`;
    const header = `<thead><tr><th scope="col">${tx("av_variant")}</th>` +
      `<th scope="col">BA ↑</th><th scope="col">Brier ↓</th></tr></thead>`;
    const methodCell = (m) => `<th scope="row">${tx("av_" + m)}</th>`;
    const overview = data.blocks.map((b) => `<div class="karte arten-mittel" data-block="${b}">` +
      `<h3>${tx("av_block_title", { b })}</h3><div class="tabelle-rahmen"><table>` +
      `<caption class="nur-leser">${tx("av_mean")} · ${b}°</caption>${header}<tbody>` +
      methods.map((m) => { const r = data.means.find((x) => x.block_degrees === b && x.method === m);
        return `<tr data-method="${m}">${methodCell(m)}${metricCells(r)}</tr>`; }).join("") +
      `</tbody></table></div></div>`).join("");
    const species = Object.entries(data.species).map(([key, name]) => {
      const rows = data.rows.filter((r) => r.taxon === key && r.block_degrees === block);
      return `<article class="karte arten-art" data-taxon="${key}"><h3><i>${esc(name)}</i></h3>` +
        `<div class="tabelle-rahmen"><table><caption class="nur-leser">${esc(name)} · ${block}°</caption>${header}<tbody>` +
        methods.map((m) => `<tr data-method="${m}">${methodCell(m)}${metricCells(rows.find((r) => r.method === m))}</tr>`).join("") +
        `</tbody></table></div></article>`;
    }).join("");
    let foldSection = "";
    if (foldMode) {
      if (!foldData || !foldData.available) {
        foldSection = `<p class="warn" role="status">${tx("av_folds_missing")}</p>`;
      } else {
        const name = data.species[Z.artenTaxon];
        const folds = Array.from({ length: data.folds }, (_, fold) => {
          const rows = foldData.rows.filter((r) => r.taxon === Z.artenTaxon && r.block_degrees === block && r.fold === fold);
          const first = rows[0];
          const classNote = first.positive_cells === 0 ? "av_no_positives" :
            first.positive_cells === first.cells ? "av_no_negatives" : null;
          return `<article class="karte arten-fold" data-fold="${fold}"><h3>${tx("av_fold", { n: fold + 1 })}</h3>` +
            `<p class="hinweis-klein" data-counts>${tx("av_fold_counts", { n: fmt(first.cells, 0),
              p: fmt(first.positive_cells, 0), a: fmt(first.cells - first.positive_cells, 0) })}</p>` +
            (classNote ? `<p class="warn">${tx(classNote)}</p>` : "") +
            `<div class="tabelle-rahmen"><table><caption class="nur-leser">${esc(name)} · ${block}° · ${tx("av_fold", { n: fold + 1 })}</caption>` +
            header + `<tbody>` + methods.map((m) => `<tr data-method="${m}">${methodCell(m)}${metricCells(rows.find((r) => r.method === m))}</tr>`).join("") +
            `</tbody></table></div></article>`;
        }).join("");
        foldSection = `<section aria-labelledby="arten-fold-titel"><h2 id="arten-fold-titel">${esc(name)} · ${tx("av_block_title", { b: block })}</h2>` +
          `<p>${tx("av_fold_intro")}</p><p class="hinweis-klein">${tx("av_fold_numbering")}</p>` +
          `<div class="arten-raster">${folds}</div><p class="warn">${tx("av_fold_limits")}</p></section>`;
      }
    }
    const links = Object.entries(data.sources).map(([key, url]) =>
      `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${tx("av_source_" + key)}</a>`).join(" · ");
    content.innerHTML = `<div class="karte arten-einordnung"><div class="karte-kopf"><h2>${tx("av_title")}</h2>` +
      `<span class="forschung-status" data-status="experimental">${tx("fr_experimental")}</span></div>` +
      `<p>${tx("av_result")}</p><p class="warn">${tx("av_limit")}</p>` +
      `<dl class="arten-definition"><dt>Balanced Accuracy (BA) ↑</dt><dd>${tx("av_ba")}</dd>` +
      `<dt>Brier ↓</dt><dd>${tx("av_brier")}</dd></dl>` +
      `<p class="hinweis-klein">${tx("av_design", { f: data.folds, km: data.buffer_km, date: data.date })}</p></div>` +
      (foldMode ? foldSection : `<section aria-labelledby="arten-mittel-titel"><h2 id="arten-mittel-titel">${tx("av_mean")}</h2>` +
      `<p class="hinweis-klein">${tx("av_mean_note")}</p><div class="arten-raster">${overview}</div></section>` +
      `<section aria-labelledby="arten-detail-titel"><h2 id="arten-detail-titel">${tx("av_detail", { b: block })}</h2>` +
      `<div class="arten-raster">${species}</div></section>`) +
      (Z.artenExtra ? `<p class="warn">${tx("av_extra_note")}</p>` : "") +
      `<div class="karte arten-quellen"><h2>${tx("av_sources")}</h2><p>${links || tx("fr_source_missing")}</p>` +
      `<p class="hinweis-klein">${tx("av_maps")}</p><p class="hinweis-klein">${tx("fr_links")}</p></div>`;
  }
  KR.ansicht({ key: "artenvergleich", zeige });
})(window.KR);
