// Read-only project evidence, independent of the selected reconstruction run.
(function (KR) {
  "use strict";
  const { $, D, t, esc, fmt } = KR;
  const text = (key, values = {}) => Object.entries(values).reduce(
    (s, [name, value]) => s.replaceAll("{" + name + "}", value), t(key));
  // like text(), but values are trusted markup (formatted numbers, superscripts)
  const html = (key, values = {}) => Object.entries(values).reduce(
    (s, [name, value]) => s.replaceAll("{" + name + "}", value), esc(t(key)));
  const link = (url, key) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(t(key))}</a>`;
  const exp10 = (x) => {
    const [m, e] = Number(x).toExponential(2).split("e");
    return `${fmt(Number(m), 2)}&nbsp;×&nbsp;10<sup>${Number(e) < 0 ? "−" : ""}${Math.abs(Number(e))}</sup>`;
  };
  const CENTRAL = ["final_base", "final_chronology", "historical_tf", "runtime_rng"];

  function snapshot() {
    const data = D.forschung;
    if (!data || !data.available) return `<p class="warn" role="status">${esc(t("fr_unavailable"))}</p>`;
    const s = data.summary;
    const source = (key) => data.sources[key] ? link(data.sources[key].url, "fr_source") :
      `<span class="hinweis-klein">${esc(t("fr_source_missing"))}</span>`;
    const cards = [
      { key: "baseline", status: data.baseline_available ? "available" : "missing", body: data.baseline_available ? "fr_baseline_body" : "fr_baseline_missing" },
      { key: "species", status: data.species_count ? "experimental" : "missing", body: data.species_count ? "fr_species_body" : "fr_species_missing",
        values: { n: fmt(data.species_count, 0) } },
      { key: "historical", status: "missing", body: "fr_historical_body", values: {
        pages: fmt(s.plots.pages_per_pdf, 0), fits: fmt(s.historical_models.caret_fits, 0),
        binary: fmt(s.historical_models.binary_fits, 0), multi: fmt(s.historical_models.multiclass_fits, 0) } },
      { key: "paper", status: "experimental", body: "fr_paper_body" },
      { key: "chronology", status: "open", body: "fr_chronology_body" },
    ];
    return `<p class="hinweis-klein">${esc(text("fr_date", { date: data.date }))}</p>` +
      `<p class="warn">${esc(t("fr_scope"))}</p>` + cards.map((card) =>
        `<article class="karte forschung-karte" data-research="${card.key}">` +
        `<div class="karte-kopf"><h2>${esc(t("fr_" + card.key))}</h2>` +
        `<span class="forschung-status" data-status="${card.status}">${esc(t("fr_" + card.status))}</span></div>` +
        `<p>${esc(text(card.body, card.values))}</p>${source(card.key)}` +
        (card.key === "species" ? `<p><button class="knopf" id="fr-artenvergleich">${esc(t("av_open"))}</button></p>` : "") + `</article>`).join("") +
      `<p class="hinweis-klein">${esc(t("fr_links"))}</p>`;
  }

  // Synthetic example tables: columns are artificial climate cells, rows artificial taxa or samples
  function tabelle(caption, rows, rowLabel, cell, sums, extraRow) {
    const n = rows[0].length;
    return `<p class="nachbau-titel">${caption}</p><div class="tabelle-rahmen"><table class="nachbau-tabelle"><thead><tr><th></th>` +
      Array.from({ length: n }, (_, k) => `<th scope="col">${esc(text("nb_ex_cell", { n: k + 1 }))}</th>`).join("") +
      (sums ? `<th scope="col">${esc(t("nb_ex_sum"))}</th>` : "") + `</tr></thead><tbody>${extraRow || ""}` +
      rows.map((row, i) => `<tr><th scope="row">${esc(text(rowLabel, { n: i + 1 }))}</th>` +
        row.map((v, k) => `<td>${cell(v, i, k)}</td>`).join("") + (sums ? `<td>${sums[i]}</td>` : "") + `</tr>`).join("") +
      `</tbody></table></div>`;
  }

  function beispiel(ex) {
    const keep = `<tr data-row="keep"><th scope="row">${esc(t("nb_ex_keep"))}</th>` +
      ex.keep.map((k) => `<td>${esc(t(k ? "nb_yes" : "nb_no"))}</td>`).join("") + `<td></td></tr>`;
    const exact = (m) => (v, i, k) => esc(m[i][k]);
    const rowSum = (row) => row.reduce((a, b) => a + b, 0);
    return `<details class="karte nachbau-karte" data-part="example"><summary><h3>${esc(t("nb_example"))}</h3></summary>` +
      `<p class="warn" data-synthetic="true">${esc(t("nb_example_warn"))}</p>` +
      `<h4>${esc(t("nb_ex_tf"))}</h4>` +
      tabelle(esc(t("nb_ex_raw")), ex.raw, "nb_ex_taxon", (v) => fmt(v, 0), ex.raw.map((r) => fmt(rowSum(r), 0))) +
      tabelle(esc(t("nb_ex_norm")), ex.normalized, "nb_ex_taxon", exact(ex.normalized_exact),
        ex.normalized.map((r) => `<span title="${fmt(rowSum(r), 4)}">1</span>`)) +
      tabelle(esc(t("nb_ex_mask")), ex.masked, "nb_ex_taxon", exact(ex.masked_exact),
        ex.row_masses_exact.map((m, i) => `<strong data-row-mass="${esc(m)}" title="${fmt(ex.row_masses_after_mask[i], 4)}">${esc(m)}</strong>`), keep) +
      `<p>${html("nb_ex_no_renorm", { mass: esc(ex.row_masses_exact[1]), tf: fmt(ex.renormalization_tf_difference, 4),
        pdf: fmt(ex.renormalization_pdf_difference, 4) })}</p>` +
      `<h4>${esc(t("nb_ex_chain"))}</h4><p>${esc(text("nb_ex_chain_body", { ids: ex.draw_ids_R_1_based.join(t("nb_and")) }))}</p>` +
      tabelle(esc(t("nb_ex_pdf")), ex.chain_pdf, "nb_ex_sample", (v) => fmt(v, 4), ex.chain_pdf.map((r) => fmt(rowSum(r), 4))) +
      `<p>${html("nb_ex_check", { err: exp10(ex.max_rational_error), tol: exp10(ex.tolerance) })}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: ex.sha256.result.slice(0, 12), b: ex.sha256.arithmetic.slice(0, 12) }))} ` +
      `${link(ex.url, "nb_evidence")}</p></details>`;
  }

  function nachbau() {
    const nb = D.nachbau;
    const kopf = `<div class="karte-kopf"><h2>${esc(t("nb_title"))}</h2>` +
      `<span class="forschung-status" data-status="open">${esc(t("nb_status"))}</span></div>`;
    if (!nb || !nb.available) {
      return `<section class="nachbau" data-research="reproduction">${kopf}` +
        `<p class="warn" role="status">${esc(t("nb_unavailable"))}</p></section>`;
    }
    const modules = nb.modules.map((m) => `<li data-module="${esc(m.id)}"><strong>${esc(t("nb_mod_" + m.id))}</strong> ` +
      `<span class="hinweis-klein">${esc(text("nb_tests_n", { n: fmt(m.new_tests, 0) }))}</span>` +
      `<br>${esc(t("nb_mod_" + m.id + "_body"))}<br><span class="hinweis-klein">${esc(t("nb_sources_label"))} ` +
      `${m.original_sources.map(esc).join(", ")}</span> · ${link(m.url, "nb_evidence")}</li>`).join("");
    const reqs = nb.requirements.map((r) => `<li data-requirement="${esc(r.id)}"${CENTRAL.includes(r.id) ? ' data-central="true"' : ""}>` +
      `<div class="nachbau-req-kopf"><strong>${esc(t("nb_req_" + r.id))}</strong>` +
      `<span class="forschung-status" data-status="${esc(r.status)}">${esc(t("nb_req_" + r.status))}</span>` +
      (CENTRAL.includes(r.id) ? `<span class="forschung-status" data-status="central">${esc(t("nb_central_tag"))}</span>` : "") +
      `</div>${esc(t("nb_req_" + r.id + "_body"))}</li>`).join("");
    return `<section class="nachbau" data-research="reproduction">${kopf}` +
      `<p class="hinweis-klein">${esc(text("nb_date", { date: nb.date }))}</p>` +
      `<p>${esc(text("nb_summary", { tests: fmt(nb.tests_passed, 0) }))}</p>` +
      `<article class="karte nachbau-karte" data-part="modules"><h3>${esc(t("nb_modules"))}</h3><ol class="nachbau-liste">${modules}</ol>` +
      `<p>${esc(text("nb_chain", { n: fmt(nb.chain_check.new_tests, 0), c: fmt(nb.contract.tests, 0) }))} ` +
      `${link(nb.chain_check.url, "nb_chain_link")} · ${link(nb.contract.url, "nb_contract_link")}</p></article>` +
      `<article class="karte nachbau-karte" data-part="requirements">` +
      `<h3>${esc(text("nb_requirements", { n: fmt(nb.requirements.length, 0) }))}</h3><p>${esc(t("nb_central"))}</p>` +
      `<ul class="nachbau-liste">${reqs}</ul>${link(nb.contract.url, "nb_contract_link")}</article>` +
      beispiel(nb.example) + `</section>`;
  }

  function zeige() {
    const beispielOffen = !!document.querySelector('#forschung-inhalt [data-part="example"][open]');
    $("forschung-inhalt").innerHTML = snapshot() + nachbau();
    const offen = document.querySelector('#forschung-inhalt [data-part="example"]');
    if (offen) offen.open = beispielOffen;
    const knopf = $("fr-artenvergleich");
    if (knopf) knopf.onclick = () => { KR.Z.ansicht = "artenvergleich"; KR.merke(); KR.zeige(); };
  }
  KR.ansicht({ key: "forschung", zeige });
})(window.KR);
