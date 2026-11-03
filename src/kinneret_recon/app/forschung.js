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

  // Artificial pruning case: columns are artificial taxa; R removes every taxon matching sort(r)[16:T]
  function pruning(p) {
    if (!p) return "";
    const n = p.zero_counts.length;
    const kopf = Array.from({ length: n }, (_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const zeile = (label, cells, attr = "") => `<tr${attr}><th scope="row">${esc(label)}</th>${cells.join("")}</tr>`;
    const raus = (k) => p.removed_taxa.includes(k);
    return `<h4>${esc(text("nb_ex_pruning", { n: fmt(n, 0) }))}</h4><p>${esc(text("nb_ex_pruning_body", { rows: fmt(p.rows, 0), n: fmt(n, 0) }))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="pruning"><thead><tr><th scope="col">${esc(t("nb_ex_taxa_head"))}</th>${kopf}</tr></thead><tbody>` +
      zeile(text("nb_ex_zeros", { rows: fmt(p.rows, 0) }), p.zero_counts.map((z) => `<td>${fmt(z, 0)}</td>`)) +
      zeile(t("nb_ex_ratio"), p.null_ratio_exact.map((r, k) => `<td title="${fmt(p.null_ratio[k], 4)}">${esc(r)}</td>`)) +
      zeile(t("nb_ex_removed"), p.zero_counts.map((_, k) =>
        `<td${raus(k) ? ' data-removed="true"' : ""}>${raus(k) ? `<strong>${esc(t("nb_yes"))}</strong>` : esc(t("nb_no"))}</td>`)) +
      `</tbody></table></div><p>${esc(t("nb_ex_pruning_why"))}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: p.sha256.result.slice(0, 12), b: p.sha256.rational.slice(0, 12) }))} ` +
      `${link(p.url, "nb_evidence")}</p>`;
  }

  // Artificial gamma-grid cases: columns are cases; the source takes the first which.min on the 0.1 grid
  function gamma(g) {
    if (!g) return "";
    const kurz = (v, d = 4) => Number(v).toLocaleString(KR.Z.sprache === "de" ? "de-DE" : "en-GB",
      { maximumSignificantDigits: Math.max(1, Math.ceil(Math.log10(Math.abs(v) + 1)) + d) });
    const roh = (v) => String(v).replace(".", fmt(0.5, 1).charAt(1));
    const kopf = g.cases.map((_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const zeile = (key, cell, attr = "") => `<tr${attr}><th scope="row">${esc(t(key))}</th>${g.cases.map(cell).join("")}</tr>`;
    const c = g.chain;
    return `<h4>${esc(text("nb_ex_gamma", { n: fmt(g.cases.length, 0) }))}</h4>` +
      `<p>${esc(text("nb_ex_gamma_body", { n: fmt(g.grid_length, 0), last: fmt(g.grid_last, 0) }))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="gamma"><thead><tr><th scope="col">${esc(t("nb_ex_case"))}</th>${kopf}</tr></thead><tbody>` +
      zeile("nb_ex_g_mean", (x) => `<td>${fmt(x.mean, 0)}</td>`) +
      zeile("nb_ex_g_sd", (x) => `<td>${fmt(x.sd, 0)}</td>`) +
      zeile("nb_ex_g_closed", (x) => `<td title="${esc(roh(x.closed_form))}">${kurz(x.closed_form)}</td>`) +
      zeile("nb_ex_g_nearest", (x) => `<td${x.nearest_differs ? ' data-differs="true"' : ""}>${kurz(x.nearest)}</td>`) +
      zeile("nb_ex_g_source", (x) => `<td data-shape="${esc(String(x.shape))}" title="${esc(roh(x.shape))}"><strong>${kurz(x.shape)}</strong></td>`) +
      zeile("nb_ex_g_index", (x) => `<td>${fmt(x.r_index, 0)}</td>`) +
      zeile("nb_ex_g_rate", (x) => `<td title="${esc(roh(x.rate))}">${kurz(x.rate, 3)}</td>`) +
      `</tbody></table></div><p>${esc(text("nb_ex_gamma_why", { shape: roh(g.cases[0].shape) }))}</p>` +
      `<p data-gamma-chain="true">${esc(text("nb_ex_gamma_chain", { d: roh(c.default_shape), sd: fmt(c.sd_pann, 0),
        f: roh(c.final_shape), r1281: roh(c.final_rate), r1733: roh(c.rate_after_recent_start) }))}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: g.sha256.result.slice(0, 12), b: g.sha256.rational.slice(0, 12) }))} ` +
      `${link(g.url, "nb_evidence")}</p>`;
  }

  // Artificial which.tf.missing cases: columns are cases; the NA mask is given, never computed from weightedMedian
  function fehlend(m) {
    if (!m) return "";
    const kopf = m.cases.map((_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const zeile = (key, cell) => `<tr><th scope="row">${esc(t(key))}</th>${m.cases.map(cell).join("")}</tr>`;
    const liste = (v) => v.length ? v.map((x) => fmt(x, 0)).join(", ") : "–";
    const e = m.empty;
    return `<h4>${esc(text("nb_ex_missing", { n: fmt(m.cases.length, 0) }))}</h4><p>${esc(t("nb_ex_missing_body"))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="missing"><thead><tr><th scope="col">${esc(t("nb_ex_case"))}</th>${kopf}</tr></thead><tbody>` +
      zeile("nb_ex_m_taxa", (x) => `<td>${fmt(x.taxa, 0)}</td>`) +
      zeile("nb_ex_m_missing", (x) => `<td>${liste(x.missing)}</td>`) +
      zeile("nb_ex_m_x2", (x) => `<td>${liste(x.x2_only)}</td>`) +
      zeile("nb_ex_m_kept", (x) => `<td>${esc(x.kept.join(""))}</td>`) +
      zeile("nb_ex_m_num", (x) => `<td data-line="${x.line}">${fmt(x.num_taxa, 0)}</td>`) +
      zeile("nb_ex_m_sd", (x) => `<td data-sd="${x.sd}"><strong>${fmt(x.sd, 0)}</strong></td>`) +
      zeile("nb_ex_m_half_up", (x) => `<td${x.half_up_differs ? ' data-differs="true"' : ""}>${fmt(x.sd_half_up, 0)}</td>`) +
      zeile("nb_ex_m_phi", (x) => `<td>${fmt(x.phi, 1)}</td>`) +
      zeile("nb_ex_m_phi1", (x) => `<td${x.stale_phi1 !== x.phi ? ' data-differs="true"' : ""}>${fmt(x.stale_phi1, 1)}</td>`) +
      `</tbody></table></div><p>${esc(text("nb_ex_missing_why", { list: m.half_up_differs_for.map((x) => fmt(x, 0)).join(", ") }))}</p>` +
      `<p data-missing-empty="true">${esc(text("nb_ex_missing_empty", { n: fmt(e.num_taxa, 0), line: e.line, sd: fmt(e.sd, 0) }))}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: m.sha256.result.slice(0, 12), b: m.sha256.rational.slice(0, 12) }))} ` +
      `${link(m.url, "nb_evidence")}</p>`;
  }

  // Artificial sampler preparation: columns are artificial taxa; each column divided by its maximum, then read row-wise
  function vorbereitung(s) {
    if (!s) return "";
    const n = s.column_max.length;
    const kopf = Array.from({ length: n }, (_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const zeile = (label, cells) => `<tr><th scope="row">${esc(label)}</th>${cells.join("")}</tr>`;
    const liste = (v) => v.map((x) => fmt(x, 0)).join(", ");
    const roh = s.raw.map((r) => r.join(" ")).join(" / ");
    const sp = s.special;
    const fall = sp.map((_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const szeile = (key, cell, attr = "") => `<tr${attr}><th scope="row">${esc(t(key))}</th>${sp.map(cell).join("")}</tr>`;
    return `<h4>${esc(text("nb_ex_sampler", { n: fmt(sp.length + 3, 0) }))}</h4>` +
      `<p>${esc(text("nb_ex_sampler_body", { raw: roh }))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="sampler"><thead><tr><th scope="col">${esc(t("nb_ex_taxa_head"))}</th>${kopf}</tr></thead><tbody>` +
      zeile(t("nb_ex_s_max"), s.column_max.map((v) => `<td>${fmt(v, 0)}</td>`)) +
      s.normalized_exact.map((row, i) => zeile(text("nb_ex_s_row", { n: i + 1 }),
        row.map((v, k) => `<td title="${fmt(s.normalized[i][k], 4)}">${esc(v)}</td>`))).join("") +
      zeile(t("nb_ex_s_mean"), s.means_exact.map((v, k) => s.means_independent[k] ? `<td data-acc="exact">${esc(v)}</td>` :
        `<td data-acc="dependent">${esc(v)}*</td>`)) +
      `</tbody></table></div>` +
      `<p data-sampler-values="true">${esc(text("nb_ex_sampler_flat", { values: s.values_exact.join(", "),
        bad: fmt(s.misaligned.column_major, 0), n: fmt(s.values_exact.length, 0) }))}</p>` +
      `<p class="nachbau-titel">${esc(t("nb_ex_sampler_special"))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="sampler-special"><thead><tr><th scope="col">${esc(t("nb_ex_case"))}</th>${fall}</tr></thead><tbody>` +
      szeile("nb_ex_s_dims", (x) => `<td>${fmt(x.nrow, 0)}×${fmt(x.ncol, 0)}</td>`) +
      szeile("nb_ex_s_maxima", (x) => `<td>${liste(x.column_max)}</td>`) +
      szeile("nb_ex_s_na", (x) => `<td>${fmt(x.na, 0)}</td>`) +
      szeile("nb_ex_s_nan", (x) => `<td${x.nan ? ' data-special="nan"' : ""}>${fmt(x.nan, 0)}</td>`) +
      szeile("nb_ex_s_inf", (x) => `<td${x.minus_inf ? ' data-special="-inf"' : ""}>${fmt(x.minus_inf, 0)}</td>`) +
      `</tbody></table></div>` +
      `<p>${esc(text("nb_ex_sampler_special_why", { m1: sp[0].means_exact.join(", ") }))}</p>` +
      `<p data-sampler-gibbs="true">${esc(text("nb_ex_sampler_gibbs", { w: fmt(s.how_many_weights, 0), n: fmt(s.num_taxa, 0),
        g: fmt(s.how_many_gibbs, 0), j: fmt(s.jeffreys, 1) }))}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: s.sha256.result.slice(0, 12), b: s.sha256.rational.slice(0, 12) }))} ` +
      `${link(s.url, "nb_evidence")}</p>`;
  }

  // Artificial weights stage: explicit weights instead of a Dirichlet draw; columns are artificial taxa, then cases
  function gewichtsstufe(w) {
    if (!w) return "";
    const n = w.z.length;
    const roh = (v) => String(v).replace(".", fmt(0.5, 1).charAt(1));
    const kopf = Array.from({ length: n }, (_, k) => `<th scope="col">${fmt(k + 1, 0)}</th>`).join("");
    const zeile = (label, cells) => `<tr><th scope="row">${esc(label)}</th>${cells.join("")}</tr>`;
    const summe = (i) => w.independent[i] ? `<td data-acc="exact">${esc(w.sums_exact[i])}</td>` :
      `<td data-acc="dependent" title="${esc(roh(w.sums[i]))}">${fmt(w.sums[i], 4)}*</td>`;
    const sp = w.special;
    const fall = sp.map((_, k) => `<th scope="col">${esc(text("nb_ex_w_case", { n: k + 1 }))}</th>`).join("");
    const szeile = (key, cell) => `<tr><th scope="row">${esc(t(key))}</th>${sp.map(cell).join("")}</tr>`;
    const markiert = (v, art) => `<td${v ? ` data-special="${art}"` : ""}>${fmt(v, 0)}</td>`;
    return `<h4>${esc(text("nb_ex_weights", { n: fmt(sp.length + 4, 0) }))}</h4>` +
      `<p>${esc(text("nb_ex_weights_body", { w: w.weights_exact.join(", ") }))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="weights"><thead><tr><th scope="col">${esc(t("nb_ex_taxa_head"))}</th>${kopf}` +
      `<th scope="col">${esc(t("nb_ex_sum"))}</th></tr></thead><tbody>` +
      zeile(t("nb_ex_w_weight"), [...w.weights_exact.map((v) => `<td>${esc(v)}</td>`), "<td></td>"]) +
      w.rows.map((row, i) => zeile(text("nb_ex_s_row", { n: i + 1 }), [...row.map((v, k) => w.rows_exact[i][k] ?
        `<td title="${fmt(v, 4)}">${esc(w.rows_exact[i][k])}</td>` : `<td title="${esc(roh(v))}">${fmt(v, 4)}</td>`), summe(i)])).join("") +
      `</tbody></table></div>` +
      `<p data-weights-flat="true">${esc(text("nb_ex_weights_flat", { bad: fmt(w.misaligned, 0), n: fmt(w.positions, 0),
        k: fmt(n, 0) }))}</p>` +
      `<p class="nachbau-titel">${esc(t("nb_ex_weights_special"))}</p>` +
      `<div class="tabelle-rahmen"><table class="nachbau-tabelle" data-table="weights-special"><thead><tr><th scope="col">${esc(t("nb_ex_case"))}</th>${fall}</tr></thead><tbody>` +
      szeile("nb_ex_s_dims", (x) => `<td>${fmt(x.nrow, 0)}×${fmt(x.ncol, 0)}</td>`) +
      szeile("nb_ex_w_zero", (x) => markiert(x.zero_rows, "zero")) +
      szeile("nb_ex_w_minf", (x) => markiert(x.minus_inf_rows, "-inf")) +
      szeile("nb_ex_s_na", (x) => `<td>${fmt(x.na, 0)}</td>`) +
      szeile("nb_ex_s_nan", (x) => `<td>${fmt(x.nan, 0)}</td>`) +
      szeile("nb_ex_w_inf", (x) => `<td>${fmt(x.inf, 0)}</td>`) +
      szeile("nb_ex_w_negzero", (x) => `<td>${fmt(x.neg_zero, 0)}</td>`) +
      `</tbody></table></div>` +
      `<p>${esc(text("nb_ex_weights_special_why", { s1: sp[0].sums_exact.join(", ") }))}</p>` +
      `<p data-weights-reset="true">${esc(text("nb_ex_weights_reset", { w: w.reset_weights.join(", "), third: roh(w.reset_third),
        first: roh(w.reset_first), sw: w.scale_weights.join(", "), bw: w.scale_base.join(", ") }))}</p>` +
      `<p>${esc(t("nb_ex_weights_dirichlet"))}</p>` +
      `<p class="hinweis-klein">${esc(text("nb_ex_hash", { a: w.sha256.result.slice(0, 12), b: w.sha256.rational.slice(0, 12) }))} ` +
      `${link(w.url, "nb_evidence")}</p>`;
  }

  function beispiel(ex, pex, gex, mex, sex, wex) {
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
      `${link(ex.url, "nb_evidence")}</p>${pruning(pex)}${gamma(gex)}${fehlend(mex)}${vorbereitung(sex)}${gewichtsstufe(wex)}</details>`;
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
      (m.conditional ? `<span class="forschung-status" data-status="conditional">${esc(t("nb_conditional"))}</span> ` : "") +
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
      (nb.earlier_states || []).map((s) => `<p class="hinweis-klein" data-earlier-state="${esc(s.state_sha256)}">` +
        `${esc(text("nb_earlier", { date: s.date, modules: fmt(s.modules, 0), tests: fmt(s.tests_passed, 0),
          hash: s.state_sha256.slice(0, 12) }))} ${link(s.url, "nb_evidence")}</p>`).join("") +
      `<p>${esc(text("nb_summary", { n: fmt(nb.modules.length, 0), tests: fmt(nb.tests_passed, 0) }))}</p>` +
      `<article class="karte nachbau-karte" data-part="modules"><h3>${esc(t("nb_modules"))}</h3><ol class="nachbau-liste">${modules}</ol>` +
      `<p>${esc(text("nb_chain", { n: fmt(nb.chain_check.new_tests, 0), c: fmt(nb.contract.tests, 0) }))} ` +
      `${link(nb.chain_check.url, "nb_chain_link")} · ${link(nb.contract.url, "nb_contract_link")}</p></article>` +
      `<article class="karte nachbau-karte" data-part="requirements">` +
      `<h3>${esc(text("nb_requirements", { n: fmt(nb.requirements.length, 0) }))}</h3><p>${esc(t("nb_central"))}</p>` +
      `<ul class="nachbau-liste">${reqs}</ul>${link(nb.contract.url, "nb_contract_link")}</article>` +
      beispiel(nb.example, nb.pruning_example, nb.gamma_example, nb.missing_example, nb.sampler_example,
        nb.weights_example) + `</section>`;
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
