// Read-only project evidence, independent of the selected reconstruction run.
(function (KR) {
  "use strict";
  const { $, D, t, esc, fmt } = KR;
  const text = (key, values = {}) => Object.entries(values).reduce(
    (s, [name, value]) => s.replaceAll("{" + name + "}", value), t(key));
  function zeige() {
    const data = D.forschung;
    const content = $("forschung-inhalt");
    if (!data || !data.available) {
      content.innerHTML = `<p class="warn" role="status">${esc(t("fr_unavailable"))}</p>`;
      return;
    }
    const s = data.summary;
    const source = (key) => data.sources[key] ?
      `<a href="${esc(data.sources[key].url)}" target="_blank" rel="noopener noreferrer">${esc(t("fr_source"))}</a>` :
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
    content.innerHTML = `<p class="hinweis-klein">${esc(text("fr_date", { date: data.date }))}</p>` +
      `<p class="warn">${esc(t("fr_scope"))}</p>` + cards.map((card) =>
        `<article class="karte forschung-karte" data-research="${card.key}">` +
        `<div class="karte-kopf"><h2>${esc(t("fr_" + card.key))}</h2>` +
        `<span class="forschung-status" data-status="${card.status}">${esc(t("fr_" + card.status))}</span></div>` +
        `<p>${esc(text(card.body, card.values))}</p>${source(card.key)}</article>`).join("") +
      `<p class="hinweis-klein">${esc(t("fr_links"))}</p>`;
  }
  KR.ansicht({ key: "forschung", zeige });
})(window.KR);
