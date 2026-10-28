// Ansicht Taxa-Gewichte: Posterior je Taxon (Median, 50 %, 95 %) eines Laufs und im Vergleich, mit Prior.
(function (KR) {
  "use strict";
  const { $, Z, t, L, css, fmt, esc, merke, LAEUFE, ladeLauf, ladeStandort, farbe, alpha, PLOT_CONFIG, achse, layout,
    titel, umschalter, laufOptionen, vergleichsAuswahl } = KR;

  let token = 0;

  function alphaText(d) {
    const a = d.gewichte.alpha;
    if (a.fest !== undefined) return t("alpha_fest").replace("{a}", fmt(a.fest, 2));
    return t("alpha_hier").replace("{m}", fmt(a.q50, 2)).replace("{lo}", fmt(a.q2_5, 2)).replace("{hi}", fmt(a.q97_5, 2));
  }

  async function zeige() {
    const my = ++token;
    laufOptionen($("gew-lauf"), Z.lauf);
    const ids = vergleichsAuswahl($("gew-vergleich"), zeige);
    const sort = Z.gewSort || "median";
    umschalter("gew-sort", sort, (w) => { Z.gewSort = w; merke(); zeige(); });
    $("gew-log").checked = !!Z.gewLog;
    let daten, st;
    try {
      [st, ...daten] = await Promise.all([ladeStandort(LAEUFE.get(Z.lauf).standort), ...ids.map(ladeLauf)]);
    } catch (e) {
      $("gew-info").innerHTML = `<span class="warn">${t("fehler_laden")}: ${esc(e.message)}</span>`;
      return;
    }
    if (my !== token) return;
    $("gew-info").innerHTML = ids.map((id, k) =>
      `<div><span class="farbe" style="background:${farbe(id)}"></span> ${esc(L(LAEUFE.get(id).titel))}: ` +
      `${esc(alphaText(daten[k]))}</div>`).join("");
    zeichne(ids, daten, st, sort);
  }

  function zeichne(ids, daten, st, sort) {
    const namen = st.taxa.namen, biom = st.taxa.biom, K = namen.length;
    const g0 = daten[0].gewichte;
    let order = [...Array(K).keys()];
    if (sort === "median") order.sort((a, b) => g0.q50[b] - g0.q50[a]);
    else if (sort === "biom") order.sort((a, b) => (biom[a] - biom[b]) || (g0.q50[b] - g0.q50[a]));
    else order.sort((a, b) => namen[a].localeCompare(namen[b]));
    const pos = new Map(order.map((k, i) => [k, i]));
    const n = ids.length, breite = 0.75;
    const traces = [], shapes = [];
    const log = !!Z.gewLog;
    const klein = 1e-5;
    const xw = (v) => (log ? Math.max(v, klein) : v);
    // Prior: 50-%-Bereich als Feld (erster Lauf), Median je Lauf als Linie
    const p0 = g0.prior;
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: xw(p0.q25), x1: xw(p0.q75), y0: 0, y1: 1,
      fillcolor: alpha("#888888", 0.15), line: { width: 0 }, layer: "below" });
    ids.forEach((id, r) => {
      const g = daten[r].gewichte, c = farbe(id);
      const off = n > 1 ? (r - (n - 1) / 2) * (breite / n) : 0;
      const y = (k) => pos.get(k) + off;
      const seg = (lo, hi) => {
        const xs = [], ys = [];
        for (let k = 0; k < K; k++) xs.push(xw(lo[k]), xw(hi[k]), null), ys.push(y(k), y(k), null);
        return { xs, ys };
      };
      const s95 = seg(g.q2_5, g.q97_5), s50 = seg(g.q25, g.q75);
      const name = L(LAEUFE.get(id).titel);
      traces.push({ x: s95.xs, y: s95.ys, type: "scatter", mode: "lines", line: { color: c, width: 1.2 }, hoverinfo: "skip" });
      traces.push({ x: s50.xs, y: s50.ys, type: "scatter", mode: "lines", line: { color: c, width: n > 3 ? 3 : 5 }, hoverinfo: "skip" });
      traces.push({
        x: [...Array(K).keys()].map((k) => xw(g.q50[k])), y: [...Array(K).keys()].map(y), type: "scatter", mode: "markers",
        marker: { color: c, size: 7, line: { color: css("--bg"), width: 1 } }, name,
        customdata: [...Array(K).keys()].map((k) => [namen[k], g.q25[k], g.q75[k], g.q2_5[k], g.q97_5[k]]),
        hovertemplate: `${esc(name)}<br>%{customdata[0]}: %{x:.4f}<br>50 %: %{customdata[1]:.4f}–%{customdata[2]:.4f}` +
          `<br>95 %: %{customdata[3]:.4f}–%{customdata[4]:.4f}<extra></extra>`,
      });
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: xw(g.prior.q50), x1: xw(g.prior.q50), y0: 0, y1: 1,
        line: { color: c, width: 1.2, dash: "dash" } });
    });
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: 1 / K, x1: 1 / K, y0: 0, y1: 1,
      line: { color: css("--muted"), width: 1, dash: "dot" } });
    const ticktext = order.map((k) => `<span style="color:${st.biome.farben[biom[k]]}">■</span> ${esc(namen[k])}`);
    const q99 = Math.max(0.25, ...daten.map((d) => Math.max(...d.gewichte.q97_5)));
    const div = $("gew-plot");
    div.style.height = Math.max(520, K * (16 + 6 * n) + 110) + "px";
    Plotly.react(div, traces, layout({
      hovermode: "closest", shapes, margin: { l: 10, r: 14, t: 30, b: 44 },
      annotations: [titel(`${t("ansicht_gewichte")} · ${t("prior_median")} (- -) · 1/${K} (···)`, "paper")],
      xaxis: achse(log ? { type: "log", title: { text: t("gewicht") }, range: [-4.3, 0] }
        : { title: { text: t("gewicht") }, range: [0, Math.min(1, q99 * 1.05)] }),
      yaxis: achse({ tickvals: order.map((_, i) => i), ticktext, autorange: "reversed", tickfont: { size: 11 },
        showgrid: false, range: [K - 0.5, -0.5] }),
    }), PLOT_CONFIG);
  }

  KR.ansicht({
    key: "gewichte",
    init() {
      $("gew-lauf").onchange = (e) => { Z.lauf = e.target.value; Z.art = null; merke(); zeige(); };
      $("gew-log").onchange = (e) => { Z.gewLog = e.target.checked; merke(); zeige(); };
    },
    zeige,
  });
})(window.KR);
