// Ansicht Rekonstruktion: T, P, Biomanteile und Baumpollenkurve eines Laufs; Detail je Alter bzw. Tiefe.
(function (KR) {
  "use strict";
  const { $, D, K, Z, t, L, css, fmt, fmtS, esc, merke, LAEUFE, FAKTOREN, ladeLauf, ladeStandort, farbe, alpha,
    dichteSkala, PLOT_CONFIG, achse, layout, zeilen, titel, band, umschalter, laufOptionen, stufeLabel,
    GRUPPEN_FOLGE, F, FT } = KR;
  const tx = (k) => F(t(k)); // uebersetzter Text als HTML (Formeln gesetzt)


  let rekToken = 0;
  let rekDetail = null; // Index des gewaehlten Alters bzw. der Tiefe

  function rekInfo(l, d) {
    const teile = [`<p>${F(L(l.beschreibung))}</p><dl>`];
    teile.push(`<dt>${tx("rolle")}</dt><dd>${F(L(K.rollen[l.rolle]))}</dd>`);
    if (l.faktor) {
      const f = FAKTOREN.get(l.faktor);
      teile.push(`<dt>${F(L(f.label))}</dt><dd>${F(stufeLabel(l, l.faktor))}</dd>`);
    }
    if (l.basis && LAEUFE.has(l.basis)) teile.push(`<dt>${tx("basis")}</dt><dd>${F(L(LAEUFE.get(l.basis).titel))}</dd>`);
    if (l.quelle) teile.push(`<dt>${tx("quelle")}</dt><dd>${esc(l.quelle)}</dd>`);
    if (d) {
      const kz = d.kennzahlen;
      teile.push(`<dt>${tx("spalte_T")}</dt><dd>${fmt(kz.T_niveau, 2)} °C</dd><dt>${tx("spalte_P")}</dt><dd>${fmt(kz.P_niveau, 0)} mm</dd>`);
      teile.push(`<dt>${tx("spalte_ess")}</dt><dd>${fmt(kz.ess_min, 0)}</dd>`);
    }
    teile.push("</dl>");
    $("rek-info").innerHTML = teile.join("");
    $("rek-hinweis").innerHTML = l.hinweis ? `<span class="warn">⚠ ${F(L(l.hinweis))}</span>` : "";
  }

  async function zeigeRekonstruktion() {
    const token = ++rekToken;
    const l = LAEUFE.get(Z.lauf);
    laufOptionen($("rek-lauf"), Z.lauf);
    let d, st;
    try {
      [d, st] = await Promise.all([ladeLauf(Z.lauf), ladeStandort(l.standort)]);
    } catch (e) {
      $("rek-info").innerHTML = `<span class="warn">${tx("fehler_laden")}: ${esc(e.message)}</span>`;
      return;
    }
    if (token !== rekToken) return;
    const art = Z.art || d.rekonstruktion.standard;
    umschalter("rek-achse", Z.achse, (w) => { Z.achse = w; rekDetail = null; merke(); zeigeRekonstruktion(); });
    umschalter("rek-art", art, (w) => { Z.art = w === d.rekonstruktion.standard ? null : w; merke(); zeigeRekonstruktion(); });
    $("rek-art").querySelectorAll("button").forEach((b) => {
      b.innerHTML = tx(b.dataset.wert === "mean" ? "mittelwert" : "mischverteilung") +
        (b.dataset.wert === d.rekonstruktion.standard ? `<span class="klein">${tx("standard_des_laufs")}</span>` : "");
    });
    rekInfo(l, d);
    detailOptionen(d);
    zeichneRekonstruktion(d, st, art);
    if (rekDetail !== null) zeigeDetail(d, st, art, rekDetail);
    else $("rek-detail").hidden = true;
  }

  function zeichneRekonstruktion(d, st, art) {
    const lauf = Z.lauf, ax = Z.achse, detail = rekDetail;
    const div = $("rek-plot");
    KR.plot(div, () => baueRekonstruktion(d, st, art, lauf, ax, detail), {
      name: `${lauf}_rekonstruktion_${ax}`, titel: "ansicht_rekonstruktion", laeufe: [lauf],
      hoehe: (b) => (b < 10 ? 14 : b < 15 ? 15 : 17),
      csv: () => csvRekonstruktion(d, st, art, ax),
    });
  }

  // Detail zu einem Alter bzw. einer Tiefe: eigene Auswahl (ein Tipp in das Diagramm zeigt nur die Werte)
  function detailOptionen(d) {
    const sel = $("rek-detail-wahl");
    const xs = d.rekonstruktion[Z.achse].x;
    if (rekDetail !== null && rekDetail >= xs.length) rekDetail = null;
    sel.innerHTML = "";
    const leer = document.createElement("option");
    leer.value = "";
    leer.textContent = FT(t("kein_detail"));
    sel.appendChild(leer);
    xs.forEach((x, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = FT(t(Z.achse === "alter" ? "detail_alter" : "detail_tiefe").replace("{x}", fmt(x, 0)));
      sel.appendChild(o);
    });
    sel.value = rekDetail === null ? "" : String(rekDetail);
    sel.onchange = () => { rekDetail = sel.value === "" ? null : Number(sel.value); zeigeRekonstruktion(); };
  }

  function baueRekonstruktion(d, st, art, lauf, ax, detail) {
    const rk = d.rekonstruktion[ax];
    const x = rk.x;
    const farbeLauf = farbe(lauf);
    const achsenBereich = {};
    const dom = zeilen([3, 3, 2, 2], 0.05);
    const traces = [];
    const ann = [];
    const ys = ["y", "y2", "y3", "y4"];
    const raster = d.rekonstruktion.raster;
    ["T", "P"].forEach((v, r) => {
      const q = rk[art][v];
      if (art === "mixture") {
        const dichte = rk.mixture.dichte[v];
        const g = raster[v];
        const z = g.map((_, j) => dichte.map((row) => row[j]));
        const genutzt = g.map((_, j) => dichte.some((row) => row[j] >= 3));
        const lo = genutzt.indexOf(true), hi = genutzt.lastIndexOf(true);
        traces.push({ type: "heatmap", x, y: g, z, yaxis: ys[r], colorscale: dichteSkala(), showscale: false,
          zmin: 0, zmax: 100, hovertemplate: `${tx(v + "_kurz")} %{y}: %{z}<extra></extra>` });
        achsenBereich[ys[r]] = lo >= 0 ? [g[Math.max(0, lo - 1)], g[Math.min(g.length - 1, hi + 1)]] : undefined;
      } else {
        traces.push(...band(x, q.q2_5, q.q97_5, farbeLauf, 0.15, ys[r], tx("band95")));
        traces.push(...band(x, q.q25, q.q75, farbeLauf, 0.32, ys[r], tx("band50")));
        traces.push({ x, y: q.q50, yaxis: ys[r], type: "scatter", mode: "lines", line: { color: farbeLauf, width: 2 },
          name: tx("median"), hovertemplate: `${tx(v + "_kurz")} %{y:.${v === "T" ? 2 : 0}f}<extra></extra>` });
        achsenBereich[ys[r]] = undefined;
      }
    });
    const bio = d.biome[ax];
    st.biome.farben.forEach((c, b) => {
      traces.push(...band(x, bio.q25[b], bio.q75[b], c, 0.3, "y3", st.biome.labels[b]));
      traces.push({ x, y: bio.q50[b], yaxis: "y3", type: "scatter", mode: "lines", line: { color: c, width: 2 },
        name: st.biome.labels[b], hovertemplate: `${st.biome.labels[b]} %{y:.2f}<extra></extra>` });
    });
    const ap = ax === "alter" ? d.ap.alter : d.ap.tiefe;
    if (d.ap.ausgeschlossen.length) {
      traces.push({ x, y: ax === "alter" ? st.ap.alter : st.ap.tiefe, yaxis: "y4", type: "scatter", mode: "lines",
        line: { color: css("--muted"), width: 1, dash: "dot" }, name: tx("ap_voll"),
        hovertemplate: `${tx("ap_voll")} %{y:.1f}<extra></extra>` });
    }
    traces.push({ x, y: ap, yaxis: "y4", type: "scatter", mode: "lines", line: { color: css("--fg"), width: 1.5 },
      name: tx("ap"), hovertemplate: `AP %{y:.1f}<extra></extra>` });

    const shapes = [];
    if (ax === "alter" && d.ap.kalibrierung_alter) {
      const m = d.ap.kalibrierung_alter;
      let start = null;
      for (let i = 0; i <= m.length; i++) {
        if (i < m.length && m[i] === 0 && start === null) start = i;
        if ((i === m.length || m[i] !== 0) && start !== null) {
          shapes.push({ type: "rect", xref: "x", yref: "paper", x0: x[start], x1: x[i - 1], y0: dom[3][0], y1: dom[3][1],
            fillcolor: alpha("#888888", 0.18), line: { width: 0 }, layer: "below" });
          start = null;
        }
      }
      ann.push({ text: tx("ap_kalibrierung"), xref: "paper", yref: "paper", x: 1, y: dom[3][1], xanchor: "right",
        yanchor: "bottom", showarrow: false, font: { size: 11, color: css("--muted") } });
    }
    if (detail !== null && detail < x.length && !KR.druck) {
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: x[detail], x1: x[detail], y0: 0, y1: 1,
        line: { color: css("--accent"), width: 1.5, dash: "dot" } });
    }
    const titelText = (v) => `${v === "T" ? "(a)" : "(b)"} ${tx(v)} · ${tx(art === "mean" ? "mittelwert" : "mischverteilung")}`;
    ann.push(titel(titelText("T"), "y domain"), titel(titelText("P"), "y2 domain"),
      titel("(c) " + tx("biome") + " · " + st.biome.labels.map((b, k) => `<span style="color:${st.biome.farben[k]}">■</span> ${esc(b)}`)
        .join("  "), "y3 domain"), titel("(d) " + tx("ap"), "y4 domain"));
    const lay = layout({
      hovermode: "x", shapes, annotations: ann,
      xaxis: achse({ title: { text: tx(ax === "alter" ? "alter_achse" : "tiefe_achse") }, anchor: "y4" }),
      yaxis: achse({ domain: dom[0], title: { text: "°C" }, range: achsenBereich.y }),
      yaxis2: achse({ domain: dom[1], title: { text: "mm" }, range: achsenBereich.y2, rangemode: "tozero" }),
      yaxis3: achse({ domain: dom[2], range: [0, 1], title: { text: tx("anteil") } }),
      yaxis4: achse({ domain: dom[3], title: { text: "%" } }),
    });
    ["yaxis", "yaxis2", "yaxis3", "yaxis4"].forEach((a) => { if (!lay[a].range) delete lay[a].range; });
    return [traces, lay];
  }

  // CSV: Quantile von T und P (gezeigte Rekonstruktion), Biomanteile und Baumpollenkurve je Alter bzw. Tiefe;
  // bei der Mischverteilung zusaetzlich ihre Dichte (lang: Alter, Groesse, Wert, Dichte)
  const QS = ["q2_5", "q25", "q50", "q75", "q97_5"];
  function csvRekonstruktion(d, st, art, ax) {
    const rk = d.rekonstruktion[ax], bio = d.biome[ax];
    const xName = ax === "alter" ? "alter_cal_a_bp" : "tiefe_cm";
    const spalten = [xName, ...["T_DJF", "P_ANN"].flatMap((v) => [...QS, "mittel"].map((q) => `${v}_${q}`)),
      ...st.biome.labels.flatMap((b) => QS.map((q) => `biom_${b}_${q}`)), "ap_prozent"];
    const voll = d.ap.ausgeschlossen.length > 0, kal = ax === "alter" && d.ap.kalibrierung_alter;
    if (voll) spalten.push("ap_alle_gehoelze_prozent");
    if (kal) spalten.push("in_kalibrierung");
    const ap = ax === "alter" ? d.ap.alter : d.ap.tiefe, apVoll = ax === "alter" ? st.ap.alter : st.ap.tiefe;
    const zeilen = rk.x.map((x, i) => {
      const z = [x];
      ["T", "P"].forEach((v) => [...QS, "mittel"].forEach((q) => z.push(rk[art][v][q][i])));
      st.biome.labels.forEach((_, b) => QS.forEach((q) => z.push(bio[q][b][i])));
      z.push(ap[i]);
      if (voll) z.push(apVoll[i]);
      if (kal) z.push(d.ap.kalibrierung_alter[i]);
      return z;
    });
    const out = [{ name: "quantile", titel: "csv_quantile", zeilen, spalten,
      kopf: [`rekonstruktion: ${art === "mean" ? "mittelwert" : "mischverteilung"}`] }];
    if (art === "mixture") {
      const lang = [];
      ["T", "P"].forEach((v) => rk.x.forEach((x, i) => d.rekonstruktion.raster[v].forEach((w, j) => {
        const z = rk.mixture.dichte[v][i][j];
        if (z > 0) lang.push([x, v === "T" ? "T_DJF" : "P_ANN", w, z]);
      })));
      out.push({ name: "dichte", titel: "csv_dichte", spalten: [xName, "groesse", "wert", "dichte"], zeilen: lang,
        kopf: ["dichte je Zeile auf das Maximum 100 normiert, Werte 0 weggelassen"] });
    }
    return out;
  }

  function zeigeDetail(d, st, art, i) {
    const lauf = Z.lauf, ax = Z.achse;
    const rk = d.rekonstruktion[ax], xs = rk.x[i];
    $("rek-detail").hidden = false;
    $("rek-detail-titel").innerHTML = tx(Z.achse === "alter" ? "detail_alter" : "detail_tiefe").replace("{x}", fmt(rk.x[i], 0));
    $("rek-detail-zu").onclick = () => { rekDetail = null; zeigeRekonstruktion(); };
    const c = farbe(Z.lauf);
    // Verteilung von T und P: Dichte der Mischverteilung, darueber Median, 50 % und 95 % des Mittelwerts
    const verteilung = () => { const traces = []; ["T", "P"].forEach((v, k) => {
      const xa = k ? "x2" : "x", ya = k ? "y2" : "y";
      const g = d.rekonstruktion.raster[v];
      traces.push({ x: g, y: rk.mixture.dichte[v][i].map((z) => z / 100), xaxis: xa, yaxis: ya, type: "scatter",
        mode: "lines", fill: "tozeroy", line: { color: css("--accent"), width: 1 }, fillcolor: alpha("#2a6fb0", 0.25),
        name: tx("mischverteilung"), hovertemplate: `${tx("mischverteilung")} %{x}: %{y:.2f}<extra></extra>` });
      const q = rk.mean[v];
      traces.push({ x: [q.q50[i]], y: [1.12], xaxis: xa, yaxis: ya, type: "scatter", mode: "markers",
        marker: { color: c, size: 8 }, name: tx("mittelwert"),
        error_x: { type: "data", symmetric: false, array: [q.q97_5[i] - q.q50[i]], arrayminus: [q.q50[i] - q.q2_5[i]],
          color: c, thickness: 1.2, width: 4 },
        hovertemplate: `${tx("mittelwert")}: %{x:.${v === "T" ? 2 : 0}f}<extra></extra>` });
      traces.push({ x: [q.q25[i], q.q75[i]], y: [1.12, 1.12], xaxis: xa, yaxis: ya, type: "scatter", mode: "lines",
        line: { color: c, width: 6 }, hoverinfo: "skip" });
    }); return traces; };
    const sichtbar = (v) => {
      const g = d.rekonstruktion.raster[v], z = rk.mixture.dichte[v][i], q = rk.mean[v];
      let lo = q.q2_5[i], hi = q.q97_5[i];
      z.forEach((w, j) => { if (w >= 2) { lo = Math.min(lo, g[j]); hi = Math.max(hi, g[j]); } });
      const pad = (hi - lo) * 0.08 || 1;
      return [lo - pad, hi + pad];
    };
    const name = `${lauf}_${ax}_${fmt(xs, 0).replace(/\D/g, "")}`;
    const kopf = [`${ax === "alter" ? "alter_cal_a_bp" : "tiefe_cm"}: ${xs}`];
    KR.plot($("rek-detail-verteilung"), () => [verteilung(), layout({
      margin: { l: 34, r: 10, t: 26, b: 40 },
      xaxis: achse({ domain: [0, 0.46], title: { text: tx("T") }, range: sichtbar("T") }),
      xaxis2: achse({ domain: [0.54, 1], title: { text: tx("P") }, range: sichtbar("P") }),
      yaxis: achse({ range: [0, 1.25], showticklabels: false, title: { text: tx("dichte") } }),
      yaxis2: achse({ range: [0, 1.25], showticklabels: false, anchor: "x2" }),
      annotations: [titel(tx("verteilung"), "paper")],
    })], { name: name + "_verteilung", titel: "verteilung", laeufe: [lauf], hoehe: (b) => (b < 10 ? 5 : 6),
      csv: () => [{ name: "verteilung", titel: "verteilung", kopf,
        spalten: ["groesse", "wert", "dichte_mischverteilung"],
        zeilen: ["T", "P"].flatMap((v) => d.rekonstruktion.raster[v].map((w, j) =>
          [v === "T" ? "T_DJF" : "P_ANN", w, rk.mixture.dichte[v][i][j]])) },
      { name: "mittelwert", titel: "csv_quantile", kopf, spalten: ["groesse", ...QS],
        zeilen: ["T", "P"].map((v) => [v === "T" ? "T_DJF" : "P_ANN", ...QS.map((q) => rk.mean[v][q][i])]) }] });
    // Biomanteile: Median mit 95-%-Intervall
    const bio = d.biome[ax];
    KR.plot($("rek-detail-biome"), () => [[{
      type: "bar", x: st.biome.labels, y: st.biome.labels.map((_, b) => bio.q50[b][i]),
      marker: { color: st.biome.farben },
      error_y: { type: "data", symmetric: false, color: css("--fg"),
        array: st.biome.labels.map((_, b) => bio.q97_5[b][i] - bio.q50[b][i]),
        arrayminus: st.biome.labels.map((_, b) => bio.q50[b][i] - bio.q2_5[b][i]) },
      hovertemplate: "%{x}: %{y:.2f}<extra></extra>",
    }], layout({ margin: { l: 44, r: 10, t: 26, b: 30 }, yaxis: achse({ range: [0, 1], title: { text: tx("anteil") } }),
      xaxis: achse({}), annotations: [titel(tx("biome"), "paper")] })], {
      name: name + "_biome", titel: "biome", laeufe: [lauf], hoehe: () => 6,
      csv: () => [{ name: "biome", titel: "biome", kopf, spalten: ["biom", ...QS],
        zeilen: st.biome.labels.map((b, k) => [b, ...QS.map((q) => bio[q][k][i])]) }] });
    // Spektrum: haeufigste Taxa, Farbe nach Biom
    const sp = (ax === "alter" ? st.spektrum_alter : st.spektrum_tiefe)[i];
    const idx = sp.map((v, k) => k).sort((a, b) => sp[b] - sp[a]).slice(0, 12);
    KR.plot($("rek-detail-spektrum"), () => [[{
      type: "bar", orientation: "h", y: idx.map((k) => st.taxa.namen[k]), x: idx.map((k) => sp[k]),
      marker: { color: idx.map((k) => st.biome.farben[st.taxa.biom[k]]) },
      hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
    }], layout({ margin: { l: 10, r: 10, t: 26, b: 34 }, xaxis: achse({ title: { text: tx("anteil") } }),
      yaxis: achse({ autorange: "reversed", tickfont: { size: 11 } }), annotations: [titel(tx("spektrum"), "paper")] })], {
      name: name + "_spektrum", titel: "spektrum", laeufe: [lauf], hoehe: (b) => (b < 10 ? 8 : 7),
      csv: () => [{ name: "spektrum", titel: "spektrum", kopf, spalten: ["taxon", "biom", "anteil"],
        zeilen: sp.map((v, k) => k).sort((a, b) => sp[b] - sp[a])
          .map((k) => [st.taxa.namen[k], st.biome.labels[st.taxa.biom[k]], sp[k]]) }] });
  }

  KR.ansicht({
    key: "rekonstruktion",
    init() {
      $("rek-lauf").onchange = (e) => { Z.lauf = e.target.value; Z.art = null; rekDetail = null; merke(); zeigeRekonstruktion(); };
      $("rek-info-box").open = window.matchMedia("(min-width: 1000px)").matches;
    },
    zeige: zeigeRekonstruktion,
  });
})(window.KR);
