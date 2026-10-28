// Ansicht Transferfunktionen: Klimaraeume der Biome (Prior der TF, Posterior der Klimawerte) und CI-Verhaeltnis.
(function (KR) {
  "use strict";
  const { $, Z, t, L, css, fmt, esc, merke, LAEUFE, ladeLauf, ladeStandort, farbe, dichteSkala, PLOT_CONFIG, achse,
    layout, titel, zeilen, laufOptionen, vergleichsAuswahl, schmal, F, FT } = KR;
  const tx = (k) => F(t(k)); // uebersetzter Text als HTML (Formeln gesetzt)

  let token = 0;
  const T_BEREICH = [-10, 30], P_BEREICH = [0, 1500];

  async function zeige() {
    const my = ++token;
    laufOptionen($("tf-lauf"), Z.lauf);
    const ids = vergleichsAuswahl($("tf-vergleich"), zeige);
    let daten, st;
    try {
      [st, ...daten] = await Promise.all([ladeStandort(LAEUFE.get(Z.lauf).standort), ...ids.map(ladeLauf)]);
    } catch (e) {
      $("tf-info").innerHTML = `<span class="warn">${tx("fehler_laden")}: ${esc(e.message)}</span>`;
      return;
    }
    if (my !== token) return;
    const ref = daten[0].transferfunktionen.prior_referenz;
    $("tf-info").innerHTML = tx(ref === "model" ? "tf_prior_model" : "tf_prior_marginal");
    zeichneRaeume(ids, daten, st);
    zeichneCi(ids, daten, st);
    tabelle(daten[0], st);
  }

  // CSV (fuer beide Diagramme): je Lauf und Biom Posterior und Prior von T und P, Trunkierung, CI-Verhaeltnis
  function csv(ids, daten, st) {
    const qs = ["q2_5", "q25", "q50", "q75", "q97_5"];
    const zeilen = [];
    ids.forEach((id, r) => {
      const tf = daten[r].transferfunktionen;
      st.biome.labels.forEach((b, k) => zeilen.push([id, b, ...["T", "P"].flatMap((v) => [
        ...qs.map((q) => tf[v].posterior[q][k]), ...qs.map((q) => tf[v].prior[q][k]), tf[v].grenzen[k][0],
        tf[v].grenzen[k][1], tf[v].ci_posterior[k], tf[v].ci_prior[k], tf[v].ci_verhaeltnis[k]])]));
    });
    const spalten = ["lauf", "biom", ...["T_DJF", "P_ANN"].flatMap((v) => [...qs.map((q) => `${v}_posterior_${q}`),
      ...qs.map((q) => `${v}_prior_${q}`), `${v}_trunkierung_unten`, `${v}_trunkierung_oben`, `${v}_ci95_posterior`,
      `${v}_ci95_prior`, `${v}_ci_verhaeltnis`])];
    return [{ name: "transferfunktionen", titel: "ansicht_transferfunktionen", spalten, zeilen,
      kopf: ids.map((id, r) => `prior-referenz ${id}: ${daten[r].transferfunktionen.prior_referenz}`) }];
  }

  function zeichneRaeume(ids, daten, st) {
    const div = $("tf-plot");
    div.style.height = schmal(div) ? "1050px" : "";
    KR.plot(div, () => baueRaeume(ids, daten, st, div), {
      name: `${ids[0]}_transferfunktionen`, titel: "ansicht_transferfunktionen", laeufe: ids,
      hoehe: (b) => (b < 12 ? 19 : b < 15 ? 6.5 : 7.5), csv: () => csv(ids, daten, st),
    });
  }

  function baueRaeume(ids, daten, st, div) {
    const B = st.biome.labels.length, eng = schmal(div);
    const gT = daten[0].rekonstruktion.raster.T, gP = daten[0].rekonstruktion.raster.P;
    const traces = [], shapes = [], ann = [];
    const lay = {};
    for (let b = 0; b < B; b++) {
      const sx = b ? "x" + (b + 1) : "x", sy = b ? "y" + (b + 1) : "y";
      const dom = (b0) => [b0 / B + (b0 ? 0.03 : 0), (b0 + 1) / B - (b0 < B - 1 ? 0.03 : 0)];
      lay["xaxis" + (b ? b + 1 : "")] = achse({ domain: eng ? [0, 1] : dom(b), anchor: sy, range: T_BEREICH,
        title: { text: eng && b < B - 1 ? "" : tx("T") } });
      lay["yaxis" + (b ? b + 1 : "")] = achse({ domain: eng ? zeilen(Array(B).fill(1), 0.09)[b] : [0, KR.druck ? 1 : 0.9],
        anchor: sx, range: P_BEREICH,
        title: { text: !eng && b ? "" : tx("P") } });
      ann.push(titel(`(${"abc"[b]}) ${st.biome.namen[b]}`, sy + " domain", sx + " domain"));
      // Posterior des Hauptlaufs als Flaeche
      traces.push({ type: "heatmap", x: gT, y: gP, z: daten[0].transferfunktionen.dichte_posterior[b], xaxis: sx,
        yaxis: sy, colorscale: dichteSkala(), showscale: false, zmin: 0, zmax: 100, hoverinfo: "skip" });
      // Prior der TF: Pr(Biom | Klima) = 0,5
      traces.push({ type: "contour", x: gT, y: gP, z: st.tf_prior_dichte[b], xaxis: sx, yaxis: sy, showscale: false,
        contours: { start: 50, end: 50, size: 1, coloring: "none" }, line: { color: css("--fg"), width: 1.5, dash: "dash" },
        hoverinfo: "skip", name: tx("tf_prior"), showlegend: b === 0 });
      ids.forEach((id, r) => {
        const tf = daten[r].transferfunktionen, c = farbe(id);
        if (r > 0) {
          traces.push({ type: "contour", x: gT, y: gP, z: tf.dichte_posterior[b], xaxis: sx, yaxis: sy, showscale: false,
            contours: { start: 50, end: 50, size: 1, coloring: "none" }, line: { color: c, width: 2 }, hoverinfo: "skip" });
        }
        const tq = tf.T.posterior, pq = tf.P.posterior;
        traces.push({ type: "scatter", mode: "markers", x: [tq.q50[b]], y: [pq.q50[b]], xaxis: sx, yaxis: sy,
          marker: { color: c, size: 8, line: { color: css("--bg"), width: 1 } },
          error_x: { type: "data", symmetric: false, array: [tq.q97_5[b] - tq.q50[b]], arrayminus: [tq.q50[b] - tq.q2_5[b]], color: c, thickness: 1 },
          error_y: { type: "data", symmetric: false, array: [pq.q97_5[b] - pq.q50[b]], arrayminus: [pq.q50[b] - pq.q2_5[b]], color: c, thickness: 1 },
          name: F(L(LAEUFE.get(id).titel)), showlegend: b === 0,
          hovertemplate: `${F(L(LAEUFE.get(id).titel))}<br>${tx("T_kurz")} %{x:.1f} °C<br>${tx("P_kurz")} %{y:.0f} mm<extra></extra>` });
        if (r === 0) {
          const [t0, t1] = tf.T.grenzen[b], [p0, p1] = tf.P.grenzen[b];
          shapes.push({ type: "rect", xref: sx, yref: sy, x0: t0, x1: t1, y0: p0, y1: Math.min(p1, P_BEREICH[1]),
            line: { color: c, width: 1, dash: "dot" } });
        }
      });
    }
    return [traces, layout(Object.assign({ hovermode: "closest", shapes, annotations: ann,
      showlegend: !!KR.druck && ids.length > 1, margin: { l: 60, r: 14, t: 30, b: 44 } }, lay))];
  }

  function zeichneCi(ids, daten, st) {
    KR.plot($("tf-ci"), () => baueCi(ids, daten, st), {
      name: `${ids[0]}_ci_verhaeltnis`, titel: "ci_verhaeltnis", laeufe: ids, hoehe: (b) => (b < 10 ? 5.5 : 6),
      csv: () => csv(ids, daten, st),
    });
  }

  function baueCi(ids, daten, st) {
    const traces = [];
    ids.forEach((id, r) => {
      const tf = daten[r].transferfunktionen, name = L(LAEUFE.get(id).titel);
      ["T", "P"].forEach((v, k) => traces.push({
        type: "bar", x: st.biome.labels, y: tf[v].ci_verhaeltnis, xaxis: k ? "x2" : "x", yaxis: k ? "y2" : "y",
        marker: { color: farbe(id) }, name: F(name), offsetgroup: String(r), showlegend: k === 0,
        hovertemplate: `${F(name)}<br>${tx(v + "_kurz")} %{x}: %{y:.2f}<extra></extra>`,
      }));
    });
    const ymax = Math.max(1.1, ...traces.map((tr) => Math.max(...tr.y))) * 1.05;
    const eins = (ref) => ({ type: "line", xref: ref + " domain", yref: ref === "x" ? "y" : "y2", x0: 0, x1: 1, y0: 1, y1: 1,
      line: { color: css("--muted"), width: 1, dash: "dash" } });
    return [traces, layout({
      barmode: "group", hovermode: "closest", showlegend: !!KR.druck && ids.length > 1, shapes: [eins("x"), eins("x2")],
      annotations: [titel("(a) " + tx("T_kurz"), "y domain", "x domain"), titel("(b) " + tx("P_kurz"), "y2 domain", "x2 domain")],
      xaxis: achse({ domain: [0, 0.47] }), xaxis2: achse({ domain: [0.53, 1], anchor: "y2" }),
      yaxis: achse({ range: [0, ymax], domain: [0, KR.druck ? 1 : 0.88], title: { text: tx("ci_verhaeltnis") } }),
      yaxis2: achse({ range: [0, ymax], anchor: "x2", domain: [0, KR.druck ? 1 : 0.88] }),
    })];
  }

  function tabelle(d, st) {
    const tf = d.transferfunktionen;
    const iv = (q, b, nd) => `${fmt(q.q50[b], nd)} (${fmt(q.q2_5[b], nd)}–${fmt(q.q97_5[b], nd)})`;
    const T = tx("T_kurz"), P = tx("P_kurz");
    const kopf = [tx("spalte_biom"), `${T} ${tx("median_95")}`, `${T} ${tx("prior_95")}`, `${tx("ci_kurz")} ${T}`,
      `${P} ${tx("median_95")}`, `${P} ${tx("prior_95")}`, `${tx("ci_kurz")} ${P}`, `${tx("trunkierung")} ${T}`,
      `${tx("trunkierung")} ${P}`];
    const zeilen = st.biome.namen.map((name, b) => `<tr><td>${esc(name)}</td>` +
      `<td>${iv(tf.T.posterior, b, 1)}</td><td>${iv(tf.T.prior, b, 1)}</td><td>${fmt(tf.T.ci_verhaeltnis[b], 2)}</td>` +
      `<td>${iv(tf.P.posterior, b, 0)}</td><td>${iv(tf.P.prior, b, 0)}</td><td>${fmt(tf.P.ci_verhaeltnis[b], 2)}</td>` +
      `<td>${fmt(tf.T.grenzen[b][0], 1)}–${fmt(tf.T.grenzen[b][1], 1)}</td>` +
      `<td>${fmt(tf.P.grenzen[b][0], 0)}–${fmt(tf.P.grenzen[b][1], 0)}</td></tr>`).join("");
    $("tf-tabelle").innerHTML = `<thead><tr>${kopf.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${zeilen}</tbody>`;
  }

  KR.ansicht({
    key: "transferfunktionen",
    init() {
      $("tf-lauf").onchange = (e) => { Z.lauf = e.target.value; Z.art = null; merke(); zeige(); };
      window.addEventListener("resize", () => {
        if (Z.ansicht === "transferfunktionen" && !$("ansicht-transferfunktionen").hidden) {
          const eng = schmal($("tf-plot"));
          if (eng !== $("tf-plot")._krEng) { $("tf-plot")._krEng = eng; zeige(); }
        }
      });
    },
    zeige,
  });
})(window.KR);
