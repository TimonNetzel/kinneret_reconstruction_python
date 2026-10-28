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

  function zeichneRaeume(ids, daten, st) {
    const div = $("tf-plot");
    const B = st.biome.labels.length, eng = schmal(div);
    const gT = daten[0].rekonstruktion.raster.T, gP = daten[0].rekonstruktion.raster.P;
    const traces = [], shapes = [], ann = [];
    const lay = {};
    for (let b = 0; b < B; b++) {
      const sx = b ? "x" + (b + 1) : "x", sy = b ? "y" + (b + 1) : "y";
      const dom = (b0) => [b0 / B + (b0 ? 0.03 : 0), (b0 + 1) / B - (b0 < B - 1 ? 0.03 : 0)];
      lay["xaxis" + (b ? b + 1 : "")] = achse({ domain: eng ? [0, 1] : dom(b), anchor: sy, range: T_BEREICH,
        title: { text: eng && b < B - 1 ? "" : tx("T") } });
      lay["yaxis" + (b ? b + 1 : "")] = achse({ domain: eng ? zeilen(Array(B).fill(1), 0.09)[b] : [0, 0.9], anchor: sx, range: P_BEREICH,
        title: { text: !eng && b ? "" : tx("P") } });
      ann.push(titel(`(${"abc"[b]}) ${st.biome.namen[b]}`, sy + " domain", sx + " domain"));
      // Posterior des Hauptlaufs als Flaeche
      traces.push({ type: "heatmap", x: gT, y: gP, z: daten[0].transferfunktionen.dichte_posterior[b], xaxis: sx,
        yaxis: sy, colorscale: dichteSkala(), showscale: false, zmin: 0, zmax: 100, hoverinfo: "skip" });
      // Prior der TF: Pr(Biom | Klima) = 0,5
      traces.push({ type: "contour", x: gT, y: gP, z: st.tf_prior_dichte[b], xaxis: sx, yaxis: sy, showscale: false,
        contours: { start: 50, end: 50, size: 1, coloring: "none" }, line: { color: css("--fg"), width: 1.5, dash: "dash" },
        hoverinfo: "skip", name: tx("tf_prior") });
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
          name: L(LAEUFE.get(id).titel),
          hovertemplate: `${F(L(LAEUFE.get(id).titel))}<br>${tx("T_kurz")} %{x:.1f} °C<br>${tx("P_kurz")} %{y:.0f} mm<extra></extra>` });
        if (r === 0) {
          const [t0, t1] = tf.T.grenzen[b], [p0, p1] = tf.P.grenzen[b];
          shapes.push({ type: "rect", xref: sx, yref: sy, x0: t0, x1: t1, y0: p0, y1: Math.min(p1, P_BEREICH[1]),
            line: { color: c, width: 1, dash: "dot" } });
        }
      });
    }
    div.style.height = eng ? "1050px" : "";
    Plotly.react(div, traces, layout(Object.assign({ hovermode: "closest", shapes, annotations: ann,
      margin: { l: 60, r: 14, t: 30, b: 44 } }, lay)), PLOT_CONFIG);
  }

  function zeichneCi(ids, daten, st) {
    const traces = [];
    ids.forEach((id, r) => {
      const tf = daten[r].transferfunktionen, name = L(LAEUFE.get(id).titel);
      ["T", "P"].forEach((v, k) => traces.push({
        type: "bar", x: st.biome.labels, y: tf[v].ci_verhaeltnis, xaxis: k ? "x2" : "x", yaxis: k ? "y2" : "y",
        marker: { color: farbe(id) }, name, offsetgroup: String(r),
        hovertemplate: `${F(name)}<br>${tx(v + "_kurz")} %{x}: %{y:.2f}<extra></extra>`,
      }));
    });
    const ymax = Math.max(1.1, ...traces.map((tr) => Math.max(...tr.y))) * 1.05;
    const eins = (ref) => ({ type: "line", xref: ref + " domain", yref: ref === "x" ? "y" : "y2", x0: 0, x1: 1, y0: 1, y1: 1,
      line: { color: css("--muted"), width: 1, dash: "dash" } });
    Plotly.react($("tf-ci"), traces, layout({
      barmode: "group", hovermode: "closest", shapes: [eins("x"), eins("x2")],
      annotations: [titel(tx("ci_verhaeltnis") + " · " + tx("T_kurz"), "y domain", "x domain"),
        titel(tx("ci_verhaeltnis") + " · " + tx("P_kurz"), "y2 domain", "x2 domain")],
      xaxis: achse({ domain: [0, 0.47] }), xaxis2: achse({ domain: [0.53, 1], anchor: "y2" }),
      yaxis: achse({ range: [0, ymax], domain: [0, 0.88] }),
      yaxis2: achse({ range: [0, ymax], anchor: "x2", domain: [0, 0.88] }),
    }), PLOT_CONFIG);
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
