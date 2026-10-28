// Kinneret-App: Rekonstruktion und Sensitivitaet (docs/app_konzept.md). Laeuft offline, auch ueber file://.
(function () {
  "use strict";

  const D = (window.KR_DATEN = window.KR_DATEN || {});
  const K = D.katalog;
  const TX = window.KR_TEXTE;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------------------------------------------
  // Zustand (pro Geraet im Browser gespeichert; ohne Speicher laeuft die App mit Voreinstellungen)

  const SPEICHER = {
    get(k, d) {
      try { const v = localStorage.getItem("kr-" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; }
    },
    set(k, v) { try { localStorage.setItem("kr-" + k, JSON.stringify(v)); } catch (e) { /* ohne Speicher */ } },
  };
  const LAEUFE = new Map(K.laeufe.map((l) => [l.id, l]));
  const FAKTOREN = new Map(K.faktoren.map((f) => [f.key, f]));
  const Z = Object.assign({
    sprache: (navigator.language || "de").toLowerCase().startsWith("de") ? "de" : "en",
    thema: "auto", ansicht: "rekonstruktion", lauf: K.standard_lauf, achse: "alter", art: null,
    faktor: "klima_prior", basis: null, auswahl: null, extra: [], baender: false,
  }, SPEICHER.get("zustand", {}));
  if (!LAEUFE.has(Z.lauf) || !LAEUFE.get(Z.lauf).app_json) Z.lauf = K.standard_lauf;
  if (!FAKTOREN.has(Z.faktor)) Z.faktor = K.faktoren[0].key;
  const merke = () => SPEICHER.set("zustand", Z);

  const t = (k) => (TX[Z.sprache] && TX[Z.sprache][k]) || TX.de[k] || k;
  const L = (o) => (o ? o[Z.sprache] || o.de || "" : "");
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const locale = () => (Z.sprache === "de" ? "de-DE" : "en-GB");
  const fmt = (x, nd) => (x === null || x === undefined || !isFinite(x) ? "–" :
    Number(x).toLocaleString(locale(), { minimumFractionDigits: nd, maximumFractionDigits: nd }));
  const fmtS = (x, nd) => (x === null || x === undefined || !isFinite(x) ? "–" : (x > 0 ? "+" : "") + fmt(x, nd));

  // ---------------------------------------------------------------------------------------------------------------
  // Daten laden: eingebettet (KR_DATEN), komprimiert eingebettet (eine Datei) oder als JS-Datei nachladen (Ordner)

  const cache = {};
  let laedt = 0;
  function ladeAnzeige(d) {
    laedt += d;
    $("laden").hidden = laedt <= 0;
  }
  function ladeSkript(src) {
    return new Promise((ok, fehler) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = ok;
      s.onerror = () => fehler(new Error(src));
      document.head.appendChild(s);
    });
  }
  async function entpacke(el) {
    const bin = Uint8Array.from(atob(el.textContent.trim()), (c) => c.charCodeAt(0));
    const strom = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(strom).text());
  }
  function lade(schluessel, eingebettet, datei) {
    if (!cache[schluessel]) {
      cache[schluessel] = (async () => {
        ladeAnzeige(1);
        try {
          if (D[schluessel]) return D[schluessel];
          const el = eingebettet && document.getElementById(eingebettet);
          if (el) D[schluessel] = await entpacke(el);
          else await ladeSkript(datei);
          if (!D[schluessel]) throw new Error(schluessel);
          return D[schluessel];
        } finally {
          ladeAnzeige(-1);
        }
      })();
      cache[schluessel].catch(() => delete cache[schluessel]);
    }
    return cache[schluessel];
  }
  const ladeLauf = (id) => lade("lauf/" + id, "kr-lauf-" + id, "daten/laeufe/" + id + ".js");
  const ladeStandort = (key) => lade("standort/" + key, null, "daten/standort_" + key + ".js");

  // ---------------------------------------------------------------------------------------------------------------
  // Farben: eine Farbe je Lauf ueber alle Ansichten (farbsehschwaechen-taugliche Palette nach Tol und Okabe-Ito)

  const PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#E69F00", "#009E73",
    "#D55E00", "#0072B2", "#CC79A7", "#999933", "#882255", "#44AA99"];
  const farben = new Map();
  function farbe(id) {
    if (!farben.has(id)) {
      const frei = PALETTE.filter((c) => ![...farben.values()].includes(c));
      farben.set(id, frei.length ? frei[0] : PALETTE[farben.size % PALETTE.length]);
    }
    return farben.get(id);
  }
  function alpha(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function dunkel() {
    return Z.thema === "dark" || (Z.thema === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function dichteSkala() {
    return dunkel()
      ? [[0, "rgba(0,0,0,0)"], [0.02, "#10324f"], [0.3, "#1f6aa5"], [0.7, "#69b3e7"], [1, "#e3f2ff"]]
      : [[0, "rgba(255,255,255,0)"], [0.02, "#e3eef9"], [0.3, "#7fb2dc"], [0.7, "#2a6fb0"], [1, "#0b2d5c"]];
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Plotly-Grundeinstellungen

  const PLOT_CONFIG = {
    responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverClosestCartesian",
      "hoverCompareCartesian"],
    toImageButtonOptions: { format: "png", scale: 2 },
  };
  function achse(extra) {
    return Object.assign({ gridcolor: css("--plot-grid"), zerolinecolor: css("--line"), linecolor: css("--line"),
      tickcolor: css("--line"), automargin: true }, extra);
  }
  function layout(extra) {
    return Object.assign({
      autosize: true, paper_bgcolor: css("--bg"), plot_bgcolor: css("--bg"), showlegend: false,
      font: { color: css("--fg"), family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 12 },
      margin: { l: 60, r: 14, t: 26, b: 44 }, separators: Z.sprache === "de" ? ",." : ".,",
      hoverlabel: { bgcolor: css("--bg2"), bordercolor: css("--line"), font: { color: css("--fg") } },
      dragmode: "zoom",
    }, extra);
  }
  // Zeilen eines Diagramms untereinander: Bereiche der y-Achsen (von oben), Anteile in hoehen
  function zeilen(hoehen, abstand) {
    const summe = hoehen.reduce((a, b) => a + b, 0);
    const frei = 1 - abstand * (hoehen.length - 1);
    let oben = 1;
    return hoehen.map((h) => {
      const unten = oben - (frei * h) / summe;
      const d = [Math.max(0, unten), oben];
      oben = unten - abstand;
      return d;
    });
  }
  function titel(text, ref) {
    return { text, xref: "paper", yref: ref, x: 0, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false,
      font: { size: 12, color: css("--muted") } };
  }
  function band(x, lo, hi, color, a, yaxis, name) {
    return [
      { x, y: lo, yaxis, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x, y: hi, yaxis, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: alpha(color, a), hoverinfo: "skip", name, showlegend: false },
    ];
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Texte, Thema, Reiter

  function setzeTexte() {
    document.documentElement.lang = Z.sprache;
    document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
    document.querySelectorAll("[data-t-title]").forEach((el) => { el.title = t(el.dataset.tTitle); });
    $("sprache").textContent = Z.sprache === "de" ? "EN" : "DE";
    document.title = t("app_titel") + " · " + (D["standort/" + aktuellerStandort()] ? L(D["standort/" + aktuellerStandort()].name) : "");
  }
  function setzeThema() {
    if (Z.thema === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = Z.thema;
    $("thema").textContent = t("thema_" + Z.thema);
  }
  function setzeAnsicht() {
    document.querySelectorAll(".reiter button").forEach((b) => b.setAttribute("aria-selected", b.dataset.ansicht === Z.ansicht));
    $("ansicht-rekonstruktion").hidden = Z.ansicht !== "rekonstruktion";
    $("ansicht-sensitivitaet").hidden = Z.ansicht !== "sensitivitaet";
  }
  const aktuellerStandort = () => (LAEUFE.get(Z.lauf) || K.laeufe[0]).standort;

  function umschalter(id, wert, beiWahl) {
    const el = $(id);
    el.querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", b.dataset.wert === wert);
      b.onclick = () => beiWahl(b.dataset.wert);
    });
  }

  const ROLLEN_FOLGE = ["referenz", "variante", "sensitivitaet", "validierung", "experiment"];
  const GRUPPEN_FOLGE = Object.keys(K.gruppen);
  function laufOptionen(select, gewaehlt, filter) {
    select.innerHTML = "";
    for (const rolle of ROLLEN_FOLGE) {
      const ls = K.laeufe.filter((l) => l.rolle === rolle && l.app_json && (!filter || filter(l)));
      if (!ls.length) continue;
      const grp = document.createElement("optgroup");
      grp.label = L(K.rollen[rolle]);
      ls.sort((a, b) => (GRUPPEN_FOLGE.indexOf(gruppeVon(a)) - GRUPPEN_FOLGE.indexOf(gruppeVon(b))) ||
        L(a.titel).localeCompare(L(b.titel), locale()));
      for (const l of ls) {
        const o = document.createElement("option");
        o.value = l.id;
        o.textContent = (l.id === K.standard_lauf ? "★ " : "") + L(l.titel);
        o.selected = l.id === gewaehlt;
        grp.appendChild(o);
      }
      select.appendChild(grp);
    }
  }
  const gruppeVon = (l) => (l.faktor && FAKTOREN.has(l.faktor) ? FAKTOREN.get(l.faktor).gruppe : "referenz");
  const stufeLabel = (l, faktor) => {
    const f = FAKTOREN.get(faktor);
    const s = f && l.stufen && f.stufen.find((x) => x.key === l.stufen[faktor]);
    return s ? L(s.label) : "–";
  };

  // ---------------------------------------------------------------------------------------------------------------
  // Ansicht Rekonstruktion

  let rekToken = 0;
  let rekDetail = null; // Index des gewaehlten Alters bzw. der Tiefe

  function rekInfo(l, d) {
    const teile = [`<p>${esc(L(l.beschreibung))}</p><dl>`];
    teile.push(`<dt>${t("rolle")}</dt><dd>${esc(L(K.rollen[l.rolle]))}</dd>`);
    if (l.faktor) {
      const f = FAKTOREN.get(l.faktor);
      teile.push(`<dt>${esc(L(f.label))}</dt><dd>${esc(stufeLabel(l, l.faktor))}</dd>`);
    }
    if (l.basis && LAEUFE.has(l.basis)) teile.push(`<dt>${t("basis")}</dt><dd>${esc(L(LAEUFE.get(l.basis).titel))}</dd>`);
    if (l.quelle) teile.push(`<dt>${t("quelle")}</dt><dd>${esc(l.quelle)}</dd>`);
    if (d) {
      const kz = d.kennzahlen;
      teile.push(`<dt>${t("spalte_T")}</dt><dd>${fmt(kz.T_niveau, 2)} °C</dd><dt>${t("spalte_P")}</dt><dd>${fmt(kz.P_niveau, 0)} mm</dd>`);
      teile.push(`<dt>${t("spalte_ess")}</dt><dd>${fmt(kz.ess_min, 0)}</dd>`);
    }
    teile.push("</dl>");
    $("rek-info").innerHTML = teile.join("");
    $("rek-hinweis").innerHTML = l.hinweis ? `<span class="warn">⚠ ${esc(L(l.hinweis))}</span>` : "";
  }

  async function zeigeRekonstruktion() {
    const token = ++rekToken;
    const l = LAEUFE.get(Z.lauf);
    laufOptionen($("rek-lauf"), Z.lauf);
    let d, st;
    try {
      [d, st] = await Promise.all([ladeLauf(Z.lauf), ladeStandort(l.standort)]);
    } catch (e) {
      $("rek-info").innerHTML = `<span class="warn">${t("fehler_laden")}: ${esc(e.message)}</span>`;
      return;
    }
    if (token !== rekToken) return;
    const art = Z.art || d.rekonstruktion.standard;
    umschalter("rek-achse", Z.achse, (w) => { Z.achse = w; rekDetail = null; merke(); zeigeRekonstruktion(); });
    umschalter("rek-art", art, (w) => { Z.art = w === d.rekonstruktion.standard ? null : w; merke(); zeigeRekonstruktion(); });
    $("rek-art").querySelectorAll("button").forEach((b) => {
      b.innerHTML = esc(t(b.dataset.wert === "mean" ? "mittelwert" : "mischverteilung")) +
        (b.dataset.wert === d.rekonstruktion.standard ? `<span class="klein">${esc(t("standard_des_laufs"))}</span>` : "");
    });
    rekInfo(l, d);
    zeichneRekonstruktion(d, st, art);
    if (rekDetail !== null) zeigeDetail(d, st, art, rekDetail);
    else $("rek-detail").hidden = true;
  }

  function zeichneRekonstruktion(d, st, art) {
    const rk = d.rekonstruktion[Z.achse];
    const x = rk.x;
    const farbeLauf = farbe(Z.lauf);
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
          zmin: 0, zmax: 100, hovertemplate: `${t(v + "_kurz")} %{y}: %{z}<extra></extra>` });
        achsenBereich[ys[r]] = lo >= 0 ? [g[Math.max(0, lo - 1)], g[Math.min(g.length - 1, hi + 1)]] : undefined;
      } else {
        traces.push(...band(x, q.q2_5, q.q97_5, farbeLauf, 0.15, ys[r], t("band95")));
        traces.push(...band(x, q.q25, q.q75, farbeLauf, 0.32, ys[r], t("band50")));
        traces.push({ x, y: q.q50, yaxis: ys[r], type: "scatter", mode: "lines", line: { color: farbeLauf, width: 2 },
          name: t("median"), hovertemplate: `${t(v + "_kurz")} %{y:.${v === "T" ? 2 : 0}f}<extra></extra>` });
        achsenBereich[ys[r]] = undefined;
      }
    });
    const bio = d.biome[Z.achse];
    st.biome.farben.forEach((c, b) => {
      traces.push(...band(x, bio.q25[b], bio.q75[b], c, 0.3, "y3", st.biome.labels[b]));
      traces.push({ x, y: bio.q50[b], yaxis: "y3", type: "scatter", mode: "lines", line: { color: c, width: 2 },
        name: st.biome.labels[b], hovertemplate: `${st.biome.labels[b]} %{y:.2f}<extra></extra>` });
    });
    const ap = Z.achse === "alter" ? d.ap.alter : d.ap.tiefe;
    if (d.ap.ausgeschlossen.length) {
      traces.push({ x, y: Z.achse === "alter" ? st.ap.alter : st.ap.tiefe, yaxis: "y4", type: "scatter", mode: "lines",
        line: { color: css("--muted"), width: 1, dash: "dot" }, name: t("ap_voll"),
        hovertemplate: `${t("ap_voll")} %{y:.1f}<extra></extra>` });
    }
    traces.push({ x, y: ap, yaxis: "y4", type: "scatter", mode: "lines", line: { color: css("--fg"), width: 1.5 },
      name: t("ap"), hovertemplate: `AP %{y:.1f}<extra></extra>` });

    const shapes = [];
    if (Z.achse === "alter" && d.ap.kalibrierung_alter) {
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
      ann.push({ text: t("ap_kalibrierung"), xref: "paper", yref: "paper", x: 1, y: dom[3][1], xanchor: "right",
        yanchor: "bottom", showarrow: false, font: { size: 11, color: css("--muted") } });
    }
    if (rekDetail !== null && rekDetail < x.length) {
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: x[rekDetail], x1: x[rekDetail], y0: 0, y1: 1,
        line: { color: css("--accent"), width: 1.5, dash: "dot" } });
    }
    const titelText = (v) => `${v === "T" ? "(a)" : "(b)"} ${t(v)} · ${t(art === "mean" ? "mittelwert" : "mischverteilung")}`;
    ann.push(titel(titelText("T"), "y domain"), titel(titelText("P"), "y2 domain"),
      titel("(c) " + t("biome") + " · " + st.biome.labels.join(" / "), "y3 domain"), titel("(d) " + t("ap"), "y4 domain"));
    const lay = layout({
      hovermode: "x", shapes, annotations: ann,
      xaxis: achse({ title: { text: t(Z.achse === "alter" ? "alter_achse" : "tiefe_achse") }, anchor: "y4" }),
      yaxis: achse({ domain: dom[0], title: { text: "°C" }, range: achsenBereich.y }),
      yaxis2: achse({ domain: dom[1], title: { text: "mm" }, range: achsenBereich.y2, rangemode: "tozero" }),
      yaxis3: achse({ domain: dom[2], range: [0, 1], title: { text: t("anteil") } }),
      yaxis4: achse({ domain: dom[3], title: { text: "%" } }),
    });
    ["yaxis", "yaxis2", "yaxis3", "yaxis4"].forEach((a) => { if (!lay[a].range) delete lay[a].range; });
    const div = $("rek-plot");
    Plotly.react(div, traces, lay, PLOT_CONFIG).then(() => {
      if (!div._krKlick) {
        div._krKlick = true;
        div.on("plotly_click", (ev) => {
          if (!ev.points || !ev.points.length) return;
          const xs = D["lauf/" + Z.lauf].rekonstruktion[Z.achse].x;
          const xv = ev.points[0].x;
          let best = 0;
          xs.forEach((v, i) => { if (Math.abs(v - xv) < Math.abs(xs[best] - xv)) best = i; });
          rekDetail = best;
          zeigeRekonstruktion();
        });
      }
    });
  }
  const achsenBereich = {};

  function zeigeDetail(d, st, art, i) {
    const rk = d.rekonstruktion[Z.achse];
    $("rek-detail").hidden = false;
    $("rek-detail-titel").textContent = t(Z.achse === "alter" ? "detail_alter" : "detail_tiefe").replace("{x}", fmt(rk.x[i], 0));
    $("rek-detail-zu").onclick = () => { rekDetail = null; zeigeRekonstruktion(); };
    const c = farbe(Z.lauf);
    // Verteilung von T und P: Dichte der Mischverteilung, darueber Median, 50 % und 95 % des Mittelwerts
    const traces = [];
    ["T", "P"].forEach((v, k) => {
      const xa = k ? "x2" : "x", ya = k ? "y2" : "y";
      const g = d.rekonstruktion.raster[v];
      traces.push({ x: g, y: rk.mixture.dichte[v][i].map((z) => z / 100), xaxis: xa, yaxis: ya, type: "scatter",
        mode: "lines", fill: "tozeroy", line: { color: css("--accent"), width: 1 }, fillcolor: alpha("#2a6fb0", 0.25),
        name: t("mischverteilung"), hovertemplate: `${t("mischverteilung")} %{x}: %{y:.2f}<extra></extra>` });
      const q = rk.mean[v];
      traces.push({ x: [q.q50[i]], y: [1.12], xaxis: xa, yaxis: ya, type: "scatter", mode: "markers",
        marker: { color: c, size: 8 }, name: t("mittelwert"),
        error_x: { type: "data", symmetric: false, array: [q.q97_5[i] - q.q50[i]], arrayminus: [q.q50[i] - q.q2_5[i]],
          color: c, thickness: 1.2, width: 4 },
        hovertemplate: `${t("mittelwert")}: %{x:.${v === "T" ? 2 : 0}f}<extra></extra>` });
      traces.push({ x: [q.q25[i], q.q75[i]], y: [1.12, 1.12], xaxis: xa, yaxis: ya, type: "scatter", mode: "lines",
        line: { color: c, width: 6 }, hoverinfo: "skip" });
    });
    const sichtbar = (v) => {
      const g = d.rekonstruktion.raster[v], z = rk.mixture.dichte[v][i], q = rk.mean[v];
      let lo = q.q2_5[i], hi = q.q97_5[i];
      z.forEach((w, j) => { if (w >= 2) { lo = Math.min(lo, g[j]); hi = Math.max(hi, g[j]); } });
      const pad = (hi - lo) * 0.08 || 1;
      return [lo - pad, hi + pad];
    };
    Plotly.react($("rek-detail-verteilung"), traces, layout({
      margin: { l: 34, r: 10, t: 26, b: 40 },
      xaxis: achse({ domain: [0, 0.46], title: { text: t("T") }, range: sichtbar("T") }),
      xaxis2: achse({ domain: [0.54, 1], title: { text: t("P") }, range: sichtbar("P") }),
      yaxis: achse({ range: [0, 1.25], showticklabels: false, title: { text: t("dichte") } }),
      yaxis2: achse({ range: [0, 1.25], showticklabels: false, anchor: "x2" }),
      annotations: [titel(t("verteilung"), "paper")],
    }), PLOT_CONFIG);
    // Biomanteile: Median mit 95-%-Intervall
    const bio = d.biome[Z.achse];
    Plotly.react($("rek-detail-biome"), [{
      type: "bar", x: st.biome.labels, y: st.biome.labels.map((_, b) => bio.q50[b][i]),
      marker: { color: st.biome.farben },
      error_y: { type: "data", symmetric: false, color: css("--fg"),
        array: st.biome.labels.map((_, b) => bio.q97_5[b][i] - bio.q50[b][i]),
        arrayminus: st.biome.labels.map((_, b) => bio.q50[b][i] - bio.q2_5[b][i]) },
      hovertemplate: "%{x}: %{y:.2f}<extra></extra>",
    }], layout({ margin: { l: 44, r: 10, t: 26, b: 30 }, yaxis: achse({ range: [0, 1], title: { text: t("anteil") } }),
      xaxis: achse({}), annotations: [titel(t("biome"), "paper")] }), PLOT_CONFIG);
    // Spektrum: haeufigste Taxa, Farbe nach Biom
    const sp = (Z.achse === "alter" ? st.spektrum_alter : st.spektrum_tiefe)[i];
    const idx = sp.map((v, k) => k).sort((a, b) => sp[b] - sp[a]).slice(0, 12);
    Plotly.react($("rek-detail-spektrum"), [{
      type: "bar", orientation: "h", y: idx.map((k) => st.taxa.namen[k]), x: idx.map((k) => sp[k]),
      marker: { color: idx.map((k) => st.biome.farben[st.taxa.biom[k]]) },
      hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
    }], layout({ margin: { l: 10, r: 10, t: 26, b: 34 }, xaxis: achse({ title: { text: t("anteil") } }),
      yaxis: achse({ autorange: "reversed", tickfont: { size: 11 } }), annotations: [titel(t("spektrum"), "paper")] }),
    PLOT_CONFIG);
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Ansicht Sensitivitaet

  let sensToken = 0;

  function serie(faktor) {
    const ids = new Set();
    K.laeufe.forEach((l) => {
      if (l.faktor === faktor) {
        ids.add(l.id);
        if (l.basis) ids.add(l.basis);
      }
    });
    return K.laeufe.filter((l) => ids.has(l.id) && l.app_json);
  }
  function stammtVon(id, basis) {
    for (let l = LAEUFE.get(id), n = 0; l && n < 50; l = LAEUFE.get(l.basis), n++) {
      if (l.basis === basis) return true;
    }
    return false;
  }
  function standardBasis(s) {
    const ids = new Set(s.map((l) => l.id));
    const wurzeln = s.filter((l) => !l.basis || !ids.has(l.basis));
    const n = (b) => s.filter((l) => stammtVon(l.id, b.id)).length;
    wurzeln.sort((a, b) => (n(b) - n(a)) || ((b.id === K.standard_lauf) - (a.id === K.standard_lauf)) ||
      (stammtVon(K.standard_lauf, b.id) - stammtVon(K.standard_lauf, a.id)));
    return wurzeln.length ? wurzeln[0].id : s[0].id;
  }

  function faktorOptionen() {
    const sel = $("sens-faktor");
    sel.innerHTML = "";
    for (const g of GRUPPEN_FOLGE) {
      const fs = K.faktoren.filter((f) => f.gruppe === g && serie(f.key).length > 1);
      if (!fs.length) continue;
      const grp = document.createElement("optgroup");
      grp.label = L(K.gruppen[g]);
      fs.forEach((f) => {
        const o = document.createElement("option");
        o.value = f.key;
        o.textContent = L(f.label);
        o.selected = f.key === Z.faktor;
        grp.appendChild(o);
      });
      sel.appendChild(grp);
    }
  }

  function sensAuswahl() {
    const s = serie(Z.faktor);
    if (!Z.basis || !s.some((l) => l.id === Z.basis)) Z.basis = standardBasis(s);
    if (!Z.auswahl || Z.auswahl.faktor !== Z.faktor || Z.auswahl.basis !== Z.basis) {
      Z.auswahl = { faktor: Z.faktor, basis: Z.basis, laeufe: s.filter((l) => stammtVon(l.id, Z.basis)).map((l) => l.id) };
      Z.extra = [];
    }
    return s;
  }

  async function zeigeSensitivitaet() {
    const token = ++sensToken;
    faktorOptionen();
    const s = sensAuswahl();
    const f = FAKTOREN.get(Z.faktor);
    $("sens-faktor-text").textContent = L(f.beschreibung);
    // Basislauf: alle Laeufe der Reihe
    const bsel = $("sens-basis");
    bsel.innerHTML = "";
    s.forEach((l) => {
      const o = document.createElement("option");
      o.value = l.id;
      o.textContent = L(l.titel) + " · " + stufeLabel(l, Z.faktor);
      o.selected = l.id === Z.basis;
      bsel.appendChild(o);
    });
    // Liste der Laeufe mit Farben
    const kandidaten = [...s.filter((l) => l.id !== Z.basis), ...Z.extra.map((id) => LAEUFE.get(id)).filter(Boolean)];
    const liste = $("sens-laeufe");
    liste.innerHTML = "";
    kandidaten.forEach((l) => {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = Z.auswahl.laeufe.includes(l.id) || Z.extra.includes(l.id);
      cb.onchange = () => {
        if (Z.extra.includes(l.id) && !cb.checked) Z.extra = Z.extra.filter((x) => x !== l.id);
        else if (cb.checked) Z.auswahl.laeufe.push(l.id);
        else Z.auswahl.laeufe = Z.auswahl.laeufe.filter((x) => x !== l.id);
        merke();
        zeigeSensitivitaet();
      };
      const chip = document.createElement("span");
      chip.className = "farbe";
      chip.style.background = cb.checked ? farbe(l.id) : "transparent";
      const txt = document.createElement("span");
      txt.textContent = L(l.titel) + (l.hinweis ? " ⚠" : "");
      if (l.hinweis) lab.title = L(l.hinweis);
      lab.append(cb, chip, txt);
      liste.appendChild(lab);
    });
    const wsel = $("sens-weitere");
    laufOptionen(wsel, null, (l) => !s.includes(l) && !Z.extra.includes(l.id));
    const leer = document.createElement("option");
    leer.value = "";
    leer.textContent = t("weitere_lauf");
    leer.selected = true;
    wsel.prepend(leer);
    $("sens-baender").checked = !!Z.baender;

    const gezeigt = [Z.basis, ...kandidaten.filter((l) => Z.auswahl.laeufe.includes(l.id) || Z.extra.includes(l.id)).map((l) => l.id)];
    let daten;
    try {
      daten = await Promise.all(gezeigt.map(ladeLauf));
    } catch (e) {
      $("sens-tabelle").innerHTML = `<caption class="warn">${t("fehler_laden")}: ${esc(e.message)}</caption>`;
      return;
    }
    if (token !== sensToken) return;
    zeichneSensitivitaet(gezeigt, daten);
    sensTabelle(gezeigt, daten);
  }

  const kurve = (d, v) => d.rekonstruktion.alter[d.rekonstruktion.standard][v];
  const mittel = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  function korrelation(a, b) {
    const ma = mittel(a), mb = mittel(b);
    let sab = 0, saa = 0, sbb = 0;
    a.forEach((x, i) => { sab += (x - ma) * (b[i] - mb); saa += (x - ma) ** 2; sbb += (b[i] - mb) ** 2; });
    return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : NaN;
  }
  const iqr = (q) => mittel(q.q75.map((x, i) => x - q.q25[i]));

  function zeichneSensitivitaet(ids, daten) {
    const x = daten[0].rekonstruktion.alter.x;
    const dom = zeilen([1, 1], 0.08);
    const traces = [], diff = [];
    ids.forEach((id, k) => {
      const d = daten[k], basis = k === 0;
      const c = basis ? css("--base") : farbe(id);
      const name = L(LAEUFE.get(id).titel);
      ["T", "P"].forEach((v, r) => {
        const q = kurve(d, v), ya = r ? "y2" : "y";
        if (Z.baender) traces.push(...band(x, q.q25, q.q75, c.startsWith("#") ? c : "#888888", basis ? 0.18 : 0.14, ya, name));
        traces.push({ x, y: q.q50, yaxis: ya, type: "scatter", mode: "lines", name,
          line: { color: c, width: basis ? 3 : 1.8 },
          hovertemplate: `${esc(name)}: %{y:.${v === "T" ? 2 : 0}f}<extra></extra>` });
        if (!basis) {
          const b = kurve(daten[0], v).q50;
          diff.push({ x, y: q.q50.map((y, i) => y - b[i]), yaxis: ya, type: "scatter", mode: "lines", name,
            line: { color: c, width: 1.8 }, hovertemplate: `${esc(name)}: %{y:+.${v === "T" ? 2 : 0}f}<extra></extra>` });
        }
      });
    });
    const xa = achse({ title: { text: t("alter_achse") }, anchor: "y2" });
    Plotly.react($("sens-plot"), traces, layout({
      hovermode: "x", xaxis: xa,
      yaxis: achse({ domain: dom[0], title: { text: t("T") } }), yaxis2: achse({ domain: dom[1], title: { text: t("P") } }),
      annotations: [titel("(a) " + t("T") + " · " + t("median"), "y domain"), titel("(b) " + t("P") + " · " + t("median"), "y2 domain")],
    }), PLOT_CONFIG);
    Plotly.react($("sens-diff"), diff, layout({
      hovermode: "x", xaxis: achse({ title: { text: t("alter_achse") }, anchor: "y2" }),
      yaxis: achse({ domain: dom[0], title: { text: t("dT") }, zerolinecolor: css("--muted"), zerolinewidth: 1.5 }),
      yaxis2: achse({ domain: dom[1], title: { text: t("dP") }, zerolinecolor: css("--muted"), zerolinewidth: 1.5 }),
      annotations: [titel("(c) " + t("dT"), "y domain"), titel("(d) " + t("dP"), "y2 domain")],
    }), PLOT_CONFIG);
  }

  function sensTabelle(ids, daten) {
    const kopf = ["spalte_lauf", "spalte_stufe", "spalte_T", "spalte_P", "spalte_dT", "spalte_dP", "spalte_iqrT",
      "spalte_iqrP", "spalte_rT", "spalte_rP", "spalte_ess"].map((k) => `<th>${esc(t(k))}</th>`).join("");
    const b = daten[0];
    const zeilenHtml = ids.map((id, k) => {
      const d = daten[k], l = LAEUFE.get(id);
      const tq = kurve(d, "T"), pq = kurve(d, "P"), tb = kurve(b, "T"), pb = kurve(b, "P");
      const it = iqr(tq), ip = iqr(pq), itb = iqr(tb), ipb = iqr(pb);
      const rel = (a, c) => (k === 0 ? "" : ` (${fmtS((a / c - 1) * 100, 0)} %)`);
      const chip = `<span class="farbe" style="background:${k === 0 ? css("--base") : farbe(id)}"></span> `;
      return `<tr class="${k === 0 ? "basis" : ""}">` +
        `<td>${chip}${esc(L(l.titel))}${l.hinweis ? ` <span class="warn-inline" title="${esc(L(l.hinweis))}">⚠</span>` : ""}</td>` +
        `<td>${esc(stufeLabel(l, Z.faktor))}</td>` +
        `<td>${fmt(mittel(tq.q50), 2)}</td><td>${fmt(mittel(pq.q50), 0)}</td>` +
        `<td>${k ? fmtS(mittel(tq.q50) - mittel(tb.q50), 2) : "–"}</td>` +
        `<td>${k ? fmtS(mittel(pq.q50) - mittel(pb.q50), 0) : "–"}</td>` +
        `<td>${fmt(it, 2)}${rel(it, itb)}</td><td>${fmt(ip, 0)}${rel(ip, ipb)}</td>` +
        `<td>${k ? fmt(korrelation(tq.q50, tb.q50), 3) : "1"}</td><td>${k ? fmt(korrelation(pq.q50, pb.q50), 3) : "1"}</td>` +
        `<td>${fmt(d.kennzahlen.ess_min, 0)}</td></tr>`;
    }).join("");
    $("sens-tabelle").innerHTML = `<thead><tr>${kopf}</tr></thead><tbody>${zeilenHtml}</tbody>`;
  }

  // ---------------------------------------------------------------------------------------------------------------

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function zeige() {
    setzeAnsicht();
    if (Z.ansicht === "rekonstruktion") zeigeRekonstruktion();
    else zeigeSensitivitaet();
  }

  function start() {
    setzeThema();
    setzeTexte();
    document.querySelectorAll(".reiter button").forEach((b) => {
      b.onclick = () => { Z.ansicht = b.dataset.ansicht; merke(); zeige(); };
    });
    $("sprache").onclick = () => { Z.sprache = Z.sprache === "de" ? "en" : "de"; merke(); setzeTexte(); setzeThema(); zeige(); };
    $("rek-info-box").open = window.matchMedia("(min-width: 1000px)").matches;
    $("thema").onclick = () => {
      Z.thema = { auto: "light", light: "dark", dark: "auto" }[Z.thema];
      merke(); setzeThema(); zeige();
    };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (Z.thema === "auto") zeige(); });
    $("rek-lauf").onchange = (e) => { Z.lauf = e.target.value; Z.art = null; merke(); zeigeRekonstruktion(); };
    $("sens-faktor").onchange = (e) => { Z.faktor = e.target.value; Z.basis = null; Z.auswahl = null; merke(); zeigeSensitivitaet(); };
    $("sens-basis").onchange = (e) => { Z.basis = e.target.value; Z.auswahl = null; merke(); zeigeSensitivitaet(); };
    $("sens-weitere").onchange = (e) => { if (e.target.value) { Z.extra.push(e.target.value); merke(); zeigeSensitivitaet(); } };
    $("sens-baender").onchange = (e) => { Z.baender = e.target.checked; merke(); zeigeSensitivitaet(); };
    ladeStandort(aktuellerStandort()).then((st) => {
      $("standort-name").textContent = L(st.name);
      setzeTexte();
    }).catch(() => {});
    zeige();
    const eineDatei = !!document.querySelector('script[type="application/gzip-base64"]');
    const sicher = location.protocol === "https:" || ["localhost", "127.0.0.1"].includes(location.hostname);
    if ("serviceWorker" in navigator && sicher && !eineDatei) navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  window.KR_APP = { zustand: Z, ladeLauf, ladeStandort }; // fuer Tests
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
