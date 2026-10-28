// Kinneret-App, Kern: Zustand, Daten, Farben, Plotly-Helfer, Texte (docs/app_konzept.md). Die Ansichten liegen in
// eigenen Dateien und melden sich mit KR.ansicht({key, init, zeige}) an; die Reihenfolge der Skripte ist die
// Reihenfolge der Reiter. Laeuft offline, auch ueber file://.
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
    faktor: "klima_prior", basis: null, auswahl: null, extra: [], baender: false, vergleich: [],
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
  // Titel oben links ueber einem Feld; ref/xref z. B. "y2 domain"/"x2 domain" fuer Felder nebeneinander
  function titel(text, ref, xref) {
    return { text, xref: xref || "paper", yref: ref, x: 0, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false,
      font: { size: 12, color: css("--muted") } };
  }
  function band(x, lo, hi, color, a, yaxis, name) {
    return [
      { x, y: lo, yaxis, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x, y: hi, yaxis, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: alpha(color, a), hoverinfo: "skip", name, showlegend: false },
    ];
  }

  // // Texte, Thema, Reiter

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
    ansichten.forEach((a) => { $("ansicht-" + a.key).hidden = a.key !== Z.ansicht; });
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

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // Vergleichslaeufe (gemeinsam fuer mehrere Ansichten): Chips zum Entfernen, Auswahl zum Hinzufuegen
  const MAX_VERGLEICH = 6;
  function vergleichsAuswahl(container, beiAenderung) {
    Z.vergleich = Z.vergleich.filter((id) => id !== Z.lauf && LAEUFE.has(id) && LAEUFE.get(id).app_json);
    container.innerHTML = "";
    const chips = document.createElement("div");
    chips.className = "chips";
    Z.vergleich.forEach((id) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.innerHTML = `<span class="farbe" style="background:${farbe(id)}"></span>${esc(L(LAEUFE.get(id).titel))}`;
      const x = document.createElement("button");
      x.textContent = "✕";
      x.title = t("entfernen");
      x.onclick = () => { Z.vergleich = Z.vergleich.filter((v) => v !== id); merke(); beiAenderung(); };
      c.appendChild(x);
      chips.appendChild(c);
    });
    container.appendChild(chips);
    if (Z.vergleich.length < MAX_VERGLEICH) {
      const sel = document.createElement("select");
      laufOptionen(sel, null, (l) => l.id !== Z.lauf && !Z.vergleich.includes(l.id));
      const leer = document.createElement("option");
      leer.value = "";
      leer.textContent = t("vergleich_hinzu");
      leer.selected = true;
      sel.prepend(leer);
      sel.onchange = () => { if (sel.value) { Z.vergleich.push(sel.value); merke(); beiAenderung(); } };
      container.appendChild(sel);
    }
    return [Z.lauf, ...Z.vergleich];
  }
  const schmal = (el) => el.clientWidth < 700;

  // Register der Ansichten
  const ansichten = [];
  window.KR = {
    D, K, Z, TX, $, t, L, css, locale, fmt, fmtS, esc, merke, LAEUFE, FAKTOREN, ladeLauf, ladeStandort, farbe, alpha,
    dunkel, dichteSkala, PLOT_CONFIG, achse, layout, zeilen, titel, band, setzeTexte, setzeThema, setzeAnsicht,
    aktuellerStandort, umschalter, laufOptionen, gruppeVon, stufeLabel, ROLLEN_FOLGE, GRUPPEN_FOLGE, ansichten,
    vergleichsAuswahl, schmal, ansicht: (a) => ansichten.push(a),
  };
})();
