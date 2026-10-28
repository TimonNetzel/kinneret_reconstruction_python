// Export: Abbildungen im Format von Climate of the Past (Copernicus) als PDF, SVG oder PNG und die Daten der
// Abbildung als CSV (docs/app_konzept.md, Abschnitt 12). Jedes mit KR.plot gezeichnete Diagramm ist exportierbar: Die
// Ansicht baut es im Druckstil neu (KR.mitDruck: hell, Masse in pt, eine Schriftgroesse), hier werden Linien und
// Schriften angepasst und die Datei geschrieben. Laeuft offline, ohne weitere Bibliotheken.
(function (KR) {
  "use strict";
  const { $, D, Z, t, L, LAEUFE, merke, umschalter, aktuellerStandort, fmt, F, FT } = KR;
  const tx = (k) => F(t(k));

  const PT_JE_CM = 72 / 2.54; // Diagramme werden in pt gebaut (1 Einheit = 1 pt)
  const BREITE = { min: 4, max: 17.5 }, HOEHE = { min: 3, max: 23 }, SCHRIFT = { min: 6, max: 12 };
  const LINIE = 0.6; // Linienstaerken: Bildschirm-px -> pt bei Abbildungsgroesse
  const einstellung = () => (Z.export = Object.assign({ breite: 17.5, schrift: 8, dpi: 300, csv: "int", kopf: true }, Z.export));
  const begrenze = (x, g) => Math.min(g.max, Math.max(g.min, x));

  let figur = null; // Diagramm (div mit _krFigur)
  let hoehe = null; // von Hand gesetzte Hoehe in cm, sonst Voreinstellung der Abbildung
  let vorschauToken = 0, vorschauUhr = null;

  // ---------------------------------------------------------------------------------------------------------------
  // Abbildung im Druckstil

  function figuren() {
    return [...document.querySelectorAll(`#ansicht-${Z.ansicht} .plot`)].filter((d) => d._krFigur && d.offsetParent !== null);
  }
  const masse = () => {
    const e = einstellung();
    const b = begrenze(e.breite, BREITE);
    return { b, h: begrenze(hoehe ?? figur._krFigur.hoehe(b), HOEHE), s: begrenze(e.schrift, SCHRIFT) };
  };

  function duenner(tr) {
    if (tr.line) tr.line.width = tr.line.width === 0 ? 0 : Math.max(0.4, (tr.line.width ?? 2) * LINIE);
    else if (tr.type === "scatter" && /lines/.test(tr.mode || "")) tr.line = { width: 2 * LINIE };
    if (tr.marker) {
      if (tr.marker.size) tr.marker.size = Math.max(2, tr.marker.size * 0.65);
      if (tr.marker.line && tr.marker.line.width) tr.marker.line.width *= LINIE;
    }
    [tr.error_x, tr.error_y].forEach((e) => { if (e) { e.thickness = (e.thickness ?? 2) * LINIE; e.width = (e.width ?? 4) * LINIE; } });
  }
  // Schriften: Titel und Beschriftungen in der gewaehlten Groesse, kleinere (Achsenzahlen, Hinweise) 1 pt kleiner
  function schriften(o, s, oben) {
    if (!o || typeof o !== "object") return;
    for (const [k, v] of Object.entries(o)) {
      if ((k === "font" || k === "tickfont") && v && v.size && !oben) v.size = v.size >= 12 ? s : Math.max(5, s - 1);
      else if (typeof v === "object") schriften(v, s, false);
    }
  }

  // Legende als Textzeilen ueber dem Diagramm (Plotlys Legende braucht je Eintrag mindestens 19 Einheiten Hoehe und
  // gleich breite Spalten; bei 8,3 cm Breite zu sperrig). Eintraege: Spuren mit Namen und showlegend !== false.
  function legende(data, layout, m) {
    const f = m.s - 1, breite = m.b * PT_JE_CM - 45, zeichen = (h) => h.replace(/<[^>]+>/g, "").length * 0.55 * f;
    const eintraege = [];
    data.forEach((tr) => {
      if (tr.showlegend === false || !tr.name) return;
      const farbe = (tr.line && tr.line.color) || (tr.marker && typeof tr.marker.color === "string" && tr.marker.color) || "#000000";
      const linie = tr.type === "contour" || /lines/.test(tr.mode || "");
      const symbol = tr.type === "bar" ? "■" : !linie ? "●" : tr.line && tr.line.dash ? "╸ ╸" : "━━";
      eintraege.push(`<span style="color:${farbe}">${symbol}</span> ${tr.name}`);
    });
    if (!eintraege.length) return;
    const zeilen = [[]];
    let frei = breite;
    eintraege.forEach((e) => {
      const w = zeichen(e) + 2.5 * f;
      if (zeilen[zeilen.length - 1].length && w > frei) { zeilen.push([]); frei = breite; }
      zeilen[zeilen.length - 1].push(e);
      frei -= w;
    });
    const hoeheZeilen = zeilen.length * f * 1.3;
    layout.margin.t += Math.ceil(hoeheZeilen + 2);
    layout.annotations = [...(layout.annotations || []), {
      text: zeilen.map((z) => z.join("   ")).join("<br>"), xref: "paper", yref: "paper", x: 0, y: 1, xanchor: "left",
      yanchor: "bottom", yshift: Math.round(m.s * 1.7), align: "left", showarrow: false, font: { size: f, color: "#000000" },
    }];
  }

  function abbildung(div, m) {
    const [traces, lay] = KR.mitDruck({ breite: m.b, schrift: m.s }, div._krFigur.baue);
    const data = JSON.parse(JSON.stringify(traces)), layout = JSON.parse(JSON.stringify(lay));
    data.forEach(duenner);
    (layout.shapes || []).forEach((sh) => { if (sh.line && sh.line.width) sh.line.width = Math.max(0.4, sh.line.width * LINIE); });
    schriften(layout, m.s, true);
    if (layout.showlegend) legende(data, layout, m);
    layout.showlegend = false;
    Object.assign(layout, { width: Math.round(m.b * PT_JE_CM), height: Math.round(m.h * PT_JE_CM) });
    return { data, layout, config: { staticPlot: true } };
  }

  // SVG mit physischer Groesse in cm (viewBox in pt), damit Schriften und Linien in pt stimmen
  async function svgText(fig, m) {
    const url = await Plotly.toImage(fig, { format: "svg", width: fig.layout.width, height: fig.layout.height });
    const komma = url.indexOf(",");
    const roh = /;base64$/.test(url.slice(0, komma)) ? atob(url.slice(komma + 1)) : decodeURIComponent(url.slice(komma + 1));
    const doc = new DOMParser().parseFromString(roh, "image/svg+xml");
    const svg = doc.documentElement;
    svg.setAttribute("viewBox", `0 0 ${fig.layout.width} ${fig.layout.height}`);
    svg.setAttribute("width", `${m.b}cm`);
    svg.setAttribute("height", `${m.h}cm`);
    return new XMLSerializer().serializeToString(svg);
  }

  // PNG mit Aufloesung im pHYs-Block (sonst nehmen Word und LaTeX 72 oder 96 dpi an)
  const CRC = (() => {
    const tab = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      tab[n] = c >>> 0;
    }
    return (bytes) => {
      let c = 0xffffffff;
      for (const b of bytes) c = tab[(c ^ b) & 255] ^ (c >>> 8);
      return (c ^ 0xffffffff) >>> 0;
    };
  })();
  function mitDpi(png, dpi) {
    const ppm = Math.round(dpi / 0.0254);
    const block = new Uint8Array(21);
    const v = new DataView(block.buffer);
    v.setUint32(0, 9);
    block.set([112, 72, 89, 115], 4); // "pHYs"
    v.setUint32(8, ppm);
    v.setUint32(12, ppm);
    block[16] = 1; // Einheit: Meter
    v.setUint32(17, CRC(block.subarray(4, 17)));
    const ende = 33; // Signatur (8) + IHDR (25)
    const out = new Uint8Array(png.length + block.length);
    out.set(png.subarray(0, ende));
    out.set(block, ende);
    out.set(png.subarray(ende), ende + block.length);
    return out;
  }
  const pixel = (cm, dpi) => Math.round((cm / 2.54) * dpi);
  async function pngBlob(fig, dpi, m) {
    // Massstab so, dass die Breite in Pixeln genau Breite x Aufloesung ist (17,5 cm bei 300 dpi: 2067 px)
    const scale = pixel(m.b, dpi) / fig.layout.width;
    const url = await Plotly.toImage(fig, { format: "png", width: fig.layout.width, height: fig.layout.height, scale });
    const bin = Uint8Array.from(atob(url.slice(url.indexOf(",") + 1)), (c) => c.charCodeAt(0));
    return new Blob([mitDpi(bin, dpi)], { type: "image/png" });
  }

  // PDF ueber den Druckdialog des Browsers („Als PDF speichern“): Seite = Abbildung, Vektorgrafik, Schriften eingebettet
  function drucke(svg, m, name) {
    const bereich = $("druck"), seite = $("druck-seite"), titel = document.title;
    bereich.innerHTML = svg;
    seite.textContent = `@page { size: ${m.b}cm ${m.h}cm; margin: 0 }`;
    document.body.classList.add("druckt");
    document.title = name; // Chrome und Edge schlagen den Titel als Dateinamen vor
    const ende = () => {
      window.removeEventListener("afterprint", ende);
      document.body.classList.remove("druckt");
      bereich.innerHTML = "";
      seite.textContent = "";
      document.title = titel;
    };
    window.addEventListener("afterprint", ende);
    window.print();
  }

  // ---------------------------------------------------------------------------------------------------------------
  // CSV der angezeigten Daten

  function csvText(tab, f, e) {
    const de = e.csv === "de", sep = de ? ";" : ",";
    const feld = (v) => {
      if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "";
      let s = typeof v === "number" && de ? String(v).replace(".", ",") : String(v);
      if (s.includes(sep) || /["\r\n]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
      return s;
    };
    const zeilen = [];
    if (e.kopf) {
      zeilen.push(`# ${FT(t("app_titel"))} (${aktuellerStandort()}): ${FT(t(f.titel))}, ${FT(t(tab.titel))}; ` +
        `exportiert ${new Date().toISOString().slice(0, 10)}`);
      (f.laeufe || []).forEach((id) => {
        const d = D["lauf/" + id], l = LAEUFE.get(id);
        if (d && l) zeilen.push(`# lauf ${id}: ${FT(L(l.titel))}; konfiguration ${l.config}; fingerabdruck ${d.fingerabdruck}; ` +
          `kinneret_recon ${d.meta.version}, git ${d.meta.git}`);
      });
      (tab.kopf || []).forEach((k) => zeilen.push("# " + k));
    }
    zeilen.push(tab.spalten.map(feld).join(sep));
    tab.zeilen.forEach((z) => zeilen.push(z.map(feld).join(sep)));
    return (de ? "﻿" : "") + zeilen.join("\r\n") + "\r\n"; // BOM: Excel erkennt sonst UTF-8 nicht
  }

  function speichere(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    $("export").appendChild(a); // im Dialog: ausserhalb waere der Link bei offenem modalem Dialog inaktiv
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  }
  const dateiname = (f, m, endung) =>
    `${aktuellerStandort()}_${f.name}_${Math.round(m.b * 10)}x${Math.round(m.h * 10)}mm.${endung}`;

  // ---------------------------------------------------------------------------------------------------------------
  // Dialog

  function zeigeWerte() {
    const e = einstellung(), m = masse();
    umschalter("export-breite", [8.3, 12, 17.5].includes(m.b) ? String(m.b) : "", (w) => {
      e.breite = +w; hoehe = null; merke(); zeigeWerte();
    });
    $("export-b").value = m.b;
    $("export-h").value = +m.h.toFixed(1);
    $("export-s").value = m.s;
    $("export-dpi").value = String(e.dpi);
    $("export-csv-format").value = e.csv;
    $("export-kopf").checked = !!e.kopf;
    $("export-masse").innerHTML = tx("export_masse").replace("{b}", fmt(m.b, 1)).replace("{h}", fmt(m.h, 1))
      .replace("{s}", fmt(m.s, 1)).replace("{px}", `${pixel(m.b, e.dpi)} × ${Math.round(pixel(m.b, e.dpi) * Math.round(m.h * PT_JE_CM) / Math.round(m.b * PT_JE_CM))}`).replace("{dpi}", e.dpi);
    // CSV-Knoepfe: eine Datei je Tabelle der Abbildung
    const f = figur._krFigur, csv = $("export-csv");
    csv.innerHTML = "";
    (f.csv ? f.csv() : []).forEach((tab) => {
      const k = document.createElement("button");
      k.type = "button";
      k.className = "knopf";
      k.dataset.csv = tab.name;
      k.innerHTML = tx(tab.titel);
      k.onclick = () => speichere(new Blob([csvText(tab, f, einstellung())], { type: "text/csv;charset=utf-8" }),
        `${aktuellerStandort()}_${f.name}_${tab.name}.csv`);
      csv.appendChild(k);
    });
    vorschau();
  }

  function vorschau() {
    clearTimeout(vorschauUhr);
    vorschauUhr = setTimeout(async () => {
      const my = ++vorschauToken, m = masse();
      try {
        const svg = await svgText(abbildung(figur, m), m);
        if (my !== vorschauToken) return;
        const bild = $("export-bild");
        bild.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
        bild.style.width = `${m.b}cm`;
        $("export-fehler").textContent = "";
      } catch (err) {
        $("export-fehler").textContent = FT(t("export_fehler")) + ": " + err.message;
      }
    }, 150);
  }

  async function exportiere(art) {
    const m = masse(), f = figur._krFigur, e = einstellung();
    try {
      const fig = abbildung(figur, m);
      if (art === "png") speichere(await pngBlob(fig, e.dpi, m), dateiname(f, m, "png"));
      else {
        const svg = await svgText(fig, m);
        if (art === "svg") speichere(new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + svg], { type: "image/svg+xml" }), dateiname(f, m, "svg"));
        else drucke(svg, m, dateiname(f, m, "pdf").slice(0, -4));
      }
    } catch (err) {
      $("export-fehler").textContent = FT(t("export_fehler")) + ": " + err.message;
    }
  }

  function oeffne(div) {
    const fs = figuren();
    if (!fs.length) return;
    figur = div && fs.includes(div) ? div : fs[0];
    hoehe = null;
    const sel = $("export-figur");
    sel.innerHTML = fs.map((d, i) => `<option value="${i}"${d === figur ? " selected" : ""}>${KR.esc(FT(t(d._krFigur.titel)))}</option>`).join("");
    sel.onchange = () => { figur = fs[+sel.value]; hoehe = null; zeigeWerte(); };
    $("export-fehler").textContent = "";
    const dlg = $("export");
    if (!dlg.open) dlg.showModal();
    zeigeWerte();
  }

  function init() {
    const e = einstellung();
    const zahl = (id, fn) => { $(id).onchange = () => { const v = parseFloat(String($(id).value).replace(",", ".")); if (isFinite(v)) fn(v); merke(); zeigeWerte(); }; };
    zahl("export-b", (v) => { e.breite = begrenze(v, BREITE); hoehe = null; });
    zahl("export-h", (v) => { hoehe = begrenze(v, HOEHE); });
    zahl("export-s", (v) => { e.schrift = begrenze(v, SCHRIFT); });
    $("export-dpi").onchange = () => { e.dpi = +$("export-dpi").value; merke(); zeigeWerte(); };
    $("export-csv-format").onchange = () => { e.csv = $("export-csv-format").value; merke(); };
    $("export-kopf").onchange = () => { e.kopf = $("export-kopf").checked; merke(); };
    $("export-pdf").onclick = () => exportiere("pdf");
    $("export-svg").onclick = () => exportiere("svg");
    $("export-png").onclick = () => exportiere("png");
    $("export").addEventListener("click", (ev) => { if (ev.target === $("export")) $("export").close(); }); // Klick neben den Dialog
  }

  KR.oeffneExport = oeffne;
  KR.exportInit = init;
  KR.exportTest = { abbildung, svgText, pngBlob, csvText, masse, figuren, mitDpi }; // fuer tools/app_pruefen.js
})(window.KR);
