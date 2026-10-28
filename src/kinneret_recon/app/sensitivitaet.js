// Ansicht Sensitivitaet: Varianten eines Faktors gegen einen Basislauf (Median, Differenz, Kennzahlen).
(function (KR) {
  "use strict";
  const { $, D, K, Z, t, L, css, fmt, fmtS, esc, merke, LAEUFE, FAKTOREN, ladeLauf, ladeStandort, farbe, alpha,
    dichteSkala, PLOT_CONFIG, achse, layout, zeilen, titel, band, umschalter, laufOptionen, stufeLabel,
    GRUPPEN_FOLGE, F, FT } = KR;
  const tx = (k) => F(t(k)); // uebersetzter Text als HTML (Formeln gesetzt)


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
      grp.label = FT(L(K.gruppen[g]));
      fs.forEach((f) => {
        const o = document.createElement("option");
        o.value = f.key;
        o.textContent = FT(L(f.label));
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
    $("sens-faktor-text").innerHTML = F(L(f.beschreibung));
    // Basislauf: alle Laeufe der Reihe
    const bsel = $("sens-basis");
    bsel.innerHTML = "";
    s.forEach((l) => {
      const o = document.createElement("option");
      o.value = l.id;
      o.textContent = FT(L(l.titel)) + " · " + FT(stufeLabel(l, Z.faktor));
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
      txt.innerHTML = F(L(l.titel)) + (l.hinweis ? " ⚠" : "");
      if (l.hinweis) lab.title = FT(L(l.hinweis));
      lab.append(cb, chip, txt);
      liste.appendChild(lab);
    });
    const wsel = $("sens-weitere");
    laufOptionen(wsel, null, (l) => !s.includes(l) && !Z.extra.includes(l.id));
    const leer = document.createElement("option");
    leer.value = "";
    leer.textContent = FT(t("weitere_lauf"));
    leer.selected = true;
    wsel.prepend(leer);
    $("sens-baender").checked = !!Z.baender;

    const gezeigt = [Z.basis, ...kandidaten.filter((l) => Z.auswahl.laeufe.includes(l.id) || Z.extra.includes(l.id)).map((l) => l.id)];
    let daten;
    try {
      daten = await Promise.all(gezeigt.map(ladeLauf));
    } catch (e) {
      $("sens-tabelle").innerHTML = `<caption class="warn">${tx("fehler_laden")}: ${esc(e.message)}</caption>`;
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
    const faktor = Z.faktor, baender = !!Z.baender, name = `sensitivitaet_${faktor}_${Z.basis}`;
    const info = {
      laeufe: ids, hoehe: (b) => (b < 10 ? 9 : 10),
      csv: () => [csvKurven(ids, daten, baender), csvTabelle(ids, daten, faktor)],
    };
    // beide Diagramme aus einem Aufbau; im Druckstil (Export) neu gebaut
    const baue = () => baueKurven(ids, daten, baender), bild = baue();
    KR.plot($("sens-plot"), () => (KR.druck ? baue() : bild)[0], Object.assign({ name, titel: "ansicht_sensitivitaet" }, info));
    KR.plot($("sens-diff"), () => (KR.druck ? baue() : bild)[1], Object.assign({ name: name + "_differenz", titel: "differenz" }, info));
  }

  function baueKurven(ids, daten, baender) {
    const x = daten[0].rekonstruktion.alter.x;
    const dom = zeilen([1, 1], 0.08);
    const traces = [], diff = [];
    ids.forEach((id, k) => {
      const d = daten[k], basis = k === 0;
      const c = basis ? css("--base") : farbe(id);
      const name = L(LAEUFE.get(id).titel);
      ["T", "P"].forEach((v, r) => {
        const q = kurve(d, v), ya = r ? "y2" : "y";
        if (baender) traces.push(...band(x, q.q25, q.q75, c.startsWith("#") ? c : "#888888", basis ? 0.18 : 0.14, ya, name));
        traces.push({ x, y: q.q50, yaxis: ya, type: "scatter", mode: "lines", name: F(name), showlegend: r === 0,
          line: { color: c, width: basis ? 3 : 1.8 },
          hovertemplate: `${F(name)}: %{y:.${v === "T" ? 2 : 0}f}<extra></extra>` });
        if (!basis) {
          const b = kurve(daten[0], v).q50;
          diff.push({ x, y: q.q50.map((y, i) => y - b[i]), yaxis: ya, type: "scatter", mode: "lines", name: F(name),
            showlegend: r === 0,
            line: { color: c, width: 1.8 }, hovertemplate: `${F(name)}: %{y:+.${v === "T" ? 2 : 0}f}<extra></extra>` });
        }
      });
    });
    const legende = !!KR.druck;
    const xa = achse({ title: { text: tx("alter_achse") }, anchor: "y2" });
    return [[traces, layout({
      hovermode: "x", xaxis: xa, showlegend: legende,
      yaxis: achse({ domain: dom[0], title: { text: tx("T") } }), yaxis2: achse({ domain: dom[1], title: { text: tx("P") } }),
      annotations: [titel("(a) " + tx("T") + " · " + tx("median"), "y domain"), titel("(b) " + tx("P") + " · " + tx("median"), "y2 domain")],
    })], [diff, layout({
      hovermode: "x", showlegend: legende, xaxis: achse({ title: { text: tx("alter_achse") }, anchor: "y2" }),
      yaxis: achse({ domain: dom[0], title: { text: tx("dT") }, zerolinecolor: css("--muted"), zerolinewidth: 1.5 }),
      yaxis2: achse({ domain: dom[1], title: { text: tx("dP") }, zerolinecolor: css("--muted"), zerolinewidth: 1.5 }),
      annotations: [titel("(c) " + tx("dT"), "y domain"), titel("(d) " + tx("dP"), "y2 domain")],
    })]];
  }

  // CSV: Median (und 50 %) von T und P je Lauf und die Differenz zum Basislauf; Spalten mit der Lauf-ID
  function csvKurven(ids, daten, baender) {
    const x = daten[0].rekonstruktion.alter.x;
    const spalten = ["alter_cal_a_bp"], spaltenWerte = [];
    ids.forEach((id, k) => ["T", "P"].forEach((v) => {
      const q = kurve(daten[k], v), b = kurve(daten[0], v).q50, gr = v === "T" ? "T_DJF" : "P_ANN";
      const qs = baender ? ["q25", "q50", "q75"] : ["q50"];
      qs.forEach((qq) => { spalten.push(`${gr}_${qq}_${id}`); spaltenWerte.push(q[qq]); });
      if (k) { spalten.push(`delta_${gr}_q50_${id}`); spaltenWerte.push(q.q50.map((y, i) => y - b[i])); }
    }));
    return { name: "kurven", titel: "csv_kurven", spalten, zeilen: x.map((a, i) => [a, ...spaltenWerte.map((w) => +w[i].toFixed(4))]),
      kopf: [`basislauf: ${ids[0]}`, "jeder Lauf mit seiner Standard-Rekonstruktion"] };
  }
  function csvTabelle(ids, daten, faktor) {
    const b = daten[0];
    const zeilen = ids.map((id, k) => {
      const d = daten[k], l = LAEUFE.get(id);
      const tq = kurve(d, "T"), pq = kurve(d, "P"), tb = kurve(b, "T"), pb = kurve(b, "P");
      const r4 = (v) => (isFinite(v) ? +v.toFixed(4) : "");
      return [id, KR.FT(L(l.titel)), KR.FT(stufeLabel(l, faktor)), r4(mittel(tq.q50)), r4(mittel(pq.q50)),
        k ? r4(mittel(tq.q50) - mittel(tb.q50)) : "", k ? r4(mittel(pq.q50) - mittel(pb.q50)) : "",
        r4(iqr(tq)), r4(iqr(pq)), k ? r4(iqr(tq) / iqr(tb) - 1) : "", k ? r4(iqr(pq) / iqr(pb) - 1) : "",
        k ? r4(korrelation(tq.q50, tb.q50)) : 1, k ? r4(korrelation(pq.q50, pb.q50)) : 1, d.kennzahlen.ess_min];
    });
    return { name: "kennzahlen", titel: "csv_kennzahlen", zeilen, kopf: [`faktor: ${faktor}`, `basislauf: ${ids[0]}`],
      spalten: ["lauf", "titel", "stufe", "T_niveau", "P_niveau", "delta_T", "delta_P", "T_iqr", "P_iqr",
        "T_iqr_rel_aenderung", "P_iqr_rel_aenderung", "r_T", "r_P", "ess_min"] };
  }

  function sensTabelle(ids, daten) {
    const kopf = ["spalte_lauf", "spalte_stufe", "spalte_T", "spalte_P", "spalte_dT", "spalte_dP", "spalte_iqrT",
      "spalte_iqrP", "spalte_rT", "spalte_rP", "spalte_ess"].map((k) => `<th>${tx(k)}</th>`).join("");
    const b = daten[0];
    const zeilenHtml = ids.map((id, k) => {
      const d = daten[k], l = LAEUFE.get(id);
      const tq = kurve(d, "T"), pq = kurve(d, "P"), tb = kurve(b, "T"), pb = kurve(b, "P");
      const it = iqr(tq), ip = iqr(pq), itb = iqr(tb), ipb = iqr(pb);
      const rel = (a, c) => (k === 0 ? "" : ` (${fmtS((a / c - 1) * 100, 0)} %)`);
      const chip = `<span class="farbe" style="background:${k === 0 ? css("--base") : farbe(id)}"></span> `;
      return `<tr class="${k === 0 ? "basis" : ""}">` +
        `<td>${chip}${F(L(l.titel))}${l.hinweis ? ` <span class="warn-inline" title="${esc(FT(L(l.hinweis)))}">⚠</span>` : ""}</td>` +
        `<td>${F(stufeLabel(l, Z.faktor))}</td>` +
        `<td>${fmt(mittel(tq.q50), 2)}</td><td>${fmt(mittel(pq.q50), 0)}</td>` +
        `<td>${k ? fmtS(mittel(tq.q50) - mittel(tb.q50), 2) : "–"}</td>` +
        `<td>${k ? fmtS(mittel(pq.q50) - mittel(pb.q50), 0) : "–"}</td>` +
        `<td>${fmt(it, 2)}${rel(it, itb)}</td><td>${fmt(ip, 0)}${rel(ip, ipb)}</td>` +
        `<td>${k ? fmt(korrelation(tq.q50, tb.q50), 3) : "1"}</td><td>${k ? fmt(korrelation(pq.q50, pb.q50), 3) : "1"}</td>` +
        `<td>${fmt(d.kennzahlen.ess_min, 0)}</td></tr>`;
    }).join("");
    $("sens-tabelle").innerHTML = `<thead><tr>${kopf}</tr></thead><tbody>${zeilenHtml}</tbody>`;
  }

  KR.ansicht({
    key: "sensitivitaet",
    init() {
      $("sens-faktor").onchange = (e) => { Z.faktor = e.target.value; Z.basis = null; Z.auswahl = null; merke(); zeigeSensitivitaet(); };
      $("sens-basis").onchange = (e) => { Z.basis = e.target.value; Z.auswahl = null; merke(); zeigeSensitivitaet(); };
      $("sens-weitere").onchange = (e) => { if (e.target.value) { Z.extra.push(e.target.value); merke(); zeigeSensitivitaet(); } };
      $("sens-baender").onchange = (e) => { Z.baender = e.target.checked; merke(); zeigeSensitivitaet(); };
    },
    zeige: zeigeSensitivitaet,
  });
})(window.KR);
