// Ansicht Moderne Validierung: gespeicherte Fehler der raeumlich blockierten Shared37-Baseline (moderne Testproben,
// keine fossilen Fehlerbaender). Hauptmodell vorab festgelegt (5 Pollenanaloge), k = 1/15 Sensitivitaeten, Vergleich
// Trainingsmittel; die regionale Tabelle ist eine nachgelagerte Teilansicht desselben Tests.
(function (KR) {
  "use strict";
  const { $, Z, t, css, fmt, fmtS, esc, merke, ladeValidierung, alpha, achse, layout, titel, umschalter, schmal, F } = KR;
  const tx = (k) => F(t(k)); // uebersetzter Text als HTML

  let token = 0;
  const ZIELE = [["tann_c", "val_tann", "°C", 2], ["pann_mm_per_year", "val_pann", "val_mm_jahr", 0]];
  const einheit = (z) => (z[2].startsWith("val_") ? t(z[2]) : z[2]);
  const modellName = (V, m) => (m === V.referenzmodell ? t("val_trainingsmittel") :
    t("val_analoge").replace("{k}", m.replace("analog_k", "")));
  const rolle = (V, m) => t(m === V.hauptmodell ? "val_hauptmodell" : m === V.referenzmodell ? "val_vergleich" : "val_sensitivitaet");
  function farbeModell(V, m) {
    if (m === V.referenzmodell) return css("--muted");
    return m === V.hauptmodell ? css("--accent") : alpha(css("--accent"), 0.4);
  }

  async function zeige() {
    const my = ++token;
    let V;
    try {
      V = await ladeValidierung();
    } catch (e) {
      $("val-info").innerHTML = `<span class="warn">${tx("val_fehlt")}</span>`;
      return;
    }
    if (my !== token) return;
    const bloecke = Object.keys(V.bloecke).sort((a, b) => a - b);
    if (!bloecke.includes(Z.valBlock)) Z.valBlock = bloecke[0];
    if (!V.gewichtungen.includes(Z.valGewichtung)) Z.valGewichtung = V.gewichtungen[0];
    const ub = $("val-block");
    if (ub.children.length !== bloecke.length) {
      ub.innerHTML = bloecke.map((b) => `<button data-wert="${b}">${b}°</button>`).join("");
    }
    umschalter("val-block", Z.valBlock, (w) => { Z.valBlock = w; merke(); zeige(); });
    umschalter("val-gewichtung", Z.valGewichtung, (w) => { Z.valGewichtung = w; merke(); zeige(); });
    const B = V.bloecke[Z.valBlock], w = Z.valGewichtung;
    info(V, B);
    zeichne(V, B, w);
    tabelle(V, B.metriken, w, $("val-tabelle"));
    folds(V, B, w);
    $("val-regional-hinweis").innerHTML = tx("val_regional_hinweis")
      .replace("{b0}", fmt(V.regional.breite[0], 0)).replace("{b1}", fmt(V.regional.breite[1], 0))
      .replace("{l0}", fmt(-V.regional.laenge[0], 0)).replace("{l1}", fmt(V.regional.laenge[1], 0))
      .replace("{n}", fmt(V.regional.proben, 0));
    tabelle(V, V.regional.metriken[Z.valBlock], w, $("val-regional"));
  }

  function info(V, B) {
    const p = V.pruefung;
    const dl = [
      [tx("val_proben"), tx("val_proben_text").replace("{n}", fmt(V.proben, 0)).replace("{a}", fmt(V.ausgeschlossen, 0))
        .replace("{g}", fmt(V.koordinatengruppen, 0)).replace("{m}", V.merkmale)],
      [tx("val_block"), tx("val_block_text").replace("{b}", Z.valBlock).replace("{n}", fmt(B.bloecke, 0)).replace("{f}", V.folds)],
      [tx("val_puffer"), `${fmt(V.puffer_km, 0)} km`],
      [tx("val_hauptmodell"), esc(modellName(V, V.hauptmodell)) + " " + tx("val_vorab")],
      [tx("val_sensitivitaet"), V.sensitivitaet.map((m) => esc(modellName(V, m))).join(", ")],
      [tx("val_vergleich"), tx("val_trainingsmittel_text")],
      [tx("val_einheiten"), `${tx("val_tann")} °C, ${tx("val_pann")} ${tx("val_mm_jahr")}`],
      [tx("val_pruefung"), tx("val_pruefung_text").replace("{f}", p.folds_checked).replace("{z}", fmt(p.prediction_rows_checked, 0))
        .replace("{km}", fmt(p.minimum_train_test_km, 2))],
    ];
    $("val-info").innerHTML = `<strong>${tx("val_titel")}</strong>` +
      `<dl>${dl.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
      `<span class="warn">${tx("val_grenze")}</span>`;
  }

  // CSV: Gesamtfehler und Fehler je Fold fuer alle Blockgroessen und Gewichtungen
  function csv(V) {
    const werte = ["rmse", "mae", "bias", "mse_skill"];
    const zeilen = [], fz = [];
    for (const [b, B] of Object.entries(V.bloecke)) {
      for (const w of V.gewichtungen) {
        for (const m of V.modelle) {
          for (const z of ZIELE) zeilen.push([b, w, m, z[0], ...werte.map((k) => B.metriken[m][z[0]][w][k])]);
        }
        for (const f of B.folds) {
          for (const m of V.modelle) {
            for (const z of ZIELE) fz.push([b, w, f.fold, f.test_samples, f.train_samples, f.buffer_excluded,
              f.minimum_train_test_km, m, z[0], ...werte.map((k) => f.metriken[m][z[0]][w][k])]);
          }
        }
      }
    }
    const kopf = [`${V.experiment} (${V.variante}), ${V.quelle}`, `hauptmodell ${V.hauptmodell}; vergleich ${V.referenzmodell}; ` +
      `puffer_km ${V.puffer_km}; folds ${V.folds}; einheiten tann_c degC, pann_mm_per_year mm/year`];
    return [
      { name: "validierung", titel: "val_fehler", kopf, zeilen,
        spalten: ["block_grad", "gewichtung", "modell", "ziel", ...werte] },
      { name: "validierung_folds", titel: "val_folds", kopf, zeilen: fz,
        spalten: ["block_grad", "gewichtung", "fold", "test", "training", "puffer_ausgeschlossen", "min_abstand_km", "modell",
          "ziel", ...werte] },
    ];
  }

  function zeichne(V, B, w) {
    const div = $("val-plot");
    div.style.height = schmal(div) ? "640px" : "";
    KR.plot(div, () => baue(V, B, w, div), {
      name: `validierung_${Z.valBlock}grad`, titel: "val_fehler", laeufe: [], hoehe: (b) => (b < 12 ? 11 : 6),
      csv: () => csv(V),
    });
  }

  function baue(V, B, w, div) {
    const traces = [], eng = schmal(div), reihen = KR.zeilen([1, 1], 0.2);
    const x = V.modelle.map((m) => modellName(V, m));
    ZIELE.forEach((z, i) => {
      const sx = i ? "x2" : "x", sy = i ? "y2" : "y", u = einheit(z);
      traces.push({ type: "bar", x, y: V.modelle.map((m) => B.metriken[m][z[0]][w].rmse), xaxis: sx, yaxis: sy,
        marker: { color: V.modelle.map((m) => farbeModell(V, m)) }, showlegend: false,
        customdata: V.modelle.map((m) => B.metriken[m][z[0]][w].mse_skill),
        hovertemplate: `%{x}<br>RMSE %{y:.${z[3] + 1}f} ${u}<br>${t("val_skill")} %{customdata:.3f}<extra></extra>` });
      // RMSE der einzelnen Folds als Punkte
      traces.push({ type: "scatter", mode: "markers", xaxis: sx, yaxis: sy, showlegend: false,
        x: V.modelle.flatMap((m) => B.folds.map(() => modellName(V, m))),
        y: V.modelle.flatMap((m) => B.folds.map((f) => f.metriken[m][z[0]][w].rmse)),
        text: V.modelle.flatMap(() => B.folds.map((f) => `Fold ${f.fold}`)),
        marker: { color: css("--fg"), size: 5, opacity: 0.6 },
        hovertemplate: `%{x}, %{text}<br>RMSE %{y:.${z[3] + 1}f} ${u}<extra></extra>` });
    });
    return [traces, layout({
      hovermode: "closest", bargap: 0.35, margin: { l: 60, r: 14, t: 30, b: 60 },
      annotations: [titel(`(a) ${t("val_tann")}, RMSE [°C]`, "y domain", "x domain"),
        titel(`(b) ${t("val_pann")}, RMSE [${einheit(ZIELE[1])}]`, "y2 domain", "x2 domain")],
      xaxis: achse({ domain: eng ? [0, 1] : [0, 0.46] }),
      xaxis2: achse({ domain: eng ? [0, 1] : [0.54, 1], anchor: "y2" }),
      yaxis: achse({ rangemode: "tozero", domain: eng ? reihen[0] : [0, KR.druck ? 1 : 0.9] }),
      yaxis2: achse({ rangemode: "tozero", anchor: "x2", domain: eng ? reihen[1] : [0, KR.druck ? 1 : 0.9] }),
    })];
  }

  function tabelle(V, metriken, w, el) {
    const kopf = [tx("val_modell"), tx("val_rolle"), ...ZIELE.flatMap((z) => {
      const u = esc(einheit(z));
      return [`${tx(z[1])} RMSE [${u}]`, `MAE [${u}]`, `Bias [${u}]`, tx("val_skill")];
    })];
    const zeilen = V.modelle.map((m) => `<tr class="${m === V.hauptmodell ? "basis" : ""}"><td>${esc(modellName(V, m))}</td>` +
      `<td>${esc(rolle(V, m))}</td>` + ZIELE.map((z) => {
      const r = metriken[m][z[0]][w];
      return `<td>${fmt(r.rmse, z[3] + 1)}</td><td>${fmt(r.mae, z[3] + 1)}</td><td>${fmtS(r.bias, z[3] + 1)}</td>` +
        `<td>${fmt(r.mse_skill, 3)}</td>`;
    }).join("") + "</tr>").join("");
    el.innerHTML = `<thead><tr>${kopf.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${zeilen}</tbody>`;
  }

  function folds(V, B, w) {
    const h = V.hauptmodell, r = V.referenzmodell;
    const kopf = ["Fold", tx("val_test"), tx("val_training"), tx("val_puffer_aus"), tx("val_min_km"),
      ...ZIELE.map((z) => `${tx(z[1])} RMSE [${esc(einheit(z))}]: ${esc(modellName(V, r))} → ${esc(modellName(V, h))}`)];
    const zeilen = B.folds.map((f) => `<tr><td>${f.fold}</td><td>${fmt(f.test_samples, 0)}</td><td>${fmt(f.train_samples, 0)}</td>` +
      `<td>${fmt(f.buffer_excluded, 0)}</td><td>${fmt(f.minimum_train_test_km, 1)}</td>` +
      ZIELE.map((z) => `<td>${fmt(f.metriken[r][z[0]][w].rmse, z[3] + 1)} → ${fmt(f.metriken[h][z[0]][w].rmse, z[3] + 1)}</td>`).join("") +
      "</tr>").join("");
    $("val-folds").innerHTML = `<thead><tr>${kopf.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${zeilen}</tbody>`;
  }

  KR.ansicht({
    key: "validierung",
    init() {
      window.addEventListener("resize", () => {
        if (Z.ansicht === "validierung" && !$("ansicht-validierung").hidden) {
          const eng = schmal($("val-plot"));
          if (eng !== $("val-plot")._krEng) { $("val-plot")._krEng = eng; zeige(); }
        }
      });
    },
    zeige,
  });
})(window.KR);
