// Ansicht Lauf: Steckbrief (Beschreibung, Code-Stand, Laufzeit), Methodentext, Kalibrierung (R2, T0, P0),
// Einstellungen im Vergleich zum Basislauf und Konvergenzdiagnostik (eingeklappt).
(function (KR) {
  "use strict";
  const { $, D, K, Z, t, L, css, fmt, esc, merke, LAEUFE, FAKTOREN, ladeLauf, alpha, PLOT_CONFIG, achse, layout, titel,
    laufOptionen, stufeLabel, formel, F, FT } = KR;
  const tx = (k) => F(t(k)); // uebersetzter Text als HTML (Formeln gesetzt)

  let token = 0;
  const KETTEN_FARBEN = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"];
  const n0 = (x) => fmt(x, 0);
  const RH = F("$\\hat{R}$");
  // Parameternamen der Diagnostik (z. B. T0, T_Med, w[Olea europaea]) in Formelschreibweise
  function parName(p) {
    const fest = { R2: "$R^2$", T0: "$T_0$", P0: "$P_0$", alpha: "$\\alpha$", log_target: "log target" };
    if (fest[p]) return fest[p];
    let m = /^([TP])_(.+)$/.exec(p);
    if (m) return `$${m[1]}_\\mathrm{${m[2]}}$`;
    m = /^w\[(.+)\]$/.exec(p);
    return m ? `$w$[${m[1]}]` : p;
  }

  async function zeige() {
    const my = ++token;
    laufOptionen($("lauf-lauf"), Z.lauf);
    const l = LAEUFE.get(Z.lauf);
    let d, b = null;
    try {
      d = await ladeLauf(Z.lauf);
      if (l.basis && LAEUFE.has(l.basis) && LAEUFE.get(l.basis).app_json) b = await ladeLauf(l.basis);
    } catch (e) {
      $("lauf-status").innerHTML = `<span class="warn">${tx("fehler_laden")}: ${esc(e.message)}</span>`;
      return;
    }
    if (my !== token) return;
    $("lauf-hinweis").innerHTML = l.hinweis ? `<span class="warn">⚠ ${F(L(l.hinweis))}</span>` : "";
    $("lauf-status").innerHTML = diagnoseText(d);
    ueberblick(l, d);
    $("lauf-methoden").innerHTML = F(L(d.methoden));
    kalibrierung(d);
    einstellungen(d, b, l);
    if ($("lauf-diagnostik-box").open) diagnostik(d);
  }

  function diagnoseText(d) {
    const tab = d.diagnostik.tabelle;
    const rh = Math.max(...tab.map((r) => r.rhat ?? Infinity));
    const ess = Math.min(...tab.map((r) => r.ess_bulk ?? 0));
    const p = d.diagnostik.probleme;
    const ers = (s) => s.replace("{r}", fmt(rh, 3)).replace("{e}", n0(ess));
    return p.length ? ers(tx("diag_schlecht")).replace("{n}", p.length).replace("{m}", tab.length)
      .replace("{liste}", esc(p.slice(0, 8).join(", ")) + (p.length > 8 ? " …" : "")) : ers(tx("diag_ok"));
  }

  function zeile(k, v) {
    return `<tr><th>${k}</th><td>${v}</td></tr>`; // k und v sind bereits HTML
  }

  function ueberblick(l, d) {
    const e = d.einstellungen, m = d.meta;
    const z = [zeile(tx("beschreibung"), F(L(l.beschreibung))), zeile(tx("rolle"), F(L(K.rollen[l.rolle])))];
    if (l.faktor) z.push(zeile(F(L(FAKTOREN.get(l.faktor).label)), F(stufeLabel(l, l.faktor))));
    if (l.basis && LAEUFE.has(l.basis)) z.push(zeile(tx("basis"), F(L(LAEUFE.get(l.basis).titel))));
    if (l.quelle) z.push(zeile(tx("quelle"), esc(l.quelle)));
    z.push(zeile(tx("konfiguration"), `<code>${esc(l.config)}</code>`));
    z.push(zeile(tx("fingerabdruck"), `<code>${esc(d.fingerabdruck)}</code> (<code>${esc(d.fingerabdruck_ketten)}</code>)`));
    z.push(zeile(tx("code_stand"), `kinneret_recon ${esc(m.version)}, Git <code>${esc(m.git)}</code>`));
    z.push(zeile(tx("umgebung"), esc(`Python ${m.python}, NumPy ${m.numpy}, ${m.platform}`)));
    z.push(zeile(tx("ketten_iter"), `${e["sampler.n_chains"]} × ${n0(e["sampler.sample_length"])} (${esc(e["sampler.kernel"])})`));
    z.push(zeile(tx("burnin_thin"), `${n0(e["sampler.burn_in"])} / ${e["sampler.thin"]}`));
    z.push(zeile(tx("stichproben"), n0(d.diagnostik.n_stichproben)));
    z.push(zeile(tx("akzeptanz"), d.diagnostik.akzeptanz.map((a) => fmt(a, 3)).join(", ")));
    const zeiten = Object.entries(m.timings_s || {}).filter(([, v]) => v > 0)
      .map(([k, v]) => `${esc(k)} ${fmt(v, 0)} ${tx("sekunden")}`).join(", ");
    z.push(zeile(tx("laufzeit"), zeiten));
    if (m.posterior_aus) z.push(zeile(tx("ketten_aus"), `<code>${esc(m.posterior_aus.split("/").slice(-2).join("/"))}</code>`));
    $("lauf-ueberblick").innerHTML = `<tbody>${z.join("")}</tbody>`;
  }

  function kalibrierung(d) {
    const k = d.kalibrierung;
    const teile = [["R2", F("$R^2$"), 2], ["T0", F("$T_0$") + " [°C]", 2], ["P0", F("$P_0$") + " [mm]", 0]];
    const traces = [], lay = {}, ann = [];
    const breit = $("lauf-kalibrierung").clientWidth >= 700;
    teile.forEach(([key, label, nd], i) => {
      const g = k[key], kv = g.kurve, sx = i ? "x" + (i + 1) : "x", sy = i ? "y" + (i + 1) : "y";
      traces.push({ x: kv.x, y: kv.posterior, xaxis: sx, yaxis: sy, type: "scatter", mode: "lines", fill: "tozeroy",
        line: { color: css("--fg"), width: 1.5 }, fillcolor: alpha("#888888", 0.3), name: tx("tf_posterior"),
        hovertemplate: `${tx("tf_posterior")} %{x}: %{y:.2f}<extra></extra>` });
      traces.push({ x: kv.x, y: kv.prior, xaxis: sx, yaxis: sy, type: "scatter", mode: "lines",
        line: { color: "#E69F00", width: 2 }, name: tx("tf_prior"), hovertemplate: `${tx("tf_prior")} %{x}: %{y:.2f}<extra></extra>` });
      const dom = breit ? [i / 3 + (i ? 0.03 : 0), (i + 1) / 3 - (i < 2 ? 0.03 : 0)] : null;
      const rows = KR.zeilen([1, 1, 1], 0.14);
      lay["xaxis" + (i ? i + 1 : "")] = achse({ domain: breit ? dom : [0, 1], anchor: sy, title: { text: breit ? label : "" } });
      lay["yaxis" + (i ? i + 1 : "")] = achse({ domain: breit ? [0, 0.85] : rows[i], anchor: sx, range: [0, 1.05],
        showticklabels: false });
      ann.push(titel(`${label}: ${fmt(g.q50, nd)} (95 %: ${fmt(g.q2_5, nd)}–${fmt(g.q97_5, nd)})`, sy + " domain", sx + " domain"));
    });
    ann.push({ text: `<span style="color:#E69F00">━</span> ${tx("tf_prior")}   ▇ ${tx("tf_posterior")}`, xref: "paper",
      yref: "paper", x: 1, y: 1.0, xanchor: "right", yanchor: "bottom", showarrow: false,
      font: { size: 11, color: css("--muted") } });
    Plotly.react($("lauf-kalibrierung"), traces, layout(Object.assign({ hovermode: "closest", annotations: ann,
      margin: { l: 20, r: 14, t: 34, b: 44 } }, lay)), PLOT_CONFIG);
  }

  function wertText(v) {
    if (v === null || v === undefined) return "–";
    if (Array.isArray(v)) return v.length ? "[" + v.map(wertText).join(", ") + "]" : "[ ]";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(+v.toPrecision(10));
    return String(v);
  }

  function einstellungen(d, b, l) {
    const kopf = `<tr><th>${tx("feld")}</th><th>${tx("wert")}</th>${b ? `<th>${tx("basis")}: ${F(L(LAEUFE.get(l.basis).titel))}</th>` : ""}</tr>`;
    const z = Object.keys(d.einstellungen).sort().map((k) => {
      const v = wertText(d.einstellungen[k]), w = b ? wertText(b.einstellungen[k]) : null;
      const anders = b && v !== w;
      return `<tr class="${anders ? "anders" : ""}"><td><code>${esc(k)}</code></td><td>${esc(v)}</td>` +
        (b ? `<td>${anders ? esc(w) : ""}</td>` : "") + "</tr>";
    });
    $("lauf-einstellungen").innerHTML = `<thead>${kopf}</thead><tbody>${z.join("")}</tbody>` +
      (b ? `<caption>${tx("abweichend_basis")}</caption>` : "");
  }

  function diagnostik(d) {
    const dg = d.diagnostik, sp = dg.spuren;
    $("lauf-diag-status").innerHTML = diagnoseText(d) + " " + tx("akzeptanz") + ": " + dg.akzeptanz.map((a) => fmt(a, 3)).join(", ");
    const sel = $("lauf-spur");
    const namen = Object.keys(sp.werte);
    if (!namen.includes(Z.spur)) Z.spur = namen.includes("R2") ? "R2" : namen[0];
    sel.innerHTML = namen.map((n) => `<option value="${esc(n)}"${n === Z.spur ? " selected" : ""}>${esc(FT(parName(n)))}</option>`).join("");
    const traces = sp.werte[Z.spur].map((kette, c) => ({ x: sp.iteration, y: kette, type: "scatter", mode: "lines",
      line: { color: KETTEN_FARBEN[c % KETTEN_FARBEN.length], width: 1 }, name: `${tx("kette")} ${c + 1}`,
      hovertemplate: `${tx("kette")} ${c + 1}: %{y}<extra></extra>` }));
    Plotly.react($("lauf-spuren"), traces, layout({
      hovermode: "closest", xaxis: achse({ title: { text: tx("iteration") } }), yaxis: achse({ title: { text: F(parName(Z.spur)) } }),
      shapes: [{ type: "line", xref: "x", yref: "paper", x0: sp.burn_in, x1: sp.burn_in, y0: 0, y1: 1,
        line: { color: css("--muted"), dash: "dot", width: 1.5 } }],
      annotations: [{ text: tx("burn_in_ende"), x: sp.burn_in, xref: "x", y: 1, yref: "paper", xanchor: "left",
        yanchor: "bottom", showarrow: false, font: { size: 11, color: css("--muted") } }],
    }), PLOT_CONFIG);
    // R-hat und ESS aller Groessen
    const tab = dg.tabelle, y = tab.map((r) => F(parName(r.parameter)));
    const warn = css("--warn"), grau = css("--muted");
    const rh = tab.map((r) => r.rhat), es = tab.map((r) => r.ess_bulk);
    const div = $("lauf-rhat");
    div.style.height = Math.max(300, tab.length * 15 + 90) + "px";
    Plotly.react(div, [
      { type: "bar", orientation: "h", y, x: rh.map((v) => v ?? 0), marker: { color: rh.map((v) => (v === null || v > 1.01 ? warn : grau)) },
        hovertemplate: `%{y}: ${RH} %{x:.4f}<extra></extra>` },
      { type: "bar", orientation: "h", y, x: es.map((v) => v ?? 0), xaxis: "x2", marker: { color: es.map((v) => (v === null || v < 400 ? warn : grau)) },
        hovertemplate: "%{y}: ESS %{x:.0f}<extra></extra>" },
    ], layout({
      hovermode: "closest", margin: { l: 10, r: 14, t: 24, b: 44 },
      xaxis: achse({ domain: [0, 0.48], title: { text: `${RH} (${fmt(1.01, 2)})` },
        range: [0.99, Math.max(1.05, ...rh.filter((v) => v !== null)) + 0.005] }),
      xaxis2: achse({ domain: [0.52, 1], type: "log", title: { text: "ESS bulk (400)" } }),
      yaxis: achse({ autorange: "reversed", tickfont: { size: 10 }, showgrid: false }),
      shapes: [{ type: "line", xref: "x", yref: "paper", x0: 1.01, x1: 1.01, y0: 0, y1: 1, line: { color: css("--fg"), dash: "dash", width: 1 } },
        { type: "line", xref: "x2", yref: "paper", x0: 400, x1: 400, y0: 0, y1: 1, line: { color: css("--fg"), dash: "dash", width: 1 } }],
    }), PLOT_CONFIG);
    const kopf = [tx("groesse"), tx("mittel"), tx("sd"), "2,5 %", tx("median"), "97,5 %", RH, tx("ess_bulk"), tx("ess_tail")];
    const kopfHtml = kopf.map((k) => `<th>${k}</th>`).join("");
    const f3 = (v) => (v === null ? "–" : Math.abs(v) >= 100 ? fmt(v, 1) : fmt(v, 4));
    $("lauf-diagnostik").innerHTML = `<thead><tr>${kopfHtml}</tr></thead><tbody>` +
      tab.map((r) => `<tr><td>${F(parName(r.parameter))}</td><td>${f3(r.mean)}</td><td>${f3(r.sd)}</td>` +
        `<td>${f3(r.q2_5)}</td><td>${f3(r.median)}</td><td>${f3(r.q97_5)}</td>` +
        `<td class="${r.rhat === null || r.rhat > 1.01 ? "schlecht" : ""}">${r.rhat === null ? "–" : fmt(r.rhat, 4)}</td>` +
        `<td class="${r.ess_bulk === null || r.ess_bulk < 400 ? "schlecht" : ""}">${r.ess_bulk === null ? "–" : n0(r.ess_bulk)}</td>` +
        `<td>${r.ess_tail === null ? "–" : n0(r.ess_tail)}</td></tr>`).join("") + "</tbody>";
  }

  // Methodentext kopieren: formatiert (HTML mit Indizes) und als Unicode-Text, oder als LaTeX-Quelltext
  async function kopieren(knopf, latex) {
    const quelle = L(D["lauf/" + Z.lauf].methoden);
    const text = formel(quelle, latex ? "latex" : "text");
    let ok = false;
    try {
      if (!latex && window.ClipboardItem && navigator.clipboard.write) {
        await navigator.clipboard.write([new ClipboardItem({
          "text/html": new Blob([formel(quelle, "html")], { type: "text/html" }),
          "text/plain": new Blob([text], { type: "text/plain" }),
        })]);
      } else await navigator.clipboard.writeText(text);
      ok = true;
    } catch (e) { /* Rueckfall unten (z. B. ohne Berechtigung) */ }
    if (!ok) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      ta.remove();
    }
    if (ok) {
      const alt = knopf.innerHTML;
      knopf.innerHTML = tx("kopiert");
      setTimeout(() => { knopf.innerHTML = alt; }, 1500);
    }
  }

  KR.ansicht({
    key: "lauf",
    init() {
      $("lauf-lauf").onchange = (e) => { Z.lauf = e.target.value; Z.art = null; merke(); zeige(); };
      $("lauf-kopieren").onclick = () => kopieren($("lauf-kopieren"), false);
      $("lauf-kopieren-latex").onclick = () => kopieren($("lauf-kopieren-latex"), true);
      $("lauf-diagnostik-box").addEventListener("toggle", () => {
        if ($("lauf-diagnostik-box").open) ladeLauf(Z.lauf).then(diagnostik);
      });
      $("lauf-spur").onchange = (e) => { Z.spur = e.target.value; merke(); ladeLauf(Z.lauf).then(diagnostik); };
    },
    zeige,
  });
})(window.KR);
