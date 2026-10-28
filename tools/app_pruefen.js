// Prueft die gebaute App im Browser (Playwright, Chromium): Laden ueber file:// ohne Netz, Ansichten, Sprache,
// Thema, Handy- und PC-Groesse; meldet Fehler der Konsole. Optional Bildschirmfotos.
//
// Aufruf: node tools/app_pruefen.js results/app/index.html [results/kinneret-app.html] [--fotos ORDNER]
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const fi = args.indexOf("--fotos");
const fotos = fi >= 0 ? args.splice(fi, 2)[1] : null;
if (fotos) fs.mkdirSync(fotos, { recursive: true });

async function pruefe(browser, datei, groesse, thema, name) {
  const ctx = await browser.newContext({ viewport: groesse, colorScheme: thema, offline: true, deviceScaleFactor: 1, locale: "de-DE" });
  const page = await ctx.newPage();
  const fehler = [];
  page.on("console", (m) => { if (m.type() === "error") fehler.push(m.text()); });
  page.on("pageerror", (e) => fehler.push(String(e)));
  page.on("request", (r) => { if (!/^(file|data|blob):/.test(r.url())) fehler.push("Netz: " + r.url()); });
  await page.goto("file://" + path.resolve(datei));
  const warte = (sel) => page.waitForSelector(sel + " .main-svg", { timeout: 20000 });
  await warte("#rek-plot");
  const n = await page.evaluate(() => document.querySelectorAll("#rek-plot .scatterlayer .trace").length);
  if (n < 5) fehler.push("Rekonstruktion: zu wenige Kurven " + n);
  // Alter antippen -> Detail
  const box = await page.locator("#rek-plot .nsewdrag").first().boundingBox();
  await page.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.5);
  await page.waitForSelector("#rek-detail:not([hidden]) #rek-detail-spektrum .main-svg", { timeout: 10000 })
    .catch(() => fehler.push("Detail erscheint nicht"));
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_rekonstruktion.png`, fullPage: true });
  await pruefeExport(page, fehler, fotos && `${fotos}/${name}`, groesse.width > 1000 && thema === "light");
  // Mischverteilung
  await page.click('#rek-art button[data-wert="mixture"]');
  await page.waitForSelector("#rek-plot .heatmaplayer image", { timeout: 10000 }).catch(() => fehler.push("keine Dichte"));
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_mischverteilung.png` });
  // Anderer Lauf (mit Kalibrierungsbereich), Tiefe
  await page.selectOption("#rek-lauf", "p_kalibrierung_ab_4ka");
  await page.click('#rek-achse button[data-wert="tiefe"]');
  await page.waitForTimeout(500);
  // Sensitivitaet
  await page.click('.reiter button[data-ansicht="sensitivitaet"]');
  await warte("#sens-plot");
  await page.selectOption("#sens-faktor", "klima_prior");
  await page.waitForFunction(() => document.querySelectorAll("#sens-tabelle tbody tr").length >= 7, null, { timeout: 15000 })
    .catch(() => fehler.push("Tabelle Sensitivitaet unvollstaendig"));
  const zeilen = await page.evaluate(() => [...document.querySelectorAll("#sens-tabelle tbody tr")].map((r) => r.innerText));
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_sensitivitaet.png`, fullPage: true });
  for (const f of ["korrekturen", "r2_prior", "rekonstruktion", "ap_taxa"]) {
    await page.selectOption("#sens-faktor", f);
    await page.waitForTimeout(400);
  }
  // Taxa-Gewichte mit Vergleich, log. Achse
  await page.click('.reiter button[data-ansicht="gewichte"]');
  await warte("#gew-plot");
  await page.selectOption("#gew-vergleich select", "standard_aktuell");
  await page.waitForFunction(() => document.querySelectorAll("#gew-vergleich .chip").length === 1, null, { timeout: 10000 })
    .catch(() => fehler.push("Vergleich: kein Chip"));
  await page.waitForTimeout(400);
  const nGew = await page.evaluate(() => document.querySelectorAll("#gew-plot .scatterlayer .trace").length);
  if (nGew < 6) fehler.push("Gewichte: zu wenige Spuren " + nGew);
  await page.check("#gew-log");
  await page.waitForTimeout(300);
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_gewichte.png`, fullPage: true });
  // Transferfunktionen (Vergleich bleibt erhalten)
  await page.click('.reiter button[data-ansicht="transferfunktionen"]');
  await warte("#tf-plot");
  await page.waitForSelector("#tf-ci .main-svg", { timeout: 10000 });
  const nTf = await page.evaluate(() => [document.querySelectorAll("#tf-plot .contourlayer .trace, #tf-plot g.contour").length,
    document.querySelectorAll("#tf-tabelle tbody tr").length]);
  if (nTf[0] < 3 || nTf[1] !== 3) fehler.push("Transferfunktionen: " + nTf);
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_transferfunktionen.png`, fullPage: true });
  // Lauf: Steckbrief, Diagnostik aufklappen, Spur wechseln; Lauf mit haengender Kette
  await page.click('.reiter button[data-ansicht="lauf"]');
  await warte("#lauf-kalibrierung");
  await page.click("#lauf-diagnostik-box summary");
  await warte("#lauf-spuren");
  await page.selectOption("#lauf-spur", "T0");
  await page.waitForTimeout(300);
  const nDiag = await page.evaluate(() => document.querySelectorAll("#lauf-diagnostik tbody tr").length);
  if (nDiag < 40) fehler.push("Diagnostik: zu wenige Zeilen " + nDiag);
  const meth = await page.textContent("#lauf-methoden");
  if (!meth.startsWith("Die Posterior-Verteilung")) fehler.push("Methodentext: " + meth.slice(0, 40));
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_lauf.png`, fullPage: true });
  await page.selectOption("#lauf-lauf", "h_legacy_r2_025");
  await page.waitForFunction(() => document.querySelector("#lauf-hinweis").textContent.includes("⚠"), null, { timeout: 10000 })
    .catch(() => fehler.push("Hinweis bei haengender Kette fehlt"));
  await page.selectOption("#lauf-lauf", "standard_v3");
  await page.click('.reiter button[data-ansicht="sensitivitaet"]');
  await page.waitForTimeout(300);
  // Sprache
  await page.click("#sprache");
  await page.waitForTimeout(300);
  const reiter = await page.textContent('.reiter button[data-ansicht="sensitivitaet"]');
  if (reiter.trim() !== "Sensitivity") fehler.push("Sprache: " + reiter);
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_englisch.png` });
  await page.click('.reiter button[data-ansicht="lauf"]');
  await page.waitForTimeout(500);
  const methEn = await page.textContent("#lauf-methoden");
  if (!methEn.startsWith("The posterior distribution")) fehler.push("Methodentext en: " + methEn.slice(0, 40));
  await page.click("#sprache");
  // Export in jeder Ansicht: Dialog, Vorschau, CSV-Knoepfe
  for (const ansicht of ["rekonstruktion", "sensitivitaet", "gewichte", "transferfunktionen", "lauf"]) {
    await page.click(`.reiter button[data-ansicht="${ansicht}"]`);
    await page.waitForTimeout(400);
    await page.click(`#ansicht-${ansicht} .export-knopf`);
    const ok = await page.waitForFunction(() => $vorschauFertig(), null, { timeout: 15000 }).then(() => true, () => false);
    const n = await page.evaluate(() => [document.querySelectorAll("#export-figur option").length,
      document.querySelectorAll("#export-csv button").length, document.querySelector("#export-fehler").textContent]);
    if (!ok || n[0] < 1 || n[1] < 1 || n[2]) fehler.push(`Export ${ansicht}: ${ok} ${n}`);
    await page.click('#export button[value="zu"]');
  }
  // Formeln: nirgends rohe LaTeX-Schreibweise, aber gesetzte Indizes in Diagrammen und Text
  for (const ansicht of ["rekonstruktion", "sensitivitaet", "gewichte", "transferfunktionen", "lauf"]) {
    await page.click(`.reiter button[data-ansicht="${ansicht}"]`);
    await page.waitForTimeout(400);
    const roh = await page.evaluate(() => {
      const txt = document.body.innerText + [...document.querySelectorAll("svg text")].map((e) => e.textContent).join(" ");
      return (txt.match(/\\(mathrm|hat|alpha|kappa|sim|Delta)|\$[A-Za-z\\]/g) || []).slice(0, 5);
    });
    if (roh.length) fehler.push(`rohe Formel in ${ansicht}: ${roh.join(" ")}`);
  }
  const indizes = await page.evaluate(() => [document.querySelectorAll("#lauf-methoden sub, #lauf-methoden sup").length,
    document.querySelectorAll("#lauf-diagnostik sub").length]);
  if (indizes[0] < 3 || indizes[1] < 5) fehler.push("Indizes fehlen: " + indizes);
  await ctx.close();
  return { fehler, zeilen };
}

// Export der Rekonstruktion: SVG in cm, PNG mit 300 dpi, CSV, PDF ueber den Druckweg (nur PC hell), Abbildung immer hell
async function pruefeExport(page, fehler, foto, pdf) {
  await page.evaluate(() => {
    window.$vorschauFertig = () => { const b = document.querySelector("#export-bild"); return b.src.startsWith("data:") && b.complete; };
  });
  await page.click("#ansicht-rekonstruktion .export-knopf");
  await page.click('#export-breite button[data-wert="17.5"]');
  await page.waitForFunction(() => $vorschauFertig() && document.querySelector("#export-b").value === "17.5", null, { timeout: 15000 })
    .catch(() => fehler.push("Export: keine Vorschau"));
  if (foto) await page.screenshot({ path: `${foto}_export.png` });
  const svg = await page.evaluate(async () => { const T = KR.exportTest, m = T.masse(); return T.svgText(T.abbildung(T.figuren()[0], m), m); });
  if (!/^<svg[^>]+width="17.5cm"/.test(svg) || !/viewBox="0 0 496 /.test(svg)) fehler.push("Export SVG: " + svg.slice(0, 160));
  if (!svg.includes("fill: rgb(255, 255, 255)") || /rgb\(21, 23, 26\)|#15171a/.test(svg)) fehler.push("Export SVG nicht hell");
  const lade = async (knopf) => {
    const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 15000 }), page.click(knopf)]);
    return [dl.suggestedFilename(), fs.readFileSync(await dl.path())];
  };
  const [pngName, png] = await lade("#export-png");
  const breite = png.readUInt32BE(16), phys = png.indexOf("pHYs");
  if (!pngName.endsWith("_175x170mm.png") || breite !== 2067 || phys < 0 || png.readUInt32BE(phys + 4) !== 11811) {
    fehler.push(`Export PNG: ${pngName} ${breite} ${phys}`);
  }
  const [csvName, csv] = await lade('#export-csv button[data-csv="quantile"]');
  const zeilen = csv.toString("utf8").trim().split("\r\n");
  const kopf = zeilen.filter((z) => z.startsWith("#")), daten = zeilen.filter((z) => !z.startsWith("#"));
  if (!csvName.endsWith("_quantile.csv") || kopf.length < 2 || daten.length !== 182 ||
    !daten[0].startsWith("alter_cal_a_bp,T_DJF_q2_5") || daten[1].split(",").length !== daten[0].split(",").length) {
    fehler.push(`Export CSV: ${csvName} ${kopf.length} ${daten.length} ${daten[0]}`);
  }
  if (pdf) {
    // window.print ersetzen, dann die Druckansicht als PDF ausgeben: eine Seite in der Groesse der Abbildung
    await page.evaluate(() => { window.print = () => { window.$gedruckt = true; }; });
    await page.click("#export-pdf");
    await page.waitForFunction(() => window.$gedruckt && document.querySelector("#druck svg"), null, { timeout: 15000 })
      .catch(() => fehler.push("Export PDF: kein Druck"));
    const buf = await page.pdf({ preferCSSPageSize: true, printBackground: true });
    const text = buf.toString("latin1");
    const box = /\/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]/.exec(text);
    const seiten = (text.match(/\/Type \/Page\b/g) || []).length;
    if (!box || Math.abs(box[1] - 496.06) > 1 || Math.abs(box[2] - 481.89) > 1 || seiten !== 1) {
      fehler.push(`Export PDF: ${box && box.slice(1)} Seiten ${seiten}`);
    }
    await page.evaluate(() => window.dispatchEvent(new Event("afterprint")));
    if (await page.evaluate(() => document.body.classList.contains("druckt"))) fehler.push("Export PDF: Druckzustand bleibt");
  }
  await page.click('#export button[value="zu"]');
}

(async () => {
  const browser = await chromium.launch();
  let fehlerGesamt = 0;
  for (const datei of args) {
    for (const [name, groesse] of [["handy", { width: 390, height: 844 }], ["pc", { width: 1366, height: 900 }]]) {
      for (const thema of ["light", "dark"]) {
        const tag = `${path.basename(datei, ".html")}_${name}_${thema}`;
        const r = await pruefe(browser, datei, groesse, thema, tag);
        console.log(`${tag}: ${r.fehler.length ? "FEHLER" : "ok"}`);
        r.fehler.forEach((f) => console.log("   " + f));
        if (thema === "light" && name === "pc") r.zeilen.forEach((z) => console.log("   | " + z.replace(/\s+/g, " ")));
        fehlerGesamt += r.fehler.length;
      }
    }
  }
  await browser.close();
  process.exit(fehlerGesamt ? 1 : 0);
})();
