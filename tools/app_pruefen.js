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
  page.on("request", (r) => { if (!r.url().startsWith("file:") && !r.url().startsWith("data:")) fehler.push("Netz: " + r.url()); });
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
  // Sprache
  await page.click("#sprache");
  await page.waitForTimeout(300);
  const reiter = await page.textContent('.reiter button[data-ansicht="sensitivitaet"]');
  if (reiter.trim() !== "Sensitivity") fehler.push("Sprache: " + reiter);
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_englisch.png` });
  await page.click("#sprache");
  await ctx.close();
  return { fehler, zeilen };
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
