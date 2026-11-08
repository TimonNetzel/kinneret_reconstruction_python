// Prueft die gebaute App im Browser (Playwright, Chromium): Laden ueber file:// ohne Netz, Ansichten, Sprache,
// Thema, Handy- und PC-Groesse; meldet Fehler der Konsole. Optional Bildschirmfotos.
//
// Aufruf: node tools/app_pruefen.js results/app/index.html [results/kinneret-app.html] [--fotos ORDNER]
//         node tools/app_pruefen.js --zwei-standorte ORDNER/index.html  (Standort-Waehler; Ordner mit einem zweiten
//         Standort "zweit", wie ihn tests/test_app.py anlegt)
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const ki = args.indexOf("--kanal");
const kanal = ki >= 0 ? args.splice(ki, 2)[1] : null;
const fi = args.indexOf("--fotos");
const fotos = fi >= 0 ? args.splice(fi, 2)[1] : null;
if (fotos) fs.mkdirSync(fotos, { recursive: true });
const zi = args.indexOf("--zwei-standorte");
const zweiStandorte = zi >= 0 ? args.splice(zi, 1).length > 0 : false;

async function neueSeite(browser, datei, groesse) {
  const ctx = await browser.newContext({ viewport: groesse, offline: true, deviceScaleFactor: 1, locale: "de-DE" });
  const page = await ctx.newPage();
  const fehler = [];
  page.on("console", (m) => { if (m.type() === "error") fehler.push(m.text()); });
  page.on("pageerror", (e) => fehler.push(String(e)));
  await page.goto("file://" + path.resolve(datei));
  await page.waitForSelector("#rek-plot .main-svg", { timeout: 20000 });
  return [page, fehler];
}

// Gesten: auf dem Handy (Touch) scrollt Wischen ueber ein Diagramm die Seite, Antippen zeigt nur Werte (Achsen und
// Kurven bleiben, in jeder Ansicht); mit der Maus zieht man einen Zoom-Rahmen auf
async function wische(ctx, page, sel, dx, dy) {
  const box = await page.locator(sel + " .nsewdrag").first().boundingBox();
  const x = box.x + box.width / 2, y = box.y + box.height * 0.8;
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y }] });
  for (let k = 1; k <= 10; k++) {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: x + dx * k / 10, y: y + dy * k / 10 }] });
  }
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await page.waitForTimeout(600);
}
const bereich = (page, sel) => page.evaluate((sel) => JSON.stringify(document.querySelector(sel)._fullLayout.xaxis.range), sel);
// alle Achsenbereiche und die Zahl gezeichneter Linien eines Diagramms
const bild = (page, sel) => page.evaluate((sel) => {
  const fl = document.querySelector(sel)._fullLayout;
  return JSON.stringify([Object.keys(fl).filter((k) => /^[xy]axis\d*$/.test(k)).sort().map((k) => fl[k].range),
    document.querySelectorAll(sel + " .scatterlayer path.js-line").length]);
}, sel);

async function pruefeGesten(browser, datei) {
  const fehler = [];
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true,
    offline: true, locale: "de-DE" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => fehler.push(String(e)));
  await page.goto("file://" + path.resolve(datei));
  await page.waitForSelector("#rek-plot .main-svg", { timeout: 20000 });
  await page.locator("#rek-plot").scrollIntoViewIfNeeded();
  const vorher = [await page.evaluate(() => scrollY), await bereich(page, "#rek-plot")];
  await wische(ctx, page, "#rek-plot", 60, -100);
  const nachher = [await page.evaluate(() => scrollY), await bereich(page, "#rek-plot")];
  if (nachher[0] < vorher[0] + 50 || nachher[1] !== vorher[1]) fehler.push(`Touch: Wischen ${vorher} -> ${nachher}`);
  // Antippen in jeder Ansicht (Diagramm unter den fixierten Kopf geschoben): nur Werte, Achsen und Kurven bleiben
  for (const [ansicht, sel] of [["rekonstruktion", "#rek-plot"], ["sensitivitaet", "#sens-plot"], ["gewichte", "#gew-plot"],
    ["transferfunktionen", "#tf-plot"], ["lauf", "#lauf-kalibrierung"], ["validierung", "#val-plot"]]) {
    await page.click(`.reiter button[data-ansicht="${ansicht}"]`);
    await page.waitForSelector(sel + " .main-svg", { timeout: 15000 });
    await page.waitForTimeout(400);
    await page.evaluate((sel) => scrollBy(0, document.querySelector(sel).getBoundingClientRect().top -
      document.querySelector(".kopf").getBoundingClientRect().bottom - 10), sel);
    const vor = await bild(page, sel);
    const box = await page.locator(sel + " .nsewdrag").first().boundingBox();
    await page.touchscreen.tap(box.x + box.width * 0.45, box.y + box.height * 0.5);
    await page.waitForTimeout(800);
    const nach = await bild(page, sel);
    if (vor !== nach) fehler.push(`Touch: Antippen veraendert ${ansicht}: ${vor} -> ${nach}`);
    // Werte bleiben stehen (Diagramme mit Werten je Alter; die uebrigen zeigen sie nur nahe an einem Punkt)
    if (["rekonstruktion", "sensitivitaet"].includes(ansicht) &&
      !(await page.evaluate((sel) => document.querySelectorAll(sel + " .hoverlayer .hovertext").length, sel))) {
      fehler.push(`Touch: Antippen zeigt keine Werte (${ansicht})`);
    }
    if (ansicht === "rekonstruktion" && !(await page.evaluate(() => document.querySelector("#rek-detail").hidden))) {
      fehler.push("Touch: Antippen oeffnet Detail");
    }
  }
  await ctx.close();
  // Maus: Ziehen zoomt
  const ctx2 = await browser.newContext({ viewport: { width: 1366, height: 900 }, offline: true, locale: "de-DE" });
  const p2 = await ctx2.newPage();
  await p2.goto("file://" + path.resolve(datei));
  await p2.waitForSelector("#rek-plot .main-svg", { timeout: 20000 });
  const b2 = await p2.locator("#rek-plot .nsewdrag").first().boundingBox();
  const r0 = await bereich(p2, "#rek-plot");
  await p2.mouse.move(b2.x + b2.width * 0.3, b2.y + b2.height * 0.5);
  await p2.mouse.down();
  await p2.mouse.move(b2.x + b2.width * 0.6, b2.y + b2.height * 0.5, { steps: 10 });
  await p2.mouse.up();
  await p2.waitForTimeout(600);
  if (await bereich(p2, "#rek-plot") === r0) fehler.push("Maus: Ziehen zoomt nicht");
  await ctx2.close();
  return fehler;
}

// Standort-Waehler mit zwei Standorten: Wechsel, Lauflisten nur des Standorts, Sensitivitaet ohne Reihe
async function pruefeStandorte(browser, datei) {
  const [page, fehler] = await neueSeite(browser, datei, { width: 1366, height: 900 });
  // Lauf-Liste der sichtbaren Ansicht (die anderen fuellen sich erst beim Anzeigen)
  const zustand = (liste = "#rek-lauf") => page.evaluate((liste) => {
    const st = (id) => KR.LAEUFE.get(id).standort;
    return { lauf: KR.Z.lauf, sichtbar: !document.querySelector("#standort-wahl").hidden,
      optionen: [...document.querySelectorAll("#standort-wahl option")].map((o) => o.value),
      fremd: [...document.querySelectorAll(liste + " option")].filter((o) => st(o.value) !== st(KR.Z.lauf)).length,
      n: document.querySelectorAll(liste + " option").length, name: document.querySelector("#standort-name").hidden };
  }, liste);
  let z = await zustand();
  if (!z.sichtbar || z.optionen.join() !== "kinneret,zweit" || !z.name) fehler.push("Waehler: " + JSON.stringify(z));
  if (z.fremd || z.n < 20) fehler.push("Lauf-Liste Kinneret: " + JSON.stringify(z));
  await page.selectOption("#standort-wahl", "zweit");
  await page.waitForTimeout(500);
  z = await zustand();
  if (z.lauf !== "zweit_lauf" || z.fremd || z.n !== 1) fehler.push("nach Wechsel: " + JSON.stringify(z));
  await page.waitForSelector("#rek-plot .main-svg", { timeout: 10000 });
  await page.click('.reiter button[data-ansicht="sensitivitaet"]');
  await page.waitForTimeout(500);
  const text = await page.textContent("#sens-tabelle");
  if (!text.includes("keine Sensitivitätsläufe")) fehler.push("Sensitivitaet ohne Reihe: " + text);
  await page.click('.reiter button[data-ansicht="gewichte"]');
  await page.waitForSelector("#gew-plot .main-svg", { timeout: 10000 }).catch(() => fehler.push("Gewichte am 2. Standort"));
  await page.selectOption("#standort-wahl", "kinneret");
  await page.waitForTimeout(500);
  z = await zustand("#gew-lauf");
  if (z.lauf !== "standard_v3" || z.fremd || z.n < 20) fehler.push("zurueck: " + JSON.stringify(z));
  await page.click('.reiter button[data-ansicht="sensitivitaet"]');
  await page.waitForFunction(() => document.querySelectorAll("#sens-tabelle tbody tr").length >= 2, null, { timeout: 15000 })
    .catch(() => fehler.push("Sensitivitaet nach Rueckwechsel"));
  // Species evidence is project-wide and stays identical when the selected site changes.
  await page.click('.reiter button[data-ansicht="artenvergleich"]');
  await page.waitForSelector(".arten-art");
  const artenVorher = await page.locator("#arten-inhalt").innerText();
  await page.selectOption("#standort-wahl", "zweit");
  await page.waitForTimeout(300);
  if (await page.locator("#arten-inhalt").innerText() !== artenVorher) fehler.push("Artenvergleich aendert sich mit Standort");
  await page.click('#arten-ebene button[data-wert="folds"]');
  await page.selectOption("#arten-taxon", "taxus_baccata");
  const foldsVorher = await page.locator("#arten-inhalt").innerText();
  await page.selectOption("#standort-wahl", "kinneret");
  await page.waitForTimeout(300);
  if (await page.locator("#arten-inhalt").innerText() !== foldsVorher) fehler.push("Foldwerte aendern sich mit Standort");
  // Research snapshot and dissertation reproduction state are project-wide as well
  await page.click('.reiter button[data-ansicht="forschung"]');
  await page.waitForSelector('[data-research="reproduction"] [data-module]');
  await page.click('[data-research="reproduction"] [data-part="example"] summary');
  const forschungVorher = await page.locator("#forschung-inhalt").innerText();
  await page.selectOption("#standort-wahl", "zweit");
  await page.waitForTimeout(300);
  if (await page.locator("#forschung-inhalt").innerText() !== forschungVorher) fehler.push("Forschungs-/Nachbaustand aendert sich mit Standort");
  await page.selectOption("#standort-wahl", "kinneret");
  return fehler;
}

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
  // ein Standort: kein Waehler, nur der Name
  const kopf = await page.evaluate(() => [document.querySelector("#standort-wahl").hidden,
    document.querySelector("#standort-name").textContent]);
  if (!kopf[0] || kopf[1] !== "See Genezareth") fehler.push("Kopf Standort: " + kopf);
  const n = await page.evaluate(() => document.querySelectorAll("#rek-plot .scatterlayer .trace").length);
  if (n < 5) fehler.push("Rekonstruktion: zu wenige Kurven " + n);
  // Klick in das Diagramm zeigt nur Werte; das Detail kommt ueber die Auswahl
  const box = await page.locator("#rek-plot .nsewdrag").first().boundingBox();
  await page.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.5);
  await page.waitForTimeout(300);
  if (!(await page.evaluate(() => document.querySelector("#rek-detail").hidden))) fehler.push("Klick oeffnet Detail");
  await page.selectOption("#rek-detail-wahl", "40");
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
  // Moderne Validierung: Beschriftung (Gitter, Puffer, Proben, Vergleich), Hauptmodell k = 5, Wechsel auf 10 Grad
  await page.click('.reiter button[data-ansicht="validierung"]');
  await warte("#val-plot");
  const val = () => page.evaluate(() => [document.querySelector("#val-info").innerText,
    document.querySelector("#val-tabelle tr.basis").innerText.replace(/\s+/g, " "),
    document.querySelectorAll("#val-folds tbody tr").length, document.querySelector("#val-regional-hinweis").textContent]);
  let v = await val();
  if (!["7.078 moderne Proben", "5°-Gitter", "100 km", "Trainingsfolds", "°C", "mm/Jahr", "keine fossilen"]
    .every((x) => v[0].includes(x))) fehler.push("Validierung Info: " + v[0].slice(0, 120));
  if (!v[1].startsWith("k = 5 Analoge Hauptmodell 3,765") || v[2] !== 5 || !v[3].includes("kein zusätzlicher unabhängiger Test")) {
    fehler.push(`Validierung: ${v.slice(1)}`);
  }
  await page.click('#val-block button[data-wert="10"]');
  await page.waitForTimeout(300);
  v = await val();
  if (!v[1].includes("3,926") || !v[0].includes("10°-Gitter")) fehler.push("Validierung 10 Grad: " + v[1]);
  await page.click('#val-block button[data-wert="5"]');
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_validierung.png`, fullPage: true });
  // Forschungsstand: five read-only evidence cards, source links, no figure export.
  await page.click('.reiter button[data-ansicht="forschung"]');
  await page.waitForSelector('[data-research="historical"]');
  const researchText = await page.textContent("#forschung-inhalt");
  if (await page.locator(".forschung-karte").count() !== 5 ||
      !["167", "161", "278", "279", "Brier", "Marginalarrays"].every((x) => researchText.includes(x))) {
    fehler.push("Forschungsstand unvollstaendig");
  }
  if (await page.locator('#ansicht-forschung .export-knopf').count() !== 0 ||
      await page.locator('[data-research="historical"] [data-status="missing"]').count() !== 1) {
    fehler.push("Forschungsstand: Kontroll-/Statusgrenze fehlt");
  }
  const sources = await page.locator('.forschung-karte a').evaluateAll((links) => links.map((a) => a.href));
  if (sources.length !== 5 || sources.some((url) => !url.includes('/blob/1cec59f1c37ac59f1440969feeaf4451e1b93dcc/'))) {
    fehler.push("Forschungsstand: Quellen nicht gebunden");
  }
  // Dissertation reproduction: newest state with twelve checked submodules, seven frozen earlier states, R cross check, synthetic examples kept apart
  const nb = '[data-research="reproduction"]';
  const nbText = await page.textContent(nb);
  if (!nbText.includes("Teilmodule geprüft; vollständiger Originalfall offen") || !nbText.includes("934") ||
      await page.locator(nb + " [data-module]").count() !== 12 ||
      await page.locator(nb + ' [data-module="gamma_prior_grid"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + ' [data-module="tf_missing_priors"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + ' [data-module="sampler_preparation"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + ' [data-module="weights_stage"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + ' [data-module="dirichlet_ratio"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + ' [data-module="weighted_median"] [data-status="conditional"]').count() !== 0 ||
      await page.locator(nb + " [data-earlier-state]").count() !== 7 ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-03, 11 Teilmodule, 752 Tests, SHA-256 cf1b77f22bdd") ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-03, 10 Teilmodule, 689 Tests, SHA-256 7eb9ac5e4eb6") ||
      await page.locator(nb + ' [data-part="runtime-median"]').count() !== 1 ||
      !["matrixStats 0.56.0", "439 von 439 Werten bitgleich", "höchstens 0 ulp"].every((x) => nbText.includes(x)) ||
      await page.locator(nb + ' [data-part="runtime"]').count() !== 1 ||
      !["R version 3.5.0", "389 bitgleiche Prüfungen", "59 von 62", "höchstens 4 ulp", "Slot 3", "LaplacesDemon 16.1.6"
        ].every((x) => nbText.includes(x)) ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-02, 9 Teilmodule, 555 Tests, SHA-256 61c6f23142df") ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-02, 8 Teilmodule, 437 Tests, SHA-256 7b7bf1f449a1") ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-02, 7 Teilmodule, 352 Tests, SHA-256 43edfcb67a64") ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-02, 6 Teilmodule, 297 Tests, SHA-256 3656316de421") ||
      await page.locator(nb + ' [data-module="taxon_pruning"] [data-status="conditional"]').count() !== 1 ||
      !nbText.includes("Früherer Stand eingefroren: 2026-11-02, 5 Teilmodule, 251 Tests, SHA-256 c358b2c3382f") || await page.locator(nb + " [data-requirement]").count() !== 11 ||
      await page.locator(nb + " [data-central]").count() !== 4 ||
      await page.locator(nb + ' [data-requirement] [data-status="conflicting"]').count() !== 4) {
    fehler.push("Nachbau: Teilmodule/Voraussetzungen unvollstaendig");
  }
  const nbLinks = await page.locator(nb + " a").evaluateAll((links) => links.map((a) => a.href));
  const frueher = ["/blob/47a926c3c8a6abc0db7c947b993f5e97f7408cdd/docs/thesis_research/app_reproduction_2026_11_02/reproduction_state.json",
    "/blob/3d013b276cb1b7af37ce4058d6c10e499a34b504/docs/thesis_research/app_reproduction_2026_11_02_pruning/reproduction_state.json",
    "/blob/7d3d0f6852dbdf869b6e479d4eeffe80faaf2038/docs/thesis_research/app_reproduction_2026_11_02_gamma_grid/reproduction_state.json",
    "/blob/0a3dcdb8fde7292f6b9c1596dbade5cc7c455e34/docs/thesis_research/app_reproduction_2026_11_02_tf_missing/reproduction_state.json",
    "/blob/235f79784d01f0a613a0d3f76f6fd690c245ab21/docs/thesis_research/app_reproduction_2026_11_02_sampler_prep/reproduction_state.json",
    "/blob/3104f3c9e9a7870138abcc301bdd4f108c009e73/docs/thesis_research/app_reproduction_2026_11_03_weights_stage/reproduction_state.json",
    "/blob/cfd0ad8f1c0cb3f06909f50f21cc1e9a65592a68/docs/thesis_research/app_reproduction_2026_11_03_dirichlet_ratio/reproduction_state.json"];
  const istFrueher = (url) => frueher.some((f) => url.endsWith(f));
  if (nbLinks.length !== 31 || frueher.some((f) => nbLinks.filter((url) => url.endsWith(f)).length !== 1) ||
      nbLinks.some((url) => !istFrueher(url) && !url.includes("/blob/a21c8f1ea7f6c285cbed2695a8b3750e4d0bba10/"))) {
    fehler.push("Nachbau: Quellen nicht gebunden " + nbLinks.length);
  }
  if (await page.locator(nb + ' [data-part="example"]').evaluate((d) => d.open)) fehler.push("Nachbau: Beispiel nicht eingeklappt");
  await page.click(nb + ' [data-part="example"] summary');
  const exText = await page.textContent(nb + ' [data-part="example"]');
  if (!(await page.locator('[data-synthetic="true"]').isVisible()) || await page.locator(nb + " table").count() !== 14 ||
      await page.locator(nb + ' [data-table="median"] tbody tr').count() !== 9 ||
      await page.locator(nb + ' [data-table="median"] [data-special="na"]').count() !== 4 ||
      await page.locator(nb + ' [data-table="median"] [data-ulps="0"]').count() !== 9 ||
      await page.locator(nb + ' [data-table="median-mask"] tr[data-removed="true"]').count() !== 3 ||
      await page.locator(nb + ' [data-table="median-mask"] [data-special="na"]').count() !== 4 ||
      await page.locator(nb + ' [data-table="dirichlet"] [data-special="nan-zero"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="dirichlet"] [data-ulps="4"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="dirichlet"] [data-ulps="0"]').count() !== 3 ||
      await page.locator(nb + ' [data-table="weights"] [data-acc="dependent"]').count() !== 2 ||
      await page.locator(nb + ' [data-table="weights"] [data-acc="exact"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="weights-special"] [data-special]').count() !== 2 ||
      await page.locator(nb + ' [data-table="sampler"] [data-acc="dependent"]').count() !== 3 ||
      await page.locator(nb + ' [data-table="sampler"] [data-acc="exact"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="sampler-special"] [data-special]').count() !== 3 ||
      await page.locator(nb + ' [data-table="missing"] [data-differs="true"]').count() !== 6 ||
      await page.locator(nb + ' [data-table="missing"] [data-sd="0"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="missing"] [data-line="1678"]').count() !== 4 ||
      await page.locator(nb + ' [data-table="gamma"] [data-differs="true"]').count() !== 3 ||
      await page.locator(nb + ' [data-table="gamma"] [data-shape="25.000000000000004"]').count() !== 1 ||
      await page.locator(nb + ' [data-table="pruning"] [data-removed="true"]').count() !== 2 ||
      await page.locator('[data-row-mass="2/3"]').count() !== 1 || await page.locator('[data-row-mass="1"]').count() !== 1 ||
      !["Kunsttaxon 2", "1/3", "Kunstprobe 4", "0,5005", "0,1000", "0,3333", "2,22", "keine Kalenderalter",
        "Nullanteil", "Nullen (von 20)", "3/20", "16:T abwärts", "ohne Taxon-Pruning", "Kunstfall", "Form laut Quelle",
        "25,000000000000004", "62.500", "0,0124", "Zeile 1733", "500.000", "Fehlende Transferfunktionen: 4 Kunstfälle",
        "ACDF", "aufgerundet", "Zeile 1678 (nrow)", "5 aus Zeile 1435", "2, 10, 18, 26, 34", "-integer(0)",
        "phi1 (alt)", "Sampler-Vorbereitung: 6 Kunstfälle", "2 1 4 3 / 4 3 2 6 / 1 6 8 9", "1/6", "7/3*",
        "8 von 12 Positionen", "Sonderwerte: drei Kunstfälle", "4×3", "8/3, 8/3, 4", "floor(5/2) = 2",
        "how_many_taxa = num_taxa = 5", "0,5 je Taxon",
        "ohne Taxon-Pruning, Gamma-Raster, TF-Entfernung, Sampler-Vorbereitung, Gewichtsstufe, Dirichlet-Quotient und gewichteter Median",
        "Gewichteter Median: 9 Kunstfälle", "M1", "M9", "NA-Maske: Kunstmatrix mit 5 Taxa", "Maske: 2, 3, 5", "1,2857", "stabil sortiert käme 1,4000",
        "Dirichlet-Quotient: 4 Kunstfälle", "D1", "D4", "R (ulp)", "1,1547", "0,7071", "Inf", "2/√3, √6/4 und 1/√2",
        "weist aber nichts zu", "in Zeile 1273 zu 0",
        "Kern der Gewichtsstufe: 7 Kunstfälle", "1/8, 3/8, 1/4, 1/4", "1/29", "29/32", "0,3333*", "0,5417*",
        "8 von 12 Positionen ein fremdes Gewicht", "z = 1:4", "Zeilensummen mit Sonderwerten: drei Kunstfälle",
        "G1", "G3", "1/4, 3/8, 5/8", "1 − 2⁻⁵⁴", "0,3333333333333333", "0,5714285714285715", "1, 1/2, 1/2",
        "rdirichlet (Zeile 1248)"
        ].every((x) => exText.includes(x))) {
    fehler.push("Nachbau: synthetisches Beispiel unvollstaendig");
  }
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) fehler.push("Forschungsstand: horizontaler Ueberlauf");
  const nbUeberlauf = await page.locator(nb + " .tabelle-rahmen").evaluateAll((r) => r.filter((x) => x.scrollWidth > x.clientWidth + 1).length);
  if (nbUeberlauf) fehler.push("Nachbau: " + nbUeberlauf + " Tabellen breiter als ihr Rahmen");
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_forschung.png`, fullPage: true });
  await page.click("#sprache");
  const enText = await page.textContent("#forschung-inhalt");
  if (!enText.includes("not 279 independent species functions")) {
    fehler.push("Forschungsstand: englische Grenze fehlt");
  }
  if (!enText.includes("Submodules checked; full original case open") || !enText.includes("Artificial sample 4") ||
      !enText.includes("Taxon pruning by zero share") || !enText.includes("Gamma prior from a shape grid") ||
      !enText.includes("Shape per source") || !enText.includes("Earlier state frozen") ||
      !enText.includes("Removal of missing TFs and taxon-count priors") || !enText.includes("half up") ||
      !enText.includes("Sampler preparation from taxon count and taxon matrix") ||
      !enText.includes("Special values: three artificial cases") || !enText.includes("8 of 12 positions") ||
      !enText.includes("Core of the weights stage with given weights") ||
      !enText.includes("Row sums with special values: three artificial cases") || !enText.includes("0.5714285714285715") ||
      !enText.includes("Dirichlet ratio of the weights stage") || !enText.includes("Dirichlet ratio: 4 artificial cases") ||
      !enText.includes("R cross check: in R version 3.5.0") ||
      !enText.includes("Weighted median and the NA mask of the transfer functions") ||
      !enText.includes("Weighted median: 9 artificial cases") || !enText.includes("439 of 439 values bit-identical") ||
      !(await page.locator(nb + ' [data-part="example"]').evaluate((d) => d.open))) {
    fehler.push("Nachbau: englische Fassung oder Aufklappzustand fehlt");
  }
  await page.click("#sprache");
  // Absent or partial evidence stays visible as such; the two bindings do not depend on each other
  const nbKontrolle = await page.evaluate(() => {
    const [f, n] = [KR.D.forschung, KR.D.nachbau];
    KR.D.nachbau = { schema_version: 1, available: false };
    KR.zeige();
    const ohneNachbau = [document.querySelectorAll(".forschung-karte").length,
      !!document.querySelector('[data-research="reproduction"] [role="status"]'), document.querySelectorAll("[data-module]").length];
    KR.D.nachbau = n; KR.D.forschung = { available: false };
    KR.zeige();
    const ohneSnapshot = [document.querySelectorAll(".forschung-karte").length, document.querySelectorAll("[data-module]").length];
    KR.D.forschung = f;
    KR.zeige();
    return JSON.stringify([ohneNachbau, ohneSnapshot]);
  });
  if (nbKontrolle !== JSON.stringify([[5, true, 0], [0, 12]])) fehler.push("Nachbau: Fehlstandskontrolle " + nbKontrolle);
  await page.click('.reiter button[data-ansicht="sensitivitaet"]');
  await page.waitForTimeout(300);
  // Stored species scores: both measures together and both spatial designs.
  await page.click('.reiter button[data-ansicht="artenvergleich"]');
  await page.waitForSelector(".arten-art");
  if (await page.locator(".arten-art").count() !== 7) fehler.push("Artenvergleich: sieben Arten fehlen");
  await page.click('#arten-block button[data-wert="10"]');
  await page.check("#arten-extra");
  if (await page.locator(".arten-art tbody tr").count() !== 28) fehler.push("Artenvergleich: Varianten fehlen");
  if (await page.locator("#ansicht-artenvergleich .export-knopf").count()) fehler.push("Artenvergleich: unerwarteter Abbildungsexport");
  if (fotos) await page.screenshot({ path: `${fotos}/${name}_artenvergleich.png`, fullPage: true });
  await page.click('#arten-ebene button[data-wert="folds"]');
  await page.selectOption("#arten-taxon", "abies_alba");
  if (await page.locator(".arten-fold").count() !== 5 || await page.locator(".arten-fold tbody tr").count() !== 20) {
    fehler.push("Artenvergleich: Foldwerte fehlen");
  }
  await page.click('#arten-ebene button[data-wert="summary"]');
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
  for (const ansicht of ["rekonstruktion", "sensitivitaet", "gewichte", "transferfunktionen", "lauf", "validierung"]) {
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
  for (const ansicht of ["rekonstruktion", "sensitivitaet", "gewichte", "transferfunktionen", "lauf", "validierung"]) {
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
  const browser = await chromium.launch(kanal ? { channel: kanal } : {});
  let fehlerGesamt = 0;
  if (zweiStandorte) {
    for (const datei of args) {
      const f = await pruefeStandorte(browser, datei);
      console.log(`${path.basename(path.dirname(datei))} zwei Standorte: ${f.length ? "FEHLER" : "ok"}`);
      f.forEach((x) => console.log("   " + x));
      fehlerGesamt += f.length;
    }
    await browser.close();
    process.exit(fehlerGesamt ? 1 : 0);
  }
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
    const g = await pruefeGesten(browser, datei);
    console.log(`${path.basename(datei, ".html")} Gesten: ${g.length ? "FEHLER" : "ok"}`);
    g.forEach((f) => console.log("   " + f));
    fehlerGesamt += g.length;
  }
  await browser.close();
  process.exit(fehlerGesamt ? 1 : 0);
})();
