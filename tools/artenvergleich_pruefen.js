// Offline browser contract for the preserved seven-species scores.
// node tools/artenvergleich_pruefen.js APP/index.html SINGLE.html [--kanal msedge]
//      [--fotos DIR] [--bericht FILE.json]
const fs = require("fs");
const path = require("path");
const assert = require("node:assert/strict");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");
const args = process.argv.slice(2);
function option(name) {
  const i = args.indexOf(name);
  return i < 0 ? null : args.splice(i, 2)[1];
}
const channel = option("--kanal"), photos = option("--fotos"), report = option("--bericht");
if (!args.length) throw Error("App HTML paths required");
if (photos) fs.mkdirSync(photos, { recursive: true });
const dir = path.resolve(__dirname, "../docs/thesis_research/smote_nnet_2026_10_30");
const lines = fs.readFileSync(path.join(dir, "metrics_summary.csv"), "utf8").trim().split(/\r?\n/);
const columns = lines.shift().split(",");
const source = lines.map((line) => Object.fromEntries(line.split(",").map((v, i) => [columns[i], v])));
const foldLines = fs.readFileSync(path.join(dir, "scores.csv"), "utf8").trim().split(/\r?\n/);
const foldColumns = foldLines.shift().split(",");
const foldSource = foldLines.map((line) => Object.fromEntries(line.split(",").map((v, i) => [foldColumns[i], v]))).filter((r) => r.scope !== "all");
const species = [...new Set(source.map((r) => r.taxon))];
const cases = [];

async function checkTables(page, block, methods, language) {
  const locale = language === "de" ? "de-DE" : "en-GB";
  const format = (v, n) => Number(v).toLocaleString(locale, { minimumFractionDigits: n, maximumFractionDigits: n });
  const tables = await page.locator(".arten-art").evaluateAll((els) => els.map((e) => ({
    taxon: e.dataset.taxon, rows: [...e.querySelectorAll("tbody tr")].map((r) => ({
      method: r.dataset.method, values: [...r.querySelectorAll("td")].map((c) => c.textContent)
    }))
  })));
  assert.equal(tables.length, 7);
  for (const table of tables) {
    assert.deepEqual(table.rows.map((r) => r.method), methods);
    for (const row of table.rows) {
      const expected = source.find((r) => r.taxon === table.taxon && Number(r.block_degrees) === block && r.method === row.method);
      assert(expected);
      assert.deepEqual(row.values, [format(100 * Number(expected.balanced_accuracy), 2) + " %", format(expected.brier, 5)]);
    }
  }
  const means = await page.locator(".arten-mittel").evaluateAll((els) => els.map((e) => ({
    block: Number(e.dataset.block), rows: [...e.querySelectorAll("tbody tr")].map((r) => ({
      method: r.dataset.method, values: [...r.querySelectorAll("td")].map((c) => c.textContent)
    }))
  })));
  assert.deepEqual(means.map((x) => x.block), [5, 10]);
  for (const mean of means) for (const row of mean.rows) {
    const expected = source.filter((r) => Number(r.block_degrees) === mean.block && r.method === row.method);
    assert.equal(expected.length, 7);
    const avg = (key) => expected.reduce((sum, r) => sum + Number(r[key]), 0) / 7;
    assert.deepEqual(row.values, [format(100 * avg("balanced_accuracy"), 2) + " %", format(avg("brier"), 5)]);
  }
}

async function checkFolds(page, taxon, block, methods, language) {
  const locale = language === "de" ? "de-DE" : "en-GB";
  const fmt = (v, n) => Number(v).toLocaleString(locale, { minimumFractionDigits: n, maximumFractionDigits: n });
  const tables = await page.locator(".arten-fold").evaluateAll((els) => els.map((e) => ({
    fold: Number(e.dataset.fold), counts: e.querySelector("[data-counts]").textContent,
    rows: [...e.querySelectorAll("tbody tr")].map((r) => ({ method: r.dataset.method,
      values: [...r.querySelectorAll("td")].map((c) => c.textContent) }))
  })));
  assert.deepEqual(tables.map((r) => r.fold), [0, 1, 2, 3, 4]);
  assert.equal(await page.locator("#arten-taxon").inputValue(), taxon);
  for (const table of tables) {
    assert.deepEqual(table.rows.map((r) => r.method), methods);
    for (const row of table.rows) {
      const r = foldSource.find((x) => x.taxon === taxon && Number(x.block_degrees) === block && Number(x.scope) === table.fold && x.method === row.method);
      assert(r);
      const ba = r.balanced_accuracy === "" ? (language === "de" ? "Nicht definiert" : "Undefined") : fmt(100 * Number(r.balanced_accuracy), 2) + " %";
      assert.deepEqual(row.values, [ba, fmt(r.brier, 5)]);
      const n = fmt(r.cells, 0), p = fmt(r.positive_cells, 0), a = fmt(Number(r.cells) - Number(r.positive_cells), 0);
      const counts = language === "de" ? `${n} ursprüngliche Testzellen: ${p} Präsenzen und ${a} kartierte Außerhalb-Zellen.` :
        `${n} original test cells: ${p} presences and ${a} mapped outside-range cells.`;
      assert.equal(table.counts, counts);
    }
  }
}

(async () => {
  const browser = await chromium.launch(channel ? { channel } : {});
  try {
    for (const file of args) for (const width of [390, 1366]) for (const theme of ["light", "dark"]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme,
        offline: true, locale: "de-DE" });
      const page = await context.newPage(), errors = [], requests = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
      page.on("request", (r) => { if (!/^(file|data|blob):/.test(r.url())) requests.push(r.url()); });
      await page.goto(pathToFileURL(path.resolve(file)).href);
      await page.waitForSelector("#rek-plot .main-svg");
      const run = await page.evaluate(() => KR.Z.lauf);
      await page.click('.reiter button[data-ansicht="forschung"]');
      await page.click("#fr-artenvergleich");
      await page.locator(".arten-art").first().waitFor();
      assert.equal(await page.locator("#ansicht-artenvergleich .export-knopf").count(), 0);
      const methods = ["nnet_original", "nnet_smote"];
      const allMethods = [...methods, "pr28_quadratic", "nnet_smote_prior_adjusted"];
      for (const language of ["de", "en"]) {
        if (language === "en") await page.click("#sprache");
        const title = await page.locator('[data-ansicht="artenvergleich"]').innerText();
        assert.equal(title, language === "de" ? "Artenvergleich" : "Species comparison");
        assert.equal(await page.locator('#arten-inhalt [data-status="experimental"]').innerText(), language === "de" ? "Experimentell" : "Experimental");
        for (const block of [5, 10]) {
          await page.click(`#arten-block button[data-wert="${block}"]`);
          assert.equal(await page.locator(`#arten-block button[data-wert="${block}"]`).getAttribute("aria-pressed"), "true");
          await page.uncheck("#arten-extra");
          await checkTables(page, block, methods, language);
          await page.check("#arten-extra");
          await checkTables(page, block, allMethods, language);
        }
        await page.click('#arten-ebene button[data-wert="folds"]');
        for (const block of [5, 10]) {
          await page.click(`#arten-block button[data-wert="${block}"]`);
          for (const taxon of species) {
            await page.selectOption("#arten-taxon", taxon);
            await page.uncheck("#arten-extra");
            await checkFolds(page, taxon, block, methods, language);
            await page.check("#arten-extra");
            await checkFolds(page, taxon, block, allMethods, language);
          }
        }
        await page.selectOption("#arten-taxon", "abies_alba");
        assert.equal(await page.locator(".arten-fold .tabelle-rahmen").evaluateAll(
          (els) => els.some((e) => e.scrollWidth > e.clientWidth + 1)), false, "fold tables overflow");
        const foldText = await page.locator("#arten-inhalt").innerText();
        assert(foldText.includes(language === "de" ? "kein Konfidenzintervall" : "neither a confidence interval"));
        assert(!/av_[a-z_]+|fr_[a-z_]+|\{[a-z]+\}/.test(foldText));
        if (photos) await page.screenshot({ path: path.join(photos, `${path.basename(file, ".html")}_${width}_${theme}_${language}_folds.png`), fullPage: true });
        await page.click('#arten-ebene button[data-wert="summary"]');
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, "page overflows");
        assert.equal(await page.locator(".arten-art .tabelle-rahmen, .arten-mittel .tabelle-rahmen").evaluateAll(
          (els) => els.some((e) => e.scrollWidth > e.clientWidth + 1)), false, "score tables overflow");
        const text = await page.locator("#arten-inhalt").innerText();
        assert(!/av_[a-z_]+|fr_[a-z_]+|\{[a-z]+\}/.test(text), "untranslated keys or placeholders");
        assert(text.includes(language === "de" ? "keine garantierte Kalibrierung" : "not guaranteed calibration"));
        assert(text.includes("Monticchio"));
        const name = `${path.basename(file, ".html")}_${width}_${theme}_${language}`;
        if (photos) await page.screenshot({ path: path.join(photos, name + ".png"), fullPage: true });
      }
      const links = await page.locator(".arten-quellen a").evaluateAll((els) => els.map((e) => e.href));
      assert.equal(links.length, 6);
      assert(links.every((u) => u.includes("/blob/2d2fdfd019d7af09fd55fd8e60e429d1c2b2f7a0/")));
      // Selection survives language, theme, view changes and a reload; it never changes the fossil run.
      await page.click("#thema");
      await checkTables(page, 10, allMethods, "en");
      await page.reload();
      await page.locator(".arten-art").first().waitFor();
      await checkTables(page, 10, allMethods, "en");
      await page.click('#arten-ebene button[data-wert="folds"]');
      await page.selectOption("#arten-taxon", "betula_pendula");
      await page.reload();
      await page.locator(".arten-fold").first().waitFor();
      await checkFolds(page, "betula_pendula", 10, allMethods, "en");
      await page.click('#arten-ebene button[data-wert="summary"]');
      assert.equal(await page.evaluate(() => KR.Z.lauf), run);
      await page.click('.reiter button[data-ansicht="rekonstruktion"]');
      await page.waitForSelector("#rek-plot .main-svg");
      await page.click('.reiter button[data-ansicht="artenvergleich"]');
      await checkTables(page, 10, allMethods, "en");
      // Missing scores must not leave stale tables or active controls; recovery stays possible.
      await page.evaluate(() => { window.savedSpecies = KR.D.artenvergleich; KR.D.artenvergleich = { available: false }; KR.zeige(); });
      assert.equal(await page.locator(".arten-art").count(), 0);
      assert.equal(await page.locator("#arten-inhalt [role=status]").isVisible(), true);
      assert.equal(await page.locator("#arten-auswahl").isVisible(), false);
      await page.evaluate(() => { KR.D.artenvergleich = window.savedSpecies; KR.zeige(); });
      await checkTables(page, 10, allMethods, "en");
      // Missing fold files preserve the overall scores. Hypothetical classless folds stay undefined.
      await page.evaluate(() => { window.savedFoldScores = KR.D.artenvergleich.fold_scores;
        KR.D.artenvergleich.fold_scores = { available: false }; });
      await page.click('#arten-ebene button[data-wert="folds"]');
      assert.equal(await page.locator(".arten-fold").count(), 0);
      assert.equal(await page.locator("#arten-inhalt [role=status]").isVisible(), true);
      await page.click('#arten-ebene button[data-wert="summary"]');
      await checkTables(page, 10, allMethods, "en");
      await page.evaluate(() => { KR.D.artenvergleich.fold_scores = structuredClone(window.savedFoldScores);
        for (const r of KR.D.artenvergleich.fold_scores.rows) if (r.taxon === "betula_pendula" && r.block_degrees === 10 && r.fold === 0) {
          r.balanced_accuracy = null; r.positive_cells = 0;
        } });
      await page.click('#arten-ebene button[data-wert="folds"]');
      const nullScores = await page.locator('.arten-fold[data-fold="0"] tbody tr td:first-of-type').allTextContents();
      assert.deepEqual(nullScores, ["Undefined", "Undefined", "Undefined", "Undefined"]);
      assert((await page.locator('.arten-fold[data-fold="0"]').innerText()).includes("No positive test cells"));
      await page.evaluate(() => { KR.D.artenvergleich.fold_scores = window.savedFoldScores; KR.zeige(); });
      await checkFolds(page, "betula_pendula", 10, allMethods, "en");
      assert.deepEqual(errors, []);
      assert.deepEqual(requests, []);
      const result = { file: path.basename(file), width, theme, languages: ["de", "en"],
        offline: true, checked_metric_rows: 56, checked_metric_values: 112, checked_mean_rows: 8,
        checked_fold_rows: 280, checked_fold_metric_values: 560, fold_counts_checked: true, undefined_class_control: true,
        missing_scores_and_recovery: true, selection_persisted: true, page_errors: errors.length, network_requests: requests.length };
      cases.push(result);
      console.log(`${path.basename(file)} ${width} ${theme}: ok`);
      await context.close();
    }
    if (report) fs.writeFileSync(report, JSON.stringify({ status: "passed", browser: channel || "chromium", cases }, null, 2) + "\n");
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
