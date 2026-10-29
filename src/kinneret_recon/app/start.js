// Start der App: Reiter aus dem Register der Ansichten, Sprache, Thema, Service Worker.
(function (KR) {
  "use strict";
  const { $, K, Z, t, L, merke, ladeStandort, setzeTexte, setzeThema, setzeAnsicht, aktuellerStandort, waehleStandort,
    ansichten, FT } = KR;

  function zeige() {
    if (!ansichten.some((a) => a.key === Z.ansicht)) Z.ansicht = ansichten[0].key;
    setzeAnsicht();
    ansichten.find((a) => a.key === Z.ansicht).zeige();
  }

  // Standort: Name im Kopf; der Waehler erscheint erst ab zwei Standorten im Katalog
  function zeigeStandort() {
    const sel = $("standort-wahl");
    sel.hidden = K.standorte.length < 2;
    $("standort-name").hidden = !sel.hidden;
    sel.innerHTML = "";
    K.standorte.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.key;
      o.textContent = FT(L(s.name));
      o.selected = s.key === aktuellerStandort();
      sel.appendChild(o);
    });
    return ladeStandort(aktuellerStandort()).then((st) => {
      $("standort-name").innerHTML = KR.F(L(st.name));
      setzeTexte();
    }).catch(() => {});
  }

  function start() {
    const reiter = $("reiter");
    ansichten.forEach((a) => {
      const b = document.createElement("button");
      b.setAttribute("role", "tab");
      b.dataset.ansicht = a.key;
      b.dataset.t = "ansicht_" + a.key;
      b.onclick = () => { Z.ansicht = a.key; merke(); zeige(); };
      reiter.appendChild(b);
    });
    // Export in jeder Ansicht (export.js)
    document.querySelectorAll(".ansicht > .steuerung").forEach((st) => {
      const b = document.createElement("button");
      b.className = "knopf export-knopf";
      b.dataset.t = "export_knopf";
      b.onclick = () => KR.oeffneExport(null);
      st.appendChild(b);
    });
    KR.exportInit();
    setzeThema();
    setzeTexte();
    $("sprache").onclick = () => {
      Z.sprache = Z.sprache === "de" ? "en" : "de"; merke(); setzeTexte(); setzeThema(); zeigeStandort(); zeige();
    };
    $("standort-wahl").onchange = (e) => { waehleStandort(e.target.value); zeigeStandort(); zeige(); };
    $("thema").onclick = () => {
      Z.thema = { auto: "light", light: "dark", dark: "auto" }[Z.thema];
      merke(); setzeThema(); zeige();
    };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (Z.thema === "auto") zeige(); });
    ansichten.forEach((a) => a.init && a.init());
    zeigeStandort();
    zeige();
    const eineDatei = !!document.querySelector('script[type="application/gzip-base64"]');
    const sicher = location.protocol === "https:" || ["localhost", "127.0.0.1"].includes(location.hostname);
    if ("serviceWorker" in navigator && sicher && !eineDatei) navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  KR.zeige = zeige;
  KR.zeigeStandort = zeigeStandort;
  window.KR_APP = { zustand: Z, ladeLauf: KR.ladeLauf, ladeStandort }; // fuer Tests
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})(window.KR);
