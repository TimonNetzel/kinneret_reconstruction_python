// Start der App: Reiter aus dem Register der Ansichten, Sprache, Thema, Service Worker.
(function (KR) {
  "use strict";
  const { $, Z, t, L, merke, ladeStandort, setzeTexte, setzeThema, setzeAnsicht, aktuellerStandort, ansichten } = KR;

  function zeige() {
    if (!ansichten.some((a) => a.key === Z.ansicht)) Z.ansicht = ansichten[0].key;
    setzeAnsicht();
    ansichten.find((a) => a.key === Z.ansicht).zeige();
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
    setzeThema();
    setzeTexte();
    $("sprache").onclick = () => { Z.sprache = Z.sprache === "de" ? "en" : "de"; merke(); setzeTexte(); setzeThema(); zeige(); };
    $("thema").onclick = () => {
      Z.thema = { auto: "light", light: "dark", dark: "auto" }[Z.thema];
      merke(); setzeThema(); zeige();
    };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (Z.thema === "auto") zeige(); });
    ansichten.forEach((a) => a.init && a.init());
    ladeStandort(aktuellerStandort()).then((st) => {
      $("standort-name").innerHTML = KR.F(L(st.name));
      setzeTexte();
    }).catch(() => {});
    zeige();
    const eineDatei = !!document.querySelector('script[type="application/gzip-base64"]');
    const sicher = location.protocol === "https:" || ["localhost", "127.0.0.1"].includes(location.hostname);
    if ("serviceWorker" in navigator && sicher && !eineDatei) navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  KR.zeige = zeige;
  window.KR_APP = { zustand: Z, ladeLauf: KR.ladeLauf, ladeStandort }; // fuer Tests
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})(window.KR);
