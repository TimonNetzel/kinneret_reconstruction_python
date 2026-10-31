# T1-Anschlussdesign für Zählkomposition und Typgemische

31.10.2026, nach Nutzer-OK **hier im bestehenden Chat**. Basis PR #39 bei
`35185534bf7f171c6f39537ebbc0449b28f5421a`; neuer Worktree
`/workspace/netzel-counts-design`, Branch `codex/netzel-counts-observation-design`.
Die sechs vorherigen Worktrees sind sauber geblieben; [Standprüfung](./git_snapshot.json).
PR #31–#39 offen, main unverändert `7f1826c`; kein Merge.
Gesichert als [PR #40](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/40)
gegen den #39-Branch, Inhaltscommit `62e0ee6`; danach nur PR-/Übergabeverweise.

**Dieses begrenzte Designpaket ist abgeschlossen.** Es definiert die mathematische
Rolle von Vegetation, Pollenfluss, Klassifikationsmatrix, Rest und Zählnenner.
**Eine reale Rekonstruktionsregel bleibt unvollständig**, weil die identifizierenden
Eingänge fehlen. Die vorhandenen Arealwahrscheinlichkeiten liefern diese Brücke
nicht automatisch. Keine echten Art-/Typzuordnungen oder Parameter geschätzt.

## Was konkret gesichert ist

- [DESIGN_COUNTS.md](./DESIGN_COUNTS.md): Flussmodell, normierte Typkomposition,
  bedingte Multinomial-Likelihood und Integration gemeinsamer latenter Größen;
  proper Priors, Support, Nullcount, fehlende Kategorien und Nennerrollen.
- [INPUT_GAPS.md](./INPUT_GAPS.md): nach Kettenglied getrennte reale Datenlücken
  und Belege. W96 und T0/Artbrücken sowie reales T1/T3/T4, Monticchio-P1 offen.
- [SYNTHETIC_PLAN.md](./SYNTHETIC_PLAN.md) mit [festen Parametern](./synthetic_plan.json):
  52 zukünftige, anonyme Posteriorfälle; vollständige PMF-Enumeration für N=8,
  Gegenbeispiele, Support-/Konfliktfälle und separate Nachrechnung geplant.
  **Noch nicht implementiert oder ausgeführt**, keine neuen Ergebnisdateien.

Wesentliche Folgerung: Relative Typcounts identifizieren keine absolute
Pollenflussskala. Unbekannte Produktionsfaktoren können ein anderes Klima
vollständig kompensieren; gleiche Typzuordnungen lassen Artenanteile untrennbar.
Diese Aussagen sind algebraisch belegt, keine empirischen Erfolgs-/Fehlerscores.
Ein enger Posterior unter fest gesetzten Faktoren wäre nur bedingt aussagekräftig.

Der spätere Zähltest setzt alle diese Faktoren ausdrücklich künstlich.
Sein gemeinsamer-H-Vergleich hält die Einzelkorn-Marginalen exakt gleich und
variiert nur die Abhängigkeit über Körner derselben Probe. Beide Auswertungen
werden berichtet; keine Auswahl eines realen Modells anhand dieses Tests.
Reale Überdispersion bleibt offen, keine automatische Dirichlet-Multinomial-Wahl.

## Tatsächlicher Prüfstand

Dokumentationsprüfung: Gleichungen/Dimensionen, Zählnenner und Restlogik,
algebraische Gegenbeispiele, 52-Fälle-Inventar gegen JSON, lokale Links,
Hashvergleich aller geschützten vorhandenen Git-Blobs und begrenzter Diffumfang.
[Prüfprotokoll](./verification.json) mit Eingangs- und Dokumenthashes.
674 geschützte bestehende Blobs einschließlich des vorhandenen Symlinks unverändert.
Kein unabhängiger menschlicher oder palynologischer Review behauptet.

**Nicht ausgeführt:** Prototypen, Simulationen, Fits, Posterioren, bestehende
Repositorytests, Browser-/CLI-Tests oder Kernrekonstruktionen. Der frühere
Testabbruch aus #39 bleibt als historischer Prüfstand unverändert erhalten;
in diesem Paket wurde kein solcher Standardtest gestartet. GitHub-CI ist davon
getrennt und wird beim Push/PR vom unveränderten Repository ausgelöst.

Bestehende Modelle, Quellregister, Resultate, Fossilcounts/Gruppen, Baseline,
CLI/App und alte Designs unverändert. Keine äußeren Scores verwendet,
keine Unteragenten oder Autorennachrichten. Nächster Auftrag, separat vorbereitet
und noch nicht autorisiert: genau den hier festgelegten kleinen synthetischen
Zählprototyp implementieren und prüfen; [HANDOFF](../HANDOFF.md).
