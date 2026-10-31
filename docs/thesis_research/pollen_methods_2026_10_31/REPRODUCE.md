# Quellen- und Schutzprüfung wiederholen

Basis: `f7e5290081dedb775a5cae7e210db88b71d9b1cb`. Kein Fit, Import oder
Countlauf erforderlich. Die Quellenprüfung ist Textlektüre, keine automatische
palynologische Validierung. [Quellenregister](./SOURCES.md) enthält URLs und
Fundstellen; nur dort genannte Abschnitte erneut lesen. Insbesondere W96 nicht
als geprüft behandeln, bis ein lesbarer Methoden-/Referenztext vorliegt.
Keine Volltextbytes dieser Literatur als archiviert behauptet; eigene Notizen
und Locators sind in Git gesichert. Online-Versionen können sich ändern.

## Git/PR und unveränderte Schutzbereiche

Im Repository ausführen:

```bash
git status --short --branch
git worktree list --porcelain
git branch -avv
git ls-remote --heads origin
gh pr view 36 --json state,mergedAt,baseRefName,headRefName,headRefOid,mergeStateStatus
git diff --check f7e5290
git diff --exit-code f7e5290 -- . \
  ':(exclude)docs/thesis_research/HANDOFF.md' \
  ':(exclude)docs/thesis_research/PROJECT_STATUS.md' \
  ':(exclude)docs/thesis_research/pollen_methods_2026_10_31/**'
```

Netzwerkbefehle benötigen Netzwerkzugang der Ausführungsumgebung. Ein
Sandbox-Socketfehler ist kein Beleg für Remotezustand oder HTTP-Zugang.
Neue unversionierte Dateien separat mit `git ls-files --others --exclude-standard`
prüfen. Kein Reset, Clean, Stash oder Merge zum Erreichen eines sauberen Status.

## Belegregister nachprüfen

[verification.json](./verification.json) speichert SHA-256 der unveränderten
Eingänge aus #35 und der gesicherten Tabellen. Hashes der Eingänge zusätzlich
gegen deren Bytes in `git show f7e5290:<Pfad>` vergleichen, nicht neu festlegen,
um abweichende Dateien zu akzeptieren.

- Sieben Modell-IDs/-namen mit config.json und species_name_links.csv abgleichen.
- Fossilfeld und Metadatenzeile mit monticchio_definitions.csv vergleichen;
  Original-/akzeptierte IDs mit species_name_links.csv. Diese Prüfung bestätigt
  die Übernahme der Metadaten, keine botanische Artbestimmung.
- Alle literature_source_ids aus evidence.csv müssen in sources.json/SOURCES.md
  vorkommen; jede Fundstelle am benannten Abschnitt/Objekt wiederfinden.
- Alle sieben fossil_identification_evidence_class bleiben ein fehlender
  datensatzbezogener Beleg; beide Freigabe-/Synonymiefelder bleiben false.
- W96: methods_read false und keine bestätigten Bestimmungsreferenzen.
  Zugriffslücke nicht mit „keine Referenzen in W96“ gleichsetzen.
- Lokale Markdown-Links auf existierende Dateien prüfen. Neue
  Literaturevidenz gegen die eingetragenen Lesebeschränkungen gegenlesen.

Durchgeführte Kontrolle durch denselben Bearbeiter; kein zweiter unabhängiger
Review. Repository-/Modelltests nicht ausgeführt, da ausschließlich Dokumentation
und Belegregister geändert wurden. Keine Literaturbilder diagnostisch beurteilt.
