# Netzel: Herkunft der EMPD2-Klimawerte recherchiert

## Ergebnis und korrigierter Arbeitsauftrag, 01.11.2026

Der Nutzer hat klargestellt, dass er kein bestimmtes Herkunftsdokument kennt
und Codex selbst im Internet danach suchen soll. Die bisherige Startbedingung
„vom Nutzer benanntes, bereits bereitgestelltes Belegobjekt“ und das zugehörige
Rechercheverbot sind für dieses neue Recherchepaket aufgehoben.
Keine erneute Bestätigung für dieses eindeutig beauftragte Paket verlangt.

[A4_E_PROVENANCE_EVIDENCE_REVIEW](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md)
ordnet konkrete Primärquellen R01–R03 zu: zusätzlicher Periodenbezug
1970–2000 bei Davis 2024, veröffentlichte PostgreSQL-Feldabbildung und
historische WorldClim2.0-/spätere 2.1-Dokumentation der Bibliothek.
**Recherche dokumentarisch abgeschlossen; die konkrete Raster-/Versions-/
Perioden-/Ausführungskette des PANGAEA-SQLite-Archivs bleibt teilweise
unbelegt.** Keine angewandte Unterversion oder Archivperiode ergänzt,
bestehende Fachdateien und `unverified` unverändert.

Basis `76edec21a7f4caa13eb64c054f67033718d479c7`,
[PR #60](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/60)
offen/ungemergt, mergebar; dessen Head-CI
[41284372025](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41284372025)
erfolgreich. Nach Inhaltscommit `1244c84` dort nur Sicherungsverweise.
main `7f1826c`; fünf alte Worktrees sauber erhalten.
Git-Fetch wegen nicht erreichbarem Cloud-Proxy fehlgeschlagen.
GitHub-Anbindung liefert Remote-Metadaten und Dokumente.
Eigener Worktree `/workspace/netzel-provenance-source-search`, Branch
`codex/netzel-provenance-source-search`; separate lokale Git-Objektablage
mit exakt hashgeprüftem Basiscommit/allen Bäumen und sparsamer Dokumentausgabe.
Übrige Dateiobjekte nicht heruntergeladen; keine Datenprüfung.
Inhaltscommit `4eb90f9e2dd6fe4659dc8f04436b541aed5a415c` über
[PR #63](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/63)
remote gesichert, offen/ungemergt und mergebar, gegen #60; kein Merge.
Automatische [CI 41286663993](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41286663993)
an diesem Inhaltscommit bei Abfrage laufend. Danach nur Sicherungsverweise;
aktuellen PR-Head/CI prüfen.

Nur Ergebnisdokument und kurze HANDOFF-/Statuspflege.
Quellen-/Versions-/Feldbezüge, sieben lokale Links/Anker und Diff geprüft;
753 geschützte Blobs/Dateimodi einschließlich Symlink und alte Statushistorie
bytegleich erhalten.
Keine manuellen Projekt-/Modelltests oder Rechnungen; automatische CI
gesondert. A1/A2 dokumentarisch abgeschlossen, A3 nur endlich-synthetisch;
A4-Kalibrierung/A5/A6 nicht begonnen. Reale Gates bleiben offen.

## Genau ein Folgeauftrag: historischen Erzeugungs-/Exportstand belegen – vorbereitet, wartet auf Nutzer-OK

**In einfachen Worten:** herausfinden, ob die Herausgeber öffentlich
dokumentiert haben, mit welchem damaligen Programm-/Klimadatenstand der
bei PANGAEA veröffentlichte Datenbankexport erzeugt wurde. Codex sucht selbst;
der Nutzer muss kein Dokument und keinen technischen Pfad liefern.
Ein „weiter“ im passenden Kontext beauftragt genau dieses Paket.

**Ziel:** die fehlenden Verbindungen aus R01–R03 an ihrer konkreten Lücke
prüfen: Bibliothek/Rasterstand/Auflösung → dokumentierte Ausführung →
PANGAEA.909130 bzw. abgegrenzter Release/Export → benannte Klimafelder.
Die neuen Belege aus dem Review nicht als bereits geschlossene Archivkette
übernehmen.

**Eingänge und Leseplan:** zuerst Git-/PR-/Merge-/CI-/Branch-/Worktree-/
Änderungsstand prüfen, ungesicherte Arbeit erhalten; eigener Worktree/Branch/PR,
kein Merge. AGENTS, diese HANDOFF, nur neuesten PROJECT_STATUS-Abschnitt;
Review insbesondere S5/S7–S10 und R01–R03; unveränderte Anforderungen
R01–R03/Herkunftsregeln; MODEL_CONTRACT Eingänge/Stop; Plan §1, §2 nur A4, §4.
Danach nur gezielt öffentlich auffindbare historische Release-/Build-/
Exportdokumentation, relevante Commit-Metadaten und statische
Verarbeitungspassagen der Herausgeber: PANGAEA.909130-Metadaten,
EMPD2/EMPD-data v2.0 (`737b76e`, Zenodo.4066498),
EMPD2/EMPD-admin und die dort tatsächlich verknüpfte Build-/Abhängigkeitskette,
Chilipp/latlon-utils zu dem damit belegten historischen Stand.
Keine ungeprüfte Annahme, dass heutige master-Stände historische Exporte erklären.

**Erlaubter Umfang:** das bereits angelegte
`A4_E_PROVENANCE_EVIDENCE_REVIEW.md` um einen klar datierten Abschnitt zu
genau dieser historischen Bindung ergänzen, plus kurze HANDOFF-/Statuspflege.
Neue Fundstellen mit dauerhaftem Dokument-/Commit-/Objektbezug,
Originalaussage/Übernahme, Archiv-/Feldgeltung und Grenze dokumentieren.
R01–R03 getrennt belegt/teilweise belegt/unbelegt bewerten; keine reale
Version/Periode auswählen oder Verträge ausfüllen.

**Begrenzung und Stop:** Recherche auf die drei Verbindungsschritte
(1) historische Build-/Bibliotheks-/Rasterbindung,
(2) dokumentierte Anwendung/Export,
(3) Zuordnung PANGAEA versus GitHub-/Zenodo-Ausgabe begrenzen.
Nach Prüfung der dafür auffindbaren relevanten Herausgeberunterlagen
abschließen, auch bei Nichtbeleg. Keine allgemeinen WorldClim-Texte als
weitere Evidenzschleife. Fehlende veröffentlichte Ausführungs-/ID-Bindung
explizit ausweisen; kein Nutzer-Dokument als neue Startbedingung verlangen.
Wenn nur reale Dateiprüfung oder Kontakt weiterhelfen könnte, das als
Grenze benennen und hier stoppen.

Alle übrigen bestehenden Fachdateien einschließlich Anforderungen,
MODEL_CONTRACT, Designs, Modelle, Resultate, Register, Importer, Flags,
Counts/Gruppen, Baseline und CLI/App bleiben unverändert.
Keine Rohdaten, Raster, Datenbank-Dumps, SQLite-/TSV-Zeilen oder ZIP-Inhalte
öffnen/beschaffen/prüfen; keine reale Daten-/Modellwahl, Joins, Reparatur,
Implementierung, manuelle Tests, Rechnungen, Fits oder Simulationen.
R04–R06 und alle übrigen Gates außerhalb dieses Pakets; A4 nicht umsetzen,
A5/A6 nicht beginnen. Keine Freigabe, Unteragenten, Autoren-/andere Nachrichten,
Beleganforderungsversand oder Merges.

**Abnahme:** konkrete historische Herkunftsverbindung oder begrenzter
Nichtbeleg mit genauer Restlücke; keine Gesamtfreigabe.
Quellen-/Link-/Diff-/Blobprüfung, Statushistorie erhalten; Commit/eigener PR
ohne Merge. Genau einen anschließenden sinnvollen Auftrag vorbereiten,
dessen Eingänge Codex selbst beschaffen kann, soweit öffentlich verfügbar.
Keine Dokumentationsschleife allein aufgrund fehlender Nutzerdatei.

## Kopierfertiger Folgeprompt

```text
Setze Netzel auf codex/netzel-provenance-source-search fort.
Inhaltscommit: 4eb90f9e2dd6fe4659dc8f04436b541aed5a415c
PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/63
Danach nur Sicherungsverweise; aktuellen PR-Head/CI prüfen.
Aktuellen PR-Head/CI prüfen; kein Merge.
Prüfe den in HANDOFF vorbereiteten historischen Erzeugungs-/Exportnachweis:
Mit welchem damaligen Programm-, Raster- und Auflösungsstand wurden die
EMPD2-Klimafelder des PANGAEA-Archivs 909130 erzeugt, und ist die Verbindung
zum GitHub-/Zenodo-Release v2.0 öffentlich dokumentiert?
Suche die passenden historischen Herausgeberunterlagen selbst im Internet.
Ich muss kein Dokument benennen oder bereitstellen.
Leseplan, genau begrenzten Umfang, Abnahme und Stop aus HANDOFF einhalten.
Nur A4_E_PROVENANCE_EVIDENCE_REVIEW.md ergänzen und HANDOFF/Status kurz pflegen.
Keine Rohdaten-/Datenbank-/Raster-/ZIP-Prüfung, Auswahl, Implementierung,
manuellen Tests oder Rechnungen; bestehende übrige Fachdateien unverändert.
Keine A4-Umsetzung, A5/A6, Gatefreigabe, Unteragenten, Nachrichten oder Merges.
Bei fehlendem öffentlichen Nachweis die konkrete Lücke festhalten und stoppen.
Ergebnis prüfen, per Commit/eigenem PR sichern, genau einen Folgeauftrag
vorbereiten und den Folgeprompt automatisch mitliefern.
Kurz und verständlich auf Deutsch berichten. Für dieses Paket kein erneutes OK.
```
