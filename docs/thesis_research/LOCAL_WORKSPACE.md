# Lokaler Einstieg: Dissertation und Rekonstruktion

> **Aktueller Arbeitsplan, 02.11.2026:** Zuerst lokale Dissertationsfälle
> nachbauen, danach verbessern: [Reihenfolge und nächster Einstieg](LOCAL_DISS_RECONSTRUCTION_PLAN.md).
> Zunächst eine quellengebundene Fallmatrix erstellen. Der unten dokumentierte
> Papervergleich bleibt technische Referenz und ist keine vollständige
> taxonbasierte Dissertationsreproduktion. Keine vorhandenen Läufe wiederholen.


Stand: 01.11.2026. Dies ist die technische Bestandsaufnahme des neu eingebundenen
Originalbestands. Wissenschaftlicher Gesamtstand weiterhin in
[PROJECT_STATUS.md](PROJECT_STATUS.md) und [HANDOFF.md](HANDOFF.md).

## Fortsetzung nach Schritt 1

Auftrag zur selbstständigen Fortsetzung am 01.11.2026. Die damalige
Compilerblockade ist **behoben**: offizielles LLVM-MinGW-Archiv 20261020 lokal
heruntergeladen, veröffentlichter SHA256 geprüft und außerhalb des Originalbestands
entpackt. Keine Systeminstallation. `tools/build_windows.py` baut den unveränderten
Kern mit C++17, `-O3`, `-ffp-contract=off` und dem Windows-Schalter
`_USE_MATH_DEFINES`. Build und Binärhash stehen lokal in `.local/build-windows.json`.

Quellenabgleich für den ausgewählten Fall dokumentiert in
[LOCAL_SOURCE_CROSSWALK.md](LOCAL_SOURCE_CROSSWALK.md). Neue relevante Befunde:
gewichtete Summen im lokalen R-Quellstand; unterschiedliche Dissertationsfassungen;
eine auch im Archiv leere Vegetationstabelle; lokale Klimaraster CRU TS4.00 mit
anderem Bezugszeitraum. Keine Quellen substituiert.

Original und Paket-Legacy stimmen nach separatem Build unter derselben Laufzeit
im 20.000-Iterationen-Test in allen sechs Ausgabefeldern exakt überein. Der
archivierte plattformübergreifende Bitgleichheitstest scheitert weiterhin:
8.141 von 20.000 Akzeptanzentscheidungen verschieden, erster Unterschied Index 3.
Die aktive libc++ verwendet einen anderen `default_random_engine` als libstdc++.
Referenz und Test wurden nicht verändert oder abgeschwächt.

Diagnostischer Lauf abgeschlossen: ca. 30,9 s inklusive Bericht und Abbildungen,
maximal ca. 280 MB Working Set; erwartet unzureichende Konvergenz der kurzen Kette.
Danach vollständiger Originallauf abgeschlossen: eine Million Iterationen,
62,4 s Sampling, ca. 126,2 s insgesamt, maximal ca. 446 MB Working Set.
Ausgaben unter `results/local_original_full/`; wissenschaftlich vorläufig.
Die Split-Diagnostik einer Originalkette ersetzt keine unabhängigen Startketten.

**Abgeschlossen:** korrigierte vorhandene v2-Variante mit vier Ketten à zwei Millionen
Iterationen (`results/local_corrected_full/`), vollständige Wiederholungen beider
Posterior-Sampler, unabhängige Nachrechnung aller 48 geprüften CSV-Felder und
362 Alters-/Klimadiagnosen. Konvergenzkriterien erfüllt; alle 43 korrigierten
Parameter R̂ ≤1,000844 und Bulk-ESS ≥8552,6. Kein wissenschaftlicher Kern geändert.
Prüfdesign vor Vollergebnissen lokal gespeichert. Zusatzprüfung des Samplers:
sechs Tests bestanden, ein kurzer Prior-Test der optionalen Dirichlet-Variante
fehlgeschlagen; vierfache Längendiagnose besteht ohne Toleranzänderung.
Archivierter Bitvergleich weiterhin fehlgeschlagen und dokumentiert.

Ergebnis und fachliche Grenzen: [LOCAL_REFERENCE_RESULT.md](LOCAL_REFERENCE_RESULT.md).
Vollständiger lesbarer Nachweis unter `results/local_reference_review_v2/review.html`.
Der erste Leser-Versuch im Nachbarverzeichnis ist als unvollständig markiert.
Nächster Schritt: fachliche Bewertung des konkreten vorläufigen Beispiels;
keine weiteren Fossilfälle oder automatische Pilotfreigabe.

Werkzeuge: `tools/run_local_reference.py` verweigert bestehende Zielverzeichnisse
und speichert Konfiguration, Seeds, Eingabe-/Quell-/Binärhashes, Build, Umgebung,
Status, Laufzeit, Speicherbedarf und Ausgabehashes. `tools/verify_local_legacy.py`
führt den lokalen Originalvergleich reproduzierbar aus.

## Schritt 1: Projektgrundlage

- Tatsächlich ausgeführte lokale Windows-Prozesse: Windows 10.0.26200,
  Git 2.56.0.windows.1, Python 3.12.10. Kein lediglich aus einem Pfad abgeleiteter
  PC-Zugriff.
- Originalquelle: 1.783 Dateien, 2.576.569.660 Bytes (2,58 GB dezimal).
  Eingabebereich: 72 Dateien / 1.412.412.056 Bytes; Ausgabebereich:
  1.601 Dateien / 1.150.091.668 Bytes; Quellcodebereich: 93 Dateien /
  3.122.140 Bytes; Wurzel: 17 Dateien / 10.943.796 Bytes.
- Wesentliche Formate: 95 R-Skripte, 13 RData-Dateien, 17 CSV-Dateien,
  zwei NetCDF-Dateien, 1.496 PDFs, neun ZIP-Archive und sechs LaTeX-Dateien.
  Diese Zahlen sind ein Metadateninventar, keine fachliche Datenprüfung.
- Quelle bleibt unverändert und ist kein Git-Repository. Keine Initialisierung,
  Verschiebung, Umbenennung oder pauschale Datenkopie. ZIPs zunächst nur über
  das Inhaltsverzeichnis geprüft; vier gezielt ausgewählte LaTeX-Dateien mit
  zusammen 207.976 Bytes in den ausgeschlossenen Arbeitsbereich übernommen.
- Separater neuer Checkout des bekannten App-/Code-Repositories. Keine an
  diesen Chat angehängten vorhandenen Worktrees gefunden. Flacher Clone mit
  Tiefe 1; historische Vorfahren sind deshalb lokal nicht vollständig vorhanden.
- GitHub-Anbindung und lesender Git-Abruf bestätigen `main` auf
  `7f1826cffddc8893ba6354cd3cca3fa43954b9be`.
  [PR #30](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/30)
  ist gemergt. Neuer lokaler Branch: `dissertation-reconstruction`.
  Vor Branchanlage war der Checkout sauber.
- `AGENTS.md` sowie aktueller Forschungsstand und Übergaben gelesen.
  Bestehende wissenschaftliche Baseline und CLI/App-Integration bleiben erhalten.
  Die Zuständigkeit aus dem Repository bleibt berücksichtigt; die App wird hier
  nicht neu entwickelt.
- Projektumgebung `.venv/` eingerichtet. Python-Abhängigkeiten separat installiert;
  genaue Versionen lokal in `.local/requirements-lock.txt`. `pip check` erfolgreich.

## Lokale Dateien und Schutz vor versehentlicher Versionierung

Persönliche Pfade und Originaldateinamen stehen ausschließlich in `.local/`.
Die Pfadkonfiguration wird nicht automatisch vom Sampler verwendet: Sie dient
der Quellinventarisierung. Der reproduzierbare Paketfall verwendet weiter
`data/in`, damit bestehende Lauf-Fingerprints unverändert bleiben.

```powershell
# Beispiel kopieren, beide Pfade lokal eintragen; kein Originalbestand im Repo.
Copy-Item docs/thesis_research/local_workspace.example.json .local/paths.json
python tools/local_inventory.py --config .local/paths.json
```

`inventory_dir` ist relativ zum aktuellen Arbeitsverzeichnis oder absolut.
Ein tatsächlicher absoluter Quellenpfad gehört nur in `.local/paths.json`.
Der Katalog enthält relative Quellpfade, Dateityp, Bytezahl, Änderungszeit und
ausdrücklich ungeprüfte Rollenvorschläge. Er liest Metadaten und ZIP-Verzeichnisse,
führt keine Quellskripte aus und kopiert keine Rohdaten.

Mit `git check-ignore` geprüft: `.local/paths.json`, Katalog, `.venv/`,
`_local_data/`, Laufverzeichnisse, Build-Artefakte und Windows-Erweiterung `.pyd`.
Bestehende Ignore-Regeln bleiben bestehen.

**Bereits versioniert:** Das Repository enthält schon ausgewählte Paket-Eingaben,
Paper-/Dissertationsausschnitte und Referenzergebnisse (25 getrackte Einträge
im geprüften Teilbestand einschließlich Platzhaltern und Symlink).
Diese wurden nicht neu hochgeladen, aus dem Index entfernt oder aus der Historie
gelöscht. Ein Ignore-Muster würde solche Einträge nicht schützen.
Der neu erschlossene Originalbestand liegt vollständig außerhalb dieses Checkouts.

## Schritt 2: Begonnener Quellenabgleich

- Eine vollständige lokale Dissertationsfassung (176 PDF-Seiten, Titelblatt
  September 2022) und ein passendes LaTeX-Archiv gefunden. Mehrere ältere
  Fassungen und eine PDF mit Vorlagenplatzhaltern sind ebenfalls vorhanden;
  Dateiname oder Änderungsdatum allein gilt nicht als Versionsnachweis.
- Die Dissertationsteile im Repository und der vorhandene Abgleich bleiben
  Quellen. Ein endgültiger Versionsabgleich ist noch offen. Lokale PDF-Auszüge,
  ausgewählte Archivmitglieder, Quellpositionen und elf gezielte SHA256-Prüfsummen
  sind unter `.local/` dokumentiert; kein vollständiges Hashing des Gesamtbestands.
- Vorläufiger **technischer Reproduktionsfall**: vorhandener Kinneret-Paperfall
  aus `data/in`, `configs/original.toml`, `tests/data/legacy_short_chain.npz`
  und `reference/v1.0-paper/`. Eingaben: 160 Probentiefen, 33 Taxa, drei Biome,
  181 Altersstufen von -50 bis 8.950 cal BP in 50-Jahres-Abständen.
  Paket-Eingabevalidierung bestanden; alle numerischen Eingaben endlich,
  Alters-Tiefen-Zeilen auf Rundungsniveau auf eins normiert.
- Das ist das vorhandene **Biommodell mit Wintertemperatur**, keine Reproduktion
  der taxonbasierten Multiproxy-Methode mit Jahresmitteltemperatur aus der
  Dissertation. Es wird nicht als fertige Hauptmethode oder fachliche Freigabe
  ausgegeben.
- Drei kleine RData-Quellen sowie die beiden lokalen NetCDF-Dateien lesend
  geöffnet. Große RData-Container noch nicht vollständig geladen. Lokale
  NetCDF-Metadaten und Variablendimensionen gespeichert; Jahres-/Monatssummen
  werden nicht allein aus Dateinamen abgeleitet.
- Begrenzte Analyse expliziter erster Dateiparameter zweier R-Einstiegsskripte:
  53 Vorkommen, davon 28 innerhalb der Quelle vorhanden, acht dort fehlend,
  17 außerhalb des ausgewählten Quellbereichs. Bedingte Zweige und dynamische
  Pfade sind damit nicht vollständig aufgelöst. Kein außerhalb liegender
  Datenordner durchsucht und kein R-Skript ausgeführt.

## Windows-Reproduzierbarkeit

Der erste unabhängige Python-Testlauf meldete 14 bestandene Tests und einen
Prüfsummenfehler der bestehenden Analog-Baseline. Nachgewiesene Ursache:
systemweit aktiviertes `core.autocrlf=true` erzeugte beim Klonen CRLF-Bytes.
Der gespeicherte Git-Blob hat genau den im Referenzmanifest erwarteten SHA256.
Keine wissenschaftliche Referenz war dafür anzupassen.

Neu: `.gitattributes` hält Text-Checkouts auf LF. Im frischen lokalen Checkout
wurden ausschließlich Dateien auf LF gebracht, deren so normalisierte Bytes
exakt dem bereits gespeicherten Git-Objekt entsprachen. Abweichende Bearbeitungen
wurden übersprungen. Das Änderungsprotokoll liegt lokal; keine globalen
Git-Einstellungen wurden geändert.

Nach der belegten Checkout-Korrektur: **15 von 15 ausgewählten bestehenden
Python-Tests bestanden** (5,48 s). Getestet wurden `test_smote_network.py`,
`test_taxon_response.py` und `test_thesis_baseline.py`, mit separatem
`--basetemp` unter `.local/`. Das sind mathematische/technische Prüfungen mit
synthetischen Fixtures und Referenzdateien, keine neu trainierten Modelle und
keine fachliche Validierung einer Fossilrekonstruktion. Der C++-Teil und die
Browser-App wurden in dieser Sitzung nicht erfolgreich ausgeführt.
Die vollständigen Testprotokolle liegen lokal.

## Historischer Buildbefund am Ende von Schritt 1

Die folgenden Absätze dokumentieren den inzwischen überholten Zwischenstand.
Aktueller Stand und abgeschlossene Rechenprüfungen stehen oben.

Der tatsächlich ausgeführte Befehl
`.venv/Scripts/python.exe setup.py build_ext --inplace` endet mit Exitcode 1:
`Microsoft Visual C++ 14.0 or greater is required`.
Kein passender Compiler über PATH oder die begrenzt geprüften Standardorte
gefunden. Laut Nutzer kann auf diesem PC noch keiner installiert sein; der
Quellordner stammt von einem anderen PC. Keine Systeminstallation durchgeführt.
Buildprotokoll: `.local/build.log`. Noch kein neuer Samplerlauf.

1. Verfügbaren C++17-Compiler bereitstellen bzw. seinen konkreten Pfad feststellen,
   Compileroptionen des vorhandenen Builds unter Windows prüfen und lokal bauen.
2. Vorhandene kurze Legacy-Äquivalenztests gegen die gesperrte Referenz ausführen;
   Versionen, tatsächliche Befehle, Speicherbedarf und Laufzeiten protokollieren.
3. Erst nach diesem diagnostischen Lauf den vollständigen vorhandenen
   Kinneret-Referenzfall reproduzieren. Ergebnisse lokal unter `results/`, keine
   großen Standard- oder Batchläufe vor der Aufwandsprüfung.
4. Parallel den neuen lokalen Originalbestand weiter mit Dissertation und
   bestehendem Quellenregister abgleichen. Fehlende externe R-Eingaben gezielt
   benennen; vorhandene NetCDF-Dateien nicht stillschweigend substituieren.
5. Für die taxonbasierte Hauptmethode gelten weiterhin die dokumentierten offenen
   Punkte: Typ-/Taxonzuordnung, Wahrscheinlichkeitskalibrierung, Unsicherheit
   und insbesondere der gesperrte Monticchio-Chronologieanschluss.
   Keine automatische Pilotfreigabe, keine zusätzlichen Fossilrekonstruktionen.

Kein Push, PR, Merge, Kontakt zu Dritten oder Upload lokaler Forschungsdaten.
Keine neuen wissenschaftlichen Parameter oder fachlichen Toleranzen festgelegt.
