# Claude: konkreter nächster Phase-1-Arbeitsauftrag

Stand 02.11.2026. Dieser Auftrag ist der aktuelle Einstieg; ältere „als Nächstes“-
Abschnitte in den Verlaufsberichten sind nachgeordnet. Nutzerauftrag:
Dissertation zuerst historisch nachvollziehen, erst später verbessern.
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten.

## Arbeitsbasis zuerst prüfen

Repository `TimonNetzel/kinneret_reconstruction_python`, Branch
`dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Der CDF-/EMD-Teil baut auf `d6e3343decbb1e74409a6c10e179da4e9da7be9f` auf und
ist mit diesem Startauftrag im selben Branch gesichert. Den tatsächlichen
aktuellen PR-Head vor Arbeit prüfen; nicht `main` als bereits aktualisiert ansehen.
Den Sicherungscommit dieses Pakets findet `git log -1 -- tools/historical_cdf_discrepancy.py`.

Ausschließlich den vom Nutzer benannten vorhandenen Checkout verwenden.
Keinen Klon oder Worktree anlegen. Vor Beginn `git status --short --branch`,
`git rev-parse HEAD`, `git log -5 --oneline` und den tatsächlichen GitHub-PR-
Stand prüfen. Lokale Änderungen erhalten; keine pauschalen Resets oder Updates.

Zuerst lesen:

1. Rootdateien `CLAUDE.md` und `AGENTS.md`.
2. Neuesten Abschnitt von [PROJECT_STATUS.md](PROJECT_STATUS.md) und
   [verbindlichen Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md).
3. [Aktuelles CDF-/EMD-Ergebnis](literal_cdf_discrepancy_2026_11_02/README.md)
   samt `source_binding.json`, `synthetic_fixture.json`, `synthetic_result.json`
   und `verification.json`.
4. Die vier Module und ihre unten zugeordneten Tests. Die Quellenpakete
   enthalten die benötigten R-Auszüge; keine erneute Gesamtinventur.

Originalquellen ausschließlich lesen; Pfade ausschließlich über die private
`.local/my_reconst_paths.json`. Fehlender Zugriff auf Originalordner ist kein
Anlass für Ersatzdaten, neue Laufdaten oder Speicherortfragen. Die bisherigen
Nachweise und synthetischen Tests sind portabel. Keine absoluten privaten
Quellpfade oder Originaldaten nach GitHub exportieren.

## Was jetzt geprüft vorliegt

| Modul | Geprüfter Ausschnitt | Tests |
| --- | --- | --- |
| [historical_age_depth.py](../../tools/historical_age_depth.py) | Explizite Listenposition; Altersduplikate einmal +0,5; Tiefenauswahl/Normierung; 1e-9 nach Normierung ohne Renormierung; Pollennullen 1e-6; Alter × Tiefe mal Tiefe × Taxon | [43 Fälle](../../tests/test_historical_age_depth.py) |
| [historical_age_selection.py](../../tools/historical_age_selection.py) | Exakte Periodenraster aus erster Altersdifferenz; vorläufige und bei Erreichen der Zeilen wirksame Maske; boolesche RW-Zeilenauswahl | [39 Fälle](../../tests/test_historical_age_selection.py) |
| [historical_final_pdf.py](../../tools/historical_final_pdf.py) | Taxonmaximum, RW-/Gibbs-Koeffizienten, TF-Produkt, erste exakte Energieposition, Rohdichtesumme dann Normierung | [22 Fälle](../../tests/test_historical_final_pdf.py) |
| [historical_cdf_discrepancy.py](../../tools/historical_cdf_discrepancy.py) | Einseitiger Platzhaltertest ersetzt beide Referenzen; kumulierte Rohdichten; `sum(abs(C−R))/dims`; keine Zellbreiten oder vorgezogene Normierung | [43 Fälle](../../tests/test_historical_cdf_discrepancy.py) |

Mit den [20 Belegvertragstests](../../tests/test_kinneret_phase1_spec.py) sind
**167 Tests bestanden**. Geprüft sind Formeln im dokumentierten endlichen
Eingabebereich, keine historische R-/Paketlaufzeitgleichheit. Kein Originalwert
wurde in den synthetischen Numerikprüfungen als Eingabe verwendet.

Die Fallstudienaufrufe übergeben `ecdf_ref1=1,ecdf_ref2=1`; damit werden beide
Referenzen zu Nullmatrizen. `dims` für deren Anlage wird vor der lokalen
Zuweisung benutzt. Der erhaltene Haupteinstieg setzt zwar 300, aber die finale
Aufrufumgebung/TF-Breite ist nicht gebunden. Beim Test explizit zwischen
`inherited_dims` und späterem `dims` unterscheiden.

`age`, `site.age` und freie Masken-/Schaltvariablen bleiben getrennt. Die
vorläufige Holozänmaske wird im erhaltenen Vorbereitungsablauf bei Erreichen
der Zeilen auf alle Alter zurückgesetzt; `taxa_change` bleibt wegen `& FALSE`
aus. Das ist kein Beleg einer konkreten historischen Ausführung.

## Nächster konkreter Auftrag: gemeinsame synthetische Rechenkette

**Ziel:** Die vier getrennten Module an einem gemeinsamen künstlichen Beispiel
auf ihre Schnittstellen und die belegte Reihenfolge prüfen. Erstelle
`tests/test_historical_phase1_chain.py` und ein neues Belegpaket
`docs/thesis_research/literal_phase1_chain_2026_11_02/` mit README, gemeinsamer
Fixture, Ergebnissen und Prüfnachweis. Verwende die vorhandenen reinen
Funktionen; keine neue produktive Runner-/App-/CLI-Schnittstelle.

Ein geeigneter, ausdrücklich künstlicher Aufbau ist:

- Eine Liste mit mehreren kleinen Altersrastern; ausgewählte Matrix **4 × 5**,
  drei eindeutige bereits ausgerichtete Probentiefen und Pollen **3 × 2**.
  Ein Raster explizit per einsbasierter Position wählen; Matrix/Flachvektor
  bleibt als Negativfall ausgeschlossen. Keine reale Chronologie auswählen.
- Vier Alterslabels beispielsweise vor der einmaligen Verschiebung
  `[-50,0,0,11500]`. `site.age` als getrennten Zustand erhalten und die
  Periodenmasken aus dem danach verschobenen `age` ableiten.
- Künstlichen `taxa_merge`-Kontrollzustand explizit nennen. Den erhaltenen
  All-TRUE-Endzustand nur unter der Bedingung abbilden, dass die relevanten
  Vorbereitungszeilen erreicht sind; kein reales Taxonmatching ausführen.
- Tiefenkern **4 × 3**, Pollen auf Alter **4 × 2**, RW-Koeffizienten **4 × 2**,
  bereits vorbereitete synthetische TFs **2 × 4** und Rohdichten **4 × 4**.
  Taxon- und Zellreihenfolge explizit festhalten. Die TF-Eingaben und deren
  Massen sind ein synthetischer Vertrag, keine wiedergewonnenen Original-TFs.
- Zwei oder drei vorgegebene RW-Gewichtsvektoren, kein Zufall. Energiebeispiel
  `[4,4,9]` mit erhaltenen einsbasierten Originalpositionen `[1,3]` für die
  bereits geprüfte exakte Duplikatregel. In R werden diese IDs vor der aktiven
  Ziehungsschleife bestimmt (375/428); CDF-Scores nur den verbliebenen Original-
  IDs zuordnen. Ein vollständig vorgegebener Dichtetensor im Test ist keine
  Behauptung, die verworfene Ziehung sei historisch berechnet worden. Die
  Energie wird kein Mischungsgewicht.
- Pro Ziehung die rohe kumulative Abweichung vor der späteren PDF-Summe/
  Normierung prüfen. Platzhalter mit ausdrücklich passenden geerbten und
  späteren Breiten einsetzen; keine implizite Referenzerzeugung.

Die Quellreihenfolge muss sichtbar bleiben: Chronologieliste und Alterslabels →
Probentiefenkern mit historischer Normierung/Nullersetzung → Pollen-Nullersetzung
und Altersprodukt → Maximumskalierung und RW-Gewichtung → logische Zeilenauswahl
und TF-Produkt → rohe kumulative Abweichung je verbliebener Ziehung →
Rohdichtesumme der zuvor bestimmten Original-IDs → Normierung der endgültigen
synthetischen PDF-Zeilen. Die exakten Energieduplikate werden wie oben beschrieben
vor der aktiven Ziehungsschleife bestimmt; Original-IDs nicht neu durchnummerieren.
Periodenmasken sind dabei ein separater Kontrollzustand; sie ändern nicht
rückwirkend die Chronologiematrix oder die Alterslabels.

Alle Sollwerte unabhängig aus kleinen rationalen Zahlen herleiten, etwa mit
handgeschriebenen Brüchen und `fractions.Fraction`. Keine erwarteten Werte
allein aus denselben zu testenden Modulaufrufen gewinnen. Zwischenstände und
Dimensionen im Nachweis festhalten. Das Beispiel verwendet ausschließlich
synthetische Eingaben; es wird keine historische Fossilkurve ausgegeben.

## Abnahmekriterien und typische Verwechslungen

- Gemeinsame Fixture, eindeutige Alters-/Tiefen-/Taxon-/Zellachsen und explizite
  Normierungszeitpunkte. Keine Achsenausrichtung allein aus gleicher Länge
  behaupten. TF-Einheiten bleiben abstrakt; keine echte Temperatur-/PANN-Skala.
- Positive Kontrolle mit rationalen Sollwerten für Zwischenstufen und Ausgabe.
  1e-9 im Tiefenkern, 1e-6 in Pollen und Reihenfolge der Schritte müssen wirksam
  sein. Eine verfrühte Normierung muss durch ein gezieltes Gegenbeispiel auffallen.
- Negative Kontrollen für vertauschte/inkonsistente Dimensionen, Matrix statt
  Listenchronologie, falsch lange Maske und widersprüchliche Platzhalterbreite.
  Bereits geprüfte Fehlerpfade nur dort wiederholen, wo die Verbindung selbst
  eine neue Verwechslung ermöglichen würde.
- Roh-CDF-Abweichung und normierte PDF-Mischung bleiben verschiedene Ergebnisse.
  Keine Umdeutung des Nullreferenzscores als Abstand zu einer echten älteren
  Referenz-CDF oder als physikalische °C-/mm-Distanz.
- Vorpakete und ihre Hashbindungen unverändert. App/CLI, Baseline, Referenzen
  und Originale erhalten. Der Phase-1-Prüfer bleibt ohne fachliche Freigabe.
- Neue Tests sowie die bisherigen 167 Fälle prüfen. README/Status/Übergabe um
  Ergebnis, genauen Testumfang, Grenzen und nächsten belegten Einstieg ergänzen.
  Geprüftes Paket auf dem bestehenden Branch committen und pushen, PR #77
  aktualisieren, keinen Merge ausführen.

Eine gemeinsame synthetische Kontrolle ist kein vollständiger Referenzlauf und
schließt keine der fehlenden Originaleingaben. Falls die Verbindung einen
Quellkonflikt zeigt, diesen mit Zeilen und Beispiel dokumentieren; nicht still
reparieren oder eine neue fachliche Variante auswählen.

## Prüfkommandos und erwarteter Ausgang

Im vorhandenen Windows-Checkout:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_historical_cdf_discrepancy.py tests/test_historical_age_selection.py tests/test_historical_age_depth.py tests/test_historical_final_pdf.py tests/test_kinneret_phase1_spec.py -q -o cache_dir=.local/phase1_spec_v1/pytest_cache
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py
.venv/Scripts/python.exe tools/check_kinneret_phase1_spec.py --with-local
```

Erstes Kommando vor neuen Änderungen: 167 Tests bestanden. Die Prüfer liefern
bewusst **Exitcode 2**: Belege intakt, Voraussetzungen offen. Erwartet werden
26 Belegdateien, 19 Befunde, 50 Snapshotdateien, elf offene Nachweisbündel und
`selected_variant=null`. Der lokale Aufruf benötigt die private Konfiguration
und prüft nur bereits dokumentierte Pfade; bisher 28 unverändert vorhanden,
zwölf fehlend. Ohne lokale Quellen den portablen Prüfer verwenden und die
lokale Prüfung ausdrücklich als nicht ausgeführt kennzeichnen.

Nach Ergänzung der gemeinsamen Prüfung das neue Testmodul zum ersten Kommando
hinzufügen; neuen Gesamtumfang tatsächlich zählen. Es gibt keinen Auftrag,
Trainings-/Sampler-/Bacon- oder Fossilläufe zum Testen zu starten.

## Unverändert offene fachliche Grenzen

Die finale Datei `core_data_additional_sites_v1.2.rdata`, erwartete Liste mehrerer
Altersraster, numerische historische TFs und tatsächliche Laufzuordnung fehlen.
Der Nutzer kennt keinen weiteren Speicherort/Erzeuger der Basisdatei. Nicht
nochmals fragen oder pauschal inventarisieren. Ältere Biom-/Sammelcontainer und
sieben moderne Artenmodelle ersetzen diese Belege nicht. Der Paper-Biomfall ist
keine vollständige taxonbasierte Dissertationsreproduktion.

Alle elf Nachweisbündel bleiben offen, keine finale Kinneret-Variante gewählt.
Keine neuen Altersmodelle, Trainings, Sampler oder Fossilrekonstruktionen,
keine stille Quellenkorrektur, Ersatzdatenwahl, Kalibrierung oder Phase-2-
Diagnostik. Monticchio-/Chronologie-Gates bleiben fallbezogen erhalten.
Plots jung links, alt rechts. Keine automatischen Merges.
