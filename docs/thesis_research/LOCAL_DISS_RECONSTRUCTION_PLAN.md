# Arbeitsplan: zuerst lokale Dissertation rekonstruieren, danach verbessern

Stand 02.11.2026. Vom Nutzer im Chat ausdrücklich neu priorisiert:
**Zuerst die lokalen, standortbezogenen Rekonstruktionen der Dissertation
nachbauen; anschließend Verbesserungen vornehmen.** Dieser Plan bestimmt die
Reihenfolge der weiteren Arbeit. Frühere Vorschläge für neue Kalibrierung,
Reliabilitätsansicht, modernisierte Multiproxy-Modelle oder zusätzliche Piloten
sind nachgeordnet. Vorhandene wissenschaftliche Befunde bleiben gültig.
Fortsetzung auf `dissertation-reconstruction` in
[PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77).
Geprüfter Ausgangsstand dieser Planänderung: `4b75ead`.

## Aktuell: geprüfte Rechenteile in der App, 02.11.2026

Die erste App-Anbindung ist abgeschlossen:
[App-Nachbaupaket](app_reproduction_2026_11_02/README.md). Die Forschungsansicht
zeigt Nachbaustatus, fünf geprüfte Teilmodule, elf offene Originaleingaben und
ein ausdrücklich synthetisches Rechenbeispiel, offline in beiden App-Formaten.
Das ist keine Fallreproduktion. Reihenfolge ab jetzt:

1. Nächsten begrenzten **deterministischen** Quellschritt (Kandidaten
   `weightedMedian`, Taxon-Pruning, Splines; Zufallsschritte nur dokumentieren)
   isolieren und gegen unabhängige Referenz prüfen.
2. Danach neuen datierten Nachbaustand in der App ergänzen (alter Stand bleibt
   eingefroren).
3. Fachlicher Meilenstein unverändert: vollständigen Eingabevertrag eines lokalen
   Originalfalls schließen, erst dann historischen Ablauf reproduzieren.

[Konkrete Claude-Übergabe](CLAUDE_NEXT_STEP_AFTER_APP_REPRODUCTION_2026_11_02.md).
Der folgende Abschnitt ist Verlauf.

## Aktueller Meilenstein: geprüfte Rechenteile zur App führen, 02.11.2026

Der Nutzer bekräftigt den vorrangigen Dissertationsnachbau **mit Implementierung
in die App**. Übergaben erfolgen pro abgeschlossenem geprüftem Paket als neue
konkrete Claude-Datei; alte Aufgaben-/Hashnachweise bleiben eingefroren.

[TF-Vorbereitung](literal_tf_preparation_2026_11_02/README.md) ist jetzt nachgebildet:
Zeilennormierung, danach explizite Klimazellmaske ohne Renormierung; 51 neue und
alle früheren 200 Teiltests grün. Eine neue Kunstkette verbindet fünf Module,
mit maximal 2,22e-16 Abweichung zur unabhängigen Bruchrechnung. Die Original-
Basis/Listenchronologie, numerischen historischen TFs und Laufbindung bleiben
offen; keine vollständige Fallreproduktion und alle elf Nachweisbündel offen.

[Nächster konkreter App-Auftrag für Claude](CLAUDE_NEXT_STEP_APP_REPRODUCTION_2026_11_02.md):
separaten aktuellen Nachbaustand und das ausdrücklich synthetische Beispiel an
die vorhandene Forschungsansicht anbinden, beide Offlineformate prüfen. Keine
historischen Eingaben ersetzen oder vorhandene Läufe als taxonbasierten Nachbau
umdeklarieren. Gesicherte Teile → geprüfte App-Anbindung; vollständiger lokaler
Originalfall mit Eingabevertrag und Ergebnisabgleich bleibt der fachliche
Meilenstein. Neue moderne Modelle/Verbesserungen erst nach Phase 1.


## Aktuell: zusätzlicher Quellenabgleich abgeschlossen, 02.11.2026

[climate_modelling-Belegpaket](climate_modelling_audit_2026_11_02/README.md):
1.828 Dateien, 161 R-Quellen und ein ZIP tatsächlich zusätzlich geprüft, privat
absolut gebunden. Neuer Collector erweitert den Suchumfang ausdrücklich;
der alte Zwei-Wurzeln-Scanner und seine Nachweise bleiben eingefroren.
main.R-Verbreitungsbasis bytegleich zur vorhandenen Kinneret-Kopie (275 Taxa).
F14/F15 sind strukturell passende CSV-Kandidaten ohne gesicherten SISAL-V2-Bezug.
Übrige 37 priorisierte Namen, acht benannte Posteriorvarianten und v4.0 ohne
neue Namenskopie. Zwei v1.4-Fassungen mit verschiedenem Hash nicht gleichsetzen;
sftlf ersetzt keine mask2-Eismaske, regionale CRU-Dateien keine globalen Anker.
Alle elf Voraussetzungen bleiben offen; keine finale Laufvariante gewählt.

23 Parser-/Lesertests bestanden, eingefrorene Belege/Startaufträge erhalten.
Keine Neuerschaffung der beiden CSVs planen, solange ihre Versionsprüfung offen
ist. Finale Basis/Listenchronologie, numerische TFs und übrige benötigte Lücken
nur anhand konkreter Erzeuger-/Eingabe-/Achsenverträge für Wiederaufbau planen.
[Neuer Übergabeauftrag](NEXT_STEP_AFTER_CLIMATE_AUDIT_2026_11_02.md): begrenzte
synthetische Prüfung der belegten TF-Vorbereitung als nächster unabhängiger
Phase-1-Schritt. Keine Trainings/Sampler/Altersmodelle/Fossilläufe oder Phase 2.
Frühere Ankündigungen „Ordner ungeprüft“ unten dokumentieren den Altstand.


## Aktuelle Quellenlage und erster Auftrag nach dem Claude-Wechsel

Der Datenträger hinter `/automount/...` ist gelöscht; **lokale Kopien können
weiterhin in bereitgestellten Ordnern/Dateien existieren**. Erst solche Kopien
samt Struktur/Version prüfen, nur verbleibende notwendige Lücken neu erstellen.
[Präzisierter Wiederaufbauplan](SOURCE_RECOVERY_PLAN.md).

**Neueste Nutzerinformation, 02.11.2026:** In der nächsten Sitzung wird ein
zusätzlicher Ordner `climate_modelling` verfügbar. Dieser neue Bestand ist noch
nicht geprüft, sein genauer Pfad noch nicht gebunden. **Erste Aufgabe für Claude
ist der gezielte Abgleich**, insbesondere mit den Ladepfaden in `main.R` und
`main_v1.2.R`, vor weiteren Neuerstellungen oder dem alten TF-Teilauftrag.
[Vollständige Übergabe](CLAUDE_RESUME_CLIMATE_MODELLING_2026_11_02.md).

Die neue [Fehlenden-/Kopienliste](MISSING_PROJECT_FILES_2026_11_02.md) enthält
39 priorisierte Anforderungen und acht getrennte historische Ergebnisvarianten;
16 direkte Ladelücken betreffen den bisherigen lokalen R-Einstieg. Sie ist ein
Snapshot **vor climate_modelling**, kein endgültiges Urteil über den neuen
Bestand. Relevante `src`-Dateien mit mtime nach 2021 sind priorisiert;
Erstellungs-/Kopierzeiten sind kein Nachweis historischer Ausführung.

Eingefrorene Nachweise/Startaufträge erhalten. Ein Dateifund schließt keine
Voraussetzung automatisch. Keine Speicherortfrage zu der bereits als unbekannt
beantworteten finalen Basisdatei, keine pauschale PC-Suche oder neuen Läufe.

## Phase 1: lokale Rekonstruktionen der Dissertation nachbauen

Ziel ist der nachvollziehbare Nachbau des historischen lokalen
Rekonstruktionsablaufs: Eingaben, Taxonfunktionen, Zuordnungen, Klimagrößen,
Referenzkurven, Priors, Rechenschritte und Nachbearbeitung anhand der tatsächlich
belegten Dissertation und ihrer Quellfassungen zusammenführen. Zunächst einen
hinreichend belegten Originalfall vollständig erschließen, danach die weiteren
lokalen Fälle derselben Methodik. Der vorhandene Kinneret-Paperfall mit
Biommodell und Wintertemperatur ist eine technische Referenz, kein bereits
abgeschlossener Nachbau der taxonbasierten Dissertation.

### Reihenfolge und konkrete Ergebnisse

| Schritt | Arbeit | Prüffähiges Ergebnis |
| --- | --- | --- |
| 1. Fallmatrix | Die lokalen Fälle aus Kapitel 3.3 anhand vorhandener Abgleiche und Quellen erfassen: Totes Meer mit getrenntem Holozän-/Glazialfall, Kinneret, Birkat Ram, Van, Iznik und Ohrid. Die genaue Fall-/Zeitfensterliste an der belegten Dissertationsfassung prüfen. | Je Fall: Fossil-/Polleneingabe, Chronologie, benötigte Taxa/TFs, Klimadaten und Bezugszeitraum, Referenzen, Anker, Codefassung, gespeicherte Ausgaben sowie Status belegt/fehlt/widersprüchlich. Keine neue pauschale Gesamtinventur. |
| 2. Ersten Referenzfall festlegen | Fall anhand belegter Originaleingaben und Laufbindung auswählen, nicht anhand günstiger Ergebnisse. Bereits gerechneten Paperfall zur technischen Kontrolle weiterverwenden. | Ein festgehaltener Fall mit Zeitfenster, Quellbelegen, vollständiger Abhängigkeitsliste und genau benannten Lücken. Nur bei einer fachlich nicht ableitbaren Auswahl konkret nachfragen. |
| 3. Historischen Datenvertrag schließen | Benötigte numerische TFs samt Taxonreihenfolge, Art-/Pollentyp-Zuordnung, Achsen, Einheiten, Klima-Zeitgrundlage, Altersbehandlung, Referenzkurven und Anker an Erzeuger und Verbraucher binden. | Geprüfte Eingabetabellen, Quellen-/Prüfsummenbindung und belegte Transformationsregeln. Fehlende Eingaben oder widersprüchliche Quellstände ausdrücklich ausweisen. |
| 4. Historischen Ablauf nachbilden | Belegte Taxon-/Pollentyp-Zusammenführung, Kombination zwischen Taxa, Sampler und Auswertung einschließlich historischer Nachbearbeitung getrennt implementieren bzw. vorhandene Teile zuordnen. | Eigener, klar gekennzeichneter Reproduktionsmodus mit kleinen technischen Rechenvergleichen; erhaltene Baseline und bisherige Standards bleiben unverändert. Keine Verbesserung still in den Referenzablauf mischen. |
| 5. Lokalen Referenzfall reproduzieren | Bei belegten Eingaben und geklärter fallbezogener Freigabe einen abgegrenzten Lauf gegen vorhandene Originalausgaben prüfen. Bereits vorhandene passende Läufe wiederverwenden. | Gesicherte Konfiguration, Quell-/Eingabehashes, Laufzeitumgebung, Seeds, Ergebnisse, technische Diagnostik und Abweichungsbericht. Exakte, numerisch angenäherte und nur grafisch prüfbare Übereinstimmung auseinanderhalten. |
| 6. Weitere lokale Fälle und App | Den belegten Ablauf auf die übrigen Fälle übertragen, jeweils erst nach Prüfung ihrer eigenen Eingaben und Chronologie; vorhandene App-/CLI-Muster verwenden. | Fallweise reproduzierte Ergebnisse, Quellen, Grenzen und Abweichungen. Nicht vorhandene Fälle bleiben offen und werden nicht durch bloße Darstellung als reproduziert gewertet. |

**Schritt 1 abgeschlossen, 02.11.2026:**
[Fallmatrix, Belege und Grenzen](local_case_matrix_2026_11_02/README.md).
Noch kein Fall vollständig reproduzierbar; Kinneret ist der erste gezielte
Klärungsfall. Nächster Auftrag: seinen historischen Basiseintrag und die
geordnete Proben-/Taxon-/TF-Zuordnung an erhaltene Originaleingaben binden
(Schritt 2/3), ohne neue Läufe.
**Teilfortschritt:** [Kinneret-Eingabebindung](kinneret_binding_2026_11_02/README.md)
belegt 160 Proben und zwölf ältere Taxonspalten. Finale Basisdatei und TFs
bleiben offen; weiterer Speicherort ist dem Nutzer nicht bekannt. Nächster
unabhängiger Schritt in Phase 1: Referenz-/Ankervertrag konkretisieren.
**Weiterer Teilfortschritt:** [Referenz-/Ankerpfad](kinneret_references_2026_11_02/README.md)
ist dokumentiert; numerische Eingaben und Laufbindung bleiben offen. Als Nächstes
historische Nachbearbeitung (Ziehungsauswahl und Dichtekombination) gezielt binden.
**Nachbearbeitung abgeglichen:** [Quellvarianten und Grenzen](kinneret_postprocessing_2026_11_02/README.md).
Originalposterioren und finale Auswahl-/Kombinationsfassung offen.
**Belegvertrag abgeschlossen:** [Maschinenprüfbare Phase-1-Spezifikation](kinneret_phase1_spec_2026_11_02/README.md)
mit lesendem Prüfer, 26 Belegdateien, 19 Befunden und elf offenen Nachweisbündeln.
Keine finale Laufvariante gewählt; der vollständige historische Datenvertrag
und Referenzlauf bleiben fachlich offen. Neue Belege für Originalbasis/Chronologieliste,
numerische TFs und Laufzuordnung notwendig. Bekannte Speicherortfrage nicht
wiederholen. Ohne Originalbelege keine Ersatzdatenreproduktion oder Phase 2 voraussetzen.
**Isolierter Teilfortschritt in Schritt 4:** [Deterministische Taxongewichtung/PDF-Summe](literal_final_pdf_2026_11_02/README.md)
als getrenntes Modul nachgebildet, 22 synthetische Tests bestanden. Weder finale
Variante noch Fallreproduktion daraus ableiten. Unabhängig vom fehlenden
Originaleingabestand als nächsten klar begrenzten Schritt die bereits gebundene
Alters-/Tiefenübertragung aus `src/check_ages.R` mit synthetischen Matrizen
prüfen. Keine neue Chronologie oder Fossilrekonstruktion.
**Weiterer isolierter Teilfortschritt in Schritt 4:**
[Alters-/Tiefenübertragung](literal_age_depth_2026_11_02/README.md)
aus `check_ages.R` und `mcmc_prepare.R` begrenzt nachgebildet; 43 neue
synthetische Tests, zusammen mit bisherigen Teilprüfungen 85 grün. Quelle,
Dimensionen, rho(D|A)-Orientierung, Einheiten und Grenzen sind dokumentiert.
Mittelwertmatrix bleibt von Listenchronologie getrennt, keine finale Variante
gewählt. Nächster unabhängiger Einstieg: Verbraucher von `age`/`site.age` und
`reconst_ages` im gebundenen `src/mcmc_functions_v1.6.R` auf Zeilen-/Zeitfenster-
auswahl prüfen, bevor die getrennten Module verbunden werden. Elf offene
Nachweisbündel und fehlende Originaleingaben bleiben unberührt.
**Altersraster und Zeilenauswahl abgeglichen:**
[Getrennter technischer Ausschnitt](literal_age_selection_2026_11_02/README.md)
mit 39 neuen synthetischen Tests, einschließlich Vorpaketen 124 grün.
Exakte Rastermitgliedschaft, vorläufige/wirksame Maske und deaktivierter
Taxonwechselzweig sind getrennt; keine automatische Achsensynchronisierung.
Vier bisherige Quellhashbindungen bestätigt. Elf offene Nachweisbündel und
fehlende Laufzuordnung bleiben unverändert. Als nächsten unabhängigen Teil
den kumulativen Dichte-/EMD-Ausschnitt in `src/mcmc_functions_v1.6.R:366–370,485–490`
gegen skalare Referenzargumente der Fallstudien belegen und synthetisch prüfen.
Keine neue Referenzkurve oder Kalibrierung; Module noch nicht zum Vollrunner verbinden.
**Kumulative Rohdichte-/EMD-Teilprüfung abgeschlossen:**
[Nullreferenz, Breitenbindung, Formeln und Prüfung](literal_cdf_discrepancy_2026_11_02/README.md).
43 neue synthetische Tests, mit Vorpaketen 167 grün. Keine neue Referenzkurve,
Normierung oder physikalische Distanz hinzugefügt. Drei Quellhashbindungen
bestätigt. [Claude-Startauftrag](CLAUDE_NEXT_STEP_2026_11_02.md) beschreibt den
nächsten begrenzten Schritt: ein gemeinsames künstliches Beispiel zur Prüfung
der vorhandenen Modulschnittstellen, Reihenfolge und Normierungszeitpunkte.
Nur Test/Belegpaket; kein historischer Vollrunner oder neue App-/CLI-Integration.
Elf Nachweisbündel bleiben offen, keine finale Laufbindung oder Phase 2.
**Gemeinsame synthetische Modulprüfung abgeschlossen:**
[Rechenkette und unabhängige Bruchrechnung](literal_phase1_chain_2026_11_02/README.md):
33 neue Tests, mit Vorpaketen 200 grün. Vier Module unverändert, keine Original-
Fallrechnung. TF-Eingaben mit ungleichen Massen sind bewusst ein Schnittstellen-
beispiel; die historische TF-Vorbereitung ist damit nicht belegt.
[Neuer Claude-Auftrag](CLAUDE_NEXT_STEP_TF_PREPARATION_2026_11_02.md): als nächsten
begrenzten Teil TF-Zeilennormierung und explizite Zellmaskierung aus den
bereits gebundenen Quellzeilen prüfen; keine unbelegte Randkonvention wählen.
Der alte Startauftrag und die bisherigen Belegpakete bleiben eingefroren.
Elf Nachweisbündel und fehlende Originaleingaben bleiben offen; kein Vollrunner.
Für diese Fortsetzung die bereits abgeschlossenen Quellenprüfungen und lokalen
Bestandsnachweise verwenden. Erst aus einer konkreten Lücke neue gezielte
Quellenarbeit ableiten. Keine bereits abgeschlossenen Gesamtprüfungen wiederholen.

### Trennung von Nachbau und wissenschaftlicher Bewertung

- Historisch belegte Einstellungen und Rechenschritte zunächst als solche
  reproduzieren, auch wenn ihre statistischen Grenzen bereits dokumentiert sind.
  Beispielsweise Auswahl/Nachbearbeitung von Ziehungen nicht automatisch durch
  volle Posterior-Auswertung ersetzen. Quellenwidersprüche oder unbelegte
  Rechenschritte nicht durch eine eigene Modellentscheidung auflösen.
- Reproduktionsprüfungen bleiben notwendig: Eingaben, Einheiten, Taxonreihenfolge,
  Transformationen und Rechenwege gegen Originalbelege prüfen. Eine neue
  trainingsinterne Wahrscheinlichkeitskalibrierung oder verbesserte äußere
  Validierung ist keine vorgeschaltete Voraussetzung für den historischen Nachbau.
- Die sieben modernen Artenmodelle, räumlichen Tests und App-Metrikansichten
  bleiben erhaltene experimentelle Vergleichsarbeiten. Sie ersetzen keine
  fehlenden historischen Taxonfunktionen oder CRU-Eingaben. WorldClim nicht
  stillschweigend als historische CRU-Quelle verwenden.
- Grafische TF-Belege sind keine exakten numerischen Marginalarrays. Fehlen
  notwendige Originalwerte dauerhaft, die konkrete Grenze benennen. Ein
  methodischer Nachbau mit neu erzeugten oder approximierten Eingaben benötigt
  eine ausdrückliche fachliche Entscheidung und bleibt getrennt von einer
  historischen Ergebnisreproduktion.

### Bekannte Eingabegrenzen und Startquellen

Die numerischen `mixed_partial_arts_emd.rdata` und
`mixed_partial_arts_emd_info.rdata`, die zugehörige geordnete TF-/Taxonbindung
und exakten globalen Klimaeingaben samt Zeitachse bleiben offene Punkte.
Welche davon für den ersten Fall benötigt werden, wird in der Fallmatrix
konkret gebunden. Die 279 historischen Funktionen gelten weiterhin nicht als
wiederhergestellt. Ohrid-spezifische Lösch-/Umbenennungsregeln nicht ungeprüft
auf Kinneret oder andere Standorte übertragen. TANN und TDJF getrennt halten.

Vorhandene Einstiege:
[Dissertationsabgleich](../abgleich_thesis.md),
[lokaler Quellenabgleich](LOCAL_SOURCE_CROSSWALK.md),
[Taxonverfahren](LOCAL_TAXON_METHOD_AUDIT.md),
[gebundene TF-Plot-/Klimabefunde](HISTORICAL_SECOND_TF_PLOT_RESULT.md),
[Konsolen-/EMD-Bindung](HISTORICAL_CONSOLE_TF_LINK_RESULT.md),
[vorhandener Papervergleich](LOCAL_REFERENCE_RESULT.md) und
[portabler Forschungsstand](research_snapshot_2026_11_02/README.md).

Chronologie- und Quellen-Gates bleiben fallbezogen bestehen. Insbesondere ist
Monticchio-P1 mit LGM90D→MON16-Anschluss weiterhin offen; der Monticchio-Pilot
wird durch diese Priorisierung weder freigegeben noch zur Voraussetzung für
jeden anderen lokalen Dissertationsfall. Keine stillen Datenkorrekturen.

Phase 1 ist für einen Fall erst mit belegtem Ablauf, gesicherten Eingaben,
reproduzierten Ausgaben und dokumentierten Abweichungen abgeschlossen.
Eine reine Machbarkeitsprüfung oder der Paper-Biomfall reicht dafür nicht.
Der Phasenabschluss benennt genau, welche lokalen Fälle reproduziert sind
und welche wegen konkreter fehlender Originale offenbleiben. Offene Fälle
werden nicht stillschweigend als erledigt behandelt.

## Phase 2: Verbesserungen gegen die gesicherte Reproduktion vergleichen

Nach dem gesicherten Referenzstand und seinem Abweichungsbericht folgen
separat geplante Verbesserungen. Die historischen Ergebnisse bleiben als
Vergleich erhalten. Jede Änderung erhält eine eigene Konfiguration und wird
gegen denselben Referenzfall geprüft.

Mögliche Pakete, erst dann fachlich priorisieren:

1. Neue trainingsinterne Wahrscheinlichkeitskalibrierung und unabhängige
   räumliche Bewertung; äußere Testergebnisse nicht zur Optimierung verwenden.
2. Änderungen an Sampler, Taxongewichtung und Nachbearbeitung mit transparentem
   Vergleich zum historischen Ablauf.
3. Erweiterte Unsicherheitsmodelle, weitere Taxa/Proxys und modernisierte
   Multiproxy-Modelle gemäß [Verbesserungsentwurf](../plan_multiproxy.md).
4. Zusätzliche App-Diagnostik, etwa Reliabilitätsansicht, sowie Komfort- und
   Darstellungsverbesserungen außerhalb der für den Nachbau nötigen Integration.

Die laufende Priorität wird nicht allein wegen leichter verfügbarer
App-Erweiterungen oder moderner Ersatzmodelle wieder umgekehrt.

## Arbeitsweise und Sicherung

Keine Subagenten; kurze deutsche Fortschrittsmeldungen. Ausschließlich den
bestehenden, vom Nutzer benannten Checkout verwenden; keine neuen Klone oder
Worktrees. Lokale Änderungen und Originale erhalten. Abgeschlossene, geprüfte
Pakete jeweils committen und auf dem bestehenden Branch in PR #77 sichern;
keine automatischen Merges. Plots immer jung links, alt rechts.

Diese Änderung aktualisiert den Plan; sie startet keine neuen Trainings,
Sampler oder Fossilrekonstruktionen. Bestehende fallbezogene Freigaben und
wissenschaftliche Gates werden vor einem späteren konkreten Lauf geprüft.
