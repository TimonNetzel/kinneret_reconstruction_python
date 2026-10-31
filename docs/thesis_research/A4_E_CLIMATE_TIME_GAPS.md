# A4: dokumentierte Klimaprovenienz und offene Zeitverbindung

Stand 31.10.2026; Basis [PR #57](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/57),
`e504d89c159ec02459643ff37b8be657af453f37`.
**Die beauftragte dokumentarische Präzisierung von E12/E13/E15 ist abgeschlossen.**
WorldClim2, 30 Bogensekunden und nächster Gitterpunkt sind als publizierte
Zuordnung dokumentiert; konkrete angewandte Klima-/Produktversion,
Referenzperiode und zeitliche Eignung der Proben bleiben offen. Kein Klimaeingang,
Zeitraum oder reales E wurde gewählt. A4-Umsetzung/reale Kalibrierung nicht
begonnen; reale Counts-Bereitschaft weiter blockiert.

## Umfang und Aussagegrenze

Maßgeblich sind [A4_E_EVIDENCE_GAPS E12/E13/E15](./A4_E_EVIDENCE_GAPS.md#belegmatrix-e-anforderung-reichweite-und-konkret-fehlender-nachweis),
[MODEL_CONTRACT „Rollen und unbesetzte Eingänge“](./MODEL_CONTRACT.md#rollen-und-unbesetzte-eingänge),
[Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
ausschließlich §2-Zeile A4 und [§4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
AGENTS, HANDOFF und nur der neueste PROJECT_STATUS-Abschnitt für den Einstieg.

Dieses Paket prüft **vorhandene Dokumentation**. Eine frühere Originallesung
wird als solche zugeordnet; sie wurde hier nicht wiederholt. Keine externe
Originalquelle, Datenbank, Archiv-/Klimadatei, Importausgabe oder Dissertation
geöffnet; kein Importer oder Modellcode geladen/ausgeführt, keine Daten-/Probenprüfung,
Rechnung oder manuell gestartete Projekt-/Modellprüfung. Die berichteten
historischen Zahlen sind keine neuen Auszählungen. Ein zunächst zu breiter
Leseausschnitt des alten Quellenberichts enthielt auch andere Projektabschnitte;
für diese Nachweiskette wurden ausschließlich seine EMPD2-Klimaangaben verwendet.
Keine neue fachliche Prüfung dieser anderen Abschnitte.

## Benutzte Dokumentstellen und ihr damaliger Prüfstatus

| Kürzel | Konkreter Beleg | Herkunft der Aussage und Reichweite |
| --- | --- | --- |
| K1 | [source_audit_2026_10_29/README, „Moderne Kalibrierung“](./source_audit_2026_10_29/README.md#moderne-kalibrierung), EMPD2-Absätze | Früherer Quellen-/Dateiprüfbericht vom 29.10.2026. Referiert die Publikationsangabe WorldClim2/30″/nächster Gitterpunkt; kennzeichnet die konkrete Referenzperiode der heruntergeladenen Klimatabelle als nicht separat dateiintern belegt. Dokumentiert unterschiedliche TSV-/SQLite-Proben-IDs. |
| K2 | [tools/thesis_import/README, „CSV-Schnittstelle, Version 1“](../../tools/thesis_import/README.md#csv-schnittstelle-version-1), Absatz nach der Dateitabelle; „Gesicherte Sonderfälle“ | Bestehender Import-/Schnittstellenvertrag: `tann_c/pann_mm_per_year` als WorldClim2, Referenzperiode `unverified`, raw-Token erhalten. Technische Feld- und Fehlerbehandlung; keine neue Herkunftsprüfung oder wissenschaftliche Trainingsfreigabe. |
| K3 | [INPUT_CONTRACT.json](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json), ausschließlich `climate` und `sample_context` | Dokumentarischer Stand des früheren Readiness-Pakets. Hält Feldrollen/Importereinheiten und übernommene Quellenangaben fest; `method_original_rechecked_this_package=false` bezieht sich auf jenes Paket. `reference_period=unverified`, keine angenommene Gleichzeitigkeit. |
| K4 | [MODERN_COUNTS_READINESS, „Klima, Standort und Zeit: Rollen statt stiller Reparatur“](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md#klima-standort-und-zeit-rollen-statt-stiller-reparatur) | Frühere Prüfung genau dreier Zeugen und begrenzter Metadaten. Verweist für Klimaherkunft auf K1/K2; erklärt diese Übernahme ausdrücklich nicht zur neuen Volltextprüfung. Unbekanntes Alter/Altersklasse der drei Zeugen, Kontext-/Integrationsgrenzen und gespeicherte Standortklassen. |
| K5 | [DENOMINATOR_RULES, „Zwei eng begrenzte Quellenobjekte“](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md#zwei-eng-begrenzte-quellenobjekte), Punkt 1 | Dokumentierte frühere direkte Lesung des Davis-et-al.-Original-HTML, DOI 10.5194/essd-12-2423-2020; unter anderem §2/§2.2. Berichtet in §2 WorldClim2/30″/nächster Gitterpunkt, dort keine Klimanormalperiode. Begrenzter Nichtbeleg, keine Aussage über sämtliche Originalmaterialien. |

K4 und K3 übernehmen für Teile der Klimaprovenienz K1/K2. Diese Wiederholungen
sind keine zusätzlichen unabhängigen Herkunftsnachweise. K5 dokumentiert eine
spätere eigene Originallesung, beschränkt deren Perioden-Nichtbeleg aber auf
die damals gelesenen Stellen. Die Zahl der Dokumente ersetzt keine fehlende
Verbindung zum tatsächlich angewandten Klimaeingang des Archivs.

## Nachweiskette für Klimaquelle, Version und Referenzperiode

| ID / Aussage | Fundstelle / damaliger Quellenstatus | Vereinbare Aussage und Geltungsbereich | Fehlender Nachweis / Konsequenz |
| --- | --- | --- | --- |
| C01 – publizierte Zuordnung | K1 „Moderne Kalibrierung“; K5 „Zwei eng begrenzte Quellenobjekte“, Punkt 1, frühere Originallesung §2 | **Dokumentiert:** WorldClim2, 30 Bogensekunden, nächster Gitterpunkt als im Artikel genannte Zuordnung. K1 und K5 sind darin vereinbar. | Diese Produkt-/Methodenangabe allein identifiziert keine konkrete angewandte Rasterversion oder Referenzperiode der archivierten Felder. |
| C02 – technischer Feldbezug | K3 `climate.fields`, `different_temperature_field_not_interchangeable`; K2 „CSV-Schnittstelle“ | **Dokumentiert:** `climate.t_ann/p_ann` und Importrollen `tann_c/pann_mm_per_year`; `t_djf` hat eine andere Rolle. | Ein technischer Join bzw. Feldname ersetzt keinen archivgenauen Verarbeitungs-/Versionsbeleg. Keine Spalten vertauschen. |
| C03 – Einheiten | K3 `climate.importer_contract_units`; K2 Feldnamen des Schnittstellenabsatzes | **Dokumentierter Importvertrag:** `t_ann` in Grad Celsius, `p_ann` in mm pro Jahr. K4 begrenzt das SQLite-Schema: kein eigener Klima-/Perioden-/Einheitennachweis dort. | Die Einheit ist als Repositoryvertrag zugeordnet; hier keine neue Bestätigung aus Rohdatei oder Originalmetadatum. Unveränderten Vertrag nicht als selbständigen Archivnachweis ausgeben. |
| C04 – konkrete Klima-/Produktversion | K1 WorldClim2-Angabe; K3 `source_recorded_in_repository`; K4 Warnung vor WorldClim-2.1-Label | **Offen:** „WorldClim2“ ist die dokumentierte Zuordnung; eine konkrete angewandte Unterversion wird in diesen Belegen nicht gesichert. | Beleg, welche konkrete Klimaversion in genau die betrachteten Archivfelder einging; keine Unterversion aus Produktnamen ergänzen. |
| C05 – Klimanormalperiode | K1 expliziter Nichtbeleg; K2 `unverified`; K3 `climate.reference_period`; K5 begrenzte Originallesung | **Offen und vereinbar:** keine bestätigte konkrete Referenzperiode. K5 schließt die Lücke aus K1/K2 nicht. | Eine zum tatsächlich angewandten Klimaeingang belegte Normalperiode; keine Übernahme von 1970–2000, 1961–1990 oder einer anderen Standardperiode. |
| C06 – Anwendung auf genau das Archiv | K1 dokumentiert unterschiedliche IDs von normalem PANGAEA-Klima-TSV und SQLite; K2 hält die gelieferten Felder/Qualitätsflags fest; K4 nennt fehlende Provenienz im Schema | **Offen:** Publikationsbeschreibung, TSV-Datei und konkrete SQLite-Felder sind keine automatisch identische vollständige Anwendungskette. Die schon in E02 offene Erzeugungskette wird hier nicht erneut geprüft. | Konkreter Nachweis Quelle/Version/Zeitraum → angewandter Zuordnungsschritt → betreffende Archivfelder. Kein Ersatz-TSV, neuer Klima-Join oder neue Rasterzuordnung. |
| C07 – CRU-/Dissertationsbezug | K1 unterscheidet die WorldClim2-Zuordnung von CRU TS 4.01 mit 1961–1990 in der Dissertation; K2/K4 verbieten stilles CRU-Label | **Abweichende Kalibrierungsbasis dokumentiert:** Die CRU-Angabe beschreibt den dortigen Vergleichsbezug, keine EMPD2-Referenzperiode. Hier keine Dissertation neu geprüft. | Keine Gleichsetzung von WorldClim2 und CRU, keine Periodenübertragung oder erneute Originalprüfung der Dissertation. |
| C08 – numerische Verfügbarkeit | K1 korrigierter Text-NULL-Befund; K2 „Gesicherte Sonderfälle“; K3 `climate.text_NULL_witness`; K4 übernimmt historische Angaben | **Technischer Teilbefund:** historische 529 Text-NULL-Klimazuordnungen und 8.217 numerische TANN/PANN-Paare; hier nicht neu gezählt. | Numerische Verfügbarkeit beweist weder Quellen-/Periodengeltung noch Eignung. Keine Zahl freigegebener Trainingsproben daraus ableiten und keine fehlenden Werte reparieren. |

**Fehlende Verbindung:** Die dokumentierte publizierte Methode ist mit einer
archivgenauen angewandten Klimaversion und deren gesicherter Normalperiode zu
verbinden. Erst dann lässt sich ein belastbarer Zeitbezug einer konkreten
Probe prüfen. Ein generischer WorldClim-Produkttext allein wäre ohne diese
Anwendungszuordnung noch keine Schließung der Archivlücke.

## Nachweiskette für Probenzeit, Integration und räumlichen Bezug

| ID / Aussage | Konkreter bestehender Beleg | Geltungsbereich und offener Nachweis |
| --- | --- | --- |
| T01 – Zeit-/Kontextfelder | K3 `sample_context.available_fields` nennt `sampletype/samplecontext/samplemethod/agebp/ageuncertainty/locationreliability`; K2 erhält Alter/raw-Felder | **Dokumentierte Feldrollen.** Feldexistenz belegt keine gemessene oder geeignete Zeitzuordnung einer Probe; negative `age_bp`-Werte dürfen laut K2 vorkommen. Keine Zeitkonvention oder Eignungsgrenze neu festgelegt. |
| T02 – drei bekannte Zeugen | K3 `sample_context.age_in_three_witnesses`; K4 „Klima, Standort und Zeit“ | **Begrenzter früherer Befund:** Alter und Altersklasse der drei Zeugen unbekannt. Keine Aussage, alle Archivproben seien zeitlich ungeeignet oder unbekannt; keine neue Auswahl oder Zählung. |
| T03 – Ablagerung/Sammlung/Integration | K4 nennt Moss-/Mehrfachsammlung und unbekannte zeitliche Integration; K3 `contemporaneous_with_climate_assumed=false` | **Offen:** Ablagerungs-/Sammlungszeit und Integrationsbereich sind nicht aus dem Datenbanknamen „modern“ oder dem Klima-Join abzuleiten. Passender Probenzeit-/Integrationsnachweis samt Unsicherheit fehlt. |
| T04 – Klima-/Probenzeitverbindung | K3 `reference_period=unverified` und keine angenommene Gleichzeitigkeit; K4 grenzt beides explizit ab | **Offen:** selbst eine später belegte Normalperiode würde unbekannte Probenzeit nicht ersetzen; eine bekannte Probenzeit belegt umgekehrt nicht die verwendete Klimanormalperiode. Beider Zuordnung benötigt eine begründete gemeinsame Geltung. |
| T05 – räumliche Methode/Ort | K1/K5 nächster Gitterpunkt bei 30″; K4 gespeicherte Standortklassen A–D/X, Text-NULL und fehlende Standortangaben | **Unterschiedliche Rollen:** Rasterauflösung und Zuordnungsmethode belegen keine Positionsgenauigkeit, tatsächliche Probenlage oder zeitliche Repräsentativität. Klassenbeschreibungen sind keine hier nachgemessenen Fehler. Keine neue räumliche Analyse. |
| T06 – bibliographisches Jahr | K4 berichtet für Champvillair einen begrenzten Publikations-Join und gelesene Metadaten, keine dort geprüfte Methodik | **Kein Probenzeitnachweis:** Publikationsjahr oder DOI ersetzt weder Beprobungsdatum noch Integrationszeit. Ein fehlender direkter Publikations-Join beweist auch keine Quellenlosigkeit. |

Kein ungelöster Widerspruch zwischen den entsprechend begrenzten Klima- und
Zeitangaben festgestellt. Die vorhandenen Texte halten `unverified`,
Feld-/Schnittstellenrollen und lokale Zeugenbefunde auseinander. Ihre
Nichtbelege betreffen bestimmte Stellen bzw. Zeugen; sie sind keine universelle
Behauptung, dass ein Beleg nirgendwo existiert. Keine Quelle wurde korrigiert.

## Abschluss, Stopgrenze und verbleibender nächster Schritt

Die beauftragte Nachweiskette ist innerhalb des festgelegten Dokumentumfangs
abgeschlossen. E12 ist als begrenzte Quellen-/Feld-/Einheitenzuordnung präzisiert;
E13 und E15 bleiben offen. Es wurde kein fehlender realer Versions-, Perioden-
oder Probenzeitnachweis gefunden oder durch eine Annahme ersetzt. Für deren
fachliche Schließung wären zusätzliche tragfähige Evidenz oder echte
Eingangsprüfungen nötig; das liegt außerhalb dieses Auftrags. Die hier
festgelegten Dokumente werden nicht als Anlass zur erneuten externen Suche
oder zu einer fortgesetzten Wiederholung derselben Prüfung benutzt.

MODEL_CONTRACT und alle bestehenden Designs, Modelle, Resultate, Register,
Verträge, Importer, Flags, Counts/Gruppen, Baseline und CLI/App unverändert.
Nur dieses neue Dokument und kurze HANDOFF-/Statuspflege. Keine reale Auswahl,
Beschaffung, Bereinigung, Datenmatrix/Kohorte, Typbrücke, Implementierung,
Klima-Joins, neuen Rechnungen, Tests, Fits, Scores, Optimierung, Simulationen
oder Kernläufe. A1/A2 dokumentarisch abgeschlossen, A3 nur endlich-synthetisch
bestanden; A4-Umsetzung/reale Kalibrierung und A5/A6 nicht begonnen.
Parnell-Zugang, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben Grenzen; kein eigener Neuheitsbeitrag
abgegrenzt. Keine Unteragenten, Autorennachrichten oder Merges.

Standprüfung: Basis-PR #57 offen/ungemergt und mergebar; automatische CI
[41238724576](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41238724576)
bei Einstieg laufend, vor Sicherung erfolgreich. Acht vorgefundene Worktrees
sauber, keine Stashes/Resets,
main unverändert `7f1826c`. Eigener Worktree `/workspace/netzel-a4-e-climate-time-gaps`,
Branch `codex/netzel-a4-e-climate-time-gaps`, eigener PR gegen #57 ohne Merge.
Quellen-/JSON-Feldbezüge, 16 lokale Links/Anker und Drei-Dateien-Umfang
geprüft; 751 geschützte Blobs/Dateimodi einschließlich Symlink unverändert,
Statushistorie erhalten. Commit/PR und eigene automatische CI in HANDOFF/Status.

Genau ein Folgeauftrag wird in [HANDOFF](./HANDOFF.md) vorbereitet: konkrete
Wiederöffnungskriterien für die fehlende archivgenaue Klima-/Zeitverbindung
als prüfbare Nachweisanforderung dokumentieren. Keine neue Belegbeschaffung
oder reale Freigabe in diesem Folgeauftrag; wartet auf Nutzer-OK.
