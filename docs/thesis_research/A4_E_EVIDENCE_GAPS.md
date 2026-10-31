# A4: vorhandene EMPD2-Belege und offene moderne E-Anforderungen

Stand 31.10.2026; Basis [PR #56](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/56),
`e69d95cd2833716987065833e5284d4430316b2f`.
**Die beauftragte dokumentarische Belegzuordnung ist abgeschlossen.** Technische
Eingänge und begrenzte Herkunfts-/Darstellungsbelege sind vorhanden. Der reale
Counts-Einstieg bleibt blockiert; kein moderner Trainingssatz oder realer
Datenvertrag ist freigegeben. EMPD2 wurde nicht als E gewählt. A4-Umsetzung,
reale Kalibrierung und A5/A6 sind nicht begonnen.

## Auftrag, Lesegrenze und Statusbedeutung

Maßgeblich sind [A4_TASK_BOUNDARY](./A4_TASK_BOUNDARY.md),
[MODEL_CONTRACT](./MODEL_CONTRACT.md), [Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
ausschließlich §2-Zeile A4 und [§4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
AGENTS, HANDOFF und nur der neueste PROJECT_STATUS-Abschnitt wurden für den
Einstieg gelesen. NOVELTY_MATRIX und A3-Details wurden nicht nachgeladen.

Gelesen wurden die unten bezeichneten **bestehenden Repository-Dokumente**.
Ihre früheren Originalprüfungen, Datenvergleiche und Zahlen werden als solche
zugeordnet. Keine SQLite-, SQL-, Sample-TSV-, Archiv- oder Klimadatei geöffnet,
keine ursprünglichen Datenbytes neu gehasht, keine URLs der Originalquellen
abgerufen und keine damalige Rechnung wiederholt. Die dauerhafte Sicherung
hier betrifft dieses neue Dokument, nicht eine neue Archivierung der Rohdaten.

- **belegt:** die eng bezeichnete Aussage wird von der vorhandenen Dokumentation
  getragen. Das ist keine erneute unabhängige Originalprüfung oder Freigabe.
- **offen:** die benötigte reale Aussage ist in diesem Dokumentumfang nicht
  hinreichend belegt; ein bekannter Teilbeleg bleibt trotzdem erhalten.
- **widersprüchlich:** unvereinbare Aussagen über denselben Gegenstand und
  Geltungsbereich wären ungelöst zu erhalten. Unterschiedliche Formate oder
  begrenzte Release-Abweichungen werden nicht automatisch so eingestuft.
- **nicht geprüft:** die konkrete fachliche Prüfung liegt außerhalb dieser
  Belegzuordnung. Daraus folgt weder Eignung noch das Fehlen jeglicher Belege.

Die realen OFFEN-Felder des MODEL_CONTRACT bleiben unverändert. Ein Status
„belegt“ in dieser Matrix besetzt keines dieser Felder.

## Festgelegte Belege und zusätzliche präzise Zuordnungsfrage

| Kürzel | Bestehendes Dokument | Tatsächlich genutzte Abschnitte / Reichweite |
| --- | --- | --- |
| B1 | [MODERN_COUNTS_READINESS](./empd2_counts_readiness_2026_10_31/MODERN_COUNTS_READINESS.md) | „Quellen und tatsächlich gelesener Umfang“, „Schema: Werte sind nicht automatisch Pollencounts“, „Fehlend, Null und Typkonzept“, „Klima, Standort und Zeit: Rollen statt stiller Reparatur“, „Möglicher direkter moderner Weg – nur prospektiv“. Frühere Schema-/Metadatenprüfung und genau drei Zeugen, keine neue Kohorte. |
| B2 | [INPUT_CONTRACT.json](./empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json) | Dokumentarische Felder `readiness`, `source`, `keys`, `response`, `denominator`, `missingness`, `type_definition`, `climate`, `sample_context`, `prospective_roles`, `blockers`; kein ausführbarer oder jetzt ausgefüllter Vertrag. |
| B3 | [ARCHIVE_LINEAGE](./empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md) | §1 „Offizielle Archivmetadaten zuerst“, §2 „Genau ein offiziell verwiesenes zweites Originalobjekt“, „Quelle → Version → angewandter Schritt → Archivfeld“. Frühere DOI-/Formatzuordnung und begrenzter Dumppräfix, keine vollständige Exportreproduktion. |
| B4 | [DECISION_GAPS](./empd2_archive_lineage_2026_10_31/DECISION_GAPS.md) | Einleitung, Tabelle „Offener Beleg“ und folgende Entscheidungspassage vor dem historischen Folgeauftrag. Suche begrenzt abgeschlossen, Counts-Bereitschaft weiter blockiert. |
| B5 | [DENOMINATOR_RULES](./empd2_denominator_rules_2026_10_31/DENOMINATOR_RULES.md) | „Zwei eng begrenzte Quellenobjekte“, „Was tatsächlich angewandt erscheint: gelieferte Prozentwerte“, „Gruppenfelder: Herkunft belegt, Hierarchie-Anwendung offen“, „Sparse-Zeilen: sichtbarer Importpfad ist kein universeller Nullbeleg“, „Drei Nennerrollen ausdrücklich auseinanderhalten“. Nur zur unten genannten Frage nachgeladen. |
| B6 | [COUNTS_DECISION_BRIEF](./empd2_counts_decision_2026_10_31/COUNTS_DECISION_BRIEF.md) | „Entscheidungsgrundlage und Nennerrollen“; zur begrenzten Aussage über `N_original`, `S_flag`, `n_ausgewählt` und den blockierten realen Einstieg. Keine historische Option oder Forschungszieländerung gewählt. |

**Konkrete Frage für B5/B6:** Belegt der in B3 genannte kompatible Prozentnenner
`S_flag` das ursprüngliche Proben-N, und erlauben 36/40 Gruppen sowie erhaltenes
Volumen 7,4 die Gleichsetzung des gepinnten Release-Codes mit der tatsächlichen
SQLite-Erzeugung? Die dortigen Abschnitte verneinen diese weitergehenden
Schlüsse ausdrücklich. Keine zusätzliche Evidenz-JSON war zur dokumentarischen
Zuordnung nötig; verlinkte Originale, Codes und Vergleichsdateien wurden nicht
neu geöffnet oder ausgeführt.

## Belegmatrix: E-Anforderung, Reichweite und konkret fehlender Nachweis

Vertragsbezüge: **R** = MODEL_CONTRACT „Rollen und unbesetzte Eingänge“;
**M** = „Beobachtungsstatus, Nullen und Abdeckung“;
**K** = „Kalibrier-Cut und gemeinsame bedingte Normierung“;
**D** = „Proxy-Abhängigkeiten und gemeinsame Evidenz“;
**S** = „Support und Identifizierbarkeit“. Die Anforderungen stehen außerdem
in A4_TASK_BOUNDARY „Anforderungen, vorhandene Belege und fehlende Entscheidungen“.
Die Abschnitts- und JSON-Feldangaben beziehen sich auf die obige Belegliste.

| ID / E-Anforderung, Vertrag | Vorhandene Aussage mit genauer Fundstelle | Reichweite und Status | Konkret noch fehlender Nachweis |
| --- | --- | --- | --- |
| E01 – Archividentität/Herkunft, R | B2 `source.file/doi/url/sha256/bytes`; B3 §1 sowie „Quelle → Version → angewandter Schritt → Archivfeld“: DOI 10.1594/PANGAEA.909130, konkrete SQLite-URL und gesperrte Bytes sind zugeordnet. | **belegt:** dokumentierte Identität dieses Archivobjekts und früherer Bezugs-/Hashnachweis. Hier keine neue Byteprüfung. | Für einen späteren Eingang zusätzlich E02 und die beabsichtigte Auswahl-/Verwendungsrolle; keine Rohdatenarchivierung allein aus diesem Dokument ableiten. |
| E02 – angewandte Daten-/Exportversion, R | B3 §1/§2 und Verbindungstabelle; B4 Tabelle: konkrete PostgreSQL→SQLite-Verarbeitung und Zuordnung des v2.0-Releases zu diesem Build bleiben unbelegt. | **offen:** offizieller Formatverweis, `pg_dump` 11.2 und DataCite-Relation belegen keine tatsächlich angewandte Konvertierung. | Archivgenaue Quelle→Daten-/Codeversion→ausgeführter Export→Feldsemantik-Kette; keine neue Suche hier. |
| E03 – Messbeleg-/Konzeptschlüssel, R/D | B1 „Schema“; B2 `keys`: Messschlüssel `(p_counts.samplename,p_counts.var_)`, Original-/akzeptierte Konzepte und Klima-Join über `samplename` sind dokumentiert. | **belegt:** technische Zuordnungsrollen; keine botanische Identität, Zusammenlegung oder Unabhängigkeit daraus. | Nachvollziehbare Herkunft und Abhängigkeitsrollen der später tatsächlich zugelassenen Messereignisse; Schlüssel erhalten. |
| E04 – zulässige Messgröße/Countstatus, M | B1 „Schema“; B2 `response`: numerischer Speichertyp, `metadata.ispercent=0`, aber heterogene Messgrößen; dokumentiertes Volumen 7,4 ist keine Pollenkornzahl. | **offen:** Nicht-Prozentflag und Ganzzahligkeit einzelner Zeugen erlauben keine pauschale Count-Likelihood. Der Volumenzeuge belegt keine gebrochene Pollenzählkohorte. | Beleg des Mess-/Erfassungsprozesses und der zulässigen Reaktionsgrößen; keine Rundung oder neue Teilmenge. |
| E05 – ursprünglicher Nenner `N_original`, M | B2 `denominator`; B3 letzter Absatz; B5 „Drei Nennerrollen“; B6 „Entscheidungsgrundlage und Nennerrollen“: kein unabhängiger ursprünglicher Messumfang belegt. | **offen:** weder alle gespeicherten Werte noch Gruppenflag oder spätere Teilmengensumme ersetzen N. | Originaler Mess-/Ausschlussprozess und zum jeweiligen Beobachtungssystem passendes Proben-N. |
| E06 – Prozentdarstellung/`S_flag`, M | B5 „Was tatsächlich angewandt erscheint“: bei genau drei bekannten Zeugen stimmen 108 Messwert-Tupel überein; 98 gelieferte Prozentwerte sind innerhalb ihrer gedruckten Präzision mit `S_flag` kompatibel. B6 Nennerrollentabelle begrenzt den Schluss. | **belegt:** dokumentierte lokale Kompatibilität der Prozentdarstellung. Keine neue Berechnung, keine globale Algorithmus- oder N-Aussage. | Für reale Counts E05; für Prozent-only E09; keine Verallgemeinerung auf weitere Proben oder Pseudo-Counts. |
| E07 – Gruppenhierarchie/disjunkte Abdeckung/Rest, M | B1 „Schema“; B2 `denominator.candidate_metadata/rest_category_defined`; B5 „Gruppenfelder“: Eltern-ID und beide Flags der 36 gemeinsamen Gruppen stimmen überein, SQLite hat 40 Gruppen. | **offen:** Feldherkunft ist dokumentiert; tatsächliche Hierarchieanwendung und erschöpfende disjunkte Beobachtungspartition fehlen. | Angewandte Hierarchie-/Nennerregel einschließlich Ausschlüssen und Rest; keine neue Aggregation. |
| E08 – fehlend, Messnull, biologische Absenz, M | B1 „Fehlend, Null und Typkonzept“; B2 `missingness`; B5 „Sparse-Zeilen“: fehlendes Paar AlbaSanchez_a1/Code 7 ohne belegten Status; sichtbarer Neuanlagepfad unterscheidet verschiedene Auslassungsursachen nicht. | **offen:** dokumentierter Nichttreffer und Codepfad liefern keine universelle Archivnullregel. Positive Eingangsprüfung mit anschließender Rundung kann zudem gespeicherte Null erzeugen. | Archivbezogener Erfassungs-/Auslassungsprozess und gültiger Status je betroffenem Messfeld; keine Nullfüllung oder Absenzinterpretation. |
| E09 – Prozent-only/Rundung/Nachweisgrenze, M | B1 „Schema“ trennt Proben- und Gruppenflags; B2 `response.rounding_or_percentage_to_pseudocounts_allowed=false`; B5 Prozentvergleich betrifft nur gelieferte Präzision der drei Zeugen. | **offen:** der frühere Präzisionsvergleich ist kein realer Mess-/Präzisionsvertrag. | Begründeter Prozent-/Rundungs-/Nachweisgrenzenprozess für einen später tatsächlich verwendeten Eingang; kein N=100. |
| E10 – Original-/akzeptierte Namen, R/M | B1 „Fehlend, Null und Typkonzept“; B2 `keys/type_definition`: beide Konzeptschlüssel und Namen sind dokumentiert; numerische ID-Räume dürfen nicht gleichgesetzt werden. | **belegt:** vorhandene Namens-/Schlüsselrelation, keine morphologische Synonymie oder Artbrücke. | Für konkrete Beobachtungstypen E11; keine automatische Poolbildung aus akzeptierten Namen. |
| E11 – Typdefinition/Beobachtungspartition, R/M | B2 `type_definition`: morphologische Definition und moderne Partition nicht freigegeben; B1 „Fehlend, Null und Typkonzept“ und B6 Grundlage erhalten W96-/Artbrückengrenzen. | **offen:** vorhandene Datenbankrelation identifiziert keine eindeutigen Artenanteile oder gemeinsame Messabdeckung. | Belegte Bestimmungskonzepte, Partitions-/Abdeckungsregel und nachvollziehbarer Umgang mit Mischtypen; keine Typ-/Artwahl. |
| E12 – vorhandene Klimafelder und dokumentierte Zuordnung, R | B1 „Klima, Standort und Zeit“; B2 `climate.fields/importer_contract_units/source_recorded_in_repository`; B5 „Zwei eng begrenzte Quellenobjekte“ berichtet die frühere Originalangabe WorldClim2, 30″, nächster Gitterpunkt. | **belegt:** technische Felder `t_ann/p_ann`, dokumentierte Importereinheiten °C/mm pro Jahr und begrenzte Quellenzuordnung. Keine neue Originalprüfung. | Konkrete Klima-/Zeitgeltung in E13/E15; `t_djf` nicht als `t_ann` oder TANN/PANN als GDD5/MTCO/AET-PET behandeln. |
| E13 – Klimaversion/Normalperiode, R/S | B1 „Klima, Standort und Zeit“; B2 `climate.reference_period=unverified`; B5 „Zwei eng begrenzte Quellenobjekte“: in den damals gelesenen Originalabschnitten keine Normalperiode genannt. | **offen:** WorldClim2-Zuordnung und räumliche Auflösung bestimmen keine gesicherte konkrete Version/Periode der Archivfelder. | Präziser, zum gelieferten Klima passender Versions-/Periodenbeleg; kein WorldClim-2.1-/1970–2000- oder CRU-1961–1990-Label ableiten. |
| E14 – Standort-/Kontextmetadaten vorhanden, R | B1 „Klima, Standort und Zeit“; B2 `sample_context.available_fields`: Probenart, Kontext, Methode, Alter/Unsicherheit, Standortzuverlässigkeit sind als Felder dokumentiert. | **belegt:** Rollen und gespeicherte Klassenbeschreibungen; kein jetzt gemessener Positionsfehler und keine geeignete Kohorte. | E15/E16 und tatsächliche Eignung/Unsicherheitsbedeutung der später verwendeten Proben. |
| E15 – zeitliche Eignung/Repräsentativität, R/S | B1 „Klima, Standort und Zeit“; B2 `sample_context.age_in_three_witnesses/contemporaneous_with_climate_assumed`: Alter/Altersklasse bei drei Zeugen unbekannt, Mehrfachsammlung/zeitliche Integration ungeklärt. | **offen:** „modern“ oder numerischer Klima-Join belegen keine Gleichzeitigkeit zur Klimanormalperiode. | Probenzeit, Integration und belegter Bezug zu einer gesicherten Klima-Referenzperiode samt Unsicherheit. |
| E16 – Ortsbezug/Eignung einzelner Proben, R/S | B1 „Klima, Standort und Zeit“: Klassen A–D/X und Text-NULL-Zeuge; B2 `sample_context.location_X_or_text_NULL_repaired=false`. | **offen:** Koordinaten/Standortklassen können fehlen oder ungeeignet sein; keine neue Eignungszählung und kein Rettungs-Join. | Nachvollziehbare räumliche Zuordnung und Unsicherheits-/Eignungsregel vor realer Auswahl; keine Freitextkorrektur. |
| E17 – gemeinsame Herkunft/Abhängigkeiten, R/D | B1 „Klima, Standort und Zeit“ beschreibt begrenzte Publikations-Joins; „Möglicher direkter moderner Weg“ fordert Standort-/Sammlungs-/Quelltrennung; B2 `prospective_roles.split_policy`. | **offen:** ein bibliographischer Join oder fehlende Publikationszeile belegen weder vollständige Provenienz noch unabhängige Klimareplikate. | Mess-/Standort-/Sammlungs-/Quellenrollen, Überschneidungen und Abhängigkeiten vor Training; keine Herkunft aus fehlendem Join verneinen. |
| E18 – räumliche Aufteilung/Support, S | B1 „Möglicher direkter moderner Weg“; B2 `prospective_roles.split_policy`, `sample_context.cohort_or_filters_selected=false`: Rollen vor Fits festlegen; keine Kohorte/Hülle berechnet. | **offen:** vorhandene Datenfelder sind kein vorab festgelegter Split oder empirischer Supportnachweis. | Datenrollen, Trennung abhängiger Einheiten und Klima-/Typ-/Kontextsupport samt Auswertungsregeln; keine Block-/Schwellenwahl anhand alter Scores. |
| E19 – unabhängige Bestätigung, S | B1 „Möglicher direkter moderner Weg“; B2 `blockers`, `prospective_roles.known_outer_scores_used_for_decisions=false`: neue geeignete Evidenz erforderlich. | **offen:** kein neuer bestätigender empirischer Nachweis in diesem Dokumentumfang. | Vorab gesperrter Prüfplan und geeignete unabhängige Evidenz; bekannte Regionen/Scores nicht als ungesehene Bestätigung darstellen. |
| E20 – Menezes-nahe Basis, reale Pollenantwort und q, K | A4_TASK_BOUNDARY „Anforderungen“ und MODEL_CONTRACT K lassen reale Verteilungen/Parameter offen; B2 `readiness.models_fitted=false`, `prospective_roles.new_model_family_chosen=false`. | **nicht geprüft:** keine Modellspezifikation, Reproduktion, Antwort-/Unsicherheitsprüfung oder Kalibrierung Gegenstand dieses Pakets. | Gesondert begründetes und autorisiertes reales Design samt Quellen-/Klimazielkompatibilität und Prüfung nach geklärten Eingängen; kein q aus A3 übernehmen. |

## Dokumentierte Abweichungen und nicht tragfähige Gleichsetzungen

Im gelesenen Umfang wurde kein ungelöster Widerspruch zwischen den
**entsprechend begrenzten Dokumentaussagen** festgestellt. Insbesondere sind
„technisch vorhanden“ und `modern_count_training_ready=false` vereinbar.
Zwei konkrete Abweichungen verhindern weitergehende Gleichsetzungen:

| Dokumentierter Befund | Welche Gleichsetzung damit nicht belegt ist | Erhaltene Grenze |
| --- | --- | --- |
| B5 „Gruppenfelder“: 36 Releasegruppen, 40 SQLitegruppen; gemeinsame 36 Eltern-/Flagdefinitionen stimmen überein. | Das gepinnte Release sei als Ganzes mit dem SQLite-Archiv identisch. | Keine Gruppen entfernen/ergänzen oder Daten-/Buildversion angleichen. |
| B5 „Sparse-Zeilen“: der sichtbare Neuanlagepfad würde 7,4 auf 7 runden; SQLite und Release-TSV erhalten 7,4. Vorhandene Proben werden übersprungen. | Genau diese Neuanlage erkläre den erhaltenen Archivwert; ebenso wenig folgt „Skript niemals benutzt“. | Tatsächlich angewandte Verarbeitung bleibt offen; kein Nachbauen, Runden oder Reparieren. |

`N_original`, `S_flag` und `n_ausgewählt` bleiben drei unterschiedliche Rollen.
Der dokumentierte Prozentvergleich sichert einen engen Darstellungsbefund;
seine Präzision ist kein Beleg für den ursprünglichen Messumfang, eine
allgemeine Nullregel oder eine zulässige reale Count-Likelihood. Diese
Zuordnung löst weder die Archivkette noch den Erfassungsprozess.

## Ergebnis, Grenzen und Sicherung

Die beauftragte Belegmatrix ist vollständig innerhalb des festgelegten
Dokumentumfangs erstellt; offene Felder und nicht geprüfte Modellfragen sind
benannt. Es wurde keine neue empirische Aussage durch Datenprüfung gewonnen.
Für eine präzisere Archivkette, einen ursprünglichen Messnenner oder eine
allgemeine Nullregel reichen diese Dokumente weiterhin nicht. Eine neue
externe Recherche oder echte Datenprüfung wurde deshalb nicht begonnen.

Bestehende Designs einschließlich A3_SYNTHETIC_DESIGN, MODEL_CONTRACT, Modelle,
Resultate, Register, Verträge einschließlich B2, Importer, Flags, Counts/Gruppen,
Baseline und CLI/App unverändert. Nur dieses neue Dokument und kurze
HANDOFF-/Statuspflege. Keine reale Auswahl/Beschaffung/Bereinigung, Datenmatrix/
Kohorte, Typbrücke, Implementierung, neuen Rechnungen, Tests, Fits, Scores,
Optimierung, Zufallssimulationen oder Kernläufe. A1/A2 dokumentarisch
abgeschlossen, A3 ausschließlich endlich-synthetisch bestanden. Parnell-Zugang,
Menezes-(2)/(5), W96 und reale T0/T1/T3/T4/Monticchio-P1 bleiben Grenzen;
kein eigener Neuheitsbeitrag abgegrenzt. Keine Unteragenten, Autorennachrichten
oder Merges.

Basis-PR #56 offen/ungemergt und mergebar, automatische
[CI 41237720645](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41237720645)
erfolgreich; sieben vorgefundene Worktrees sauber, keine Stashes/Resets;
main unverändert `7f1826c`. Eigener Worktree `/workspace/netzel-a4-e-evidence-gaps`,
Branch `codex/netzel-a4-e-evidence-gaps`, eigener PR gegen #56 ohne Merge.
Quellenzuordnung, JSON-Feldbezüge, 22 lokale Links/Anker und Drei-Dateien-Umfang
geprüft; 750 geschützte Blobs/Dateimodi einschließlich Symlink unverändert,
Statushistorie erhalten. Sechs eng begrenzte Aussagen belegt, 13 Anforderungen
offen, eine Modellfrage nicht geprüft. Commit/PR und eigene automatische CI
in HANDOFF/Status.

Genau ein Folgeauftrag wird in [HANDOFF](./HANDOFF.md) vorbereitet:
vorhandene Klimaprovenienz-/Zeitbezugsangaben zu E12/E13/E15 gezielt in der
bestehenden Repository-Dokumentation präzisieren, ohne neue Quellen, reale
Auswahl oder Klima-Join. Er wartet auf das nächste Nutzer-OK.
