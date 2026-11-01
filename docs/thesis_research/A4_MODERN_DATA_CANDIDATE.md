# A4: öffentliches modernes Menezes-Datenobjekt, begrenzt geprüft

01.11.2026. Nutzerauftrag: vom gesicherten Gesamtstand aus den nächsten
bearbeitbaren Datenengpass begrenzt bearbeiten. Basis PR #74,
`815b7f88e73efed00e52a9c88aaad954714bb68a`, offen/ungemergt,
vier Head-Checks erfolgreich. **Datenobjekt gefunden und strukturell geprüft;
kein moderner Eingang E gewählt, keine reale Kalibrierung freigegeben.**

## Warum dieser Schritt

A1/A2 sind dokumentarisch abgeschlossen, A3 ausschließlich endlich-synthetisch
bestanden. Diese Prüfungen werden übernommen. Die
[EMPD2-Belegzuordnung](./A4_E_EVIDENCE_GAPS.md) und
[Counts-Entscheidungsvorlage](./empd2_counts_decision_2026_10_31/COUNTS_DECISION_BRIEF.md)
lassen Messstatus, ursprünglichen Nenner und weitere Eingangsrollen offen;
die abgeschlossene Archivsuche wird nicht wiederholt. Die jüngeren
[Herkunfts-/Zeitbelege](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md) werden ebenfalls
übernommen. Furlanetto bleibt separat vorgemerkt und zugangsabhängig.

Die [A4-Zeile im Gesamtplan](./MODULAR_RECONSTRUCTION_PLAN.md#2-arbeitspakete-mit-prüfbarem-ende)
verlangt einen begründeten modernen Eingang und eine Menezes-nahe Basis oder
präzise dokumentierte Abweichung. Die bisherige
[Methodenmatrix](./NOVELTY_MATRIX.md#quellen-und-beleggrenze) prüfte keine
Daten-/Codeobjekte. Die neue Frage lautet deshalb: Ist der von Menezes v1
verwendete moderne Eingang öffentlich identifizierbar, und welche benötigten
Mess-/Klima-/Probenrollen bringt er tatsächlich mit? Dies ist Quellenprüfung
eines konkreten Kandidaten, keine Entscheidung für Ersatzdaten oder ein neues
annahmenabhängiges Forschungsziel.

**Umfang/Stop:** Nur v1 §1.1 zum modernen Datensatz und Anhang D zur
Bereitstellung, die zwei unmittelbar verlinkten Autoren-Repositories,
sieben kleine Dokumentations-/Ladeskriptdateien und genau ein modernes
Binärobjekt. Letzteres nur auf Container, Dimensionen, Speichertypen,
Spaltennamen und Attribute prüfen. Bei Objektidentität und konkreter Restlücke
stoppen. Keine Einzelwerte, Summen, Wertebereiche oder Datenqualität auswerten.

## Neue Belege

Das [Evidenzprotokoll](./a4_menezes_data_2026_11_01/evidence.json) enthält
Quellen/Versionen, genaue Zeilen, Blob-/SHA-256-Prüfungen, Struktur und Lesegrenzen.

| Beleg | Fundstelle und begrenzter Befund |
| --- | --- |
| M1 | [Menezes et al., arXiv:2609.08866v1](https://arxiv.org/html/2609.08866v1#S1.SS1), §1.1, moderne Daten; [Anhang D](https://arxiv.org/html/2609.08866v1#A4): erweitertes RS10, 7.832 Proben/28 Taxa, GDD5/MTCO/AET-PET, ungefähr 30-jährige stationsbasierte Klimaaggregation; Links zu Paket und Reproduktion. Posteriorziehungen werden auf Anfrage angeboten, hier nicht beschafft. |
| M2 | [Autoren-Reproduktion, README Z.3–15](https://github.com/AndrMenezes/reproduce__palaeoclimate_zanim_ln_bart/blob/42429c039c33bb9872d4bf8f2f7402802d376ed1/README.md#L3) und [Ladeskript Z.16–20](https://github.com/AndrMenezes/reproduce__palaeoclimate_zanim_ln_bart/blob/42429c039c33bb9872d4bf8f2f7402802d376ed1/application/03__fit_zanim_ln_bart.R#L16): direkte v1-Verknüpfung und Laden von `pollen_climate$Y/$X` aus `zanicc`; Klimaspalten `gdd5`, `mtco`, `aet.pet`. Installationsbeispiel ohne Commitpin; kein Ausführungs-/Umgebungsnachweis. |
| M3 | [zanicc, R/datasets.R Z.1–44](https://github.com/AndrMenezes/zanicc/blob/de7259b433c37a5628bca49254316fe63b2761c3/R/datasets.R#L1): dokumentiert Y/X, Einheiten und RS10-Bezug; beschreibt oberste 5–10 mm Seesediment und ungefähr 30-jährige stationsgewichtete Klimamittel. Generierte `man/pollen_climate.Rd` ist dieselbe Beleglinie, keine unabhängige Bestätigung. |
| M4 | [data/pollen_climate.rda](https://github.com/AndrMenezes/zanicc/blob/de7259b433c37a5628bca49254316fe63b2761c3/data/pollen_climate.rda): 182.619 Bytes; Git-Blob `e7a7d8b6fec3711af7a1888ba83871009ad8e981`, SHA-256 `2a1e8ecaac28a3f88ccffcec4000eba9db2c8ec612f3ce29afe5fa9bface5daa`. Bytes auch am datumsgefilterten Stand `e051432081cc20dd5990ad3c54cad7f8265d89f4` vom 04.09.2026 identisch. Das belegt Objektstabilität an zwei Revisionen, keine exakte historische Analyseumgebung. |

## Tatsächliche Struktur und Aussagegrenzen

Passiv mit Python/`rdata` 1.1.0 deserialisiert; keine R-Skripte ausgeführt.
Parser samt Abhängigkeiten nur temporär installiert, Projektabhängigkeiten
unverändert. Hash vor/nach gleich. Auch rohe Serialisierungsattribute geprüft,
damit das Fehlen von Kennungen nicht allein aus einer Parser-Konvertierung folgt.

| Rolle | Im Objekt tatsächlich vorhanden | Weiter offen |
| --- | --- | --- |
| Beobachtung | Ein Listeneintrag `Y`, 7.832 × 28, R-Integer; 28 Spaltennamen gesichert. | Ganzzahliger Speicher belegt weder ursprüngliche Kornzählung noch N, Messnullen, Disjunktheit oder Vollständigkeit. Keine numerischen Zellen geprüft. |
| Klima/Ort | `X`, 7.832 × 8, R-Real; `longitude`, `latitude`, `altitude`, `gdd0`, `gdd5`, `mtco`, `mtwa`, `aet.pet`. | Konkrete Stations-/Produktversion, genaue Kalenderperiode, Zuordnungs-/Unsicherheitsbeleg und Herkunft je Zeile. TANN/PANN nicht umbenennen. |
| Probenbindung | Y und X im selben Objekt; gleiche Zeilenzahl und dokumentierte rollenweise Paarung. | Beide Matrizen haben **NULL-Zeilennamen**. Externe Originalproben-/Standort-/Quellkennungen fehlen; Zeilenposition allein trägt keinen externen Metadatenjoin oder Unabhängigkeitsnachweis. |
| Zeit/Messkontext | Allgemeine Beschreibung in M3. Im Objekt nur Y/X mit `names` bzw. `dim`/`dimnames`, keine weiteren Attribute. | Keine probenbezogenen Sammlungs-/Ablagerungszeiten, Integration, Unsicherheiten, Herkunftsgruppen oder eigener N-Vektor. 5–10 mm sind eine Materialtiefe, keine Integrationsdauer; ungefähr 30 Jahre definieren keine Kalendernormalperiode. |

Diese Abwesenheitsbefunde gelten genau für das bezeichnete Objekt und die
gelesenen Dokumentationsstellen. Sie behaupten nicht, dass anderswo keine
passende Herkunftstabelle existiert. Die veröffentlichte Countbeschreibung
bleibt ein Autorenbeleg; unser Strukturcheck validiert keinen Erfassungsprozess.
Keine Gleichheit mit EMPD2 oder fossilen Typkonzepten abgeleitet.

## Ergebnis und nächster Datenengpass

**Verfügbarkeit und Objektidentität sind für diesen Kandidaten geklärt.**
Der unmittelbar nächste Engpass ist die **belegte Zuordnung jeder Datenzeile
zu Originalprobe, Sammlung/Quelle und Klima-Zeitbezug**. Erst diese Verbindung
ermöglicht eine belastbare Prüfung von Abhängigkeiten und zeitlicher Eignung.
Null-/Nenner-/Typregeln bleiben zusätzlich offen. Der unveränderte
[Modellvertrag](./MODEL_CONTRACT.md#stopbedingungen-und-abschlussgrenze) gilt;
dieser Befund wählt E nicht und öffnet kein Gate.

Hier endet die Prüfung. Als genau ein Anschluss ist die begrenzte Suche nach
dieser **zeilenbezogenen RS10-Herkunfts-/Metadatentabelle** in HANDOFF vorbereitet;
keine Wiederholung der Objektprüfung und keine breite Ersatzdatensuche.
Furlanetto bleibt vorgemerkt; keine PhD-Neulesung oder unveränderten Abrufe.

Gesichert werden Bericht und kompakte Evidenz. Originaldaten und vollständige
Autorenquellen liegen nur im Sitzungscache; dauerhafte Vollablage nicht belegt.
Keine Matrix exportiert, Daten-/Taxon-/Klimazielwahl, Bereinigung, Nennerrechnung,
Implementierung, Tests, Fits, Scores, Simulationen, Kernläufe, Kontakte oder Merges.
Abnahme bestanden: sieben Textquellen mit Blob/SHA-256 und Literalzeilen,
identische Binärbytes an zwei Revisionen und Hash vor/nach Dekodierung;
20 Index-Pointer, sechs Evidenz-Blobs und 13 lokale Links/Anker geprüft.
Alte Indexeinträge und Statushistorie erhalten; übrige Worktrees sauber,
Diff auf fünf Dateien begrenzt. Keine unabhängige palynologische Prüfung.
Sicherung und CI stehen in HANDOFF/Status.
