# Übergabe für einen neuen Chat

Stand 30.10.2026. Alle unten aufgeführten PRs #18–#25 sind bei der Prüfung
offen, ihre Ergebnisse sind auf GitHub gespeichert. Vor neuer Arbeit diesen
Stand erneut prüfen. Dieser Index und [NEW_CHAT_PROMPT.md](./NEW_CHAT_PROMPT.md)
liegen in PR #25 auf `codex/netzel-epd-sample-identity`.

## Was gespeichert ist und wo

Fachliche Ergebnisse, Entscheidungen, Quellenregister, Prüfcode, abgeleitete
Tabellen und Prüfgrenzen sind in den PRs dokumentiert. Es handelt sich um
**getrennte Branches**, keine kumulative Folge: #19–#25 basieren jeweils auf
#18. Ein Checkout von #25 enthält daher nicht automatisch die Zusatzdateien
aus #20–#24. Diese gezielt über die folgenden festgehaltenen Commits lesen.
Der vollständige Chatverlauf und reproduzierbare Lauf-Zwischendateien sind
nicht eingecheckt und für die Fortsetzung nicht erforderlich.

| PR | Inhalt | Festgehaltener Ergebnisstand / Einstieg |
| --- | --- | --- |
| [#18](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/18) | Berechnete, unabhängig geprüfte moderne Baseline | [Bericht auf f9acc16](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/f9acc16c58a47a51b1063690a69a2db01381c76a/docs/thesis_research/baseline2025_2026_10_29/README.md) |
| [#19](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/19) | Claudes CLI-/App-Integration | Head `c8977a57fecabcef8f54abd7eafd6545f53487f9`; unabhängiges Review in #20 |
| [#20](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/20) | Analogabdeckung der 69 Proben, Altersdiagnose und Claude-Review | [Bericht auf d7f151a](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/d7f151a761eeb46d377f9323d559b40fd9ee083e/docs/thesis_research/monticchio_2026_10_29/README.md) |
| [#21](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/21) | Ältere Quellen-/Probenkorrespondenz, übertragene AMS-Anker | [Bericht auf 170d77f](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/170d77f2bdbbe60217dc8342f34beb252ccb67de/docs/thesis_research/chronology_crosswalk_2026_10_29/README.md) |
| [#22](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/22) | Vom Nutzer bestätigter Pilotplan P1–P4 | [Plan auf a710e86](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/a710e86dded84514d06d75d2769fe25dcdd5ec9e/docs/thesis_research/PILOT_PLAN.md) |
| [#23](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/23) | Audit der vom Nutzer hochgeladenen MON16-Supplemente | [Bericht auf 6b22422](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/6b22422f95856a7cf7cdc6d2aa2de3ed05fadbd7/docs/thesis_research/mon16_intake_2026_10_29/README.md) |
| [#24](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/24) | Sechs historische Markerketten, ETH-12464-Originalquelle | [Bericht auf 8f484ef](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/8f484ef0da80e329b4ff29ab128e1aa760d827d3/docs/thesis_research/marker_crosswalk_2026_10_30/README.md) |
| [#25](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/25) | EPD-Datenbankkennungen, exakte Tiefen und dieser Übergabeindex | [Audit auf 9aeafe4](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/9aeafe411d1dd27f4d87d037346d61a797b03e52/docs/thesis_research/epd_sample_identity_2026_10_30/README.md) |

Die Ergebnisberichte enthalten Quellen-/Ausgabe-Hashes und Reproduktion.
Die vollständige GitHub-CI auf den oben verlinkten Ergebniscommits von
#18–#21 und #23–#25 war erfolgreich (Python 3.11/3.13, jeweils Push und PR).
#22 ist ein Dokumentationsstand. Diese Aussage bezieht sich auf die genannten
Ergebniscommits, nicht pauschal auf spätere Änderungen oder neue Branchstände.

## Rohdaten-Ausnahme: Upload bitte behalten

Das Originalpaket `ScienceDirect_files_29Oct2026_22-15-14.667.zip`
(2.928.099 Bytes, SHA256
`2329fab3f4fc196174c70960cb9f1fddcbe9f9158ab099a3351193986f7ebdf8`)
wurde **nicht im Repository weiterverteilt**. Es enthält:

- `1-s2.0-S0277379126003252-mmc1.xlsx`: A-1/A-2/A-3.
- `1-s2.0-S0277379126003252-mmc2.xlsx`: B-1 bis B-8.
- `1-s2.0-S0277379126003252-mmc3.docx`: C-1/C-2.

Inventar, Datei-Hashes, Diagnosen und beide Leser sind in #23 gespeichert.
Eine bytegleiche Wiederbeschaffung vom Verlag wurde nicht nachgewiesen.
Für das Lesen der Ergebnisse ist kein Upload nötig; für eine erneute Prüfung
dieser Originaldateien das ZIP im neuen Chat wieder anhängen. Neue Serverbytes
gegebenenfalls als neuen Eingang behandeln, nicht mit dem Nutzer-ZIP gleichsetzen.
Andere öffentliche Eingänge sind über Quellenadressen und Hashes beschrieben;
auch dort werden heruntergeladene Roharchive/PDFs nicht vollständig eingecheckt.

## Fachlicher Stand

Die Baseline ist fertig; keine neue Modellentwicklung anstoßen. Claude betreut
CLI/App, OpenAI Datenrecherche und unabhängige Prüfung. Das
[Review von #19](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/d7f151a761eeb46d377f9323d559b40fd9ee083e/docs/thesis_research/monticchio_2026_10_29/INTEGRATION_REVIEW.md)
fand im geprüften Umfang keine blockierenden Codebefunde. Der echte CLI-Lauf
mit Originaldaten war referenzgleich. Die Browserdarstellung wurde hier wegen
fehlendem Chromium nicht unabhängig ausgeführt. Unveränderte Köpfe nicht ohne
neuen Anlass erneut vollständig prüfen.

Der akzeptierte Ablauf lautet P1 Chronologie/Provenienz/Tiefenzuordnung →
P2 diagnostischer Fossilpilot mit fester Baseline → P3 Claude-Integration →
P4 weitere Standorte/Proxys. **P1 ist noch offen**, P2 wurde nicht still freigegeben.

- Analogdiagnose für alle 69 Monticchio-Proben liegt vor; gleiche
  Standortkoordinaten wurden aus den modernen Kandidaten ausgeschlossen.
  Distanzflags sind keine validierten Nichtanalog-Grenzen.
- Alle 69 EPD-Datenbankkennungen (Entity 932, Proben 4–72) und exakten
  Mittelpunkttiefen sind bestätigt; acht gerundete Pilot-Exporttiefen weichen
  um 5 mm ab. Die gesamte 245×135-Zahlenmatrix stimmt mit dem Archiv überein.
  Datenbankkennungen sind keine ursprünglichen physischen Probenetiketten.
- Die 66/69 älteren Prozent-Fingerprint-Zuordnungen aus #21 bleiben ein anderer
  Vergleich. Die drei dort offenen Fälle werden durch #25 nicht aufgelöst.
- MON16 A-3 hat 13.024 Zeilen mit marginalen Alterszusammenfassungen, keine
  gemeinsamen Posterior-Kurven. Mercato-/Tiefen- und Quellenkonflikte bleiben
  dokumentiert; keine stille Korrektur oder Altersübertragung.
- Sechs historische Markerketten und die Herkunft von ETH-12464 sind belegt.
  Das ergibt noch keine physische Transformation der 69 Proben auf MON16.
- EPD-Tabellen für Kernabschnitte/Bohrsegmente sind für Entity 932 leer.
  Alte Standard-14C-Alter nicht anstelle der beibehaltenen Cut-Alter einlesen.

## Nächster konkreter Eingang und Arbeitsregeln

Vorrangig fehlt die ursprüngliche Proben-/Kompositierungsliste, die diese
Datenbankproben mit physischen Kernen, Segmenten und Abschnitten verbindet,
einschließlich Tiefenbezug und Korrelationsunsicherheit. Genauer Bedarf:
[MISSING_SOURCE_REQUEST.md](./epd_sample_identity_2026_10_30/MISSING_SOURCE_REQUEST.md).
Daneben MON16-Quellenkonflikte und für volle Altersfortpflanzung das originale
Modellprojekt bzw. zusammengehörige Alterskurven klären. Die bereits leeren
Archivtabellen und erfolglosen Abrufe nicht unverändert erneut durchsuchen.
Für weitere Suche einen neuen, konkreten Quellenansatz benennen; wenn dafür
Originalmaterial von Datenhaltern nötig ist, den fehlenden Eingang klar sagen.

Keine Autorenanfrage wurde versandt. Keine Merges wurden beauftragt.
Keine fossile Klimakurve oder fossilen Unsicherheitsintervalle berechnet.
Moderne RMSE sind keine fossilen Fehlerbänder; marginale Altersgrenzen nicht
unabhängig ziehen und als zeitliche Altersensembles ausgeben.

Für neue Forschungsarbeit den Branch von #18 verwenden, solange dieser offen
ist, sonst `main`; benötigte Forschungsergebnisse gezielt aus den anderen PRs
lesen/übernehmen. Eigener Branch und PR, keine gleichzeitige Bearbeitung von
Claudes Integrationsdateien. Keine temporären Dateien oder Erinnerungen aus
dem alten Chat voraussetzen. Kurz im Chat berichten, ausführlich im Repository.
