# Sicherung und Zusammenführung #18–#29

Stand 30.10.2026. Ausdrücklicher Nutzerauftrag: „Bitte lade alles hoch und
merge es so dass die sachen aus dem chat nicht verloren gehen.“ Damit ist der
technische Merge dieser Arbeiten beauftragt; fachliche P1-/Pilotfreigaben
werden dadurch nicht erteilt. Aktueller Einstieg: [HANDOFF](../HANDOFF.md).

## Erhaltene Arbeiten

#18 wurde zuerst mit Merge-Commit auf `main` übernommen. Die übrigen offenen
PRs wurden auf `main` umgestellt und ihre Original-Heads im Integrationsbranch
zunächst lokal mit Merge-Commits integriert. Veröffentlicht wird derselbe
geprüfte Dateibaum als gemeinsamer Merge-Commit mit `main` und den elf
ursprünglichen PR-Heads als Eltern. Kein Squash, kein Force-Push, keine Löschung
der Ergebnisbranches. Alle zwölf ursprünglichen Heads sind Vorfahren des
Integrationsstands. Konflikte betrafen ausschließlich AGENTS.md,
PROJECT_STATUS.md und HANDOFF.md; zunächst wurde jeweils der jüngste
Paketstand übernommen und danach eine gemeinsame aktuelle Übergabe erstellt.

| PR | Arbeit | Ursprünglicher gesicherter Head |
| --- | --- | --- |
| [#18](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/18) | Netzel: moderne Pollen-Baseline räumlich prüfen und Claude auf Integration begrenzen | `f9acc16c58a47a51b1063690a69a2db01381c76a` |
| [#19](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/19) | Netzel: moderne Shared37-Baseline in CLI und App einbinden | `c8977a57fecabcef8f54abd7eafd6545f53487f9` |
| [#20](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/20) | Netzel: Monticchio-Analogabdeckung und Chronologie unabhängig prüfen | `d7f151a761eeb46d377f9323d559b40fd9ee083e` |
| [#21](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/21) | Netzel: Monticchio-Quellenzuordnung und übertragene AMS-Anker prüfen | `170d77f2bdbbe60217dc8342f34beb252ccb67de` |
| [#22](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/22) | Netzel: bestätigten Pilotplan und Chronologie-Abnahme festhalten | `a710e86dded84514d06d75d2769fe25dcdd5ec9e` |
| [#23](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/23) | Netzel: hochgeladene MON16-Chronologie prüfen und Quellenkonflikte belegen | `6b22422f95856a7cf7cdc6d2aa2de3ed05fadbd7` |
| [#24](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/24) | Netzel: sechs Tephra-Markerketten und ETH-12464-Herkunft belegen | `8f484ef0da80e329b4ff29ab128e1aa760d827d3` |
| [#25](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/25) | Netzel: EPD-Probenkennungen und exakte Monticchio-Tiefen unabhängig prüfen | `00c9f037642470c7d1a9a44e488f062109292c2d` |
| [#26](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/26) | Netzel: 28 Monticchio-Probenkandidaten mit archivierten Abschnittsnamen prüfen | `4c9f1150e766a7f855dc9e28fd121638a710ff9a` |
| [#27](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/27) | Netzel: ursprüngliche Variablen prüfen und 47 Monticchio-Differenzen eingrenzen | `5fd3bad46d5108e8986c054171fba66a05c26589` |
| [#28](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/28) | Netzel: sieben botanische Arten-Transferfunktionen trainieren und räumlich prüfen | `8f02794434f561f5b30228360247935f3c025cc8` |
| [#29](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/29) | SMOTE + einfaches Netz: bessere Klassifikation, schlechtere rohe Wahrscheinlichkeiten | `833e566df062a1ee9f215b2e0db90f5ef08d1a48` |

[Maschinenlesbarer Dateinachweis](./preservation.json): 183 unterschiedliche
von den zwölf PRs bearbeitete Pfade. Vor der zentralen Dokumentationsfortschreibung
entsprachen alle 183 exakt ihrem jeweils jüngsten ursprünglichen PR-Blob.
Danach bleiben 174 Dateien bytegleich; neun Einstiegs-/Statusdateien erhalten
explizite Aktualisierungen. Sämtliche früheren Fassungen dieser neun Dateien
bleiben durch die übernommenen Originalcommits lesbar. Wissenschaftlicher Code,
Design, Parameter, Tabellen, Grafiken, Quellregister und Auditberichte bleiben
unverändert. Die Original-PR-Prüfungen waren sämtlich erfolgreich.

## Gesicherter Chatinhalt und fachliche Bewertung

Der gemeinsame HANDOFF hält die erledigten Pakete, Modellvarianten, Zahlen,
Einschränkungen, Nutzerentscheidungen und nächsten Arbeitsschritte fest.
Er enthält keine Zusage einer fertigen Multiproxy-Fossilrekonstruktion.

Der Vergleich ist als reproduzierbarer Forschungszwischenstand überzeugend:
SMOTE steigert die Balanced Accuracy bei Schwelle 0,5 für alle sieben Arten in
beiden räumlichen Prüfungen auf etwa 90 %. Zugleich verschlechtert es bei allen
14 Art×Block-Fällen den Brier gegenüber demselben Netz ohne SMOTE. Für die
Klimarekonstruktion ist deshalb die Wahrscheinlichkeitskalibrierung offen.
Arten-/Pollentyp-Zuordnung und Unsicherheit sind weitere notwendige Arbeiten.

P1 bleibt offen: physische LGM90D→MON16-Zuordnung samt Korrelationsunsicherheit.
28 Kandidaten, 41 native Nichttreffer und 47 offene Provenienzfragen bleiben
unverändert. Technischer Merge ersetzt keine Quellenklärung oder Pilotabnahme.

Rohdaten, große reproduzierbare Laufprotokolle und Originalpublikationen waren
bewusst nicht Bestandteil der PRs und werden nicht neu weiterverteilt. Die
vorhandenen Ergebnisdateien, Parameter, Eingangs-/Ausgabehashes und vollständigen
Reproduktionswege werden erhalten. Alte temporäre Chatdateien sind nicht nötig.
Für erneute Original-ScienceDirect-Rohdatenprüfung muss das Nutzer-ZIP erneut
angehängt werden. Es ist keine Volltranskription des Chats und keine zusätzliche
Archivierung ursprünglich nicht veröffentlichter Rohdateien behauptet.

## Gemeinsame Prüfung

Der Dateierhalt und die Abstammung aller zwölf Original-Heads sind geprüft.
Gemeinsame Prüfung unter Python 3.12: 74 Repositorytests erfolgreich, zwei
Browserprüfungen zunächst durch ein nicht verfügbares `/tmp` blockiert; Wiederholung
mit beschreibbarem TMPDIR. Zusätzlich 52 Werkzeug-/GIS-/Metadatenprüfungen
erfolgreich. GitHub-CI und abschließende Browserergebnisse werden vor dem
Merge im Integrations-PR festgehalten. Die früheren
vollständigen Trainings-/Quellenaudits werden hier nicht neu gerechnet;
ihre unveränderten Prüfnachweise bleiben gespeichert.
