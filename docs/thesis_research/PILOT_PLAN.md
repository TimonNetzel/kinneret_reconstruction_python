# Verbindlicher Ablauf für den Monticchio-Piloten

Vom Nutzer am 29.10.2026 im Chat bestätigt. Dieser Plan konkretisiert den
nächsten Pilotabschluss. Der [langfristige Multiproxy-Plan](../plan_multiproxy.md)
bleibt Hintergrund; seine Voraussetzung verfügbarer ursprünglicher
Taxon-Transferfunktionen ist durch die Shared37-Baseline nicht erfüllt.
Die verlorenen 279 Funktionen werden nicht als wiederhergestellt behandelt.

## Ziel und Zuständigkeiten

Ein reproduzierbarer, wissenschaftlich nachvollziehbarer Monticchio-Pilot
mit ausgewiesener Analogabdeckung und nachvollziehbarer Altersunsicherheit.
Die Gesamtvision Europa/0–150 ka wird erst nach diesem Pilot erweitert.

- **OpenAI:** Quellen, Zuordnungen, Rechenlauf, Unsicherheitsprüfung und
  unabhängiger Ergebnis-Audit.
- **Claude:** CLI-/App-Integration eines geprüften Datenvertrags. Seine
  Integrationsdateien werden hier nicht gleichzeitig bearbeitet.
- Jeder Arbeitsabschnitt endet mit einem konkreten Ergebnis und einer
  Entscheidung zum nächsten Meilenstein. Neue wissenschaftliche Blocker
  werden nicht durch still geänderte Abnahmekriterien übergangen.

## Meilensteine und Abnahme

| Phase | Ergebnis | Abnahme zum nächsten Schritt |
|---|---|---|
| **P1: Chronologie klären** | Quellenpaket, belegte Tiefen-/Probenzuordnung, Herkunft und Abhängigkeit der Anker, ausdrückliche Unsicherheitsdarstellung; Status jeder der 69 Proben | Prüfsummen und Einheiten geprüft; Tiefenskalen verbunden; direkte und übertragene Anker unterschieden; zulässige Teilmenge und Grenzen fachlich begründet. Aus 66 numerischen Kandidaten wird nicht automatisch eine freigegebene Teilmenge. |
| **P2: Diagnostischen Pilot berechnen und prüfen** | Anwendung der festgehaltenen Baseline auf die begründete Teilmenge; getrennte Prüfung von Analogabdeckung, Zähl-, Modell- und Altersunsicherheit; maschinenlesbare Ergebnisse und unabhängiger Audit | Eingaben und Annahmen gesperrt; reproduzierbare Ergebnisse; keine modernen Test-RMSE als fossile Fehlerbänder; keine gemeinsam abhängigen Alter aus unabhängigen Randziehungen erfinden. Fehlende Unsicherheitskomponenten bleiben ausdrücklich fehlend. |
| **P3: Geprüften Pilot integrieren** | Kompakter Integrationsauftrag für Claude mit Beispieldaten, Schema und geprüfter Referenz; Herkunft und Qualitätskennzeichnungen in der App | Wissenschaftlicher Inhalt mit Vertrag abgeglichen; Kennzeichnung der Altersgrenzen und Analogdiagnose korrekt; Integrationsprüfungen bestanden. |
| **P4: Schrittweise erweitern** | Weitere Standorte und unabhängige Proxys; später größere Zeitabdeckung | Erst nach Pilot-Abnahme; neue Kalibrierung, Chronologien und Abhängigkeiten jeweils separat prüfen. |

## Erreichter Stand

- [PR #18](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/18):
  moderne Baseline bereits berechnet und unabhängig geprüft.
- [PR #19](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/19):
  Claude-Integration dieser modernen Ergebnisse; geprüfter Head `c8977a5`.
- [PR #20](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/20):
  Analogdiagnose der 69 Proben und Prüfung der bisherigen Altersgrenzen.
- [PR #21](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/21):
  66 numerische Zuordnungskandidaten, drei offene Fälle, Archivpräzision und
  übertragene AMS-Anker. Die Kandidaten ersetzen keinen physischen Crosswalk.

Diese PRs sind beim erneuten Abgleich am 29.10.2026 weiterhin offen und ihre
Heads unverändert. Sie werden hier nicht ungefragt gemergt. Für neue Arbeit
gilt weiterhin: #18-Branch verwenden, solange er offen ist, sonst `main`;
vor jedem neuen Abschnitt den Stand erneut prüfen.

## Aktuelle Entscheidung: P1 wartet auf konkrete Eingaben

Die Supplementtabellen **A-1/A-2/A-3** zu
[Wulf et al. 2026, doi:10.1016/j.quascirev.2026.110116](https://doi.org/10.1016/j.quascirev.2026.110116)
sind weiterhin nicht als Originaldateien bezogen. Bei erneuten Zugriffen
liefern Verlag bzw. getestete Download-Kandidaten HTTP 403. Die getesteten
CDN-Adressen sind lediglich Kandidaten; 403 bestätigt weder ihre Existenz
noch Dateiendung oder Inhalt. Die Websuche bei Verlag, Autoreninstitution
und nach einer alternativen Datenablage hat keine tatsächlich abrufbare
Supplementdatei ergeben. Es wird nicht behauptet, dass sie nirgends existiert.

Auch eine bezogene A-3-Tabelle allein würde den Anschluss von LGM90D an
MON16 nicht beweisen: MON16 besitzt eine eigene Komposit- und eine für die
Modellierung von Tephren bereinigte Tiefenskala.

Der erste abzuarbeitende Eingang ist in
[CHRONOLOGY_DATA_REQUEST.md](./CHRONOLOGY_DATA_REQUEST.md) konkret beschrieben.
Die erfolglosen Zugriffe stehen in [pilot_chronology_gate.json](./pilot_chronology_gate.json).
Kein weiterer unveränderter Downloadversuch am selben gesperrten Endpunkt
ist ein neuer Forschungsfortschritt.

## Ablauf nach Eingang der Dateien

1. Originale unverändert sichern; Größe, SHA256, Quelle und Lizenz erfassen.
   Archivbestand auflisten und die tatsächlichen Tabellen-/Spaltennamen prüfen.
2. A-1/A-2/A-3 gegen Publikation und Einheiten abgleichen; historische,
   kalibrierte und modellierte Alter getrennt behandeln. Gemeinsame
   Modellziehungen und bloße Randintervalle ausdrücklich unterscheiden.
3. Tiefentransformation einschließlich Nullpunkten, Kernabschnitten und
   ausgeschlossener Tephramächtigkeit prüfen. Unklare Segmente und Proben
   bleiben gesperrt; keinen Crosswalk aus Klimakurven oder gewünschten
   zeitlichen Übereinstimmungen anpassen.
4. Je Probe und Anker eine begründete Entscheidung dokumentieren und
   unabhängig nachrechnen. P1 endet mit **weiter / begrenzt weiter nur nach
   ausdrücklich geänderter Zieldefinition / Monticchio als Pilot zurückstellen**.
5. Bei positiver P1-Abnahme folgt P2 mit der bestehenden Baseline, ohne neue
   Modellwahl anhand der Fossilproben. Claude erhält den Pilotauftrag erst
   nach der wissenschaftlichen Prüfung.

Sollten Quellen und Zuordnung nicht beschaffbar sein, wird eine explizite
Entscheidung über einen anderen Pilotdatensatz getroffen. Es beginnt keine
unbegrenzte Ersatzquellensuche. Die geprüfte moderne Baseline und die bisherige
App bleiben unabhängig davon nutzbar.
