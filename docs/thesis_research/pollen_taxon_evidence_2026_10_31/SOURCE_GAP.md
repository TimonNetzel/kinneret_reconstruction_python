# Präziser Quellenbedarf nach der Sieben-Arten-Inventur

## 1. Originale frisch gegen die bestehenden Hashes prüfen

Die erwarteten Bytes und URLs stehen in [sources.json](./sources.json), nicht
in alten temporären Dateien suchen. Bei Abweichung stoppen und Ursache prüfen;
keinen Referenzhash ersetzen. Keine kompletten Zähltabellen im Chat ausgeben.

- **Monticchio PANGAEA.739452:** die sieben Parameterdefinitionen einschließlich
  Abkürzung, originalem Parameternamen, Archivkommentar, Bestimmungsnotizen und
  eventuellem Bestimmungsschlüssel prüfen. Schwerpunkt `Tax [#]`: Ist die
  Bezeichnung T. baccata ein ursprünglicher Bestimmungsbefund oder eine spätere
  akzeptierte Katalogbezeichnung? Archivkommentar allein klärt das nicht.
- **EMPD2 PANGAEA.909130 / EMPD2.sqlite3:** `p_vars` für sämtliche passenden
  Originalnamen/Untertypen und deren akzeptierte Konzepte samt Notizen auslesen;
  `var_`, `original_varname`, `acc_var_`, `acc_varname`, `groupid` getrennt erhalten.
  Nicht nur die sieben Kandidaten-IDs auswählen: diese sind keine vollständigen
  Gattungs-/Untertypenpools. Herkunft, Bedeutung und etwaige Alias-/Synonymregeln
  am Original dokumentieren, noch keine Counts aggregieren.
- **Zumaque et al. 2025, S1 / mmc1.docx:** die vollständigen Zeilen zu PABIE,
  BBETU, BCORY, FFAGU, PPICE, TTAXU und TTILI direkt prüfen, Seiten-/Tabellenzeile
  und originale Namen erhalten. Das finale XLSX-Blatt
  `1) EMPD2_90taxa_7634sites_cnts` auf die Gruppenkopfzeilen prüfen.
  Alte SQLite-IDs nicht in diese neue Matrix hineininterpretieren.

Für die ersten beiden Zugriffe ist zunächst die Netzfreigabe für
`doi.pangaea.de` und `store.pangaea.de` erforderlich; CONNECT wurde abgewiesen.
Die späteren Originalquellen verwenden `download.pangaea.de` und
`www.ebi.ac.uk` (Zugriff hier noch nicht getestet). Weiterleitungen anhand der
tatsächlichen Antwort prüfen, keine unbekannten Domains oder Zugangsdaten erfinden.
Kein Kontakt zu Autoren beauftragt.

## 2. Artspezifische Belege, nicht aus Areal oder Scores ableiten

| Offener Fall | Benötigter Beleg | Was nicht genügt |
| --- | --- | --- |
| G-abies_alba | Dokumentierter Bestimmungsumfang von Abies im konkreten fossilen Datensatz; geeigneter Schlüssel/Atlas mit Merkmalen, Verwechslungsmöglichkeiten und geographisch-zeitlicher Geltung für A. alba | Gattungslabel oder bloße gegenwärtige Arealüberlappung; andere PABIE-Mitglieder nicht still verwerfen |
| G-betula_pendula | Bestimmungs-/Synonymiebeleg zu B. pendula und der tatsächlich verwendeten Betula-Untergliederung; Originalcodebezug | B. pubescens-type oder Gattungswert allein; B. pendula nicht aus fehlendem Namenseintrag als abwesend folgern |
| G-corylus_avellana | Nachweis, warum das konkrete Corylus-Pollenkonzept genau C. avellana repräsentieren darf; Ausschluss oder Behandlung anderer möglicher Taxa | Nur ein publiziertes Gruppenmitglied mit dem Namen Corylus |
| G-fagus_sylvatica | Bestimmungsumfang des fossilen Fagus-Feldes; Artabgrenzung und Quellenbeleg für beteiligte Namensverwendungen | Gleichsetzung eines Gattungsnamens mit F. sylvatica; die botanische Schreibweise Fagus silvatica allein |
| G-picea_abies | Definition und Diagnose von Picea abies-type sowie Nachweis getrennt erhaltener Fossiluntertypen, falls diese Grundlage der Artzuordnung sein sollen | Moderne Gruppe PPICE oder der darin vorkommende Untertypname; Gruppensumme nicht rückwärts zerlegen |
| G-taxus_baccata | Ursprüngliche Definition/Bestimmung und Synonymiehistorie von Tax [#] → Taxus baccata samt morphologischer und regionaler Reichweite | Übereinstimmende akzeptierte Artnamen in Archivkommentar und Datenbank |
| G-tilia_cordata | Definition von Tilia cordata- type gegenüber T. platyphyllos- type, Bestimmungsbeleg und getrennte Originalzählung im fossilen Material | Breites Tilia-Feld oder die zusammengeschlossene moderne TTILI-Gruppe |

Für jedes Dokument exakte Fundstelle, untersuchte Namen/Merkmale, Aussageumfang,
DOI/Version und soweit rechtlich/technisch möglich Prüfsumme sichern. Ein
Schlüssel kann Bestimmbarkeit stützen, ersetzt aber keinen Nachweis, wie die
konkreten Originalproben tatsächlich bestimmt wurden. Falls die Artbrücke nicht
belegt werden kann: den breiteren Pollentyp ausdrücklich erhalten und die
Nichtzuordenbarkeit dokumentieren, keine Ersatzregel erfinden.

## Nächstes begrenztes Paket

Zuerst die sieben Monticchio-Originaldefinitionen und die vollständigen
EMPD2-Namensketten beschaffen und mit der jetzigen Inventur abgleichen; Taxus
als explizite Namens-vs.-Bestimmungsfrage markieren. Für alle sieben Original-
und akzeptierte Namen/IDs sowie Notizen mit präzisen Fundstellen sichern.
Anschließend gezielt die nötigen morphologischen Primärbelege recherchieren.
Nicht automatisch eine Art priorisieren, weil ihr äußerer Score besser ist.

Abschluss: Original-Namensketten und Zugangslücken je Art prüfbar gesichert,
keine automatische neue Zuordnung. Modelle, Gruppen, Fossilcounts und CLI/App
bleiben unangetastet. Vollständige Unsicherheits-/Beobachtungsmodellierung,
weitere Arten/Areale und physischer MON16-Anschluss sind getrennte Arbeitspakete.
