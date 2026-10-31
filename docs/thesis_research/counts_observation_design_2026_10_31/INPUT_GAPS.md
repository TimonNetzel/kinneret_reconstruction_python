# Eingangsvertrag: verfügbar, angenommen und weiterhin gesperrt

31.10.2026. Nur gezielte Repository-Dokumente gelesen; keine Originalcounts,
Modellkoeffizienten, Areal-/Klimaraster oder neuen externen Quellen ausgewertet.
Die Tabelle ist eine Lückenprüfung für das Design, keine neue Dateninventur
und kein Quellennachweis durch bloßes Wiederholen alter Referenzen.

| Schritt / Größe | Vorhandener belegter Kontext | Fehlender realer Eingang / Konsequenz |
| --- | --- | --- |
| Kartiertes Areal m_j(c) | [Design](../taxon_transfer_2026_10_30/DESIGN.md), [config](../taxon_transfer_2026_10_30/config.json): sieben Arten, Kartenindikatoren unter festem Gebiet, WorldClim 1970–2000 | Keine lokale Präsenz-/Abundanzfunktion. Keine Auswahl einer Varianten-/Kalibrierungskombination für Pollencounts anhand alter äußerer Scores. |
| Lokale Vegetation G(V\|c) | Kein in diesem Paket belegtes gemeinsames lokales Vegetationsmodell. #39 benutzt vier anonyme Bernoulli-Kanäle, nicht lokale Pflanzenmengen. | Verknüpfung Karte→lokales Vorkommen→Vegetationsanteile einschließlich Standort, sonstiger Quellen und gemeinsamer Abhängigkeit. Ohne proper G keine ausführbare reale Counts-Likelihood. |
| Produktion r | Im vorigen binären Spielzeugmodell kein Pollenproduktionsmodell | Unabhängige relative Produktionsbelege mit Referenzmaß, Taxon-/Typumfang und Geltungsbereich. Freie r können ein anderes Klima vollständig kompensieren; Counts allein identifizieren diese Größen nicht. |
| Transport B, Erhaltung d | Nicht durch kartierte Areale oder #32-Kalibrierungsbänder beschrieben | Lokaler/Ferntransport, Erhaltungs-/Detektionsprozess und Einheiten; sonst Verwechslung von lokalem Vorkommen mit eingetragenem Pollen. Ein pauschal gesetztes d=1/B=0 wäre eine Annahme, kein nachgewiesener Eingang. |
| Art→Typ A | [Literaturgrenzen](../pollen_methods_2026_10_31/README.md), [präzise Quellenlücke](../pollen_methods_2026_10_31/SOURCE_GAP.md) | W96-Methodenteil und tatsächlich angewandte Bestimmungskette fehlen. Sieben datensatzbezogene Artbrücken weiterhin offen; weder akzeptierter EPD-Name noch EMPD2-Link liefern eine numerische Klassifikationsmatrix. |
| Typgemische / andere Quellen U | Sechs Fossilkommentare auf Gattungsebene, ein akzeptierter Taxus-Artkommentar; Herkunft/Bestimmungsumfang weiter offen | Vollständiger biologischer Umfang je beobachtetem Typ. Nicht modellierte Arten können zu ausgewählten Typen beitragen; nicht automatisch nur „Rest“. Keine Gattungscounts auf einzelne Artfunktionen verteilen. |
| Counts y, Nenner N, Rest R | Bestehende Imports/Originaldefinitionen sind im Repository dokumentiert, in diesem Designpaket nicht neu geprüft | Probenbezogener, belegter ganzzahliger Nenner und Partition einschließlich ausgeschlossener/fehlender Kategorien. Keine Behauptung, dass sieben Felder einen geschlossenen Pollennenner liefern. Keine Prozente zu scheinbaren Counts umrechnen. |
| Stichprobenmodell | Multinomial als ausdrücklich synthetische Minimalannahme mathematisch festgelegt | Reale Klassifikations-/Zählregel, Replikate und ggf. überdisperse Fehlerstruktur. N unabhängige Körner sind eine Annahme; N ist keine Anzahl unabhängiger Klimareplikate. |
| Klima und Support | WorldClim-TANN/PANN und Hüllensperren sind im Artenfunktionsvertrag definiert | Gemeinsame Hülle aller tatsächlich benötigten Modelle, zeitlich/räumliche Transportierbarkeit, Vegetations-/Pollenbrückensupport und priorer Klimabezug. Eine Hülle allein legitimiert keine fossile Anwendung. |
| Probe–Tiefe–Alter | [Roadmap T3](../TAXON_PILOT_ROADMAP.md), [Monticchio-Quellenbedarf](../variable_provenance_2026_10_30/SOURCE_GAP.md) | Physischer LGM90D→MON16-Anschluss einschließlich Korrelationsunsicherheit weiterhin offen; 28 Kandidaten/41 Nichttreffer/47 Herkunftsfragen sind keine Freigabe. Keine neue Kernwahl. |
| Unsicherheit | #32 als bedingte Kalibrierungssensitivität dokumentiert; hier nicht neu ausgewertet | Gemeinsame Transfer-/Typ-/Produktions-/Transport-/Chronologieunsicherheit und unabhängige Bestätigung. Fehlende Unsicherheit nicht auf 0 setzen und moderne Scores nicht in fossile Fehlerbänder umdeuten. |

## Wodurch die Mehrdeutigkeiten eingeschränkt werden könnten

Benötigt werden **unabhängige** Daten oder begründete feste Beschränkungen zu
den offenen Richtungen: Materialbestimmung für A, lokale Vegetation/Präsenz für
G, relative Produktion/Transport/Erhaltung im passenden Referenzsystem und ein
prüfbarer Nenner. Ein Referenztaxon kann eine globale Skalenkonvention setzen,
beweist aber keine bekannten relativen Faktoren. Eine Summe-1-Beschränkung der
Vegetation identifiziert weder r noch unbekannte Typgemische.

Eine moderne gemeinsame Vegetations-/Pollenprobe mit geklärter Bestimmung könnte
relevante Zusatzinformation enthalten; welche Beobachtungen welche Parameter
tragen, muss in einem eigenen Datenrollen-/Quellendesign geprüft werden.
Hier keine Behauptung, dass ein solcher passender Datensatz vorhanden oder
freigegeben sei. Schon für moderne Daten wären Quellen-/Regionenrollen vor einer
Bestätigung zu trennen; bekannte äußere Testregionen sind keine neue Evidenz.

Proper informative Priors können einen rechnerisch eindeutigen Posterior erzeugen.
Das muss als priorabhängige/bedingte Sensitivität ausgewiesen werden, solange
die Counts selbst die betreffende Richtung nicht identifizieren.
Mehr Pollenkörner beheben keine strukturelle Beobachtungsäquivalenz.

## Freigabestatus

- **In diesem Paket erledigt:** mathematischer Kandidatenvertrag, explizite
  Gegenbeispiele für Skala, r/V/Klima, Typmischung und Hintergrund; Trennung von
  Nullcount/fehlend/Rest; fester Plan für einen zukünftigen anonymen Zähltest.
- **Vorbereitet, nicht ausgeführt:** der 52-Posterior-Test in SYNTHETIC_PLAN.md.
  Keine Ergebnisdateien, keine geschätzten Parameter und keine neue SBC-Aussage.
- **Separat blockiert:** W96-Methodenteil/Referenzen und datensatzbezogene
  Artbrücken; Monticchio-P1 und reale T0/T1/T3/T4 bleiben offen.
- **Unverändert:** sieben Modelle und alle bisherigen Resultate/Quellregister,
  Fossilcounts/Gruppierungen, Analog-Baseline und CLI/App. Keine Autorennachricht,
  kein Merge und keine echte Pilotwahl aus diesem Design.
