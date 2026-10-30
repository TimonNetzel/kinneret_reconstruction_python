# Netzel-Projekt: kurzer Arbeitsstand

Stand 30.10.2026. Vor Fortsetzung PR-/Merge-Stand erneut prüfen.

**Ziel:** App für taxonbasierte Multiproxy-Klimarekonstruktionen europäischer
Fundstellen mit belegten Daten und Unsicherheiten.

| Bereich | Stand |
| --- | --- |
| Moderne Analog-Baseline #18 | Fertig und unabhängig geprüft; weiterhin offene Arbeitsbasis. Keine neue Entwicklung. |
| Monticchio-P1 | Offen. 69 Datenbankkennungen bestätigt; physische LGM90D→MON16-Zuordnung und Korrelationsunsicherheit fehlen. |
| Native Korrespondenz aus #26 | Unverändert 28 numerische Kandidaten mit Abschnittsnamen; 41 Proben ohne vollständigen Treffer, 47 Differenzzellen. |
| Neue Variablenprüfung | Fertig: ursprüngliche EPD-Definitionen und 483 Zellen in drei Quellen geprüft; 35 Plantago-Bilanzen und neun Digitalis-Zahlengleichheiten an festen Diagnosepaaren belegt. Keine Umcodierung als bewiesen akzeptiert. |
| Taxon-Transferfunktionen | Noch neu zu erstellen und zu prüfen; die ursprünglichen 279 Funktionen sind nicht wiederhergestellt. |
| Fossile Klimarekonstruktion / Multiproxy-Gesamtdurchlauf | Noch nicht berechnet bzw. abgeschlossen. P2 nicht freigegeben. |
| CLI/App | Claude; Codex bearbeitet Quellen und unabhängige Prüfung. |

**Fertig bei Codex:** [Variablenbericht](./variable_provenance_2026_10_30/README.md),
alle 69 festen Paare und 47 erhaltenen Differenzen, Quellen-/Hashnachweise,
zwei bytegleiche Produzentenläufe, separate Prüfung und vier neue Tests.
Eigener Branch `codex/netzel-variable-source-audit`; PR-Verweis folgt beim
Veröffentlichen im selben Arbeitspaket.

**Aktuell offen:** Zwei Abies- und eine Sambucus-Zelle bleiben numerisch
unerklärt. Die Entstehung **aller 47** Differenzen ist ohne ursprüngliche
Einreichungs-/Umcodierungsbelege nicht bestätigt; auch die 44 Zahlenmuster
sind keine Quellenkorrektur. MON16-Felder bleiben leer.

**Als Nächstes:** [konkreten Einreichungsbeleg](./variable_provenance_2026_10_30/SOURCE_GAP.md)
für Neotoma 26608/47525 beschaffen: Originalvariablen/Regeln,
datensatzbezogene Synonymie, Einreichungshistorie und Zählnotizen. Danach
historische Abschnitts-/Kompositierungstabellen und MON16-Korrelation samt
Unsicherheit. Keine unveränderte Wiederholung ausgeschöpfter Abrufe.

PRs #18–#26 beim Arbeitsbeginn offen und ungemergt; #19 unverändert auf
`c8977a5`. Die [Übergabe](./CHAT_RESTART_VARIABLES.md) verlinkt die getrennten
Stände aus #25/#26. Bestehende Baseline, Taxonomie, Alter und Claudes Dateien
unverändert. Keine Pilotfreigaben, Autorennachrichten oder Merges. Das
Original-ScienceDirect-ZIP ist nur für erneute Original-Rohdatenprüfungen nötig.
