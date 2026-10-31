# Pollentyp–Taxon: Bestandsaufnahme für sieben Arten

31.10.2026. Basis `6f35df1cb751a2e516e75b94c1cb3bcb05050a9f` auf dem
vollständigen Forschungsstand `9ee2f84`. Abgegrenztes Quellenpaket:
vorhandene Belege inventarisieren, Namensräume und Aussageebenen trennen,
fehlende Artbrücken benennen. **Keine neue taxonomische Zuordnung freigegeben.**

## Ergebnis

[Sieben prüfbare Datensätze](./inventory.csv) mit je botanischem Modellziel,
Fossilspalte/Archivkommentar, bisherigem EMPD2-Katalogkandidaten, publiziertem
2025-Gruppencode, Namensbestandteilen und Beleg-IDs. Die 21 Fundstellen in
[sources.json](./sources.json) nennen Dateipfad und CSV-Zeile oder JSON-Pointer
am festen Basiscommit; elf Eingangsdateien sind zusätzlich SHA-256-gesperrt.
Das sind überprüfte Repository-Belege, keine frisch gelesenen Originalarchive.

| Modellart | Fossilspalte / erhaltener Archivkommentar | Belegter Namensbezug | Fehlende Artbrücke |
| --- | --- | --- | --- |
| Abies alba | `Abi [#]` / Abies | Gattungsfeld; EMPD2-Kandidat 321; 2025 PABIE | Gattung nicht als A. alba verwenden; Gruppierungsbeleg nennt auch andere Arten |
| Betula pendula | `Bet [#]` / Betula | Gattungsfeld; Kandidat 12; BBETU | B. pendula steht nicht unter den erhaltenen S1-Mitgliedsnamen dieser Gruppe; keine Absenzbehauptung, kein Synonym zu B. pubescens-type |
| Corylus avellana | `Cor [#]` / Corylus | Gattungsfeld; Kandidat 31; BCORY | Ein einziges S1-Mitglied mit Gattungsnamen ist kein Einzelartbeleg |
| Fagus sylvatica | `Fag [#]` / Fagus | Gattungsfeld; Kandidat 42; FFAGU | Dasselbe Problem; botanische Originalschreibweise Fagus silvatica bleibt zusätzlich erhalten |
| Picea abies | `Pic [#]` / Picea | Gattungsfeld; Kandidat 126; PPICE enthält u. a. Picea abies-type | Ein Untertypname innerhalb der breiten Gruppe belegt weder exakte Artgleichheit noch eine erhaltene getrennte Fossilzählung |
| Taxus baccata | `Tax [#]` / Taxus baccata | **Artname im Archivkommentar** und EMPD2-Kandidat 111; TTAXU mit Mitglied Taxus baccata | Stärkerer Namensbeleg, aber kein hier geprüfter morphologischer Bestimmungs-/Zählbeleg für die fossile Artzuweisung |
| Tilia cordata | `Til [#]` / Tilia | Gattungsfeld; Kandidat 187; TTILI enthält u. a. Tilia cordata- type | Gruppe enthält auch Tilia und Tilia platyphyllos- type; Artzuweisung und getrennte Originalzählungen nicht belegt |

Bilanz: **7/7 Namensbezüge dokumentiert, 6/7 Fossilkommentare nur auf
Gattungsebene, 1/7 mit Artname; 0/7 neue fossile Artmodell-Zuordnungen genehmigt.**
Die sechs breiteren Konzepte sind nicht automatisch biologisch unbestimmbar;
es fehlt in diesem Nachweis die begründete Verengung auf genau die Modellart.
Für alle sieben bleibt diese Brücke offen. Die Artenliste stammt aus der festen
#28-Konfiguration, nicht aus äußeren Modell-Scores.

## Was die vorhandenen Belege tatsächlich sagen

- `fossil_decisions.csv` ordnet alle sieben Felder mit `include` nur einem
  **fossilen Arbeitskatalog** zu; `scope=fossil_catalogue_only` und
  `scientific_status=not_training_approved` bleiben unverändert. Der Grund
  `exact_archive_and_modern_accepted_concept` dokumentiert einen damaligen
  Katalognamensabgleich, keine unabhängig nachgewiesene Artgrenze.
- Die Kandidatenzahlen stammen aus dem alten EMPD2-Namensraum akzeptierter
  Konzepte. Sie sind **keine vollständigen Originalcode-Pools**, keine BfN-IDs
  und keine Neotoma-IDs. Insbesondere nicht von der Zahl auf Identität schließen.
  Der getrennte historische EPD-Auszug führt Abies als `var_ = 1`, während der
  EMPD2-Katalogkandidat 321 lautet. Auch BfN-Abies trägt 1 – kein gemeinsamer Schlüssel.
- Die 2025-Gruppen gehören zu einer anderen Kohorte/Version. Die unveränderte
  Shared37-Baseline verwendet sie als breitere Vergleichsmerkmale. Das begründet
  keine Anwendung einer einzelnen neuen Artfunktion auf diese Gruppen.
- `member_examples` im alten 2025-Nachweis wird im Produzenten auf acht Namen
  begrenzt (`prepare_shared2025.py`). Für diese sieben Gruppen liegen die
  gespeicherten Anzahlen zwischen eins und fünf und entsprechen jeweils der
  Zahl der getrennten Namen. Hier fehlt also kein Name allein wegen dieses
  Limits; die Vollständigkeit des ursprünglichen Supplements ist dadurch
  **nicht erneut unabhängig geprüft**. `published_member_names` bewahrt dessen
  gespeicherte Schreibweisen, etwa `Abies marocana`, ohne stille Korrektur.
- Ein `-type` oder `sect.` ist nicht automatisch synonym zu einer botanischen
  Art. Gattungs-/Gruppensummen lassen die ursprünglichen Untertypen nicht ohne
  weitere Daten zurückgewinnen. Keine Pollensumme oder Beobachtungsregel geändert.

## Quellenzugang und Reichweite

Originalregistrierungen: Monticchio PANGAEA.739452, EMPD2 PANGAEA.909130,
2025-Matrix PANGAEA.973997 und Supplement S1 (Zumaque et al., 2025,
DOI 10.1016/j.dib.2025.111657). URLs, übernommene historische Hashes/Größen,
Lizenzangaben und benötigte Inhalte stehen in sources.json. Keine Rohdatei
wird neu weiterverteilt; die ursprünglichen Lizenzbedingungen bleiben maßgeblich.

Die beiden ersten Downloads wurden am 31.10.2026 über den vorgesehenen Proxy
geprüft: **CONNECT HTTP 403** für `doi.pangaea.de` und `store.pangaea.de`.
Es fand kein Download und deshalb keine neue Originalbyte- oder Headerprüfung
statt. Die 2025-Matrix und das S1-DOCX wurden in diesem Paket nicht angefordert.
Kein alter temporärer Rohdatenbestand wird vorausgesetzt. Die im alten
`source_evidence.json` referierte EMPD2-Methodenarbeit warnt vor der Gleichsetzung
akzeptierter Namen mit identischen Bestimmungs-/Zählkonzepten; sie wurde hier
nicht frisch bezogen und liefert keine sieben artspezifischen Bestimmungsnachweise.

**Abgeschlossen:** Repository-Inventur und Nachweiskettenprüfung.
**Offen/blockiert:** Originalprüfung und artspezifische morphologische Brücke;
präzise Anforderungen in [SOURCE_GAP.md](./SOURCE_GAP.md). Kein vollständiger
Literaturreview oder Beweis, dass passende Belege grundsätzlich nicht existieren.

## Prüfung und Reproduktion der Bestandsprüfung

Aus der Repository-Wurzel ohne Netz oder wissenschaftliche Neuberechnung:

```bash
python docs/thesis_research/pollen_taxon_evidence_2026_10_31/verify.py
```

Der Prüfer liest die gespeicherten Quellen direkt, prüft sieben eindeutige
Modellziele, alle übertragenen Feldwerte, Locator- und Hashketten, die
unveränderten Katalog-Sperren und sämtliche offenen Zuordnungen.
[verification.json](./verification.json) ist der aktuelle Prüfnachweis. Er ist
kein unabhängiger menschlicher oder palynologischer Review. Kein Modellfit,
keine Verwendung äußerer Scores, keine Änderung der Analog-Baseline oder CLI/App,
keine Änderung von #28–#32-Quellen oder -Ergebnissen. Monticchio-P1 und die
28/41/47-Befunde bleiben unverändert.
