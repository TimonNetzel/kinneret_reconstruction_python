# Sieben Arten: Originaldefinitionen und EMPD2-Namensketten

31.10.2026. Folgepaket zu [Inventur #34](../pollen_taxon_evidence_2026_10_31/README.md),
Basis `98ef1bf980aaa71903fa8fe40ef7797bd9d16b87`. **Die beiden registrierten
Originaldateien wurden frisch bezogen und hashgeprüft.** Keine neue Modell-
oder Pollentyp–Art-Freigabe. Keine Fossilzählung, Gruppierung oder CLI/App geändert.

## Was die Originale nun belegen

### Monticchio: Kommentare sind akzeptierte EPD-Namen

PANGAEA.739452, Metadatenzeile **9**, sagt ausdrücklich:

> The EPD (http://www.europeanpollendatabase.net) accepted species name is given in the parameter comment. This dataset was archived on 2010-05-11 from the EPD database.

Die sieben Parameter stimmen mit den bisher erhaltenen Archivnamen überein:
[Originaldefinitionen](./data/monticchio_definitions.csv) mit Metadatenzeile,
Tabellenkopfzeile und 1-basiertem Spaltenindex. Abies, Betula, Corylus, Fagus,
Picea und Tilia stehen als Gattungen in den Kommentaren; `Tax [#]` hat den
Kommentar `Taxus baccata` (Zeile 115). Die Formulierung „species name“ im
allgemeinen Archivhinweis macht die sechs Gattungsnamen nicht zu Artbestimmungen.

**Taxus-Präzisierung:** Die Rolle des Kommentars als akzeptierter EPD-Name ist
jetzt am Original belegt. Ob die Proben ursprünglich auf Artebene bestimmt
wurden, welches Bestimmungsmerkmal benutzt wurde und wie die Zuordnung historisch
entstand, bleibt offen. Der Parameter enthält kein Bestimmungsprotokoll. Das
im Event genannte METHOD/DEVICE beschreibt das Bohrgerät, keinen Pollenschlüssel.
Die verwandte Publikation im Original ist Watts, Allen & Huntley (1996),
DOI [10.1016/0277-3791(95)00093-3](https://doi.org/10.1016/0277-3791(95)00093-3);
ihre Bestimmungsmethoden wurden in diesem Paket nicht gelesen.

### EMPD2: ursprüngliche Namen und akzeptierte Pollenkonzepte trennen

[Sieben direkte Originalnamenslinks](./data/species_name_links.csv), jeweils
mit den ursprünglichen `p_vars`-Schlüsseln (kein BfN-/Neotoma-Namensraum):

| Botanisches Modellziel | Original `var_`, Name | Akzeptiertes `acc_var_`, Name | Bisheriger Fossilkatalog-Kandidat |
| --- | --- | --- | --- |
| Abies alba | 759, Abies alba | 321, Abies | 321, gleicher akzeptierter Schlüssel |
| Betula pendula | 1731, Betula pendula | 128, Betula pubescens-type | 12, Betula; **anderer** Schlüssel |
| Corylus avellana | 31, Corylus avellana | 31, Corylus | 31, gleicher akzeptierter Schlüssel |
| Fagus sylvatica | 42, Fagus sylvatica | 42, Fagus | 42, gleicher akzeptierter Schlüssel |
| Picea abies | 783, Picea abies | 783, Picea abies-type | 126, Picea; **anderer** Schlüssel |
| Taxus baccata | 111, Taxus baccata | 111, Taxus baccata | 111, gleicher akzeptierter Schlüssel |
| Tilia cordata | 1406, Tilia cordata | 153, Tilia cordata-type | 187, Tilia; **anderer** Schlüssel |

Vier gleiche und drei unterschiedliche akzeptierte IDs sind **Namensketten-
befunde**, keine vier genehmigten Artbrücken. Eine Selbstreferenz wie 31→31
bedeutet nicht, dass ursprünglicher und akzeptierter Name gleich spezifisch sind.
`Betula pendula → Betula pubescens-type` ist keine hier bestätigte botanische
Synonymie. Auch das originale `Betula pendula/pubescens` (3615) verweist auf 128.
Das schließt die enge Namenslücke aus #34: B. pendula ist in der alten EMPD2-
Datei vorhanden. Die Zugehörigkeit zum 2025-Supplement wird damit nicht frisch
verifiziert; dort wurde in #34 nur der erhaltene Gruppierungsbeleg inventarisiert.

Weitere prüfbare Unterschiede in [empd_name_chains.csv](./data/empd_name_chains.csv):

- `Taxus` kommt zweimal als Originalname vor: 198→111 (`Taxus baccata`) und
  3433→3433 (`Taxus`). Nicht automatisch zusammenlegen. Der alte Kandidat 111
  entstand in einer begrenzten Katalogsuche mit positivem Datennachweis und
  beanspruchte keine vollständige Dictionary-Schließung.
- Fagus orientalis (3963) verweist ebenfalls auf Fagus (42); Picea obovata
  (2130) auf Picea abies-type (783). Ein akzeptierter Typname allein garantiert
  daher nicht, dass ausschließlich die gleichnamige Modellart eingereicht wurde.
- Die Datenbank führt auch auffällige Ketten wie `Melia` (1951)→Abies (321)
  und `Caryo. Trizoncolpate indet.` (2191)→Abies (321). Diese Quelldaten sind
  unverändert erhalten; keine Ursache, botanische Richtigkeit oder Korrektur
  wird daraus abgeleitet.
- `p_vars` hat genau fünf Felder: `var_`, `acc_var_`, `original_varname`,
  `acc_varname`, `groupid`. Es gibt dort **keine Notiz-/Schlüssel-/Synonymie-
  Begründungsspalte**. Die anderen Tabellen und probenspezifischen Notizen wurden
  nicht als Ersatz für einen taxonspezifischen Bestimmungsbeleg interpretiert.

## Umfang der vollständigen Namenssuche

Alle **5.230** `p_vars`-Zeilen wurden auf die sieben Gattungswörter in beiden
Namensfeldern geprüft, ohne Beschränkung auf die alten Kandidaten-IDs. Danach
wurden alle `acc_var_`-Verweise bis zur Selbstreferenz verfolgt. Ergebnis:
**128 lexikalische Treffer über sieben Abfragen, 135 Artabfrage×Zeilen-Einträge,
120 unterschiedliche Dictionary-Zeilen** einschließlich referenzierter Zielzeilen.
Alle ausgewählten Ketten schließen; keine fehlende ID oder Endlosschleife.
Anzahlen je Abfrage stehen im [Manifest](./data/manifest.json).

„Vollständig“ gilt für diese deklarierte Namenssuche und Referenzverfolgung,
**nicht** für alle botanischen Synonyme oder einen freigegebenen Gattungspool.
Wortgrenzen, keine beliebigen Teilzeichenketten; Groß-/Kleinschreibung nur für
die Suche normalisiert, ausgegebene Originalnamen unverändert.
Bewusst breite Suche: `Abies` trifft auch das Epitheton in `Picea abies`.
Solche Treffer sowie Mischkonzepte, cf.-Felder, Antheren/Stomata und grobe
Zielkonzepte bleiben als lexikalische Kandidaten sichtbar, niemals als automatisch
zu Abies oder zu Pollen gehörende Beobachtung. Dasselbe Original kann deshalb
in mehreren Abfragen erscheinen. Keine daraus resultierende Doppelzählung oder
Aggregation durchgeführt. Nicht namensbezogene Synonyme können fehlen.

## Sicherung, Quellen und Prüfung

Das [Manifest](./data/manifest.json) erhält beide Download-URLs, DOI, Bytezahl,
Lizenz, Hash, Datenbankschema, Archivkontext und drei gesperrte Repository-
Eingänge. [REPRODUCE.md](./REPRODUCE.md) enthält Bezugs- und Prüfbefehle.
Die frischen Originale stimmen mit den bestehenden Hashes aus #34 überein:
EMPD2.sqlite3, 21.327.872 Bytes, PANGAEA.909130 (CC-BY-4.0), und Monticchio,
86.174 Bytes, PANGAEA.739452 (CC-BY-NC-3.0). Keine Hashänderung.
Die extrahierten Metadaten behalten ihre Quellenattribution und jeweiligen
Lizenzbedingungen. Originaldateien und Zählmatrizen werden nicht eingecheckt.
Lokaler Cache ist kein nachgewiesenes dauerhaftes Bytearchiv.

[Separater Prüfer](./verify.py) importiert den [Extraktor](./extract.py) nicht,
liest die Originaldateien direkt und prüft Hashes, vollständige Abfragedeckung,
SQL-Feldwerte, Referenzketten, Originalzeilen/-spalten und sieben Modellnamenslinks.
[Prüfbericht](./data/verification.json): bestanden. Derselbe Bearbeiter und
SQLite-/Python-Decoder, kein unabhängiger menschlicher oder palynologischer
Review. Keine Modell-/Langtests nötig oder ausgeführt; keine Modelle angepasst,
keine äußeren Scores gelesen oder zur Auswahl verwendet.

**Abgeschlossen:** Originalmetadaten und Namensketten der zwei Quellen.
**Offen:** morphologische Art-/Typabgrenzung, tatsächlich benutzte Bestimmung und
kontextspezifische Artbrücke für alle sieben Fälle; präzise nächste Schritte in
[SOURCE_GAP.md](./SOURCE_GAP.md). 2025-Matrix/S1 in diesem Paket nicht frisch
bezogen. Monticchio-P1, 28/41/47 und vier große Bytearchivlücken unverändert.
