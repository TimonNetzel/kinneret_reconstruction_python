# Bestimmungsumfang der sieben Modellarten: Literaturprüfung

31.10.2026. Basis `f7e5290081dedb775a5cae7e210db88b71d9b1cb`, offener
[PR #36](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/36).
Auftrag: [SOURCE_GAP aus #35](../original_pollen_names_2026_10_31/SOURCE_GAP.md).
Eigener Worktree `/workspace/netzel-pollen-methods`, Branch
`codex/netzel-pollen-methods`; ohne Unteragenten.
Gesichert als [PR #37](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/37)
gegen den #36-Branch; kein Merge.

**Ergebnis:** Für 7/7 Arten sind gelesene morphologische Referenzbefunde,
Typgrenzen und verbleibende Fragen dokumentiert. **Die fossile Methodenkette
ist nicht geschlossen:** Von Watts, Allen & Huntley (1996), DOI
[10.1016/0277-3791(95)00093-3](https://doi.org/10.1016/0277-3791(95)00093-3),
war nur bibliographische Information/Abstract zugänglich. Methodenteil und
tatsächlich benutzte Bestimmungsreferenzen konnten nicht gelesen werden.
Keine der sieben Artbrücken ist damit freigegeben; T0 bleibt offen.

## Was die belegte Kette trägt

Die in #35 originalgeprüfte Archivmetadatenzeile 9 erklärt die Kommentare zu
akzeptierten EPD-Namen. Die Parameterdefinitionen in Zeilen 11/19/40/50/85/115/117
sind erhalten; sechs Gattungen und der Kommentar Taxus baccata belegen kein
ursprüngliches Artbestimmungsprotokoll. Grundlage hier sind die unveränderten
[Definitionen](../original_pollen_names_2026_10_31/data/monticchio_definitions.csv),
[Namensketten](../original_pollen_names_2026_10_31/data/empd_name_chains.csv) und
der [Manifestkontext](../original_pollen_names_2026_10_31/data/manifest.json).
Die Originaldateien wurden in diesem Paket nicht erneut bezogen/gehasht.

**LGM90D → verwandte Publikation W96** ist somit ein vorhandener Quellenverweis.
**W96 → tatsächlich verwendeter Schlüssel/Referenzsammlung → Bestimmungsregel
je Parameter → erhaltene getrennte Zählkategorie** bleibt unbelegt. Kein Schlüssel
wird aus üblichen Laborpraktiken, einer fremden Bibliographie oder späterer
Literatur als damals verwendet eingetragen. Auch eine Zitation allein würde
noch nicht beweisen, dass gerade ihre Artunterteilung angewandt wurde.

Je Art lautet die verlangte Kategorie deshalb **kein datensatzbezogener
Bestimmungsbeleg im hier lesbaren Material**. Die neuen Literaturbefunde sind
allgemeine Referenzevidenz, keine nachgewiesene allgemeine Methode von W96.
Das bedeutet nicht, dass W96 entsprechende Angaben nicht enthält.

## Evidenztabelle

Quellenkürzel sind im [Quellenregister](./SOURCES.md) mit stabilen URLs,
Version/Objekt und exakter Fundstelle aufgelöst. LM = Lichtmikroskopie,
SEM = Rasterelektronenmikroskopie. Referenzmerkmale sind keine hier festgelegten
Klassifikationsschwellen. Unterschiede zwischen Präparationen/Ansichten dürfen
nicht in eine einzige Größenregel umgerechnet werden.

| Modellart / Fossilfeld (Archivzeile) | Unterstützender morphologischer Befund | Grenze, Gegenbeleg oder fehlender Umfang | Konkrete offene Frage für LGM90D |
| --- | --- | --- | --- |
| **Abies alba** / Abi (11) | **A20 §3.2**, PDF S. 8–9: zwei Sacci, Leptoma, proximaler Y-Abdruck. **C215**, Visible notes: Übergang Saccus–Corpus und Wand gegenüber den Sacci helfen bei der Gattungsabgrenzung gegen Picea. | **A20 §4**, PDF S. 17–18: Ähnlichkeit mit anderen europäischen/asiatischen Abies; Y-Abdruck und getrennte Sacci ausdrücklich nicht artspezifisch. Die Studie an 48 polnischen Genotypen validiert keine italienische fossile Artbestimmung. | Welcher Schlüssel und welche Artmerkmale wurden benutzt? Ist alba gegenüber anderen Abies tatsächlich nachgewiesen? Auffällige EMPD2-Quellketten bleiben erhalten. |
| **Betula pendula** / Bet (19) | **B18 §1**, Absatz zu Blackmore et al.: pubescens-Typ umfasst pendula sowie pubescens ssp. carpatica, ssp. pubescens und ssp. tortuosa; nana-Typ umfasst nana/humilis. Kriterien betreffen Größe/Form, Exine und Poren/Vestibulum. Dies ist eine **indirekt referierte** Schlüsseldefinition. | **B18 §2.1–2.2, §3.1**: rezentes Material; pendula/pubescens/humilis ähnlich groß, keine deutlichen Form-/Exine-/Porenunterschiede in den untersuchten SEM-Proben. Kein universelles Untrennbarkeitstheorem. Blackmore (2003) selbst nicht im Volltext geprüft. EMPD2 1731→128 ist keine botanische Synonymie. | Wurde Betula überhaupt unterteilt, und blieben entsprechende Zählkategorien erhalten? Ein pubescens-Typ wäre allein noch keine pendula-Bestimmung. |
| **Corylus avellana** / Cor (40) | **C25**, Objekt 306797, Pollen Description: drei Poren, Annulus/Oncus; LM glatt bis scabrat, SEM rugulat/granulat. Picture legend 5–10 unterscheidet acetolysiertes von frischem Material. | **Q-D**, Betulaceae; **Q-N**, Anm. 24: Corylus-avellana-Typ enthält in diesem britischen Katalog auch Myrica gale; große Variationsüberlappung, nur manche Myrica-Körner sicher abtrennbar. Keine Behauptung desselben Typumfangs für EMPD2 oder Monticchio. | Belegt der historische Schlüssel die engere Bestimmung? Wie wurden Corylus/Myrica und andere Corylus behandelt? Die Selbst-ID 31→31 löst das nicht. |
| **Fagus sylvatica** / Fag (50) | **F26**, Objekt 307041, Pollen Description: drei Colpori, LM scabrat/rugulat/verrucat; Beschreibung rezenter sylvatica. | **F07**, Results and discussion → Identification of fossil pollen: Artunterscheidung durch cpDNA in anderem italienischem Fossilmaterial; dortige frühere Palynologie trennte die Arten nicht. Das belegt weder orientalis in LGM90D noch generelle morphologische Unmöglichkeit. EMPD2 3963→42 bleibt ein separater Namensbefund. | Welche morphologische sylvatica/orientalis-Abgrenzung wurde hier angewandt und dokumentiert? Geographische Plausibilität ersetzt sie nicht. |
| **Picea abies** / Pic (85) | **P21**, Objekt 306140, Pollen Description: saccat, heteropolar, Leptoma; **C215**, Visible notes: gegen Abies geringerer Winkelwechsel zwischen Saccus und Corpus und andere Saccusskulptur. | P21 ist eine Artbeschreibung, **keine Definition des abies-Typs gegen obovata**. Eine geprüfte morphologische Typumschreibung/Trennregel hierfür fehlt. EMPD2 2130→783 belegt allein, dass auch der eingereichte Name obovata auf diesen Datenbanktyp verweist. | Wurde mehr als Picea gezählt? Wo sind Typdefinition, Geltungsbereich und getrennte Originalkategorie belegt? Keine Trennbarkeit oder Untrennbarkeit aus der Datenbank ableiten. |
| **Taxus baccata** / Tax (115) | **T07**, S. 225, Characteristics of Taxus pollen: keine Sacci/Aperturen, scabrate bis microgemmate Exine; mögliche Verwechslung u. a. mit Juniperus/Populus/Rhynchospora/Quercus sowie Korrosions-/Spaltungsprobleme. | **T18**, S. 45 und 57, Conclusions: untersuchte baccata/cuspidata/×media samt Sorten anhand der Pollenmerkmale nicht einzeln trennbar, nur Gruppen. Reichweite: rezentes Gartenmaterial, keine pauschale Aussage über alle Fossilien. Die zwei EMPD2-Taxus-Einträge 198→111 und 3433→3433 werden nicht gepoolt. | Originaler Bestimmungsumfang und Behandlung ähnlicher/erodierter Körner? Der akzeptierte EPD-Artkommentar beantwortet dies nicht; Herkunft der zwei EMPD2-Konzepte bleibt offen. |
| **Tilia cordata** / Til (117) | **L74**, S. 107–109: Untersuchung zweier rezenter Herkünfte; platyphyllos hat gröbere/unregelmäßigere Skulptur, cordata häufiger einfache Trichter ohne sichtbare seitliche Öffnungen; Vergleich jeweils desselben Pols auch in LM beschrieben. | Beide Arten teilen den grundlegenden Wandbau; Präparation/Polansicht beachten. **L-C/L-P**, Objekte 306671/306672: beide mit drei Colpori und retikulierter LM-Skulptur beschrieben (Feldinkonsistenz bei L-P siehe SOURCES); rezente Referenzen, kein Ersatz für eine fossile Typdefinition. **Q-N**, Anm. 43: platyphyllos-Typ schließt dort auch ×vulgaris ein. L74s erwähnte Tafeln 11–14 sind im gelesenen dreiseitigen PDF nicht enthalten und wurden nicht geprüft. | Wurden cordata-/platyphyllos-Untertypen in LGM90D unterschieden und separat archiviert? Kriterien, Hybridumfang und Übertragung auf Erhaltungszustand fehlen; 1406→153 macht Til nicht zu cordata. |

Die Grenzen sind unterschiedlich belegt: direkte moderne Vergleichsstudien,
historische Morphologiestudie, morphologische Online-Diagnosen und begrenzte
Katalogkommentare. **Nichtbefund eines Trennschlüssels** (etwa Picea) ist kein
empirischer Nachweis fehlender Trennbarkeit. Die Datenbankrelationen aus #35
werden als Warnhinweise erhalten, ohne daraus botanische Synonyme zu machen.
Kein neuer Artenpool, keine Umcodierung und keine biologische Absenzregel.

## Prüfstand und Übergabe

[evidence.csv](./evidence.csv) enthält sieben explizite Statusentscheidungen und
ihre Fundstellen. [sources.json](./sources.json) hält gelesene Ausschnitte von
unzugänglicher Originalmethodik auseinander. [verification.json](./verification.json)
dokumentiert Hash-/Basisvergleiche, Referenzauflösung und den engen Diffumfang;
[REPRODUCE.md](./REPRODUCE.md) beschreibt die Nachprüfung.
Dies ist eine eigene Quellenprüfung, kein unabhängiger menschlicher oder
palynologischer Review. Online-Texte/ausgelesene PDF-Passagen gelesen, keine
eigene mikroskopische Untersuchung und keine Bilddiagnostik.

**Gesichert:** sieben Literaturbefunde mit Fundstellen und Gegen-/Fehlbelegen.
**Blockiert:** W96-Methodenteil, seine tatsächlich verwendeten Bestimmungshilfen
und die datensatzbezogene Artbrücke. [SOURCE_GAP.md](./SOURCE_GAP.md) begrenzt
die nötige Nachlieferung; keine erneute breite Namenssuche.
Modelle, bestehende Forschungsresultate/-register, Fossilcounts, Analog-Baseline
und CLI/App bleiben bytegleich. Keine Modell-/Langläufe oder Auswahl anhand
äußerer Testergebnisse. 2025-S1/Matrix nicht benötigt oder geprüft;
Monticchio-P1, 28/41/47 und vier große Bytearchivlücken unverändert.
