# T1-Kandidat und begrenztes synthetisches T2: Ergebnis

31.10.2026. Basis PR #38 bei `d003fb9`; eigener Worktree
`/workspace/netzel-t1-synthetic-t2`, Branch `codex/netzel-t1-synthetic-t2`.
Das [Design](./DESIGN.md) mit [Konfiguration](./config.json) wurde als
`2b41c377af45035ec8447bd2b2b435bf55121aac` **vor Implementierung und Lauf**
committed und nach origin gepusht. Keine fachlichen Designänderungen nach dem Lauf.
Der [Git-/PR-Snapshot](./git_snapshot.json) belegt #31–#38 offen, main bei
`7f1826c`, Ausgangsworktrees sauber; PR #38 technisch MERGEABLE/CLEAN nach CI.
Kein Merge und keine Unteragenten.
Gesichert als [PR #39](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/39)
gegen `codex/netzel-next-task-workflow` (#38), Inhaltscommit `953d1e9`;
nachfolgende Sicherung ergänzt ausschließlich PR-/Übergabeverweise.

**Begrenzt abgeschlossen:** ein kohärentes synthetisches Beobachtungsmodell,
diskrete Inversion und alle vorab vorgesehenen Grenzfälle. **Weiter offen:**
T1 für reale Pollen, T0/Artbrücken, T3/T4, W96-Methodenkette und Monticchio-P1.
Vier anonyme binäre Kanäle sind keine sieben echten Artenmodelle und keine
Pollenhäufigkeiten. Kein Modell wurde übernommen oder anhand äußerer Scores gewählt.

## Vollständiger synthetischer Vergleich

108 Posteriore aus 38 Datensätzen, jeweils 1.681 diskrete Zustände.
90 Auswertungen bekannter innerer Zustände: drei Zustände × fünf vorab fixierte
Seeds, je beide Priors, davon 30 independent→independent, 30 shared→shared und
30 shared→bewusst falsch independent. Alle Varianten stehen hier; kein Sieger.
Seeds werden zwischen Zuständen wiederverwendet: keine 15 unabhängigen
Bestätigungsstudien und keine empirisch validierte 95%-Deckung.

| Generator → Inferenz | Prior | n | Mittlerer Abstand Posterior-Mittel → Wahrheit | Größter Abstand | Wahrheit im diskreten 95%-HPD-Set | Mittlere HPD-Zellzahl |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| independent → independent | uniform | 15 | 0,14835 | 0,31546 | 15/15 | 114,07 |
| independent → independent | informativ | 15 | 0,11936 | 0,28287 | 15/15 | 102,73 |
| shared → shared | uniform | 15 | 0,12572 | 0,28320 | 14/15 | 98,07 |
| shared → shared | informativ | 15 | 0,12077 | 0,27763 | 14/15 | 90,20 |
| shared → independent, missspezifiziert | uniform | 15 | 0,15343 | 0,32890 | 13/15 | 81,73 |
| shared → independent, missspezifiziert | informativ | 15 | 0,16273 | 0,35083 | 11/15 | 72,60 |

Abstände sind dimensionslos im Spielzeugraum, keine °C-/mm-Fehler.
Der informative Prior verringert hier den mittleren Fehler bei beiden passenden
Modellen, erhöht ihn bei der missspezifizierten Inferenz. Auch passende Modelle
verfehlen einzelne Wahrheiten; ein kleineres HPD-Set bedeutet nicht höhere Güte.
Die Missspezifikation verändert durch Weglassen von H **auch die marginalen
Antwortkurven**. Dieser Vergleich isoliert daher keinen reinen Korrelationseffekt.
Keine Brier-/Karten-Scores oder bekannten äußeren Testregionen verwendet.

Die acht Ausschlüsse innerer Wahrheiten aus HPD-Sets sind vollständig erhalten:

| Daten-ID (Präfix `known_shared_`) | Inferenz / Prior |
| --- | --- |
| 0_104 | shared / uniform; shared / informativ |
| 1_102 | independent / informativ |
| 1_103 | independent / uniform; independent / informativ |
| 1_104 | independent / uniform; independent / informativ |
| 1_105 | independent / informativ |

0 bezeichnet (0,0), 1 bezeichnet (−0,5; 0,5), Suffix den Seed.
Alle Einzelwerte, marginalen Quantilbereiche, HPD-Massen, Log-Evidenzen und
Randmassen stehen in [results.json](./outputs/results.json). Log-Evidenzen über
verschiedene Datensätze werden nicht als Modellrangliste ausgewertet.

## Grenzfälle: numerisch bestanden, fachliche Grenzen sichtbar

| Fall | Ergebnis / Bedeutung |
| --- | --- |
| Widerspruch, 2 Posteriore | Beide u-Kanäle in 40/40 Einheiten positiv: obere Tail-Schranke 8,2718×10⁻²⁵, beide Konfliktflags gesetzt. Trotzdem u-Quantilbereich etwa [−0,15;0,15]. Ein schmaler Posterior kann ein schlechter Modellkompromiss sein; Flag ist hier nur die vorab spezifizierte Spezialdiagnose. v bleibt priorbestimmt. |
| Ein Kanal fehlt, 2 | 120 statt 160 Beobachtungen, beide Posterioren gültig, Wahrheit enthalten. Fehlend nicht als 0 behandelt. |
| Nur u beobachtet, 2 | v-Posterior gleich v-Prior bis 8,33×10⁻¹⁷. v bleibt nicht identifizierbar, obwohl u konzentriert. Mit informativem Prior liegt v-Mittel bei −0,21097, ohne v-Evidenz. |
| Alles fehlt, 4 | Beide Modelle × beide Priors: Posterior gleich Prior bis 6,51×10⁻¹⁹, Log-Evidenz exakt 0. Kein Klimasignal. |
| s=f=0,1, 2 | Trotz 160 Beobachtungen Posterior gleich Prior bis 2,15×10⁻¹⁷. Beobachtungszahl allein erzeugt keine Information. |
| Gefaltetes u, 2 | Likelihood für ±u gleich bis 5,69×10⁻¹⁴. Uniform: je etwa 0,49625 Masse bei u<0/u>0, Mittel etwa 0 statt wahrer 0,5. Informativ: negative/positive Masse 0,27775/0,71380. Die Richtungspräferenz stammt vom Prior; das Vorzeichen bleibt strukturell nicht identifizierbar. |
| Ungültige Eingänge | Supportverletzungen an allen vier Kanten, NaN/Inf, ungültige Y/Detektionsraten/Formen/Bezeichner explizit abgewiesen; keine stille Reparatur. |

Vier Randfälle sind zusätzlich vollständig ausgewertet:

| Wahrheit | Prior | Fehler des Mittels | Randmasse | Wahrheit in HPD |
| --- | --- | ---: | ---: | --- |
| (−1;1) | uniform | 0,29487 | 0,22609 | ja |
| (−1;1) | informativ | 0,40697 | 0,09659 | **nein** |
| (1;−1) | uniform | 0,31449 | 0,23055 | ja |
| (1;−1) | informativ | 0,35516 | 0,17207 | ja |

Der informative Prior verschlechtert beide Rand-Mittelfehler. Randmasse hat
keine nachträglich optimierte Sperrschwelle. Eine unbekannte außerhalb liegende
Wahrheit ist aus den Beobachtungen nicht zuverlässig diagnostizierbar;
getestet wurde die Abweisung bekannter unzulässiger Vorwärtsabfragen.
Alle 108 Auswertungen berechnet; null nicht schätzbare Rechenfälle, keine
übersprungenen geplanten Szenarien. Strukturelle Nichtidentifizierbarkeit bleibt
ein fachliches Ergebnis und wird nicht als Rechenfehler weggezählt.

## Prüfungen, Sicherung und Reichweite

- 29 gezielte Tests bestanden: direkte Enumeration der latenten Z/H-Zustände,
  gemeinsame Mischung statt Produkt gemittelter Kanäle, Missingness, Priors,
  Symmetrie, unmögliche Daten und ungültige Eingänge.
- [Separater Prüfer](../../../tools/taxon_synthetic/verify.py) ohne Import des
  Produzenten: 38 Datensätze aus dem gesperrten Plan neu erzeugt, alle
  108×1.681 Posteriormassen mit direkter Produkt-/Mischungsformel verglichen,
  maximale absolute Differenz 1,75×10⁻¹⁵, Mittelwerte 1,89×10⁻¹⁵;
  Diagnosen und sieben Manifestdateien geprüft.
  [Maschinenprotokoll](./verification.json). Python/NumPy sind gemeinsame
  Abhängigkeiten; kein unabhängiger menschlicher Review behauptet.
- Zweiter frischer Lauf: alle fünf Ausgabedateien einschließlich Manifest
  bytegleich. [Hashes und Vergleich](./repeat_verification.json).
- [Erhaltungsprüfung](./preservation.json): 654 bestehende Blobs einschließlich
  eines Symlinks bytegleich gegen den Basiscommit; nur HANDOFF/PROJECT_STATUS
  als bestehende Dokumente verändert. Fünf ursprüngliche Worktrees weiterhin
  sauber, neue Pfade auf dieses Paket begrenzt; lokale Dokumentationslinks gültig.
- Der zusätzlich versehentlich breit gestartete Standardtest wurde abgebrochen:
  56 bestanden, zwei Browserfehler, zwei Langtests abgewählt, Exit 2 nach Interrupt.
  Playwrights erwartetes Chromium-Binary fehlt. Vorhandene Tests enthalten auch
  kurze Kinneret-Regressionsketten und reale Testeingänge; sie wurden dabei
  teilweise ausgeführt. Das war außerhalb des benötigten synthetischen
  Prüfbereichs. Keine neue wissenschaftliche Kernkurve ausgewertet/gesichert,
  keine Referenzen verändert; verbleibende Standardtests nicht als bestanden
  behauptet. Einzelheiten in [TESTING.md](./TESTING.md); keine App-Reparatur.

Beobachtungen, vollständige diskrete Massen und Diagnosen liegen unter
[outputs](./outputs/manifest.json) als Git-Dateien vor, nicht nur unter /tmp.
[REPRODUCE.md](./REPRODUCE.md) nennt genaue Befehle und Umgebung.
Die ursprünglichen vier großen Bytearchivlücken bleiben unverändert.

Unsicherheit umfasst ausschließlich das festgelegte diskrete generative Modell.
Keine Nominaldeckungsprüfung, kein SBC, keine Vollunsicherheit für Fossilien,
keine automatische Art-/Pilotfreigabe. Kartenlabel→lokales Vorkommen→Pollenfluss→
Zählkomposition ist noch nicht belegt. Nächster begrenzter Auftrag ist ein
**Anschlussdesign für Zählkomposition und Typgemische**, siehe
[HANDOFF](../HANDOFF.md); wartet auf Nutzer-OK. W96 bleibt separat blockiert.
