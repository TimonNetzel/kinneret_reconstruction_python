# A3: Audit der gesperrten endlichen synthetischen Prüfung

Stand 31.10.2026. **A3 ist im begrenzten synthetischen Umfang abgeschlossen
und bestanden.** 22 Posteriorfälle, drei vollständige prädiktive Verteilungen,
33 einzelne ungültige Eingänge und 345 exakte Nachweise ohne Abweichung.
Keine Pflichtfälle fehlgeschlagen, blockiert, nicht schätzbar oder nicht
ausgeführt. Die erwartete Nichtidentifizierbarkeit in S11 bleibt bestehen.
Das Ergebnis bestätigt ausschließlich die definierten künstlichen Fälle.

## Auftrag, Sperre und Isolation

Der Nutzer hat nach [PR #54](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/54)
die Implementierung, Ausführung und getrennte Nachrechnung ausdrücklich
beauftragt. Basiscommit `8583d31843e0e6919bb0b1c03fbb7ebc9dfd7c9b`;
eigener Worktree `/workspace/netzel-a3-synthetic-execution`, Branch
`codex/netzel-a3-synthetic-execution`. Vorher fünf Worktrees sauber, keine
Stashes; main unverändert `7f1826cffddc8893ba6354cd3cca3fa43954b9be`.
PR #54 offen, ungemergt und mergebar; dessen Head entspricht der Basis.
Automatische [CI 41235319258](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41235319258)
am Basis-Head erfolgreich. Kein Reset, Merge oder Eingriff in fremde Arbeit.
Eigener Folge-PR gegen den #54-Branch; endgültiger Sicherungsverweis in
[HANDOFF](./HANDOFF.md) und neuestem [Status](./PROJECT_STATUS.md).

Maßgeblich: [gesperrtes Design](./A3_SYNTHETIC_DESIGN.md), unveränderter
[MODEL_CONTRACT](./MODEL_CONTRACT.md), [Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
ausschließlich §2-Zeile A3 und [§4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
Design-Sperrcommit **`a4c03c6752708dead6ecd98f016758538ddb71b2`**, Design-Blob
**`21879814c77f0341457f9f4790dd40e24b555d55`** vor Implementierung bestätigt.
Keine Definition, Wahrheit, Fallmenge oder Abnahme angepasst. NOVELTY_MATRIX,
Originalquellen, Dissertation und reale Eingangsdateien wurden nicht nachgeladen.

## Zwei Rechenwege und Ausführung

[Prüfkern](../../scripts/netzel_a3_synthetic/kernel.py): expliziter Markov-Prior,
Multinomialfaktoren, gemeinsame psi-/theta-Variablen, ganze tau-Kurven und
Normierung des gesamten zweiten Blocks je psi. Eingangsprüfung vor Q;
negative/nicht endliche Kernwerte und Z=0 stoppen die jeweilige Berechnung.
Die Schnittstelle akzeptiert nur die festen Fälle; auch anderweitig gültige,
aber abweichende Counts oder q werden als Erweiterung zurückgewiesen.

[Referenz](../../scripts/netzel_a3_synthetic/reference.py): eigene verschachtelte
Zustandsschleifen, direkt eingesetzte rationale Polynome, eigener Normierer und
eigene ereignisweise Marginalisierung. S06c verwendet die marginale
Binomialformel statt der Kompositionssumme des Kerns; S05 zählt geordnete
Kategoriepaare statt Multinomialkoeffizienten auf. Nur deklarierte
[Fallkonstanten](../../scripts/netzel_a3_synthetic/constants.py) und rationale
Standardarithmetik werden gemeinsam genutzt. Kein Import von Faktoren,
Normierern, Missingness- oder Marginalisierungsfunktionen zwischen den Wegen.
Zusätzliche Produkt-, Moment- und Odds-Identitäten stehen im
[Auditprogramm](../../scripts/netzel_a3_synthetic/audit.py).
Eine gemeinsame Sprache und gemeinsame Konstanten können gemeinsame Fehler
nicht ausschließen; die andere Zerlegung und algebraischen Zeugen begrenzen
dieses Risiko. Dies ist keine externe fachliche Begutachtung.

S03 und S10 haben zusätzlich einen Posterior ohne die zukünftigen Beobachtungen;
S05 ist ausschließlich eine prädiktive Komposition. Kein zukünftiger Messwert
wird erneut als Evidenz konditioniert. Alle nicht ersetzten Definitionen bleiben
wie S01. Punktfixierungen ersetzen nur ihren jeweiligen Prior durch eine
Punktmasse; S11 ersetzt den zweiten Block durch den gesperrten Zweizustandsprior.

Manuell ausschließlich diese neue Prüfung und ihr neues Prüfmodul ausgeführt:

```bash
python -m scripts.netzel_a3_synthetic.audit --output docs/thesis_research/A3_SYNTHETIC_AUDIT
python -m unittest discover -s tests -p test_netzel_a3_synthetic.py -v
```

Endstand: **345 Nachweise bestanden; sechs gezielte Tests bestanden**.
Der erste vollständige Tabellenlauf hatte bereits null Abweichungen; danach
wurden die vollständigen Marginaldifferenzen, explizite S11-Nullmassen und der
q-aus-E-Zeuge ergänzt und die betroffene endliche Prüfung erneut ausgeführt.
Keine Ergebnisabweichung oder Implementierungskorrektur an Wahrscheinlichkeiten
war nötig. Eine zusätzliche Startprüfung aus /tmp zeigte zunächst einen
fehlenden Importpfad im neuen Prüfmodul; der Test setzt nun seinen eigenen
Checkoutpfad ausdrücklich. Der erneute direkte Start aus /tmp bestand alle
sechs Tests. Keine Auswahl günstiger Fälle oder Anpassung von Erwartungswerten.
Automatische breite Repository-CI wird separat beim jeweiligen PR-Head berichtet;
es wurde keine breite Projekt-/Modellprüfung manuell gestartet.

## Vollständige Fallbilanz

Für jeden Posteriorfall sind sämtliche F-/Q-Zustände, beide Z, Summe Q und
alle Marginalen exakt zwischen beiden Wegen gleich, Q nichtnegativ, Z positiv
und Q(psi)=q. „Bestanden“ umfasst zusätzlich die folgenden jeweiligen Zeugen.
Die einzelnen tatsächlichen Brüche einschließlich Nullmassen und Abweichungen
stehen vollständig in den verlinkten Tabellen, ohne Fließkommatoleranz.

| Fall | Fester Eingang / erwarteter Nachweis | Tatsächlicher Befund | Status |
| --- | --- | --- | --- |
| S00 | P/Y/D fehlen; Produkt der Priors | Q=q*pC*pTau*pTheta exakt; C weiterhin gekoppelt | bestanden |
| S01 | Grundfall mit festgelegter Wahrheit | Z(-1)=13643/8388608; Z(+1)=32933/8388608; q aus E allein bestätigt | bestanden |
| S02-scaled | S01, Kernfaktor r=(1,3) | Jeder Q-Zustand gegenüber S01 unverändert | bestanden |
| S02-witness | Ohne Evidenz, F=r*Prior | Q wie S00; globale Normierung verletzt q | bestanden |
| S03 | C=(1,1,1),tau=a,theta=1; zwei künftige A-Indikatoren | E(B)=5/12, Cov=Var_q(pA)=1/72>0 | bestanden |
| S04 | S00 mit gemeinsamer C-Kette | Pr(C2=C1)=3/4, Cov(C1,C2)=3/8>0; direkte Markov-Tabelle gleich | bestanden |
| S05 | Künftige Komposition, N=2,p=(1/2,1/4,1/4) | Alle sechs Ausgänge summieren zu N; Cov(nA,nB)=-1/4 | bestanden |
| S06a | P1=(0,2,0),N=2 | LP1=pB², je Zustand 1/16 oder 1/4, stets positiv | bestanden |
| S06b | Nur P1 ignorierbar fehlend | LP1=1 in jedem Zustand | bestanden |
| S06c | Nur P1:nA=1,B/R fehlen,N=2 | Summe über B+R=1 exakt 2*pA*(1-pA), je 3/8 oder 1/2 | bestanden |
| S06d | Nur P1=(0,0,0),N=0 | LP1=1; Q exakt gleich S06b | bestanden |
| S07 | P1 durch gesperrten Prozent-Messkanal ersetzt,R0=0 | LP1=1-pA; Pr(H=1/4 bei R0=0)=1/2>0, kein Count/N erzeugt | bestanden |
| S08a | Nur D, keine P/Y | Q(tau=a)=3/4, Q(tau=b)=1/4 | bestanden |
| S08b | S01; ganze Kurven | Q(tau=b)=15281059/64186417; innerhalb psi normierte Kurvenformel exakt | bestanden |
| S08c-a | S01,tau=a fix | Alle je psi bedingten Verläufe stimmen mit S01 überein | bestanden |
| S08c-b | S01,tau=b fix | Alle je psi bedingten Verläufe stimmen mit S01 überein | bestanden |
| S08d | C=C*,theta=1,nur P2/D | Bei psi=+1 tau-Odds a:b=3/4 statt D-Odds 3; Faktor LP2(a)/LP2(b)=1/4 | bestanden |
| S09-base | Grund-Y=(+1,-1) | Positives Z, geschütztes q und positiver Support der Wahrheit | bestanden |
| S09-opposite | Nur Y=(-1,+1) | Z=(-1):19451/8388608,(+1):15701/8388608; Q verändert, q erhalten | bestanden |
| S09-noY | Nur Y weglassen | Z=(-1):551/65536,(+1):761/65536; Q verändert, q erhalten | bestanden |
| S10 | C=(1,1,1),tau=a; zwei künftige Y teilen theta | E(Indikator)=11/16, Cov=Var_theta(pY)=1/256>0 | bestanden |
| S11-equal | Nur UA/UB mit Prior 1:1,D fehlt | LP*LY gleich; Posterior-Odds je psi 1:1 | bestanden; nicht identifizierbar |
| S11-unequal | Derselbe Fall mit Prior 3:1 | Posterior-Odds je psi 3:1; keine zusätzliche Information | bestanden; nicht identifizierbar |
| S12 | 33 einzelne Mutationen I01–I09 | Alle erwarteten Fehlerklassen vor Q; keine Reparatur | bestanden |

**Altersverschiebung und Konflikt:** Vollständige, auch unveränderte
Marginaldifferenzen stehen in [comparisons.tsv](./A3_SYNTHETIC_AUDIT/comparisons.tsv).
Bei fixierter Kurve a ist Pr(C2=+1)=2774350/7147819, bei b dagegen
2868175/3778033; die Zuordnung von d1/d2 verschiebt die Information zwischen
den Klimaindizes. Die zeitlichen und an d1/d2 ausgewerteten C-Marginalen
sind beide gesichert. Tau vor der Cut-Normierung zu fixieren bewahrt q;
nachträgliches Konditionieren der gesamten q-Mischung auf tau würde psi
umgewichten. Deshalb erfolgt der bedingte Vergleich ausdrücklich je psi.

Beim Y-Wechsel von Grundfall zu Gegenbelegung sinkt Pr(C1=+1) von
106997259/128372834 auf 60742155/87257186, Pr(C2=+1) steigt von
211726975/449304919 auf 184854463/305400151. Pr(theta=1) sinkt von
228731436/449304919 auf 134772364/305400151; ohne Y bleibt theta bei 1/2.
Tau=b hat in den drei S09-Fällen die Massen 15281059/64186417,
11242915/43628593 und 103549/419311. Das sind bedingte Verschiebungen,
keine Verbesserung/Verschlechterung nach einem Score und kein Auswahlgrund.
Weder Wahrheitstreue noch schmalere Bänder waren Abnahmekriterien;
keine Intervall-/Deckungsbehauptung. S11 zeigt ausdrücklich, dass selbst
bekannte Wahrheit UB keinen eindeutigen Verlauf aus diesen Messungen ergibt.

## Fehlerablehnungen und absichtlich defekte Verfahren

Jede einzelne Mutation samt Eingang, Soll-/Ist-Fehlerklasse, Status und fehlender
Q-Ausgabe steht in [invalid.tsv](./A3_SYNTHETIC_AUDIT/invalid.tsv);
die Eingänge sind in [invalid_cases.py](../../scripts/netzel_a3_synthetic/invalid_cases.py)
festgelegt. Fremde Dateireferenzen werden nur als verbotener Text geprüft,
keine reale Datei geöffnet. Wiederholte Eltern-ID ist erlaubt; doppelte
Messbeleg-ID wird zurückgewiesen.

| Gruppe | Zahl | Tatsächliche Ablehnung aller Unterfälle |
| --- | --- | --- |
| I01 | 3 | INVALID_CALIBRATION: negativ, nicht normiert, fehlend |
| I02 | 1 | INVALID_NORMALIZER: Z(+1)=0 bei positiver q-Masse |
| I03 | 3 | INVALID_KERNEL: -1, +Unendlich, undefiniert am Wahrheitszustand |
| I04 | 6 | INVALID_OBSERVATION: N unbekannt/negativ/fraktional; n negativ/fraktional/falsche Summe |
| I05 | 6 | UNRESOLVED_STATUS: Sparse-Null, unbekannt/nicht berichtet/nicht bestimmt/außerhalb Abdeckung, unvollständige Kategorien |
| I06 | 4 | Drei MISSING_PRECISION_MODEL: Prozent, Rundungsnull, Nachweisgrenzennull; ein PSEUDOCOUNT_FORBIDDEN: N=100 |
| I07 | 3 | INVALID_CHRONOLOGY: gleiche, umgekehrte, außerhalb liegende Indizes |
| I08 | 3 | DUPLICATE_EVIDENCE: D, P als AP/NAP, Y-Replikat |
| I09 | 4 | OUT_OF_DESIGN: reale Referenz, fremde Kategorie, Einheit, Rückkanaloption |

| Absichtlich defektes Verfahren / Kontrolle | Korrekter Zeuge | Defekter tatsächlicher Befund | Ergebnis |
| --- | --- | --- | --- |
| Globale Normierung, S02 | psi(+1)=2/3 | 6/7; Differenz 4/21 | erkannt |
| Unabhängiges psi je Probe, S03 | Cov=1/72 | 0 | erkannt |
| Unabhängige Binomialkategorien, S05 | Pr(Summe ungleich 2)=0 | 673/1024 | erkannt |
| Separat marginalisiertes theta, S10 | Cov=1/256 | 0 | erkannt |
| Produkt der C-Marginalen, S04 | Pr(C2=C1)=3/4 | 9/16 | zusätzliche Negativkontrolle erkannt |
| Chronologie allein durch D, S08d bei psi=+1 | tau-Odds a:b=3/4 | 3 | Rückkanalverletzung erkannt |

Alle defekten Verteilungen sind vollständig und normiert in
[counterstates.tsv](./A3_SYNTHETIC_AUDIT/counterstates.tsv) erhalten.
Ihr Scheitern ist beabsichtigt, kein Ausfall einer wählbaren realen Modellvariante.

## Dauerhaft gesicherte Tabellen und Prüfgrenze

Alle folgenden Tabellenbytes gehören zum selben PR, insgesamt 191340 Bytes
TSV plus [Manifest](./A3_SYNTHETIC_AUDIT/manifest.json) mit Zeilenzahlen und
SHA256. Keine bloße Hash-/Reproduktionsreferenz anstelle der Dateien.

| Artefakt | Datenzeilen / Inhalt |
| --- | --- |
| [states.tsv](./A3_SYNTHETIC_AUDIT/states.tsv) | 1288: jeder F-/Q-Zustand beider Wege mit exakter Differenz |
| [normalizers.tsv](./A3_SYNTHETIC_AUDIT/normalizers.tsv) | 44: beide Z je Fall/psi |
| [marginals.tsv](./A3_SYNTHETIC_AUDIT/marginals.tsv) | 1056: psi, vollständiges C, C1–C3, tau, theta, C an d1/d2, psi-C/tau/theta; explizite Nullmassen |
| [predictive.tsv](./A3_SYNTHETIC_AUDIT/predictive.tsv) | 14: sämtliche Ausgänge S03/S05/S10 beider Wege |
| [likelihoods.tsv](./A3_SYNTHETIC_AUDIT/likelihoods.tsv) | 20: P1-Zeugen S06a–d/S07 für jedes psi/c |
| [percent_latent.tsv](./A3_SYNTHETIC_AUDIT/percent_latent.tsv) | 16: sämtliche H-Massen und R0-bedingte Massen |
| [witnesses.tsv](./A3_SYNTHETIC_AUDIT/witnesses.tsv) | 345: Kriterium, Soll, Ist, Differenz, Status; bei Tabellenvergleich Zahl ungleicher Einträge |
| [invalid.tsv](./A3_SYNTHETIC_AUDIT/invalid.tsv) | 33: jeder S12-Unterfall |
| [countermethods.tsv](./A3_SYNTHETIC_AUDIT/countermethods.tsv) | 6: defekte Zeugen und Unterschiede |
| [counterstates.tsv](./A3_SYNTHETIC_AUDIT/counterstates.tsv) | 105: alle Zustände der defekten Verfahren/Kontrollen |
| [comparisons.tsv](./A3_SYNTHETIC_AUDIT/comparisons.tsv) | 576: vollständige Marginaländerungen der vorgeschriebenen Variationen |
| [cases.tsv](./A3_SYNTHETIC_AUDIT/cases.tsv) | 24: alle Fallgruppen mit Eingang, Erwartung, Befund, Status, Abweichung und Grenze |

Das neue [Prüfmodul](../../tests/test_netzel_a3_synthetic.py) kontrolliert
Abnahme, unabhängige Fallabdeckung, tatsächliche Defekterkennung, archivierte
Dateibytes/Manifest, getrennte Imports/Designblob und Erweiterungsstopps.
Quellenzuordnung, 30 lokale Links/Anker, Diffumfang und Statushistorie geprüft;
729 geschützte Git-Blobs samt Dateimodi einschließlich eines Symlinks exakt
unverändert. Sicherungsstand und automatische CI stehen in HANDOFF/Status.

**Grenzen:** Keine reale Auswahl, Beschaffung, Bereinigung, Datenmatrix/Kohorte,
Typbrücke, Recherche oder erneute Dissertationprüfung; keine Fits, Scores,
Optimierung, Zufallssimulationen oder Kernläufe. Alle bestehenden Designs,
MODEL_CONTRACT, Modelle, Resultate, Register, Verträge, Importer, Flags,
Counts/Gruppen, Baseline und CLI/App unverändert. A1/A2 dokumentarisch
abgeschlossen; A4–A6 nicht ausgeführt. Parnell-Originalzugang blockiert,
Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben Grenzen; kein eigener Neuheitsbeitrag
abgegrenzt. Keine Unteragenten, Autorennachrichten oder Merges.
Genau ein begrenzter Folgeauftrag wird in HANDOFF vorbereitet; er wird hier
nicht ausgeführt und benötigt das nächste Nutzer-OK.
