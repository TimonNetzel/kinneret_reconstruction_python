# Prozentgewichtete Taxonrekonstruktion: Abgleich und verbesserter Vorschlag

31.10.2026. Vom Nutzer beauftragter Abgleich seiner neuen Idee mit Dissertation
und bisherigem Projektstand, einschließlich Bewertung und Verbesserung.
Basis `e6a1dbe3611b636edfe3f29d9e8ee5b785eb8527` aus
[PR #45](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/45).
Worktree `/workspace/netzel-percentage-review`, Branch
`codex/netzel-percentage-indicator-review`. Konzeptprüfung, keine Methodenwahl,
Implementierung oder wissenschaftliche Ausführung.
Gesichert als [PR #46](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/46)
gegen den #45-Branch, offen/ungemergt; Inhaltscommit
`4455eb01040fb2570c2c36451392929cd0ac6fad`, danach Sicherungs-/Übergabeverweise.
Eigene automatische CI bei Erstellung noch nicht abschließend geprüft.

## Bewertung und Anschluss an die Dissertation

**Die Grundarchitektur ist anschlussfähig:** Taxonspektrum → Klimaindikation,
Altersunsicherheit, Klimaprior und zusätzliche Proxys. Prozentwerte dürfen
relative Gewichte für die Auswahl von Indikatoren sein. Sie sind nicht automatisch
die von den vorhandenen Transferfunktionen beschriebenen Wahrscheinlichkeiten.
Die Idee ermöglicht einen anderen methodischen Weg als den gesperrten Counts-Pfad;
sie belegt dessen fehlende Mess-/Nennerregeln nicht nachträglich.

Originaltext gezielt gelesen: [Dissertation §2.3/2.4](../../paper/Phd_thesis_netzel(1).PDF),
insbesondere Gl. 2.10–2.15, Annahmen zu Gl. 2.13, Proxy-Pool Gl. 2.16/2.17;
[§3.2.2](../../paper/Phd_thesis_netzel_ch3.2.PDF), Nachbearbeitung und Kombination
der Rekonstruktionen, insbesondere gedruckte Seite 65. PDF-Text nur lesend
extrahiert; keine vollständige Dissertation oder deren Originalcode neu auditiert.
Der [bestehende Abgleich](../../abgleich_thesis.md) und der
[frühe Multiproxy-Plan](../../plan_multiproxy.md) dienen der Einordnung.

| Baustein der Idee | Dissertation / bestehender Stand | Verbesserung |
| --- | --- | --- |
| Taxa mit Transferfunktion auswählen | §2.4, Gl. 2.12/2.15: Mischung über ausgewählte Taxoninformation; §2.5: binäre Artenfunktionen | Beobachteten Pollentyp und botanische Art unterscheiden; zulässige Auswahl vorab festlegen, Abdeckung erhalten |
| Prozente als Wahrscheinlichkeiten | Als Verteilung für die Auswahl eines Indikators interpretierbar; frühes Planmodell verwendet relative Spektralgewichte | Auswahlwahrscheinlichkeit, Kornanteil und Artenvorkommenswahrscheinlichkeit getrennt führen |
| Übertragung Tiefe → Alter | §2.3, Gl. 2.10/2.11 integriert mit P(D\|A); §2.4 Gl. 2.15 nutzt diesen Anschluss | Ganze konsistente Alters-Tiefen-Kurven weiterreichen; geglättete Altersstufen nicht als neue unabhängige Proben zählen |
| Klima aus Transferfunktionen und Prior | §2.4 enthält einen uniformen Referenzprior und zusätzliche Klimaanker | Gemeinsames Klima pro Probe und klarer Evidenzterm; Klimavariable, Priorrolle und Support vorab festlegen |
| Zusätzliche Proxys über erklärte Varianz | Proxy-Pool Gl. 2.16/2.17; Beta(3,3) auf jeweiliger Referenzähnlichkeit | Eigenes Proxy-Beobachtungsmodell samt Reststreuung, Autokorrelation und Chronologie; Anteil erklärter Varianz als unsicherer Parameter |
| Zusammenführen der Ergebnisse | §3.2.2 wählt priornahe Stichproben und faltet PDFs unabhängiger Mittelgrößen | Vollständige Posteriorverteilung erhalten; MCMC-Ziehungen nicht zu unabhängigen zusätzlichen Rekonstruktionen erklären |

Der Originaltext spezifiziert in diesen gelesenen Abschnitten die innere
Kombination mehrerer Taxonfunktionen nicht vollständig als ausführbaren Code.
Die unten diskutierte Produktregel wird deshalb **nicht** als nachgewiesener
Dissertationsalgorithmus bezeichnet. Der alte Plan enthält Vorschläge, keine
heutigen Freigaben für Typbrücken, Probenwahl oder Varianten.

## 1. Prozentwerte: zulässige Rolle und verlorene Information

Sei p_sk der archivierte Anteil des Pollentyps k an der dokumentierten
Pollensumme von Probe s. Er beschreibt näherungsweise die relative Häufigkeit
eines klassifizierten Pollenkorns im betreffenden Nenner. Eine Funktion
m_k(c)=P(kartierter Arealindikator k=1 | Klima c) beschreibt etwas anderes.
Die sieben bestehenden Funktionen haben genau diese Kartenlabelrolle, keine
Pollenhäufigkeits- oder lokale Vegetationsmengenrolle:
[Artenfunktionsdesign](../taxon_transfer_2026_10_30/DESIGN.md).

Für eine vorab begründete, beobachtete und disjunkte Teilmenge K kann gelten:

```text
a_s = sum_{k in K} p_sk                     (Abdeckung des ursprünglichen Nenners)
w_sk = p_sk / a_s, für a_s > 0              (bedingte Indikator-Auswahlgewichte)
```

Die w summieren sich zu 1. Das ist eine zulässige Definition einer zufälligen
Indikatorauswahl, noch keine Likelihood für das gemeinsam herrschende Klima.
Nur bei kompatiblen Nennern und disjunkten Kategorien ist a_s eine Abdeckung;
überlappende Typen oder gemischte Pollensummen dürfen nicht zusammengerechnet
werden. Bei fehlender Abdeckungsinformation diese als unbekannt ausweisen.

Beispiel: Ein verwendbarer Typ mit 1 % Anteil erhält nach Einschränkung w=1,
obwohl 99 % des Spektrums nicht vertreten sind. Dieselbe normalisierte Zahl
bei 90 % Abdeckung ist nicht dieselbe empirische Evidenz. Deshalb a_s und
ausgeschlossene/fehlende Anteile mitführen, kein neues Mindestgate aus Ergebnissen
lernen. Bei a_s=0 oder unbekanntem Feldstatus keine Gleichverteilung/Nullfüllung.
Keine beliebige Schwelle oder Schärfung allein aus der Abdeckungszahl ableiten.

Prozente verlieren zudem Zählpräzision: 10/100 und 100/1000 ergeben beide 10 %,
aber unterschiedliche Zählunsicherheit unter demselben einfachen Zählmodell.
Ohne originales N darf weder N=100 noch eine andere effektive Kornzahl erfunden
werden. N_original, S_flag und n_ausgewählt bleiben verschiedene Größen.
Prozent-only-Auswertung benötigt ein eigenes Fehler-/Evidenzstärkemodell.
Ein nicht gespeicherter Wert bleibt fehlend, bis seine Bedeutung geklärt ist.
Auch beobachtete 0 % beweisen keine botanische Absenz; Rundungsnullen sind möglich.

## 2. Drei verschiedene Kombinationsregeln

**Taxonmischung:** Aus vorher festgelegten Klimadichten g_k(c) entsteht
g_s(c)=sum_k w_sk g_k(c). Dies beschreibt Klimaassoziationen eines zufällig
gewählten Indikators. Es erzwingt keinen gemeinsam verträglichen Klimazustand
aller Taxa. Wie g aus einer Vorkommensfunktion über Referenzprior und Normierung
gewonnen wird, muss explizit sein. Einen bereits enthaltenen Referenzprior
nicht still zusätzlich als neue Evidenz multiplizieren.

**Gewichtetes Klimamittel:** C_s=sum_k w_sk C_k mit taxonspezifischen C_k
definiert eine weitere Zufallsgröße. Es kann Amplituden dämpfen und zwischen
zwei verträglichen Klimabereichen liegen, obwohl der Mittelbereich selbst wenig
Unterstützung hat. Seine Verteilung ist weder die obige Mischung noch von selbst
der Posterior eines gemeinsamen lokalen Klimas. Für die gemeinsame Verteilung
der C_k sind Abhängigkeiten explizit nötig.

**Gemeinsamer Klimazustand:** Ein klar formulierter Kandidat für den nächsten
schriftlichen Entwurf wäre ein prozentgewichteter Indikatorscore:

```text
S_s(c) = sum_{k in K_s} w_sk log m_k(c)
p_lambda(c | w_s) ∝ pi_C(c) exp(lambda_s S_s(c))
```

Hier müssten gültige m_k>0 auf dem geprüften Support und proper pi_C sowie
eine begründete Evidenzstärke lambda_s vorliegen. Das ist als verallgemeinerte
Bayes-Auswertung eines Scores interpretierbar, **keine aus Pollenzählungen
abgeleitete Likelihood**. Die Prozentgewichte sind keine unabhängigen
Bernoulli-Beobachtungen. Ein q log m+(1-q) log(1-m)-Ansatz würde Anteile
stillschweigend zu binären Zielwahrscheinlichkeiten machen und wäre ebenso zu
begründen. Korrelationen und gemeinsame Fehler der Taxa bleiben relevant.
Keine willkürliche Epsilon-Reparatur bei m=0; Konflikt/Support explizit behandeln.

Der Score ist ein Vorschlag zur Diskussion, kein beschlossener realer Posterior.
lambda=100 aus Prozenten abzuleiten wäre eine erfundene Stichprobengröße.
Seine Einstellung an dieselben Referenzkurven, mit denen man später validiert,
wäre zirkulär. Unsicherheitsbänder blieben ohne gesonderte Prüfung bedingte
Modellbänder. Für eine empirische Kompositions-Likelihood wären moderne
Pollentypanteile und ein passendes Fehler-/Konzentrationsmodell nötig; eine
Normierung der vorhandenen Artenvorkommensfunktionen erzeugt das nicht.

**Empfehlung:** Für einen ersten klar begrenzten Anschluss die gemeinsame
Klimaindikation und Rolle des Scores schriftlich prüfen. Mischungsverteilung
und Klimamittel als andere Aussageziele ausdrücklich davon abgrenzen.
Keines dieser Ziele hier automatisch zur fossilen Hauptmethode erklären.

## 3. Chronologie: Unsicherheit weiterreichen statt nur glätten

Die Altersintegration der Dissertation ist konzeptionell sinnvoll. Eine einzige
geglättete Prozentkurve mit anschließend als unabhängig behandelten Altersstufen
verliert jedoch Abhängigkeiten. Eine Probe kann durch Altersunsicherheit auf
mehrere Alterszellen verteilt werden; sie liefert dadurch keine zusätzlichen
unabhängigen Beobachtungen. Nichtlineare Inversion und vorheriges Mitteln der
Spektren sind im Allgemeinen nicht vertauschbar.

Verbesserter Ablauf: Spektrum und Messstatus je Tiefe erhalten, gemeinsam gültige
monotone Alters-Tiefen-Kurven aus dem Chronologiemodell weiterreichen und für jede
Kurve die Klimaauswertung auf demselben physikalischen Probenbestand betrachten.
Anschließend Unsicherheit über die Kurven integrieren. Gemeinsame Altersfehler,
Sediment-/Probenintervalle und zeitliche Auflösung mitführen; kein unabhängiges
Altersziehen je Taxon oder Alterszelle. Übertriebene Auflösung nicht durch ein
feineres Ausgaberaster vortäuschen.

Für einen ersten modularen Entwurf kann die Chronologie allein durch die
Datierung informiert bleiben. Eine spätere Rückwirkung von Proxyähnlichkeit
auf Alter wäre eine eigene Modellentscheidung und benötigt Identifikationsprüfung.
Die Chronologie nicht nach maximalem R² verschieben oder Kurven entsprechend
auswählen. Diese Unsicherheitsintegration ist verschieden vom problematischen
Mitteln/Falten von Posteriorziehungen in Dissertation §3.2.2.

## 4. Klimaprior und zusätzliche Proxys

Mit „erklärter Varianz des Klimamittels“ wird hier das Proxy-Pool-Konzept der
Dissertation verstanden. Falls ein anderer Mechanismus gemeint ist, bleibt
diese Zuordnung eine explizite Interpretation der Nutzeridee.

Ein Klimaprior muss Zielvariable, Einheiten, Raum und Zeitraum treffen.
Die heutigen sieben Funktionen verwenden TANN/PANN und eine neue WorldClim-
Kalibrierung; die Dissertation verwendet CRU 4.01/1961–1990. Diese Versionen
nicht gleichsetzen; TDJF ist keine austauschbare Temperaturvariable. Ein
heutiger Anker betrifft zunächst den zugehörigen jungen Zeitraum, nicht alle
vergangenen Altersstufen. Priors und Klimaanker nicht pro Taxon vervielfachen.

Statt eine fertige Kurve nachträglich zu Referenzen hinzubiegen, zusätzliche
Proxys auf denselben latenten Klimaverlauf beziehen. Für Proxy r ein eigenes
Beobachtungsmodell mit Klimagröße, Richtung, Niveau/Skala, Reststreuung,
zeitlicher Autokorrelation, Geltungsfenster und eigener Chronologie angeben.
Ein Regressions-/Zeitreihenmodell kann den erklärten Varianzanteil rho_r
definieren und mit einem Prior versehen. Ein Wert um 0,5 kann eine unsichere
Vorannahme sein; er ist kein zu erzwingendes Ergebnis. Beta(R²) als Faktor
allein ist noch kein vollständiges generatives Modell der beobachteten Kurve.

Bei nur standardisierten Verlaufsreferenzen bleiben absolutes Klimaniveau und
Amplitude unbestimmt durch diese Referenz allein; Klimaindikation oder
unabhängige Anker müssen dazu Information liefern. Absolute Proxykalibrierung
wäre eine andere, separat belegte Beziehung. Gemeinsame Ursachen und Fehler
mehrerer Proxys berücksichtigen. AP/NAP aus denselben Pollendaten ist keine
unabhängige zweite Beobachtung. Zur Anpassung verwendete Referenzen nicht als
ungesehene Bestätigung ausgeben.

Volle Posteriorziehungen und ihre Mehrgipfligkeit erhalten. Keine Auswahl nur
priornaher Stichproben und kein Mitteln unabhängiger Kopien derselben Unsicherheit,
um schmalere Bänder zu erzeugen; siehe Dissertation §3.2.2 und bestehenden Abgleich.

## Konsequenz für den bisherigen Block und genau ein nächster Auftrag

Das Prozent-Indikatorziel kann **ohne behauptete Wiederherstellung von N_original**
entworfen werden. Es benötigt weiterhin belegte Prozentbasis, Feldstatus,
Pollentyp–Taxon-Bezüge, Support und Chronologie. Die offene EMPD2-Exportkette
wird damit kein erledigter Befund. W96 und Monticchio-P1 bleiben offen.
Die sieben Funktionen sind keine nachgewiesenen sieben fossilen Artzuordnungen.
Der Nutzer hat hier eine Konzeptprüfung beauftragt; die alte A/B-Frage wird
dadurch fachlich konkretisiert, aber kein realer Design-/Laufauftrag erteilt.

**Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK:** ein schriftliches
Vorabdesign für einen prozentgewichteten Indikatorscore mit gemeinsamem
Klimazustand erstellen, ausschließlich abstrakt und ohne reale Taxonauswahl.

- Neue `PERCENTAGE_INDICATOR_DESIGN.md` mit Aussageziel, Score-/Priorrolle,
  Evidenzstärke, Abdeckung/Fehlwerten, Abhängigkeiten, Chronologieintegration,
  Proxy-Anschluss und Grenzen. Mischung/Klimamittel als andere Ziele abgrenzen.
- Einen kleinen späteren synthetischen Prüfplan formulieren: widersprüchliche
  Indikatoren, gleiche Prozente bei verschiedenem N, geringe/fehlende Abdeckung,
  Prior-only, mehrdeutiges Klima sowie gemeinsame Altersverschiebungen.
  Keine Implementierung, Testausführung oder Simulation in diesem Folgepaket.
- Leseplan: AGENTS, HANDOFF, PROJECT_STATUS, dieses Konzeptpapier; bestehendes
  Artenfunktionsdesign und Counts-Design nur zur Abgrenzung. Dissertation nur
  bei einer konkreten ungeklärten Formelstelle erneut nachladen.
- Abschluss: mathematische Rollen und noch freie Entscheidungen benennen,
  keine empirische Kalibrierung/Freigabe behaupten. Alte Designs, Modelle,
  Resultate, Register, Verträge, Importer, Flags, Counts/Gruppen, Baseline und
  CLI/App unverändert. Keine Quellen-/Datenbeschaffung, Datenauswahl, Fits,
  Scores als Ergebnisläufe, Kernläufe, Kontakte, Merges oder Unteragenten.
  Eigener Worktree/Branch/PR, Status und genau ein Folgeauftrag sichern.

## Prüfung und Sicherung

Beim Einstieg #45 offen/ungemergt/mergefähig, Head entspricht der Basis;
automatische PR-CI Run 41198477399 erfolgreich. main `7f1826c` unverändert,
sieben vorherige Worktrees sauber. Original-PDF-Stellen und bestehende Designs
lesend verglichen; keine neuen Onlinequellen, Datenabfragen oder Modelle.
Nur dieses neue Papier plus Status/Übergabe verändert. Quellenstellen, Links,
Nenner-/Wahrscheinlichkeitsrollen und Diffumfang geprüft; keine Tests oder
wissenschaftliche Rechnungen. Gezielte PDF-Textextraktion ist reine Lesehilfe.
721 geschützte bestehende Git-Blobs bytegleich zur Basis; sieben vorherige
Worktrees sauber erhalten, keine ungesicherten Änderungen verworfen.
Erlaubter Drei-Dateien-Diff und `git diff --check` bestanden.
