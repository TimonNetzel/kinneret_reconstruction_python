# Ersatzvorschlag zu Dissertation Gl. 2.16: gemeinsames Prozent-/Multiproxy-Modell

31.10.2026. Nutzerauftrag: Gl. 2.16 konkret weiterentwickeln, mathematisch und
klimatologisch begründen und mit Modellbausteinen füllen. Basis
`d0c25a1e0a798c4625e132d4c22a26886c12878e` aus
[PR #46](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/46).
Eigener Worktree `/workspace/netzel-joint-equation`, Branch
`codex/netzel-joint-percentage-equation`. Schriftlicher Methodenvorschlag,
keine Implementierung, reale Modellwahl oder wissenschaftliche Ausführung.
Gesichert als [PR #47](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/47)
gegen den #46-Branch, offen/ungemergt; Inhaltscommit
`cd0591d9eb6384aeb3496972fb4786fba6c2f2ed`, danach Sicherungs-/Übergabeverweise.
Eigene automatische CI bei Erstellung noch nicht abschließend geprüft.

## Neue Gleichung

\[
\boxed{
p(C,\tau,\Theta\mid Q,Y,D,E)\propto
\underbrace{p(C\mid\Theta_C)}_{\text{Klimaprozess}}
\underbrace{p(\tau\mid\Theta_\tau)p(D\mid\tau)}_{\text{Chronologie}}
\underbrace{p(\Theta\mid E)}_{\text{Kalibrierung/Priors}}
\prod_s\underbrace{L_P\!\left(Q_s\mid C(\tau_P(d_s)),\Theta_P\right)}_{\text{Pollenspektrum}}
\prod_r\underbrace{L_r\!\left(Y_r\mid\mathcal H_r[C,\tau_r],\Theta_r\right)}_{\text{weitere Proxys}}.
}
\]

C(t)=(T_ANN(t),log P_ANN(t)) ist ein gemeinsamer Klimaverlauf. Q sind beobachtete
Prozentkompositionen, Y externe Proxywerte, D Datierungsbeobachtungen; tau sind
monotone Alters-Tiefen-Funktionen aller beteiligten Archive. Theta enthält alle
Prozess-/Beobachtungsparameter. E ist unabhängig zugeordnete moderne
Kalibrierinformation; sie steckt genau einmal im Parameterprior p(Theta|E).
Für noch unkalibrierte Teile sind explizite proper Priors erforderlich.

H_r beschreibt die passende Klimagröße, räumliche Repräsentativität und zeitliche
Auflösung des Proxys. Ein entfernter Proxy beobachtet nicht fehlerfrei das lokale
Kernklima. Die Produktform setzt bedingte Unabhängigkeit gegebener latenter
Zustände voraus; bei gemeinsamen Fehlern sind gemeinsame Likelihoodblöcke nötig.
AP/NAP aus Q nicht zusätzlich als unabhängiges Y verwenden.

## Mit Leben gefüllt

**1. Klimaprior.** Vorschlag: bivariater Matérn-3/2-Gaußprozess für TANN/log PANN,
mit expliziter Kreuzkovarianz. Proper Priors auf Amplituden, positive Zeitskalen
und Korrelation; Skalen und Support müssen vor Anwendung klimatologisch
begründet werden, nicht aus gewünschten Kurven stammen. Unabhängige Klimaanker
als eigene Beobachtungsterme genau einmal hinzufügen, nur für ihre belegte
Epoche. Keine Wiederholung des heutigen Ankers über alle Zeiten. Gemeinsamer
Support der benötigten Funktionen verbindlich; keine stille Extrapolation.

**2. Pollenprozente als Komposition.** Für K vorab definierte, disjunkte und
kompatibel erhobene Pollentypen und ein Referenzteil K:

\[
\operatorname{alr}(q_s^*)=
\left(\log(q_{s1}^*/q_{sK}^*),\ldots,\log(q_{s,K-1}^*/q_{sK}^*)\right)
\sim\mathcal N_{K-1}\!\left(a+B h(C_s),\Sigma_P\right),
\qquad Q_s=\mathcal O_s(q_s^*).
\]

q* ist eine latente positive Komposition; O_s beschreibt dokumentierte Rundung,
Nachweisgrenzen und fehlende Komponenten. L_P integriert die latente Komposition
über diesen Beobachtungsprozess, statt Nullwerte durch Epsilon zu ersetzen.
Logistic-normal ist eine konkrete empirische Modellannahme, keine belegte reale
Fehlerverteilung. Strukturelle Nullen benötigen einen getrennten Zustandsprozess;
solange der Nullstatus unbekannt ist, keine reale Likelihood als fertig ausgeben.
Zeitlich/räumlich abhängige Proben benötigen entsprechende gemeinsame Fehler.

Als erklärende Merkmale können h_j(C)=logit m_j(C) der vorhandenen, im Support
gültigen Artenfunktionen dienen. **a, B und Sigma_P müssen anhand moderner
gepaarter Pollenspektren und Klimadaten kalibriert werden**. Dies ist eine
empirische Brücke Arealindikation→beobachtete Typkomposition, kein Identitäts-
oder Produktionsgesetz. Vorhandene Artenfunktionen bleiben unverändert; ohne
geeignete moderne Kalibrierung ist dieser Baustein nicht einsatzbereit.
Die Artenfunktionen erhalten TANN und PANN in ihren Originaleinheiten:
bei log P im Prozess vor dem Funktionsaufruf zurücktransformieren.

Die Auswahl vorhandener Artenfunktionen ersetzt weder die Definition der
beobachteten Pollentypen noch deren belegte Zuordnung. Originalabdeckung a_s und
ausgeschlossene Masse erhalten; eingeschränkte Komposition nur bei begründetem
Auswahlprozess. Keine reale Teilmenge hier gewählt. Prozent-only ist möglich,
aber unterschiedliche Messpräzision benötigt Zusatzinformation/Dispersion.
Sigma_P ist keine wiedergewonnene Kornzahl; moderne Streuung nicht ungeprüft
auf fossile Präzision übertragen. Ohne Replikate/vergleichbare Protokolle bleiben
Dispersion und Übertragbarkeit annahmenabhängig. Original-N nicht erfinden.

**3. Chronologie.** Datierungsspezifisches p(D|tau) und normierter Prior für
positive Akkumulationsschritte, etwa mit bestehendem Chronologiemodell. Ganze
monotone Kurven integrieren, keine unabhängige Altersziehung je Typ und keine
Vervielfachung einer Probe auf Alterszellen. C(tau(d_s)) ist zunächst die
Punktprobenannahme. Bei längeren Ablagerungsintervallen erwartete Zusammensetzung
über das Intervall integrieren, nicht bloß Klima vor der nichtlinearen Funktion
mitteln. Gleiches gilt für Proxy-Auflösung in H_r.

Die Gleichung ist ein vollständig gemeinsamer Ansatz: Proxys können tau
mitinformieren. Soll ausschließlich D die Chronologie bestimmen, muss dies als
separater modularer Ansatz mit unterbrochener Rückkopplung definiert werden;
es ist nicht derselbe gemeinsame Posterior. Vor realer Anwendung anhand
Datierungsinformativeness und möglicher Klimazeit-/Proxymehrdeutigkeit entscheiden.

**4. Proxy-Pool als Beobachtungsmodell.** Konkreter Kandidat im gültigen Fenster:

\[
Y_r=\alpha_r+s_r\!\left[
\sigma_r\sqrt{\rho_r}\,z(\mathcal H_r[C,\tau_r])+
\sqrt{1-\rho_r}\,e_r\right],\qquad
e_r\sim\text{AR(1) mit stationärer Varianz 1}.
\]

z standardisiert den modellierten Verlauf im vorab festgelegten Fenster;
s_r>0 ist die Proxy-Skala, sigma_r die belegte/unsichere Wirkungsrichtung.
rho_r ist ein modellierter Signalvarianzanteil, kein beobachtetes Stichproben-R².
Beta(3,3) kann als ausdrücklich zu prüfender Prior für rho an die Dissertation
anknüpfen; keine Verpflichtung auf 50 % erklärte Varianz. Bei unregelmäßigen
Zeiten zeitabstandsabhängige Korrelation statt gleicher AR-Schritte verwenden.
Ein konstanter Signalverlauf besitzt keine z-Standardisierung: unstandardisierte
Regression oder explizite Nichtschätzbarkeit, keine Epsilon-Normierung.
Standardisierte Proxys identifizieren allein weder absolutes Klimaniveau noch
Amplitude; unabhängige Klimaanker/Polleninformation bleiben nötig.

## Änderung gegenüber 2.16 und dem vorherigen Konzeptpapier

Die Originalfaktorisierung in [Dissertation §2.4, Gl. 2.16](../../paper/Phd_thesis_netzel(1).PDF)
ist nicht allein wegen eines bedingten Posteriorfaktors algebraisch falsch.
Der neue Vorschlag macht Datenentstehung, Parameter und Abhängigkeiten explizit:

- P(C|P,...) und P(P|...) durch eine Pollen-Beobachtungsverteilung plus
  Klimaprozess ersetzen. Jedes Datum hat genau eine Rolle; keine nachträgliche
  Priorvervielfachung aus taxonspezifischen Klimadichten.
- Beta(R²) als alleinigen Proxyfaktor durch L_r ersetzen; rho mit Prior ist
  Bestandteil eines expliziten Modells der Proxywerte.
- Altersunsicherheit als gemeinsame latente Chronologie behandeln, nicht als
  bereits erledigte Glättung. Chronologie-Rückkopplung ausdrücklich entscheiden.
- Kein freies Mittel taxonspezifischer Klimata und keine neue Unsicherheit
  aus der Faltung unabhängiger Kopien von Posteriorziehungen. Volle gemeinsame
  Posteriorverteilung berichten, keine Auswahl priornaher Stichproben (§3.2.2).

Für das nun präzisierte Ziel eines generativen Modells wird der früher
diskutierte Prozent-Indikatorscore hier als Hauptvorschlag zurückgestellt:
er wäre eine getrennte verallgemeinerte Bayes-Methode und ersetzt die moderne
Kompositionskalibrierung nicht. Kein altes Design oder Resultat geändert.
Mathematische Kohärenz unter diesen Annahmen ist kein Nachweis empirischer
Gültigkeit, Identifizierbarkeit oder wissenschaftlicher Neuheit. Neuheit wurde
nicht durch Literaturrecherche geprüft. W96, EMPD2-Archiv-/Messstatus, reale
Gates und Monticchio-P1 bleiben offen; kein realer Lauf autorisiert.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Abstrakten Identifizierbarkeits- und Kalibriervertrag dieses Gleichungsentwurfs
schriftlich festlegen**, neue MODEL_CONTRACT.md plus Status/Übergabe.
Für a/B/Sigma, Klimaprozess, Proxy-Skalen/rho und Chronologie Evidenzbedarf,
Fixierungen/Referenzkonventionen, erlaubte Annahmen und Abbruchfälle benennen.
Insbesondere Klima–Brücke–Proxy-Skala und Klima–Alter nicht als identifiziert
voraussetzen. Kleinen späteren synthetischen Prüfplan nur beschreiben:
Prior-only, fehlende/gerundete Nullen, Klima-/Skalenmehrdeutigkeit,
taxonübergreifende Fehler, Chronologieverschiebung und korrelierte Proxys.
Leseplan: AGENTS/HANDOFF/PROJECT_STATUS, dieser Entwurf; vorheriges Konzept-
und Counts-Design nur bei konkreter Abgrenzung. Keine Recherche/Datenbeschaffung,
reale Auswahl, Code-/Test-/Fit-/Score-/Simulations-/Kernläufe, Kontakte, Merges
oder Unteragenten. Alte Designs/Modelle/Resultate/Register/Verträge/Importer/
Flags/Counts/Gruppen, Baseline und CLI/App unverändert; eigener Worktree/Branch/PR.

## Stand und Prüfung

Basis-PR #46 offen/ungemergt/mergefähig, Head wie oben, PR-CI Run 41200027713
erfolgreich; main `7f1826c`, acht vorherige Worktrees sauber. Originalstelle
2.16 gezielt nachgelesen, frühere Designs nur abgegrenzt, keine neue Recherche
oder Rechnung. Nur dieses neue Dokument plus Status/Übergabe erlaubt.
Mathematische Rollen/Dimensionen, Referenzen und Links geprüft; 722 geschützte
bestehende Git-Blobs bytegleich, acht vorherige Worktrees weiterhin sauber.
Erlaubter Drei-Dateien-Diff und git diff --check bestanden; keine Tests/Läufe.
