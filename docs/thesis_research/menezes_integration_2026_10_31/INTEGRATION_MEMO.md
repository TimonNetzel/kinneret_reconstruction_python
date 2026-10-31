# Dissertation und Menezes konsequent verbinden

31.10.2026. Nutzerauftrag: vor dem Chatwechsel eine begründete Verbindung und
einen möglichen Mehrwert konkretisieren, Schwerpunkt Menezes. Basis
`877f572e40ea028bbd12e4127fbb16ac5f4acd9b` aus
[PR #48](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/48).
Worktree `/workspace/netzel-menezes-integration`, Branch
`codex/netzel-menezes-integration`; eigener PR nach Inhaltscommit.

## Belegter Anschluss

[Menezes et al., 2026, v1](https://arxiv.org/html/2609.08866v1), gezielt gelesen:
§1.3, §2.1, Anfang §2.2/2.3 und §5. Die Arbeit verwendet modern kalibrierte
Pollen-Countsantworten mit Null-/N-inflation, logistic-normal-Abhängigkeit und
BART. Ihr cut-posterior schützt diese Kalibrierung vor fossiler Rückkopplung.
Zeitliche Abhängigkeit und Altersunsicherheit fehlen in der aktuellen
Rekonstruktion ausdrücklich (§1.3/§5); die Autoren schlagen solche Erweiterungen
selbst vor. Ihre Klimaziele sind GDD5, MTCO und AET/PET (§4.3), nicht unsere
TANN/PANN. Preprintstatus, keine vollständige Code-/Methodenprüfung behauptet.

## Empfohlene Verbindung – Vorschlag, keine gewählte reale Methode

**Kalibriermodul aus dem modernen Pollenansatz, Rekonstruktionsarchitektur aus
der Dissertation weiterentwickeln.** Moderne Antwortunsicherheit vollständig
übertragen; Fossilproben über einen gemeinsamen Klimaverlauf und ganze
Chronologiekurven koppeln. Weitere unabhängige Proxys bekommen eigene
Beobachtungsmodelle mit Fehlern und Altersunsicherheit statt eines alleinigen
Beta(R²)-Faktors. AP/NAP aus demselben Spektrum nicht erneut unabhängig zählen.

Eine präzise modulare Zielverteilung für diesen Vorschlag wäre:

\[
p_{cut}(\psi,C,\tau,\theta\mid E,P,Y,D)
=p(\psi\mid E)\,
\frac{p(C,\tau,\theta\mid D)\,
L_P(P\mid C,\tau,\psi)\,L_Y(Y\mid C,\tau,\theta)}{Z(\psi)}.
\]

psi sind die modern kalibrierten Pollenparameter, E unabhängige moderne Daten,
P zulässige Pollenbeobachtungen, Y zusätzliche Proxys, D Datierungen und theta
nachgelagerte Klima-/Proxyparameter. Z(psi) normalisiert den **gesamten**
Rekonstruktionsblock über C/tau/theta für jeden psi-Wert. Damit bleibt die
Randverteilung von psi gleich p(psi|E). Dieselbe psi-Ziehung gilt gemeinsam
für alle betroffenen Proben, keine unabhängige Neukalibrierung je Alterszelle.

Dies präzisiert den vollständig gemeinsamen Vorschlag aus #47: Dort können
Fossilien/Proxys die Kalibrierparameter mitinformieren, hier wird genau diese
Rückkopplung unterbrochen. Das ist eine normierte modulare Zielverteilung,
kein gewöhnlicher vollständiger generativer Bayes-Posterior. Cut behebt keine
falsche Kalibrierung oder zeitliche Übertragbarkeit. Chronologie darf im
geschriebenen Rekonstruktionsblock durch P/Y aktualisiert werden; auch diesen
Rückkanal zu sperren wäre ein zusätzlicher eigener Cut, nicht dieselbe Gleichung.

Nicht ungeprüft die Mischung **bedingt normierter** Klimaposterioren durch
einen einzigen normierten Mittelwert der Likelihoods ersetzen. Die
Normierungsrollen der Menezes-Gleichungen (2)/(5) im anstehenden Vergleich
explizit nachvollziehen; hier keine Fehlerbehauptung über die Arbeit.

## Was aus der Dissertation erhalten und verbessert würde

- Erhalten: Taxonbezug, Chronologie, gemeinsamer Klimaverlauf, Klimaanker und
  mehrere Referenzproxies. Mathematische Rollen und Datenabhängigkeiten explizit.
- Ersetzen: unkalibrierte Gleichsetzung Arealvorkommen/Pollenanteil durch ein
  modernes Pollen-Beobachtungsmodell. Die sieben Artenfunktionen sind optional
  später zu prüfende Merkmale einer kalibrierten Brücke oder ein separater
  Vergleich; keine zweite unabhängige Pollen-Likelihood aus denselben Informationen.
- Erhalten: Idee erklärter Varianz. Neu formulieren als unsicheren Signalanteil
  innerhalb eines Proxy-Zeitreihenmodells mit Reststreuung/Autokorrelation.
- Verwerfen: priornahe Auswahl und künstliche Bandverengung durch Faltung von
  Posterior-Kopien. Mehrgipfligkeit und gemeinsame Unsicherheit berichten.

Menezes' Countsmodell verlangt echte Counts samt gültigem Zählnenner und
Beobachtungsstatus. Prozente nicht auf N=100 hochrechnen. Bei Prozent-only
bleibt eine andere Beobachtungsverteilung mit Rundungs-/Fehlwertprozess und
eigenem Präzisionsmodell erforderlich. Bekannte Datenlücken bleiben bestehen.
Ebenso Klimavariablen und Bezugsperioden vor Verbindung vereinheitlichen;
GDD5/MTCO/AET-PET nicht als TANN/PANN umbenennen.

## Überprüfbarer Mehrwert statt bloß zusätzlicher Komponenten

Beitragskandidat: eine zeitlich und chronologisch kohärente modulare
Multiproxy-Inversion, die Kalibrierunsicherheit erhält und Widersprüche
zwischen Pollen und unabhängigen Proxys sichtbar macht. Erweiterungen in diese
Richtung sind angekündigt und teilweise aus anderen Vorarbeiten bekannt;
Neuheit muss an der konkreten Lösung und belegtem Erkenntnisgewinn liegen.

Später vorab zu planende Vergleiche: Pollen-Basisverfahren; plus Chronologie/
Klimadynamik; plus externe Proxys. Gleiche Zielgrößen/Probenrollen und vorab
festgelegte Bewertung. Mehrwert anhand unabhängiger Vorhersagen, Datierungs-
verträglichkeit, Unsicherheitskalibrierung und Konflikterkennung zeigen.
Schmalere Bänder oder glattere Kurven allein sind kein Gütenachweis.
Zunächst gezielter synthetischer Test, danach getrennt freizugebende reale
Anwendung; Monticchio-P1 und W96 nicht über neue Literatur umgehen.

## Genau ein nächster Auftrag bleibt die vorbereitete Abgrenzungsmatrix

NOVELTY_MATRIX.md zu Parnell 2015, Bowen 2020 und Menezes v1 2026 erstellen,
nach drei Arbeiten stoppen. Schwerpunkt zusätzlich: gemeinsame versus cut-
Inferenz, bedingte Normalisierung, zeitliche/Chronologie-Erweiterung und
externe Proxy-Kopplung. Höchstens einen prüfbaren Beitragskandidaten abgrenzen.
Leseplan AGENTS/HANDOFF/PROJECT_STATUS, dieses Memo, #48-Erstprüfung, #47-Entwurf
und genau diese drei Primärarbeiten. Keine Implementierung, Datenbeschaffung/
Auswahl/Bereinigung/Matrix/Kohorte, Code-/Test-/Fit-/Score-/Simulations-/
Kernläufe, Kontakte, Merges oder Unteragenten. Alte Designs/Modelle/Resultate/
Register/Verträge/Importer/Flags/Counts/Gruppen, Baseline und CLI/App unverändert.
Eigener Worktree/Branch/PR; W96 und reale Gates offen. Der im vorherigen Turn
gegebene Nutzer-OK für diese Matrix bleibt gültig; die Nachfrage hier führt
die Matrix noch nicht aus. Im neuen Chat ist dafür kein zusätzliches OK nötig.

## Stand und Prüfung

Basis-PR #48 offen/ungemergt/mergefähig, Head wie Basis, CI 41205275351
erfolgreich, main `7f1826c`, zehn vorherige Worktrees sauber. Nur dieses Memo
plus Status/Übergabe. Gezielte Online-Methodenlektüre, keine Daten oder
wissenschaftliche Rechnungen. Quellenstellen, mathematische Rollen,
Normierungsgrenzen und Verweise geprüft; 724 geschützte bestehende Git-Blobs
bytegleich zur Basis. Erlaubter Drei-Dateien-Diff und git diff --check bestanden;
vorhandene Arbeit erhalten, keine Tests oder wissenschaftlichen Läufe.
