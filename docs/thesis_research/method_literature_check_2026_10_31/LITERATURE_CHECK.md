# Literatur-Erstprüfung: Grundarchitektur bereits verwendet

31.10.2026. Nutzerfrage, ob der vorgeschlagene Ansatz bereits verwendet wurde.
Basis `6e28a8541f58a263cbd455d46b5ef8df101bae43` aus
[PR #47](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/47).
Worktree `/workspace/netzel-literature-check`, Branch
`codex/netzel-method-literature-check`;
[PR #48](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/48)
gegen den #47-Branch, offen/ungemergt; Inhaltscommit
`288d6e02eca647254c8242a671182c1ee84abffa`, danach Sicherungs-/Übergabeverweise.
Eigene automatische CI bei Erstellung noch nicht abschließend geprüft.

**Ja: Die allgemeine hierarchisch-bayessche Architektur, Polleninversion,
Zeitunsicherheit und gemeinsame Multiproxy-Inversion haben deutliche Vorläufer.**
Die neue Faktorisierung allein ist kein belegter Neuheitsbeitrag. Für die exakte
Kombination mit unserer Arealfunktions-Merkmalsbrücke, Prozent-only-Beobachtung,
Kalibrierung und Proxy-Kopplung ist Gleichheit oder Neuheit noch nicht belegt.

## Geprüfte Primärquellen und belastbare Reichweite

| Originalarbeit | Bereits verwendeter Baustein / Anwendung | Grenze des jetzigen Nachweises |
| --- | --- | --- |
| [Parnell et al., 2015, JRSS C, DOI 10.1111/rssc.12065](https://academic.oup.com/jrsssc/article-abstract/64/1/115/7067578) | Bclim: moderne Pollen–Klima-Kalibrierung, fossile Pollencounts, unsichere radiometrische Alter und Klimazeitreihenprior; Fallstudie Sluggan Moss, Nordirland | Verlagszusammenfassung und Metadaten gelesen; 2014 online, Heftjahr 2015. Keine identische Prozent-only-Likelihood oder vollständige Gleichungsübereinstimmung behauptet |
| [Werner & Tingley, 2015, DOI 10.5194/cp-11-533-2015](https://cp.copernicus.org/articles/11/533/2015/cp-11-533-2015.html) | Gemeinsame bayessche Klimainferenz und Aktualisierung der Wahrscheinlichkeiten unsicherer Altersmodelle; numerische Experimente | Abstract/Short summary gelesen. Experimenteller Methodennachweis, nicht als belegte reale Pollenanwendung ausgeben |
| [Bowen et al., 2020, DOI 10.5194/cp-16-65-2020](https://cp.copernicus.org/articles/16/65/2020/) | Joint Proxy Inversion: gemeinsame Inversion von Proxy-System- und Umweltzeitreihenmodellen; reale marine Mg/Ca- und delta18O-Anwendungen | Abstract, einschlägige Stellen §3 vor §3.1 und Beginn §3.1 gelesen. Altersunsicherheit wird in den gezeigten Anwendungen ausdrücklich nicht mitgerechnet; kein Pollen-Prozentmodell |
| [Menezes et al., 2026, arXiv:2609.08866v1](https://arxiv.org/abs/2609.08866v1) | Besonders nah: Null-/N-inflation berücksichtigendes multinomiales logistic-normal-Modell, BART-Kalibrierung, cut-posterior mit übertragener Kalibrierunsicherheit und Monticchio-Fallstudie | Preprint eingereicht 08.09.2026; arXiv-Metadaten/Abstract und HTML-Anfang gelesen. Counts-Modell, kein Nachweis identischer Prozent-only-, Multiproxy- oder gemeinsamer Chronologiekomponente. Zeitschriftenpublikation/Peer-Review hier nicht verifiziert |

Fundstellen: Parnell-Verlagsseite Summary; Werner Abstract Absätze 2–4 und
Short summary; Bowen Abstract sowie §3 Absatz vor §3.1 mit explizitem Verzicht
auf Behandlung der Altersunsicherheit; Menezes Abstract und v1-Submission.
[Menezes-HTML](https://arxiv.org/html/2609.08866v1) dient nur derselben
Preprintquelle, nicht einer weiteren unabhängigen Bestätigung.
Keine vollständige Methoden-/Codeprüfung dieser vier Arbeiten behauptet.

## Konsequenz für unseren Entwurf

Die Behauptung „Klima + Pollenkomposition + Chronologie + mehrere Proxys in einem
Bayes-Modell wurde noch nie gemacht“ ist durch diesen Erstcheck nicht tragfähig.
Insbesondere logistic-normal und Behandlung von Nullwerten sind für Pollen
bereits Gegenstand einer sehr nahen aktuellen Arbeit. Die Unterschiede Counts
versus Prozente, cut versus gemeinsamer Posterior und Arealindikator versus
kalibrierte Pollenantwort bleiben ausdrücklich bestehen; nicht alles gleichsetzen.

Ein möglicher eigener Beitrag müsste eine **präzise abgegrenzte und geprüfte
Weiterentwicklung** sein, etwa eine begründete moderne Kalibrierbrücke aus
Arealfunktionsmerkmalen mit nachgewiesenem Vorteil/Grenzen und konsistenter
Behandlung von Prozentrundung, Abdeckung und Chronologie. Dies sind Kandidaten,
keine bestätigten Alleinstellungsmerkmale. Eine andere Priorfamilie oder bloße
Zusammenstellung bekannter Terme belegt für sich keine methodische Neuheit.
Der Entwurf aus #47 bleibt ein Vorschlag; seine Beleglage wird hier ergänzt,
alte Designs/Quellenregister/Resultate nicht umgeschrieben.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Begrenzte methodische Abgrenzungsmatrix zu drei nahen Primärarbeiten erstellen.**
Neue NOVELTY_MATRIX.md plus Status/Übergabe. Ausschließlich die Methodenteile
von Parnell 2015, Bowen 2020 und Menezes v1 2026 gegen den #47-Entwurf lesen:
Beobachtung/Counts/Prozente/Nullen, Kalibrierbrücke, Klimaprozess,
Chronologie-Rückkopplung, Proxy-Abhängigkeit und Unsicherheitsvalidierung.
Je Merkmal Gleichung/Abschnitt, Gemeinsamkeit, konkrete Differenz und offene
Beleglage nennen; bei fehlendem Volltext Zugangslücke statt Vermutung.
Nach diesen drei Arbeiten stoppen, keine Schneeballsuche oder umfassende
Neuheitsbestätigung behaupten. Ergebnis: vorhandene Methoden nutzen und
höchstens einen überprüfbaren eigenen Beitragskandidaten präzisieren oder
festhalten, dass noch keiner abgegrenzt ist. Keine neue Modellentscheidung,
Implementierung, Datenbeschaffung/-auswahl, Matrix/Kohorte/Bereinigung,
Fits/Scores/Tests/Simulationen/Kernläufe, Kontakte, Merges oder Unteragenten.
Leseplan: AGENTS/HANDOFF/PROJECT_STATUS, diese Erstprüfung, #47-Entwurf und
genau die drei genannten Primärarbeiten. Eigener Worktree/Branch/PR;
alte Designs/Modelle/Resultate/Register/Verträge/Importer/Flags/Counts/Gruppen,
Baseline und CLI/App unverändert, W96 und reale Gates offen.

## Stand und Prüfung

Einstieg #47 offen/ungemergt/mergefähig, Head wie Basis, PR-CI 41201264089
erfolgreich; main `7f1826c`, neun vorherige Worktrees sauber. Recherche über
Webwerkzeug, nur Primärquellen für die Aussagen. Keine Projekt-/Quellcode-
Ausführung oder wissenschaftliche Rechnungen. Nur dieses neue Dokument plus
Status/Übergabe; Quellenkennungen, Aussagegrenzen, Links und Diffumfang geprüft.
723 geschützte bestehende Git-Blobs bytegleich zur Basis; erlaubter
Drei-Dateien-Diff und git diff --check bestanden. Vorhandene Arbeit erhalten.
