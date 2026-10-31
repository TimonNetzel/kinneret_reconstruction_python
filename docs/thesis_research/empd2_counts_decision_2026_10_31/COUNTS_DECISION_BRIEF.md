# Entscheidungsvorlage: blockierter realer EMPD2-Counts-Einstieg

31.10.2026. Nach Nutzer-OK ausschließlich schriftlich erstellt, auf Basis
`308efab1a7d55d2dcb2485f3346ba58d1d23e3bf` aus
[PR #44](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/44).
Worktree `/workspace/netzel-empd2-decision`, Branch
`codex/netzel-empd2-counts-decision`. Eigener PR nach Inhaltscommit.

**Ergebnis:** Die vorhandenen Belege tragen keine Freigabe des realen
Counts-T1. Zwei mögliche weitere Aufträge sind unten entscheidbar abgegrenzt.
Die Wahl bleibt offen. Das OK für diese Vorlage beschließt weder ein neues
Forschungsziel noch eine Null-/Nennerregel, Modellfamilie oder Datennutzung.
Bis zu einer ausdrücklichen Entscheidung gelten die bestehenden Sperren.

## Entscheidungsgrundlage und Nennerrollen

Nur bestehende Repository-Belege gelesen, keine neue Quellenrecherche oder
Originaldatenabfrage. Der [INPUT_CONTRACT](../empd2_counts_readiness_2026_10_31/INPUT_CONTRACT.json)
setzt `modern_count_training_ready=false`; [SOURCE_GAPS](../empd2_counts_readiness_2026_10_31/SOURCE_GAPS.md)
beschreibt die damaligen offenen Regeln. Die anschließenden
[Archivbefunde](../empd2_archive_lineage_2026_10_31/ARCHIVE_LINEAGE.md),
[LINEAGE_EVIDENCE](../empd2_archive_lineage_2026_10_31/LINEAGE_EVIDENCE.json) und
[DECISION_GAPS](../empd2_archive_lineage_2026_10_31/DECISION_GAPS.md) begrenzen
diese Fragen präziser, ohne den alten Vertrag zu überschreiben:

- Die Originaldatei ist identifiziert; ihre genaue Erzeugungs-/Exportkette
  bleibt im gelesenen Umfang unbelegt. Die begrenzte Suche ist beendet.
- `p_counts.count` enthält heterogene Messgrößen. Erhaltenes Volumen 7,4 ist
  keine gebrochene Pollenzählung und darf nicht gerundet oder als Kornzahl
  behandelt werden. Die 36/40-Gruppendifferenz wird nicht angeglichen.
- Ein fehlendes Sample×Typ-Paar besitzt keinen belegten Messstatus. Der
  sichtbare Importpfad allein beweist keine allgemeine archiveigene Nullregel.
- Typkonzepte/Restkategorie, Klimanormalperiode und zeitliche Eignung sowie
  unabhängige Bestätigung bleiben weitere reale Voraussetzungen.

| Größe | Bedeutung | Vorhandene Aussage / Grenze |
| --- | --- | --- |
| `N_original` | Ursprünglicher gemessener Zählumfang einschließlich Erfassungs-/Ausschlussprozess | Nicht unabhängig belegt; weder Summe aller gespeicherten Werte noch Gruppenflag allein ersetzt ihn |
| `S_flag` | Summe gespeicherter Werte in flag-1-Gruppen | Für drei bekannte Zeugen mit gelieferten Prozenten kompatibel; kein allgemeiner Nachweis von N_original oder globalem Algorithmus |
| `n_ausgewählt` | Umfang einer später fachlich definierten, disjunkten Beobachtungsteilmenge | Keine Teilmenge gewählt; benötigt eigenen belegten oder ausdrücklich angenommenen Auswahl-/Erfassungsprozess |

Die Kompatibilität der Prozentdarstellung wird aus dem Archivpaket übernommen,
nicht neu berechnet. Sie schließt weder die Messstatuslücke noch eine
Typ-/Artbrücke. Auch ein bedingtes Kompositionsziel kann diese Lücken nicht
durch Umbenennen des Nenners beseitigen.

## Zwei Handlungswege zur ausdrücklichen Wahl

| Entscheidungspunkt | A: Reales Counts-T1 zurückstellen | B: Separates annahmenabhängiges Forschungsziel entwerfen lassen |
| --- | --- | --- |
| Benötigte Nutzerentscheidung | Die Zurückstellung als Arbeitsstand festhalten; keine automatische Fortsetzung der Archivsuche | Einen eigenen schriftlichen Designauftrag für ein anderes, ausdrücklich bedingtes Ziel erteilen; Ziel und Annahmen erst darin zur Entscheidung vorlegen |
| Was unverändert bleibt | Gesperrte Datei, Vertrag, Designs, Modelle und reale Gates | Dieselben bestehenden Referenzen und Gates; keine Umdeutung des historischen Vertrags oder bestehender Resultate |
| Kleinster Evidenzbedarf für den nächsten Schritt | Für die Dokumentation der Zurückstellung keine neue Quelle. Für ein späteres Wiederöffnen: präzise neue, archivbezogene Evidenz zu angewandter Version/Verarbeitung und Mess-/Nennerstatus; wiederholte Formatgleichheit genügt nicht | Für einen rein abstrakten schriftlichen Entwurf genügen die gesicherten Lücken. Er muss Beobachtungseinheit, Wert-/Einheitenbedeutung, Typabgrenzung, Messstatus/Selektion und Nennerrolle ausdrücklich als belegt, angenommen oder nicht identifizierbar ausweisen |
| Zusätzliche Mindestbasis vor jeder realen Nutzung | Nachgewiesene zulässige disjunkte Beobachtungspartition, Mess-/Ausschlussprozess und sachgerechter Nenner; danach Klima-/Zeitkontext und vorab festgelegte Prüfung separat erfüllen | Für konkrete Eingänge müssen Wertbedeutung, Typ-/Einheitenzuordnung und Beobachtungs-/Auswahlprozess ausreichend abgesichert oder als ausdrücklich genehmigte Annahmen begründet sein. Bleibende Nichtidentifizierbarkeit und deren Folgen müssen benannt werden; Klima-/Zeitkontext und unabhängige Bestätigung bleiben erforderlich |
| Zulässige Schlussfolgerung | Die archivgetreue reale Counts-Interpretation ist mit diesen Belegen noch nicht freigegeben | Ein späteres Ergebnis dürfte nur Aussagen über das festgelegte Ziel unter seinen Annahmen tragen; die historische Erzeugungskette wäre dadurch weiterhin nicht nachgewiesen |
| Unzulässige Schlussfolgerung | Das Archiv sei fehlerhaft, wissenschaftlich wertlos oder grundsätzlich nie nutzbar | Annahmen seien Quellbelege; fehlende Zeilen seien automatisch Messnullen; S_flag oder n_ausgewählt sei N_original; empirische Validierung oder Art-/Fossilfreigabe sei bereits erreicht |
| Eindeutige Stopbedingung | Keine reale Counts-Weiterarbeit und keine Ersatzsuche ohne neuen begrenzten Auftrag und neue präzise Evidenz; bei weiterem Nichtbeleg zurückgestellt bleiben | Entwurf stoppen, wenn das Ziel nur mit stiller Nullfüllung/Rundung, unbegründeter Taxon-/Nennerwahl oder unbeobachtbarer Wahrheit als geprüftem Fakt formulierbar wäre. Ohne ausdrücklich akzeptiertes Design und reale Eingangsprüfung keine Implementierung, Datenwahl oder Läufe |

**A erhält die gegenwärtigen Sperren als bewusst dokumentierten Arbeitsstand.**
Es behauptet keine Nichtexistenz möglicher Archivbelege. Hier wird weder neue
Recherche vorbereitet noch ein Voll-Dump, anderes Release oder Autorenkontakt
als nächste Pflichtaufgabe angesetzt.

**B wäre eine Änderung der Forschungsfrage und zunächst nur ein Designauftrag.**
Ein mögliches Ziel dürfte sich nur auf klar definierte beobachtete Größen unter
offengelegten Annahmen beziehen. Diese Vorlage wählt keine konkrete Zielvariable,
Teilmenge, Klimagröße, Likelihood oder Modellfamilie. Insbesondere darf die
Zusammensetzung der jeweils gespeicherten positiven Zeilen nicht ohne eigenen
Nachweis als eine für alle Proben gleich erhobene, vollständige Komposition
behandelt werden. Bedingung auf eine Teilmenge benötigt die Bedeutung ihrer
Mitgliedschaft und Auslassungen; sie repariert unbekannte Erfassungsprozesse nicht.

## Gemeinsame Grenzen vor einem diagnostischen Kernlauf

Die [Roadmap T1/T3](../TAXON_PILOT_ROADMAP.md) verlangt vorab begründete
Beobachtungsannahmen und gesperrte reale Eingänge. Ein späterer B-Entwurf würde
daher zunächst erneut T1 betreffen und T3 nicht überspringen. Synthetische
Methodenprüfung identifiziert keine unbekannte historische Archivverarbeitung.
Moderne Pollentypkomposition ist weder eine kartierte Artvorkommenswahrscheinlichkeit
noch automatisch lokale Artenpräsenz. Produktion und Transport sind damit nicht
einzeln identifiziert. Klimanormalperiode/zeitliche Eignung dürfen nicht aus
Dateiname, räumlicher Nähe oder numerischem Join gefolgert werden.

W96-Bestimmungs-/Artbrücken, T0, reales T1/T3/T4 und Monticchio-P1 bleiben
offen. Keine Kernwahl, Modellvariante oder Mindestabdeckung wird beschlossen.
Bestehende Analog-Baseline und CLI/App bleiben Referenzen; ein grüner CI-Lauf
ersetzt keine dieser wissenschaftlichen Voraussetzungen.

## Konkrete Entscheidung und genau ein vorbereiteter Folgeauftrag

**Offene Entscheidung:** A ausdrücklich zurückstellen oder B einen separaten
schriftlichen Ziel-/Annahmenentwurf beauftragen. Beide Wege sind Optionen,
keiner ist durch das bisherige OK gewählt. Ohne Wahl bleibt der reale Einstieg
gesperrt; aus Schweigen oder technischem PR-Status folgt keine Entscheidung.

**Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzerentscheidung und OK:**
Die ausdrücklich gewählte Option in einer neuen `COUNTS_DECISION_RECORD.md`
verbindlich dokumentieren und den daraus folgenden Arbeitsumfang abgrenzen.

- **Erforderlicher Eingang:** ausdrückliche Wahl A oder B. Ein allgemeines OK
  ohne zuordenbare Wahl ersetzt diese fachliche Entscheidung nicht.
- **Umfang:** nur dieses Protokoll plus Status/Übergabe, eigener Worktree/Branch/PR.
  Bei A Zurückstellung und Wiederöffnungskriterium festhalten. Bei B nur den
  begrenzten späteren Designauftrag konkretisieren; noch kein Design entscheiden
  oder ausführen. Keine neue Recherche, Datenquelle, Originaldatenabfrage,
  Daten-/Taxon-/Nennerwahl, Matrix/Kohorte, Bereinigung, Implementierung oder Lauf.
- **Leseplan:** AGENTS, HANDOFF, PROJECT_STATUS und diese Vorlage;
  INPUT_CONTRACT/Archivbelege nur bei konkreter Zuordnungsfrage nachschlagen.
- **Abschluss:** Wahl und ihre ausdrückliche Grundlage, erlaubt/nicht erlaubt,
  erforderliche Evidenz und Stopbedingung prüfbar sichern. Keine Freigabe aus
  der Protokollierung ableiten. Keine Unteragenten, Autorennachrichten oder Merges.

## Stand- und Dokumentprüfung dieses Pakets

Basis-PR #44 beim Einstieg offen, ungemergt und mergefähig; Head `308efab…`,
automatische PR-CI Run `41197759461` erfolgreich. Remote-main unverändert
`7f1826cffddc8893ba6354cd3cca3fa43954b9be`; Remote-Head #44 entspricht der Basis.
Alle sechs vorherigen Worktrees einschließlich des Archiv-Worktrees sauber.
Nur eine neue Vorlage plus HANDOFF/PROJECT_STATUS erlaubt. JSON-Eingänge,
Verweise, Aussagen und Diffumfang geprüft; keine eigenen Tests.
720 bestehende Git-Blobs außerhalb der erlaubten Status-/Übergabedateien
bytegleich zur Basis, einschließlich alter Archivpakete/Verträge/Designs.
Keine ungesicherten Änderungen in den sechs vorherigen Worktrees; eigener
Diff auf genau drei Dateien begrenzt, `git diff --check` bestanden.
Die vorhandene SQLite wurde nur zur Erhaltungskontrolle gehasht, nicht geöffnet:
SHA-256 `5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`.
Bestehende Designs/Modelle/Resultate/Verträge/Register/Importer/Flags/Counts/
Gruppen, Baseline und CLI/App bleiben unverändert. Keine Unteragenten,
Autorennachrichten, Merges, neuen Daten oder Projekt-/Quellcodeausführung.
