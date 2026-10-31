# Begrenztes synthetisches T2: Zählkomposition und Typgemische

31.10.2026. Auftrag durch erneutes Nutzer-OK hier im bestehenden Chat;
Basis PR #40 bei `19a75426320d0143c09cd00c756bc31f97cef0d0`.
Eigener Branch `codex/netzel-synthetic-counts-t2`, Worktree
`/workspace/netzel-counts-t2`; sieben ursprüngliche Worktrees sauber erhalten.
[Git-/PR-/CI-Snapshot](./git_snapshot.json): #31–#40 offen, main `7f1826c`,
#40 technisch CLEAN/MERGEABLE mit vier erfolgreichen CI-Checks, kein Merge.
Gesichert als [PR #41](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/41)
gegen `codex/netzel-counts-observation-design` (#40), Inhaltscommit `fa61c11`;
danach nur PR-/Übergabeverweise. Eigene CI bei Erstellung noch laufend.

**Dieses synthetische Paket ist abgeschlossen:** alle 52 vorab geplanten gültigen
Posterioren berechnet, zwei zusätzliche unmögliche Fälle ausdrücklich nicht
schätzbar. Keine realen Modelle, Counts oder wissenschaftlichen Kernläufe.
Reales T1, W96/Artbrücken, T0/T3/T4 und Monticchio-P1 bleiben offen.

## Vor Ausführung gesichert

Der unveränderte [Design-/Prüfplan](../counts_observation_design_2026_10_31/SYNTHETIC_PLAN.md)
und dessen [JSON](../counts_observation_design_2026_10_31/synthetic_plan.json)
legen Parameter, 52 Fälle, Priors, Seeds und Kriterien fest. Ausführbarer Code,
Tests und separater Prüfer wurden als `5b7e67a` committed und gepusht.
Erster pytest-Aufruf: Sammlung wegen eines Syntaxfehlers im Testdekorator
abgebrochen; Reparatur als `220ef55` erneut **vor erstem Ergebnislauf** committed
und gepusht. Danach keine wissenschaftliche Parameter-/Kriterienänderung.
[Operative Umsetzung und Reparatur](./IMPLEMENTATION.md).

## Bekannte Zustände: alle vorgesehenen base-Varianten

Je drei Zustände × drei Seeds, derselbe Datensatz für beide Priors. Fehler ist
der Abstand des Posterior-Mittels zur Wahrheit im dimensionslosen Spielzeugraum.
Kein empirischer °C-/mm-Fehler und kein 95%-Deckungsnachweis. Wiederverwendete
Seeds zwischen N/Zuständen sind keine unabhängigen Bestätigungsserien.

| N | Prior | Fälle | Mittlerer Fehler | Größter Fehler | Wahrheit in 95%-HPD | Mittlere HPD-Zellzahl | Größte Randmasse |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | uniform | 9 | 0,36575 | 0,63892 | 9/9 | 130,67 | 0,31023 |
| 8 | informativ | 9 | 0,38615 | 0,61566 | 8/9 | 114,44 | 0,23120 |
| 40 | uniform | 9 | 0,19421 | 0,40111 | 9/9 | 34,56 | 0,23805 |
| 40 | informativ | 9 | 0,19704 | 0,34203 | 9/9 | 32,78 | 0,14553 |

Mehr Körner verringern hier den mittleren Fehler beider Priors; dies ist keine
universelle Garantie. Der informative Prior verschlechtert in beiden N-Gruppen
den mittleren Fehler, verbessert aber jeweils den größten Fehler und verkleinert
die HPD-Sets. Einzige verfehlte bekannte Wahrheit aller 42 Auswertungen mit
Wahrheit: `base_s0_N8_seed201:base:informative` (Wahrheit (0,0)). Alle übrigen
bekannten Fälle enthalten die Wahrheit im HPD-Set; keine Auswahl danach.

## Gemeinsames H: gleicher Einzelkornrand, andere Abhängigkeit

Ein vorab festgelegter Datensatz bei (0,0), N=40, Seed 303. Ein H wird pro
Probe gezogen. Die bewusst missspezifizierte Inferenz zieht H dagegen effektiv
je Korn neu. Einzelkorn-Marginalen stimmen in beiden Modellen überein.

| Inferenz | Prior | Fehler des Mittels | HPD-Zellen | Wahrheit in HPD |
| --- | --- | ---: | ---: | --- |
| korrekte gemeinsame Mischung | uniform | 0,14738 | 36 | ja |
| korrekte gemeinsame Mischung | informativ | 0,09218 | 34 | ja |
| neu je Korn, missspezifiziert | uniform | 0,07722 | 30 | ja |
| neu je Korn, missspezifiziert | informativ | 0,05997 | 29 | ja |

In **diesem einzelnen Datensatz** liegt das Mittel des missspezifizierten Modells
näher an der Wahrheit. Das widerlegt die generative Annahme nicht und legitimiert
keine Übernahme des falschen Modells. Beide Ergebnisse sind vollständig berichtet.
Strukturprüfung bei c=0, N=2, beide Körner Typ 1: gemeinsame Wahrscheinlichkeit
17/288, je-Korn-Modell 25/576, Unterschied 1/64. Alle fünf Einzelkorn-Marginalen
über den gesamten Support geprüft. Kein Modellwettbewerb, kein SBC.

## Alle weiteren Grenzfälle

| Fall | Ergebnis |
| --- | --- |
| merged, 2 Posteriore | ±u-Likelihoodsymmetrie bis 1,43×10⁻¹⁴. Uniform je 0,49329 Masse auf u<0/u>0, u-Mittel etwa 0 bei wahrer 0,5. Informativ negative/positive Masse 0,20351/0,77830 und u-Mittel 0,34396. Fehler 0,50177→0,16607 stammt auch aus Priorpräferenz; Vorzeichen bleibt durch die Daten strukturell unbestimmt. |
| missing_group, 2 | N=8, drei Kategorien beobachtet, vier Körner gemeinsam in zwei fehlenden Kategorien. Exakte kollabierte Likelihood stimmt mit Summe aller fünf möglichen Aufteilungen überein; nicht nullgesetzt. |
| conditional_subset, 2 | Ausgewählter Nenner n_O=5 unter ausdrücklich künstlichem Auswahlprozess; keine Wiederverwendung von vollem N=8 in der Inferenz. Posterioren vollständig gespeichert; kein Vergleich ihrer Evidenzen mit dem anderen Beobachtungsprozess als Qualitätsrangliste. |
| all_missing, 2 | N=40 bekannt, null Kategorien beobachtet. Posterior gleich Prior bis 2,61×10⁻¹⁸, Log-Evidenz exakt 0. |
| zero_total, 2 | Belegter vollständiger N=0-Datensatz: ebenfalls Prior-only. Datenstatus bleibt vom fehlenden Spektrum getrennt. |
| observed_zero, 2 | (0,2,2,2,2), N=8 gültig; beobachtete Null ist keine botanische Absenz und nicht fehlend. |
| unmöglich, 2 zusätzliche Negativfälle | Positive Kategorie hat in allen 441 Zuständen Wahrscheinlichkeit 0. Beide Priors: `not_estimable`, kein Posterior und keine Epsilon-Reparatur. Diese Fälle sind keine ausgelassenen der 52 gültigen Szenarien. |
| Ungültige Eingänge | Counts/Total, Nenner/ausgewählter Prozess, Form, Typdimension, A, Fluss/r/d/B, NaN/Inf, Klimasupport und unbekannte Bezeichner explizit abgewiesen. Strukturelles pi=0 ist nur bei positivem Count unmöglich. |

Skalierung aller Flüsse ×7, Produktions-/Klimakompensation und Austausch von
0,05 zwischen gleich zugeordneten Quellen numerisch invariant. Es wurden dabei
keine Faktoren gefittet. Bedingt bekannte Parameter erzeugen rechnerische
Identifikation, ersetzen aber keine unabhängigen realen Brückenbelege.

## Prüfung und dauerhafte Ausgaben

- **51 gezielte Tests bestanden**, null Fehler/Fehlschläge/Skips nach korrigierter
  Sammlung. [Testprotokoll](./tests.json). Kein gesamter Repositorytest gestartet.
- Separater [Prüfer](../../../tools/taxon_counts/verify.py) ohne Produzentenimport:
  25 Datensätze einschließlich Generator/Seed/Masken neu erzeugt, alle
  52×441 Posteriormassen und Diagnosen nachgerechnet; maximale absolute
  Massendifferenz 1,263×10⁻¹⁵, Mittelwertdifferenz 7,22×10⁻¹⁶.
  Zwei Negativfälle und neun Manifestdateien geprüft.
  [Prüfprotokoll](./verification.json).
- Zwölf PMF-Normierungsprüfungen für base/shared, drei Wahrheiten und N=0/8;
  bei N=8 jeweils sämtliche 495 Countvektoren. Gleichungen, probenweite Mischung,
  Missingnesssumme und analytische Invarianten zusätzlich gezielt geprüft.
- Frischer Zweitlauf: sämtliche **fünf Produktionsdateien bytegleich**.
  [Vergleich/Hashes](./repeat_verification.json).
- [Erhaltungs-/Dokumentationsprüfung](./scope_verification.json): 682 geschützte
  Git-Blobs einschließlich Symlink unverändert gegen die Basis; sieben bisherige
  Worktrees sauber. Nur Status/Übergabe als bestehende Dokumente geändert.

Gemeinsame Abhängigkeiten des Produzenten und separaten Prüfers sind Python und
NumPy; kein unabhängiger menschlicher oder palynologischer Review behauptet.
Die fünf Dateien unter [outputs](./outputs/manifest.json) enthalten Beobachtungen,
52 vollständige diskrete Massen, Diagnosen einschließlich Negativfällen,
Laufumgebung und Manifest. Alle Dateibytes in Git gesichert; /tmp ist nur
Vergleichsort. [Reproduktionsbefehle](./REPRODUCE.md).

Keine realen Modell-/Typentscheidungen aus äußeren Scores. Unsicherheit umfasst
nur den vorgegebenen künstlichen Zähl-/H-Prozess und beide diskreten Priors;
keine reale Produktions-, Transport-, Klassifikations-, Karten-, Modell- oder
Chronologieunsicherheit. Bestehende Forschungsdateien und CLI/App unverändert,
keine Unteragenten, Merges oder Autorennachrichten.

Nächster begrenzter Auftrag ist ein **Datenrollen-/Readiness-Audit eines modernen
Counts-Kalibrierungswegs**, siehe [HANDOFF](../HANDOFF.md), wartet auf Nutzer-OK.
Der synthetische Kandidat ist kein fertiges reales Verfahren.
