# Netzel: Übergabe nach dokumentarischer A3-Auftragsabgrenzung

## Gesichertes Ergebnis und Stand

Die ausdrücklich beauftragte **A3-Auftragsabgrenzung ist abgeschlossen**.
Nur diese HANDOFF und ein kurzer [Status](./PROJECT_STATUS.md) wurden gepflegt;
kein neues Konzeptpapier. A3 selbst und A4–A6 sind nicht begonnen.
A1 und A2 bleiben als Dokumentationspakete abgeschlossen.
[MODEL_CONTRACT](./MODEL_CONTRACT.md) bleibt unverändert: Planfestlegungen
und OFFEN-Felder sind getrennt, keine reale Modell-/Datenfreigabe.

Grundlage der Abgrenzung sind
[Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
**ausschließlich Zeile A3** in
[Plan §2](./MODULAR_RECONSTRUCTION_PLAN.md#2-arbeitspakete-mit-prüfbarem-ende),
[Arbeitsmodus §4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus)
und die unten benannten Vertragsabschnitte. Die A3-Zeile verlangt zuerst ein
gesperrtes Design, erst danach separat beauftragte Implementierung/Ausführung.
Hier werden nur dessen Auftrag und Abnahmegrenze festgehalten; keine
Szenarien, künstlichen Wahrheiten, Verteilungen oder Parameter festgelegt.

Basis: [PR #52](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/52),
offen/ungemergt und mergebar, Remote-Head
`1bf23cded54a101aada68b79acf89a0b7a272fb4`; gesicherter A2-Stand
`7deff9f002bd8f865f63ef5812c843d6c0a6ec9b` ist dessen Vorfahr.
Automatische CI [41219952795](https://github.com/TimonNetzel/kinneret_reconstruction_python/actions/runs/41219952795)
am Übergabe-Head erfolgreich. Main weiterhin `7f1826c`.
Fünf vorgefundene Worktrees sauber, keine Stashes; ungesicherte Arbeit nicht
angetastet. Eigener Worktree `/workspace/netzel-a3-task-boundary`, Branch
`codex/netzel-a3-task-boundary`;
[PR #53](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/53)
gegen den #52-Branch, offen/ungemergt und mergebar. Inhaltscommit
`82598ed1825771656afe5c4f4f34afb3e85e6f51`; danach nur Sicherungsverweise.
CI 41220607622 zum Inhaltscommit bei Abfrage in Arbeit; letzter Remote-Head
und aktueller CI-Stand über PR #53 prüfbar.

## Genau ein Folgeauftrag: A3-Design dokumentieren – vorbereitet, wartet auf Nutzer-OK

**Ziel:** den ersten, ausschließlich dokumentarischen Teil von A3 als vorab
gesperrtes Design für einen späteren synthetischen Methodenprototyp ausarbeiten.
Es soll festlegen, wie bekannte künstliche Wahrheit die methodischen Zusagen
des Modellvertrags prüfbar macht, einschließlich erwarteter Grenzen und
Fehlerfälle. Kein empirischer Einsatznachweis, keine reale Rekonstruktion und
keine Ausführung des Prototyps. Der jetzige Nutzerauftrag gilt nur für die erledigte
Auftragsabgrenzung; das Designpaket ist noch nicht beauftragt.

**Benötigte Eingänge und Leseplan:** zuerst Git-/PR-/Merge-/CI-Stand, Branches,
Worktrees und lokale Änderungen prüfen, ungesicherte Arbeit erhalten; eigener
Worktree/Branch/PR ohne Merge. Danach AGENTS, diese HANDOFF, ausschließlich
neuester Status, MODEL_CONTRACT sowie Plan §1, §2-Zeile A3 und §4.
NOVELTY_MATRIX nur bei einer konkreten Belegfrage nachladen; keine neue
Literatur-/Archivsuche oder erneute Originalprüfung der Dissertation.
Benötigt werden die dokumentierten Rollen und OFFEN-Felder aus
[Rollen und Eingängen](./MODEL_CONTRACT.md#rollen-und-unbesetzte-eingänge),
die Regeln zu [Beobachtungsstatus](./MODEL_CONTRACT.md#beobachtungsstatus-nullen-und-abdeckung),
[Cut und Normierung](./MODEL_CONTRACT.md#kalibrier-cut-und-gemeinsame-bedingte-normierung),
[Zeit und Chronologie](./MODEL_CONTRACT.md#zeitprozess-und-chronologie-rückkopplung),
[Proxy-Abhängigkeiten](./MODEL_CONTRACT.md#proxy-abhängigkeiten-und-gemeinsame-evidenz),
[Support/Identifizierbarkeit](./MODEL_CONTRACT.md#support-und-identifizierbarkeit)
und [Stopbedingungen](./MODEL_CONTRACT.md#stopbedingungen-und-abschlussgrenze).
Keine realen Daten oder neuen Originalbelege als Eingang beschaffen.

**Erlaubte Änderungen im späteren Paket:** genau ein neues aktives
A3-Designdokument plus kurze HANDOFF-/Statuspflege. Darin ausschließlich
synthetische Annahmen, Evidenzrollen, bekannte Wahrheit, Szenariozuordnung,
erwartete Befunde, Abnahmeregeln und getrennte Nachrechnung vorab beschreiben.
Etwaige künstliche Größen, Verteilungen und Parameter müssen ausdrücklich
lokale Designannahmen bleiben; sie füllen keine realen OFFEN-Felder des
MODEL_CONTRACT und ändern keinen bestehenden Vertrag. Noch keine Generatoren,
Datenartefakte, Implementierung, Tests oder Rechnungen erstellen/starten.

**Verbindliche Abdeckung aus Planzeile/Vertrag:** Das spätere Design muss
Normierung und Cut-Marginalen, gemeinsame Kalibrierunsicherheit, fehlende
Beobachtungen/Nullstatus, zeitliche und taxonübergreifende Abhängigkeiten,
Altersverschiebung, Proxykonflikt und nicht identifizierbare Fälle adressieren.
Prior-only und fehlerhafte Eingänge ausdrücklich berücksichtigen. Für jeden
Punkt muss vorab erkennbar sein, welche künstliche Wahrheit und welche
Prüfgröße eine Erfüllung, Verletzung oder Nichtschätzbarkeit zeigen sollen.
Die getrennte Nachrechnung und vollständige Berichterstattung aller späteren
Resultate einschließlich Ausfällen/Ausschlüssen sind zu planen, nicht
auszuführen. Keine konkreten Szenario- oder Verfahrensentscheidungen werden
durch diese Auftragsabgrenzung vorweggenommen.

**Schutzgrenzen:** Bestehende Designs, Modelle, Resultate, Register und
Verträge einschließlich MODEL_CONTRACT, Importer, Flags, Counts/Gruppen,
Baseline und CLI/App unverändert. Keine reale Daten-/Modellwahl,
Beschaffung, Bereinigung, Datenmatrix/Kohorte, Fits, Scores, Simulationen oder
Kernläufe. A4–A6 bleiben außerhalb des Auftrags. Keine Unteragenten,
Autorennachrichten oder Merges; keine stillen Literaturkorrekturen oder
Neuheitsbehauptungen.

**Offene Voraussetzungen und Stopbedingungen:** Parnell-Originalzugang bleibt
blockiert, Menezes-(2)/(5)-Formelklärpunkt offen; kein eigener Neuheitsbeitrag
ist abgegrenzt. W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben offen. Sie sind Grenzen realer Nutzung und
der Beleglage, keine pauschalen Vorbedingungen für ein rein synthetisches
Design und werden durch dessen spätere Fertigstellung nicht gelöst.

- Benötigt ein Designpunkt neue Originalbelege, reale Auswahl oder die Lösung
  eines dieser Blocker, den betroffenen Punkt als blockiert ausweisen; keine
  Ersatzannahme als belegte reale Eingabe ausgeben.
- Widerspricht eine synthetische Spezifikation den Vertragsregeln zu Cut,
  gemeinsamer Normierung/Unsicherheit, Evidenzrollen oder Beobachtungsstatus,
  nicht als vertragskonform sperren. Absichtlich fehlerhafte Eingänge nur als
  vorab gekennzeichnete Fehlerfälle mit erwarteter Zurückweisung behandeln.
- Sind künstliche Wahrheit, erwartete Konsequenz, Prüfkriterium oder getrennte
  Nachrechnung nicht eindeutig beschreibbar, Designpunkt offen lassen und
  keinen vollständigen Designabschluss behaupten. Nichtidentifizierbarkeit
  darf ein erwartetes Ergebnis sein, keine durch enge Priors verdeckte Lösung.
- Sobald Code, Datenerzeugung, Ausführung, Rechnung, reale Freigabe oder eine
  Änderung bestehender geschützter Dateien nötig würde, an der
  Dokumentationsgrenze stoppen. Keine automatische Fortsetzung nach A4–A6.

**Prüfbares Ende:** genau ein vor Ausführung gesichertes A3-Designdokument;
jeder geforderte Prüfpunkt mit Vertrags-/Planbezug, künstlicher Wahrheit,
erwartetem Befund, Abnahmeregel und geplanter getrennter Nachrechnung
nachverfolgbar. Offene/blockierte Punkte und Grenzen ausdrücklich ausweisen;
keine Abdeckung, Identifikation oder reale Gültigkeit als bereits geprüft
behaupten. Quellen-/Link-/Diffprüfung und Blobabgleich bestehen, nur neues
Designdokument plus kurze Übergabe/Status ändern, Commit und eigenen PR ohne
Merge sichern. Umsetzung/Ausführung bleiben separat zu beauftragen; der
Designabschluss erteilt dafür keine Freigabe.
