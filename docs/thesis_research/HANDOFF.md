# Netzel: kompakte Übergabe nach A2

## Gesichertes Ergebnis

[MODEL_CONTRACT](./MODEL_CONTRACT.md) ist der zusammenhängende dokumentarische
A2-Entwurf auf ausdrücklichen Nutzerauftrag. Festlegungen aus
[Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung)
sind von OFFEN-Feldern getrennt: Rollen E/P/Y/D/psi/C/tau/theta,
Beobachtungsstatus/Nullen, Kalibrier-Cut mit gemeinsamer bedingter Normierung,
Zeitprozess, begründungspflichtiger Chronologie-Rückkanal, Proxy-Abhängigkeiten,
Support, Identifizierbarkeit und Stopbedingungen. Keine reale Daten-/Modellwahl
oder ausgefüllte Datenmatrix. A2 dokumentarisch abgeschlossen; Modell- und
Datenfreigabe weiterhin offen.

A1 bleibt abgeschlossen. [NOVELTY_MATRIX](./NOVELTY_MATRIX.md) unverändert:
Parnell-Originalzugang blockiert, Menezes-(2)/(5)-Formelklärpunkt offen;
kein eigener Neuheitsbeitrag abgegrenzt. Keine neue Literatur-/Archivsuche
oder Originalprüfung der Dissertation. W96, EMPD2-Archiv-/Messstatus und
reale T0/T1/T3/T4/Monticchio-P1 offen.

Basis PR #51, offen/ungemergt und mergebar, Head
`665daede79048df7efb6fe60db85c765b45718bb`; automatische CI 41211699800
bei Standprüfung erfolgreich. Main unverändert `7f1826c`, #50-Head `64a6529`.
Fünf vorgefundene Worktrees sauber, keine Stashes; kein Reset oder Merge.
Eigener Worktree `/workspace/netzel-a2-model-contract`, Branch
`codex/netzel-a2-model-contract`; [PR #52](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/52)
gegen den #51-Branch, offen/ungemergt und mergebar. Inhaltscommit
`1aee3fd71fa50da256f372979dcc6927081fb38e`; danach nur Sicherungsverweise.
Quellen-/Link-/Diffprüfung bestanden, 727 geschützte Blobs bytegleich.
Automatische CI 41216380268 zum Inhaltscommit bei Abfrage queued;
kein abschließender CI-Erfolg behauptet. Aktueller Remote-Head über PR #52
prüfbar; Details im [neuesten Status](./PROJECT_STATUS.md).
Ältere Übergaben bleiben in Git erhalten.

## Genau ein Folgeauftrag: A3-Auftragsabgrenzung – vorbereitet, wartet auf Nutzer-OK

**Ziel:** ausschließlich den nächsten A3-Auftrag anhand seiner Planzeile und
der OFFEN-Felder des A2-Vertrags konkret abgrenzen, ohne A3 auszuführen.
Das OK hierfür autorisiert nur diese dokumentarische Auftragsklärung;
A3–A6 sind noch nicht begonnen. Für A2 ist kein weiteres OK erforderlich.

**Erlaubter Umfang und Ergebnis:** nur HANDOFF und kurzen Status pflegen;
Ziel, benötigte Eingänge, erlaubte Änderungen, Schutzgrenzen und prüfbares Ende
des späteren A3-Pakets festhalten. Fehlende Voraussetzungen explizit als
Blocker übernehmen. Kein neues Konzeptpapier, keine Wahl realer Daten,
Klimavariablen, Verteilungen, Parameter oder Modelle aus OFFEN-Feldern ableiten.

**Eingänge/Leseplan:** kompakte Git-/PR-/Merge-/CI-/Worktree-Prüfung, AGENTS,
diese HANDOFF, nur neuester Status, MODEL_CONTRACT; Plan ausschließlich
§1/§2-Zeile A3/§4. NOVELTY_MATRIX nur bei konkreter Beleggrenze nachladen.
Keine Originalquellen oder Archive erneut öffnen.

**Abschluss:** genau ein konkret ausführbarer A3-Auftrag mit expliziten offenen
Voraussetzungen und Stopbedingungen in HANDOFF; Dokumentations-/Link-/Diff-
und Blobprüfung, eigener Branch/PR ohne Merge. Noch keine Ausführung von A3
oder Vorgriff auf A4–A6.

**Gesperrt:** neue Recherche, Datenwahl/-beschaffung, Bereinigung,
Datenmatrix/Kohorte, Implementierung, Tests, Fits, Scores, Simulationen,
Kernläufe sowie Änderungen bestehender Designs/Modelle/Resultate/Register/
Verträge/Importer/Flags/Counts/Gruppen, Baseline und CLI/App.
Keine Unteragenten, Autorennachrichten oder Merges.

## Kopierfertiger Übergabeprompt

Die folgende Anweisung ist für den nächsten Chat vorbereitet; erst ihr
Absenden bzw. ein entsprechendes OK beauftragt die A3-Auftragsabgrenzung.
In diesem Paket wurde nur der Prompt gesichert.

> Setze das Netzel-Projekt nach PR #52 fort.
>
> Branch: codex/netzel-a2-model-contract
> Gesicherter A2-Stand: 7deff9f002bd8f865f63ef5812c843d6c0a6ec9b
> PR: https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/52
> Danach gegebenenfalls nur Übergabepflege; aktuellen Remote-Head prüfen.
>
> Prüfe zuerst Git-/PR-/Merge-/CI-Stand, Branches, Worktrees und lokale Änderungen; erhalte ungesicherte Arbeit. Eigener Worktree, Branch und PR, kein Merge.
>
> Tokensparend lesen: AGENTS.md, HANDOFF.md, MODEL_CONTRACT.md, ausschließlich den neuesten Abschnitt von PROJECT_STATUS.md und MODULAR_RECONSTRUCTION_PLAN.md nur §1, §2-Zeile A3 und §4. NOVELTY_MATRIX.md nur bei konkreter Belegfrage nachladen.
>
> A1 und A2 sind als Dokumentationspakete abgeschlossen. MODEL_CONTRACT trennt Planfestlegungen von OFFEN-Feldern; keine reale Modell-/Datenfreigabe. Parnell-Originalzugang und Menezes-Formelklärpunkt (2)/(5) bleiben Grenzen; kein eigener Neuheitsbeitrag ist abgegrenzt. W96, EMPD2-Archiv-/Messstatus und reale Gates bleiben offen.
>
> Führe als nächsten begrenzten Auftrag ausschließlich die dokumentarische A3-Auftragsabgrenzung aus: aus Planzeile und Modellvertrag Ziel, benötigte Eingänge, erlaubte Änderungen, Schutzgrenzen, Stopbedingungen und prüfbares Ende für das spätere A3-Paket festhalten. Nur HANDOFF und kurzen Status pflegen; kein neues Konzeptpapier. A3 selbst und A4–A6 nicht ausführen. Für diese Auftragsabgrenzung kein erneutes OK verlangen.
>
> Keine neue Literatur-/Archivsuche oder erneute Originalprüfung der Dissertation. Keine reale Daten-/Modellwahl, Datenbeschaffung, Bereinigung, Datenmatrix/Kohorte, Implementierung, Tests, Fits, Scores, Simulationen oder Kernläufe. Bestehende Designs, Modelle, Resultate, Register, Verträge einschließlich MODEL_CONTRACT, Importer, Flags, Counts/Gruppen, Baseline und CLI/App unverändert.
>
> Keine Unteragenten, Autorennachrichten oder Merges. Kurze deutsche Updates, kompakte Werkzeugausgaben. Dokumentation durch Quellen-/Link-/Diffprüfung und Blobabgleich prüfen. Ergebnis und genau einen begrenzten Folgeauftrag sichern.
