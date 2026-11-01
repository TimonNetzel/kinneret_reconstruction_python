# Netzel: Furlanetto TRS 18 – Auftrag autorisiert, Originalzugang blockiert

## Aktueller Stand, 01.11.2026

Nutzer-OK „OK, Furlanetto fortsetzen“ bleibt für denselben Auftrag gültig.
Basis [PR #70](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/70),
Head `183ac3a8473fcc3f4ba4d0cbe691a165f880fdf2`, offen/ungemergt;
alle vier Head-Checks erfolgreich abgeschlossen. Eigener Worktree
`/workspace/netzel-furlanetto-boa-review`, Branch
`codex/netzel-furlanetto-boa-review`, ab #70. Fünf vorgefundene Worktrees
sauber erhalten; kein Merge.

Beide BOA-Hosts jetzt wirksam erlaubt (`restricted/enforced`, aktuelle
Beobachtungen; Executor-Policy bestätigt). Trotz erneuter Revision 1 ist
die Hostliste verändert. Erster regulärer Executor-Aufruf des bekannten
[BOA-Handles](https://www.boa.unimib.it/handle/10281/241319): Proxy-CONNECT
200, danach HTTP/2 403 / `cf-mitigated: challenge`, HTML statt Datensatz,
keine Weiterleitung. Kein verifizierter öffentlicher Dateilink, keine
Originalbytes oder Methodenlesung. An dieser konkreten Zugriffslücke gestoppt.
Sage nicht erneut abgerufen; letzter belegter Befund bleibt ebenfalls eine
Challenge. Keine Umgehung und kein erfundener PDF-Pfad.

[Aktueller Review-Anhang](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md#boa-executorzugriff-nach-bestätigter-hostfreigabe-01112026)
und [neue Evidenz](./furlanetto_boa_review_2026_11_01/evidence.json)
sichern die neue Beobachtung. Alte Reviewtexte, Evidenz und Originale erhalten.
R04/R05 bleiben im bisherigen Teilumfang offen; keine Gatefreigabe.

## Was der Nutzer tun kann/muss

**Kein weiteres OK und keine erneute BOA-Hostfreigabe nötig.** Keine
Nutzerdatei und keine Kontakte erforderlich. Aktuell ist kein Nutzerschritt
als sichere Behebung der BOA- oder Sage-Challenge verifiziert. Optional kann
ein bereits bekannter offizieller öffentlicher Original-Link genannt werden;
Codex prüft den Zugang selbst. Für eine Fortsetzung ist ein belegter relevant
veränderter erlaubter Originalzugang oder neu verifizierter öffentlicher
Original-Link nötig. Die Hostfreigabe allein hat die Lesbarkeit nicht hergestellt.
Ohne neuen Eingang keine identischen Abrufe oder weiteren Diagnosepakete.

## Genau ein Anschluss – bereits autorisiert, wartet nur auf technischen Eingang

**Ziel:** Furlanetto-Originalmethodik/Supplement für Champvillair_a18 / TRS 18,
DOI 10.1177/0959683618810395, lesen. Kein neues Planungs-/Diagnosepaket.

**Startbedingung:** belegte relevante Änderung des erlaubten Originalzugangs
oder neu verifizierter öffentlicher Autorenoriginal-Link. Für die fachliche
Lesung müssen tatsächlich lesbare Originalbytes vorliegen. BOA-Hostfreigabe
ist bestätigt und der erste reguläre Abruf an einer Website-Challenge beendet.
Kein identischer BOA-/Sage-Abruf ohne belegte relevante Änderung, keine
erfundenen Dateipfade und keine Umgehung.

**Leseplan:** AGENTS, dieser HANDOFF und neuester Status; aktueller Review-Anhang
und neue Evidenz, bei Bedarf W3–W6/Champvillair und alte Furlanetto-Evidenz;
A4_E_REOPENING_REQUIREMENTS nur R04/R05 samt Herkunftsregeln. Bei tatsächlichem
Arbeitsbeginn aktuellen Head/CI/Worktrees prüfen. Ungesicherte Arbeit erhalten.

**Umfang/Abnahme/Stop:** nur Review/Evidenz und HANDOFF/Status pflegen.
Explizite TRS-18-/nachgewiesene Gruppenbindung, Sammlung, Ablagerung, Integration
und Unsicherheit getrennt mit Originalstelle, kurzem Zitat, Seite/Abschnitt,
Dokumentversion/DOI/URL und Beleggrenze sichern. Keine Versionsgleichheit von
Dissertation und Artikel voraussetzen. Unbekanntes erhalten, kein Probenalter
aus Publikationsjahr. Nach zugeordneten Originalstellen oder konkreter
Zugriffslücke stoppen. Keine neue Probe/Kohorte, Werteanalyse, Reparatur,
Implementierung, manuellen Tests, wissenschaftlichen Rechnungen, R01–R03-Neusuche,
R06-/Gatefreigabe, A4-Umsetzung/A5/A6, Unteragenten, Nachrichten/Kontakte oder
Merges. Nötige begrenzte schreibgeschützte Quellen-/Metadatenprüfung ohne
Extra-OK erlaubt. Alte Originale/Berichte erhalten. JSON/Links/Quellen/Hashes
und Diff prüfen; eigener Worktree/Branch/PR, kein Merge. Genau einen Anschluss
mit Prompt in HANDOFF pflegen; im bestehenden Chat nur verlinken, solange
der fachliche Auftrag unverändert ist. Nutzeraktionen konkret nennen.

## Kopierfertiger Folgeprompt

```text
Setze den bereits autorisierten Furlanetto-Auftrag im bestehenden Chat fort.
Branch: codex/netzel-furlanetto-boa-review
Sicherung: siehe nachstehenden aktuellen Commit-/PR-Verweis.
Fachliche Basis: PR #68, Inhaltscommit 0126234303f74c9e4212a354e61d0dc81eaa5ec6.
Nutzer-OK liegt vor; für denselben Auftrag nicht erneut fragen.
Nur Champvillair_a18 / TRS 18, DOI 10.1177/0959683618810395.
Vor Fortsetzung belegte relevante Zugriffsänderung oder verifizierten öffentlichen
Originalzugang verlangen. Beide BOA-Hosts wirksam erlaubt; erster regulärer
Executor-Handleabruf HTTP 403 / cf-mitigated: challenge, kein Dateilink verifiziert.
BOA und Sage ohne belegte Änderung nicht identisch erneut abrufen. Nur tatsächlich
verlinkte öffentliche Originaldateien verfolgen; keinen PDF-Pfad erfinden.
Fachliche Lesung erst bei lesbaren Originalbytes; Dissertation und Artikel
nicht ungeprüft gleichsetzen. Keine Umgehung. Aktuellen Head/CI prüfen.
Leseplan, getrennte Belege, Umfang und Stop aus HANDOFF einhalten.
Nur Review/Evidenz und HANDOFF/Status; alte Originale und ungesicherte Arbeit
erhalten. Eigener Worktree/Branch/PR, kein Merge. Alle Schutzgrenzen aus HANDOFF
einhalten. Ergebnis prüfen und sichern; genau einen Anschluss mit Prompt in
HANDOFF, im selben Chat bei unverändertem Auftrag nur verlinken.
Kurz auf Deutsch, nötige/optionale Nutzeraktionen konkret nennen.
```

## Sicherung

Basis der aktuellen Fortsetzung ist Regel-/Übergabe-PR #69, Head
`47bbcc2c2770c28040086e72286e45e37e83c83f`; die Angabe „danach nur
Sicherungsverweise“ gilt dort nur bis `9de48ac`, der spätere Regelzusatz ist
Commit `47bbcc2`. Fachliche Basis bleibt
[PR #68](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/68),
Inhaltscommit `0126234303f74c9e4212a354e61d0dc81eaa5ec6`.

Inhaltscommit `346558c1290d0a0ca0506397ce5331ad3230fdfe` remote in
[PR #70](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/70)
gesichert; offen/ungemergt gegen `codex/netzel-efficient-chat-continuation`
(#69). Danach nur Sicherungsverweise ergänzt; aktuellen Head/CI bei
tatsächlicher Fortsetzung prüfen. Dieser Absatz beschreibt die historische
Sicherung von #70. Das Nutzer-OK bleibt gültig; die neue BOA-Zugriffslücke
steht oben. Kein Merge.
