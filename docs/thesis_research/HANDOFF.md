# Netzel: Furlanetto TRS 18 – Auftrag autorisiert, Originalzugang blockiert

## Aktueller Stand, 01.11.2026

Nutzer-OK „OK, Furlanetto fortsetzen“ liegt vor und bleibt für denselben
Auftrag gültig. Basis [PR #69](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/69),
Head `47bbcc2c2770c28040086e72286e45e37e83c83f`, offen/ungemergt;
vier automatische Head-Checks beim Einstieg noch laufend. Eigener Worktree
`/workspace/netzel-furlanetto-public-original`, gleichnamiger Branch mit
Präfix `codex/`, gegen #69. Sechs vorgefundene Worktrees sauber erhalten.

Websuche/-Lesewerkzeug jetzt verfügbar; zwei begrenzte Suchrunden ausgeführt.
Suchindex führt die bekannte Furlanetto-Dissertation als öffentlich; ein
direktes Öffnen des [BOA-Datensatzes](https://www.boa.unimib.it/handle/10281/241319)
ergibt HTTP 403. Kein verifizierter PDF-Link, keine Originalbytes und keine
neue Methodenvolltextlesung. Kein erneuter Sage-Abruf; letzter belegter Befund
bleibt HTTP 403 / `cf-mitigated: challenge`. Suchauszüge sind keine
Originalmethodenbelege. An dieser konkreten Zugriffslücke gestoppt.

[Review-Anhang](./A4_E_PROVENANCE_EVIDENCE_REVIEW.md#öffentlicher-furlanetto-originalzugang-nach-nutzer-ok-01112026)
und [neue Evidenz](./furlanetto_public_original_2026_11_01/evidence.json)
sichern Suche, Zugriffsgrenze und die getrennten Nachweise. Alte Reviewtexte,
Evidenz und Originale erhalten. R04/R05 nicht geschlossen, keine Gatefreigabe.
Im bestehenden Chat weiterarbeiten; bereits bekannte unveränderte Stellen
weiterverwenden, keine weiteren Abrufe/Diagnosepakete ohne neuen Eingang.

## Was der Nutzer tun kann/muss

**Kein weiteres OK nötig.** Keine Nutzerdatei und keine Kontakte erforderlich.
Optional über die unterstützte Cloud-Umgebungskonfiguration die offiziellen
Hosts `www.boa.unimib.it` und `boa.unimib.it` erlauben; beide fehlen in der
aktuellen Executor-Hostliste. Danach kann Codex den bekannten institutionellen
Datensatz erstmals regulär über den Executor prüfen und dessen tatsächlichen
öffentlichen Dateilink ermitteln. Die Freigabe garantiert keine Lesbarkeit,
sie behebt die Sage-Challenge nicht. Hier keine Netzänderung ausgeführt.
Alternativ kann ein bereits bekannter öffentlicher Original-Link optional
genannt werden; selbst ermittelbare Quellen sucht/verifiziert Codex.
Aktuell ist kein Nutzerschritt als sichere Behebung der Sage-Sperre belegt.

## Genau ein Anschluss – bereits autorisiert, wartet nur auf technischen Eingang

**Ziel:** Furlanetto-Originalmethodik/Supplement für Champvillair_a18 / TRS 18,
DOI 10.1177/0959683618810395, lesen. Kein neues Planungs-/Diagnosepaket.

**Startbedingung:** belegte relevante Änderung des erlaubten Originalzugangs
oder neu verifizierter öffentlicher Autorenoriginal-Link. Für die fachliche
Lesung müssen tatsächlich lesbare Originalbytes vorliegen. Nach offizieller
BOA-Hostfreigabe ist eine erste reguläre Prüfung des bekannten Handles möglich;
keine erfundenen Dateipfade, keine Umgehung, kein identischer Sage-Abruf.

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
Branch: codex/netzel-furlanetto-public-original
Sicherung: siehe nachstehenden aktuellen Commit-/PR-Verweis.
Fachliche Basis: PR #68, Inhaltscommit 0126234303f74c9e4212a354e61d0dc81eaa5ec6.
Nutzer-OK liegt vor; für denselben Auftrag nicht erneut fragen.
Nur Champvillair_a18 / TRS 18, DOI 10.1177/0959683618810395.
Vor Fortsetzung belegte relevante Zugriffsänderung oder verifizierten öffentlichen
Originalzugang verlangen. BOA-Webzugriff HTTP 403; Executor-Hosts bislang nicht
erlaubt. Nach offizieller Hostfreigabe zuerst den bekannten institutionellen
Handle prüfen, keinen PDF-Pfad erfinden. Sage nicht identisch erneut abrufen;
letzter Befund HTTP 403 / cf-mitigated: challenge. Keine Umgehung.
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
tatsächlicher Fortsetzung prüfen. Kein Merge. Das Nutzer-OK bleibt gültig;
nur der technische Eingang fehlt.
