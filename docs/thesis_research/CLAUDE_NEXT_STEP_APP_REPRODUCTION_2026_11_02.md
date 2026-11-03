# Claude-Übergabe: Dissertationsnachbau in die App anbinden

Stand 02.11.2026. Gilt ebenso für ChatGPT/Codex. Vorrangiges Nutzerziel:
**die lokalen taxonbasierten Rekonstruktionen der Dissertation nachbauen und
in der App nutzbar machen; methodische Verbesserungen erst danach.**
Deutsch, kurze Fortschrittsmeldungen, keine Subagenten. Nur vorhandenen vom
Nutzer benannten Checkout verwenden; keine neuen Klone/Worktrees. Bestehender
Branch `dissertation-reconstruction`, [PR #77](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/77),
kein automatischer Merge. Git-/PR-Stand und lokale Änderungen zuerst prüfen.
Basis des aktuellen Pakets `956951b`; den Sicherungscommit liefert
`git log -1 -- tools/historical_tf_preparation.py`.

## Was jetzt tatsächlich fertig ist

1. `CLAUDE.md`, `AGENTS.md`, neuesten [Projektstatus](PROJECT_STATUS.md),
   [Arbeitsplan](LOCAL_DISS_RECONSTRUCTION_PLAN.md) und
   [TF-Vorbereitungspaket](literal_tf_preparation_2026_11_02/README.md) lesen.
2. Fünf begrenzte historische Teilmodule sind vorhanden: Alters-/Tiefenübertragung,
   Alters-/Zeilenauswahl, Taxongewichtung/PDF-Summe, Roh-CDF-Vergleich und jetzt
   TF-Zeilennormierung mit nachfolgender expliziter Klimazellmaske.
3. Das neue Modul ist `tools/historical_tf_preparation.py`. 51 neue Tests und alle
   bisherigen 200 bestanden: **251 grün**. Eine zusätzliche Kunstverbindung mit
   allen fünf Modulen stimmt mit unabhängiger Bruchrechnung bis auf höchstens
   `2.22e-16` überein. TF-Normierung findet vor der Maske statt; keine zweite
   Normierung nach dem Nullsetzen. Original-Ziehungs-IDs 1 und 3 bleiben erhalten.
4. Quellen-/Versionsbefunde zu climate_modelling sind bereits abgeschlossen:
   [neues Quellenpaket](climate_modelling_audit_2026_11_02/README.md).
   Verbreitungsbasis vorhanden, F14/F15-Version ungeklärt, v4.0 ungebunden.
   Diesen Bestand nicht nochmals pauschal inventarisieren.

**Weiter offen:** Elf Nachweisbündel. Finale Basis/Listenchronologie, numerische
historische TFs, globale Klima-/Anker-/Referenzstände, historische Laufzuordnung
und Originalergebnisabgleich fehlen weiter. Keine finale Variante gewählt und
kein vollständiger taxonbasierter Dissertationsfall reproduziert. Die 251 Tests
verwenden Kunstbeispiele und Belegverträge; sie sind keine Originalfallrechnung.

## Konkreter nächster Auftrag: erste App-Anbindung der geprüften Rechenschritte

Der Nutzer möchte sichtbaren Fortschritt zum Dissertationsnachbau in der App.
Mache die vorhandenen fünf Teile und ihre neue geprüfte Rechenkette in der
bestehenden Ansicht **Forschungsstand** zugänglich. Es handelt sich um den
Nachbaustand und ein ausdrücklich synthetisches Rechenbeispiel. Ein echter
Dissertationslauf bleibt an die offenen Originaleingaben gebunden.

Die aktuelle App liest bisher einen älteren Forschungs-Snapshot. Diesen
Snapshot, seine Hashpins und die eingefrorenen Startaufträge nicht überschreiben.
Stattdessen einen separaten datierten, an die neuen Paket-Hashes gebundenen
Nachbaustand ergänzen. Bereits gespeicherte Beispielergebnisse verwenden;
beim App-Build keine Originalcontainer, Modelle, R-Skripte oder Sampler öffnen.
Keine Hintergrundberechnung oder Abhängigkeit von privaten `.local`-Pfaden.

Vorhandene Integrationspunkte, in diesem Paket nur gelesen:

| Datei | Aufgabe im nächsten Paket |
| --- | --- |
| `src/kinneret_recon/forschungsstand.py` | Ergänzende geprüfte Nachbau-/Beispieldaten lesen oder an einen kleinen neuen Leser delegieren; bisherige Snapshotbindung erhalten. |
| `src/kinneret_recon/app/forschung.js` | Abschnitt „Dissertationsnachbau“ mit Teilmodulen, belegtem Umfang und noch offenen Voraussetzungen. |
| `src/kinneret_recon/app/i18n.js` | Deutsche/englische Texte, eindeutige Kennzeichnung der Kunstrechnung. |
| `src/kinneret_recon/app_bauen.py` | Datenpfad in Ordner- und Ein-Datei-App erhalten; bei Schemaergänzung beide Exportwege prüfen. |
| `tests/test_forschungsstand.py`, `tests/test_app.py`, `tools/app_pruefen.js` | Datenintegrität, Build und Offline-Browser prüfen; Baseline-/Artenansichten erhalten. |

Mindestinhalt und Abnahme:

- Sichtbarer Status „Teilmodule geprüft; vollständiger Originalfall offen“.
  Keine Erfolgsanzeige „Dissertation reproduziert“ aus Testzahlen ableiten.
- Fünf Schritte mit Quellenlinks; elf offene Voraussetzungen aus dem unveränderten
  Belegvertrag zuordnen. Originalbasis/Chronologie, numerische TFs und Laufbindung
  als zentrale Lücken verständlich benennen, ohne interne Variablennamen als
  Bedienkonzept aufzuzwingen.
- Ein aufklappbares **synthetisches Rechenbeispiel**: TF-Werte vor Normierung,
  nach Normierung und nach expliziter Maske; die Zeilensummen 1 und 2/3 zeigen.
  Die bereitgestellte zusätzliche Kunstkette kann Rohdichte/PDF ergänzen.
  Keine Klimakalenderjahre, realen Messdaten oder historischen TFs dafür behaupten.
  Beispielzahlen nicht in den vorhandenen Laufkatalog oder echte Rekonstruktionen
  einspeisen; keine stillen Ersatzdaten.
- Werte aus `literal_tf_preparation_2026_11_02/synthetic_result.json` und
  `arithmetic_check.json` übernehmen und an deren unveränderte SHA-256 binden.
  Fehlender Zusatzstand bleibt sichtbar nicht verfügbar; geänderte Bytes dürfen
  nicht als geprüfter Stand angezeigt werden. Bestehender Forschungs-Snapshot
  und Nachbaustand müssen getrennt erkennbar bleiben.
- Beide Offlineformate, Deutsch/Englisch, Hell/Dunkel, Desktop/Mobil prüfen.
  Keine neuen Netzanforderungen; Quellenlinks dürfen wie bisher extern öffnen.
  Forschungs-/Nachbaustand darf nicht von der Standort-/Laufwahl abhängen.
- App-/Daten-/Buildtests und passende Browserprüfungen tatsächlich ausführen;
  Prüfumgebung/Überspringungen offen dokumentieren. Die historischen 251 Tests
  sind zuletzt in diesem Paket grün; bei Numerikänderungen erneut prüfen.
  Beide Phase-1-Prüfer weiter erwarteter Exit 2.

Das ist die erste App-Anbindung des geprüften Nachbaus. Es ersetzt den nächsten
fachlichen Meilenstein nicht: einen vollständigen Eingabevertrag für einen lokalen
Originalfall schließen und erst bei belegten Eingaben den historischen Ablauf
reproduzieren. Keine weiteren modernen Modelle oder Komfortfunktionen vorschieben.

## Arbeitsgrenzen und erneute Übergabe

Originale nur lesen, private Quellpfade aus `.local/my_reconst_paths.json`.
Automount-Datenträger gelöscht; lokale Kopien zuerst prüfen, nur notwendige
verbleibende Lücken kontrolliert neu erstellen. Namensgleichheit beweist keine
historische Identität. Kinnerets Mittelwertchronologie bleibt von einer Liste
mehrerer Altersraster getrennt. Bekannte Speicherortfrage nicht wiederholen.
Keine unbelegte `inside.range`-/Median-/RNG-/Variantenwahl. Keine Trainings,
Sampler, Altersmodelle oder Fossilrekonstruktionen starten. Baseline, CLI,
Originale und fallbezogene Monticchio-/Chronologie-Gates erhalten. Plots jung
links, alt rechts; Phase 1 vor Phase 2.

Am Ende ein neues Belegpaket, aktualisierten Projektstatus/Arbeitsplan und
**eine neue konkrete Claude-Übergabedatei mit Link im Abschluss** erstellen:
Ergebnis, tatsächliche Tests, offene Originaleingaben, nächster fachlicher
Meilenstein und nächster App-Schritt. Diese und frühere Übergaben eingefroren
lassen. Fertiges geprüftes Paket auf bestehendem Branch committen/pushen,
PR #77 fortschreiben, nicht automatisch mergen.
