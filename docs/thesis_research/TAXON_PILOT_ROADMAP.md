# Gemeinsamer Meilenstein: erste lokale Rekonstruktion mit dem Taxonverfahren

Am 31.10.2026 vom Nutzer im Chat als gemeinsamer Weg festgehalten.
Diese Roadmap verbindet Gesamtziel, vorhandene Bausteine und den Weg zum ersten
Kernlauf. Sie genehmigt keine konkrete Modellvariante, Taxonzuordnung, Pilotwahl,
Absenkung bestehender Kriterien oder einen Merge. Dieses Paket ist Dokumentation;
keine Forschungsläufe oder Implementierung werden dadurch als erledigt behandelt.

## Gesamtauftrag und erster greifbarer Abschluss

Ein wissenschaftlich nachvollziehbares, verbessertes **taxonbasiertes** Verfahren
nach den Ideen der Dissertation entwickeln, prüfen und später mit mehreren Proxys
in der App nutzbar machen. Die Dissertation wird nicht ungeprüft oder bitgleich
nachgebaut: mathematische Rollen der Terme, Posterior-Auswertung, Datenrollen und
Unsicherheiten müssen klar sein. Bestehendes Biom-Verfahren und Analog-Baseline
bleiben als unveränderte Referenzen erhalten. Claude betreut CLI/App; Codex
Kalibrierung, Quellen, Methodenprüfung und Ergebnisprüfung.

**Erster Meilenstein:** ein reproduzierbarer, ausdrücklich als diagnostisch
gekennzeichneter lokaler Klimarekonstruktionslauf für einen begründet gewählten
Bohrkern mit festgelegtem neuen Taxonverfahren, nachvollziehbarer Taxonabdeckung,
Chronologie, Supportdiagnostik und offengelegten Unsicherheitsgrenzen.
Das ist noch keine fertige allgemeine Multiproxy-App oder pauschale Fossilfreigabe.

## Was bereits vorhanden ist

| Baustein | Belegter Stand / Grenze |
| --- | --- |
| Referenzverfahren | Biom-Verfahren und geprüfte moderne Analog-Baseline vorhanden, unverändert erhalten |
| Transferfunktionen | Sieben Arten: quadratische Modelle und Netze mit/ohne SMOTE berechnet und räumlich geprüft; Wahrscheinlichkeitskalibrierung untersucht. Keine universelle Überlegenheit von SMOTE belegt. [Bericht #28](./taxon_transfer_2026_10_30/README.md), [Kalibrierung #31](./calibration_2026_10_30/README.md) |
| Unsicherheit | Bedingte Kalibrierungssensitivität bei festen Netzen untersucht; keine vollständige oder nominell validierte Rekonstruktionsunsicherheit. [Bericht #32](./calibration_uncertainty_2026_10_30/README.md) |
| Pollendaten | Bereits importiert und geprüft; zuletzt Originaldefinitionen/Namensketten gesichert. Artbrücke und Bestimmungsumfang bleiben offen. [Originalprüfung #35](./original_pollen_names_2026_10_31/README.md) |
| Neues Gesamtverfahren | Noch kein abgenommener taxonbasierter Bohrkern-Gesamtdurchlauf; sieben Vorwärtsfunktionen sind noch keine geprüfte Inversion Pollenspektrum→Klima |

Die Funktionen modellieren zunächst kartierte Artenvorkommen unter Klima,
nicht direkt fossile Pollenhäufigkeiten. Die verlorenen 279 historischen
Funktionen sind nicht wiederhergestellt. Nicht Voraussetzung des ersten Piloten:
alle 279 Funktionen, alle europäischen Standorte oder die vollständige
Multiproxy-/App-Integration fertigstellen. Ob die sieben vorhandenen Arten für
einen geeigneten Pilot genügend Information liefern, ist dagegen **noch offen**.

## Weg zum ersten diagnostischen Kernlauf

| Arbeitsschritt | Konkretes Ergebnis / Kriterium vor Weitergang |
| --- | --- |
| **T0 – Bestimmungsumfang und Kern eingrenzen** | Datensatzbezogene Bestimmung und Pollentyp–Taxon-Bezüge belegen; unsichere/nicht zuordenbare Fälle erhalten. Kandidatenkern auf verfügbare Counts, Chronologie und verwendbare Taxa prüfen. Auswahl, Mindestabdeckung und Ausschlussregeln vor dem Rekonstruktionslauf festlegen; keine Auswahl anhand gewünschter Klimakurven oder äußerer Modell-Scores. |
| **T1 – Rekonstruktionsregel vorab festlegen** | Vorwärtsfunktionen, Kombination/Inversion, Beobachtungsannahmen, Klimaprior, optionale Referenzterme und Umgang mit Unsicherheit mathematisch begründen. Einheitliche Klimagrößen, Einheiten und Bezugszeiträume sichern; TANN nicht mit TDJF verwechseln. Konkrete Designentscheidung und Prüfplan sichern, bevor wissenschaftliche Anpassung/Variantenbewertung beginnt. |
| **T2 – Begrenzten Methodenprototyp prüfen** | Forschungsprototyp mit synthetischen Daten: bekannte Eingänge und Grenzfälle, Prior-only-Verhalten, numerische Korrektheit, Supportgrenzen und identifizierbare/nicht identifizierbare Fälle prüfen. SBC nur bei geeignetem kohärentem generativem Modell als solche ausweisen; eine Pseudo-Likelihood liefert nicht automatisch einen SBC-Nachweis. Bestehende Referenzverfahren unverändert. |
| **T3 – Reale Eingänge sperren** | Gewählter Kern mit belegter Probe/Tiefe/Alter-Verknüpfung; zulässige Taxonzuordnungen und ausreichende vorab definierte Abdeckung nachweisen. Unbekannte/nicht bestimmbare Taxa nicht als biologische Absenz behandeln. Falls Referenzproxies genutzt werden: Chronologie, Abhängigkeiten und Herkunft ebenfalls dokumentieren. Fehlende Unsicherheitskomponenten ausdrücklich benennen. |
| **T4 – Ersten lokalen Pilot rechnen und auditieren** | Nur nach dokumentierter Prüfung von T0–T3: diagnostischer Kernlauf mit gesperrten Eingängen/Annahmen, reproduzierbaren Ausgaben, Konvergenz-/Supportprüfung und separatem Ergebnis-Audit. Fehler, Verschlechterungen, Nichtschätzbarkeit und Grenzen vollständig berichten; keine modernen Testfehler als fossile Fehlerbänder. |
| **Danach – Integration und Erweiterung** | Geprüften Datenvertrag und klar begrenzten Auftrag an Claude für CLI/App vorbereiten; weitere Taxa, Standorte und Proxys als eigene Pakete. Kein automatischer Übergang vom diagnostischen Pilot zur allgemeinen wissenschaftlichen Freigabe. |

**Parallel sinnvoll:** Quellen-/Zuordnungsarbeit T0 und Design/synthetischer
Methodenprototyp T1–T2 können in getrennten, begrenzten Paketen vorankommen.
Synthetische Prüfungen brauchen noch keine abschließend geklärte Fossilzuordnung.
Sie ersetzen aber T3 nicht. Dies ist eine fachliche Arbeitsaufteilung, kein
Auftrag für automatische Unteragenten oder gleichzeitige Änderungen derselben Dateien.

## Monticchio-Pilot und neue Methode auseinanderhalten

Der bestätigte [PILOT_PLAN.md](./PILOT_PLAN.md) betrifft in P2 zunächst den
Monticchio-Lauf der **bestehenden Analog-Baseline**. Dieser Lauf ist nicht mit
T4 der neuen Methode identisch. Sein P1-Gate bleibt verbindlich: physische
LGM90D→MON16-Verknüpfung einschließlich Korrelationsunsicherheit fehlt weiterhin;
28 Kandidaten, 41 native Nichttreffer und 47 Herkunftsfragen sind keine Freigabe.
Für das neue Verfahren an Monticchio wird dieser Quellenblock nicht umgangen.

Der ältere [Multiproxy-Plan](../plan_multiproxy.md) nennt Kinneret als möglichen
Methodeneinstieg; der spätere Pilotplan konzentriert sich auf Monticchio.
Keiner dieser historischen Vorschläge wählt automatisch den Kern für T4.
Bleibt Monticchio blockiert, bedarf ein anderer Pilot bzw. eine geänderte
Zieldefinition einer ausdrücklichen Entscheidung; keine endlose Ersatzquellensuche
und keine still abgesenkten Kriterien. Kein belastbares Kalenderdatum für T4,
solange Kern, Artbrücke und konkrete Rekonstruktionsregel nicht feststehen.

## Nächster Einstieg und Schutzregeln

Der unmittelbar vorbereitete Quellenauftrag bleibt die Prüfung der Methoden von
Watts, Allen & Huntley (1996) und der tatsächlich verwendeten Bestimmungsreferenzen;
[genauer Umfang](./original_pollen_names_2026_10_31/SOURCE_GAP.md).
Sein Ergebnis muss T0 konkret voranbringen: belegte Art-/Typgrenzen oder präzise
verbleibende Lücken, keine bloße Verlängerung der Namensinventur.
Ein anschließendes eigenes Designpaket soll T1 und einen überschaubaren
synthetischen T2-Prototyp abgrenzen, damit die Kernrekonstruktion gemeinsames
Ziel bleibt. Noch keine neue Formel oder Variante als gewählt behandeln.

Dauerhafte Regeln: [AGENTS.md](../../AGENTS.md); belegter aktueller Stand:
[PROJECT_STATUS.md](./PROJECT_STATUS.md); nächster Auftrag: [HANDOFF.md](./HANDOFF.md).
Keine neuen Modellentscheidungen aus äußeren Scores, keine automatische Art-
oder Pilotfreigabe und kein Merge ohne ausdrücklichen Auftrag. Ältere Pläne und
Archive bleiben unverändert erhalten; neue Entscheidungen samt Begründung sichern.
