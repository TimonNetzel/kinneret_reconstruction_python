# A4: dokumentarische Auftragsgrenze vor moderner Kalibrierung

Stand 31.10.2026; Basis [PR #55](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/55),
`b1e78244bd8484004bc8f46de8ecbc09dadf5963`.
**Das Nutzer-OK autorisiert ausschließlich diese Auftragsabgrenzung.**
Sie ist dokumentarisch abgeschlossen; A4-Umsetzung und reale Kalibrierung sind
nicht begonnen. Es wird kein realer Datenvertrag ausgefüllt oder freigegeben.

## Maßgeblicher Auftrag und belastbarer Ausgangsstand

Gelesen wurden AGENTS, HANDOFF, nur der neueste PROJECT_STATUS-Abschnitt,
[A3_SYNTHETIC_AUDIT](./A3_SYNTHETIC_AUDIT.md), der unveränderte
[MODEL_CONTRACT](./MODEL_CONTRACT.md), [Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
ausschließlich **§2-Zeile A4** und [§4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
Keine neue Original-, Literatur- oder Archivprüfung; NOVELTY_MATRIX wurde
mangels konkreter Belegfrage nicht geöffnet. Aussagen über offene Eingänge
beziehen sich auf diese Dokumente, nicht auf eine neue Vollprüfung älterer
Quellenregister oder Datenbestände.

Die unveränderte A4-Zeile lautet:

> **A4 – Moderne Kalibrierung und Basisverfahren; geplant, Datenvertrag fehlt:**
> Begründeten modernen Eingang mit Typ-, Klima-, Zeit-/Orts- und Messrollen
> sperren. Vor Training Abhängigkeiten, räumliche Aufteilung/Support und
> unabhängige Prüfung festlegen. Menezes-nahe Basis mit passenden Klimazielen
> reproduzieren oder Abweichungen präzise ausweisen; moderne Pollenantwort
> und ihre Unsicherheit prüfen. Kein Vorgriff auf ungesicherte EMPD2-Counts
> oder bestehende äußere Scores.

A4 soll später ein überprüfbares modernes Pollen-Beobachtungsmodell und seine
Kalibrierunsicherheit q(psi)=p(psi|E) bereitstellen. Sein späterer fachlicher Abschluss
umfasst reale Nachweise und Prüfungen; dieses Dokument legt ausschließlich
deren benötigte Rollen und Grenzen offen. Die taxonbasierte Multiproxy-App
und ein realer diagnostischer Kernlauf bleiben nachgelagerte Ziele.

A1/A2 sind dokumentarisch abgeschlossen. A3 ist ausschließlich im gesperrten
endlichen synthetischen Umfang bestanden: laut unverändertem Audit
22 Posteriorfälle, drei prädiktive Verteilungen, 33 Fehlerablehnungen und
345 exakte Nachweise; sechs gezielte Tests. Diese Resultate wurden hier
nicht erneut berechnet. S11 bleibt erwartungsgemäß nicht identifizierbar.

## Anforderungen, vorhandene Belege und fehlende Entscheidungen

**OFFEN** bedeutet hier: im maßgeblichen Vertrag nicht besetzt bzw. durch den
synthetischen Audit nicht für reale Nutzung belegt. Es behauptet nicht,
dass in sämtlichen älteren Projektdateien keinerlei Information vorliegt.
Ein späterer Nachweis muss die konkrete Quelle/Version nennen und ihre
Tragfähigkeit prüfen; eine vorhandene Datei allein ist keine Freigabe.

| A4-Anforderung / Eingangsrolle | Bereits gesicherte Vorgabe oder begrenzter Beleg | Vor realem Einsatz fehlend; Status |
| --- | --- | --- |
| Moderner Eingang E und Herkunft | MODEL_CONTRACT: E enthält moderne Pollenbeobachtung samt Klimabezug; kein moderner Trainingssatz ausgewählt. | Quelle/Version, Messbeleg-IDs, Auswahl-/Ausschlussregel und Herkunftskette; **OFFEN**. Kein Datensatz gewählt. |
| Pollentypen und Abdeckung | MODEL_CONTRACT: Typen und Arten unterscheiden; sieben Artenfunktionen bleiben Vergleichsreferenzen. | Eingangsdefinitionen, gemeinsame Kategorie-/Abdeckungsregel, breite Typen/Mischungen und nachvollziehbare Zuordnungen; **OFFEN**. Keine Typ-/Artbrücke beschlossen. |
| Zählungen, N und Status | Vertrag verlangt belegte originale Counts/N und Feldstatus. A3 S06 prüft nur künstlich bekannte Status. | Für einen realen Eingang belegte Nenner, Messabdeckung, echte Nullcounts, Missingness und eventuelle Restkategorie; **OFFEN**. EMPD2-Archiv-/Messstatus bleibt Grenze. |
| Prozent-only und Präzision | Vertrag verlangt eigenen Präzisionsprozess; S07 ist ein künstlich festgelegtes Beispiel. | Reale Rundung, Nachweisgrenzen, Nenner-/Präzisionsinformation und daraus begründeter Messprozess; **OFFEN**. Kein N=100 und keine Übernahme von S07. |
| Klimaziel, Einheiten, Referenz | Plan §1/Vertrag: TANN/PANN sind nicht GDD5/MTCO/AET-PET. | Zielgrößen, Einheiten, Saisonalität/Bezugsperiode, Quelle/Version und belegter Bezug zur modernen Probe; **OFFEN**. Keine Zielauswahl oder Umbenennung. |
| Zeit und Ort des modernen E | Vertrag verlangt Raum-/Zeitzuordnung und Unsicherheit. | Beprobungs-/Ablagerungsbezug, Koordinaten/Unsicherheit, Klima-Bezugszeit und Zuordnungsregel; **OFFEN**. Keine Reinigung, Aggregation oder Kohorte erstellt. |
| Abhängigkeiten und Evidenzherkunft | Vertrag verbietet Doppelverwendung; gemeinsame latente Größen benötigen klare Rollen. | Mess-/Standort-/Archiv-/Kalibrierabhängigkeiten, Wiederholungen und mögliche Überschneidungen vor Training belegen; **OFFEN**. Keine Unabhängigkeit aus getrennten Spalten ableiten. |
| Räumliche Aufteilung und Support | A4 fordert Festlegung vor Training; AGENTS untersagt Auswahl anhand äußerer Testergebnisse. | Datenrollen, räumliche Aufteilung, Trennung abhängiger Einheiten, Klima-/Typ-/Raumabdeckung und Umgang mit fehlendem Support vorab festlegen; **OFFEN**. Keine Blockgröße, Schwelle oder Region gewählt. |
| Unabhängige Prüfung | Plan §1/AGENTS: bekannte Regionen nicht als vollständig ungesehene Bestätigung darstellen. | Vorab begründeter Prüfzugang, unabhängige Evidenz und feste Auswertungs-/Berichtsregeln; **OFFEN**. Frühere äußere Scores liefern keine neue bestätigende Evidenz. |
| Menezes-nahe Basis und Klimaziele | A4 verlangt Reproduktion oder präzise Abweichungsdarstellung; A1-Grenzen stehen im MODEL_CONTRACT. | Quellen-/Versionszuordnung der später konkret verwendeten Definitionen, kompatible Klimaziele und Abweichungsprotokoll; **OFFEN**. Keine Originalprüfung wiederholt und keine Modellfamilie festgelegt. |
| Moderne Pollenantwort und Unsicherheit psi/q | Vertrag: moderne Kalibrierunsicherheit gemeinsam übertragen; P/Y/D dürfen q nicht rückkoppeln. A3 prüft diese Struktur nur endlich-künstlich. | Reale Likelihood/Priors, Parameter-/Unsicherheitsrollen, properes q, moderne Antwortprüfung und aussagegemäße Unsicherheitsdiagnostik; **OFFEN**. Keine künstlichen Brüche oder Kategorien übernommen. |
| Reproduzierbarer Nachweis | AGENTS fordert separate Prüfung, vollständige Ergebnisbilanz und dauerhaft gesicherte Artefaktbytes. | Später gesperrte Eingänge/Definitionen, vollständige Resultate, unabhängige Nachrechnung mit geeigneter Zerlegung, Ausfälle/Ausschlüsse und dauerhafte Ablage; **OFFEN**. Noch kein realer Lauf erlaubt. |

Die offenen Felder gehören zusammen: Eine konkrete Modell-Likelihood benötigt
den belegten Messstatus des betreffenden Eingangs, die Kalibrierung dessen
passendes Klimaziel, und eine bewertbare Prüfung die vorab festgelegten
Datenrollen. Fehlende Belege dürfen nicht durch eine rechnerische Standardwahl,
eine synthetische Übereinstimmung oder bestehende äußere Resultate ersetzt
werden. Das ist eine Abhängigkeit der Nachweise, keine hier getroffene
Reihenfolge realer Varianten oder Auswahl einer Datenquelle.

## Modulgrenze und Reichweite der A3-Nachweise

Für das spätere A4 liegt die moderne Evidenz in E; q umfasst die relevante
moderne Kalibrierunsicherheit. Fossiles P, weitere Proxys Y und Datierung D
sind im MODEL_CONTRACT Rollen des nachgelagerten Blocks. Dieses Paket wählt
weder diese Eingänge noch C-Prozess, tau-Kurven oder theta aus. Die tatsächliche
Chronologie-Rückkopplung bleibt offen. Moderne Klima-/Zeit-/Ortszuordnung in E
ist trotzdem vor Training zu belegen.

A3 zeigt die bedingte Normierung, gemeinsame Unsicherheit, Statusunterschiede,
Abhängigkeiten, ganzen Alterskurven und Gegenbeispiele innerhalb des künstlichen
Supports. Seine Zahlen liefern weder reale Likelihood/Priors noch einen
Nachweis empirischer Übertragbarkeit. Eine zusätzliche Kalibriergröße darf
nicht als nachgelagertes theta umbenannt werden, um die q-Grenze zu umgehen.
Die sieben bestehenden Artenfunktionen und die fertige Analog-Baseline sind
keine dadurch bestätigte moderne Pollenanteils-Likelihood.

Parnell-Originalzugang ist blockiert, Menezes-(2)/(5) bleibt offen; kein eigener
Neuheitsbeitrag ist abgegrenzt. W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben Grenzen. Die jeweilige betroffene Nutzung
benötigt einen eigenen Beleg; hier werden keine dieser Grenzen neu geprüft
oder aufgehoben. Ein fehlender fossiler Eingang wird nicht still durch einen
modernen ersetzt. Aus diesen offenen Punkten wird auch keine zusätzliche
pauschale Behauptung über die Unmöglichkeit jeder modernen Kalibrierung
abgeleitet.

## Schutz- und Stopgrenzen

Das aktuelle OK erlaubt genau dieses neue Dokument und kurze HANDOFF-/Statuspflege.
Alle bestehenden Designs einschließlich A3_SYNTHETIC_DESIGN, MODEL_CONTRACT,
Modelle, Resultate, Register, Verträge, Importer, Flags, Counts/Gruppen,
Baseline und CLI/App bleiben unverändert. Keine reale Daten-/Modellwahl,
Beschaffung, Bereinigung, Datenmatrix/Kohorte, Implementierung, neue Rechnung,
Fits, Scores, Optimierung, Zufallssimulationen oder Kernläufe; keine manuell
gestarteten Projekt-/Modelltests. A4-Umsetzung und A5/A6 nicht ausgeführt.
Keine Unteragenten, Autorennachrichten oder Merges.

| Stopbedingung | Konsequenz |
| --- | --- |
| Die Abgrenzung erfordert eine neue reale Auswahl, neue Evidenz oder eine Änderung bestehender Vertrags-/Designfelder. | Betroffenen Teil als offen/blockiert festhalten; keine stille Erweiterung. |
| Messstatus, N, Kategorien/Abdeckung oder Prozentpräzision bleiben unbelegt. | Betroffene reale Eingangsnutzung gesperrt; keine Nullfüllung, Pseudo-Counts oder ergänzte Restmenge. |
| Klimaziel, Einheit, Zeit-/Ortsbezug oder Evidenzrolle fehlen bzw. widersprechen einander. | Kein Training auf diesem Eingang und keine Ersetzung durch ähnlich benannte Variablen. |
| Vorab festgelegte Abhängigkeiten, räumliche Datenrollen, Support oder unabhängige Prüfung fehlen. | Kein bestätigender Modellvergleich oder an äußeren Ergebnissen ausgerichtete Variantenwahl. |
| Methodische Definition/Quellenzuordnung reicht für eine behauptete Reproduktion nicht aus. | Reproduktionsanspruch blockiert halten; bekannte Abweichungen präzise benennen, keine Originalgleichung als geprüft ausgeben. |
| Ein späterer Rechenschritt liefert ungültiges q/Z, verwirft Kalibriermasse, verwendet Evidenz doppelt oder verdeckt Nichtidentifizierbarkeit. | Keine gültige Cut-Zielverteilung, eindeutige Wahrheit oder validierte Unsicherheit behaupten; Definition/Implementierung separat klären. |

## Prüfbares Ende und anschließender begrenzter Schritt

**Dieses Paket:** dokumentarische Abgrenzung abgeschlossen. A4-Plananforderungen,
Eingangsrollen, belegte Struktur versus fehlende reale Nachweise, Stopgrenzen
und ein späteres fachliches Ende sind festgehalten. Keine reale Einsatzreife
behauptet. Quellenzuordnung zu den oben genannten Abschnitten, 17 lokale Links/Anker
und Drei-Dateien-Umfang geprüft; 749 geschützte Git-Blobs/Dateimodi einschließlich
eines Symlinks unverändert, Statushistorie erhalten. Sicherung per Commit/eigenem PR gegen #55 ohne Merge.

**Späteres A4-Ende, hier weder autorisiert noch erreicht:** ein gesperrter,
begründeter moderner Eingang samt Typ-/Mess-/Klima-/Zeit-/Ortsrollen; vor
Training festgelegte Abhängigkeiten, räumliche Datenrollen/Support und
unabhängige Prüfung; nachvollziehbare Menezes-nahe Basis oder präzise begrenzte
Abweichung; geprüfte moderne Pollenantwort und Kalibrierunsicherheit mit
vollständiger Resultat-/Ausfallbilanz und dauerhaft gesicherten Artefakten.
Fehlende Pflichtnachweise ergeben einen belegten Teilabschluss oder eine
Blockade. Keine Gatefreigabe oder fossile Gesamtvalidierung daraus ableiten.

Genau ein anschließender Auftrag wird in [HANDOFF](./HANDOFF.md) vorbereitet:
vorhandene EMPD2-Archiv-/Messstatusdokumentation gezielt auf die hier offenen
modernen E-Anforderungen zuordnen, ausschließlich dokumentarisch, ohne reale
Auswahl oder neue Beschaffung. Dieser Folgeauftrag wartet auf Nutzer-OK.
