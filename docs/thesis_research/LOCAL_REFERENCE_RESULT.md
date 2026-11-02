# Vorläufiges Kinneret-Referenzbeispiel: Ergebnis

Stand 01.11.2026, lokaler Branch `dissertation-reconstruction`, Basis `7f1826c`.
Kein Push, PR, Merge oder Upload neuer Forschungsdaten.

Darstellungsvorgabe des Nutzers: Alter auf der X-Achse **von jung nach alt**,
also links heute/jung, rechts Vergangenheit/alt. Vergleichsgrafik und eingebetteter
Prüfbericht entsprechend aktualisiert; sämtliche numerischen Ergebnisse unverändert.

**Ein überprüfbares technisches Beispiel liegt vor. Fachliche Freigabe steht aus.**
Gerechnet wurde der vorhandene Kinneret-Paperfall mit 33 Taxa, drei Biomen,
Wintertemperatur und Jahresniederschlag. Die vollständige taxonbasierte
Multiproxy-Methode der Dissertation wurde damit nicht reproduziert.

- [Lesbarer lokaler Prüfbericht](../../results/local_reference_review_v2/review.html)
- [Vergleichsgrafik](../../results/local_reference_review_v2/comparison.png)
- [Maschinenlesbare Einzelprüfungen](../../results/local_reference_review_v2/verification.json)
- [Korrigierte Rekonstruktion nach Alter, CSV](../../results/local_corrected_full/rekonstruktion_alter.csv)
- [Quellenabgleich](LOCAL_SOURCE_CROSSWALK.md) und [Wiederholung](LOCAL_REFERENCE_REPRODUCTION.md)

## Tatsächlich ausgeführt

| Lauf | Umfang | Sampling | Gesamtzeit | Maximaler Working Set |
| --- | --- | --- | --- | --- |
| Kleiner diagnostischer Originallauf | 20.000 Iterationen | 1,2 s | 30,9 s | ca. 280 MB |
| Vollständiger Originallauf | eine Kette, eine Million Iterationen | 62,4 s | 126,2 s | ca. 446 MB |
| Vorhandene korrigierte v2-Variante | vier Ketten à zwei Millionen Iterationen | 721,7 s | 1334,2 s | ca. 426 MB |

Die korrigierte Pipeline enthält zusätzlich einen vollständigen Prior-Referenzlauf
(565,4 s) und die Auswertung. Unterschiedliche CPU-Auslastung während der Prüfungen
begrenzt die Vergleichbarkeit dieser Laufzeiten. Es wurde keine neue Modellvariante
erfunden: vorhandenes v2-Modell, feste Gewichte-Priorfamilie Dir(0,5), dieselben
Paperanker, vorhandener Log-Gamma-Vorschlag, bekannte Hastings-/Gamma-Korrekturen.
Nur die Speicherung wurde aus Speichergründen auf `thin=600` begrenzt.
Konfiguration vor Sichtung der Vollergebnisse festgelegt, 10.000 Nach-Burn-in-Ziehungen.

## Bestandene Prüfungen

- **Separater Originalvergleich:** unveränderten Original-C++-Quelltext unabhängig
  vom Paketkern gebaut. Über 20.000 Iterationen stimmen alle sechs Felder
  (einschließlich Akzeptanzentscheidungen) unter derselben Laufzeit bitgenau überein.
- **Vollständige Wiederholungen:** beide Posterior-Sampler nochmals vollständig
  ausgeführt; alle acht gespeicherten Zustands-/Iterationsfelder exakt gleich.
  Original-Wiederholung 40,0 s; korrigiert 291,1 s mit zwei Workern statt einem.
  Nachweise an Posterior-, Eingabe-, Code-, Konfigurations- und Binärhash gebunden.
- **Unabhängige Klimanachrechnung:** separate NumPy-Formel aus Eingangsspektren,
  Alters-Tiefen-Matrix, Gewichten und TF-Ziehungen. Alle **48** geprüften
  CSV-Quantil-/Mittelwertspalten über Alter und Tiefe stimmen an der vorhandenen
  Export-Rundung überein; keine Pipeline-Rekonstruktionsfunktion dafür verwendet.
- **Integrität und Invarianten:** Eingabe-/Quell-/Ausgabehashes stimmen; endliche
  Werte, normierte nichtnegative Taxongewichte, TF-Grenzen und nichtnegativer
  Niederschlag geprüft. Bestehende C++-, Python- und Legacy-Quellen unverändert.
- **Korrigierte Parameterdiagnostik:** alle 43 Größen bestehen die vorhandenen
  Kriterien R̂ ≤1,01 und Bulk-ESS ≥400. Maximaler R̂ **1,000844**, kleinste
  Bulk-ESS **8552,6**, kleinste Tail-ESS **8122,1**.
- **Korrigierte Klimakurven:** alle **362** Prüfungen (181 Alter × zwei Größen)
  bestehen dieselben Kriterien. Maximaler R̂ **1,000431**, kleinste Bulk-ESS
  **8943,3**. Originaldiagnostik aus einer geteilten Einzelkette wird nicht als
  unabhängige Mehrkettenprüfung ausgegeben.

## Quantitativer Vergleich der Stichprobenmediane

Alle Vergleichskurven hier mit derselben direkten Stichproben-Quantilmethode
nachgerechnet; KDE-Quartile und Stichprobenquartile werden nicht vermischt.

| Vergleich | Größe | Mittlere vorzeichenbehaftete Mediandifferenz | Maximale absolute Mediandifferenz |
| --- | --- | --- | --- |
| Original lokal minus Archiv | Wintertemperatur | +0,0038 °C | 0,0615 °C |
| Original lokal minus Archiv | Jahresniederschlag | +3,54 mm/Jahr | 5,85 mm/Jahr |
| Korrigiert minus Original lokal | Wintertemperatur | +0,0539 °C | 0,1341 °C |
| Korrigiert minus Original lokal | Jahresniederschlag | −11,95 mm/Jahr | 18,36 mm/Jahr |

Dies sind beschreibende Abweichungen, keine nachträglich festgelegte fachliche
Akzeptanztoleranz. Die Intervalle beschreiben Parameterunsicherheit des gewichteten
Biom-Klimamittelwerts, keine zusätzliche Streuung einer vollständigen Biom-Mischung.
Der Zusammenhang mit der aus denselben Pollen gebildeten AP-Referenz ist keine
unabhängige externe Klima-Validierung.

## Zwei bestehende Tests bleiben sichtbar auffällig

Ausgewählte vorhandene Tests: 15 Python-Prüfungen bestanden; anschließend
12 C++-/Legacy-Prüfungen bestanden und ein Archivtest fehlgeschlagen;
sechs Samplerprüfungen bestanden und ein kurzer Prior-Test fehlgeschlagen.
Insgesamt **33 bestanden, zwei fehlgeschlagen**, zwei langsame Tests regulär
abgewählt. Keine Behauptung einer vollständig grünen Gesamttestsuite.

1. **Archivierte Bitgleichheit:** 8141/20000 Akzeptanzentscheidungen verschieden,
   erster Unterschied Index 3. Der Legacy-Code nutzt Standardbibliotheks-
   Zufallszahlbausteine. Die lokale libc++ verwendet einen anderen Defaultgenerator
   als libstdc++; lokaler Probeaufruf dokumentiert. Unter derselben lokalen Laufzeit
   sind Original und Paket exakt gleich. Keine Änderung an Golden-Dateien oder Tests.
2. **Kurzer Dirichlet-Prior-Test:** 300.000 Iterationen liefern für einzelne
   Klimagrößen nur ca. 10–20 effektive Stichproben und R̂ bis 1,323. Bei vorab
   festgelegter vierfacher Länge, denselben Seeds und unveränderter Toleranz
   liegen die sechs normierten Mittel bei 0,4945–0,5148; R̂ höchstens 1,00422,
   Bulk-ESS mindestens 1225,8. Die Längendiagnose besteht. Der ursprüngliche
   Kurztest wird trotzdem nicht als bestanden umetikettiert. Diese optionale
   Vorschlagsvariante wurde nicht für den korrigierten Beispiel-Lauf benutzt.

Eine erste Ausführung des neuen Nachrechnungswerkzeugs stoppte an der zusätzlichen
Kettenachse der Archivdatei. Der Leser wurde dafür angepasst, keine Daten verändert.
Der unvollständige Versuch bleibt unter `results/local_reference_review/`
ausdrücklich als `INCOMPLETE` markiert. Der vollständige Nachweis ist ausschließlich
`results/local_reference_review_v2/`.

## Noch erforderliche fachliche Bewertung

Das Beispiel ist zur Prüfung bereit, noch nicht fachlich freigegeben. Zu bewerten
sind die vorläufige Rekonstruktion und Unsicherheitsbänder sowie die Einordnung
als technischer Paper-Referenzfall für die nächste Entwicklungsstufe.
Die dokumentierten Methoden-/Quellenfragen der taxonbasierten Dissertation bleiben
offen: endgültige Textfassung, vollständige historische Transferfunktionen und
Laufeingaben, Taxonzuordnungen, Kalibrierung, Chronologien und Unsicherheit.
Die vorhandenen Monticchio-/Provenienz-Gates werden nicht aufgehoben.
Weitere Fossilfälle beginnen erst nach expliziter fachlicher Freigabe.
