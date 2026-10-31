# A4: Nachweisanforderungen zur Wiederöffnung der Klima-/Zeitverbindung

Stand 31.10.2026; Basis [PR #58](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/58),
`0ac78de8ec007bb8e92c5ea571cefc7ba9e414a0`.
**Die dokumentarische Nachweisanforderung ist abgeschlossen. Keine der offenen
realen Voraussetzungen ist hiermit belegt oder freigegeben.** Reale Klima-/Zeit-
verbindung und Counts-Bereitschaft bleiben blockiert; A4-Umsetzung/reale
Kalibrierung und A5/A6 nicht begonnen.

## Ableitung und Geltungsbereich

Ausgangspunkt sind ausschließlich die offenen Beziehungen C04/C05/C06 und
T02–T04 aus [A4_E_CLIMATE_TIME_GAPS](./A4_E_CLIMATE_TIME_GAPS.md#nachweiskette-für-klimaquelle-version-und-referenzperiode)
und dessen [Zeitnachweiskette](./A4_E_CLIMATE_TIME_GAPS.md#nachweiskette-für-probenzeit-integration-und-räumlichen-bezug).
Die dort zugeordneten K1–K5 wurden hier nicht erneut aufgesucht. Deren
Dokumentwiederholungen bilden keine unabhängigen zusätzlichen Belege; die
begrenzten Nichtbelege behaupten keine weltweite Nichtexistenz von Evidenz.

Der unveränderte [MODEL_CONTRACT, Rollen und unbesetzte Eingänge](./MODEL_CONTRACT.md#rollen-und-unbesetzte-eingänge)
verlangt für E unter anderem Quelle/Version, Klimadefinition, Einheit,
Bezugsperiode, räumlichen und zeitlichen Bezug, Unsicherheit und Abhängigkeiten.
Seine [Supportgrenze](./MODEL_CONTRACT.md#support-und-identifizierbarkeit)
und [Stopbedingungen](./MODEL_CONTRACT.md#stopbedingungen-und-abschlussgrenze)
verbieten die Ersetzung fehlender Zuordnungen durch Annahmen oder breitere
Priors. Daraus werden unten notwendige Nachweise abgeleitet; der Vertrag wird
weder ergänzt noch als real ausgefüllt behandelt. Maßgeblich bleiben
[Plan §1](./MODULAR_RECONSTRUCTION_PLAN.md#1-ziel-und-festgehaltene-arbeitsrichtung),
§2 ausschließlich Zeile A4 und [§4](./MODULAR_RECONSTRUCTION_PLAN.md#4-tokensparender-arbeitsmodus).
Es wird keine neue numerische Schwelle oder empirische Zulässigkeitsregel gesetzt.

## Anforderungen an künftig vorzulegende Evidenz

Die Belegtypen sind mögliche dokumentarische Träger der nötigen Aussagen;
ihre Nennung bestätigt weder ihre Existenz noch ihren Inhalt. Alle sechs
Zeilen sind **offene Nachweisanforderungen**. Eine spätere Prüfung muss für
jede beanspruchte Verbindung ihre konkrete Fundstelle und Reichweite ausweisen.

| Anforderung | Vorzulegender Belegtyp und notwendige Angaben | Identitätsbezug | Zulässiger Schluss bei tragfähigem Beleg | Weiterhin offen bei Nichtbeleg |
| --- | --- | --- | --- | --- |
| **R01 – Klimaeingang eindeutig benennen (C04)** | Archivbezogene Herkunfts-/Erzeugungsdokumentation oder eindeutig zugeordneter Release-/Verarbeitungsnachweis: Produkt, genaue angewandte Version bzw. unverwechselbarer Stand, Eingangsobjekt, Klimagrößen/Definitionen, Einheiten und räumlicher Bezug. Ein generischer Produkttext genügt ohne Anwendungsbezug nicht. | Eingang muss dauerhaft wiedererkennbar sein, etwa durch Archivkennung, Release und Dateikennung/Prüfsumme; Bezug zum bezeichneten Zielarchiv ausweisen. Der Name „WorldClim2“ allein unterscheidet die offene Unterversion nicht. | Genau der dokumentierte Klimaeingang wäre identifizierbar; keine Übertragung auf andere Versionen, Releases oder Felder. Die tatsächliche Anwendung erfordert zusätzlich R03. | Konkrete angewandte Version und Eingangsidentität bleiben offen; kein WorldClim-2.1-, CRU- oder anderes Ersatzlabel ergänzen. |
| **R02 – Referenzperiode diesem Eingang zuordnen (C05)** | Eingangsbezogene Metadaten oder explizit verknüpfte Dokumentation: genaue Referenz-/Normalperiode, Bedeutung der zeitlichen Aggregation/Saisonalität der betreffenden Größen und Fundstelle der Zuordnung. Abweichende Periodenangaben mit ihrem jeweiligen Objekt erhalten. | Periodenangabe muss denselben Klimaeingang/Stand wie R01 betreffen; eine Kette über eindeutig verknüpfte Dokumente ist auszuweisen. Keine Standardperiode aus einem bloßen Produktfamiliennamen ableiten. | Die Referenzperiode dieses Eingangs wäre dokumentarisch belegt. Noch kein Nachweis, dass sie für die Archivfelder oder Proben gilt. | Periodenbezug bleibt `unverified`; weder eine generische Produktperiode noch ein Dissertations-/CRU-Vergleichszeitraum schließt die Lücke. |
| **R03 – angewandte Kette bis zu Archivfeldern belegen (C06)** | Archivbezogener Erzeugungs-/Verarbeitungsnachweis: welcher Eingang aus R01/R02 durch welchen tatsächlich angewandten Zuordnungs-/Übernahmeschritt in welche Zielobjekte/-felder gelangte; dokumentierte Methode, räumlicher Zuordnungsbezug, Feldabbildung, Einheiten sowie gegebenenfalls angewandte Transformationen und Versions-/Releaseübergänge. Eine mögliche Importfunktion oder Methodenskizze allein belegt keine Anwendung. | Zielarchiv mit Stand/Release und Objektkennung, Tabelle/Felder sowie betroffene Proben-/Standortschlüssel oder ausdrücklich abgegrenzter Geltungsbereich. Verbindungen zwischen abweichenden TSV-/SQLite-IDs müssen belegt sein; keine Identität aus ähnlichen Namen ableiten. `climate.t_ann/p_ann` und `t_djf` nicht gleichsetzen. | Quelle/Version/Periode → angewandter Schritt → genau bezeichnete Archivfelder wäre für den belegten Umfang nachvollziehbar. Kein neuer Klima-Join und keine Aussage über ungeprüfte Releases/Proben. | Anwendungskette und Archivgeltung bleiben offen, auch bei bekannter Produktperiode. Feldexistenz, numerische Werte oder ein ausführbarer Join ersetzen den Nachweis nicht. |
| **R04 – Probenzeit und ihre Unsicherheit zuordnen (T02)** | Probenbezogene Sammlungs-/Datierungsdokumentation oder eindeutig zugeordnete Metadaten: welche Zeit beobachtet bzw. datiert wurde, Zeitangabe/-bereich, Konvention/Referenz und Einheit, Methode, Unsicherheit und bekannte Lücken. Unbekanntes ausdrücklich erhalten. | Eindeutige Verbindung zur jeweiligen Archivprobe und ihrem Stand; Herkunft/Probenkennung und Zuordnungsschritt nennen. Ein Befund zu den drei bekannten Zeugen gilt nur für sie; keine allgemeine Archivaussage daraus. | Für genau die belegte Probe wäre die bezeichnete Zeit samt Aussagegrenze zugeordnet; keine Auswahl geeigneter Proben und noch kein Integrations- oder Klimaperiodennachweis. | Unbekannte Zeugenalter bleiben unbekannt. Feldname, negative Alterszahl ohne erklärten Bezug, Publikationsjahr oder Datenbankname „modern“ reichen nicht. |
| **R05 – Sammlung, Ablagerung und Integration unterscheiden (T03)** | Proben-/methodenbezogene Dokumentation: Verhältnis von Sammlungszeit, repräsentierter Ablagerungszeit und zeitlicher Integration, gegebenenfalls Mehrfachsammlung/Mischung, Unsicherheiten und Grenzen. Eine nicht quantifizierbare Integration muss als solche benannt sein; keine Nullunsicherheit unterstellen. | Methode/Kontext müssen genau den betreffenden Proben oder einer nachweislich zugeordneten Gruppe gelten. Allgemeine Eigenschaften eines Probentyps ohne diesen Bezug reichen nicht. | Die repräsentierte Zeitspanne und ihre belegbaren Grenzen wären zugeordnet; ein Sammlungsdatum allein wird nicht zur Integrationsdauer. | Integrationsbereich und zeitliche Repräsentativität bleiben offen; weder Moss-/Kontextlabel noch ein Klima-Join belegen Gleichzeitigkeit. |
| **R06 – gemeinsame Klima-/Probenzeitgeltung begründen (T04)** | Nachvollziehbare fachliche Begründung, die die gesicherte Klimaperiode/-definition aus R01–R03 mit Probenzeit und Integration/Unsicherheit aus R04/R05 verbindet: beanspruchter zeitlicher Bezug, Reichweite, verbleibende Unsicherheiten, Annahmen und gegebenenfalls Abhängigkeiten. Eine bloße Jahresüberschneidung ersetzt diese Begründung nicht. | Begründung muss dieselben identifizierten Klima-/Archivfelder und Proben/Probenbereiche zusammenführen; unterschiedliche Räume, Zeitkonventionen oder Stände dürfen nicht stillschweigend vermischt werden. | Nur die begründete zeitliche Zuordnung im belegten Umfang wäre später bewertbar. Keine automatische Auswahl von E, Schwelle für Zeitabstände oder Freigabe einer Kalibrierung. | Gemeinsame Geltung bleibt offen, sobald eine Seite oder ihre Verbindung unbelegt ist. Bekannte Klimaperiode ersetzt keine Probenzeit und bekannte Probenzeit keine angewandte Klimaperiode. |

## Beleg, Annahme, Verfügbarkeit und Freigabe auseinanderhalten

Ein künftig vorgelegter Beleg muss seine Herkunft erkennen lassen: Urheber/
herausgebende Stelle, Dokument-/Objektkennung und Stand, genaue Fundstelle,
Originalaussage oder ausdrücklich gekennzeichnete Übernahme, beanspruchte
Archiv-/Feld-/Probenidentität und Aussagegrenzen. Bei indirekter Übernahme ist
die Verknüpfung zum Ursprungsbeleg offenzulegen; Wiederholung allein stärkt den
Nachweis nicht. Nicht zuordenbare oder widersprüchliche Angaben gleichen
Geltungsbereichs halten die betroffene Verbindung offen.

Eine Annahme ist als Annahme auszuweisen und schließt keine Nachweislücke.
Technische Verfügbarkeit bedeutet lediglich, dass ein Feld/Objekt vorhanden
oder verknüpfbar ist; sie belegt weder Herkunft noch Eignung. Die spätere
dokumentarische Bewertung kann eine konkrete Verbindung als belegt, nur
teilweise belegt oder unbelegt einordnen. **Auch vollständig zugeordnete
Klima-/Zeitbelege wären keine fachliche Gesamtfreigabe:** Counts/N,
Null-/Missingness-/Nennerstatus, Typ-/Messstatus, Abdeckung, Archivstatus,
Abhängigkeiten und übrige reale Gates bleiben zusätzliche Voraussetzungen.
Keine nachträgliche Entscheidung anhand äußerer Testergebnisse ableiten.

Die vorliegenden Texte enthalten keine neue reale Evidenz. Die Anforderungen
bezeichnen, was eine spätere gesondert beauftragte Prüfung benötigen würde;
sie wählen keine reale Version, Periode, Probe, Kohorte, Datenrolle oder
Modellvariante. Ein zusätzlich nötiger realer Entscheid wird OFFEN gehalten.
Es wird keine Beleganforderung versandt und keine Quelle beschafft.

## Abnahme und Stopgrenze dieses Pakets

C04/C05/C06 und T02–T04 sind mit Belegtyp, Angaben, Identitätsbezug, begrenztem
Schluss und Nichtbelegfolge abgedeckt. Abnahme betrifft ausschließlich diese
aus dem unveränderten Vertrag abgeleitete Dokumentation. Quellenbezüge,
Links/Anker, Drei-Dateien-Diff, erhaltene Statushistorie und geschützte
Git-Blobs/Dateimodi werden geprüft; Sicherung und CI-Stand stehen in
[PROJECT_STATUS](./PROJECT_STATUS.md) und [HANDOFF](./HANDOFF.md).

Alle bestehenden Fachdateien einschließlich MODEL_CONTRACT, Designs,
Modelle, Resultate, Register, Verträge, Importer, Flags, Counts/Gruppen,
Baseline und CLI/App bleiben unverändert. Keine neue Recherche, Original-/
Archiv-/Literatursuche, Beschaffung, reale Daten-/Dateiprüfung, Auswahl,
Klima-Joins, Bereinigung, Matrix/Kohorte, Typbrücke, Implementierung, Tests,
Rechnungen, Fits, Scores, Optimierung, Simulationen oder Kernläufe.
Keine A3-Wiederholung; A1/A2 dokumentarisch abgeschlossen, A3 ausschließlich
endlich-synthetisch bestanden. A4 wird nicht umgesetzt, A5/A6 nicht begonnen.
Parnell, Menezes-(2)/(5), W96, EMPD2-Archiv-/Messstatus und reale
T0/T1/T3/T4/Monticchio-P1 bleiben Grenzen; kein eigener Neuheitsbeitrag
abgegrenzt. Keine Gatefreigabe, Unteragenten, Autorennachrichten oder Merges.

Genau ein begrenzter Folgeauftrag ist in HANDOFF vorbereitet: ein künftig vom
Nutzer eindeutig bezeichnetes archivbezogenes Herkunftsdokument gegen
R01–R03 dokumentarisch bewerten. Ohne dieses neue Belegobjekt bleibt das Paket
blockiert; dieselben Altangaben werden nicht erneut als neue Evidenz geprüft.
