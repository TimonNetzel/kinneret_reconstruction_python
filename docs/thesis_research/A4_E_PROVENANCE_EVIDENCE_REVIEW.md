# A4: recherchierte Herkunftsnachweise für die EMPD2-Klimafelder

Stand 01.11.2026; Basis [PR #60](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/60),
`76edec21a7f4caa13eb64c054f67033718d479c7`.

**Ergebnis:** Die Internetrecherche liefert zusätzliche Belege zur Periodenangabe,
zur Versionsgeschichte des Hilfsprogramms und zur dokumentierten Feldabbildung.
R01–R03 sind für das konkrete PANGAEA-/SQLite-Archiv weiterhin nur teilweise
belegt. Keine angewandte Unterversion oder Archivperiode wird festgelegt;
`reference_period=unverified` in bestehenden Verträgen bleibt unverändert.

## Auftrag und abgegrenzte Recherche

Der Nutzer hat den übernommenen Startauftrag im Chat ausdrücklich korrigiert:
Codex soll selbst im Internet nach entsprechenden Dokumenten suchen.
Damit entfällt für dieses Paket die Startbedingung eines vom Nutzer benannten
Belegobjekts und die bisherige Recherchesperre. Bewertet werden die unten
identifizierten öffentlich zugänglichen Dokumente gemeinsam; die Auswahl von
Belegdokumenten ist keine Auswahl eines realen Klima-/Trainingsdatensatzes.

Maßstab bleiben unverändert [R01–R03 und Herkunftsregeln](./A4_E_REOPENING_REQUIREMENTS.md#anforderungen-an-künftig-vorzulegende-evidenz),
[MODEL_CONTRACT: Eingänge](./MODEL_CONTRACT.md#rollen-und-unbesetzte-eingänge)
und [Stopbedingungen](./MODEL_CONTRACT.md#stopbedingungen-und-abschlussgrenze).
Zielbezug ist PANGAEA.909130, insbesondere dessen SQLite-Klimafelder
`climate.t_ann/p_ann`; TSV, PostgreSQL, GitHub-Release und SQLite werden
nicht aufgrund ähnlicher Bezeichnungen als identisch behandelt.

Gesucht wurde gezielt nach EMPD2/WorldClim, Herkunft/Verarbeitung,
`climate.tsv`, Periodenangaben und dem im Herausgebercode genannten
`latlon_utils.get_climate`. Gelesen wurden Veröffentlichungs-/Archivmetadaten,
README-Passagen und statischer Herausgebercode. Keine Datenzeilen,
Klimaraster, Datenbank-Dumps, SQLite-Dateien, Samples oder Downloadarchive
wurden geöffnet, beschafft oder ausgewertet. Kein fremder Code ausgeführt.
Ein Suchtreffer, eine Versionsbezeichnung oder ein Repository-Pfad ist kein
Nachweis der tatsächlichen Anwendung auf das Zielarchiv.

## Quellen und genaue Fundstellen

Abruf aller folgenden Stellen am 01.11.2026. Aussagen sind eigene knappe
Übernahmen/Paraphrasen; die Codebezeichnungen dienen als Locator.

| ID | Herkunft / Dokument und Stand | Fundstelle und begrenzter Befund |
| --- | --- | --- |
| S1 | Davis et al. 2020, [EMPD2-Originalartikel](https://essd.copernicus.org/articles/12/2423/2020/essd-12-2423-2020.html), DOI 10.5194/essd-12-2423-2020 | §2 erster Absatz: monatliche, saisonale und jährliche Temperatur/Niederschlag aus WorldClim2; nächster Rasterpunkt, 30 Bogensekunden, kein Klima für marine Proben. §6 verknüpft die Publikation mit PANGAEA.909130 und nennt Excel/PostgreSQL/SQLite. Bestätigt den bereits bekannten Publikationsbeleg, keine neue unabhängige Versions-/Periodenangabe. |
| S2 | Chevalier et al. 2019, [PANGAEA.909130, Metadatenseite](https://doi.pangaea.de/10.1594/PANGAEA.909130), veröffentlicht 26.11.2019, DOI registriert 22.02.2020 | Abstract und „Original version“: drei laut Herausgeber äquivalente Ausgabeformate einschließlich SQLite. „Comment“ bezeichnet die separat präsentierten Klimadaten. Parametertabelle: Jahresmitteltemperatur °C, Jahresniederschlag mm/a; Monatsniederschlag mm/month, Jahreszeiten djf/mam/jja/son. Diese Metadaten benennen keine konkrete angewandte WorldClim-Unterversion, Rasterdatei oder Normalperiode. Die Äquivalenzaussage ersetzt keine ID-/Feld-/Exportprüfung. |
| S3 | Davis et al. 2024, [LGM-Originalartikel](https://cp.copernicus.org/articles/20/1939/2024/), DOI 10.5194/cp-20-1939-2024 | §2.3, letzter Absatz vor §2.4: Klima-Anomalien gegenüber WorldClim 2, Mittel 1970–2000; dasselbe Produkt sei zur Klimazuordnung der modernen Pollenproben in der dortigen Transferfunktion verwendet worden, mit Verweis auf Davis 2020. Neu gelesener, expliziter Perioden-/Anwendungsbezug für diese Studie. Keine Release-/Exportkennung der hier verwendeten SQLite-Datei und keine genaue Unterversion. |
| S4 | WorldClim-Herausgeber, [Historical climate data](https://www.worldclim.org/data/worldclim21.html), beim Abruf aktuelle Produktseite | Einleitungsabsatz: WorldClim 2.1, 1970–2000, Veröffentlichung Januar 2020. Tabelle: monatliche Temperatur °C, Niederschlag mm; Auflösungen einschließlich 30 Sekunden. Gilt für dieses Produkt, belegt dessen Verwendung im Zielarchiv nicht. Nicht aus dem Publikationsnamen WorldClim2 auf 2.1 schließen. |
| S5 | EMPD2-Herausgeber, [Release v2.0](https://github.com/EMPD2/EMPD-data/releases/tag/v2.0), 05.10.2020; [Zenodo.4066498](https://zenodo.org/records/4066498) | GitHub-Tagobjekt `f543a1bf60fefb705e46db400f59ebdac96edaf7` verweist auf Commit `737b76eb269b071422edded6ac332654ff83c103`. Release und Zenodo beschreiben Supplemente zu Davis 2020; Zenodo „Related works“ nennt PANGAEA.909130. Dauerhaft identifizierter Dokumentationsstand, aber die Supplementbeziehung beweist keine Byte-/ID-Gleichheit mit dem PANGAEA-SQLite-Export von 2019. ZIP nicht heruntergeladen. |
| S6 | EMPD2/EMPD-data, [README.rst](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/README.rst), Blob `974b609a7c942a464aa978724ac278be3de02d14` | „Accessing the data“, „Postgres dump“ und „Tab-delimited database“ beschreiben Rohformat, PostgreSQL-Dump und tabellarische Ausgabe sowie veränderliche/latest gegenüber stabilen Releases. Ein dokumentierter Zugangs-/Formatbezug, kein archivgenauer Ausführungs- oder SQLite-Exportbeleg. |
| S7 | Derselbe Commit, [tests/fixes.py](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/tests/fixes.py#L169), Blob `6aa38774423bbd2625cc2b2e5001db86b1b3a20d` | Import `get_climate` aus `latlon_utils` (Z.8); `fix_temperature` Z.169–194 und `fix_precipitation` Z.197–226: bei fehlenden/nan-Feldern mit Koordinaten Abruf von tavg/prec; Speicherung als Temperatur-/Niederschlagsfolgen. Z.216–218 multipliziert das letzte Niederschlagsfeld mit 12 und die vier vorigen mit 3. Statisch dokumentierte Umrechnung, hier weder ausgeführt noch numerisch bestätigt. Kein Laufprotokoll für alle Archivproben; nicht als universelle Neuerzeugung sämtlicher Werte ausgeben. |
| S8 | Derselbe Commit, [Importskript](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/import_into_empd2.py#L362), Blob `f911713f780ab43f129e9b4acc5e94d415b2a94f`; [Erzeugungsschema](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/create_empd2.sql#L249), Blob `c0f1bba5fbb1897678ebd6f218e3e6a1f6679672` | Import Z.362–385: Temperature/Precipitation → Tabelle climate; Updatefolge jan…dec,djf,mam,jja,son,ann, getrennt für t/p; Schlüssel SampleName. Schema Z.249–287 bestätigt die Feldreihenfolge beim INSERT. Dokumentiert PostgreSQL-Feldabbildung einschließlich T_ann/P_ann; kein dokumentierter Lauf und keine belegte Überführung auf die abweichenden SQLite-/TSV-IDs. |
| S9 | Chilipp/latlon-utils, [historische README.rst](https://github.com/Chilipp/latlon-utils/blob/3dfe2f50171c80c7b1af45e609695ec1a3542f46/README.rst#L99), Stand 27.03.2019, Blob `bf2614edbe240f492befd991ab1c93cf90d7cdf2` | „Data download“ Z.99–103 nennt WorldClim2.0. „WorldClim resolutions“ Z.125–131 nennt 10m als Default, änderbar über Argumente/Umgebung LATLONRES. Historischer Bibliotheksstand, keine gesicherte in EMPD2 verwendete Paketversion oder Auflösung. |
| S10 | Dieselbe Bibliothek, [neuere README.rst](https://github.com/Chilipp/latlon-utils/blob/1acecc10a94b0d53be056c355d981358912c9d66/README.rst#L99), Commit 17.01.2022, Blob `96b136e20cef0450a5192465e102c674d3d2e674` | Entsprechende Passage nennt WorldClim2.1. Bibliotheksdokumentation hat sich gegenüber S9 geändert. Diese spätere Nennung darf keine ältere Archivherkunft überschreiben. |

S7/S8 wurden ausschließlich als statische technische Herkunftsdokumente gelesen,
keine Tests importiert oder gestartet. Ergänzend gelesene Dockerfile-/Umgebungs-
texte von EMPD-data/EMPD-admin und die Viewer-Datenseite lieferten in diesen
Passagen keine archivgenaue Paket-/Rasterbindung oder Exportkette.
Ein zunächst angenommener README.md-Pfad im Datenrepository war nicht vorhanden;
die per Baumverzeichnis identifizierte README.rst wurde verwendet.
Die allgemeine WorldClim-2.0-URL `/data/v2.0.html` war im Webwerkzeug
nicht erreichbar; daraus folgt kein inhaltlicher Nichtbeleg.

## Bewertung gegen die unveränderten Anforderungen

| Anforderung | Neue Verbindung und genaue Reichweite | Bewertung für das Zielarchiv / fehlender Nachweis |
| --- | --- | --- |
| **R01 – Klimaeingang identifizieren** | S1/S2 sichern Produktfamilie, räumliche Publikationsmethode, benannte Klimagrößen/Einheiten und Archivkennung. S5 identifiziert den untersuchten GitHub-Stand; S7 → S9/S10 macht die Bibliotheksabhängigkeit und unterschiedliche historische Produktangaben sichtbar. | **Teilweise belegt.** Keine Bindung des PANGAEA-SQLite-Exports an eine konkrete Bibliotheks-/Rasterversion, Eingangsdatei oder Prüfsumme. Weder WorldClim2.0 noch 2.1 als tatsächlich angewandte Unterversion festgelegt. |
| **R02 – Periode genau diesem Eingang zuordnen** | S3 nennt 1970–2000 ausdrücklich im Anwendungszusammenhang der dortigen EMPD2-basierten Transferfunktion. S4 bestätigt diese Periode für das Produkt 2.1. Das ist mehr als die bisher allein betrachtete Periodenlücke im EMPD2-Artikel von 2020. | **Teilweise belegt.** Für die in S3 beschriebene Studie ist die Periodenangabe dokumentiert; deren genaue Bindung an unser SQLite-Eingangsobjekt fehlt. Die zwei Angaben werden nicht zu einem archivgenauen Nachweis zusammengesetzt. Bestehendes `unverified` bleibt erhalten. |
| **R03 – tatsächlich angewandte Kette zu den Archivfeldern** | S7 dokumentiert eine mögliche Befüllungs-/Korrekturkette für fehlende Metafelder samt Niederschlagsaggregation. S8 ordnet die Folgen den PostgreSQL-Klimafeldern zu. S1/S2/S5 belegen Publikations-, Format- und Supplementbeziehungen. | **Teilweise belegt.** Statischer Code ist kein Ausführungsbeleg. Raster-/Bibliotheksstand, angewandte Auflösung, reale Lauf-/Exportkennung und konkrete SQLite-/TSV-Schlüsselbrücke bleiben unbelegt. T_djf und T_ann bleiben getrennt. Keine Gleichheit von Datenausgaben behauptet. |

**Kein bestätigter Widerspruch gleicher Archivgeltung:** 10m in S9 ist ein
konfigurierbarer Bibliotheksdefault; 30 Sekunden in S1 eine publizierte
Anwendungsaussage. Die fehlende Konfiguration verhindert die Gleichsetzung,
beweist aber keine falsche Archivauflösung. Ebenso sind 2.0/2.1 unterschiedliche
Dokumentationsstände, keine hier nachgewiesenen widersprüchlichen
Rastereingänge derselben Archivdatei. Das Veröffentlichungsdatum 2019 bzw.
die Produktfreigabe 2020 ersetzt keinen Versionsbeleg; Archivrevisionen sind
hier nicht geprüft.

## Ergebnis, Grenze und konkreter Anschluss

Die neue Evidenz ist dokumentiert und R01–R03 zugeordnet; das Recherchepaket
ist dokumentarisch abgeschlossen. Die vollständige reale Herkunftskette
bleibt offen. R04–R06, Counts-/Messstatus und alle übrigen Gates bleiben
außerhalb der Abnahme. A4-Kalibrierung/A5/A6 wurden nicht begonnen.
Keine fachliche Freigabe, Quelle still korrigiert, Daten-/Modellwahl,
Klima-Join, Bereinigung, Implementierung, manuelle Projekt-/Modellprüfung,
Rechnung, Autorenkontakt oder Merge.

Der nächste eng begrenzte Dokumentationsschritt ist die **historische
Export-/Versionsbindung**: öffentlich dokumentierte Erzeugungs-/Exportspuren
zu PANGAEA.909130 und dem nachgewiesenen Release v2.0 suchen, einschließlich
der damals wirksamen Bibliotheks-/Auflösungsbindung, sofern veröffentlicht.
Eine nur bekannte heutige Standardkonfiguration reicht nicht.
Keine Wiederholung allgemeiner Produkttexte und keine Aufforderung an den
Nutzer, selbst ein Dokument zu benennen. Konkreter Umfang, Abnahme und Stop
stehen in [HANDOFF](./HANDOFF.md); Sicherung in [PROJECT_STATUS](./PROJECT_STATUS.md).

Prüfung dieses Pakets: Quellen-/Zeilen-/Versionsbezüge, lokale Links/Anker,
genau drei geänderte/neue Dateien, erhaltene Statushistorie und geschützte
Git-Blobs/Dateimodi einschließlich Symlink. Diese Git-Prüfung ist keine
Prüfung realer Klima-/Pollen-Dateien. Automatische Repository-CI getrennt
von nicht ausgeführten manuellen Projekt-/Modelltests berichten.
