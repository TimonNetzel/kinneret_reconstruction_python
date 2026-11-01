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

## Historischer Erzeugungs- und Exportnachweis, 01.11.2026

Fortsetzung ab PR #64, Inhaltscommit `65bd8b267300daf7cb8d55bd0c09f901d033391b`,
geprüfter Head/Basis `7628f1267ff60ec24f191365826e87a6add9caa5`.
Dieser Abschnitt ergänzt S1–S10; deren damalige Bewertung bleibt als Historie
erhalten. Recherchiert wurden ausschließlich die drei beauftragten Verbindungen:
historischer Build/Rasterstand, Anwendung/Export und PANGAEA↔GitHub/Zenodo.

**Antwort:** Öffentlich belegt sind eine historische **Buildvorgabe mit 5m**,
ein **WorldClim-2.0-Downloader im historischen Bibliothekscode** und eine
**Supplementbeziehung des Release v2.0 zu PANGAEA.909130**. Damit ist weiterhin
**nicht belegt, mit welchem tatsächlich installierten Programm-/Rasterstand
und welcher wirksamen Auflösung die PANGAEA-SQLite-Klimafelder erzeugt wurden**.
Die publizierten 30 Bogensekunden bleiben eine Anwendungsaussage (S1), die
5m eine andere, hier nicht archivgenau zugeordnete Buildvorgabe. Keine der
beiden Angaben wird zur nachgewiesenen Laufkonfiguration des Zielobjekts erklärt.

### Historische Quellen und Reichweite

Alle Abrufe am 01.11.2026; knappe eigene Übernahmen, kein fremder Code ausgeführt.
Commit-Datumsfilter identifizieren erreichbare historische Quellstände, keine
damals installierten Images/Pakete. Unversionierte Image-/master-Verweise
werden nicht nachträglich zu historischen Pins umgedeutet.

| ID | Herausgeberquelle, unverwechselbarer Stand und Fundstelle | Dokumentierter Befund und Beleggrenze |
| --- | --- | --- |
| H1 | EMPD2/EMPD-data v2.0, Commit `737b76eb269b071422edded6ac332654ff83c103`: [Dockerfile Z.1–14](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/Dockerfile#L1), Blob `61d8336743b32dfcdb8140f37815675ae2119a9e`; [.travis.yml Z.13–17](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/.travis.yml#L13), Blob `a2249bccef63f7705bfd72f342b2397aa28c26b9` | Verweist auf `empd2/empd-admin-base` ohne Tag/Digest; CI beschreibt Build und Testaufruf. Dauerhafter Quellstand, aber kein Image-/Ausführungsbeleg für den PANGAEA-Export. |
| H2 | EMPD2/EMPD-admin, [Installationsdokumentation Z.64–99](https://github.com/EMPD2/EMPD-admin/blob/5cc6071f372cc48cbdad3ccc10a7558af20b2c84/docs/install.rst#L64), Blob `0afb540c7b7bef0c48a574891b365aa1fc89a4a4`, Commit `5cc6071f372cc48cbdad3ccc10a7558af20b2c84` vom 20.10.2019 | Verknüpft `latlon-utils` und die Umgebungsdatei in EMPD2/empd-admin-base. Damit ist der zusätzliche Buildquellenpfad tatsächlich vom Herausgeber dokumentiert. [Vergleich zum Stand 76c73ae vom 12.03.2020](https://github.com/EMPD2/EMPD-admin/compare/5cc6071f372cc48cbdad3ccc10a7558af20b2c84...76c73aea3a0fcd293770a0ce96031a82685d4e41) ändert nur common.py/diff.py, nicht diese Installations- oder unten genannten Exportpassagen. Keiner dieser Stände ist damit als PANGAEA-Lauf identifiziert. |
| H3 | EMPD2/empd-admin-base, Commit `10cfa4e71348a1cec1af429a8cab1e2694a87f3b` vom 01.05.2019: [Dockerfile Z.1–4, 23–34](https://github.com/EMPD2/empd-admin-base/blob/10cfa4e71348a1cec1af429a8cab1e2694a87f3b/Dockerfile#L1), Blob `12dcf02e52926717296ff97211a1a11482428409`; [empd-admin-environment.yml Z.5–26](https://github.com/EMPD2/empd-admin-base/blob/10cfa4e71348a1cec1af429a8cab1e2694a87f3b/empd-admin-environment.yml#L5), Blob `b0c85a7b419dee1f8bc11a76b92066e71da6ffaa` | Basis `postgres:11.2`, Python 3.7; `LATLONRES=5m`; Downloadvorgabe `tavg prec`, Breiten 0–90, `-res 5m`. Sowohl Downloadumgebung als auch Anwendungsumgebung installieren `latlon-utils` ohne Versionspin. Dies ist der per Commit-Datumsfilter vor 26.11.2019 und 05.10.2020 erreichbare letzte Quellstand. Belegt eine konkrete historische Bauanweisung, weder deren Ausführung noch den verwendeten Paket-/Rasterdateistand. |
| H4 | Chilipp/latlon-utils, Commit `510511d207fc30c8a38a457f5b4494daaf361b3c` vom 13.11.2019: [download.py Z.16, 68–110](https://github.com/Chilipp/latlon-utils/blob/510511d207fc30c8a38a457f5b4494daaf361b3c/latlon_utils/download.py#L16), Blob `f56127f8ef8c054324cfb534b07236454bca79c1`; [__init__.py Z.10, 52–60, 185–224, 329–343](https://github.com/Chilipp/latlon-utils/blob/510511d207fc30c8a38a457f5b4494daaf361b3c/latlon_utils/__init__.py#L52), Blob `001e4349b711ce05af8e566c93cd637ab34d36c7` | Quellversion 0.0.5; Downloader adressiert `/worldclim/v2.0/tif/base/` und `wc2.0_<res>_<var>.zip`, wandelt GeoTIFF in NetCDF. Auflösung: Funktionsargument vor `LATLONRES`, sonst 10m; alternative `data_files` möglich. Der Downloader-Blob ist auch in den historischen Ständen `cc98059fd1f624852bdbed9ff7e8bfd834df8c71` (28.03.2019) und `9182d1f214c8efbb4769c1021c481c3703be4b61` (28.04.2020) identisch. Das belegt historische WorldClim-2.0-Programmlogik, **keinen** tatsächlich installierten Stand aus H3 und keine Identität realer Rasterbytes. ZIP/Raster/NetCDF nicht geöffnet. |
| H5 | EMPD2/EMPD-admin im H2-Stand: [finish.py Z.131–178](https://github.com/EMPD2/EMPD-admin/blob/5cc6071f372cc48cbdad3ccc10a7558af20b2c84/empd_admin/finish.py#L131), Blob `9e8f31be1bef710eb38ac285670704f087ee4324`; [repo_test.py Z.302–343](https://github.com/EMPD2/EMPD-admin/blob/5cc6071f372cc48cbdad3ccc10a7558af20b2c84/empd_admin/repo_test.py#L302), Blob `4a306c6c1b1cda3b937a7e2cd7dedb44665f77be` | Statische Folge: Importskript → PostgreSQL/pg_dump → Export öffentlicher Tabellen als TSV. S7 wurde gezielt bestätigt: Klima-Fixaufrufe Z.184–186/213–215 übergeben keine eigene Auflösung und betreffen fehlende/nan-Felder. Weder diese mögliche Kette noch S8 dokumentiert hier den konkreten PANGAEA-SQLite-Export mit seiner Schlüssel-/Feldidentität. |
| H6 | EMPD2/EMPD-data, ausschließlich [Export-Commitmetadaten 8b28577](https://api.github.com/repos/EMPD2/EMPD-data/git/commits/8b28577fa7e64d284180816202f8ffaa73cc3546) und [direkter Vorgänger b5a97bb](https://api.github.com/repos/EMPD2/EMPD-data/git/commits/b5a97bb41c5a96e82dd50665c6504d0d1e73e40c), 05.10.2020, 08:54:01 bzw. 08:53:59 UTC | Committexte dokumentieren TSV-Aktualisierung aus PostgreSQL sowie vorherigen Dump für update.tsv. Der erste Commit ist über die pfadbezogene Commit-Historie von `tab-delimited/climate.tsv` am Release-Ref auffindbar. Eine konkrete Exportspur für diese GitHub-Ausgabe, stärker als bloß vorhandener Code; kein Laufmanifest mit Bibliotheks-/Rasterversion, wirksamer Auflösung oder Bezug zum PANGAEA-SQLite-Objekt. Keine Datenblobs oder Datenpatches gelesen. |
| H7 | [GitHub-Release v2.0](https://github.com/EMPD2/EMPD-data/releases/tag/v2.0), [Tagobjekt f543a1b](https://api.github.com/repos/EMPD2/EMPD-data/git/tags/f543a1bf60fefb705e46db400f59ebdac96edaf7), [Zenodo.4066498, Description/Related works](https://zenodo.org/records/4066498), [PANGAEA.909130, Abstract/Original version](https://doi.pangaea.de/10.1594/PANGAEA.909130) | Erneut gezielt gelesen: GitHub verlinkt Zenodo; Tag verweist auf `737b76e`; Zenodo nennt ausdrücklich die Supplementbeziehung zu PANGAEA und zum Artikel. PANGAEA beschreibt seine drei äquivalenten Originalformate, nennt dort aber keinen GitHub-Commit/Zenodo-Build als Erzeuger. **Öffentliche bibliografische Verbindung belegt; konkrete Export-/ID-/Feldgleichheit nicht belegt.** Die Zenodo-ZIP-Prüfsumme identifiziert nur das dortige Objekt; sie verknüpft es nicht mit den PANGAEA-SQLite-Bytes. |

Ergänzend gezielt geprüft: EMPD-admin-Dockerfile, environment.yml und setup.py
im H2-Stand; EMPD-data-Quellbaum am Release-Ref als Pfad-/Objektverzeichnis,
nicht als Dateninhalt. Zwei begrenzte Websuchrunden zu 909130/Export/SQLite/
WorldClim und Committextsuche in EMPD-data vor 06.10.2020 nach WorldClim,
sqlite und resolution lieferten keinen zusätzlichen archivgenauen Beleg.
Leere Suchresultate beweisen keine weltweite Nichtexistenz. S1 §2/§6 und S2/S5
wurden nur zur konkreten Archiv-/Releasegrenze nachgelesen; keine erneute
allgemeine Produktrecherche. Alle 14 gelesenen vollständigen Quelltexte der
Build-/Bibliothekskette stimmen mit ihren Git-Blob-SHAs überein.

### R01–R03 und Stop

| Anforderung | Stand nach historischer Prüfung | Konkrete Restlücke |
| --- | --- | --- |
| R01 | **Teilweise belegt.** Historische WorldClim-2.0-Downloaderlogik und Buildvorgabe sind genauer identifiziert. | Tatsächlich für PANGAEA verwendete Programm-/Bibliotheksversion, Image/Umgebung, Rastereingang mit Stand/Dateiidentität und wirksame Auflösung **unbelegt**. 0.0.5/2.0/5m sind keine Auswahl für das Archiv. |
| R02 | **Teilweise belegt**, durch H1–H7 kein neuer archivgenauer Periodenbeleg. S3 behält seinen begrenzten Studienbezug. | Verknüpfung der Referenzperiode mit genau dem angewandten PANGAEA-Klimaeingang **unbelegt**; `unverified` bleibt bestehen. |
| R03 | **Teilweise belegt.** Statische Feld-/Verarbeitungskette und eine GitHub-Exportspur sowie Supplementbeziehung sind dokumentiert. | Ausführung mit festgelegten Eingängen → identifizierter PANGAEA-SQLite-Export → `climate.t_ann/p_ann` samt Proben-/Schlüsselgeltung **unbelegt**. TSV-/PostgreSQL-/SQLite-Identität nicht aus Namen abgeleitet. |

Die 5m-Bauvorgabe und die 30″-Publikationsangabe haben keinen nachgewiesenen
identischen Lauf-/Objektbezug. Deshalb weder Bestätigung gleicher Archivauflösung
noch bestätigter Widerspruch am selben Objekt. Ein Datum allein beweist keine
verwendete WorldClim-Unterversion. Keine heutige Konfiguration wurde rückdatiert.

**Begrenzter Nichtbeleg und Stop:** In den oben tatsächlich geprüften
Herausgeberunterlagen fehlt ein Export-/Ausführungsnachweis, der Programm,
Rasterstand und Auflösung mit PANGAEA.909130 und dessen SQLite-Klimafeldern
verbindet. Weitere reale Datei-/Raster-/Datenbankprüfung oder Herausgeberkontakt
wären eigene, hier nicht ausgeführte Arbeit; auch ein Bytevergleich allein
würde den damaligen Programmlauf nicht beweisen. Keine weitere Suchschleife,
kein Nutzer-Dokument als künstliche Startbedingung. Das beauftragte Paket ist
dokumentarisch abgeschlossen; die reale Herkunftskette bleibt blockiert.

Keine Datenzeilen, Datenbank-Dumps, ZIP-Inhalte oder Raster beschafft/geöffnet,
keine Auswahl, Implementierung, manuellen Projekt-/Modelltests oder Rechnungen.
Keine Gatefreigabe/A4-Umsetzung/A5/A6, Unteragenten, Nachrichten oder Merges.
Übrige bestehende Fachdateien bleiben unverändert. Quellen-/Link-/Diff-/
Blobprüfung und Sicherung stehen in Status/HANDOFF. Genau ein anschließendes
Paket ist dort vorbereitet: öffentliche Probenzeit-/Integrationsdokumentation
gegen R04/R05 prüfen; kein erneuter R01–R03-Durchlauf und keine Ausführung hier.

## Probenzeit und Integration: R04/R05, 01.11.2026

Fortsetzung ab PR #65, Inhaltscommit `6a147f10313f92b022853a98375ecad93c1bf301`,
geprüfter Head/Basis `6cb9603fa3dc6262f81f1ddb7af7d4492219aa8a`.
Danach in #65 nur HANDOFF-/Statusverweise; Head-CI erfolgreich.
Die historischen Abschnitte oben bleiben unverändert. Dieses Paket prüft nur
öffentliche Zeit-/Sammlungsdefinitionen gegen die unveränderten
[R04/R05](./A4_E_REOPENING_REQUIREMENTS.md#anforderungen-an-künftig-vorzulegende-evidenz)
und [T01–T06](./A4_E_CLIMATE_TIME_GAPS.md#nachweiskette-für-probenzeit-integration-und-räumlichen-bezug).
Keine erneute R01–R03-Suche und keine R06-Bewertung/Freigabe.

**Ergebnis:** R04 und R05 sind **teilweise belegt**: öffentliche allgemeine
Definitionen und eine begrenzte Integrationsregel sind nachvollziehbar.
Eine tatsächliche Zeit-/Unsicherheits-/Integrationszuordnung zu identifizierten
PANGAEA-Archivproben ist dadurch **unbelegt**. Die früheren unbekannten Alter
der drei Zeugen werden nicht ergänzt; keine Aussage über alle Archivproben.

### Gelesene Herausgeberstellen und ihre Reichweite

Abrufe am 01.11.2026; eigene knappe Paraphrasen. Allgemeine Dokumentation,
beabsichtigter Erfassungsweg und tatsächlicher Probennachweis werden getrennt.
Keine Datenlinks, Datenzeilen, Datenbank-, Raster- oder ZIP-Inhalte geöffnet.

| ID | Quelle, Stand und genaue Fundstelle | Zeitbedeutung / Unsicherheit und Grenze |
| --- | --- | --- |
| Z1 | Davis et al. 2020, DOI [10.5194/essd-12-2423-2020](https://essd.copernicus.org/articles/12/2423/2020/essd-12-2423-2020.html), Endfassung 09.10.2020: §1, Absatz unmittelbar vor Tabelle 1; §2, Absatz zum Datenbankprotokoll; §2.2, erste Absätze. [PDF S.2426–2427, Tabelle 1](https://essd.copernicus.org/articles/12/2423/2020/essd-12-2423-2020.pdf#page=5), als Text gelesen. | §1 nennt das Sammlungsjahr als Metadateninhalt; Tabelle 1 unterscheidet Probenalter BP (numerisch), Altersunsicherheit (Liste), Material und Entnahmemethode. §2 beschreibt jünger als 200 BP und bei Fallen mindestens zehnjährig gemittelte Daten statt Monats-/Jahresdaten. §2.2 beschreibt die Übernahme von Beiträgerdaten und Ergänzung fehlender Angaben aus Beiträger-/Originalquellen. Das ist ein allgemeines Protokoll, kein Datierungsprotokoll oder Integrationsintervall einer konkreten Probe. |
| Z2 | EMPD2/EMPD-data v2.0, Commit `737b76eb269b071422edded6ac332654ff83c103` vom 05.10.2020, [README.rst Z.81–86](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/README.rst#L81) und [Z.112–144](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/README.rst#L112), Blob `974b609a7c942a464aa978724ac278be3de02d14`. | `SampleName` ist der EMPD-Schlüssel, `OriginalSampleName` die Autorenkennung. `AgeBP` bezeichnet Jahre vor (+) bzw. nach (−) AD 1950. `AgeUncertainty` ist ein Zuverlässigkeitscode: A ab 0 BP bis „Present“; B letzte 100 Jahre / 0–50 BP; C letzte 250 Jahre / 50–200 BP. Kontext, Material und Sammlungs-/Entnahmemethode sind getrennte Codefelder; Notizen und Publikationsfelder erlauben weitere Angaben. Keine numerische Fehlerverteilung, Datierungsmethode oder Integrationsdauer definiert; keine tatsächliche Befüllung/Zuordnung geprüft. |
| Z3 | Derselbe Release-Commit, [CONTRIBUTING.md, „Quick and Dirty“, Z.22–27](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/CONTRIBUTING.md#L22) sowie „Contributing new data through Github“, Blob `807c210e1d3144d3bdfd3bd13ed97e218bccab88`. | Verweist auf eine Metadatenvorlage und beschreibt die Einreichung einer Metadatenzeile je Probe mit `SampleName`-Bezug. Nur dokumentierter Einreichungsweg; keine geprüfte Einreichung, Sammlung oder Datierung. Die dortige Vorlage ist als veränderlicher master-Link angegeben, kein eigener archivgenauer Zeitbeleg. Vorlage und verlinkte Beispieldaten nicht geöffnet; keine Nachricht versandt. |
| Z4 | Chevalier et al. 2019, [PANGAEA.909130](https://doi.pangaea.de/10.1594/PANGAEA.909130), veröffentlicht 26.11.2019, DOI registriert 22.02.2020: Abstract, Original version, Comment und vollständige Parameterliste der Landingpage. | Die Parameterliste gehört laut Comment zur Klimatabelle; sie definiert kein Probenalter, Sammlungsjahr, Datierungsverfahren oder Integrationsintervall. Links zu drei Originalformaten und deren behauptete Äquivalenz schließen die fehlende Probenzeit-/Methodenzuordnung nicht. Veröffentlichungs-/Registrierungsdatum sind keine Sammlungsdaten. Nur Metadatenseite gelesen; keine Tabellenzeilen oder Originaldateien beschafft. |

**Auslegung und Grenzen:** Die Vorzeichen-/Referenzkonvention aus Z2 ist
dokumentiert, ihre reale Anwendung auf eine bestimmte Archivprobe hier nicht.
Die relativen Zeitformulierungen der Klassen B/C stehen neben festen
BP-Bereichen; ihr Bezugsdatum wird in dieser Passage nicht angegeben.
Die Originalangaben bleiben nebeneinander erhalten, ohne Neuberechnung,
Aktualisierung auf 2026 oder Umdeutung in ±Jahre, Standardabweichung oder
Konfidenzintervall. Ein möglicherweise historischer relativer Wortlaut ist
kein hier bestätigter Widerspruch zweier Datierungen derselben Probe.

Die Sammlungsjahr-Aussage in Z1 und das Probenalterfeld in Z2 liefern allein
keine allgemeine Gleichsetzungsregel. Material-/Methodencodes sind kein
Nachweis der repräsentierten Ablagerungszeit. Die Fallenregel aus Z1 belegt
eine Protokollanforderung, keine tatsächliche Mindestintegration jedes
Moos-, Boden- oder Sedimentdatensatzes und keine konkrete Mischung.
Nicht quantifizierte Integration bleibt unbekannt, niemals Nullunsicherheit.

### Begrenzte Suche und Zugriffsgrenze

Zunächst zwei gezielte Websuchen zu Davis 2020/EMPD2-Alter/Sammlung und
PANGAEA.909130; danach Endfassung, Tabelle 1, Landingpage und README am
festen Release-Ref gelesen. Die direkt verlinkte CONTRIBUTING-Dokumentation
wurde nachgelesen. Der Git-Quellbaum diente nur der Identifikation von
Dokumentationspfaden/Blobkennungen; keine darin bezeichneten Datenblobs gelesen.
README und CONTRIBUTING vollständig abgerufen und gegen ihre Git-Blob-SHAs
geprüft; keine EMPD-Programmlogik ausgeführt.

§2 von Z1 verweist für Struktur/Metadaten unmittelbar auf Davis et al. 2013,
DOI [10.1007/s00334-012-0388-5](https://link.springer.com/article/10.1007/s00334-012-0388-5).
Die Verlagsseite bietet Abstract und Zugangsvorschau, keinen gelesenen
Methodenvolltext. Eine weitere begrenzte Suchrunde nach diesem historischen
Definitionstext und den Zeitfeldern führte zur
[Autorenkopie der Davis-Gruppe](https://wp.unil.ch/davisgroup/files/2018/08/Davis_et_al_2013.pdf).
Der erste Webabruf lieferte Anfangsseiten; die gezielten Folgeabrufe nach
Alter/Moos/Sammlung sowie auf den Methodenteil scheiterten mit HTTP 404.
Deshalb **keine Prüfung der relevanten 2013-Methodenpassagen behauptet**.
Der erste PDF-Abruf von Z1 hatte ein Timeout; der zweite lieferte die gelesene
Tabelle 1. Keine Ersatzbehauptung aus Suchtreffern oder nicht lesbaren Bildern.
Die Zugriffslücke beweist weder fehlende öffentliche Existenz noch, dass der
historische Text die konkrete Probenlücke schließen würde. Keine weitere
Literatur-/Dokumentationsschleife und kein Nutzerdatei-Blocker.

### Getrennte Abnahme und Stop

| Anforderung | Bewertung für den beauftragten Umfang | Konkreter weiterhin fehlender Nachweis |
| --- | --- | --- |
| **R04 – Probenzeit und Unsicherheit** | **Teilweise belegt.** Allgemeine Alterskonvention und kategoriale Unsicherheitsrolle sind mit Z1/Z2 dokumentiert; Z3 beschreibt einen möglichen Herkunftsweg. | Archivstand + eindeutige Proben-/Originalkennung + tatsächlich beobachtete/datierte Zeit und Methode + Unsicherheit/Lücken + belegter Zuordnungsschritt zu genau dieser PANGAEA-Probe. Weder Publikationsjahr noch Code-/Felddefinition ersetzt diesen Nachweis. Die drei bekannten unbekannten Zeugenalter bleiben unbekannt. |
| **R05 – Sammlung, Ablagerung und Integration** | **Teilweise belegt.** Z1/Z2 unterscheiden Erfassungsrollen und liefern eine begrenzte methodische Protokollregel. | Nachweislich zugeordnete Proben-/Gruppenmethode mit Verhältnis von Sammlungsdatum zu repräsentierter Ablagerungszeit, Integrationsspanne und gegebenenfalls Mehrfachsammlung/Mischung samt Unsicherheit. Für konkrete Archivproben in den gelesenen Dokumenten **unbelegt**. |

**Stop erreicht:** Öffentliche allgemeine Definitionen wurden bewertet; die
fehlende tatsächliche Proben-/Methodenbindung lässt sich daraus nicht ableiten.
Eine reale Proben-/Dateiprüfung, zusätzliche konkret zugeordnete Originale oder
Kontakt wären gesonderte Arbeit. Hier kein Nachladen realer Daten, keine
Datierung aus dem Namen „modern“, keine Auswahl geeigneter Proben und keine
fachliche Schließung der Lücke. Dokumentarisch abgeschlossen; reale
Zeit-/Integrationsverbindung weiter blockiert. R01–R03 unverändert, R06 und
übrige Gates offen; keine A4-Umsetzung/A5/A6.

Nur dieser Abschnitt ergänzt, HANDOFF/Status kurz gepflegt. Übrige Fachdateien
unverändert; Quellen-/Link-/Diff-/Git-Blobprüfung, keine manuellen Projekt- oder
Modelltests, Rechnungen, Implementierung, Unteragenten, Nachrichten oder Merges.
Genau ein nächster **dokumentarischer Stop-/Eingangsplan** ist in
[HANDOFF](./HANDOFF.md) vorbereitet; er führt keine weitere Suche oder reale
Prüfung aus und erteilt keine R06-/Gatefreigabe.

## Konkrete Zeitprüfung der drei Zeugen, 01.11.2026

**Aktueller Nutzerauftrag:** notwendige Metadaten-/Originalquellenprüfung
zulassen und den konkreten Drei-Zeugen-Plan ausführen. Der oben noch
vorbereitete reine Stop-/Eingangsplan ist damit ersetzt. Die Umfangsverbote
der früheren Pakete bleiben historische Angaben; die aktuelle
[AGENTS-Regel](../../AGENTS.md#eigenständige-internetrecherche-als-standard)
erlaubt nötige, begrenzte schreibgeschützte Provenienzprüfungen ohne neues OK.
Basis ist PR #66, geprüfter Head `a9177cf84b2cc3d0b1e1983c7315880d93e5c5e6`;
offen/ungemergt und mergebar, vier Head-Checks erfolgreich.

**Ausgeführt:** genau `AlbaSanchez_a1`, `Champvillair_a18` und `Barboni_a1000`
aus dem bestehenden Readiness-Bericht, ohne neue Probenauswahl. Original-
Metadaten samt Notizen und direkter Publikationsrelation frisch gelesen,
dieselben Zeugen am festen Herausgeber-Release verglichen und zugeordnete
öffentliche Quellen gezielt gesucht. Pollenmesswerte und Klimawerte wurden
in diesem Paket nicht gelesen. Alle drei tatsächlichen Altersangaben bleiben
unbekannt. Das Ergebnis ist ein gesichertes, probenbezogenes Teilergebnis;
die reale Zeit-/Integrationsfrage ist weiterhin offen.

### Quellen, Zugriff und nachprüfbare Identität

Das [Evidenzprotokoll](./witness_time_check_2026_11_01/evidence.json) sichert
die exakten SQL-Abfragen/Parameter, Rohmetadaten der drei Zeugen, drei
Release-Auszüge mit Zeilennummern, Quellen-/Zugriffsstatus und Einzelergebnisse.
Wörtliches `"NULL"`, leeres Feld und SQLite-NULL werden nicht still gleichgesetzt.
Die unten genannten unbekannten Alter/Altersklassen sind in W1 Text `"NULL"`;
W2 enthält dafür leere Felder. Keine Datenergänzung oder Originalkorrektur.

| ID | Quelle / fester Stand / genaue Stelle | Gelesener Befund und Grenze |
| --- | --- | --- |
| W1 | Vorhandenes [PANGAEA.909130-SQLite-Original](https://store.pangaea.de/Publications/ChevalierM-etal_2019/EMPD2.sqlite3), 21.327.872 Bytes; SHA-256 vor/nach `5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`. `metadata` nur die drei festen Schlüssel; `metapubli` → `publications` nur für sie. Abfragen in `evidence.json: queries`. | Mit `mode=ro&immutable=1` und `PRAGMA query_only=ON` geöffnet. Genau drei Metadatenzeilen; genau eine direkte Publikationsrelation für Champvillair, `publiid=22`, DOI 10.1177/0959683618810395. Kein direkter Publikationstreffer für AlbaSanchez/Barboni; das beweist keine Quellenlosigkeit. Notizen enthalten keine Datierungs-/Integrationsangabe. Keine Counts-, Klima- oder Arbeitertabellen gelesen. |
| W2 | EMPD2/EMPD-data v2.0, Commit `737b76eb269b071422edded6ac332654ff83c103`, [meta.tsv](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/meta.tsv), ausschließlich Zeilen 2, 265 und 2609. Vollständiger Metadatenblob `1dca11169396f2eb2f1960ccd600cbd07705a4bf`, 7.405.566 Bytes; gegen Git-Blob-SHA geprüft. | Zum Herausgebervergleich den vollständigen Metadatenblob temporär heruntergeladen, nur dieselben drei Zeilen in den benötigten Feldern inhaltlich ausgewertet/gesichert; keine Bestandsstatistik oder weiteren Proben bewertet. `SampleName` + `OriginalSampleName` + `SiteName` stimmen für die drei Zeugen mit W1 überein. Alters-/Unsicherheitsfelder leer, gleiche Champvillair-Publikation; bei den anderen je fünf Publikations-/DOI-Felder leer. Diese konkrete Metadatenverbindung beweist keine Identität vollständiger SQLite-/TSV-Exporte. |
| W3 | Furlanetto et al., [DOI 10.1177/0959683618810395](https://journals.sagepub.com/doi/10.1177/0959683618810395), online 19.11.2018, The Holocene 29(2), 2019, S.271–286. Verlagsseite: Abstract, Data availability statement, Supplementary Material. | Abstract beschreibt Moos-Oberflächenproben in alpinen Höhentransekten; die Daten sollen in EMPD Stage 2 archiviert werden. W1/W2 verknüpfen die konkrete Probe bibliographisch mit dieser Arbeit. Die gelesenen Abschnitte liefern weder Sammlungsdatum noch Datierung/Unsicherheit oder Integrationsintervall für `TRS 18`. Hauptmethoden hinter Zugangsschranke, nicht gelesen. 2018/2019 sind Publikationsstände, keine Probenjahre. |
| W4 | Institutioneller [FLORE-Datensatz 2158/1408933](https://flore.unifi.it/handle/2158/1408933), bibliographischer Datensatz und „File in questo prodotto“. | Derselbe DOI; Verlags-PDF als „Accesso chiuso“ geführt. Keine Kopie angefordert und kein Autorenkontakt. Der Datensatz liefert keinen zusätzlichen gelesenen Methodenvolltext. |
| W5 | Unmittelbar auf W3 verlinktes [Supplement-DOCX](https://journals.sagepub.com/doi/suppl/10.1177/0959683618810395/suppl_file/furlanetto_et_al._revised_supplementary_material.docx). | Webwerkzeug: nicht unterstützter DOCX-Inhaltstyp. Executor: HTTP 403; Diagnose präzisiert `Tunnel connection failed: 403 Forbidden` beim Proxy-CONNECT. `journals.sagepub.com` steht nicht in der geltenden Executor-Hostliste. Deshalb Inhalt nicht gelesen; **technische Zugriffslücke, kein Nichtbeleg im Supplement**. Repo-Regeln ändern diese Netzwerkgrenze nicht; keine Umgehung. |
| W6 | Gefundene Furlanetto-Dissertation 2019, [Bicocca 10281/241319](https://www.boa.unimib.it/handle/10281/241319), sekundärer [Pflichtabgabe-Datensatz](https://tesidottorato.depositolegale.it/handle/20.500.14242/77337). | Suchtreffer bezeichnet eine öffentliche Dissertation; beide Datensatzabrufe scheitern im Webwerkzeug mit HTTP 403. Auch der aus bekanntem Dateinamen gebildete PDF-Kandidat war nicht abrufbar und ist kein verifizierter Dateilink. Kein Volltext gelesen, keine Zuordnung einer Dissertationstabelle oder Methode zu `TRS 18` behauptet. |

Die neue Metadatenprüfung bestätigt die schon berichteten unbekannten
Zeugenalter jetzt direkt am unveränderten Original. W2 schließt für genau
diese drei Kennungen eine dokumentierte Metadatenbrücke, keine historische
Programm-/Raster-/Exportkette. R01–R03 wurden nicht erneut recherchiert.
Die Vollbytes der neu bezogenen Metadaten liegen nur temporär im Cache;
im Repository sind die begrenzten Auszüge und Abfragen gesichert, kein
dauerhaftes Vollarchiv der Quelldateien behauptet.

### Ergebnis je Zeuge gegen R04/R05

| Archivprobe und Originalkennung | Sammlungs-/Probenzeit, Methode und Unsicherheit (R04) | Ablagerung, Mehrfachsammlung und Integration (R05) | Verbleibende konkrete Lücke |
| --- | --- | --- | --- |
| **AlbaSanchez_a1 → `1 béticas`**, Sierra de las Nieves Natural Park, Málaga, Spain; Identitätsfelder in W1/W2 gleich. | **Unbelegt für diese Probe:** Alter, Altersklasse, Sammlungsdatum und Datierungsmethode fehlen in den geprüften Metadaten; keine direkte Publikationsrelation. | **Teilweise belegt:** Material `moss`, Kontext `closed forest`, Vegetation Pinsapo-Tannenwald dokumentiert. Entnahmemethode und repräsentierte Ablagerungs-/Integrationsspanne unbekannt. | Ein öffentliches Original muss genau `1 béticas`/diesen EMPD-Schlüssel identifizieren und Zeit, Methode samt Unsicherheit sowie Integrationsbezug liefern. Eine ähnliche Studie aus demselben Naturpark reicht nicht. |
| **Champvillair_a18 → `TRS 18`**, La Thuile Valley, Italy; W1/W2 gleich, direkte Relation zu W3. | **Unbelegt für diese Probe:** Metadatenalter/Altersklasse unbekannt. Bibliographische Bindung belegt; gelesene W3-Abschnitte datieren `TRS 18` nicht. Methodenvolltext/Supplement in W4–W6 nicht gelesen. | **Teilweise belegt:** Material `moss`, Methode `hand picking - multiple samples`. Zahl, räumliche Zusammensetzung, Gleichzeitigkeit der Teilentnahmen und repräsentierte Zeitspanne damit nicht belegt. | Lesbare Originalmethoden/Erhebungsmetadaten mit explizitem `TRS 18`- oder nachweislichem Gruppengeltungsbezug. Anschließend Sammlung, Ablagerungszeit und Integration getrennt beurteilen; kein Alter aus Publikationsjahr ergänzen. |
| **Barboni_a1000 → `MAROC124`**, `Moroc`, Morocco; Identitätsfelder in W1/W2 gleich. | **Unbelegt für diese Probe:** Alter, Altersklasse, Datierungs-/Sammlungsangaben fehlen. Keine direkte Publikationsrelation; Notiz leer. | **Unbelegt:** Material, Kontext, Entnahmemethode und Integration fehlen. Die Ortswarnung bleibt unverändert erhalten. | Zuerst ein Original mit eindeutiger `MAROC124`-Herkunfts-/Probenbindung und den fehlenden Zeit-/Methodenangaben. Unzuverlässige Lageangaben werden weder repariert noch zur Quellenidentität umgedeutet. |

Die probenbezogenen R04-Nichtbelege sind von der bereits teilweise belegten
allgemeinen R04-Definition in Z1/Z2 zu unterscheiden. R05 ist für AlbaSanchez
und Champvillair nur im oben bezeichneten Material-/Entnahmeumfang teilweise
belegt, für Barboni unbelegt. **Kein Zeuge hat jetzt eine belegte vollständige
Zeit-/Integrationszuordnung.** Fehlende Integration bleibt unbekannt und
ist keine Nullunsicherheit; keine geeignete Trainingsprobe daraus ausgewählt.

### Begrenzter Abschluss und Anschluss

Die gezielten Suchfragen stehen im Evidenzprotokoll: Originalkennungen,
Autor-/Ortskombinationen und die direkt verknüpfte DOI sowie öffentliche
Autoren-/Institutionsfassungen und das Verlags-Supplement. Jahreszahlen in
Suchanfragen waren Suchhilfen, keine Datierungshypothesen. Für AlbaSanchez
und Barboni wurde keine gelesene öffentliche Originalmethode mit eindeutiger
Zeugenbindung ermittelt; Viewer-Suchauszüge, verwandte/später erschienene
Studien und leere Trefferlisten wurden nicht als Probenbeleg übernommen.
Für Champvillair ist die Quellenidentität konkret, die relevante Lesung
an den oben genannten Zugriffsgrenzen offen. Keine allgemeine Literatur-
oder erneute Definitionssuche, keine Behauptung weltweiter Nichtexistenz.

Der beauftragte Prüfplan ist bis zu diesen konkreten Nachweis-/Zugriffsgrenzen
ausgeführt. Die frühere Repo-Sperre ist behoben; R04/R05 sind fachlich nicht
geschlossen. Als genau ein sachlicher Folgeauftrag ist die **gezielte
Beschaffung/Lesung der öffentlich verlinkten Furlanetto-Originalmethodik für
`TRS 18`** vorbereitet, mit expliziter technischer Startbedingung, ohne neue
Metadateninventur oder zusätzliches reines Planungspapier. Kein Nutzer muss
ein Dokument benennen oder bereitstellen; ein erlaubter lesbarer Zugriff
muss tatsächlich verfügbar sein. R06/übrige Gates und R01–R03 bleiben offen.

Original-Hash vor/nach und Release-Blob geprüft; Evidenz-JSON, Identitätsfelder,
Quellen-/Linkbezüge und zulässiger Diffumfang geprüft. Bestehender Reviewtext,
Statushistorie und übrige Fachdateien erhalten. Keine Datenreparatur,
Modell-/Kohortenwahl, Projektimplementierung, manuellen Projekt-/Modelltests,
wissenschaftlichen Rechnungen/Fits/Simulationen, A4-Umsetzung/A5/A6,
Gatefreigabe, Unteragenten, Nachrichten/Kontakte oder Merges.


## Furlanetto-Zugriffsprüfung für TRS 18, 01.11.2026

**Auftrag und Basis:** nur Originalmethoden/Supplement zu DOI
10.1177/0959683618810395 für Champvillair_a18 / TRS 18. PR #67 offen,
Head `947ae20621d929b51b13250b6a9665ff16778385`, vier erfolgreiche
Head-Checks; nach `42d365f` ausschließlich Sicherungsverweise. Fünf
vorgefundene Worktrees sauber erhalten; eigener Worktree/Branch
`codex/netzel-furlanetto-access-review` ab diesem Head. Leseplan aus HANDOFF,
aktueller Drei-Zeugen-Abschnitt, W3–W6/Champvillair und R04/R05 samt
Herkunftsregeln gelesen; keine neue Inventur.

**Tatsächlich geänderter Zugriff:** `journals.sagepub.com` steht jetzt in
`/etc/codex/network-policy.json` unter `egress_rules`. Laufzeitstatus meldet
aktuelle Beobachtungen, verbunden, eingeschränkte Policy `enforced`.
Deshalb genau ein neuer Abruf des bekannten W5-Supplements über den
unveränderten Proxy mit TLS-Prüfung; keine Wiederholung unter gleichen
Bedingungen. [Neues Evidenzprotokoll](./furlanetto_access_review_2026_11_01/evidence.json)
sichert URL, Abrufbefehl, ausgewählte Antwortfelder und Antwort-Hash.

**Konkrete neue Zugriffslücke:** Proxy-CONNECT nun HTTP 200, danach HTTP/2
403, `Content-Type: text/html; charset=UTF-8`, `cf-mitigated: challenge`,
Titel „Just a moment...“, 6.443 Antwortbytes. Keine Weiterleitung/Location;
kein DOCX erhalten. Die Hostfreigabe allein schafft keinen lesbaren Zugriff.
Keine Challenge-Lösung, URL-Varianten, Proxyumgehung oder Wiederholungsabrufe.
An dieser Stelle gemäß Auftrag gestoppt. Hauptmethoden nicht neu gelesen;
die historische Zugangsschranke aus W3/W4 bleibt. W6 bietet hier keinen
neu verfügbaren verifizierten Autoren-Volltextzugriff; bekannte Hosts nicht
in der geltenden Quellenfreigabe, kein Web-Lesewerkzeug verfügbar.
Kein Nutzerdokument vorausgesetzt und keine Datei angefordert.

| Getrennter Nachweis | Ergebnis und genaue Grenze |
| --- | --- |
| TRS-18-/Gruppenbindung | W1/W2 verbinden Archivkennung, Originalkennung und DOI. Keine gelesene Originalstelle belegt die Geltung einer Methodengruppe für TRS 18. |
| Sammlung | Nur vorhandene W1/W2-Angaben `moss` und `hand picking - multiple samples`; Sammlungsdatum, Anzahl, Zusammensetzung und Gleichzeitigkeit offen. |
| Ablagerung | Repräsentierte Ablagerungszeit weiterhin nicht belegt. |
| Integration | Kein gelesenes Originalintervall; Mehrfachentnahme allein belegt keine Dauer. |
| Unsicherheit | Nicht bestimmt; keine Nullunsicherheit und kein Alter aus Publikationsjahr. |

**Abnahme/Stop:** Zugriffsversuch und Diagnose dokumentarisch abgeschlossen;
Originalmethodenprüfung **blockiert**, keine inhaltliche Nichtbelegprüfung
des unlesbaren Supplements behauptet. Keine neue Methoden-/Versions-/Seiten-
oder Zitatangabe erfunden. R04/R05 unverändert offen im bisherigen Teilumfang,
R06/übrige Gates nicht freigegeben. Alte Quellen und Berichte einschließlich
W1–W6 bleiben unverändert; neuer Befund ergänzt die historische Proxy-Diagnose.
Keine Werteanalyse, neue Probe/Kohorte, Reparatur, Implementierung, manuellen
Tests, wissenschaftlichen Rechnungen, R01–R03-Neusuche, A4-Umsetzung/A5/A6,
Unteragenten, Kontakte/Nachrichten oder Merges. JSON, Referenzen und Diffumfang
geprüft; keine dauerhafte Archivierung der temporären Fehlerantwort behauptet.


## Öffentlicher Furlanetto-Originalzugang nach Nutzer-OK, 01.11.2026

**Basis und Autorisierung:** PR #69, Head
`47bbcc2c2770c28040086e72286e45e37e83c83f`, offen/ungemergt; vier
Head-Checks zu Beginn noch laufend, kein Erfolg behauptet. Nutzer hat mit
„OK, Furlanetto fortsetzen“ genau den vorbereiteten Auftrag autorisiert.
Sechs bestehende Worktrees sauber erhalten; eigener Worktree/Branch
`codex/netzel-furlanetto-public-original` ab #69. AGENTS/HANDOFF/Status,
vorhandene Furlanetto-Evidenz, W3–W6/Champvillair sowie R04/R05 und
Herkunftsregeln berücksichtigt. Nur vorgeschaltete begrenzte Zugangssuche;
keine neue Probe, keine Originaldateninventur.

**Neue tatsächlich verfügbare Fähigkeit:** Websuche/-Lesewerkzeug jetzt
aufrufbar; die historische Angabe „kein Web-Lesewerkzeug“ bleibt als damaliger
Befund erhalten. Das beweist keine neue Lesbarkeit einer Website. Zwei
begrenzte Suchrunden mit sechs genauen Anfragen sind in der
[Evidenz](./furlanetto_public_original_2026_11_01/evidence.json) dokumentiert.
Suchauszüge bestätigen nur bekannte Quellenkandidaten, keine gelesenen Methoden.
Kein neuer Sage-Abruf; dessen letzte belegte Challenge-Diagnose unverändert.

**Konkrete Zugriffslücke:** W6, institutioneller
[BOA-Datensatz 10281/241319](https://www.boa.unimib.it/handle/10281/241319),
Giulia Furlanetto, Dissertation 2019. Der Suchindex führt
`phd_unimib_809217.pdf` als „Accesso Aperto“ (12.37 MB). Genau ein direktes
Öffnen des Datensatzes im jetzt verfügbaren Webwerkzeug ergibt HTTP 403.
Keine Seiten-/Dateilinks gelesen, kein verifizierter PDF-Link, keine
Originalbytes erhalten. Kein Volltext oder Gleichstand mit der publizierten
DOI-/Supplementfassung behauptet; kein Downloadpfad aus Dateinamen gebildet.
FLORE bleibt im aktuellen Suchauszug als geschlossene Verlagsfassung geführt;
kein weiterer Direktabruf. Andere Werke/Zitate sind kein Ersatzoriginal.

| Getrennter Nachweis | Ergebnis nach dieser Zugangssuche |
| --- | --- |
| TRS-18-/Gruppenbindung | Vorhandene W1/W2-Identitäts-/DOI-Brücke erhalten; keine neu gelesene Originalstelle zur Methodengruppe. |
| Sammlung | Nur bisheriges Material-/Entnahmelabel; Datum, Anzahl, Zusammensetzung und Gleichzeitigkeit weiter offen. |
| Ablagerung | Repräsentierte Zeit unbekannt. |
| Integration | Dauer unbekannt; Mehrfachentnahme belegt kein Intervall. |
| Unsicherheit | Unbekannt, keine Nullunsicherheit; kein Probenalter aus 2019 abgeleitet. |

**Abnahme/Stop:** Zugangssuche an konkreter HTTP-403-Lücke beendet;
Startbedingung für die Methodenlesung nicht erfüllt. R04/R05 fachlich
unverändert, ungelesener Inhalt nicht als Nichtbeleg bewertet. Keine
Originalzitate/Seiten oder Dokumentversion erfunden. Alte Evidenz und
Reviewtexte unverändert, nur dieser Abschnitt ergänzt. Keine Umgehung oder
weiteren Abrufvarianten. JSON/Referenzen/Links/Diff geprüft, keine manuellen
Projekt-/Modelltests, Rechnungen, R01–R03-Neusuche, R06-/Gatefreigabe,
A4-Umsetzung/A5/A6, Unteragenten, Kontakte oder Merges.

**Nutzeraktion und Anschluss:** Das OK bleibt gültig. Optional kann der Nutzer
über die unterstützte Cloud-Konfiguration `www.boa.unimib.it` und
`boa.unimib.it` erlauben; beide fehlen in der Executor-Hostliste. Das würde
einen ersten regulären Executor-Aufruf des institutionellen Datensatzes
ermöglichen, garantiert jedoch keinen lesbaren Inhalt und behebt keine
Verlagssperre. Keine Konfigurationsänderung hier ausgeführt. Keine Nutzerdatei
oder Kontakte erforderlich. Kein verifizierter Nutzerschritt zur Behebung der
Sage-Challenge. Genau ein Anschluss bleibt die bereits autorisierte
Originalmethodenlesung bei tatsächlich geändertem erlaubtem Originalzugang;
kein erneutes OK für denselben Auftrag und keine identischen Abrufe.


## BOA-Executorzugriff nach bestätigter Hostfreigabe, 01.11.2026

**Basis:** [PR #70](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/70),
Head `183ac3a8473fcc3f4ba4d0cbe691a165f880fdf2`, offen/ungemergt,
vier erfolgreiche Head-Checks in den Läufen 41291662668 und 41291667788.
Inhaltscommit `346558c1290d0a0ca0506397ce5331ad3230fdfe` erhalten.
Fünf vorgefundene Worktrees sauber erhalten; eigener Worktree/Branch
`codex/netzel-furlanetto-boa-review` ab diesem Head. Bestehendes Nutzer-OK
gilt weiter. AGENTS, HANDOFF, neuester Status, aktueller Review/Evidenz und
gezielt W3–W6/Champvillair sowie R04/R05/Herkunftsregeln berücksichtigt.

**Neue wirksame Freigabe:** Laufzeitstatus meldet `observations_current=true`,
Policy `restricted/enforced`, beide offiziellen BOA-Hosts erlaubt. Auch die
Executor-Policy Version 1 enthält `www.boa.unimib.it` und `boa.unimib.it`.
Obwohl die Revisionsnummer erneut 1 lautet, ist die Hostliste gegenüber dem
vorigen Befund verändert. Keine Netzkonfiguration durch Codex geändert.
Die bestätigte Freigabe erlaubt erstmals den regulären Executor-Aufruf des
bekannten [institutionellen Handles](https://www.boa.unimib.it/handle/10281/241319).

**Genau ein Abruf / konkrete Zugriffslücke:** Über den unveränderten Proxy
mit TLS-Prüfung liefert CONNECT HTTP 200, danach HTTP/2 403,
`Content-Type: text/html; charset=UTF-8`, `cf-mitigated: challenge`, Titel
„Just a moment...“, 6.166 HTML-Antwortbytes. Keine Weiterleitung/Location.
[Evidenz](./furlanetto_boa_review_2026_11_01/evidence.json) sichert Befehl,
ausgewählte Antwortfelder und SHA-256 der Fehlerantwort. Kein lesbarer
Datensatz, kein tatsächlich verlinkter öffentlicher Dateilink und keine
Originalbytes. Kein PDF-Pfad aus dem historischen Dateinamen konstruiert;
keine Hostvariante, Challenge-Lösung oder weitere Quelle verfolgt. Sage
nicht erneut abgerufen; dessen letzte belegte Challenge bleibt historisch.

| Getrennter Nachweis | Ergebnis / Beleggrenze |
| --- | --- |
| TRS-18-/Gruppenbindung | W1/W2-Identitäts-/DOI-Brücke erhalten; keine neue gelesene Originalstelle zur Methodengruppe. |
| Sammlung | Nur bisheriges Material-/Entnahmelabel; Datum, Anzahl, Zusammensetzung und Gleichzeitigkeit offen. |
| Ablagerung | Repräsentierte Zeit weiterhin unbekannt. |
| Integration | Dauer weiterhin unbekannt; Mehrfachentnahme belegt kein Intervall. |
| Unsicherheit | Unbekannt, keine Nullunsicherheit; kein Alter aus Publikationsjahr. |

**Stop und Grenze:** Reguläre BOA-Zugriffsprüfung abgeschlossen;
fachliche Originalmethodenlesung weiterhin **blockiert**. Kein Originalzitat,
keine Seite oder Dokumentversion verifiziert; Dissertation und Artikel/
Supplement nicht gleichgesetzt. Ungelesener Inhalt ist kein Nichtbeleg.
R04/R05 bleiben im bisherigen Teilumfang offen; keine R06-/Gatefreigabe.
Alte Review-/Statushistorie, Evidenz und Originale erhalten. Antwortbytes
nur temporär unter `/tmp`, keine dauerhafte Archivierung behauptet.
Nur Review/Evidenz und HANDOFF/Status geändert. JSON/Hashes/Links/Diffumfang
geprüft; keine manuellen Projekt-/Modelltests, wissenschaftlichen Rechnungen,
neuen Proben/Kohorten, Werteanalysen, Reparaturen, Implementierungen,
R01–R03-Neusuche, A4-Umsetzung/A5/A6, Unteragenten, Kontakte oder Merges.

**Nutzeraktion / genau ein Anschluss:** Kein weiteres OK und keine erneute
BOA-Hostfreigabe nötig; beide Hosts sind erlaubt. Keine Nutzerdatei/Kontakte
erforderlich. Aktuell kein verifizierter Nutzerschritt zur Behebung der
BOA- oder Sage-Challenge. Optional kann ein bereits bekannter offizieller
öffentlicher Original-Link genannt werden. Codex übernimmt dessen Prüfung
bzw. nach belegter relevanter Zugriffsänderung die Fortsetzung desselben
Auftrags; ohne neuen Eingang keine identischen Abrufe oder Diagnosepakete.
