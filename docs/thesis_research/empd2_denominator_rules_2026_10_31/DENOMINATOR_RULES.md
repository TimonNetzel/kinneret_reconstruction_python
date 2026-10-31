# EMPD2: Nenner- und Sparse-Regeln, belegter Teilstand

31.10.2026. Autorisierter Quellenauftrag aus PR #42 bei
`e1e8ebe32f17487c4c91d2044db845112a8365bb`, hier im Chat ausgeführt.
Worktree `/workspace/netzel-empd2-rules`, Branch
`codex/netzel-empd2-denominator-rules`. Neun vorherige Worktrees erhalten;
main `7f1826c`, #31–#42 offen, #42-CI grün. Kein Merge/Autorenkontakt.
[Standprüfung](./git_snapshot.json).

**Ergebnis:** Der bereitgestellte Prozentnenner ist für die drei bekannten
Zeugen datenbezogen nachvollzogen. Eine allgemeine archiveigene Nullregel und
der ursprünglich im Labor gemessene Gesamtzählumfang sind weiterhin nicht belegt.
Es gibt keine Count-Likelihood-, Trainings- oder Pilotfreigabe.

## Zwei eng begrenzte Quellenobjekte

1. **Davis et al. (2020), Original-HTML**, DOI
   [10.5194/essd-12-2423-2020](https://essd.copernicus.org/articles/12/2423/2020/),
   §2, §2.2, §4/§6 und Literaturverweis Sommer et al. (2020). §2 beschreibt
   vollständige Rohzähl-Assemblagen statt Prozent-/Teilspektren; §2.2 erklärt
   Verarbeitung und Namenszuordnung. §2 belegt WorldClim2, 30″ und nächsten
   Gitterpunkt, nennt dort aber keine Klimanormalperiode. Eine explizite
   Zuordnung fehlender Sample×Typ-Paare zu Null sowie eine Formel für die drei
   Gruppenfelder wurde in diesen gelesenen Abschnitten nicht gefunden.
   Das ist ein begrenzter Nichtbeleg, keine Behauptung über alle Materialien.
2. **Das offiziell zugeordnete Release EMPD2/EMPD-data v2.0**, in der
   Publikationsbibliographie mit DOI 10.5281/zenodo.4066498 identifiziert.
   Annotierter Tag `f543a1bf60fefb705e46db400f59ebdac96edaf7`, datiert
   05.10.2020, zeigt auf Commit
   [`737b76eb269b071422edded6ac332654ff83c103`](https://github.com/EMPD2/EMPD-data/tree/737b76eb269b071422edded6ac332654ff83c103).
   Zusätzlich verknüpft der Artikel den offiziellen Viewer, der genau dieses
   Repository nennt. Keine heutige master-Version als historischer Beleg benutzt.

Die Zenodo-Landingpage war per Webwerkzeug nicht zugänglich; deren Inhalt/Archiv
wird nicht als gelesen behauptet. Der Tag und alle zwölf gelesenen Dateien wurden
über die GitHub-Verbindung am festen Commit geprüft. Code und Daten des Releases
stehen gemäß README unter CC BY 4.0; Attribution Sommer, Chevalier und Davis (2020).
Originalcode wurde **nur gelesen**, weder importiert noch ausgeführt.
Keine Archiv-/Viewerformulare ausgefüllt, keine Nachricht versendet.

## Was tatsächlich angewandt erscheint: gelieferte Prozentwerte

Die Dateien [AlbaSanchez_a1.tsv](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/samples/AlbaSanchez_a1.tsv),
[Champvillair_a18.tsv](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/samples/Champvillair_a18.tsv)
und [Barboni_a1000.tsv](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/samples/Barboni_a1000.tsv)
enthalten neben Original-/akzeptiertem Namen und count eine percentage-Spalte.
Ihre 20/49/39 Messwert-Tupel stimmen an den Schlüsseln
`(original_varname,acc_varname,groupid)` und Wert **vollständig** mit den gleichen
drei Proben des gesperrten SQLite-Originals überein. Keine weiteren Proben gelesen.

Für diese Prüfung ist definiert, ohne einen neuen Nenner freizugeben:

```text
S_flag(s) = Summe der gespeicherten Werte der Probe,
            deren ursprüngliche Gruppe used_in_sum=True trägt
P_check(s,j) = 100 * gespeicherter Wert(s,j) / S_flag(s)
```

| Zeuge | S_flag | Geprüfte bereitgestellte Prozente | Leere Prozentfelder | Größte absolute Differenz in Prozentpunkten |
| --- | ---: | ---: | ---: | ---: |
| AlbaSanchez_a1 | 311 | 20 | 0 | 4,824×10⁻⁷ |
| Champvillair_a18 | 731 | 39 | 10 | 2,326×10⁻⁷ |
| Barboni_a1000 | 840 | 39 | 0 | 4,762×10⁻⁷ |

Alle **98** gelieferten Zahlen stimmen mit P_check innerhalb einer halben Einheit
der jeweils letzten gedruckten Dezimalstelle überein. Die Toleranz folgt der
Quellpräzision pro Wert, nicht einem gewählten Rekonstruktionsfehler. Alle zehn
leeren Prozentfelder betreffen Gruppen mit percent_values=False; alle 98 gefüllten
Felder solche mit True. AQUA bei Barboni liegt außerhalb der Nennergruppen,
besitzt aber ein Prozentfeld. Beispiel: 3/840×100 ist kompatibel mit 0,35714286.

Damit ist die **Kompatibilität der gelieferten Prozentdarstellung mit S_flag**
für genau diese Zeugen stark belegt. Das beweist weder einen eindeutigen globalen
Berechnungsalgorithmus noch N_original für jede Probe. Die Prozentwerte wurden
nicht zurück in Pseudo-Counts umgerechnet. Die Messwerte blieben unverändert.
[Genaue Vergleiche und Quellfingerprints](./source_comparison.json).

## Gruppenfelder: Herkunft belegt, Hierarchie-Anwendung offen

Im [Release-GroupID.tsv](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/tables/GroupID.tsv)
stehen 36 Gruppen. Im SQLite-Original stehen 40, zusätzlich MANG/PALM/SEED/SUCC.
Für alle 36 gemeinsamen IDs stimmen Eltern-ID und beide Flags überein.
Diese Differenz verhindert die Behauptung eines insgesamt identischen Releases.

[makeFixedTables.py, Zeilen 113–124](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/makeFixedTables.py#L113-L124)
überträgt diese Felder in die relationale Gruppentabelle;
[create_empd2.sql, 209–217](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/create_empd2.sql#L209-L217)
definiert sie. Das belegt die Feldherkunft, aber keinen Algorithmus zum rekursiven
Zusammenlegen von Eltern-/Kindgruppen oder zur Prozentberechnung.
Aus higher_groupid allein keine doppelt gezählten Aggregatcounts erzeugen.

Der große postgres/EMPD2.sql-Dump war über die begrenzte Connector-Anfrage nur mit
Blobmetadaten, ohne lesbaren Inhalt erhältlich. Er wurde nicht als ausgewertet
ausgegeben; keine Aussage, dass dort oder anderswo keine weitere Logik existiere.
Keine zusätzliche Codebasis als drittes Quellenobjekt nachgeladen.

## Sparse-Zeilen: sichtbarer Importpfad ist kein universeller Nullbeleg

Im [historischen Importskript, Zeilen 435–497](https://github.com/EMPD2/EMPD-data/blob/737b76eb269b071422edded6ac332654ff83c103/postgres/scripts/import_into_empd2.py#L435-L497)
gilt für **neu angelegte** Proben:

- 435–439: vorhandene Proben werden übersprungen, neue Sample-TSVs eingelesen.
- 463–469: nur Eingangswert >0 wird übernommen, danach gerundet und eingefügt.
  Ein positiver Wert <0,5 könnte dabei als 0 gespeichert werden; aus der
  Eingangsschranke nicht „alle gespeicherten Werte strikt positiv“ ableiten.
- 474–484: beim passenden Duplikatfehler wird der gerundete Wert addiert.
- 485–497: Fehler werden gesammelt und am Ende per Assertion geprüft.

Eingangsnull, fehlende/NaN-Eingänge und gar nicht vorhandene Taxonzeilen erhalten
auf diesem sichtbaren Neuanlagepfad keinen unterscheidbaren Messstatus.
Ein SQL-Nichttreffer ist daher ohne zusätzliche Provenienz nicht eindeutig.
Die dokumentierte vollständige Assemblage als Datenprotokoll hebt diese
implementierungs- und archivbezogene Unterscheidung nicht stillschweigend auf.

Zudem würde die Neuanlage des Volumenwerts 7,4 hier 7 erzeugen; das SQLite-Original
und das Release-TSV bewahren 7,4. Vorhandene Proben werden aber übersprungen:
Dies beweist **nicht**, dass das Skript niemals auf einer vorbefüllten Datenbank
gelaufen sein könnte. Es belegt die fehlende Kette von genau diesem Code zu genau
diesem Archivwert. Kein Nachbauen/Ausführen, Runden oder „Reparieren“ der Daten.
AlbaSanchez_a1/Originalcode 7 bleibt ein nicht gespeichertes Paar, kein neu
eingetragener Nullcount. Weder lokale botanische Absenz noch Messnull bestätigt.

## Drei Nennerrollen ausdrücklich auseinanderhalten

| Größe | Was sie hier bedeutet | Stand |
| --- | --- | --- |
| N_original | Ursprünglicher gemessener Zählumfang samt Erfassungs-/Ausschlussprozess | Für die Archivproben weiterhin nicht unabhängig belegt |
| S_flag / kompatibler Prozentnenner | Summe dokumentierter flag-1-Messwerte im geprüften Release | Für die drei Zeugen gegen bereitgestellte Prozentwerte geprüft; nicht pauschal N_original |
| n_ausgewählt | Summe einer später fachlich gewählten, disjunkten Beobachtungsteilmenge | Noch keine Teilmenge gewählt; bedingtes Modell benötigt belegten Auswahlprozess |

Eine Summe aller p_counts-Werte bleibt für Champvillair mit Volumen und anderen
Messgrößen vermischt. Die symbolischen Nennerrollen sind keine Erlaubnis, neue
Matrixfelder, Taxonpools oder Trainingskohorten zu erzeugen.

## Abschluss und verbleibende Grenze

Der begrenzte Quellenauftrag ist mit einem **gesicherten Teilergebnis** erledigt:
Originalmethodik gelesen, historisches offizielles Release identifiziert,
Prozentnenner-Kompatibilität belegt und Sparse-/Versionsgrenze präzisiert.
[RULE_EVIDENCE.json](./RULE_EVIDENCE.json) trennt Protokoll, Code, Datenvergleich
und nicht bewiesene Anwendung. [SOURCE_GAPS.md](./SOURCE_GAPS.md) beschreibt den
nächsten engen Archiv-Provenienzauftrag, wartet auf neues Nutzer-OK.

Keine ganze Matrix/Kohorte, Fits, Scores, Simulationen oder Tests. Nur Quellen-/
Metadatenprüfung an denselben drei Zeugen und Dokument-/Hash-/Diffkontrolle.
Die Quelldatenbank blieb vor/nach read-only-Zugriff bytegleich, alle bestehenden
Designs/Modelle/Resultate/Register/Importer/Flags/Fossilcounts/Gruppen, Baseline
und CLI/App unverändert. W96, reales T1/T0/T3/T4/Monticchio-P1 bleiben offen.
Kein unabhängiger menschlicher Review und keine automatische Nullfüllung oder
Trainingsfreigabe. [Reproduktion](./REPRODUCE.md), [Prüfung](./verification.json).
Quellen-/Dokumentationsprüfung bestanden: zwölf gepinnte Dateien, 108 Messwert-
Tupel und 98 Prozentwerte geprüft; 709 geschützte bestehende Git-Blobs
einschließlich Symlink unverändert, neun vorherige Worktrees sauber erhalten.
