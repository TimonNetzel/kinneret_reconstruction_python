# Entscheidungspunkt nach begrenztem Archivabschluss

**Quellensuche abgeschlossen; reale Counts-Bereitschaft weiterhin blockiert.**
Die zwei gezielten Quellen ordnen Originalformate zu und belegen einen
PostgreSQL-Dump. Sie liefern im gelesenen Umfang keine tatsächlich angewandte
Erzeugungskette zur gesperrten SQLite-Datei.
[Ergebnis](./ARCHIVE_LINEAGE.md), [Belege und Lesegrenzen](./LINEAGE_EVIDENCE.json).

| Offener Beleg | Konsequenz für genau dieses Archiv |
| --- | --- |
| Konkrete Daten-/Exportversion samt angewandter Konvertierung | Weder v2.0-Code noch pg_dump 11.2 als SQLite-Buildnachweis verwenden |
| Herkunft des erhaltenen Volumens 7,4 und der 40 Gruppen | Werte/Gruppen erhalten; keine Rundung oder Anpassung an 36 Releasegruppen |
| Erfassungsstatus und Nullunterdrückung bei fehlenden Sample×Typ-Paaren | Keine Nullfüllung, Messnull oder biologische Absenz ableiten |
| Ursprünglicher gemessener Umfang N_original | S_flag und n_ausgewählt nicht als Ersatz ausgeben |
| Tatsächlich angewandte Hierarchie-/Nennerlogik | Gruppenflags allein erlauben keine neue Aggregation oder Count-Likelihood |

**Konkrete Entscheidung:** Bleibt ein reales Counts-T1 bis zu archivgenauen
Belegen zurückgestellt, oder soll später ausdrücklich ein anderes, klar als
annahmenabhängig bezeichnetes Forschungsziel entworfen werden? Ein solches
Ziel müsste Beobachtungseinheit, Messstatus, Nennerrolle und Aussagegrenzen vorab
festlegen; es würde die fehlende historische Kette nicht nachträglich belegen.
Hier ist weder diese Zieländerung noch eine reale Nutzung beschlossen.
Bis zur Entscheidung gelten die bestehenden Sperren unverändert. Keine
Ersatzquellensuche oder Wiederholung der abgeschlossenen zwei Quellen.

## Genau ein Folgeauftrag – vorbereitet, wartet auf Nutzer-OK

**Eine dokumentarische Entscheidungsvorlage zum blockierten realen EMPD2-Counts-Einstieg erstellen.**

- **Ziel:** Die beiden obigen Handlungswege anhand der bereits gesicherten
  Belege hinsichtlich benötigter Annahmen, zulässiger Aussagen und verbleibender
  Sperren prüfbar gegenüberstellen. Keine Wahl oder Umsetzung vorwegnehmen.
- **Umfang:** genau eine neue `COUNTS_DECISION_BRIEF.md` plus Status/Übergabe.
  Keine weitere Quellenrecherche, neue Datenquelle, Datenauswahl, Matrix/Kohorte,
  Datenbereinigung, Code-/Test-/Modell-/Kernlauf oder Änderung vorhandener Designs,
  Verträge, Register, Importer, Flags, Counts, Gruppen, Baseline oder CLI/App.
- **Leseplan:** AGENTS, HANDOFF, PROJECT_STATUS; die drei Dateien dieses Pakets;
  anschließend nur `INPUT_CONTRACT.json` und `SOURCE_GAPS.md` des
  `empd2_counts_readiness_2026_10_31`-Pakets und Roadmap T1/T3.
  Bei einer konkreten Regelbehauptung `RULE_EVIDENCE.json` aus #43 nachschlagen.
- **Abschluss:** Für jeden Handlungsweg notwendige Nutzerentscheidung,
  minimale noch erforderliche Evidenz, erlaubte/nicht erlaubte Schlussfolgerung
  und eindeutige Stopbedingung festhalten; N_original/S_flag/n_ausgewählt trennen.
  Keine Null-/Nennerregel oder wissenschaftliche Freigabe beschließen.
- **Sicherung:** eigener Worktree/Branch/PR, Quellen-/Link-/JSON-/Diffprüfung,
  keine Unteragenten, Autorennachrichten oder Merges. W96 und reale
  T0/T1/T3/T4/Monticchio-P1 bleiben offen.

Ein OK autorisiert nur diese schriftliche Vorlage. Es autorisiert weder
das annahmenabhängige Forschungsziel noch seine Implementierung oder einen Lauf.
