# Original-Namensprüfung reproduzieren

Python >=3.11, nur Standardbibliothek (SQLite, CSV, JSON, Hashes), kein C++-Kern,
GIS-Paket oder Modellfit. Aus der Repository-Wurzel des Quellenpakets starten.

## Eingänge frisch beziehen und prüfen

Die URL-/Hashregistrierungen stehen in `data/manifest.json`; sie entsprechen
den unveränderten Sperren aus #34. Zielordner frei wählen; bestehende Dateien
nur nach gültigem Hash wiederverwenden, niemals blind überschreiben. Über den
vorgesehenen Netzwerkproxy mit unveränderter TLS-Prüfung beziehen:

```bash
set -e
mkdir -p /workspace/netzel-source-cache/pollen-originals-2026-10-31
if [ ! -e /workspace/netzel-source-cache/pollen-originals-2026-10-31/EMPD2.sqlite3 ]; then
  curl --fail --location 'https://store.pangaea.de/Publications/ChevalierM-etal_2019/EMPD2.sqlite3' \
    --output /workspace/netzel-source-cache/pollen-originals-2026-10-31/EMPD2.sqlite3
fi
if [ ! -e /workspace/netzel-source-cache/pollen-originals-2026-10-31/monticchio_739452.tab ]; then
  curl --fail --location 'https://doi.pangaea.de/10.1594/PANGAEA.739452?format=textfile' \
    --output /workspace/netzel-source-cache/pollen-originals-2026-10-31/monticchio_739452.tab
fi
```

Die Originalbytes wurden am 31.10.2026 mit TLS-/Hashprüfung über diese URLs
frisch geladen. Lokale Netzwerksandbox kann unterstützte Freigabe für den
Proxyzugriff erfordern; nicht Proxy, TLS- oder Hashprüfung umgehen. Erwartet:

- EMPD2.sqlite3: `5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2`
- monticchio_739452.tab: `46ab6cea3a636d1a25b8c15f4b9b79a94c8fd1bd41f2943ae5c8941f9050a4de`

Der Extraktor und der Prüfer verlangen beide diese Sperren und Bytezahlen;
bei jeder Abweichung stoppen. Kein neuer Hash als Ersatz. Roharchive nicht
ins Repository aufnehmen; Quellenlizenzen und Attribution im Manifest beachten.

## Extraktion und separate Prüfung

Das Ausgabeziel muss neu sein. SQLite wird ausschließlich read-only geöffnet;
keine Messwerte ausgewertet oder Taxa harmonisiert. Der Manifest-Basiscommit
`98ef1bf` muss für die Prüfung der drei Repository-Eingänge lokal verfügbar sein.

```bash
python docs/thesis_research/original_pollen_names_2026_10_31/extract.py \
  --raw-dir /workspace/netzel-source-cache/pollen-originals-2026-10-31 \
  --output-dir /workspace/netzel-source-cache/neue-namensextraktion
python docs/thesis_research/original_pollen_names_2026_10_31/verify.py \
  --raw-dir /workspace/netzel-source-cache/pollen-originals-2026-10-31 \
  --output-dir /workspace/netzel-source-cache/neue-namensextraktion \
  --report /workspace/netzel-source-cache/neue-namensextraktion/verification.json
```

Den gespeicherten Repository-Auszug selbst prüfen:

```bash
python docs/thesis_research/original_pollen_names_2026_10_31/verify.py \
  --raw-dir /workspace/netzel-source-cache/pollen-originals-2026-10-31 \
  --output-dir docs/thesis_research/original_pollen_names_2026_10_31/data
```

Keine alten Chat-Zwischendateien erforderlich. Die gespeicherten CSVs und das
Manifest sind deterministische Metadatenauszüge; ein zweiter vollständiger
wissenschaftlicher Lauf wurde weder benötigt noch behauptet. Die ursprüngliche
EMPD-Datei ist nicht der aktualisierte 2025-Datensatz. Keine Zählungen aus beiden
Versionen oder ursprüngliche und akzeptierte Konzept-IDs ungeprüft vermischen.
