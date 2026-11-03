# Portabler Forschungsstand vom 02.11.2026

Dieses Paket sichert ausgewählte, bereits berechnete Befunde für die
[Claude-/App-Übergabe](../CLAUDE_HANDOFF_2026_11_02.md).
**Kein wissenschaftlich freigegebener neuer Fossilfall und keine
Wiederherstellung numerischer 279-Taxon-Transferfunktionen.**

- `summary.json`: kompakter Status für Befundprüfung und spätere App-Einbindung.
- `manifest.json`: Herkunft jeder Exportkopie, Originalhash, Exporthash,
  Transformation und Umfang. Absolute maschinenspezifische Pfade wurden ersetzt.
- `historical_provenance/`: Raster-/Klima-/Modellleserbefunde.
- `console/`: 271 Kardinalitäten, 278 EMD-Seitenbindungen, unabhängiger PDF-Check.
- `second_plot/`: zweiter Plot, exakte Dezimalvergleiche und Quellbindung.
  `vector_comparison.json.gz` enthält den vollständigen verlustfreien JSON-Export.
- `paper_verification/`: vorhandene Paper-/Samplerprüfungen und Laufzeitgrenzen.
- `paper_results/`: CSV-Ergebnisse und Diagnostik beider vorhandenen Paperläufe,
  Konfigurationen, Provenienz, Vergleichsgrafik und lesbarer Prüfbericht.
- `handoff_tests/`: erneuter erfolgreicher Lauf der 67 betroffenen Tests.

Manifest prüfen, nur Python-Standardbibliothek nötig:

```bash
python tools/verify_research_handoff.py
```

Für Quellen-/Lesertests mit den normalen Projekt-Testabhängigkeiten:

```bash
python -m pytest tests/test_console_cardinality.py tests/test_historical_provenance.py tests/test_local_rdata_bytecode.py tests/test_local_rdata_metadata.py tests/test_local_rdata_stream.py tests/test_tf_plot_marks.py tests/test_tf_plot_comparison.py -q
```

Das setzt keine Original-RData-/NetCDF-/PDF-Dateien voraus. Für den PDF-Audit
selbst werden die Originale sowie pypdf und der dokumentierte MuPDF-Runtime
benötigt. Diese optionalen Lesewerkzeuge sind nicht Teil des normalen App-Builds.

Eine große JSON-Datei ohne zusätzliche Pakete lesen:

```python
import gzip, json
from pathlib import Path
p = Path('docs/thesis_research/research_snapshot_2026_11_02/second_plot/vector_comparison.json.gz')
with gzip.open(p, 'rt', encoding='utf-8') as stream:
    evidence = json.load(stream)
print(evidence['counts'])
```

Gespeicherte Abschlussprüfungen sind historische Ergebnisse. Ihre internen
Originalhashes beziehen sich auf die unveränderten lokalen Dateien, nicht auf
hier neu serialisierte oder um Pfade bereinigte Exportkopien. Das Exportmanifest
trennt beide Byteidentitäten. Originaldaten, vollständige Samplerarchive,
Runtimes und persönliche Einstellungen sind absichtlich nicht neu enthalten.
