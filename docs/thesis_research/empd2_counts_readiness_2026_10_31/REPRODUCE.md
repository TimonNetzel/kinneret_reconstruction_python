# Lesende Nachprüfung des EMPD2-Audits

Standardbibliothek Python/SQLite/Decimal/Hashlib. Kein Anwendungspaket,
Modellimport, GIS oder wissenschaftlicher Lauf. Vorhandenes Original bevorzugen;
21.327.872 Bytes/SHA-256 aus INPUT_CONTRACT.json vor jeder Lektüre prüfen.
Beschaffung nur bei fehlenden Bytes nach unverändertem
[#35-Nachweis](../original_pollen_names_2026_10_31/REPRODUCE.md).
In diesem Paket war keine Beschaffung nötig. Das Original wurde nicht in Git
aufgenommen oder als neues dauerhaftes Bytearchiv bezeichnet.

Verbindung ausschließlich lesend:

```python
import hashlib, sqlite3
from pathlib import Path
p = Path('/workspace/netzel-source-cache/pollen-originals-2026-10-31/EMPD2.sqlite3')
assert p.stat().st_size == 21327872
expected = '5634e85d922ee0741c41a77d081010c9cc1c113966b283bc52e2051a9d3722e2'
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
con = sqlite3.connect(p.as_uri() + '?mode=ro&immutable=1', uri=True)
con.execute('PRAGMA query_only=ON')
```

Die SQL-Abfragen des gesicherten Belegauszugs samt Parametern und Ergebniszeilenzahl
stehen in evidence.json/query_log. Die vorherige explorative Schema-/Metadaten-
Lektüre und punktuelle Wörterbuchabfragen führten zu denselben drei Zeugen,
nicht zu einer zusätzlichen Probenkohorte. Quellenfelder und Strings einschließlich
Text-NULL bleiben erhalten. Relevante neun Tabellenschemata, 40 Gruppen und
Codebeschreibungen sind als Metadaten gespeichert; keine Worker-/Kontaktwerte.

Die drei Auswahlabfragen liefern lexikalisch die ersten Fälle für ihre jeweilige
Eigenschaft, ohne spätere Nachwahl. Nur deren 20+49+39 Messzeilen wurden gelesen.
Kein vollständiger Sample×Taxon-Export; gespeichert sind je fünf Beispielwerte,
der eine gebrochene Volumenwert und arithmetische Gruppensummen. Summen mittels
`Decimal(CAST(count AS TEXT))`, nicht mit gerundeten Pseudo-Counts. Summen sind
ausdrücklich keine genehmigten N. Zusätzlicher punktueller Nichttreffer:
AlbaSanchez_a1/Originalcode 7, dessen Wörterbuchname Artemisia lautet.

Vor/nach dem Lesezugriff identischer Originalhash ist in evidence.json/source
festgehalten. Historische globale Importkennzahlen aus reference_run.json werden
nicht neu berechnet oder ersetzt. Volltexte der Methoden-/Probenpublikationen
nicht als gelesen ausweisen: hier nur vorhandene Bibliographie/Repositorykontext.

Dokumentationsprüfung: JSON-Konsistenz, drei Zeugen/108 gelesene Messzeilen,
Grundgesamtheit gegenüber Beispielen, Quell-/Hashbelege, lokale Links und
begrenzter Diff. Alle geschützten Git-Blobs gegen Basis `62ffeaf` vergleichen;
einziger erlaubter bestehender Dokumentdiff HANDOFF/PROJECT_STATUS.
[Prüfprotokoll](./verification.json). Keine Tests/Fits/Simulationen ausgeführt.
GitHub-CI wird beim Push/PR getrennt vom unveränderten Repository ausgelöst.
