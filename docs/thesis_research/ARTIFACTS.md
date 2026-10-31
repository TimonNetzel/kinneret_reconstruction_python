# Artefaktverzeichnis: vier große Laufausgaben

Stand 31.10.2026, geprüft am Forschungscommit `9ee2f84e41a04fafe5af203e809deeaaf329dba3`.
Die Nachweise sind veröffentlichte Manifeste und Wiederholungsberichte, **keine
neue Byteprüfung der vier Dateien**. Kein großer Lauf wurde neu gerechnet.

| Datei | SHA-256 der Dateibytes | Bytes laut repeat.json | Bekannter dauerhafter Speicherort der Bytes | Erzeugung / Umgebung |
| --- | --- | --- | --- | --- |
| `predictions.csv` (K) | `73cd156b1775216e81d2aaf016f2ddaa5c8a75a8d7ace88ed2c0fbe5657b14ca` | 53225677 | **Nicht belegt**; kein Git-Blob laut Sicherungsinventar | Befehl K, Umgebung E |
| `smote_trace.csv.gz` (K) | `e28abaf9fde7d8ab581a880bcc1fd840f31f7373e04f534fb292ff0d0a33e1c3` | 32294815 | **Nicht belegt**; kein Git-Blob laut Sicherungsinventar | Befehl K, Umgebung E |
| `block_draws.csv.gz` (U) | `1b8117c5052408613c953a4e63d80537a34410bcaa6663a4f71aaae5b374885b` | 6209076 | **Nicht belegt**; kein Git-Blob laut Sicherungsinventar | Befehl U, Umgebung E |
| `intervals.csv.gz` (U) | `645ee0d98b27f97b42ef98eb4a87234a8317ee39ad1cf331942e02fd6c54ca8b` | 56825733 | **Nicht belegt**; kein Git-Blob laut Sicherungsinventar | Befehl U, Umgebung E |

## Nachweiskette und offene Speicherlücke

K = [#31-Manifest](./calibration_2026_10_30/run_manifest.json),
[Wiederholungsbericht](./calibration_2026_10_30/repeat.json),
[Ausführungsbefunde](./calibration_2026_10_30/execution_notes.json),
[Checkpoint-Wiederherstellung](./calibration_2026_10_30/checkpoint_recovery.json).
U = [#32-Manifest](./calibration_uncertainty_2026_10_30/run_manifest.json),
[Wiederholungsbericht](./calibration_uncertainty_2026_10_30/repeat.json),
[Ausführungsbefunde](./calibration_uncertainty_2026_10_30/execution_notes.json).
Alle vier Hashes stimmen zwischen jeweiligem Manifest, repeat.json und
[Chat-Inventar](./CHAT_WORK_ARCHIVE_2026_10_30.json) überein. Das Inventar führt
sie als `hash_and_reproduction_only`; keine der vier Dateien ist im geprüften
Git-Baum enthalten. Im Arbeitsverzeichnis dieses Dokumentationspakets sind sie
nicht vorhanden. Ein späteres privates Backup ist damit weder belegt noch ausgeschlossen.

**Offen:** dauerhafte URI/Repository-/Objektversion, tatsächliche Verfügbarkeit
und erneute Bytehashprüfung eines Archivs. Die unten genannten `work/`-Pfade sind
nur frische lokale Ausgabeziele, keine bekannten dauerhaften Speicherorte.
Alte temporäre Dateien werden nicht vorausgesetzt. Ein Hash sichert keine Bytes.
Ein Vollarchiv wäre ein separates Paket: vorhandene Originalausgaben auffinden
oder ausdrücklich beauftragte Reproduktion, Hash/Gzip prüfen und einen geeigneten
dauerhaften Speicherort samt Version dokumentieren. Fremde Roharchive nicht
ungeprüft weiterverteilen.

## Erzeugungsbefehle (Nachweis, jetzt nicht ausführen)

Arbeitsverzeichnis ist die Wurzel eines frischen Checkouts des gesperrten
Forschungsstands. K ist am #31-Ergebniscommit
`9458011198c69368764b8698c966c0ac23cc381c` dokumentiert; U am #32-Ergebniscommit
`6d35bbfe06cf54572dba2cb439495a8556eddb4d` mit unveränderter #31-Basis.
Vorher Quellen frisch und hashgeprüft mit `tools/taxon_smote/prepare.py`
beschaffen; vollständiger Befehl und Quellenregister stehen in
[K-Reproduktion](./calibration_2026_10_30/REPRODUCE.md) und
[U-Reproduktion](./calibration_uncertainty_2026_10_30/REPRODUCE.md).
Keine Eingänge aus früheren Chats voraussetzen. Neue Ausgabeziele verwenden.

Befehl K erzeugt gemeinsam `predictions.csv` und `smote_trace.csv.gz`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_calibration/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_2026_10_30/config.json \
  --output work/calibration_run1
```

Befehl U erzeugt gemeinsam `block_draws.csv.gz` und `intervals.csv.gz`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/taxon_uncertainty/run.py \
  --previous work/pr28 --maps work/maps --climate work/wc2.1_10m_bio.zip \
  --config docs/thesis_research/calibration_uncertainty_2026_10_30/config.json \
  --output work/uncertainty_run1
```

Umgebung E: beide Laufmanifeste nennen Python **3.12.14**, NumPy **2.5.4** und
SciPy **1.17.0**. Die Reproduktionsanleitungen verweisen zusätzlich auf das feste
[Forschungsregister](./smote_nnet_2026_10_30/requirements-research.txt): Shapely
2.2.0, Rasterio 1.5.2, scikit-learn 1.8.0, Matplotlib 3.10.8. Pro Fit ein
BLAS-Thread; sechs Prozesse laut Anleitungen. Konkrete CPU, Betriebssystem,
BLAS-Build und vollständiger transitiver Lock sind in diesen Nachweisen
**nicht angegeben**; keine plattformübergreifende Bitgleichheit zugesichert.

K: Elf Endausgaben bytegleich erst nach ausdrücklich dokumentierter
Checkpoint-Wiederherstellung; verworfene unvollständige Gzip-Spur nicht als
Erfolgsnachweis verwenden. U: sieben bytegleiche Enddateien zweier frischer
Läufe mit gültigen Gzip-Streams; frühere unvollständige Versuche ausgeschlossen.
Diese historischen Befunde ersetzen weder das fehlende Bytearchiv noch einen
neuen Audit. Für künftige Reproduktion auch die separaten Prüfbefehle der
jeweiligen REPRODUCE.md verwenden; hier werden sie nicht ausgeführt.
