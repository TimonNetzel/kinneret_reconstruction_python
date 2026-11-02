# Lokales Referenzbeispiel erneut ausführen

Stand 01.11.2026; Windows x64, CPython 3.12.10. Es wird der vorhandene
Kinneret-Paperfall geprüft, keine vollständige taxonbasierte Multiproxy-Rekonstruktion
der Dissertation. Einstieg und wissenschaftliche Grenzen:
[Quellenabgleich](LOCAL_SOURCE_CROSSWALK.md).

## Build ohne Systeminstallation

In dieser Sitzung: LLVM-MinGW 20261020 / Clang 23.1.4, pybind11 3.1.0,
NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2. Vollständige installierte Versionen
in `.local/requirements-lock.txt` und in der Provenienz jedes Laufs.

Offizielles portables Compilerarchiv:
[Release 20261020](https://github.com/mstorsjo/llvm-mingw/releases/tag/20261020),
`llvm-mingw-20261020-ucrt-x86_64.zip`, 190.612.575 Bytes,
SHA256 `fd5462b7f5d93b2ff981fcd3f9c7e427e47f3203c64ea91c27bcc444eebd74cd`.
Lokal vor dem Entpacken gegen den von GitHub veröffentlichten Digest geprüft.
Entpackter Umfang ca. 750 MB. Ein Compilerpfad ist eine lokale Einstellung;
kein persönlicher Pfad wird hier versioniert.

In der eingerichteten Projektumgebung, vom Repositoryverzeichnis:

```powershell
$compiler = (Get-Content .local/compiler.json -Raw | ConvertFrom-Json).compiler
.venv/Scripts/python.exe tools/build_windows.py --compiler $compiler
.venv/Scripts/python.exe tools/build_windows.py --compiler $compiler --original
.venv/Scripts/python.exe tools/verify_local_legacy.py
```

Der normale Paketbuild per `setup.py` bleibt unverändert. Der zusätzliche lokale
Buildhelfer verwendet die bestehenden C++-Quellen, C++17, `-O3`,
`-ffp-contract=off`, statische Laufzeitbibliotheken und `_USE_MATH_DEFINES` für
die vorhandenen UCRT-Mathematikkonstanten. Die resultierende `.pyd` wird nicht
versioniert. Das separat gebaute Original bleibt unter `build/windows/`.
Keine Änderung an Registry, globalem PATH oder globalen Git-Einstellungen.

## Vollständige Läufe und unabhängige Prüfung

Beide Zielverzeichnisse müssen neu sein. Die Helfer verweigern das Überschreiben.
Alle wissenschaftlichen Eingaben bleiben in `data/in`, Originalbestand und
vorhandene Baselines unverändert.

```powershell
.venv/Scripts/python.exe tools/run_local_reference.py --config configs/original.toml --output results/replica_original --threads 1
.venv/Scripts/python.exe tools/run_local_reference.py --config configs/local_reference_v2.toml --output results/replica_corrected --threads 1
.venv/Scripts/python.exe tools/verify_local_reference.py --baseline results/replica_original --corrected results/replica_corrected --output results/replica_review --replay
```

Die vollständig ausgeschriebene Konfiguration `configs/local_reference_v2.toml`
fixiert die tatsächlich gewählte Variante: vorhandenes v2-Modell, fester Dir(0,5)-Prior,
Paperanker, vier Ketten à zwei Millionen Iterationen, Burn-in 500.000, Seed 2023,
Speicherintervall 600 statt 150. Das Speicherintervall ändert nicht die
Zielverteilung. Es verbleiben 10.000 Ziehungen nach dem Burn-in.
Der zusätzliche Prior-Referenzlauf gehört zur vorhandenen Pipeline und wird
ebenfalls ausgeführt; dies ist ein wesentlicher Teil der gesamten Laufzeit.

`run_local_reference.py` setzt nur Ausgabepfad und Threadzahl, importiert die
unveränderte wissenschaftliche Pipeline und protokolliert Exitstatus, Konfiguration,
Eingabe-, Quell-, Binär- und Ausgabehashes, Seed, Build, Versionen, Laufzeit und
maximalen Working Set. Kein Originaldaten-Upload und keine pauschale Datenkopie.

Der Prüfer berechnet Klimakurven direkt aus Eingangsspektren, Biomzuordnung und
gespeicherten Ziehungen mit einer separaten NumPy-Formel. Er verwendet dafür
weder `postprocess.reconstruct` noch die Tabellenfunktion der Pipeline.
Alle CSV-Quantile und Mittel werden mit der schon vorhandenen Export-Rundung
verglichen. Er prüft Eingabe-/Quell-/Ausgabeintegrität, Gewichtsnormalisierung,
Endlichkeit, TF-Grenzen, nichtnegativen Niederschlag sowie die vorab festgelegten
Konvergenzkriterien. `--replay` wiederholt beide Posterior-Sampler vollständig.
Die korrigierten Ketten werden bei der Wiederholung mit zwei Workern ausgeführt;
die Arraygleichheit prüft damit auch Unabhängigkeit von dieser Threadzahl.

Lange Wiederholungen lassen sich getrennt protokollieren, während die Pipeline
noch ihren unabhängigen Prior-Referenzlauf auswertet:

```powershell
.venv/Scripts/python.exe tools/replay_local_sampler.py --config results/replica_original/config.toml --posterior results/replica_original/posterior.npz --output .local/replica_original_replay.json --threads 1
.venv/Scripts/python.exe tools/replay_local_sampler.py --config configs/local_reference_v2.toml --posterior results/replica_corrected/posterior.npz --output .local/replica_corrected_replay.json --threads 2
.venv/Scripts/python.exe tools/verify_local_reference.py --baseline results/replica_original --corrected results/replica_corrected --output results/replica_review_cached --baseline-replay-record .local/replica_original_replay.json --corrected-replay-record .local/replica_corrected_replay.json
```

Solche Nachweise werden nur übernommen, wenn Posteriorbytes, Samplerkonfiguration,
Eingaben, wissenschaftlicher Quellcode und Binärdatei zum geprüften Lauf passen.
Bei der ersten Zeile muss der Originallauf abgeschlossen sein, damit dessen
exportierte Konfiguration existiert. Die zweite Zeile kann nach vollständiger
Bereitstellung von `posterior.npz` beginnen. Die abschließende Integritätsprüfung
setzt den Abschluss beider kompletten Pipeline-Läufe voraus.

## Grenzen der Bitgleichheit

Der Legacy-Code benutzt `std::default_random_engine` und
`std::gamma_distribution`. Die lokale libc++ definiert den Defaultgenerator als
[`minstd_rand`](https://github.com/llvm/llvm-project/blob/main/libcxx/include/__random/default_random_engine.h),
libstdc++ als
[`minstd_rand0`](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/random.h).
Der lokale Probeaufruf bestätigt die abweichenden ersten Zahlen.
Eine identische Seedzahl garantiert damit keinen identischen Legacy-Pfad über
diese Laufzeiten hinweg; auch mathematische Bibliotheken können Unterschiede
verursachen. Der fehlgeschlagene Archivtest wird nicht umgeschrieben.
Der Originalvergleich unter derselben Laufzeit und vollständige Wiederholungen
belegen eine andere, genau benannte Reproduzierbarkeit.

Die ausgewählten bestehenden Testprotokolle und eine ergänzende Diagnose der
optionalen Dirichlet-Vorschlagsvariante werden im Bericht eingelesen, falls sie
lokal vorliegen. Sie werden weder vorausgesetzt noch als neue Testergebnisse
erfunden. Fehlende Zusatzprotokolle sind im neuen Arbeitsbereich separat auszuführen.
Die eigentlichen unabhängigen Rekonstruktionsprüfungen laufen ohne diese
historischen Zusatzprotokolle erneut.
