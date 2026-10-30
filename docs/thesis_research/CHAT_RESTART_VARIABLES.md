# Fortsetzung nach der Variablenprüfung

Stand 30.10.2026. Eigener Branch `codex/netzel-variable-source-audit`, Basis
#18 auf `f9acc16c58a47a51b1063690a69a2db01381c76a`. Vor neuer Arbeit PR-/Merge-Stand
erneut prüfen; #18–#26 waren beim Beginn dieser Arbeit offen und ungemergt.
Claude-#19 unverändert auf `c8977a57fecabcef8f54abd7eafd6545f53487f9`.

Ergebnis: [PR #27](https://github.com/TimonNetzel/kinneret_reconstruction_python/pull/27)
auf Ergebniscommit `49534dbc9e72c5c2ad2ca22e92a57d1a7519a985`.
Spätere reine Status-/Übergabecommits verändern die numerischen Ergebnisse nicht.

Zuerst [PROJECT_STATUS.md](./PROJECT_STATUS.md) und
[neuen Quellenbericht](./variable_provenance_2026_10_30/README.md) lesen.
Die Branches sind weiterhin getrennt. Frühere vollständige Übergaben:
[PR #26](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/4c9f1150e766a7f855dc9e28fd121638a710ff9a/docs/thesis_research/CHAT_RESTART_PR26.md),
[PR #25](https://github.com/TimonNetzel/kinneret_reconstruction_python/blob/00c9f037642470c7d1a9a44e488f062109292c2d/docs/thesis_research/CHAT_RESTART.md).

Neu belegt: 483 Originalzellen in drei Quellen gleich. Alle 35 Plantago-Differenzen
bilanzieren sich bei den festen #26-Diagnosepaaren als vier alte Felder gegen
ein neues; neun alte Scrophulariaceae-Werte sind gleich Digitalis-type. EPD
763 verweist auf Morphofeld 918 (colporate-reticulate). Zwei Abies- und ein
Sambucus-Zahlenunterschied bleiben; Originaleinreichung/Umcodierung fehlen
auch für die 44 Zahlenmuster. **Keine Quelle korrigiert oder Zuordnung freigegeben.**
28 Kandidaten / 41 native Nichttreffer aus #26 bleiben unverändert. Alle 47
Herkunftsfragen sind weiterhin offen; P1 und MON16-Anschluss nicht gelöst.

Zwei Produzentenläufe bytegleich, separater Rohslot-/Fraction-/SQL-COPY-Prüfer
bestanden, vier neue Werkzeugtests bestanden. Gemeinsamer RDS-Binärparser
bleibt eine benannte Abhängigkeit. Quellen mit URL, Bytezahl und SHA256 sowie
die zwei aus #26 zu beziehenden Leser im neuen Quellenregister. Keine alten
temporären Dateien nötig; ScienceDirect-ZIP nicht benötigt.

Nächster Eingang: [präzisierter Einreichungsbeleg](./variable_provenance_2026_10_30/SOURCE_GAP.md)
für 26608/47525 mit datensatzbezogener Synonymie, Einreichungshistorie und
Originalzählnotizen, danach historischer Kompositanschluss samt Unsicherheit.
Neue Taxon-Transferfunktionen sind noch nicht erstellt. Baseline fertig;
Claude betreut CLI/App, Codex Quellen und unabhängige Prüfung. Keine stillen
Korrekturen, Pilotfreigaben, Autorennachrichten oder Merges. Eigener PR,
kurze deutsche Updates: fertig / aktuell / als Nächstes.
