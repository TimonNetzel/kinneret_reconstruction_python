# Monticchio: Diagnoseplan vor dem Ergebnislauf

Basis: PR #18, `f9acc16c58a47a51b1063690a69a2db01381c76a`.
Die moderne Klimabaseline wird weder verändert noch erneut entwickelt.

- Dieselben 37 Gruppen, Zählsummen einschließlich Pinus und derselbe moderne
  Schwellenfilter (7.078 Proben); alle 69 fossilen Proben separat prüfen.
- Hellinger-Abstand `sqrt(sum((sqrt(p)-sqrt(q))**2)/2)`, Ranggleichstände
  nach moderner Proben-ID. Die ersten fünf Analoge samt Koordinaten und
  Abstand offenlegen; Zahl unterschiedlicher Koordinatengruppen berichten.
- Rein beschreibender moderner Distanzvergleich: jede moderne Probe gegen
  alle anderen Koordinatengruppen; d1 und d5, empirische 95-/99-%-Quantile
  mit linearer Interpolation. Überschreitungen sind Diagnoseflags, keine
  validierte Nichtanalog-Grenze oder fossile Erfolgswahrscheinlichkeit.
- Geographische Sensitivität: modernes Referenzprofil und fossile Suche
  zusätzlich mit 100-km-Ausschluss um die jeweilige Probe. Kein neuer
  Klimatest; dieser gepoolte Distanzvergleich ist keine räumliche OOF-Validierung.
- Alle Fossilränge unabhängig mit direkter Python-Distanzrechnung prüfen;
  moderne Referenzstichprobe: erste, mittlere, letzte sortierte Probe.
- Chronologie: Cut gegen All und Entity 932 prüfen; Tiefenoffsets, Monotonie,
  Altersgrenzen, Klassen, Lücken und Überlappung benachbarter Intervalle.
  Control_P 804606 frisch beziehen und auswerten. Keine Altersensembles
  aus unabhängig gezogenen Randintervallen erfinden; kein Extrapolieren.
- Die Bedeutung der Altersgrenzen und Kontrollpunktcodes in Primärquellen
  recherchieren; spätere/andere Kompositkerne nicht still zuordnen.

Keine fossilen Klimavorhersagen, keine Änderung an Claudes Integrationsdateien.
