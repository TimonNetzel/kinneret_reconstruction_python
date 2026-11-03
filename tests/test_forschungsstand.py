"""Pinned research evidence must remain separate from predictions in both offline app builds."""
import json
from pathlib import Path
import pytest
from kinneret_recon import forschungsstand
from kinneret_recon.app_bauen import baue_eine_datei, baue_ordner

ROOT=Path(__file__).resolve().parents[1]


def test_pinned_evidence_retains_scientific_limits_and_sources():
    d=forschungsstand.app_daten(ROOT)
    assert d['available'] and d['species_count']==7 and d['baseline_available']
    assert d['summary']['historical_models']['caret_fits']==167
    assert not d['summary']['plots']['numeric_TF_arrays_recovered']
    assert not d['summary']['gates']['fossil_pilot_release']
    assert d['summary']['gates']['monticchio_P1']=='open'
    assert set(d['sources'])=={'baseline','species','historical','paper','chronology'}
    assert all('/blob/'+forschungsstand.SOURCE_COMMIT+'/' in s['url'] for s in d['sources'].values())
    assert '.local/' not in json.dumps(d) and 'C:\\' not in json.dumps(d)


def test_missing_snapshot_is_visibly_unavailable(tmp_path):
    assert forschungsstand.app_daten(tmp_path)=={'schema_version':1,'available':False}


def test_changed_snapshot_is_rejected(tmp_path, monkeypatch):
    content=(ROOT/forschungsstand.SNAPSHOT).read_bytes()
    monkeypatch.setattr(forschungsstand, "SNAPSHOT", Path("snapshot.json"))
    (tmp_path/forschungsstand.SNAPSHOT).write_bytes(content+b' ')
    with pytest.raises(ValueError,match='hash mismatch'):forschungsstand.app_daten(tmp_path)


def test_snapshot_does_not_claim_missing_models_or_metrics_are_packaged(tmp_path, monkeypatch):
    content=(ROOT/forschungsstand.SNAPSHOT).read_bytes()
    monkeypatch.setattr(forschungsstand, "SNAPSHOT", Path("snapshot.json"))
    (tmp_path/forschungsstand.SNAPSHOT).write_bytes(content)
    data=forschungsstand.app_daten(tmp_path)
    assert data['available'] and not data['baseline_available'] and data['species_count']==0
    assert data['sources']=={}


def test_research_evidence_in_both_offline_formats_and_service_worker(tmp_path):
    out=tmp_path/'app';folder=baue_ordner(out,ROOT)
    html=baue_eine_datei(tmp_path/'single.html',ROOT).read_text(encoding='utf-8')
    js=(out/'daten/forschung.js').read_text(encoding='utf-8')
    prefix='(window.KR_DATEN=window.KR_DATEN||{})["forschung"]='
    data=json.loads(js[len(prefix):-2])
    assert data==forschungsstand.app_daten(ROOT)
    assert prefix in html and 'src="daten/forschung.js"' not in html
    assert 'daten/forschung.js' in (out/'sw.js').read_text()
    assert 'forschung.js' in folder.read_text(encoding='utf-8')
    assert '"fr_historical_body"' in html and 'not 279 independent species functions' in html
    assert 'fr_baseline_missing' in html and 'fr_species_missing' in html