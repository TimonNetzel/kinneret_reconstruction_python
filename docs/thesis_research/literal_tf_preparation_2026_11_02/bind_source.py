"""Bind only the existing preparation source and already frozen excerpts."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASIS='956951b9546f3120620bf375bf073066c9f8bf44'


def main():
    prior=ROOT/'docs/thesis_research/literal_phase1_chain_2026_11_02/next_step_source_excerpt.json'
    excerpt=json.loads(prior.read_text())
    cfg=json.loads((ROOT/'.local/my_reconst_paths.json').read_text(encoding='utf-8-sig'))
    source=Path(cfg['original_data_root'])/'src/mcmc_prepare.R'
    raw=source.read_bytes();before=hashlib.sha256(raw).hexdigest()
    if before!=excerpt['sha256']:raise ValueError('Original source differs from frozen binding')
    lines=raw.decode('utf-8-sig').splitlines()
    for span in excerpt['spans']:
        assert '\n'.join(lines[span['first_line']-1:span['last_line']])==span['text']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    (OUT/'source_excerpt.json').write_bytes(prior.read_bytes())
    binding={'status':'bounded_finite_historical_source_fragment_not_case_reproduction','basis_commit':BASIS,
      'source':'original:src/mcmc_prepare.R','source_bytes':len(raw),'source_sha256':before,
      'source_hash_before_equals_after_and_prior':True,'portable_excerpt':'source_excerpt.json',
      'portable_excerpt_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
      'implemented_spans':{'normalize_tf_rows':[1581,1586],'mask_tf_cells':[1613,1618]},
      'conditional_change_branch':'Same formulas only if taxa_change; no branch activated or final variant selected',
      'excluded_context':'sample/quantile/spline/weightedMedian/inside.range/taxon pruning are not executed',
      'axes':'ordered taxon rows x ordered climate-cell columns; separate explicit TANN and PANN masks',
      'units':'discrete row sums without cell widths; no climate unit conversion or physical integration',
      'supported_domain':'nonempty real numeric matrices converted to float64; finite nonnegative values; finite positive row sums before normalization; exactly one boolean per cell',
      'NA_Inf_zero_row_recycling_or_dimension_drop_semantics_reproduced':False,
      'inside_range_boundary_convention_or_package_selected':False,
      'actual_original_inputs_used_for_numerics':False,'R_or_package_runtime_equivalence_claimed':False,
      'run_binding':None,'historical_reproduction_ready':False}
    (OUT/'source_binding.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Source hash and all four frozen source spans verified; originals read only')


if __name__=='__main__':main()
