"""Synthetic parser checks: prevent commented/quoted paths becoming dependencies."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/thesis_research/required_files_2026_11_02/collect_evidence.py"
spec = spec_from_file_location("required_files_audit", PATH)
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


def test_comments_and_function_names_inside_strings_do_not_create_reads():
    text = "# load(\"comment.rdata\")\nmessage(\"load('quoted.rdata')\")\nload(\"real.rdata\") # nc_open('comment.nc')\n"
    items = list(audit.calls(text))
    assert [(c["function"], c["line"]) for c in items] == [("load", 3)]
    assert audit.render_path(audit.path_expression(items[0])) == "real.rdata"


def test_hash_and_parentheses_in_strings_and_multiline_arguments_are_preserved():
    text = 'load(\n paste0("folder/#(x)/", id, ".rdata"),\n envir=parent.frame())'
    item = list(audit.calls(text))[0]
    assert item["line"] == 1
    assert audit.render_path(audit.path_expression(item)) == "folder/#(x)/{id}.rdata"


def test_namespaced_reader_and_named_filename_are_recognized():
    item = list(audit.calls('ncdf4::nc_open(filename="a.nc", write=FALSE)'))[0]
    assert item["function"] == "nc_open"
    assert audit.render_path(audit.path_expression(item)) == "a.nc"


def test_writes_use_output_file_argument_and_are_distinct_from_reads():
    items = list(audit.calls('save(x, y, file="out.rdata"); saveRDS(x, "out.rds"); load("in.rdata")'))
    assert [audit.render_path(audit.path_expression(c)) for c in items] == ["out.rdata", "out.rds", "in.rdata"]
    assert [c["function"] in audit.WRITERS for c in items] == [True, True, False]


def test_literal_prefix_binding_preserves_unresolved_model_and_period_variables():
    text = 'in_path <- "/automount/lost/cmip_pmip"\nnc_open(paste0(in_path,"/",period,"/x_",model,".nc"))'
    item = list(audit.calls(text))[0]
    values = audit.constants_before(item["code"], item["position"])
    assert audit.render_path(audit.path_expression(item), values) == "/automount/lost/cmip_pmip/{period}/x_{model}.nc"


def test_conditionals_are_recorded_without_claiming_execution():
    item = list(audit.calls('if(FALSE) load("optional.rdata")'))[0]
    assert audit.render_path(audit.path_expression(item)) == "optional.rdata"
    # This parser deliberately does not label the branch as a mandatory runtime input.


def test_file_path_and_escaped_quote_keep_argument_boundaries():
    parts = audit.split_args(r'"a\"b,#.rdata", nested(x,y), option=FALSE')
    assert len(parts) == 3
    assert audit.render_path('file.path("a", folder, "x.rdata")') == "a/{folder}/x.rdata"


def test_unknown_expressions_remain_unresolved_and_are_not_executed():
    assert audit.render_path('system("do not execute")').startswith("{system(")
    assert audit.render_path('unknown_file') == "{unknown_file}"
