"""Bounded read-only source/data dependency audit, not an R interpreter.

Scope: configured original/additional roots and the existing checkout only.
Inspect R source, data filenames/metadata and ZIP directories; never load RData,
NetCDF arrays or execute R. Comments are excluded; branches/function reachability
remain explicit limitations. Outputs distinguish absence from copy candidates.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA_EXT = {'.rdata', '.rda', '.rds', '.nc', '.nc4', '.csv', '.txt', '.xlsx', '.xls', '.tif', '.grd', '.shp'}
PRIMARY_EXT = {'.rdata', '.rda', '.rds', '.nc', '.nc4'}
READERS = {'load', 'readRDS', 'nc_open', 'read.csv', 'read.csv2', 'read.table', 'read.delim', 'read_csv', 'read_excel', 'raster', 'brick', 'stack', 'source'}
WRITERS = {'save', 'saveRDS', 'write.csv', 'write.table', 'nc_create'}
EXCLUDED_DIRS = {'.git', '.venv', '.local', 'node_modules', '__pycache__', 'build'}
CUTOFF = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def mask_comments(text):
    chars = list(text); quote = None; escaped = False; comment = False
    for i, ch in enumerate(text):
        if comment:
            if ch == '\n': comment = False
            else: chars[i] = ' '
        elif quote:
            if escaped: escaped = False
            elif ch == '\\': escaped = True
            elif ch == quote: quote = None
        elif ch in "\"'": quote = ch
        elif ch == '#': chars[i] = ' '; comment = True
    return ''.join(chars)


def mask_strings(code):
    chars = list(code); quote = None; escaped = False
    for i, ch in enumerate(code):
        if quote:
            if ch != '\n': chars[i] = ' '
            if escaped: escaped = False
            elif ch == '\\': escaped = True
            elif ch == quote: quote = None
        elif ch in "\"'": quote = ch; chars[i] = ' '
    return ''.join(chars)


def split_args(text):
    parts = []; start = 0; depth = 0; quote = None; escaped = False
    for i, ch in enumerate(text):
        if quote:
            if escaped: escaped = False
            elif ch == '\\': escaped = True
            elif ch == quote: quote = None
        elif ch in "\"'": quote = ch
        elif ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        elif ch == ',' and depth == 0: parts.append(text[start:i].strip()); start = i + 1
    parts.append(text[start:].strip())
    return parts


def calls(text):
    code = mask_comments(text); masked = mask_strings(code)
    pattern = r'(?<![\w.])(?:(?:\w+)::)?(' + '|'.join(re.escape(n) for n in sorted(READERS | WRITERS)) + r')\s*\('
    for match in re.finditer(pattern, masked):
        start = match.end(); depth = 1; i = start
        while i < len(masked) and depth:
            if masked[i] == '(': depth += 1
            elif masked[i] == ')': depth -= 1
            i += 1
        if depth: continue
        yield {'function': match.group(1), 'line': text.count('\n', 0, match.start()) + 1,
               'position': match.start(), 'args': split_args(code[start:i-1]), 'code': code}


def path_expression(call):
    args = call['args']; fun = call['function']
    for arg in args:
        m = re.match(r'^(?:file|filename|con)\s*=\s*(.*)$', arg, re.S)
        if m: return m.group(1).strip()
    if fun == 'save': return None
    if fun in {'saveRDS', 'write.csv', 'write.table'}: return args[1] if len(args) > 1 else None
    return args[0] if args else None


def literal(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        # Only the path escapes needed here; never evaluate arbitrary R expressions.
        return value[1:-1].replace('\\\\', '\\').replace('\\"', '"').replace("\\'", "'")
    return None


def render_path(expr, constants=None):
    constants = constants or {}; expr = expr.strip()
    direct = literal(expr)
    if direct is not None: return direct
    if re.fullmatch(r'[A-Za-z_.][\w.]*', expr): return constants.get(expr, '{' + expr + '}')
    m = re.fullmatch(r'(paste0|paste|file.path)\s*\((.*)\)', expr, re.S)
    if m:
        fun, body = m.groups(); parts = split_args(body); sep = '' if fun == 'paste0' else ('/' if fun == 'file.path' else ' ')
        values = []
        for arg in parts:
            named = re.match(r'^(sep|collapse)\s*=\s*(.*)$', arg, re.S)
            if named:
                if named.group(1) == 'sep': sep = literal(named.group(2)) or ''
                continue
            values.append(render_path(arg, constants))
        return sep.join(values)
    return '{' + re.sub(r'\s+', ' ', expr) + '}'


def constants_before(code, position):
    result = {}
    for m in re.finditer(r'\b([A-Za-z_.][\w.]*)\s*<-\s*([\"\'])([^\n]*?)\2', code[:position]):
        result[m.group(1)] = m.group(3)
    return result


def public(value, roots):
    value = str(value).replace('\\', '/')
    for scope, root in sorted(roots.items(), key=lambda item: len(str(item[1])), reverse=True):
        value = value.replace(root.as_posix(), '<' + scope.upper() + '_ROOT>')
    value = re.sub(r'/automount/[^\"\s]*?/dissertation/', '/automount/<LOST_DISK>/dissertation/', value)
    value = re.sub(r'(?:C:)?/Users/[^/\"\s]+', '<USER_HOME>', value, flags=re.I)
    value = re.sub(r'/home/[^/\"\s]+', '<HISTORICAL_HOME>', value)
    return value


def write_csv(name, rows):
    if not rows: return
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def collect():
    cfg = json.loads((ROOT / '.local/my_reconst_paths.json').read_text(encoding='utf-8'))
    roots = {'original': Path(cfg['original_data_root']), 'additional': Path(cfg['additional_source_root']), 'repository': ROOT}
    index = []; source_paths = []; archives = []; errors = []
    for scope, root in roots.items():
        def onerror(error): errors.append({'scope': scope, 'error_type': type(error).__name__})
        for folder, dirs, files in os.walk(root, onerror=onerror):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and Path(folder) / d != OUT]
            for name in files:
                path = Path(folder) / name; ext = path.suffix.lower(); rel = path.relative_to(root).as_posix()
                if ext in DATA_EXT:
                    stat = path.stat(); index.append({'label': scope + ':' + rel, 'basename': name.lower(), 'path': path,
                                                      'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
                if ext == '.zip' and scope != 'repository': archives.append((scope, root, path))
                if ext == '.r' and scope != 'repository' and 'wind_power_potential' not in rel:
                    source_paths.append((scope, root, path))
    by_name = defaultdict(list)
    for entry in index: by_name[entry['basename']].append(entry)
    zip_index = defaultdict(list); zip_reports = []
    for scope, root, path in archives:
        try:
            with zipfile.ZipFile(path) as archive:
                records = []
                for info in archive.infolist():
                    if Path(info.filename).suffix.lower() in DATA_EXT:
                        label = scope + ':' + path.relative_to(root).as_posix() + '!' + info.filename
                        zip_index[Path(info.filename).name.lower()].append(label)
                        records.append([info.filename, info.CRC, info.file_size])
                zip_reports.append({'archive': scope + ':' + path.relative_to(root).as_posix(), 'data_members': len(records),
                                    'directory_digest': digest(json.dumps(records, ensure_ascii=True).encode())})
        except (OSError, zipfile.BadZipFile) as error:
            errors.append({'archive': scope + ':' + path.relative_to(root).as_posix(), 'error_type': type(error).__name__})
    sources = []; dependencies = []; writers = []; source_edges = []; unresolved = []
    for scope, root, path in sorted(source_paths, key=lambda v: (v[0], v[2].as_posix())):
        rel = path.relative_to(root).as_posix(); label = scope + ':' + rel
        raw = path.read_bytes(); stat = path.stat(); text = raw.decode('utf-8-sig', errors='replace')
        if '/src/' in '/' + rel: project_rel = rel.split('/src/')[0] if '/src/' in rel else ''
        else: project_rel = str(Path(rel).parent).replace('\\', '/'); project_rel = '' if project_rel == '.' else project_rel
        base = root / project_rel if scope == 'additional' else root
        in_src = '/src/' in '/' + rel
        group = 'dissertation_original' if scope == 'original' else ('site_inputs_and_generators' if rel.split('/')[0] in {'my_kinneret','my_ohrid','my_birkat_ram','my_van','my_iznik','my_prespa','my_gedi'} else 'paper_biom_or_other_variant')
        sources.append({'source': label, 'sha256': digest(raw), 'bytes': len(raw),
                        'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                        'filesystem_created_utc_not_historical_proof': datetime.fromtimestamp(getattr(stat, 'st_birthtime', stat.st_ctime), timezone.utc).isoformat(),
                        'is_src': in_src, 'modified_after_2021': stat.st_mtime >= CUTOFF, 'group': group})
        for call in calls(text):
            expr = path_expression(call)
            if not expr: continue
            constants = constants_before(call['code'], call['position'])
            rendered = render_path(expr, constants).replace('\\', '/')
            dynamic = '{' in rendered
            if call['function'] == 'source':
                if not dynamic:
                    target = base / rendered
                    if target.is_file():
                        try: target_label = scope + ':' + target.relative_to(root).as_posix()
                        except ValueError: target_label = public(target, roots)
                        source_edges.append({'source': label, 'line': call['line'], 'target': target_label})
                continue
            row = {'source': label, 'line': call['line'], 'function': call['function'],
                   'expression': public(expr, roots), 'path_or_pattern': public(rendered, roots),
                   'modified_utc': sources[-1]['modified_utc'], 'recent_src': in_src and stat.st_mtime >= CUTOFF,
                   'group': group, 'dynamic': dynamic, 'automount_origin': rendered.startswith('/automount/'),
                   'static_only_not_execution_proof': True}
            if call['function'] in WRITERS:
                writers.append(row); continue
            # Data calls with no filename extension are retained as unresolved, not declared missing.
            basename = rendered.rsplit('/', 1)[-1]
            ext = Path(basename).suffix.lower()
            if ext not in DATA_EXT and not any(e in rendered.lower() for e in PRIMARY_EXT):
                unresolved.append({**row, 'reason': 'Path/extension not statically resolvable; not declared missing'})
                continue
            expected_exists = False
            if not dynamic and not rendered.startswith('/automount/'):
                expected = Path(rendered) if Path(rendered).is_absolute() else base / rendered
                expected_exists = expected.is_file()
            candidates = []
            members = []
            if not dynamic:
                candidates = by_name.get(basename.lower(), [])
                members = zip_index.get(basename.lower(), [])
            else:
                pattern = re.sub(r'\{[^}]*\}', '*', basename).lower()
                if pattern not in {'*', '*.rdata', '*.nc'}:
                    candidates = [entry for name, values in by_name.items() if fnmatch.fnmatchcase(name, pattern) for entry in values]
                    members = [member for name, values in zip_index.items() if fnmatch.fnmatchcase(name, pattern) for member in values]
            status = ('present_at_expected_path' if expected_exists else
                      'local_copy_candidate_needs_binding' if candidates else
                      'archive_copy_candidate_needs_binding' if members else
                      'dynamic_family_unresolved_or_missing' if dynamic else 'not_found_in_checked_scope')
            tail_matches = []
            match = re.search(r'/(my_(?!reconst/)[^/]+)/(.*)', rendered)
            if match:
                tail = match.group(1) + '/' + match.group(2)
                tail_matches = [entry['label'] for entry in candidates if entry['label'].split(':', 1)[1].endswith(tail)]
            row.update({'same_site_path_tail_candidates': ' | '.join(tail_matches), 'extension': ext, 'expected_path_exists': expected_exists, 'availability': status,
                        'local_candidates': ' | '.join(entry['label'] for entry in candidates),
                        'archive_candidates': ' | '.join(members)})
            dependencies.append(row)
        if path.read_bytes() != raw: raise ValueError('Source changed while reading: ' + label)
    graph = defaultdict(set)
    for edge in source_edges: graph[edge['source']].add(edge['target'])
    reachable = {'original:main_v1.2.R'}; frontier = list(reachable)
    while frontier:
        for target in graph[frontier.pop()]:
            if target not in reachable: reachable.add(target); frontier.append(target)
    for row in dependencies:
        row['source_reachable_from_main_v1_2_static'] = row['source'] in reachable
        basename = row['path_or_pattern'].rsplit('/', 1)[-1]
        hints = [f"{w['source']}:{w['line']}" for w in writers if w['path_or_pattern'].rsplit('/', 1)[-1] == basename]
        row['producer_hints_not_execution_proof'] = ' | '.join(hints)
    # Keep the full metadata-only index private; publish only data-copy matches and source facts.
    private = ROOT / '.local/required_files_2026_11_02'; private.mkdir(exist_ok=True)
    (private / 'data_index.json').write_text(json.dumps([{k:v for k,v in entry.items() if k!='path'} for entry in index], indent=2)+'\n', encoding='utf-8')
    write_csv('source_dates.csv', sources); write_csv('dependency_calls.csv', dependencies); write_csv('producer_calls.csv', writers)
    write_csv('source_edges.csv', source_edges); write_csv('unresolved_reads.csv', unresolved)
    missing = [r for r in dependencies if r['availability'] in {'not_found_in_checked_scope','dynamic_family_unresolved_or_missing'}]
    write_csv('missing_references.csv', missing)
    summary = {'scope': 'Configured two source roots, existing checkout excluding private/runtime dirs, and ZIP directories under source roots; no whole-PC search',
               'source_count': len(sources), 'recent_src_count_by_scope': dict(Counter(s['source'].split(':')[0] for s in sources if s['is_src'] and s['modified_after_2021'])),
               'date_limit': 'mtime after 2021 is priority only, not historical creation/runtime proof; filesystem creation may reflect copying',
               'indexed_data_files': len(index), 'indexed_primary_data_files': sum(Path(e['basename']).suffix.lower() in PRIMARY_EXT for e in index),
               'zip_directories_checked': zip_reports, 'read_calls': len(dependencies), 'unresolved_reads_without_renderable_data_extension': len(unresolved), 'availability_counts': dict(Counter(d['availability'] for d in dependencies)),
               'statically_reachable_sources_from_main_v1_2': sorted(reachable), 'errors': errors,
               'limitations': ['No R execution or evaluation of control flow, function reachability, setwd or branch-dependent constants',
                               'No complete R dependency graph; package-internal reads and unrenderable filenames may be absent',
                               'Same basename is only a candidate; identity/schema/axes and actual run binding remain unverified',
                               'Missing means not found in this bounded scope, not absent everywhere on the PC',
                               'No separate upload artifacts were listed in this chat; no claim of inspecting unavailable attachments'],
               'model_or_reconstruction_runs': False}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: summary[k] for k in ['source_count','recent_src_count_by_scope','indexed_data_files','indexed_primary_data_files','read_calls','availability_counts','errors']}))


if __name__ == '__main__':
    collect()
