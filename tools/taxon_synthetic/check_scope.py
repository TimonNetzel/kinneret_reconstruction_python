"""Check unchanged base blobs, allowed paths, worktrees and documentation links."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "docs/thesis_research/t1_synthetic_t2_2026_10_31"
BASE = "d003fb97a21c967a44161230eed807e745ed5b0b"
ALLOWED = {"docs/thesis_research/HANDOFF.md", "docs/thesis_research/PROJECT_STATUS.md"}
PREFIXES = ["docs/thesis_research/t1_synthetic_t2_2026_10_31/", "tools/taxon_synthetic/"]
NEW_TEST = "tests/test_synthetic_inversion.py"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    unchanged = []
    symlinks = []
    for row in git("ls-tree", "-rz", BASE).split(b"\0"):
        if not row:
            continue
        info, name = row.split(b"\t", 1)
        name = name.decode()
        mode, kind, expected = info.decode().split()
        if name in ALLOWED:
            continue
        assert kind == "blob", name
        path = ROOT / name
        if mode == "120000":
            assert path.is_symlink(), name
            payload = os.readlink(path).encode()
            symlinks.append(name)
        else:
            assert not path.is_symlink(), name
            payload = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        assert actual == expected, name
        unchanged.append(name)
    changed = set(git("diff", "--name-only", BASE).decode().splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    for name in changed:
        assert name in ALLOWED or name == NEW_TEST or any(name.startswith(p) for p in PREFIXES), name
    paths = ["kinneret_reconstruction_python", "netzel-original-names", "netzel-pilot-roadmap",
             "netzel-pollen", "netzel-workflow"]
    statuses = {str(ROOT.parent / p): subprocess.check_output(
        ["git", "-C", str(ROOT.parent / p), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True) for p in paths}
    assert not any(statuses.values()), statuses
    record = {"base_commit": BASE, "status": "passed", "existing_files_byte_identical": len(unchanged),
              "unchanged_symlinks": symlinks, "allowed_existing_document_changes": sorted(ALLOWED),
              "original_worktree_statuses": statuses, "new_paths_allowed": PREFIXES + [NEW_TEST],
              "git_blobs_sha1_compared": True,
              "note": "Existing model/result/source/count/CLI/app files unchanged; no reference hashes replaced."}
    (PACKAGE / "preservation.json").write_text(json.dumps(record, indent=2) + "\n")
    checks = []
    for p in sorted([ROOT / x for x in ALLOWED] + list(PACKAGE.glob("*.md"))):
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            assert (p.parent / target.split("#")[0]).resolve().exists(), (p, target)
            checks.append([str(p.relative_to(ROOT)), target])
    (PACKAGE / "documentation_verification.json").write_text(json.dumps(
        {"status": "passed", "local_links_checked": len(checks), "links": checks}, indent=2) + "\n")
    print(f"Preserved {len(unchanged)} existing blobs including {len(symlinks)} symlinks; "
          f"five original worktrees clean; {len(checks)} local links valid; diff scope passed")


if __name__ == "__main__":
    main()
