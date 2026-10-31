"""Verify isolation and documentation of the new synthetic counts package."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "docs/thesis_research/counts_synthetic_t2_2026_10_31"
BASE = "19a75426320d0143c09cd00c756bc31f97cef0d0"
ALLOWED = {"docs/thesis_research/HANDOFF.md", "docs/thesis_research/PROJECT_STATUS.md"}
PREFIXES = ["tools/taxon_counts/", "docs/thesis_research/counts_synthetic_t2_2026_10_31/"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    protected, symlinks = [], []
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
        protected.append(name)
    changed = set(git("diff", "--name-only", BASE).decode().splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    for name in changed:
        assert name in ALLOWED or name == "tests/test_synthetic_counts.py" or any(name.startswith(p) for p in PREFIXES), name
    paths = [r[9:] for r in git("worktree", "list", "--porcelain").decode().splitlines()
             if r.startswith("worktree ") and r[9:] != str(ROOT)]
    statuses = {p: subprocess.check_output(["git", "-C", p, "status", "--porcelain=v1", "--untracked-files=all"], text=True) for p in paths}
    assert len(statuses) == 7 and not any(statuses.values()), statuses
    links = []
    for p in sorted([ROOT / x for x in ALLOWED] + list(PACKAGE.glob("*.md"))):
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            dest = (p.parent / target.split("#")[0]).resolve()
            assert dest == PACKAGE / "scope_verification.json" or dest.exists(), (p, target)
            links.append([str(p.relative_to(ROOT)), target])
    result = {"status": "passed", "base_commit": BASE, "protected_blobs_unchanged": len(protected),
              "symlinks_unchanged": symlinks, "allowed_existing_changes": sorted(ALLOWED),
              "original_worktree_statuses": statuses, "local_links_checked": len(links), "links": links}
    (PACKAGE / "scope_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(protected)} protected blobs unchanged; seven old worktrees clean; {len(links)} links passed")


if __name__ == "__main__":
    main()
