"""Every repo-root path expression in a script must resolve to the repo root.

Scripts locate the repository by walking up from ``__file__``. Moving a script
into a subfolder changes how far up the root is, and a stale depth does not
raise on import: it silently points at the wrong directory, so the script reads
missing inputs or writes outputs somewhere nobody looks. This test pins the
depth for every script at once.
"""
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENTS_CALL = re.compile(r'Path\(__file__\)(?:\.resolve\(\))?\.parents\[(\d+)\]')

PY_FILES = sorted(
    p for p in REPO_ROOT.rglob('*.py')
    if '__pycache__' not in p.parts and 'archive' not in p.parts
)


def _repo_root_uses():
    """Yield (file, line number, depth) for each parents[N] expression."""
    for path in PY_FILES:
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            for match in PARENTS_CALL.finditer(line):
                yield path, number, int(match.group(1))


USES = list(_repo_root_uses())


@pytest.mark.skipif(not USES, reason='no parents[N] expressions found')
@pytest.mark.parametrize(
    'path,line,depth',
    USES,
    ids=[f'{p.relative_to(REPO_ROOT)}:{n}' for p, n, _ in USES],
)
def test_parents_depth_reaches_repo_root(path, line, depth):
    resolved = path.resolve().parents[depth]
    expected = len(path.resolve().relative_to(REPO_ROOT).parts) - 1
    assert resolved == REPO_ROOT, (
        f'{path.relative_to(REPO_ROOT)}:{line} uses parents[{depth}], which resolves to '
        f'{resolved}. For this depth it should be parents[{expected}].'
    )
