"""enex_analysis package.

The shared thermodynamic/heat-pump physics models live in the vendored
``tmhp`` git submodule, which uses a ``src`` layout
(``tmhp/src/tmhp/*.py``).  Make that package importable both as the
top-level ``tmhp`` (matching the convention used by the docs/scripts) and
as the ``enex_analysis.tmhp`` subpackage, so internal modules can keep
using ``from .tmhp.X import ...`` after the constants/physics refactor.
"""

import sys as _sys
from pathlib import Path as _Path

_tmhp_src = _Path(__file__).resolve().parent / "tmhp" / "src"
if _tmhp_src.is_dir() and str(_tmhp_src) not in _sys.path:
    _sys.path.insert(0, str(_tmhp_src))

# Expose the embedded package as ``enex_analysis.tmhp`` so relative imports
# (``from .tmhp.constants import ...``) resolve to the vendored submodule.
import tmhp as _tmhp  # noqa: E402

_sys.modules[__name__ + ".tmhp"] = _tmhp
# Also bind as a real attribute so ``enex_analysis.tmhp`` resolves via
# attribute access (e.g. ``enex_analysis.tmhp.calc_util``), not only via
# the import machinery's sys.modules lookup.
tmhp = _tmhp

del _sys, _Path, _tmhp_src, _tmhp
