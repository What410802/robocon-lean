from .utils import knobs as _knobs

# Fail loudly on environment variables that look like training knobs but are
# not registered (renamed/deprecated); they would otherwise silently do nothing.
_knobs.validate()

# Cross-knob consistency: MjLab imports task packages through entry points and
# catches Exception while registering tasks, so an invalid recipe raised inside a
# task builder would degrade into a confusing "task not registered" KeyError
# later.  Abort the package import (SystemExit is not caught by that handler)
# before any config is built.
try:
    _config_problems = _knobs.validate_config()
except Exception as _exc:  # unparsable value -> same loud, pre-GPU failure
    raise SystemExit(
        "[knobs] invalid velocity-tracking configuration: "
        f"{type(_exc).__name__}: {_exc}"
    ) from _exc
if _config_problems:
    raise SystemExit(
        "[knobs] invalid velocity-tracking configuration:\n  - "
        + "\n  - ".join(_config_problems)
    )

from . import tasks as tasks  # noqa: E402  (import after the knob validation)
