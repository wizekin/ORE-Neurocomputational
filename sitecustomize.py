"""Opt-in-by-default startup hook for ORE repository script logging.

Set ORE_AUTO_LOG=0 to disable automatic tracing for a process.
"""
try:
    # Importing runtime installs one tracing session when argv targets this repo.
    import ore_logger.runtime  # noqa: F401
except Exception:
    # Logging is diagnostic; a failure must never prevent a script from starting.
    pass
