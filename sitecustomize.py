"""Opt-in-by-default startup hook for ORE repository script logging.

Set ORE_AUTO_LOG=0 to disable automatic tracing for a process.
"""
try:
    from ore_logger.runtime import install_for_current_script

    install_for_current_script()
except Exception:
    # Logging is diagnostic; a failure must never prevent a script from starting.
    pass
