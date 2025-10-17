import subprocess, sys
def _git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""
def get_version_info():
    return {"git_commit": _git_commit(), "py_ver": sys.version.split()[0], "np_ver": _numpy_ver()}
def _numpy_ver():
    try:
        import numpy as np
        return str(np.__version__)
    except Exception:
        return ""
