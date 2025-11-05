# scripts/resolve_runner.py
import importlib, sys
candidates = [
  "runner.cli",
  "project.runner.cli",
  "src.runner.cli",
  "decum.runner.cli",
  "app.runner.cli",
]
for m in candidates:
  try:
    importlib.import_module(m)
    print(m); sys.exit(0)
  except Exception:
    pass
print("NO_MODULE", file=sys.stderr); sys.exit(1)
