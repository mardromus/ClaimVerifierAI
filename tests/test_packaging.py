"""Guard against package modules that exist locally but are never committed (e.g. hidden by .gitignore)."""

import importlib
import pkgutil
import shutil
import subprocess
from pathlib import Path

import pytest

import claimverifier

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("torch", "transformers", "sentence_transformers", "fastapi")


def test_every_module_imports():
    for mod in pkgutil.walk_packages(claimverifier.__path__, "claimverifier."):
        if mod.name.endswith("__main__"):  # runs the CLI on import
            continue
        try:
            importlib.import_module(mod.name)
        except ImportError as e:  # optional heavy dependencies are allowed to be missing
            if not any(name in str(e) for name in OPTIONAL):
                raise


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout")
def test_package_sources_are_not_gitignored():
    sources = [str(p.relative_to(ROOT)) for p in (ROOT / "claimverifier").rglob("*.py")
               if "__pycache__" not in p.parts]
    result = subprocess.run(["git", "check-ignore", "--no-index", *sources], cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip() == "", f"package files ignored by .gitignore:\n{result.stdout}"
