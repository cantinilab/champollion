import sys
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

ROOT_TUTORIALS = ROOT / "tutorials"
DOCS_TUTORIALS = DOCS_SOURCE / "tutorials" / "notebooks"
if ROOT_TUTORIALS.exists():
    if DOCS_TUTORIALS.is_symlink():
        DOCS_TUTORIALS.unlink()
    shutil.copytree(ROOT_TUTORIALS, DOCS_TUTORIALS, dirs_exist_ok=True)

project = "Champollion"
author = "Jules Samaran"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Champollion"

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
