"""Verbatim copies of official MolecularIQ evaluation code.

Nothing in this package may be edited. See VENDOR.json for the upstream
repository, path and commit of every file, and `scripts/check_vendor.py`
for the integrity check that re-downloads upstream and diffs it.
"""

from .moleculariq_extractors import extract_moleculariq_answer
from .moleculariq_system_prompt import SYSTEM_PROMPT

__all__ = ["extract_moleculariq_answer", "SYSTEM_PROMPT"]
