"""LLM prompt templates loaded by the engine at runtime.

This module is empty by design — its only purpose is to make
``agents/`` a Python package so the .md prompt templates ship inside
the wheel at ``<site-packages>/agents/*.md``. The runtime path
resolution in ``core/engine.py`` (``Path(__file__).parent.parent /
"agents"``) then works for both source-tree and wheel installs.
"""
