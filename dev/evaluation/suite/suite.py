"""The suite definition: which topics, in which groups, and the one set of settings every topic is run with.

A suite lives in a folder OUTSIDE the repository (the harness is general; the topics and the models a person pays for
are theirs). ``suite.yaml`` there reads::

    rounds_dir: C:/fi_runs/suite/rounds      # one folder per round is made under it
    env_file: C:/dev/FrontierInsight/.env    # KEY=VALUE lines passed to every quest process (never printed)
    config_template: config_template.yaml    # the settings every topic gets (see below)
    replace_keys: [provider]                 # keys the template REPLACES; any other key is merged into the topic's
    groups:
      dev:     {topics: [a, b]}
      heldout: {acceptance_only: true, topics: [c]}   # runs only with --acceptance
    topics:
      a: topics/a.yaml                       # a normal FI config; its provider block is replaced by the template's

The config of a topic in a round is made from the topic file and the template every time a round starts, so the model
block of two topics (or two rounds) cannot drift apart. Only ``output.output_dir`` differs: ``<round>/<topic>/out``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class SuiteError(Exception):
    """A suite file or a request that cannot be used; the message says what to change."""


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # noqa: ANN401 -- the dumper's signature
        return True


def _str_representer(dumper: yaml.SafeDumper, value: str) -> Any:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_NoAliases.add_representer(str, _str_representer)


def dump_yaml(data: Any) -> str:
    return yaml.dump(data, Dumper=_NoAliases, sort_keys=False, allow_unicode=True, width=120)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SuiteError(f"{path} is not a YAML mapping")
    return value


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``extra`` merged into a copy of ``base``: mappings merge key by key, anything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Suite:
    """A loaded ``suite.yaml``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.root = self.path.parent
        raw = _read_yaml(self.path)
        self.raw = raw
        topics = raw.get("topics")
        groups = raw.get("groups")
        if not isinstance(topics, dict) or not topics:
            raise SuiteError(f"{self.path}: `topics` must map a topic name to its config file")
        if not isinstance(groups, dict) or not groups:
            raise SuiteError(f"{self.path}: `groups` must list at least one group")
        self.topic_files: dict[str, Path] = {str(k): self._rel(v) for k, v in topics.items()}
        self.groups: dict[str, dict[str, Any]] = {}
        for name, spec in groups.items():
            spec = spec if isinstance(spec, dict) else {"topics": spec}
            names = [str(t) for t in (spec.get("topics") or [])]
            missing = [t for t in names if t not in self.topic_files]
            if missing:
                raise SuiteError(f"{self.path}: group `{name}` names topics that are not defined: {', '.join(missing)}")
            self.groups[str(name)] = {"topics": names, "acceptance_only": bool(spec.get("acceptance_only"))}
        self.rounds_dir = self._rel(raw.get("rounds_dir") or "rounds")
        self.env_file = self._rel(raw["env_file"]) if raw.get("env_file") else None
        self.template_file = self._rel(raw["config_template"]) if raw.get("config_template") else None
        self.replace_keys = [str(k) for k in (raw.get("replace_keys") or ["provider"])]

    def _rel(self, value: Any) -> Path:
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def group_topics(self, group: str) -> list[str]:
        if group not in self.groups:
            raise SuiteError(f"no group `{group}` in {self.path}; the groups are: {', '.join(self.groups)}")
        return list(self.groups[group]["topics"])

    def require_allowed(self, group: str, acceptance: bool) -> None:
        """A group marked ``acceptance_only`` (the held-out topics) is not run unless the caller asked for the acceptance run."""
        self.group_topics(group)
        if self.groups[group]["acceptance_only"] and not acceptance:
            raise SuiteError(f"group `{group}` is held back for the acceptance run; it is not run without --acceptance")

    def topic_text(self, topic: str) -> str:
        """The topic's own ``topic:`` block (what the grader needs for the goals and reference values)."""
        return str(_read_yaml(self.topic_files[topic]).get("topic") or "")

    def make_config(self, topic: str, out_dir: Path) -> dict[str, Any]:
        """The topic's config for one round: its own file, the template's settings on top, and its output folder."""
        if topic not in self.topic_files:
            raise SuiteError(f"no topic `{topic}` in {self.path}")
        cfg = _read_yaml(self.topic_files[topic])
        if self.template_file is not None:
            template = _read_yaml(self.template_file)
            for key, value in template.items():
                if key in self.replace_keys or not isinstance(cfg.get(key), dict) or not isinstance(value, dict):
                    cfg[key] = copy.deepcopy(value)
                else:
                    cfg[key] = _merge(cfg[key], value)
        out = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
        out["output_dir"] = str(out_dir).replace("\\", "/")
        cfg["output"] = out
        return cfg

    def write_configs(self, group: str, round_dir: Path) -> dict[str, Path]:
        """One config per topic of the group under ``round_dir/<topic>/config.yaml``; returns topic -> path."""
        written: dict[str, Path] = {}
        for topic in self.group_topics(group):
            folder = round_dir / topic
            folder.mkdir(parents=True, exist_ok=True)
            cfg = self.make_config(topic, folder / "out")
            path = folder / "config.yaml"
            path.write_text(dump_yaml(cfg), encoding="utf-8")
            written[topic] = path
        return written
