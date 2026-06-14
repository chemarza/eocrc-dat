from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any

from tap import Tap

from eocrc_dat.agency.commands import run_evaluate, run_export, run_infer, run_score, run_train
from eocrc_dat.agency.config import load_experiment
from eocrc_dat.agency.schema import ExperimentConfig
from eocrc_dat.support.io import write_json
from eocrc_dat.support.logging import get_logger

_LOG = get_logger("eocrc_dat.agency")
_VERBS = ("train", "evaluate", "infer", "score", "export")


class _Base(Tap):
    experiment: str = "main"
    config_dir: str = "configs"
    out: str | None = None


class TrainArgs(_Base):
    pass


class EvaluateArgs(_Base):
    interactions: bool = False
    compare: str | None = None


class InferArgs(_Base):
    describe_cohort: bool = False
    data: str = "synthetic"
    seed: int | None = None


class ScoreArgs(_Base):
    pass


class ExportArgs(_Base):
    pass


def _emit(payload: dict[str, Any], out: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    if out is not None:
        write_json(payload, out)


def _with_seed(cfg: ExperimentConfig, seed: int | None) -> ExperimentConfig:
    if seed is None:
        return cfg
    return dataclasses.replace(cfg, seed=seed, data=dataclasses.replace(cfg.data, seed=seed))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in _VERBS:
        sys.stderr.write(f"usage: eocrc-dat {{{'|'.join(_VERBS)}}} [options]\n")
        return 2
    verb, rest = arguments[0], arguments[1:]

    if verb == "train":
        targs = TrainArgs().parse_args(rest)
        cfg = load_experiment(targs.experiment, targs.config_dir)
        _emit(run_train(cfg), targs.out)
    elif verb == "evaluate":
        eargs = EvaluateArgs().parse_args(rest)
        cfg = load_experiment(eargs.experiment, eargs.config_dir)
        _emit(
            run_evaluate(cfg, with_interactions=eargs.interactions, compare=eargs.compare),
            eargs.out,
        )
    elif verb == "infer":
        iargs = InferArgs().parse_args(rest)
        cfg = _with_seed(load_experiment(iargs.experiment, iargs.config_dir), iargs.seed)
        _emit(run_infer(cfg), iargs.out)
    elif verb == "score":
        sargs = ScoreArgs().parse_args(rest)
        cfg = load_experiment(sargs.experiment, sargs.config_dir)
        _emit(run_score(cfg), sargs.out)
    else:
        xargs = ExportArgs().parse_args(rest)
        cfg = load_experiment(xargs.experiment, xargs.config_dir)
        _emit(run_export(cfg, xargs.out or "artifacts/model.pt"), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
