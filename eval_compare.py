"""
Transformer vs other models: config-driven benchmarking + re-runnable visualization.

This script is intentionally self-contained:
- It reads `eval_config.yaml` (see repo root) describing envs/roles/algorithms/seeds/etc.
- It runs training+evaluation for a set of *unique* experiment variants per environment.
- It saves all raw results under `save_data/`, plus one run-level `metadata.json`.
- It can be re-run in visualization-only mode to regenerate tables/plots/report from saved data.

Examples:
  # Run benchmark
  python eval_compare.py --mode run --config eval_config.yaml

  # Re-run visualization from a previous run directory
  python eval_compare.py --mode viz --run_dir save_data/transformer_eval/20260114_120102
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("save_data") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch
from loguru import logger

from env.env_core import EconomicSociety
from main import select_agent, setup_government_agents
from runner import Runner
from utils.config import load_config
from utils.seeds import set_seeds


# IMPORTANT: marl-macro-modeling uses loguru too and calls `logger.remove(0)` on import.
# If we remove handler 0 ourselves, their import crashes.
# We therefore install a no-op handler as id=0 (removable), then our real stderr handler as id=1.
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logger.configure(
    handlers=[
        {"sink": lambda _m: None, "level": _LOG_LEVEL},  # id=0 (safe for marl to remove)
        {"sink": sys.stderr, "level": _LOG_LEVEL},       # id=1
    ]
)


ROLE_TO_TRAINER_KEY: dict[str, str] = {
    "households": "house_alg",
    "market": "firm_alg",
    "bank": "bank_alg",
    "government": "gov_alg",
    # government sub-roles
    "tax": "tax_gov_alg",
    "central_bank": "central_bank_gov_alg",
    "pension": "pension_gov_alg",
}


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def _scalarize(x: Any) -> float:
    """Convert metric value to a scalar float for plotting/table."""
    if x is None:
        return float("nan")
    try:
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_eval_config(config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(config_path))
    return OmegaConf.to_container(cfg, resolve=True)  # plain python dict


def _env_cfg(eval_cfg: dict[str, Any], env_name: str) -> dict[str, Any]:
    envs_cfg = (eval_cfg.get("environments") or {})
    if env_name not in envs_cfg:
        raise KeyError(f"Environment '{env_name}' missing in eval_config under environments.*")
    return dict(envs_cfg[env_name])


def _trainer_overrides_from_role_algs(role_algs: dict[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for role, alg in role_algs.items():
        if role not in ROLE_TO_TRAINER_KEY:
            raise KeyError(f"Unknown role '{role}'. Expected one of: {sorted(ROLE_TO_TRAINER_KEY)}")
        overrides[ROLE_TO_TRAINER_KEY[role]] = alg
    return overrides


def _variant_signature(env_name: str, role_algs: dict[str, str]) -> tuple:
    return (env_name, tuple(sorted(role_algs.items())))


def _build_canonical_variants_for_env(env_name: str, env_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    For each transformer-target role, create a set of variants that differ ONLY in that role's algorithm.
    This guarantees the 'PPO vs SAC vs transformer' style comparisons for tables/plots.
    """
    transformer_roles: list[str] = list(env_cfg.get("transformer_roles", []))
    role_algorithms: dict[str, list[str]] = dict(env_cfg.get("role_algorithms", {}))
    defaults: dict[str, str] = dict(env_cfg.get("default_role_algs", {}))

    variants: list[dict[str, Any]] = []
    for target_role in transformer_roles:
        algs = list(role_algorithms.get(target_role, []) or [])
        if not algs:
            continue
        # Ensure transformer is always included early (important for truncated runs).
        preferred = list(
            env_cfg.get(
                "canonical_alg_order",
                [
                    "transformer",
                    "ppo",
                    "sac",
                    "ddpg",
                    "rule_based",
                    "bc",
                    "llm",
                    "data_based",
                    "real",
                    "saez",
                    "us_federal",
                ],
            )
            or []
        )
        algs_ordered = [a for a in preferred if a in set(algs)]
        algs_ordered += [a for a in algs if a not in set(algs_ordered)]

        for alg in algs_ordered:
            ra = dict(defaults)
            ra[target_role] = alg
            variant_id = f"target={target_role}__alg={alg}"
            variants.append(
                {
                    "env": env_name,
                    "variant_id": variant_id,
                    "role_algs": ra,
                    "target_role": target_role,
                    "target_alg": alg,
                    "kind": "canonical",
                }
            )
    return variants


def _train_runner(runner: Runner, *, n_epochs: int, epoch_length: int) -> None:
    """Train for n_epochs x epoch_length steps (no intermediate eval)."""
    obs_dict = runner.envs.reset()

    for _epoch in range(n_epochs):
        transition_dict = {"obs_dict": [], "action_dict": [], "reward_dict": [], "next_obs_dict": [], "done": []}
        on_policy_process = all(runner.envs.recursive_decompose_dict(runner.agents_policy, lambda a: a.on_policy))

        for t in range(epoch_length):
            obs_dict_tensor = runner._get_tensor_inputs(obs_dict)
            action_dict, processed_actions_dict = runner.agents_get_action(obs_dict_tensor)
            next_obs_dict, reward_dict, done = runner.envs.step(processed_actions_dict, t)
            runner.agents_observe(reward_dict, done)

            if on_policy_process:
                transition_dict["obs_dict"].append(obs_dict)
                transition_dict["action_dict"].append(action_dict)
                transition_dict["reward_dict"].append(reward_dict)
                transition_dict["next_obs_dict"].append(next_obs_dict)
                transition_dict["done"].append(done)
            else:
                runner.buffer.add(
                    {
                        "obs_dict": obs_dict,
                        "action_dict": action_dict,
                        "reward_dict": reward_dict,
                        "next_obs_dict": next_obs_dict,
                        "done": done,
                    }
                )

            obs_dict = next_obs_dict
            if done:
                obs_dict = runner.envs.reset()

        for agent_name in runner.agents_policy:
            sub_agent_policy = runner.agents_policy[agent_name]
            batch_size = epoch_length if on_policy_process else int(getattr(runner.args, "batch_size", 64))

            if on_policy_process:
                agent_data = runner.buffer.sample(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    batch_size=batch_size,
                    on_policy=True,
                    transition_dict=transition_dict,
                )
            else:
                agent_data = runner.buffer.sample(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    batch_size=batch_size,
                    on_policy=False,
                )

            if isinstance(sub_agent_policy, dict):
                for name in sub_agent_policy:
                    runner.sub_agent_training(
                        agent_name=name,
                        agent_policy=sub_agent_policy[name],
                        transitions=agent_data[name],
                        loss={"actor_loss": {}, "critic_loss": {}},
                    )
            else:
                runner.sub_agent_training(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    transitions=agent_data,
                    loss={"actor_loss": {}, "critic_loss": {}},
                )


def _merge_episode_series(episodes: list[list[float]]) -> list[float]:
    if not episodes:
        return []
    # Flatten and ensure all elements are scalars
    normalized_episodes = []
    for s in episodes:
        # Ensure s is a flat list of floats
        flat_list = []
        for item in s:
            try:
                if isinstance(item, (list, tuple, np.ndarray)):
                    # If item is an array/list, take the mean to scalarize
                    arr = np.asarray(item, dtype=np.float32)
                    flat_list.append(float(np.mean(arr)) if arr.size > 0 else float("nan"))
                elif isinstance(item, (int, float, np.floating, np.integer)):
                    flat_list.append(float(item))
                else:
                    # Try to convert to float, fallback to nan
                    flat_list.append(float(item) if item is not None else float("nan"))
            except Exception:
                flat_list.append(float("nan"))
        normalized_episodes.append(flat_list)
    
    if not normalized_episodes:
        return []
    
    max_len = max(len(x) for x in normalized_episodes) if normalized_episodes else 0
    if max_len == 0:
        return []
    mat = []
    for s in normalized_episodes:
        try:
            # Ensure s is a 1D array of floats
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim > 1:
                # If somehow multi-dimensional, flatten it
                arr = arr.flatten()
            if arr.shape[0] < max_len:
                arr = np.pad(arr, (0, max_len - arr.shape[0]), constant_values=np.nan)
            mat.append(arr)
        except Exception as e:
            # If conversion fails, create a NaN-filled array of the right length
            logger.warning(f"Failed to convert episode series to array: {e}, using NaN padding")
            mat.append(np.full(max_len, np.nan, dtype=np.float32))
    
    if not mat:
        return []
    
    mat = np.stack(mat, axis=0)  # [E, T]
    return np.nanmean(mat, axis=0).tolist()


def _evaluate_timeseries(runner: Runner, *, metrics: list[str], eval_episodes: int) -> dict[str, list[float]]:
    """
    Evaluate and return per-step time series for requested metrics.
    Each element of the list corresponds to one environment step (like viz/data/*.json).
    """
    metrics = list(dict.fromkeys(metrics))  # de-dup keep order
    per_metric_episodes: dict[str, list[list[float]]] = {m: [] for m in metrics}

    for _ in range(int(eval_episodes)):
        obs_dict = runner.eval_env.reset()
        t = 0

        ep_series: dict[str, list[float]] = {m: [] for m in metrics}
        while True:
            with torch.no_grad():  # type: ignore[name-defined]
                obs_dict_tensor = runner._get_tensor_inputs(obs_dict)
                _raw, processed_actions_dict = runner.agents_get_action(obs_dict_tensor)
                next_obs_dict, rewards_dict, done = runner.eval_env.step(processed_actions_dict, t)
                runner.agents_observe(rewards_dict, done)
            t += 1

            runner.init_economic_dict(rewards_dict)
            for m in metrics:
                if m in runner.econ_dict:
                    ep_series[m].append(_scalarize(runner.econ_dict[m]))
                else:
                    ep_series[m].append(float("nan"))

            obs_dict = next_obs_dict
            if done:
                break

        for m in metrics:
            per_metric_episodes[m].append(ep_series[m])

    return {m: _merge_episode_series(per_metric_episodes[m]) for m in metrics}


def _sample_additional_variants_for_env(
    *,
    env_name: str,
    env_cfg: dict[str, Any],
    already: list[dict[str, Any]],
    max_variants: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Optionally add more unique variants by sampling combinations of role algorithms.
    Constraint: roles listed under transformer_roles are forced to 'transformer' for these extra variants
    (keeps the focus on transformer while still varying the rest).
    """
    role_algorithms: dict[str, list[str]] = dict(env_cfg.get("role_algorithms", {}))
    defaults: dict[str, str] = dict(env_cfg.get("default_role_algs", {}))
    transformer_roles: set[str] = set(env_cfg.get("transformer_roles", []))

    existing = { _variant_signature(env_name, v["role_algs"]) for v in already }
    variants: list[dict[str, Any]] = []

    roles = sorted({*defaults.keys(), *role_algorithms.keys()})
    if not roles:
        return variants

    # If the requested max is larger than the number of unique combos possible, avoid infinite loops.
    # We estimate an upper bound on unique combinations (with transformer_roles forced to "transformer").
    max_unique = 1
    for role in roles:
        algs = role_algorithms.get(role)
        if not algs:
            continue
        if role in transformer_roles:
            max_unique *= 1
        else:
            max_unique *= max(1, len(algs))

    remaining_target = max(0, max_variants - len(already))
    # Hard cap attempts so we never hang.
    max_attempts = max(1000, remaining_target * 500)
    attempts = 0

    while len(already) + len(variants) < max_variants and attempts < max_attempts:
        attempts += 1
        ra = dict(defaults)
        for role in roles:
            algs = role_algorithms.get(role)
            if not algs:
                continue
            if role in transformer_roles:
                ra[role] = "transformer"
            else:
                ra[role] = rng.choice(list(algs))

        sig = _variant_signature(env_name, ra)
        if sig in existing:
            continue
        existing.add(sig)
        variant_id = "sampled__" + "__".join([f"{k}={v}" for k, v in sorted(ra.items())])
        variants.append(
            {
                "env": env_name,
                "variant_id": variant_id,
                "role_algs": ra,
                "target_role": None,
                "target_alg": None,
                "kind": "sampled",
            }
        )

    if len(already) + len(variants) < max_variants:
        logger.warning(
            "Env {}: requested max_launches_per_env={} but could only sample {} extra unique variants "
            "(already={}, attempts={}, max_unique_estimate={}).",
            env_name,
            max_variants,
            len(variants),
            len(already),
            attempts,
            max_unique,
        )
    return variants


def build_variants(eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    env_list: list[str] = list(eval_cfg.get("environments_to_benchmark", []))
    if not env_list:
        raise ValueError("eval_config.yaml: environments_to_benchmark must be a non-empty list")

    max_launches_default = int(eval_cfg.get("max_launches_per_env", 0) or 0)
    rng = random.Random(int(eval_cfg.get("sampling_seed", 0) or 0))

    all_variants: list[dict[str, Any]] = []
    envs_cfg = (eval_cfg.get("environments") or {})
    for env_name in env_list:
        # Skip environments that are not configured (e.g., commented out)
        if env_name not in envs_cfg:
            logger.warning(f"Skipping environment '{env_name}' - not found in environments config (may be commented out)")
            continue
        
        cfg_e = _env_cfg(eval_cfg, env_name)
        max_variants = int(cfg_e.get("max_launches_per_env", max_launches_default) or 0)
        if max_variants <= 0:
            max_variants = 1

        canonical = _build_canonical_variants_for_env(env_name, cfg_e)

        # De-dup canonical while preserving order
        canonical_unique: list[dict[str, Any]] = []
        seen = set()
        for v in canonical:
            sig = _variant_signature(env_name, v["role_algs"])
            if sig in seen:
                continue
            seen.add(sig)
            canonical_unique.append(v)

        # Select canonical variants in a round-robin across transformer_roles so that
        # `max_launches_per_env` doesn't accidentally drop later roles (e.g., bank) entirely.
        transformer_roles: list[str] = list(cfg_e.get("transformer_roles", []) or [])
        per_role: dict[str, list[dict[str, Any]]] = {r: [] for r in transformer_roles}
        for v in canonical_unique:
            tr = v.get("target_role")
            if tr in per_role:
                per_role[str(tr)].append(v)

        base: list[dict[str, Any]] = []
        if transformer_roles:
            i = 0
            while len(base) < max_variants and any(per_role[r] for r in transformer_roles):
                role = transformer_roles[i % len(transformer_roles)]
                if per_role[role]:
                    base.append(per_role[role].pop(0))
                i += 1
        else:
            base = canonical_unique[:max_variants]

        extra = []
        if len(base) < max_variants:
            extra = _sample_additional_variants_for_env(
                env_name=env_name, env_cfg=cfg_e, already=base, max_variants=max_variants, rng=rng
            )

        all_variants.extend(base + extra)

    # Global uniqueness within the run
    uniq: list[dict[str, Any]] = []
    seen_all = set()
    for v in all_variants:
        sig = _variant_signature(v["env"], v["role_algs"])
        if sig in seen_all:
            continue
        seen_all.add(sig)
        uniq.append(v)
    return uniq


def _train_runner_collect_metrics(
    runner: Runner,
    *,
    n_epochs: int,
    epoch_length: int,
    eval_every: int,
    tracked_metric_keys: set[str],
) -> list[dict[str, Any]]:
    """
    Train for n_epochs and evaluate every `eval_every` epochs (including epoch 0).
    Returns a list of dicts: {"epoch": int, "metrics": {...subset...}}.
    """
    obs_dict = runner.envs.reset()
    history: list[dict[str, Any]] = []
    # Prevent Runner._evaluate_agent from writing to viz/data during benchmarking.
    # (Runner may force write_evaluate_data=True if it thinks a "best" eval happened.)
    # Use very large integers (Runner casts to int internally).
    runner.eva_year_indicator = 10**18
    runner.eva_reward_indicator = 10**18

    for epoch in range(n_epochs):
        transition_dict = {"obs_dict": [], "action_dict": [], "reward_dict": [], "next_obs_dict": [], "done": []}

        # Determine on/off-policy once per epoch (agent types don't change)
        on_policy_process = all(runner.envs.recursive_decompose_dict(runner.agents_policy, lambda a: a.on_policy))

        for t in range(epoch_length):
            obs_dict_tensor = runner._get_tensor_inputs(obs_dict)
            action_dict, processed_actions_dict = runner.agents_get_action(obs_dict_tensor)
            next_obs_dict, reward_dict, done = runner.envs.step(processed_actions_dict, t)
            runner.agents_observe(reward_dict, done)

            if on_policy_process:
                transition_dict["obs_dict"].append(obs_dict)
                transition_dict["action_dict"].append(action_dict)
                transition_dict["reward_dict"].append(reward_dict)
                transition_dict["next_obs_dict"].append(next_obs_dict)
                transition_dict["done"].append(done)
            else:
                runner.buffer.add(
                    {
                        "obs_dict": obs_dict,
                        "action_dict": action_dict,
                        "reward_dict": reward_dict,
                        "next_obs_dict": next_obs_dict,
                        "done": done,
                    }
                )

            obs_dict = next_obs_dict
            if done:
                obs_dict = runner.envs.reset()

        # Train each agent (or sub-agent)
        for agent_name in runner.agents_policy:
            sub_agent_policy = runner.agents_policy[agent_name]
            batch_size = epoch_length if on_policy_process else int(getattr(runner.args, "batch_size", 64))

            if on_policy_process:
                agent_data = runner.buffer.sample(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    batch_size=batch_size,
                    on_policy=True,
                    transition_dict=transition_dict,
                )
            else:
                agent_data = runner.buffer.sample(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    batch_size=batch_size,
                    on_policy=False,
                )

            if isinstance(sub_agent_policy, dict):
                for name in sub_agent_policy:
                    runner.sub_agent_training(
                        agent_name=name,
                        agent_policy=sub_agent_policy[name],
                        transitions=agent_data[name],
                        loss={"actor_loss": {}, "critic_loss": {}},
                    )
            else:
                runner.sub_agent_training(
                    agent_name=agent_name,
                    agent_policy=sub_agent_policy,
                    transitions=agent_data,
                    loss={"actor_loss": {}, "critic_loss": {}},
                )

        if epoch % max(1, eval_every) == 0:
            m = runner._evaluate_agent(write_evaluate_data=False)
            m = dict(m)
            subset = {k: _scalarize(m.get(k)) for k in tracked_metric_keys if k in m}
            history.append({"epoch": int(epoch), "metrics": subset})

    return history


def run_benchmark(eval_cfg: dict[str, Any], *, config_path: Path) -> Path:
    run_id = _now_id()
    run_dir = Path("save_data") / "transformer_eval" / run_id
    _safe_mkdir(run_dir)
    logger.add(str(run_dir / "run.log"), level=os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("Starting benchmark run_id={} run_dir={}", run_id, run_dir)

    seeds: list[int] = [int(s) for s in (eval_cfg.get("seeds") or [])]
    if not seeds:
        raise ValueError("eval_config.yaml: seeds must be a non-empty list")

    n_epochs = int(eval_cfg.get("n_epochs", 0) or 0)
    epoch_length = int(eval_cfg.get("epoch_length", 0) or 0)
    if n_epochs <= 0 or epoch_length <= 0:
        raise ValueError("eval_config.yaml: n_epochs and epoch_length must be positive integers")

    eval_every = int(eval_cfg.get("eval_every", 1) or 1)
    wandb_flag = bool(eval_cfg.get("wandb", False))
    cuda_flag = bool(eval_cfg.get("cuda", False))

    variants = build_variants(eval_cfg)
    env_list: list[str] = list(eval_cfg.get("environments_to_benchmark", []))
    logger.info(
        "Config: envs={} variants={} seeds={} n_epochs={} epoch_length={} wandb={} cuda={}",
        len(env_list),
        len(variants),
        len(seeds),
        n_epochs,
        epoch_length,
        wandb_flag,
        cuda_flag,
    )

    # Run-level metadata (single file)
    metadata = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "eval_config": eval_cfg,
        "resolved": {
            "n_epochs": n_epochs,
            "epoch_length": epoch_length,
            "eval_every": eval_every,
            "seeds": seeds,
            "wandb": wandb_flag,
            "cuda": cuda_flag,
        },
        "variants": variants,
    }
    _json_dump(run_dir / "metadata.json", metadata)

    total_jobs = len(variants) * len(seeds)
    job_i = 0

    for v in variants:
        env_name = v["env"]
        env_cfg = _env_cfg(eval_cfg, env_name)
        logger.info(
            "Env {} | variant {}/{}: {}",
            env_name,
            sum(1 for _v in variants if _v["env"] == env_name and _v["variant_id"] <= v["variant_id"]),  # best-effort
            sum(1 for _v in variants if _v["env"] == env_name),
            v["variant_id"],
        )

        tracked = set(env_cfg.get("metrics_to_plot", []) or [])
        # Table-friendly defaults
        tracked |= {"gov_reward", "house_reward", "bank_reward", "social_welfare", "GDP", "house_income"}

        role_algs = dict(v["role_algs"])
        trainer_overrides = _trainer_overrides_from_role_algs(role_algs)
        trainer_overrides.update(
            {
                "n_epochs": n_epochs,
                "epoch_length": epoch_length,
                "wandb": wandb_flag,
                "cuda": cuda_flag,
                "test": False,
                # Avoid writing lots of model checkpoints during benchmarking
                "save_interval": n_epochs + 1,
                # Keep evaluation frequent for dynamics
                "display_interval": 1,
            }
        )

        for seed in seeds:
            job_i += 1
            t0 = time.time()
            row_meta = {
                "ts_start": datetime.now().isoformat(timespec="seconds"),
                "env": env_name,
                "variant_id": v["variant_id"],
                "kind": v.get("kind"),
                "target_role": v.get("target_role"),
                "target_alg": v.get("target_alg"),
                "seed": int(seed),
                "role_algs": role_algs,
                "trainer_overrides": trainer_overrides,
            }

            out_path = run_dir / env_name / v["variant_id"] / f"seed_{seed}.json"
            try:
                logger.info(
                    "[{}/{}] Run env={} variant={} seed={} role_algs={}",
                    job_i,
                    total_jobs,
                    env_name,
                    v["variant_id"],
                    seed,
                    role_algs,
                )
                cfg = load_config(env_name)
                cfg = deepcopy(cfg)

                for k, vv in trainer_overrides.items():
                    cfg.Trainer[k] = vv
                cfg.Trainer.seed = int(seed)

                set_seeds(cfg.Trainer.seed, cuda=cfg.Trainer.cuda)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_num)

                env = EconomicSociety(cfg.Environment)

                house_agent = select_agent(cfg.Trainer.house_alg, "households", env.households.type, env, cfg.Trainer)
                firm_agent = select_agent(cfg.Trainer.firm_alg, "market", env.market.type, env, cfg.Trainer)
                bank_agent = select_agent(cfg.Trainer.bank_alg, "bank", env.bank.type, env, cfg.Trainer)
                gov_agents = setup_government_agents(cfg, env)

                runner = Runner(
                    env,
                    cfg.Trainer,
                    house_agent=house_agent,
                    government_agent=gov_agents,
                    firm_agent=firm_agent,
                    bank_agent=bank_agent,
                )

                _train_runner(runner, n_epochs=n_epochs, epoch_length=epoch_length)

                # Avoid Runner writing viz/data even if it considers something "best".
                runner.eva_year_indicator = 10**18
                runner.eva_reward_indicator = 10**18

                final_metrics = dict(runner._evaluate_agent(write_evaluate_data=False))
                row_meta["final_metrics"] = {k: _scalarize(v) for k, v in final_metrics.items()}

                metrics_to_plot = list(env_cfg.get("metrics_to_plot", []) or [])
                eval_eps = int(getattr(cfg.Trainer, "eval_episodes", 1) or 1)
                ts = _evaluate_timeseries(
                    runner, metrics=metrics_to_plot, eval_episodes=eval_eps
                )
                # Explicit year axis from the generated JSON arrays (1..T).
                # Each element in `timeseries[*]` is one environment step (= one year).
                any_metric = next(iter(ts.keys()), None)
                T = len(ts[any_metric]) if any_metric is not None else 0
                row_meta["timeseries_years"] = list(range(1, T + 1))
                row_meta["timeseries"] = ts
                row_meta["ts_end"] = datetime.now().isoformat(timespec="seconds")
                _json_dump(out_path, row_meta)
                logger.info(
                    "[{}/{}] Done env={} variant={} seed={} saved={} duration_s={:.1f}",
                    job_i,
                    total_jobs,
                    env_name,
                    v["variant_id"],
                    seed,
                    out_path,
                    time.time() - t0,
                )
            except Exception as e:
                row_meta["error"] = str(e)
                row_meta["ts_end"] = datetime.now().isoformat(timespec="seconds")
                _json_dump(out_path, row_meta)
                logger.exception(
                    "[{}/{}] FAILED env={} variant={} seed={} saved={} duration_s={:.1f}",
                    job_i,
                    total_jobs,
                    env_name,
                    v["variant_id"],
                    seed,
                    out_path,
                    time.time() - t0,
                )

    logger.info("Benchmark complete. run_dir={}", run_dir)
    return run_dir


def _find_latest_run_dir() -> Path | None:
    base = Path("save_data") / "transformer_eval"
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    return candidates[0] if candidates else None


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for p in run_dir.rglob("seed_*.json"):
        try:
            results.append(_json_load(p))
        except Exception:
            continue
    return results


def _pretty_metric_name(k: str) -> str:
    mapping = {
        "gov_reward": "Acc. rew.",
        "bank_reward": "Acc. rew.",
        "house_reward": "Acc. rew.",
        "social_welfare": "SW",
        "GDP": "GDP",
        "house_income": "Income",
    }
    return mapping.get(k, k)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without external deps (no tabulate)."""
    if df is None or df.empty:
        return "_(empty)_"

    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False, name=None)]

    def esc(s: str) -> str:
        return s.replace("\n", " ").replace("|", "\\|")

    cols = [esc(c) for c in cols]
    rows = [[esc(v) for v in row] for row in rows]

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join(["| " + " | ".join(r) + " |" for r in rows])
    return "\n".join([header, sep, body])


def _role_display_name(role: str) -> str:
    if role in {"tax", "pension", "central_bank"}:
        return "gov"
    if role == "households":
        return "house"
    return role


def visualize(run_dir: Path) -> Path:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in run_dir={run_dir}")
    metadata = _json_load(meta_path)
    eval_cfg = metadata.get("eval_config", {})

    results = _load_results(run_dir)
    if not results:
        raise ValueError(f"No results found under {run_dir}")
    n_err = sum(1 for r in results if r.get("error"))
    logger.info("Loaded results: total={} errors={} run_dir={}", len(results), n_err, run_dir)

    # Helpful diagnostics: are transformer runs missing due to errors?
    try:
        env_list: list[str] = list(eval_cfg.get("environments_to_benchmark", []))
        missing_transformer: list[str] = []
        for env_name in env_list:
            env_cfg = _env_cfg(eval_cfg, env_name)
            transformer_roles: list[str] = list(env_cfg.get("transformer_roles", []) or [])
            if not transformer_roles:
                continue
            ok = 0
            err = 0
            for r in results:
                if r.get("env") != env_name:
                    continue
                role_algs = dict(r.get("role_algs", {}) or {})
                if not any(role_algs.get(tr) == "transformer" for tr in transformer_roles):
                    continue
                if r.get("error"):
                    err += 1
                else:
                    ok += 1
            if err > 0 and ok == 0:
                missing_transformer.append(f"{env_name} (transformer runs all failed: {err})")
        if missing_transformer:
            logger.warning("Transformer missing in report because all transformer runs failed for: {}", missing_transformer)
    except Exception:
        pass

    # --- Build a table similar to the attached one ---
    # Important: we average across *all* launches (canonical + sampled), not just across seeds.
    # Grouping is done by (env, target_role, algorithm used for that role).
    table_rows: list[dict[str, Any]] = []
    for r in results:
        if r.get("error"):
            continue
        env = str(r.get("env"))
        try:
            env_cfg = _env_cfg(eval_cfg, env)
        except Exception:
            continue

        transformer_roles: list[str] = list(env_cfg.get("transformer_roles", []) or [])
        role_algs: dict[str, Any] = dict(r.get("role_algs", {}) or {})
        final = dict(r.get("final_metrics", {}) or {})

        for target_role in transformer_roles:
            if target_role not in role_algs:
                continue
            target_alg = str(role_algs.get(target_role))

            if target_role in {"tax", "pension", "central_bank"}:
                acc = final.get("gov_reward")
            elif target_role == "bank":
                acc = final.get("bank_reward")
            elif target_role == "market":
                acc = final.get("firm_reward")
            else:  # households or other
                acc = final.get("house_reward")

            table_rows.append(
                {
                    "Environment": env,
                    "Role": _role_display_name(str(target_role)),
                    "TargetRole": str(target_role),
                    "Algorithm": target_alg,
                    "Acc. rew.": acc,
                    "SW": final.get("social_welfare"),
                    "GDP": final.get("GDP"),
                    "Income": final.get("house_income"),
                }
            )

    df = pd.DataFrame(table_rows)
    # Average across seeds *and* launches (any additional variants that match canonical grouping)
    if not df.empty:
        df_agg = (
            df.groupby(["Environment", "Role", "TargetRole", "Algorithm"], as_index=False)
            .agg(
                {
                    "Acc. rew.": "mean",
                    "SW": "mean",
                    "GDP": "mean",
                    "Income": "mean",
                }
            )
        )
    else:
        df_agg = df

    figures_dir = Path("reports") / "figures"
    _safe_mkdir(figures_dir)

    plot_paths: list[Path] = []

    # --- Dynamics plots over steps (mean +/- std across seeds and launches) ---
    env_list = list(eval_cfg.get("environments_to_benchmark", []))
    envs_cfg = (eval_cfg.get("environments") or {})
    for env_name in env_list:
        # Skip environments that are not configured (e.g., commented out)
        if env_name not in envs_cfg:
            logger.warning(f"Skipping visualization for environment '{env_name}' - not found in environments config (may be commented out)")
            continue
        env_cfg = _env_cfg(eval_cfg, env_name)
        metrics_to_plot: list[str] = list(env_cfg.get("metrics_to_plot", []) or [])
        transformer_roles: list[str] = list(env_cfg.get("transformer_roles", []) or [])

        for target_role in transformer_roles:
            for metric in metrics_to_plot:
                # series_by_alg: alg -> list[series] (each series is one run's per-step values)
                series_by_alg: dict[str, list[list[float]]] = {}
                for rr in results:
                    if rr.get("error"):
                        continue
                    if rr.get("env") != env_name:
                        continue
                    role_algs = dict(rr.get("role_algs", {}) or {})
                    if target_role not in role_algs:
                        continue
                    alg = str(role_algs.get(target_role))
                    ts = (rr.get("timeseries") or {}).get(metric)
                    if not isinstance(ts, list) or len(ts) == 0:
                        continue
                    series_by_alg.setdefault(alg, []).append([_scalarize(x) for x in ts])

                if not series_by_alg:
                    continue

                plt.figure(figsize=(9, 5))
                for alg, series_list in series_by_alg.items():
                    if not series_list:
                        continue
                    max_len = max(len(s) for s in series_list)
                    mat = []
                    for s in series_list:
                        arr = np.asarray(s, dtype=np.float32)
                        if arr.shape[0] < max_len:
                            arr = np.pad(arr, (0, max_len - arr.shape[0]), constant_values=np.nan)
                        mat.append(arr)
                    arr = np.stack(mat, axis=0)  # [R, T]
                    mean = np.nanmean(arr, axis=0)
                    std = np.nanstd(arr, axis=0)
                    # Use year index from JSON arrays (1..T).
                    x = np.arange(1, max_len + 1, dtype=np.int32)
                    plt.plot(x, mean, label=alg)
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

                plt.title(f"{env_name} | target={target_role} | {_pretty_metric_name(metric)}")
                plt.xlabel("year")
                plt.ylabel(metric)
                plt.legend()
                plt.tight_layout()

                out = figures_dir / f"transformer_eval__{env_name}__target={target_role}__metric={metric}.png"
                plt.savefig(out, dpi=160)
                plt.close()
                plot_paths.append(out)
                logger.info("Wrote plot {}", out)

    # --- Write markdown report ---
    report_path = Path("reports") / "transformer_eval_report.md"
    _safe_mkdir(report_path.parent)

    md: list[str] = []
    md.append(f"# Transformer benchmark report\n")
    md.append(f"- Run dir: `{run_dir}`\n")
    md.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n")
    if n_err:
        md.append(f"- Errors: `{n_err}` (see failed JSONs under run dir)\n")

    md.append("\n## Performance table (env-averaged across seeds and launches)\n")
    if df_agg.empty:
        md.append("_No results found (check errors in result files)._ \n")
    else:
        # Create a compact table per env+role with algorithm columns (similar to paper-style tables)
        for (env, role, target_role), grp in df_agg.groupby(["Environment", "Role", "TargetRole"]):
            md.append(f"\n### {env} — {role}\n")

            # desired algorithm ordering for readability
            preferred_alg_order = [
                "ppo",
                "sac",
                "ddpg",
                "rule_based",
                "bc",
                "transformer",
                "llm",
                "data_based",
                "real",
                "saez",
                "us_federal",
            ]
            algs_present = [a for a in preferred_alg_order if a in set(grp["Algorithm"].tolist())]
            # append any remaining algs deterministically
            algs_present += sorted([a for a in set(grp["Algorithm"].tolist()) if a not in set(algs_present)])

            metrics = ["Acc. rew.", "SW", "GDP", "Income"]
            # wide table: rows=metrics, cols=algorithms
            wide = pd.DataFrame({"Metric": metrics})
            for alg in algs_present:
                row = grp[grp["Algorithm"] == alg]
                vals = {m: (row[m].iloc[0] if (not row.empty) else np.nan) for m in metrics}
                wide[alg] = [vals[m] for m in metrics]

            def _fmt(x: Any) -> str:
                if pd.isna(x):
                    return ""
                try:
                    xf = float(x)
                except Exception:
                    return str(x)
                return f"{xf:,.0f}" if abs(xf) >= 1e3 else f"{xf:.0f}"

            for c in wide.columns:
                if c == "Metric":
                    continue
                wide[c] = wide[c].map(_fmt)

            md.append(_df_to_markdown(wide))
            md.append("\n")

    md.append("\n## Metric dynamics over years (mean ± std across seeds and launches)\n")
    if not plot_paths:
        md.append("_No plots were generated (check metrics_to_plot and saved histories)._ \n")
    else:
        for p in plot_paths:
            # report is in `reports/`, figures are in `reports/figures/`
            rel = f"figures/{p.name}"
            md.append(f"\n### {p.name}\n")
            md.append(f"![{p.name}]({rel})\n")

    report_path.write_text("\n".join(md))
    logger.info("Wrote report {}", report_path)
    return report_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["run", "viz"], default="run")
    p.add_argument("--config", type=str, default="eval_config.yaml")
    p.add_argument("--run_dir", type=str, default="")
    args = p.parse_args()

    if args.mode == "run":
        cfg_path = Path(args.config)
        logger.info("Mode=run config={}", cfg_path)
        eval_cfg = load_eval_config(cfg_path)
        run_dir = run_benchmark(eval_cfg, config_path=cfg_path)
        report_path = visualize(run_dir)
        print(f"Wrote results to: {run_dir}")
        print(f"Wrote report to: {report_path}")
        return

    # viz-only
    run_dir = Path(args.run_dir) if args.run_dir else (_find_latest_run_dir() or Path())
    logger.info("Mode=viz run_dir={}", run_dir)
    report_path = visualize(run_dir)
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()

