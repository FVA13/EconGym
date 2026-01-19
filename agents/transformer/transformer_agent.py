import copy
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from loguru import logger as _loguru_logger


@dataclass(frozen=True)
class TransformerAdapterSpec:
    """
    Minimal spec to make a marl-macro-modeling transformer checkpoint usable in EconGym.

    Notes:
    - EconGym observations are *vectors*; marl-macro-modeling expects *named tokens*.
      We bridge that via STATE_MAPPING/ACTION_MAPPING with lots of "Empty" fallbacks.
    - The checkpoint MUST match the instantiated model architecture.
    """

    marl_project_root: str
    checkpoint_path: str

    # Model architecture (must match checkpoint) if loading raw state_dict (.pt/.pth).
    # If using a Lightning checkpoint (.ckpt) like in `examples/infer_model.ipynb`,
    # these can be left unset (0) and will be derived from the checkpoint.
    state_dim: int = 0
    action_dim: int = 0
    num_tasks: int = 0
    d_model: int = 0
    nhead: int = 0
    num_layers: int = 0
    max_seq_len: int = 0
    model_params_dim: int = 0
    pinn_output_dim: int = 0
    has_pinn: bool = False

    # Conditioning
    task_id: int = 0
    model_params: Optional[list[float]] = None  # padded/truncated to model_params_dim

    # Token descriptions (names must exist in marl STATE_MAPPING/ACTION_MAPPING or be "Empty")
    state_description: Optional[list[str]] = None
    action_description: Optional[list[str]] = None


def _safe_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


class transformer_agent:
    """
    EconGym agent wrapper around marl-macro-modeling's AlgorithmDistillationTransformer.

    Interface contract (per EconGym Runner):
      - get_action(obs_tensor) -> np.ndarray action
      - on_policy attribute (True so Runner stores trajectories similarly)
      - optional observe(reward, done=False) hook (we add Runner support for this)
    """

    def __init__(self, envs, args, agent_name="government", type=None):
        self.name = "transformer"
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent_type = type  # for government: "tax" | "central_bank" | "pension"

        self.device = "cuda" if getattr(self.args, "cuda", False) else "cpu"
        self.on_policy = True
        # Action scaling contract:
        # - If True: model outputs are normalized in [-1, 1] and Runner should scale to real_action_min/max.
        # - If False: model outputs are already in the "real" action units expected by EconGym entities.
        # Default False because pretrained checkpoints may not follow RL normalization.
        self.outputs_normalized = bool(getattr(self.args, "transformer_outputs_normalized", False))
        self.debug = bool(getattr(self.args, "transformer_debug", False))

        spec = self._build_spec_from_args()

        # Import marl-macro-modeling modules dynamically (local project, not a pip package)
        marl_root = os.path.abspath(spec.marl_project_root)
        if not os.path.isdir(marl_root):
            raise ValueError(
                f"Invalid marl_project_root='{spec.marl_project_root}'. "
                f"Expected a directory containing 'lib/'."
            )
        if marl_root not in sys.path:
            sys.path.insert(0, marl_root)

        # marl-macro-modeling config uses loguru and calls `logger.remove(0)` at import time.
        # Depending on the host app's logging setup, handler id 0 may not exist and loguru raises.
        # Make this import robust by temporarily making `remove()` tolerant to missing handler ids.
        _orig_remove = getattr(_loguru_logger, "remove", None)
        if callable(_orig_remove):
            def _safe_remove(handler_id=None):  # type: ignore[no-redef]
                try:
                    return _orig_remove(handler_id)
                except ValueError:
                    # Ignore "There is no existing handler with id X"
                    return None

            _loguru_logger.remove = _safe_remove  # type: ignore[assignment]

        # Token-ID helpers can differ across marl revisions.
        # Prefer helper fns if present; otherwise fall back to direct mappings.
        try:
            from lib.dataset import action_token_id, state_token_id  # type: ignore

            self._state_token_id = state_token_id
            self._action_token_id = action_token_id
        except Exception:
            # ACTION_MAPPING has been moved to Tokenizer class in lib.dataset
            try:
                from lib.dataset import Tokenizer  # type: ignore
                tokenizer = Tokenizer()
                # Use the tokenizer's instance methods
                self._state_token_id = tokenizer.state_token_id
                self._action_token_id = tokenizer.action_token_id
            except Exception as e:
                raise ImportError(
                    f"Could not import Tokenizer from lib.dataset. "
                    f"Original error: {e}"
                )

        # Restore original loguru behavior after successful marl imports.
        if callable(_orig_remove):
            _loguru_logger.remove = _orig_remove  # type: ignore[assignment]

        # --- Load model the same way as `examples/infer_model.ipynb` whenever possible ---
        ckpt_path = str(spec.checkpoint_path)
        if ckpt_path.endswith(".ckpt"):
            try:
                from pipeline.run_pipeline import EconomicPolicyModel  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Failed to import marl-macro-modeling Lightning pipeline loader. "
                    "Your checkpoint is a '.ckpt', so we need marl's deps installed "
                    "(notably `lightning` and `hydra`). Original error: "
                    f"{e}"
                )

            policy_model = EconomicPolicyModel.load_from_checkpoint(ckpt_path, weights_only=False)
            policy_model.eval()
            self.model = policy_model.model
            self.model.eval()
            # marl's transformer inference uses `self.device` (string), so set it
            setattr(self.model, "device", self.device)
            self.model.to(self.device)

            # Derive dimensions from the loaded model (no manual hyperparams required)
            state_dim = int(getattr(self.model, "state_dim"))
            action_dim = int(getattr(self.model, "action_dim"))
            max_seq_len = int(getattr(self.model, "max_seq_len"))
            model_params_dim = int(getattr(self.model, "model_params_dim"))
        else:
            # Raw state_dict path: require explicit architecture config
            from lib.models.transformer import AlgorithmDistillationTransformer  # type: ignore

            state_dim = int(spec.state_dim)
            action_dim = int(spec.action_dim)
            max_seq_len = int(spec.max_seq_len)
            model_params_dim = int(spec.model_params_dim)

            if state_dim <= 0 or action_dim <= 0 or max_seq_len <= 0:
                raise ValueError(
                    "When using a raw transformer checkpoint (not '.ckpt'), you must set "
                    "Trainer.transformer_state_dim, transformer_action_dim, transformer_max_seq_len, "
                    "transformer_model_params_dim, transformer_num_tasks, transformer_d_model, transformer_nhead, "
                    "transformer_num_layers."
                )

            self.model = AlgorithmDistillationTransformer(
                state_dim=state_dim,
                action_dim=action_dim,
                num_tasks=int(spec.num_tasks),
                d_model=int(spec.d_model),
                nhead=int(spec.nhead),
                num_layers=int(spec.num_layers),
                max_seq_len=max_seq_len,
                model_params_dim=model_params_dim,
                pinn_output_dim=int(spec.pinn_output_dim),
                has_pinn=bool(spec.has_pinn),
            ).to(self.device)
            self.model.eval()
            setattr(self.model, "device", self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state_dict, strict=True)

        # Build default descriptions if not provided.
        state_desc = spec.state_description or self._default_state_description_for_role()
        action_desc = spec.action_description or self._default_action_description_for_role()

        # Pad/truncate descriptions to match model dims.
        self.state_description = (state_desc + ["Empty"] * state_dim)[:state_dim]
        self.action_description = (action_desc + ["Empty"] * action_dim)[:action_dim]

        self.states_info = torch.tensor(
            [self._safe_state_token_id(n) for n in self.state_description],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)  # [1, state_dim]
        self.actions_info = torch.tensor(
            [self._safe_action_token_id(n) for n in self.action_description],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)  # [1, action_dim]

        model_params = _safe_float_list(spec.model_params)
        model_params = (model_params + [0.0] * model_params_dim)[:model_params_dim]
        self.model_params = torch.tensor(model_params, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.task_ids = torch.tensor([int(spec.task_id)], dtype=torch.long, device=self.device)

        self.max_seq_len = int(max_seq_len)
        self._states_hist: deque[torch.Tensor] = deque(maxlen=self.max_seq_len)
        self._actions_hist: deque[torch.Tensor] = deque(maxlen=self.max_seq_len)
        self._rewards_hist: deque[torch.Tensor] = deque(maxlen=self.max_seq_len)

        self._last_action = torch.zeros(action_dim, dtype=torch.float32, device=self.device)
        self._last_reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)

    # -----------------------
    # Runner hook
    # -----------------------
    def observe(self, reward: Any, done: bool = False):
        """Optional hook called by Runner after env.step()."""
        try:
            r = float(reward)
        except Exception:
            # If reward is vector-like, fall back to mean.
            r = float(np.mean(np.array(reward, dtype=np.float32)))
        self._last_reward = torch.tensor([r], dtype=torch.float32, device=self.device)
        if done:
            self._reset_history()

    def _reset_history(self):
        self._states_hist.clear()
        self._actions_hist.clear()
        self._rewards_hist.clear()
        self._last_reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)

    # -----------------------
    # EconGym interface
    # -----------------------
    def train(self, transition_dict):
        # This agent is meant for evaluation / transfer; training stays in marl-macro-modeling.
        return torch.tensor(0.0), torch.tensor(0.0)

    def save(self, dir_path):
        pass

    def get_action(self, obs_tensor):
        """
        Produce a normalized action in [-1, 1] compatible with EconGym action_space.

        Important:
        - This adapter expects the checkpoint to have been trained on similarly-normalized actions.
        - Reward conditioning works only if Runner calls observe() after each step.
        """
        obs = obs_tensor.detach().cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else np.array(obs_tensor)

        # EconGym can provide 2D obs for multi-entities (e.g., households [N, d], firms [firm_n, d]).
        # For now we keep a *single* sequence model instance and (if needed) broadcast its action
        # to all entities; this keeps integration simple and avoids per-entity history management.
        obs_1d = obs if obs.ndim == 1 else np.mean(obs, axis=0)

        state_vec = self._obs_to_state_vec(obs_1d).to(self.device)  # [state_dim]

        # Align action/reward with current state (use previous action/reward as context).
        self._states_hist.append(state_vec)
        self._actions_hist.append(self._last_action)
        self._rewards_hist.append(self._last_reward)

        states = torch.stack(list(self._states_hist), dim=0).unsqueeze(0)   # [1, T, state_dim]
        actions = torch.stack(list(self._actions_hist), dim=0).unsqueeze(0)  # [1, T, action_dim]
        rewards = torch.stack(list(self._rewards_hist), dim=0).unsqueeze(0)  # [1, T, 1]

        with torch.no_grad():
            out, _pinn = self.model.forward(
                states=states,
                states_info=self.states_info,
                actions=actions,
                actions_info=self.actions_info,
                rewards=rewards,
                task_ids=self.task_ids,
                model_params=self.model_params,
            )
            action = out[0, -1].detach().cpu().numpy()
            # _loguru_logger.debug(f"action: {action} (before clipping)")

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action = self._adapt_action_to_env(obs_tensor, action)
        # _loguru_logger.debug(f"[transformer_agent] role={self.agent_name}:{self.agent_type} action(sample)={action if np.ndim(action)==1 else action[0]}")

        # Keep last action in model's action_dim space for recurrence.
        # If we sliced/padded for the env, re-pad to model action dim so the history stays consistent.
        model_action_dim = int(self.actions_info.shape[-1])
        action_for_history = np.asarray(action, dtype=np.float32)
        if action_for_history.ndim != 1:
            # if env expects 2D (e.g., households), store mean action for recurrence
            action_for_history = np.mean(action_for_history, axis=0)
        if action_for_history.shape[0] >= model_action_dim:
            action_for_history = action_for_history[:model_action_dim]
        else:
            action_for_history = np.concatenate(
                [action_for_history, np.zeros(model_action_dim - action_for_history.shape[0], dtype=np.float32)],
                axis=0,
            )

        self._last_action = torch.tensor(action_for_history, dtype=torch.float32, device=self.device)
        return action

    # -----------------------
    # Spec + mapping helpers
    # -----------------------
    def _build_spec_from_args(self) -> TransformerAdapterSpec:
        # Where marl-macro-modeling lives on your machine.
        marl_root = getattr(self.args, "marl_project_root", None) or getattr(self.args, "transformer_marl_root", None)
        ckpt = getattr(self.args, "transformer_ckpt_path", None)

        if not marl_root or not ckpt:
            raise ValueError(
                "Missing transformer configuration. Please set in cfg YAML under Trainer:\n"
                "- marl_project_root: '/Users/.../marl-macro-modeling'\n"
                "- transformer_ckpt_path: '/path/to/checkpoint.pt'\n"
                "And the architecture fields: transformer_state_dim, transformer_action_dim, transformer_num_tasks, "
                "transformer_d_model, transformer_nhead, transformer_num_layers, transformer_max_seq_len, "
                "transformer_model_params_dim."
            )

        # Architecture (must match checkpoint) if using a raw state_dict checkpoint.
        # For Lightning `.ckpt`, these can be left as 0 and will be derived from checkpoint.
        state_dim = int(getattr(self.args, "transformer_state_dim", 0))
        action_dim = int(getattr(self.args, "transformer_action_dim", 0))
        num_tasks = int(getattr(self.args, "transformer_num_tasks", 0))
        d_model = int(getattr(self.args, "transformer_d_model", 0))
        nhead = int(getattr(self.args, "transformer_nhead", 0))
        num_layers = int(getattr(self.args, "transformer_num_layers", 0))
        max_seq_len = int(getattr(self.args, "transformer_max_seq_len", 0))
        model_params_dim = int(getattr(self.args, "transformer_model_params_dim", 0))
        pinn_output_dim = int(getattr(self.args, "transformer_pinn_output_dim", 0))
        has_pinn = bool(getattr(self.args, "transformer_has_pinn", False))

        task_id = int(getattr(self.args, "transformer_task_id", 0))
        model_params = getattr(self.args, "transformer_model_params", None)

        # Optional explicit token descriptions (advanced)
        state_desc = getattr(self.args, "transformer_state_description", None)
        action_desc = getattr(self.args, "transformer_action_description", None)

        return TransformerAdapterSpec(
            marl_project_root=str(marl_root),
            checkpoint_path=str(ckpt),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            num_tasks=int(num_tasks),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            max_seq_len=int(max_seq_len),
            model_params_dim=int(model_params_dim),
            pinn_output_dim=pinn_output_dim,
            has_pinn=has_pinn,
            task_id=task_id,
            model_params=_safe_float_list(model_params) if model_params is not None else None,
            state_description=list(state_desc) if state_desc is not None else None,
            action_description=list(action_desc) if action_desc is not None else None,
        )

    def _safe_state_token_id(self, name: str) -> int:
        try:
            return int(self._state_token_id(name))
        except Exception:
            return int(self._state_token_id("Empty"))

    def _safe_action_token_id(self, name: str) -> int:
        try:
            return int(self._action_token_id(name))
        except Exception:
            return int(self._action_token_id("Empty"))

    def _default_state_description_for_role(self) -> list[str]:
        """
        Default mapping from EconGym obs vector positions to marl state tokens.

        EconGym global_obs (see env/set_observation.py):
          [top10_wealth, top10_edu, bot50_wealth, bot50_edu, wage, price, lending_rate, deposit_rate]

        We map unknown semantics to "Empty" but still keep the numeric value.
        """
        global_desc = [
            "Capital",      # top10 wealth ~ capital proxy
            "Empty",        # education (no canonical token)
            "Capital",      # bottom wealth ~ capital proxy
            "Empty",        # education
            "Real Wage",    # wage
            "Price Level",  # price
            "InterestRate", # lending rate (proxy)
            "InterestRate", # deposit rate (proxy)
        ]

        if self.agent_name == "government":
            if self.agent_type == "tax":
                return global_desc + ["Debt"]
            if self.agent_type == "central_bank":
                return global_desc + ["Inflation Rate", "Output Growth Rate"]
            if self.agent_type == "pension":
                # pension obs is custom; map what we can, fallback to Empty
                return [
                    "Empty",  # accumulated pension account
                    "Empty",  # population
                    "Empty",  # old_n
                    "Empty",  # retire_age
                    "Empty",  # contribution_rate
                    "Debt",   # Bt
                    "Output", # GDP
                ]
        elif self.agent_name == "bank":
            # [base_interest_rate, reserve_ratio, lending_rate, deposit_rate, current_loans, total_account]
            return [
                "InterestRate",
                "Empty",
                "InterestRate",
                "InterestRate",
                "Debt",   # proxy: credit stock
                "Money Stock",
            ]
        elif self.agent_name == "market":
            # firm obs: [capital, productivity, lending_rate]
            return ["Capital", "Productivity", "InterestRate"]

        # Fallback: just treat everything as "Empty" to avoid token errors.
        return ["Empty"]

    def _default_action_description_for_role(self) -> list[str]:
        """
        Default mapping from EconGym action vector positions to marl action tokens.
        Unknown dims become "Empty".
        """
        if self.agent_name == "government":
            if self.agent_type == "tax":
                # [tau, xi, tau_a, xi_a, Gt_prob, (optional firm spending shares...)]
                base = ["tax_rate_change", "Empty", "tax_rate_change", "Empty", "gov_spending_change"]
                return base
            if self.agent_type == "central_bank":
                # [base_interest_rate, reserve_ratio]
                return ["Nominal Interest Rate", "Empty"]
            if self.agent_type == "pension":
                # [retire_age, contribution_rate]
                return ["Empty", "Empty"]
        elif self.agent_name == "bank":
            # [lending_rate, deposit_rate]
            return ["Nominal Interest Rate", "Nominal Interest Rate"]
        elif self.agent_name == "market":
            # [price, wage] (for monopoly/oligopoly variants)
            return ["Empty", "Real Consumption"]
        return ["Empty"]

    def _obs_to_state_vec(self, obs_1d: np.ndarray) -> torch.Tensor:
        """
        Convert a 1D EconGym observation vector into a fixed-length state vector for the transformer.
        We keep raw values and pad/truncate to state_dim.
        """
        x = np.asarray(obs_1d, dtype=np.float32).flatten()
        # Pad/truncate to model state_dim
        # (state_dim is implied by states_info length)
        state_dim = int(self.states_info.shape[-1])
        if x.shape[0] >= state_dim:
            x = x[:state_dim]
        else:
            x = np.concatenate([x, np.zeros(state_dim - x.shape[0], dtype=np.float32)], axis=0)
        return torch.tensor(x, dtype=torch.float32)

    def _adapt_action_to_env(self, obs_tensor, action_1d: np.ndarray):
        """
        Ensure returned actions match EconGym's expected shapes and action_dim for this role.

        Key cases:
        - households: expects (N, action_dim)
        - market with firm_n>1: expects (firm_n, action_dim)
        - others: expects (action_dim,)
        """
        # Determine expected env action_dim for the current role
        if self.agent_name == "government":
            expected_dim = int(self.envs.government[self.agent_type].action_dim)
        elif self.agent_name == "households":
            expected_dim = int(self.envs.households.action_dim)
        elif self.agent_name == "market":
            expected_dim = int(self.envs.market.action_dim)
        elif self.agent_name == "bank":
            expected_dim = int(self.envs.bank.action_dim)
        else:
            expected_dim = int(action_1d.shape[-1])

        a = np.asarray(action_1d, dtype=np.float32).flatten()
        # Slice/pad to expected_dim
        if a.shape[0] >= expected_dim:
            a = a[:expected_dim]
        else:
            a = np.concatenate([a, np.zeros(expected_dim - a.shape[0], dtype=np.float32)], axis=0)

        # Broadcast to entity dimension when obs is 2D
        if isinstance(obs_tensor, torch.Tensor) and obs_tensor.ndim == 2:
            n = int(obs_tensor.shape[0])
            if self.agent_name in {"households", "market"} and n > 0 and expected_dim > 0:
                a2 = np.tile(a.reshape(1, -1), (n, 1)).astype(np.float32)
                return self._clip_action_semantics(a2)

        return self._clip_action_semantics(a.astype(np.float32))

    def _clip_action_semantics(self, action: np.ndarray) -> np.ndarray:
        """
        Light semantic safety rails for EconGym action meaning (mainly for households).
        This prevents obviously-invalid values from exploding the simulation.
        """
        if self.agent_name != "households":
            return action

        a = np.asarray(action, dtype=np.float32)
        # households expects at least: [saving_share, labor_share] (+ optional invest_share)
        # Clip to [0,1] for stability/BC-like semantics.
        if a.ndim == 1:
            if a.shape[0] >= 1:
                a[0] = np.clip(a[0], 0.0, 1.0)
            if a.shape[0] >= 2:
                a[1] = np.clip(a[1], 0.0, 1.0)
            if a.shape[0] >= 3:
                a[2] = np.clip(a[2], 0.0, 1.0)
            return a

        if a.ndim == 2:
            if a.shape[1] >= 1:
                a[:, 0] = np.clip(a[:, 0], 0.0, 1.0)
            if a.shape[1] >= 2:
                a[:, 1] = np.clip(a[:, 1], 0.0, 1.0)
            if a.shape[1] >= 3:
                a[:, 2] = np.clip(a[:, 2], 0.0, 1.0)
            return a

        return a
