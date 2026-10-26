"""Small-language-model causal world model, plus the non-LLM comparison policies.

`SLMPolicy` wraps a Hugging Face causal LM (optionally 4-bit quantised, optionally
with a LoRA adapter from `causlm.slm.distill`) and measures the wall-clock
latency and the NVML Joule count of every invocation, because those numbers are
what the meta-controller trades against control quality.

`TemplatePolicy` is the CPU-only stand-in required when no GPU is available:
the same interface, backed by the distilled rule set plus a small gradient-boosted
ATE regressor rather than a transformer.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ACTUATORS, SLM_MODELS, VARS, WEIGHTS
from ..utils import EnergyMeter
from .prompts import SYSTEM_PROMPT, build_prompt
from .schema import CausalAdvice, parse_advice


@dataclass
class InvocationCost:
    latency_ms: float = 0.0
    energy_j: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mode: str = "n/a"


class BasePolicy:
    name = "base"
    uses_stat_graph = True

    def advise(self, ctx: Dict) -> Tuple[CausalAdvice, InvocationCost]:
        raise NotImplementedError

    def advise_batch(self, ctxs: Sequence[Dict]):
        return [self.advise(c) for c in ctxs]


# ------------------------------------------------------------------- HF SLM
class SLMPolicy(BasePolicy):
    name = "slm"

    def __init__(self, model_key: str = "qwen2.5-0.5b",
                 adapter: Optional[str | Path] = None,
                 quantize: Optional[str] = None,       # None | "4bit" | "8bit"
                 device: str = "cuda", max_new_tokens: int = 232,
                 use_stat_graph: bool = True, dtype: str = "bfloat16",
                 device_index: int = 0, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = SLM_MODELS.get(model_key, model_key)
        self.model_key = model_key
        self.uses_stat_graph = use_stat_graph
        self.max_new_tokens = max_new_tokens
        self.device = device if torch.cuda.is_available() else "cpu"
        self.device_index = device_index
        self.quantize = quantize

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        kw = {"dtype": getattr(torch, dtype)}
        if quantize in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantize == "4bit"),
                load_in_8bit=(quantize == "8bit"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": device_index}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
            self.model = self.model.merge_and_unload() if quantize is None else self.model
            self.name = f"slm+lora({model_key})"
        else:
            self.name = f"slm({model_key})"
        if quantize is None:
            self.model = self.model.to(f"{self.device}:{device_index}"
                                       if self.device == "cuda" else "cpu")
        self.model.eval()
        self.meter = EnergyMeter(device_index=device_index)
        torch.manual_seed(seed)

    def _chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)

    def _prompt_from_ctx(self, ctx: Dict) -> str:
        A = ctx["A"] if self.uses_stat_graph else np.zeros_like(ctx["A"])
        return build_prompt(
            A, ctx["X"],
            pvals=ctx.get("pvals") if self.uses_stat_graph else None,
            scores=ctx.get("scores") if self.uses_stat_graph else None,
            var_names=ctx.get("var_names", VARS),
            device_meta=ctx.get("device_meta"),
            candidates=ctx.get("candidates"),
            regime=ctx.get("regime"))

    @staticmethod
    def _strip(text: str) -> str:
        return text.strip()

    def advise(self, ctx: Dict):
        return self.advise_batch([ctx])[0]

    def advise_batch(self, ctxs: Sequence[Dict]):
        import torch
        prompts = [self._chat(self._prompt_from_ctx(c)) for c in ctxs]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self.meter.measure() as m:
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        n = len(ctxs)
        res = []
        for i, t in enumerate(texts):
            adv = parse_advice(self._strip(t))
            cost = InvocationCost(
                latency_ms=m["latency_ms"] / n, energy_j=m["energy_j"] / n,
                prompt_tokens=int(enc["attention_mask"][i].sum()),
                completion_tokens=int((gen[i] != self.tok.pad_token_id).sum()),
                mode=m["mode"])
            res.append((adv, cost))
        return res


# -------------------------------------------------- CPU stand-in ("distilled")
class TemplatePolicy(BasePolicy):
    """Distilled rule set + learned ATE regressor. Runs on CPU in microseconds.

    This is the fallback the constraints ask for: same JSON contract, same
    call site, no transformer. It also doubles as an ablation -- it shows how
    much of CauSLM-Edge's benefit survives without a language model at all.
    """
    name = "template"

    #: features the ATE regressor consumes, in this order
    CTX_KPIS = ["indoor_temp", "outdoor_temp", "pmv", "latency_ms", "pdr",
                "occupancy", "humidity", "hvac_power", "it_energy_j",
                "hvac_setpoint", "sample_rate_hz", "duty_cycle",
                "offload_frac", "queue_util", "suspicion", "tariff"]
    ATE_KEYS = ["energy_cost", "indoor_temp", "latency_ms", "pmv"]

    def __init__(self, rules_path: Optional[Path] = None,
                 ate_model_path: Optional[Path] = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.rules = self._default_rules()
        if rules_path and Path(rules_path).exists():
            self.rules.update(json.loads(Path(rules_path).read_text()))
        self.ate_model = None
        if ate_model_path is None:
            ate_model_path = WEIGHTS / "template_ate.joblib"
        if Path(ate_model_path).exists():
            import joblib
            from .ate_model import MultiATE  # noqa: F401 (pickle needs it)
            self.ate_model = joblib.load(ate_model_path)

    @staticmethod
    def _default_rules() -> Dict:
        # Known confounding structure of a building gateway. `occupancy` and
        # `outdoor_temp` are the two exogenous drivers; any edge between two of
        # their children is a candidate spurious edge.
        return {"exogenous": ["occupancy", "outdoor_temp"],
                "children_of_occupancy": ["hvac_power", "wifi_rssi",
                                          "indoor_temp", "latency_ms",
                                          "humidity"],
                "children_of_outdoor": ["indoor_temp", "hvac_power",
                                        "energy_kwh"]}

    def _spurious(self, A: np.ndarray, names: Sequence[str]):
        out = []
        co = set(self.rules["children_of_occupancy"])
        ct = set(self.rules["children_of_outdoor"])
        for i, u in enumerate(names):
            for j, v in enumerate(names):
                if i == j or A[i, j] <= 0:
                    continue
                if (u in co and v in co) or (u in ct and v in ct):
                    # only flag pairs with no plausible mechanism
                    if not _mechanistic(u, v):
                        out.append((u, v))
        return out

    def advise(self, ctx: Dict):
        t0 = time.perf_counter()
        names = list(ctx.get("var_names", VARS))
        A = np.asarray(ctx["A"])
        cands = list(ctx.get("candidates") or [{"hvac_setpoint": 23.0}])
        kpis = ctx.get("kpis") or {}
        best, best_ate = None, None
        if self.ate_model is not None and kpis:
            cv = np.array([float(kpis.get(k, 0.0)) for k in self.CTX_KPIS])
            pmv_now = float(kpis.get("pmv", 0.0))
            X = np.stack([np.concatenate([cv, _cand_vec(c)]) for c in cands])
            A_hat = self.ate_model.predict(X)
            from ..utility import intervention_utility
            util = [intervention_utility(
                dict(zip(self.ATE_KEYS, [float(z) for z in a])), kpis)
                for a in A_hat]
            # "hold the current settings" is always an option, scored at zero
            k = int(np.argmax(util))
            if util[k] > 0:
                best = cands[k]
                best_ate = dict(zip(self.ATE_KEYS,
                                    [float(z) for z in A_hat[k]]))
        if best is None:
            best = {}
            best_ate = {k: 0.0 for k in self.ATE_KEYS}
        adv = CausalAdvice(
            do=dict(best), predicted_ATE=best_ate, valid=True,
            confounders=[c for c in self.rules["exogenous"] if c in names],
            spurious_edges=self._spurious(A, names),
            safety="ok", explain="distilled rule set")
        dt = (time.perf_counter() - t0) * 1e3
        return adv, InvocationCost(latency_ms=dt, energy_j=dt / 1000.0 * 2.5,
                                   mode="estimated_cpu")


def _mechanistic(u: str, v: str) -> bool:
    """Edges a building engineer would accept without a confounder."""
    ok = {("hvac_power", "indoor_temp"), ("hvac_power", "energy_kwh"),
          ("indoor_temp", "humidity"), ("hvac_power", "humidity"),
          ("wifi_rssi", "latency_ms"), ("latency_ms", "energy_kwh"),
          ("indoor_temp", "hvac_power"), ("occupancy", "hvac_power"),
          ("occupancy", "wifi_rssi"), ("occupancy", "indoor_temp"),
          ("occupancy", "latency_ms"), ("occupancy", "humidity"),
          ("outdoor_temp", "indoor_temp"), ("outdoor_temp", "hvac_power"),
          ("outdoor_temp", "energy_kwh"), ("humidity", "energy_kwh"),
          ("hvac_power", "latency_ms")}
    return (u, v) in ok


def _cand_vec(c: Dict[str, float]) -> np.ndarray:
    return np.array([c.get(k, np.nan) if k in c else 0.0 for k in ACTUATORS] +
                    [1.0 if k in c else 0.0 for k in ACTUATORS], float)


class ParrotPolicy(BasePolicy):
    """Baseline 3: the same small model, but it never sees the PCMCI+ skeleton."""
    name = "slm-parrot"

    def __init__(self, **kw):
        kw["use_stat_graph"] = False
        self.inner = SLMPolicy(**kw)
        self.name = f"slm-parrot({self.inner.model_key})"
        self.uses_stat_graph = False

    def advise(self, ctx):
        return self.inner.advise(ctx)

    def advise_batch(self, ctxs):
        return self.inner.advise_batch(ctxs)


def load_policy(kind: str, **kw) -> BasePolicy:
    kind = kind.lower()
    if kind in ("template", "rules", "mock"):
        return TemplatePolicy(**{k: v for k, v in kw.items()
                                 if k in ("rules_path", "ate_model_path", "seed")})
    if kind == "parrot":
        return ParrotPolicy(**kw)
    return SLMPolicy(**kw)
