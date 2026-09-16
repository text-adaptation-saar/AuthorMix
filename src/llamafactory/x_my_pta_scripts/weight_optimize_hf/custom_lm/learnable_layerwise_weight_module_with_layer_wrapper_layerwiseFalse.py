"""
Adapter-wise learnable weight module: one trainable weight per adapter (not per layer).
GRPO-compatible. Same weight is applied to all layers for each adapter.
"""

import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft.peft_model import PeftModel
from typing import List, Optional


class WeightedLoRALinear(nn.Module):
    """
    Wrapper for a linear layer with multiple LoRA adapters.
    Uses adapter-wise weights: one weight per adapter, same for all layers.
    """

    def __init__(self, base_layer, lora_modules, adapter_names, weight_getter):
        """
        Args:
            base_layer: The base linear layer
            lora_modules: Dict of {adapter_name: {lora_A, lora_B, scaling, dropout}}
            adapter_names: List of adapter names in order
            weight_getter: Callable() -> tensor of shape (num_adapters,) with no args
        """
        super().__init__()
        self.base_layer = base_layer
        self.lora_modules = lora_modules
        self.adapter_names = adapter_names
        self.weight_getter = weight_getter

    def forward(self, x):
        """
        Forward with weighted LoRA using adapter-wise weights (same for all layers).
        W_eff = W_base + sum_a w_a * (B_a @ A_a) * scaling_a.
        """
        output = self.base_layer(x)
        weights = self.weight_getter()
        weights = weights.to(device=output.device, dtype=output.dtype)

        for idx, adapter_name in enumerate(self.adapter_names):
            if adapter_name not in self.lora_modules:
                continue
            lora = self.lora_modules[adapter_name]
            lora_input = lora['dropout'](x)
            lora_out = lora['lora_A'](lora_input.to(lora['lora_A'].weight.dtype))
            lora_out = lora['lora_B'](lora_out)
            lora_out = lora_out.to(output.dtype) * lora['scaling']
            output = output + weights[idx] * lora_out
        return output

class AdapterwiseLearnableWeightModule(nn.Module):
    """
    Adapter-wise weighted LoRA: one trainable weight per adapter.
    The same weight is applied to all layers for each adapter.
    """

    model: PeftModel

    def __init__(
            self,
            base_model_path: str,
            adapter_paths: List[str],
            adapter_names: Optional[List[str]] = None,
            device: str = "auto",
            weight_init: str = "zeros",
            custom_init_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if adapter_names is None:
            adapter_names = [f"adapter_{i}" for i in range(len(adapter_paths))]

        self.adapter_names = adapter_names
        self.num_adapters = len(adapter_paths)

        print("\n" + "="*80)
        print("ADAPTER-WISE WEIGHTED LORA (one weight per adapter)")
        print("="*80)
        print(f"Base: {base_model_path}")
        print(f"Adapters: {self.num_adapters}")
        print(f"Trainable: {self.num_adapters} adapter weights (same for all layers)")
        print("="*80 + "\n")

        print("Loading base LLaMA model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )

        for param in base_model.parameters():
            param.requires_grad = False

        print(f"\nLoading {len(adapter_paths)} LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_paths[0],
            adapter_name=adapter_names[0],
            is_trainable=False,
        )

        for adapter_path, adapter_name in zip(adapter_paths[1:], adapter_names[1:]):
            print(f"  Loading {adapter_name} from {adapter_path}")
            self.model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_layers = self._infer_num_layers()
        model_device = next(self.model.parameters()).device

        # One weight per adapter, shape (num_adapters,)
        self.weight_logits = nn.Parameter(
            self._init_weight_logits(
                init_mode=weight_init,
                num_adapters=self.num_adapters,
                device=model_device,
                custom_init_weights=custom_init_weights,
            )
        )

        self._wrap_lora_projections()

        print(f"\nModel ready:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Adapters: {self.num_adapters}")
        print(f"  TRAINABLE params: {self.weight_logits.numel()} (one per adapter)")
        print(f"  Frozen params: All model & adapter parameters")
        print(f"  Wrapped projections: All q/k/v/o and gate/up/down projections")
        print(f"  Device: {model_device}\n")

    def _wrap_lora_projections(self):
        """
        Wrap each LoRA projection with WeightedLoRALinear.
        All wrappers share the same adapter-wise weights (one per adapter).
        """
        print("Wrapping LoRA projections with adapter-wise weights...")

        wrapped_count = 0
        self._wrapped_modules = []

        for module_name, module in self.model.named_modules():
            if isinstance(module, WeightedLoRALinear):
                continue
            if not hasattr(module, "lora_A"):
                continue
            if not self._has_multiple_loras(module):
                continue
            if not hasattr(module, "base_layer"):
                continue

            layer_idx = self._infer_layer_idx(module_name)
            if layer_idx is None:
                continue

            lora_modules = self._extract_lora_modules(module)
            wrapped = WeightedLoRALinear(
                base_layer=module.base_layer,
                lora_modules=lora_modules,
                adapter_names=self.adapter_names,
                weight_getter=self.get_weights,
            )

            parent, attr_name = self._get_parent_module(module_name)
            setattr(parent, attr_name, wrapped)
            wrapped_count += 1
            self._wrapped_modules.append((module_name, layer_idx))

        print(f"Wrapped {wrapped_count} projections across {self.num_layers} layers")

    def _has_multiple_loras(self, module):
        """Check if module has LoRA for multiple adapters"""
        if not hasattr(module, 'lora_A'):
            return False
        count = sum(1 for name in self.adapter_names if name in module.lora_A)
        return count >= 1

    def _extract_lora_modules(self, module):
        """
        Extract LoRA components from a module.

        Returns dict: {adapter_name: {lora_A, lora_B, scaling, dropout}}
        """
        lora_modules = {}

        for adapter_name in self.adapter_names:
            if adapter_name in module.lora_A:
                lora_modules[adapter_name] = {
                    'lora_A': module.lora_A[adapter_name],
                    'lora_B': module.lora_B[adapter_name],
                    'scaling': module.scaling[adapter_name],
                    'dropout': module.lora_dropout[adapter_name]
                }

        return lora_modules

    def _init_weight_logits(
        self,
        init_mode: str,
        num_adapters: int,
        device: torch.device,
        custom_init_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Initialize adapter-wise weight logits. Shape: (num_adapters,).

        Supported modes:
        - "zeros", "ones", "equal", "randint_0_1", "randint_-2_2", "uniform_-2_2"
        - "random_0_1", "random_-1_1", "random_0.5_1.3"
        - "random_0_1_adapter_level", "random_-1_1_adapter_level", "random_0.5_1.3_adapter_level"
        - "style_similarity": requires custom_init_weights of length num_adapters
        """
        import random as _random

        shape = (num_adapters,)

        if init_mode == "zeros":
            return torch.zeros(shape, dtype=torch.float32, device=device)
        if init_mode == "ones":
            return torch.ones(shape, dtype=torch.float32, device=device)
        if init_mode == "equal":
            return torch.full(shape, 1.0 / num_adapters, dtype=torch.float32, device=device)
        if init_mode == "randint_0_1":
            return torch.randint(0, 2, shape, device=device).to(torch.float32)
        if init_mode == "randint_-2_2":
            return torch.randint(-2, 3, shape, device=device).to(torch.float32)
        if init_mode == "uniform_-2_2":
            return torch.empty(shape, dtype=torch.float32, device=device).uniform_(-2.0, 2.0)
        if init_mode == "random_0_1":
            return torch.empty(shape, dtype=torch.float32, device=device).uniform_(0.0, 1.0)
        if init_mode == "random_-1_1":
            return torch.empty(shape, dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
        if init_mode == "random_0.5_1.3":
            return torch.empty(shape, dtype=torch.float32, device=device).uniform_(0.5, 1.3)

        if init_mode == "random_0_1_adapter_level":
            per_adapter = [_random.random() for _ in range(num_adapters)]
            print(f"> Init: random_0_1_adapter_level, per-adapter weights: {per_adapter}")
            return torch.tensor(per_adapter, dtype=torch.float32, device=device).clone()
        if init_mode == "random_-1_1_adapter_level":
            per_adapter = [_random.uniform(-1.0, 1.0) for _ in range(num_adapters)]
            print(f"> Init: random_-1_1_adapter_level, per-adapter weights: {per_adapter}")
            return torch.tensor(per_adapter, dtype=torch.float32, device=device).clone()
        if init_mode == "random_0.5_1.3_adapter_level":
            per_adapter = [_random.uniform(0.5, 1.3) for _ in range(num_adapters)]
            print(f"> Init: random_0.5_1.3_adapter_level, per-adapter weights: {per_adapter}")
            return torch.tensor(per_adapter, dtype=torch.float32, device=device).clone()

        if init_mode == "style_similarity":
            if custom_init_weights is None or len(custom_init_weights) != num_adapters:
                raise ValueError(
                    f"style_similarity init requires custom_init_weights of length {num_adapters}, "
                    f"got {custom_init_weights}"
                )
            print(f"> Init: style_similarity, per-adapter weights: {custom_init_weights}")
            return torch.tensor(custom_init_weights, dtype=torch.float32, device=device).clone()

        raise ValueError(
            f"Unknown weight_init '{init_mode}'. "
            "Choose from: zeros, ones, equal, randint_0_1, randint_-2_2, uniform_-2_2, "
            "random_0_1, random_-1_1, random_0.5_1.3, "
            "random_0_1_adapter_level, random_-1_1_adapter_level, random_0.5_1.3_adapter_level, "
            "style_similarity."
        )

    def _get_parent_module(self, module_name: str):
        """
        Return (parent_module, child_name) for a dotted module name.
        """
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _infer_layer_idx(self, module_name: str):
        """
        Infer layer index from module name using common HF patterns.
        Supports LLaMA/Qwen-like models.
        """
        patterns = [
            r"\bmodel\.layers\.(\d+)\b",
            r"\btransformer\.h\.(\d+)\b",
            r"\bmodel\.h\.(\d+)\b",
            r"\blayers\.(\d+)\b",
            r"\bh\.(\d+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, module_name)
            if match:
                return int(match.group(1))
        return None

    def _infer_num_layers(self) -> int:
        """
        Infer number of layers from config or module names.
        """
        config_layers = getattr(self.model.config, "num_hidden_layers", None)
        if config_layers is not None:
            return int(config_layers)

        max_idx = -1
        for module_name, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            layer_idx = self._infer_layer_idx(module_name)
            if layer_idx is not None:
                max_idx = max(max_idx, layer_idx)
        if max_idx >= 0:
            return max_idx + 1
        raise ValueError("Unable to infer number of layers from model/config.")

    def get_weights(self):
        """
        Return adapter-wise weights, shape (num_adapters,). Same for all layers.
        """
        return self.weight_logits

    def print_weights(self, verbose: bool = False):
        """Print current adapter-wise weights."""
        weights = self.get_all_weights().detach().cpu().numpy()
        print("\n" + "="*80)
        print("ADAPTER-WISE WEIGHTS")
        print("="*80)
        weights_str = ", ".join(
            [f"{name}={w:.4f}" for name, w in zip(self.adapter_names, weights)]
        )
        print(weights_str)
        print("="*80 + "\n")
        return weights.tolist()

    def get_all_weights(self):
        """Get all weights as a tensor of shape (num_adapters,)"""
        return self.weight_logits

    def print_wrapped_modules(self, max_items: int = 50):
        """
        Print wrapped module names with inferred layer indices.
        """
        if not hasattr(self, "_wrapped_modules"):
            print("No wrapped module records found.")
            return []

        total = len(self._wrapped_modules)
        print(f"\nWRAPPED MODULES (showing up to {max_items} of {total})")
        for module_name, layer_idx in self._wrapped_modules[:max_items]:
            print(f"layer {layer_idx:2d}: {module_name}")
        if total > max_items:
            print(f"... {total - max_items} more not shown")
        return self._wrapped_modules

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass - wrapped modules apply adapter-wise weighted LoRA (same weights for all layers).
        output = base(x) + sum_a w_a * lora_a(x). Gradients flow through the adapter weights.
        """

        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        # if self.training:
        #     g = self.weight_logits.grad
        #     if g is None:
        #         print(f"[grad] step: None (no gradient!)")
        #     else:
        #         g = g.detach().float().cpu().tolist()
        #         print(f"[grad] step: {g}")

        # Standard forward - wrappers handle weighted LoRA.
        return model_out

# CRITICAL: I need to add these methods for GRPO generation.

    def generate(self, *args, **kwargs):
        """Delegate generation to inner PeftModel model (required by GRPO!)"""
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Required for generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    @property
    def config(self):
        """Delegate config to inner model."""
        return self.model.config

    @property
    def warnings_issued(self):
        """Delegate warnings_issued to inner model."""
        return self.model.warnings_issued

    def add_model_tags(self, *args, **kwargs):
        """Delegate add_model_tags to inner model."""
        return self.model.add_model_tags(*args, **kwargs)  # type: ignore[misc]

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Delegate gradient_checkpointing_enable to inner model."""
        return self.model.gradient_checkpointing_enable(*args, **kwargs)  # type: ignore[misc]

    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Delegate gradient_checkpointing_disable to inner model."""
        return self.model.gradient_checkpointing_disable(*args, **kwargs)  # type: ignore[misc]

    @property
    def is_gradient_checkpointing(self):
        """Delegate is_gradient_checkpointing to inner model."""
        return self.model.is_gradient_checkpointing
