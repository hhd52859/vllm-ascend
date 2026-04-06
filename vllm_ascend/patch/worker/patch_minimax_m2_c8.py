#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MiniMax-M2 C8 KV cache weight interception patch.
#
# The W8A8C8 checkpoint stores per-channel KV cache scales/offsets:
#   layers.N.self_attn.k_proj.kv_cache_scale  (shape [kv_heads * head_dim])
#   layers.N.self_attn.k_proj.kv_cache_offset
#   layers.N.self_attn.v_proj.kv_cache_scale
#   layers.N.self_attn.v_proj.kv_cache_offset
#
# MiniMaxM2Model.load_weights has a stacked_params_mapping that replaces
# "k_proj" → "qkv_proj" in any key containing "k_proj", causing
# k_proj.kv_cache_offset → qkv_proj.kv_cache_offset → KeyError.
#
# patch_minimax_m2.py wraps MiniMaxM2Model.load_weights and stores the
# original as module-level `_original_load_weights`.  We must replace
# that reference with our filtered version so the stacked_params_mapping
# never sees the kv_cache_scale/offset keys.

from collections.abc import Iterable

import torch
import vllm_ascend.patch.worker.patch_minimax_m2 as _mm2_patch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model

# Mapping from checkpoint suffix to vLLM attention parameter suffix
_C8_SUFFIX_MAP = {
    "self_attn.k_proj.kv_cache_scale":  "self_attn.attn.k_cache_scale",
    "self_attn.k_proj.kv_cache_offset": "self_attn.attn.k_cache_offset",
    "self_attn.v_proj.kv_cache_scale":  "self_attn.attn.v_cache_scale",
    "self_attn.v_proj.kv_cache_offset": "self_attn.attn.v_cache_offset",
}

# The true original (pre-patch_minimax_m2) load_weights
_bare_original_load_weights = _mm2_patch._original_load_weights


def _c8_filtered_load_weights(
    self: "MiniMaxM2Model",
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    params_dict = dict(self.named_parameters())
    c8_active = any("attn.k_cache_scale" in k for k in params_dict)

    if not c8_active:
        # Not a C8 model: filter orphan kv_cache keys to prevent KeyError
        def _filter(w):
            for name, t in w:
                if "kv_cache_scale" not in name and "kv_cache_offset" not in name:
                    yield name, t
        return _bare_original_load_weights(self, _filter(weights))

    c8_loaded: set[str] = set()

    def _intercept(raw_weights: Iterable[tuple[str, torch.Tensor]]):
        for name, loaded_weight in raw_weights:
            intercepted = False
            for src_sfx, dst_sfx in _C8_SUFFIX_MAP.items():
                if name.endswith(src_sfx):
                    dst_name = name[: -len(src_sfx)] + dst_sfx
                    if dst_name in params_dict:
                        param = params_dict[dst_name]
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, loaded_weight.squeeze())
                        c8_loaded.add(dst_name)
                    intercepted = True
                    break
            if not intercepted:
                yield name, loaded_weight

    loaded_params = _bare_original_load_weights(self, _intercept(weights))
    loaded_params.update(c8_loaded)
    return loaded_params


# Replace the reference that patch_minimax_m2._patched_load_weights uses.
# That wrapper calls `_original_load_weights(self, weights)` — by replacing
# the module-level name we redirect it through our C8 filter.
_mm2_patch._original_load_weights = _c8_filtered_load_weights
