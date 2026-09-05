# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine compatibility layered over the unchanged P/D metadata schema."""

from vllm.config import VllmConfig
from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)

from .transport import get_transfer_engine


def compute_p2p_compatibility_hash(
    vllm_config: VllmConfig,
    attn_backend_name: str,
    transfer_mode: str = "pull",
) -> str:
    protocol_hash = compute_nixl_compatibility_hash(
        vllm_config, attn_backend_name, transfer_mode
    )
    engine = get_transfer_engine(vllm_config)
    if engine == "nixl":
        return protocol_hash
    return hash_factors(
        {
            "protocol": protocol_hash,
            "transfer_engine": engine,
            "transport_protocol_version": 1,
        }
    )
