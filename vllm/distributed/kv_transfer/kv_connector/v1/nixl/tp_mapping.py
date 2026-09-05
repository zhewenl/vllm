# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tp_mapping import (
    ReadSpec as ReadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tp_mapping import (
    TPMapping as TPMapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tp_mapping import (
    _is_attention_spec as _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tp_mapping import (
    _is_ssm_spec as _is_ssm_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tp_mapping import (
    compute_tp_mapping as compute_tp_mapping,
)

__all__ = [
    "ReadSpec",
    "_is_attention_spec",
    "_is_ssm_spec",
    "TPMapping",
    "compute_tp_mapping",
]
