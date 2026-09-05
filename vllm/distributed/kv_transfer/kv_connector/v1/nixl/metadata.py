# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    GET_META_MSG as GET_META_MSG,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    P2P_CONNECTOR_VERSION as NIXL_CONNECTOR_VERSION,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    PUSH_REG_NOTIF_PREFIX as PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    HeartbeatInfo as HeartbeatInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    P2pAgentMetadata as NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    P2pConnectorMetadata as NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    P2pHandshakePayload as NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    RemoteMeta as RemoteMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    ReqId as ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    ReqMeta as ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    TransferHandle as TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    _get_speculative_compatibility_factors as _get_speculative_compatibility_factors,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    compute_p2p_compatibility_hash as compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
    logger as logger,
)

__all__ = [
    "NixlAgentMetadata",
    "NixlHandshakePayload",
    "_get_speculative_compatibility_factors",
    "compute_nixl_compatibility_hash",
    "HeartbeatInfo",
    "RemoteMeta",
    "ReqMeta",
    "NixlConnectorMetadata",
    "logger",
    "TransferHandle",
    "ReqId",
    "GET_META_MSG",
    "PUSH_REG_NOTIF_PREFIX",
    "NIXL_CONNECTOR_VERSION",
]
