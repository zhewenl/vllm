# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.connector import (
    P2pBaseConnector as NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.connector import (
    P2pConnector as NixlConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.connector import (
    P2pPullConnector as NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.connector import (
    P2pPushConnector as NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.connector import (
    logger as logger,
)

__all__ = [
    "NixlBaseConnector",
    "NixlPullConnector",
    "NixlPushConnector",
    "logger",
    "NixlConnector",
]
