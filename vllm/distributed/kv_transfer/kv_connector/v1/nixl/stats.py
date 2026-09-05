# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.stats import (
    P2pKVConnectorStats as NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.stats import (
    P2pPromMetrics as NixlPromMetrics,
)

__all__ = ["NixlKVConnectorStats", "NixlPromMetrics"]
