# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of P2pPullConnectorWorker."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_worker import (
    P2pPullConnectorWorker,
)

# Backward compatibility: P2pConnectorWorker is the pull-based worker.
P2pConnectorWorker = P2pPullConnectorWorker


__all__ = ["P2pConnectorWorker", "P2pPullConnectorWorker"]
