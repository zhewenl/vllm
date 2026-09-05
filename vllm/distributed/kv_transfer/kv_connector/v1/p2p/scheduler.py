# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of P2pPullConnectorScheduler."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_scheduler import (
    P2pPullConnectorScheduler,
)

# Backward compatibility: P2pConnectorScheduler is the pull-based scheduler.
P2pConnectorScheduler = P2pPullConnectorScheduler

__all__ = ["P2pConnectorScheduler", "P2pPullConnectorScheduler"]
