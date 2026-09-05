# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared P/D protocol with lazily loaded connector implementations."""

import importlib
from typing import Any

_EXPORTS = {
    "P2pBaseConnectorScheduler": ".base_scheduler",
    "P2pBaseConnectorWorker": ".base_worker",
    "P2pBaseConnector": ".connector",
    "P2pConnector": ".connector",
    "P2pPullConnector": ".connector",
    "P2pPushConnector": ".connector",
    "P2pAgentMetadata": ".metadata",
    "P2pConnectorMetadata": ".metadata",
    "P2pHandshakePayload": ".metadata",
    "P2pPullConnectorScheduler": ".pull_scheduler",
    "P2pPullConnectorWorker": ".pull_worker",
    "P2pPushConnectorScheduler": ".push_scheduler",
    "P2pPushConnectorWorker": ".push_worker",
    "P2pConnectorScheduler": ".scheduler",
    "P2pKVConnectorStats": ".stats",
    "P2pConnectorWorker": ".worker",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(importlib.import_module(_EXPORTS[name], __package__), name)
    globals()[name] = value
    return value
