# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.scheduler import (
    P2pConnectorScheduler as NixlConnectorScheduler,
)

__all__ = ["NixlConnectorScheduler"]
