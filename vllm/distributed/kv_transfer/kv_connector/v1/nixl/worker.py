# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.worker import (
    P2pConnectorWorker as NixlConnectorWorker,
)

__all__ = ["NixlConnectorWorker"]
