# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_worker import (
    _KV_BLOCKS_EXPIRY_SAFETY_MARGIN as _KV_BLOCKS_EXPIRY_SAFETY_MARGIN,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_worker import (
    P2pPullConnectorWorker as NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_worker import (
    logger as logger,
)

__all__ = ["NixlPullConnectorWorker", "logger", "_KV_BLOCKS_EXPIRY_SAFETY_MARGIN"]
