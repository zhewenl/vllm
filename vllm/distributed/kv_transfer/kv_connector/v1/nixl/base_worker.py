# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_worker import (
    P2pBaseConnectorWorker as NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_worker import (
    _share_storage_and_block_stride as _share_storage_and_block_stride,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_worker import (
    _tensor_byte_span_end as _tensor_byte_span_end,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_worker import (
    _uses_dense_virtual_transfer_pages as _uses_dense_virtual_transfer_pages,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_worker import (
    logger as logger,
)

__all__ = [
    "_share_storage_and_block_stride",
    "_tensor_byte_span_end",
    "_uses_dense_virtual_transfer_pages",
    "NixlBaseConnectorWorker",
    "logger",
]
