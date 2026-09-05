# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility imports for the shared P/D connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.utils import (
    _RANDOM_SUFFIX_RE as _RANDOM_SUFFIX_RE,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.utils import (
    get_base_request_id as get_base_request_id,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.utils import (
    get_representative_spec_type as get_representative_spec_type,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.utils import (
    zmq_ctx as zmq_ctx,
)
from vllm.platforms import current_platform

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


__all__ = [
    "zmq_ctx",
    "get_representative_spec_type",
    "get_base_request_id",
    "_NIXL_SUPPORTED_DEVICE",
    "_RANDOM_SUFFIX_RE",
]
