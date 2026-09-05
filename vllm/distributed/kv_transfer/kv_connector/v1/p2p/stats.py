# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse existing metrics with engine-selected names and normalized telemetry."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)

from .transport import TransferTelemetry, get_transfer_engine


class P2pKVConnectorStats(NixlKVConnectorStats):
    def record_transfer(self, res: TransferTelemetry):
        self.data["transfer_duration"].append(res.duration_seconds)
        self.data["post_duration"].append(res.post_seconds)
        self.data["bytes_transferred"].append(res.bytes_transferred)
        self.data["num_descriptors"].append(res.descriptor_count)


class P2pPromMetrics(NixlPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
            metric_prefix=get_transfer_engine(vllm_config),
        )
