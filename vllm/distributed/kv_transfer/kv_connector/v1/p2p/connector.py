# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in connectors; existing NixlConnector classes keep their native path."""

from typing import ClassVar

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)

from .stats import P2pPromMetrics
from .worker import P2pPullConnectorWorker, P2pPushConnectorWorker


class P2pBaseConnector(NixlBaseConnector):
    _push: ClassVar[bool] = False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self._push and vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError("P2pPushConnector does not support DCP > 1")
        if role == KVConnectorRole.SCHEDULER:
            scheduler_cls = (
                NixlPushConnectorScheduler if self._push else NixlPullConnectorScheduler
            )
            scheduler = scheduler_cls(vllm_config, self.engine_id, kv_cache_config)
            self.connector_scheduler = scheduler
            scheduler.side_channel_host = self.kv_transfer_config.get_from_extra_config(
                "side_channel_host", scheduler.side_channel_host
            )
            port = self.kv_transfer_config.get_from_extra_config(
                "side_channel_port", None
            )
            if port is not None:
                scheduler.side_channel_port = (
                    port + vllm_config.parallel_config.data_parallel_index
                )
        elif role == KVConnectorRole.WORKER:
            worker_cls = (
                P2pPushConnectorWorker if self._push else P2pPullConnectorWorker
            )
            self.connector_worker = worker_cls(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    @classmethod
    def build_prom_metrics(
        cls, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ):
        return P2pPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def start_load_kv(self, forward_context, **kwargs):
        assert isinstance(
            self.connector_worker, (P2pPullConnectorWorker, P2pPushConnectorWorker)
        )
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)


class P2pPullConnector(P2pBaseConnector):
    """READ-based P/D transfer using the selected engine."""


class P2pPushConnector(P2pBaseConnector):
    """WRITE-based P/D transfer using the selected engine."""

    _push = True


P2pConnector = P2pPullConnector
