# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import threading
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)


def _make_store_sending_thread(
    store: MagicMock,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    disk_offload_buffer_budget_bytes: int | None = None,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_load_req(
    req_id: str,
    block_hashes: list[bytes],
    *,
    token_len: int,
    vllm_cached_tokens: int = 0,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=list(range(len(block_hashes))),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
        ),
    )


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=[0, 1],
        block_hashes=block_hashes,
        can_save=True,
        original_block_size=16,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = (
    mooncake_store_worker._estimate_disk_offload_staging_bytes([256])
)
_DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS = 4 * _DISK_OFFLOAD_SINGLE_KEY_BYTES
_DISK_OFFLOAD_BUDGET_FOR_SPLIT = math.ceil(
    2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES / _DISK_OFFLOAD_USABLE_BUDGET_RATIO
)  # Allows two 256-byte chunks but not the third.
_DISK_OFFLOAD_BUDGET_TOO_SMALL = (
    _DISK_OFFLOAD_SINGLE_KEY_BYTES - 1
)  # Smaller than a single 256-byte chunk.


def test_store_sending_thread_skips_request_during_cpu_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        [256, 256],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-b")
    thread._handle_request(_make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    thread.add_stored_request("req-a")
    thread._handle_request(_make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


def test_get_disk_offload_buffer_budget_bytes_uses_effective_offload_flag(
    monkeypatch,
):
    monkeypatch.delenv("MOONCAKE_ENABLE_OFFLOAD", raising=False)
    monkeypatch.setenv("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "2mb")

    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes(enable_offload=True)
        == 2 * 1024 * 1024
    )
    assert (
        mooncake_store_worker._get_disk_offload_buffer_budget_bytes(
            enable_offload=False
        )
        is None
    )


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert (
        mooncake_store_worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288
    )


def test_recv_thread_uses_single_batch_when_no_disk_offload_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    assert sizes == [[256], [256], [256]]


def test_recv_thread_uses_ratio_scaled_budget_for_first_pass_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]


def test_recv_thread_splits_disk_offload_loads_by_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2

    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    first_addrs = store.batch_get_into_multi_buffers.call_args_list[0].args[1]
    second_addrs = store.batch_get_into_multi_buffers.call_args_list[1].args[1]
    first_sizes = store.batch_get_into_multi_buffers.call_args_list[0].args[2]
    second_sizes = store.batch_get_into_multi_buffers.call_args_list[1].args[2]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]
    base_addr = thread.token_database.kv_caches_base_addr[0]
    block_len = thread.token_database.block_len[0]
    assert first_addrs == [[base_addr], [base_addr + block_len]]
    assert second_addrs == [[base_addr + 2 * block_len]]
    expected_size = block_len
    assert first_sizes == [[expected_size], [expected_size]]
    assert second_sizes == [[expected_size]]


def test_recv_thread_stops_after_first_failing_disk_offload_sub_batch():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1


def test_recv_thread_uses_soft_key_cap_for_disk_offload_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6131",
    ]
    assert store.batch_get_into_multi_buffers.call_args_list[1].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0"],
        token_len=16,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0
