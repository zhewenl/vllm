# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeStoreConnector HMA support."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)

from .utils import create_vllm_config, make_kv_cache_config


def _make_scheduler(
    swa_enabled: bool,
    sw_size: int = 2048,
    block_size: int = 16,
    disable_hma: bool = False,
):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled, sw_size=sw_size,
    )
    return MooncakeStoreScheduler(vllm_config, kv_cache_config=kv_cache_config)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_blocks_per_sw",
    [
        (True, [0, 2048 // 16 + 1]),  # FA=0, SWA=128+1=129
        (False, [0]),
    ],
)
def test_blocks_per_sw(swa_enabled, expected_blocks_per_sw):
    scheduler = _make_scheduler(swa_enabled=swa_enabled)
    assert scheduler.blocks_per_sw == expected_blocks_per_sw


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,disable_hma,expected_is_hma",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_is_hma_required(swa_enabled, disable_hma, expected_is_hma):
    scheduler = _make_scheduler(swa_enabled=swa_enabled, disable_hma=disable_hma)
    assert scheduler._is_hma_required is expected_is_hma


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_clips_swa_keeps_fa():
    # sw_size=128 tokens / block_size=16 = 8 blocks + 1 = 9 blocks_per_sw
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128)
    assert scheduler.blocks_per_sw == [0, 9]

    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))
    clipped = scheduler.get_sw_clipped_blocks((fa_blocks, sw_blocks))

    assert clipped[0] == fa_blocks            # FA untouched
    assert clipped[1] == sw_blocks[-9:]       # SWA clipped to last 9
    assert len(clipped[1]) == 9


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_noop_when_not_hma():
    scheduler = _make_scheduler(swa_enabled=False)
    assert scheduler._is_hma_required is False
    block_ids = ([1, 2, 3],)
    assert scheduler.get_sw_clipped_blocks(block_ids) == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# Task 2: per-group block_ids end-to-end
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (  # noqa: E402,E501
    RequestTracker,
)


@pytest.mark.cpu_test
def test_request_tracker_update_extends_each_group():
    """update() must extend the matching group, not flatten."""
    tracker = RequestTracker(
        req_id="r1", token_len=0, allocated_block_ids=[[1, 2], [10, 11]],
    )
    tracker.update(([3, 4], [12, 13]))
    assert tracker.allocated_block_ids == [[1, 2, 3, 4], [10, 11, 12, 13]]


@pytest.mark.cpu_test
def test_request_tracker_update_accepts_flat_list_as_single_group():
    """Legacy flat list extends group 0 (single-group compat)."""
    tracker = RequestTracker(
        req_id="r1", token_len=0, allocated_block_ids=[[1, 2]],
    )
    tracker.update([3, 4])
    assert tracker.allocated_block_ids == [[1, 2, 3, 4]]


@pytest.mark.cpu_test
def test_scheduler_request_finished_clips_swa_group():
    """request_finished must clip SWA group on the way to delay-free state."""
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128, block_size=16)
    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))

    request = MagicMock()
    request.request_id = "r-finished"
    request.kv_transfer_params = {}
    scheduler._request_trackers["r-finished"] = RequestTracker(
        req_id="r-finished",
        token_len=20 * 16,
        allocated_block_ids=[fa_blocks, sw_blocks],
        num_saved_tokens=20 * 16,
    )

    delay, _ = scheduler.request_finished(request, (fa_blocks, sw_blocks))
    assert delay is True
