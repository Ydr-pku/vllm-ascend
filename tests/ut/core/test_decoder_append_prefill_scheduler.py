# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow tests executing scheduler source with fake cache/worker I/O.

The pinned vLLM checkout (or VLLM_SOURCE_ROOT) supplies the real base scheduler,
async methods, queues, and request status. Only imports and hardware/model
construction are excluded. These tests do not establish NPU or API support.
"""

import ast
import enum
import importlib.util
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/core/decoder_append_prefill_scheduler.py"
VLLM = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm")) / "vllm"
if not VLLM.is_dir():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the target vLLM or set VLLM_SOURCE_ROOT to its checkout.")
    VLLM = Path(spec.origin).parent


class Output(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(outputs=[], has_structured_output_requests=False, pending_structured_output_tokens=False)
        self.__dict__.update(kwargs)


class TestRequest(SimpleNamespace):
    __test__ = False
    __hash__ = object.__hash__

    def is_finished(self):
        return self.status.is_finished(self.status)

    def get_finished_reason(self):
        return "stop"

    def append_output_token_ids(self, token):
        self.num_output_tokens += 1
        self.num_tokens += 1

    def take_prefill_stats(self):
        return None

    def take_events(self):
        return ["queued"]


def execute_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def runtime():
    namespace = dict(
        __name__="admission_test_runtime",
        ABC=ABC,
        abstractmethod=abstractmethod,
        deque=deque,
        enum=enum,
        Enum=enum.Enum,
        time=time,
        defaultdict=defaultdict,
        SchedulerInterface=object,
        MULTIMODAL_REGISTRY=None,
        Request=TestRequest,
        EngineCoreOutputs=Output,
        EngineCoreOutput=Output,
        SchedulerOutput=Output,
        KVConnectorBlockState=SimpleNamespace,
        logger=Mock(),
        check_stop=lambda request, max_model_len: False,
        record_function_or_nullcontext=lambda *args: nullcontext(),
        remove_all=lambda seq, removed: [req for req in seq if req not in removed],
        NewRequestData=SimpleNamespace(from_request=lambda request, *args, **kwargs: request),
    )
    execute_classes(VLLM / "v1/request.py", {"RequestStatus"}, namespace)
    execute_classes(VLLM / "v1/core/sched/interface.py", {"PauseState"}, namespace)
    execute_classes(
        VLLM / "v1/core/sched/request_queue.py",
        {"SchedulingPolicy", "RequestQueue", "FCFSRequestQueue", "PriorityRequestQueue"},
        namespace,
    )
    namespace["create_request_queue"] = lambda policy: namespace["FCFSRequestQueue"]()
    execute_classes(VLLM / "v1/core/sched/scheduler.py", {"Scheduler"}, namespace)
    execute_classes(VLLM / "v1/core/sched/async_scheduler.py", {"AsyncScheduler"}, namespace)
    execute_classes(SOURCE, {"DecoderAppendPrefillScheduler", "AsyncDecoderAppendPrefillScheduler"}, namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture(params=["DecoderAppendPrefillScheduler", "AsyncDecoderAppendPrefillScheduler"])
def scheduler(runtime, request):
    cls = getattr(runtime, request.param)
    scheduler = cls.__new__(cls)
    empty_blocks = SimpleNamespace(get_block_ids=lambda: ([],), blocks=([],))
    scheduler.__dict__.update(
        current_step=0,
        max_num_scheduled_tokens=64,
        vllm_config=SimpleNamespace(speculative_config=None, kv_transfer_config=None),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=64, long_prefill_token_threshold=0, enable_chunked_prefill=True
        ),
        _pause_state=runtime.PauseState.UNPAUSED,
        max_num_encoder_input_tokens=0,
        running=[],
        waiting=runtime.FCFSRequestQueue(),
        skipped_waiting=runtime.FCFSRequestQueue(),
        policy=runtime.SchedulingPolicy.FCFS,
        kv_cache_manager=Mock(),
        prefill_capacity_bound=False,
        max_num_active_reqs=16,
        max_num_running_reqs=16,
        num_waiting_for_streaming_input=0,
        lora_config=None,
        connector=None,
        ec_connector=None,
        aux_output_connector=None,
        prefix_replay_tokens=0,
        num_lookahead_tokens=0,
        is_encoder_decoder=False,
        need_mamba_block_aligned_split=False,
        scheduler_reserve_full_isl=False,
        num_spec_tokens=0,
        num_sampled_tokens_per_step=1,
        num_prefill_lookahead=1,
        max_model_len=256,
        block_size=16,
        log_stats=False,
        return_sampling_mask=False,
        is_mm_encoder_only=False,
        structured_output_manager=Mock(accept_tokens=Mock(return_value=True)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
        use_v2_model_runner=False,
        model_uses_mrope=False,
        prev_step_scheduled_req_ids=set(),
        reset_preempted_req_ids=set(),
        finished_req_ids=set(),
        finished_req_ids_dict=defaultdict(set),
        encoder_cache_manager=Mock(),
        dynamic_sd_lookup=None,
        defer_block_free=False,
        sched_step_seq=0,
        processed_step_seq=0,
        deferred_frees=deque(),
        _inflight_prefills=set(),
        requests={},
        _pending_decoder_cache_rejections={},
        _decoder_cache_rejection_batches={},
        perf_metrics=None,
        grammar_compile_error_reqs=set(),
        kv_event_publisher=Mock(),
    )
    manager = scheduler.kv_cache_manager
    manager.empty_kv_cache_blocks = empty_blocks
    manager.get_computed_blocks.return_value = (empty_blocks, 0, 0)
    manager.allocate_slots.return_value = empty_blocks
    manager.get_num_common_prefix_blocks.return_value = [0]
    manager.take_boundary_state_offloads.return_value = {}
    manager.take_kv_cache_block_copies.return_value = ([], [])
    manager.take_events.return_value = None
    scheduler._make_cached_request_data = Mock(return_value=None)
    scheduler._get_new_block_ids_to_zero = Mock(return_value=None)
    scheduler.make_stats = Mock(return_value=None)
    return scheduler


def enqueue(runtime, scheduler, request_id="miss", *, marked=True, hits=0, output_tokens=0):
    request = TestRequest(
        request_id=request_id,
        sampling_params=SimpleNamespace(
            extra_args={"require_decoder_cache_hit": 1} if marked else {}, num_logprobs=None
        ),
        status=runtime.RequestStatus.PREEMPTED if output_tokens else runtime.RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        num_in_flight_tokens=0,
        num_output_placeholders=0,
        drop_stale_output=False,
        lora_request=None,
        shared_prefix_boundary=0,
        num_tokens=32,
        num_prompt_tokens=32,
        prefill_stats=None,
        has_encoder_inputs=False,
        num_output_tokens=output_tokens,
        client_index=2,
        trace_headers={"trace": "test"},
        num_preemptions=int(bool(output_tokens)),
        is_prefill_chunk=True,
        use_structured_output=False,
        spec_token_ids=[],
        skip_reading_prefix_cache=False,
        last_sched_seq=0,
        pooling_params=None,
        stop_reason=None,
        num_nans_in_logits=0,
    )
    scheduler.requests[request_id] = request
    scheduler.waiting.add_request(request)
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        hits,
        0,
    )
    return request


def empty_model_output():
    return SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
    )


def test_miss_is_not_allocated_and_zero_token_batch_finishes(runtime, scheduler):
    req = enqueue(runtime, scheduler)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {}
    assert output.total_num_scheduled_tokens == 0
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    results = scheduler.update_from_output(output, empty_model_output())
    terminal = results[2].outputs[0]
    assert (terminal.request_id, terminal.new_token_ids, terminal.stop_reason) == ("miss", [], "decoder_cache_miss")
    assert terminal.trace_headers == req.trace_headers
    assert terminal.events == ["queued"]
    assert results[2].finished_requests == {"miss"}
    assert not scheduler.requests and not scheduler.waiting and not scheduler.skipped_waiting
    scheduler.kv_cache_manager.free.assert_called_once_with(req)
    assert not scheduler._pending_decoder_cache_rejections
    assert not scheduler._decoder_cache_rejection_batches


@pytest.mark.parametrize("marked,hits,expected_tokens", [(False, 0, 32), (True, 16, 16)])
def test_normal_or_resident_request_is_scheduled(runtime, scheduler, marked, hits, expected_tokens):
    req = enqueue(runtime, scheduler, marked=marked, hits=hits)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"miss": expected_tokens}
    assert scheduler.kv_cache_manager.allocate_slots.call_args.kwargs["num_new_computed_tokens"] == hits
    assert not scheduler._pending_decoder_cache_rejections
    assert req.num_in_flight_tokens == expected_tokens
    expected_placeholders = int(isinstance(scheduler, runtime.AsyncScheduler))
    assert req.num_output_placeholders == expected_placeholders


def test_connector_cannot_rescue_a_local_miss(runtime, scheduler):
    enqueue(runtime, scheduler)
    scheduler.connector = Mock(supports_divergent_local_hybrid_hits=True)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {}
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
    scheduler.kv_cache_manager.get_computed_blocks_for_connector.assert_not_called()
    scheduler.connector.update_state_after_alloc.assert_not_called()


def test_marked_hit_also_bypasses_remote_matching(runtime, scheduler):
    enqueue(runtime, scheduler, hits=16)
    scheduler.connector = Mock(supports_divergent_local_hybrid_hits=True)
    scheduler.connector_prefix_cache_stats = None
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"miss": 16}
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
    scheduler.connector.update_state_after_alloc.assert_called_once()


def test_allocation_retry_rechecks_residency(runtime, scheduler):
    enqueue(runtime, scheduler, hits=16)
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {}
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        0,
        0,
    )
    second = scheduler.schedule()
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 1
    assert not scheduler.update_from_output(first, empty_model_output())
    assert scheduler.update_from_output(second, empty_model_output())[2].outputs[0].stop_reason == "decoder_cache_miss"


def test_pending_rejection_waits_for_its_own_batch(runtime, scheduler):
    first = scheduler.schedule()
    enqueue(runtime, scheduler)
    rejected = scheduler.schedule()
    later = scheduler.schedule()
    assert not scheduler.update_from_output(first, empty_model_output())
    assert "miss" in scheduler.requests
    assert later.num_scheduled_tokens == {}
    assert scheduler.kv_cache_manager.get_computed_blocks.call_count == 1
    assert (
        scheduler.update_from_output(rejected, empty_model_output())[2].outputs[0].stop_reason == "decoder_cache_miss"
    )
    assert not scheduler.update_from_output(later, empty_model_output())


@pytest.mark.parametrize("reuse_id", [False, True])
def test_abort_does_not_emit_rejection_or_finish_reused_id(runtime, scheduler, reuse_id):
    req = enqueue(runtime, scheduler)
    output = scheduler.schedule()
    scheduler.finish_requests("miss", runtime.RequestStatus.FINISHED_ABORTED)
    if reuse_id:
        enqueue(runtime, scheduler)
    results = scheduler.update_from_output(output, empty_model_output())
    assert all(not result.outputs for result in results.values())
    assert req.status == runtime.RequestStatus.FINISHED_ABORTED
    assert bool(scheduler.requests) == reuse_id


def test_post_output_miss_cannot_be_replayed(runtime, scheduler):
    enqueue(runtime, scheduler, output_tokens=3)
    output = scheduler.schedule()
    result = scheduler.update_from_output(output, empty_model_output())[2].outputs[0]
    assert result.stop_reason == "decoder_cache_lost_after_output"
    assert result.new_token_ids == []


def test_deliverable_stale_output_defers_admission(runtime, scheduler):
    req = enqueue(runtime, scheduler)
    req.num_stale_output_tokens = 1
    output = scheduler.schedule()
    scheduler.kv_cache_manager.get_computed_blocks.assert_not_called()
    assert not scheduler.update_from_output(output, empty_model_output())
    req.num_stale_output_tokens = 0
    req.num_output_tokens = 1
    output = scheduler.schedule()
    assert (
        scheduler.update_from_output(output, empty_model_output())[2].outputs[0].stop_reason
        == "decoder_cache_lost_after_output"
    )


def test_miss_does_not_block_an_unmarked_request(runtime, scheduler):
    enqueue(runtime, scheduler)
    enqueue(runtime, scheduler, "normal", marked=False)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"normal": 32}


@pytest.mark.parametrize("hits,rejected", [(0, True), (16, True), (32, False)])
def test_prefix_replay_requires_a_useful_local_prefix(runtime, scheduler, hits, rejected):
    enqueue(runtime, scheduler, hits=hits)
    scheduler.prefix_replay_tokens = 16
    output = scheduler.schedule()
    assert bool(scheduler._pending_decoder_cache_rejections) == rejected
    assert bool(output.num_scheduled_tokens) != rejected


def test_only_documented_admission_deltas_change_schedule():
    """Catch accidental loss of upstream scheduling behavior during refreshes."""

    def schedule_tree(path, class_name):
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")

    class StripAdmission(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "rejected_requests":
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            condition = ast.unparse(node.test)
            if "_pending_decoder_cache_rejections" in condition or "_should_reject_decoder_cache_miss" in condition:
                return None
            if "_requires_decoder_cache_hit" in condition:
                node.test = node.test.values[0]
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if "self._record_decoder_cache_rejections(" in ast.unparse(node):
                return None
            return self.generic_visit(node)

    actual = StripAdmission().visit(schedule_tree(SOURCE, "DecoderAppendPrefillScheduler"))
    upstream = schedule_tree(VLLM / "v1/core/sched/scheduler.py", "Scheduler")
    assert ast.dump(actual) == ast.dump(upstream)


@pytest.mark.parametrize("async_enabled", [False, True])
def test_constructor_preserves_async_setup(runtime, monkeypatch, async_enabled):
    policy = SimpleNamespace(
        enable_balance_scheduling=False,
        recompute_scheduler_enable=False,
        **{
            key: SimpleNamespace(enabled=False)
            for key in (
                "short_request_first_config",
                "profiling_chunk_config",
                "batch_job_sched_config",
                "dyntra_lb_config",
            )
        },
    )
    init_globals = runtime.DecoderAppendPrefillScheduler.__init__.__globals__
    monkeypatch.setitem(init_globals, "init_ascend_config", lambda config: SimpleNamespace(scheduler_config=policy))

    def base_init(self, *args, **kwargs):
        self.num_spec_tokens = 2
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    monkeypatch.setattr(runtime.Scheduler, "__init__", base_init)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_enabled))
    cls = runtime.AsyncDecoderAppendPrefillScheduler if async_enabled else runtime.DecoderAppendPrefillScheduler
    instance = cls(vllm_config=config)
    assert instance._pending_decoder_cache_rejections == {}
    assert instance._decoder_cache_rejection_batches == {}
    if async_enabled:
        assert instance._spec_token_placeholders == [-1, -1]
    config.scheduler_config.async_scheduling = not async_enabled
    with pytest.raises(ValueError, match="matching async_scheduling"):
        cls(vllm_config=config)
    policy.recompute_scheduler_enable = True
    with pytest.raises(ValueError, match="recompute"):
        cls(vllm_config=config)


def test_inflight_preempted_output_is_delivered_before_cache_loss(runtime, scheduler):
    req = enqueue(runtime, scheduler, hits=16)
    first = scheduler.schedule()
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        0,
        0,
    )
    waiting = scheduler.schedule()
    assert not scheduler._pending_decoder_cache_rejections
    model_output = empty_model_output()
    model_output.sampled_token_ids = [[42]]
    model_output.req_id_to_index = {"miss": 0}
    result = scheduler.update_from_output(first, model_output)
    assert result[2].outputs[0].new_token_ids == [42]
    assert req.num_stale_output_tokens == req.num_output_placeholders == 0
    assert not scheduler.update_from_output(waiting, empty_model_output())
    rejected = scheduler.schedule()
    result = scheduler.update_from_output(rejected, empty_model_output())
    assert result[2].outputs[0].stop_reason == "decoder_cache_lost_after_output"


def test_mixed_batch_delivers_normal_output_and_rejection(runtime, scheduler):
    enqueue(runtime, scheduler)
    normal = enqueue(runtime, scheduler, "normal", marked=False)
    output = scheduler.schedule()
    model_output = empty_model_output()
    model_output.sampled_token_ids = [[42]]
    model_output.req_id_to_index = {"normal": 0}
    result = scheduler.update_from_output(output, model_output)
    assert [(item.request_id, item.new_token_ids) for item in result[2].outputs] == [("normal", [42]), ("miss", [])]
    assert scheduler.running == [normal]
    assert set(scheduler.requests) == {"normal"}
    if isinstance(scheduler, runtime.AsyncScheduler):
        assert normal.num_output_placeholders == 0
        scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(normal, 32)
