import ast
import asyncio
import importlib.util
import logging
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TokenTensor:
    """Only the tensor size is relevant; no token contents are retained."""

    def __init__(self, size=1):
        self.size = size

    def numel(self):
        return self.size


def output(**changes):
    values = dict(
        hidden_states=None,
        all_hidden_states=None,
        output_ids=TokenTensor(),
        input_ids=None,
        finished=True,
        aux_info=SimpleNamespace(input_len=8, output_len=1, reuse_len=0, pd_sep=False),
        loss=None,
        logits=None,
        all_probs=None,
        prompt_logits=None,
        custom_output=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def frame(*outputs):
    return SimpleNamespace(generate_outputs=list(outputs))


async def stream(frames, error=None):
    for item in frames:
        yield item
    if error is not None:
        raise error


def invalid_streams():
    yield "zero frames", []
    yield "empty terminal", [frame()]
    yield "empty after valid", [frame(output()), frame()]
    yield "unfinished", [frame(output(finished=False))]
    yield "missing aux", [frame(output(aux_info=None))]
    for field, value in (
        ("input_len", 7),
        ("output_len", 0),
        ("output_len", 2),
        ("reuse_len", 1),
        ("pd_sep", True),
    ):
        terminal = output()
        setattr(terminal.aux_info, field, value)
        yield f"invalid {field}={value}", [frame(terminal)]
    for size in (0, 2):
        yield f"output tensor size={size}", [
            frame(output(output_ids=TokenTensor(size)))
        ]
    yield "missing output tensor", [frame(output(output_ids=None))]
    yield "invalid second output", [frame(output(), output(finished=False))]


def load_startup_functions():
    """Exercise real orchestration without importing the engine/GPU dependencies."""
    path = Path(__file__).resolve().parents[2] / "start_server.py"
    source = ast.parse(path.read_text(), filename=str(path))
    names = {
        "_run_startup_real_warmup_grpc",
        "_maybe_run_startup_real_warmup",
        "start_server",
    }
    functions = [node for node in source.body if getattr(node, "name", None) in names]
    namespace = dict(
        logging=logging,
        time=time,
        traceback=traceback,
        PyEnvConfigs=object,
        STARTUP_REAL_WARMUP_TOKEN_ID=100,
        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS=1,
        StartupRealWarmupAddressResolutionError=type(
            "AddressError", (RuntimeError,), {}
        ),
        _get_startup_real_warmup_token_lens=lambda _: [8],
        _get_startup_real_warmup_max_len=lambda _: 16,
        _get_startup_real_warmup_speculative_reserve_step=lambda _: 1,
        _get_startup_real_warmup_grpc_addresses=lambda _: ["test:1"],
        _get_startup_real_warmup_timeout_s=lambda: 10,
        _new_startup_real_warmup_request_id=lambda _: 123,
        _get_startup_real_warmup_request_token_len=lambda *_: 8,
        _get_startup_real_warmup_max_new_tokens=lambda: 1,
        _should_run_startup_real_warmup=lambda _: True,
        _run_startup_real_warmup_async=asyncio.run,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class StartupWarmupCompletionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "startup_warmup.py"
        spec = importlib.util.spec_from_file_location("startup_warmup_under_test", path)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)

    async def test_rejects_invalid_terminal_completion(self):
        for name, frames in invalid_streams():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                await self.helper.consume_startup_warmup(stream(frames), 8)

    async def test_accepts_single_frame(self):
        terminal = output()
        frame_count, aux = await self.helper.consume_startup_warmup(
            stream([frame(terminal)]), 8
        )
        self.assertEqual(frame_count, 1)
        self.assertIs(aux, terminal.aux_info)

    async def test_accepts_multiple_rpc_frames_without_scheduler_chunk_assumption(self):
        terminal = output()
        frame_count, aux = await self.helper.consume_startup_warmup(
            stream([frame(), frame(output(finished=False)), frame(terminal)]), 8
        )
        self.assertEqual(frame_count, 3)
        self.assertIs(aux, terminal.aux_info)

    async def test_propagates_stream_failure_after_valid_frame(self):
        error = LookupError("RPC failed")
        with self.assertRaises(LookupError) as caught:
            await self.helper.consume_startup_warmup(
                stream([frame(output())], error), 8
            )
        self.assertIs(caught.exception, error)

    async def test_propagates_cancellation(self):
        error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await self.helper.consume_startup_warmup(stream([], error), 8)
        self.assertIs(caught.exception, error)


class StartupWarmupIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_startup_functions()
        self.frames = [frame(output())]
        self.error = None
        self.closed = False
        self.requests = []
        test = self

        class RpcClient:
            def __init__(self, **_):
                pass

            def enqueue(self, request):
                test.requests.append(request)
                return stream(test.frames, test.error)

            async def close(self):
                test.closed = True

        modules = {
            "torch": SimpleNamespace(
                full=lambda *_args, **_kwargs: TokenTensor(8), int32=object()
            ),
            "rtp_llm.config.generate_config": SimpleNamespace(
                GenerateConfig=SimpleNamespace
            ),
            "rtp_llm.cpp.model_rpc.model_rpc_client": SimpleNamespace(
                ModelRpcClient=RpcClient
            ),
            "rtp_llm.utils.base_model_datatypes": SimpleNamespace(
                GenerateInput=SimpleNamespace
            ),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.config = SimpleNamespace(grpc_config=None)

    def run_warmup(self):
        asyncio.run(self.namespace["_run_startup_real_warmup_grpc"](self.config))

    def test_rejects_invalid_completion_and_closes_client(self):
        for name, frames in invalid_streams():
            with self.subTest(name=name):
                self.frames = frames
                self.closed = False
                with self.assertRaises(RuntimeError):
                    self.run_warmup()
                self.assertTrue(self.closed)

    def test_accepts_single_and_multiple_rpc_frames(self):
        for frames in (
            [frame(output())],
            [frame(output(finished=False)), frame(output())],
        ):
            with self.subTest(frame_count=len(frames)):
                self.frames = frames
                self.run_warmup()
                self.assertTrue(self.closed)
        config = self.requests[0].generate_config
        self.assertEqual(config.max_new_tokens, 1)
        self.assertFalse(config.can_use_pd_separation)
        self.assertFalse(config.reuse_cache)
        self.assertTrue(config.aux_info)

    def test_original_stream_error_propagates_and_client_closes(self):
        self.error = LookupError("RPC failed")
        with self.assertRaises(LookupError) as caught:
            self.run_warmup()
        self.assertIs(caught.exception, self.error)
        self.assertTrue(self.closed)

    def test_maybe_warmup_propagates_validation_failure(self):
        self.frames = []
        with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError):
            self.namespace["_maybe_run_startup_real_warmup"](self.config)

    def test_start_server_does_not_mark_ready_after_invalid_completion(self):
        self.frames = []
        ready = []
        failed = []
        released = []
        manager = SimpleNamespace(
            add_process=lambda *_args, **_kwargs: None,
            add_processes=lambda *_args, **_kwargs: None,
            run_health_checks=lambda: True,
            shutdown_requested=False,
            request_failure_shutdown=lambda: failed.append(True),
            monitor_and_release_processes=lambda: released.append(True),
        )
        noop = lambda *_args, **_kwargs: None
        self.namespace.update(
            configure_warmup=noop,
            multiprocessing=SimpleNamespace(set_start_method=noop),
            torch=SimpleNamespace(
                multiprocessing=SimpleNamespace(set_start_method=noop)
            ),
            init_controller=noop,
            _sync_server_shutdown_timeout=noop,
            ProcessManager=lambda **_: manager,
            VitSeparation=SimpleNamespace(VIT_SEPARATION_ROLE="vit"),
            RoleType=SimpleNamespace(
                VIT="vit", FRONTEND="frontend", PDFUSION="pdfusion"
            ),
            _setup_startup_warmup_health_gate=lambda _: "gate",
            _mark_startup_warmup_health_gate_ready=lambda gate: ready.append(gate),
            start_backend_server_impl=noop,
            start_frontend_server_impl=noop,
            start_dash_sc_server_impl=noop,
        )
        self.config.runtime_config = SimpleNamespace(warm_up=True, model_warm_up=True)
        self.config.parallelism_config = SimpleNamespace(dp_size=1)
        self.config.concurrency_config = None
        self.config.vit_config = SimpleNamespace(vit_separation=None)
        self.config.role_config = SimpleNamespace(role_type="prefill")
        self.config.server_config = SimpleNamespace(
            shutdown_timeout=1,
            monitor_interval=1,
            frontend_pre_stop_drain_seconds=0,
            dash_sc_grpc_pre_stop_drain_seconds=0,
            pre_stop_drain_headroom_seconds=0,
            pre_stop_drain_signal=0,
            backend_post_frontend_drain_seconds=0,
            validate_port_layout=noop,
        )
        with self.assertLogs(level="ERROR"), patch.dict(
            sys.modules, {"rtp_llm.telemetry": SimpleNamespace(resolve_pod_ip=noop)}
        ):
            self.namespace["start_server"](self.config)
        self.assertEqual(ready, [])
        self.assertEqual(failed, [True])
        self.assertEqual(released, [True])


if __name__ == "__main__":
    unittest.main()
