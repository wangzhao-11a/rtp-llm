import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.perf_test import batch_decode_test, batch_perf_impl
from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _parse_grid_cases,
    _positive_int_arg,
)
from rtp_llm.test.perf_test.dataclass import ResponseInfo
from rtp_llm.test.perf_test.test_util import target_reuse_len_for_hit_rate


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_parse_grid_cases_dedupes_in_order(self):
        self.assertEqual(
            _parse_grid_cases("1:8192, 2:262144,1:8192"),
            [(1, 8192), (2, 262144)],
        )

    def test_parse_grid_cases_empty_uses_cartesian_grid(self):
        self.assertIsNone(_parse_grid_cases(""))

    def test_parse_grid_cases_rejects_malformed_items(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("1:8192,bad")

    def test_parse_grid_cases_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("0:8192")

    def test_positive_int_arg_parses_forwarded_server_args(self):
        self.assertEqual(
            _positive_int_arg(["--seq_size_per_block", "64"], "seq_size_per_block", 1),
            64,
        )
        self.assertEqual(_positive_int_arg([], "seq_size_per_block", 8), 8)

    def test_target_reuse_len_rounds_to_block(self):
        self.assertEqual(target_reuse_len_for_hit_rate(1048576, 0.85, 64), 891264)

    def test_failed_run_collects_timelines_stops_server_and_writes_info(self):
        with tempfile.TemporaryDirectory() as result_dir:
            args = argparse.Namespace(
                generate_config="{}",
                result_dir=result_dir,
                dp_size=1,
                max_seq_len=8192,
                concurrency_limit=2,
                decode_test_length=10,
                num_measures=1,
                prefill_reuse_cache_hit_rate=0.0,
                dataset_name="",
                dataset_path="",
                dataset="",
            )
            trace = Path(result_dir) / "profiler_mock_wr0.json"
            trace.write_text('{"traceEvents": []}')
            with (
                mock.patch.object(
                    batch_decode_test, "parse_args", return_value=(args, [])
                ),
                mock.patch.object(batch_decode_test, "EngineServer") as server_cls,
                mock.patch.object(
                    batch_decode_test,
                    "_run_tests",
                    side_effect=RuntimeError("PERF_REQUEST_FAILURE"),
                ),
                mock.patch.object(batch_decode_test.time, "sleep"),
                mock.patch("rtp_llm.config.log_config.setup_logging"),
            ):
                with self.assertRaisesRegex(RuntimeError, "PERF_REQUEST_FAILURE"):
                    batch_decode_test.main()
                server_cls.return_value.stop.assert_called_once_with()
            self.assertTrue((Path(result_dir) / "timelines" / trace.name).is_file())
            info = json.loads((Path(result_dir) / "test_info.json").read_text())
            self.assertEqual(info["decode_test_length"], 10)


class PerfRequestSuccessTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(
            os.environ,
            {
                "PERF_REQUIRE_ALL_REQUESTS_SUCCESS": "1",
                "PERF_PROFILE_FLUSH_SLEEP": "0",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        pool = mock.patch.object(batch_perf_impl, "ProcessPoolExecutor")
        pool.start()
        self.addCleanup(pool.stop)

    def make_impl(self, *, warmup_runs=0, measure_runs=1, profile_runs=0):
        impl = batch_perf_impl.BatchPerfImpl(
            12345,
            1,
            2,
            "query",
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            profile_runs=profile_runs,
        )
        impl._set_concurrency = mock.Mock()
        impl._prearm_profile = mock.Mock()
        return impl

    @staticmethod
    def responses(*success_flags):
        return [ResponseInfo({}, flag) for flag in success_flags]

    def test_all_failed_measure_requests_raise(self):
        impl = self.make_impl()
        impl._curl_server_responses = mock.Mock(
            return_value=self.responses(False, False)
        )
        with self.assertRaisesRegex(RuntimeError, "phase=measure.*success=0/2"):
            impl.run()

    def test_partial_failure_raises(self):
        impl = self.make_impl()
        impl._curl_server_responses = mock.Mock(
            return_value=self.responses(True, False)
        )
        with self.assertRaisesRegex(RuntimeError, "success=1/2"):
            impl.run()

    def test_missing_responses_raise(self):
        impl = self.make_impl()
        impl._curl_server_responses = mock.Mock(return_value=[])
        with self.assertRaisesRegex(RuntimeError, "success=0/0 expected=2"):
            impl.run()

    def test_failed_warmup_is_not_ignored(self):
        impl = self.make_impl(warmup_runs=1)
        impl._curl_server_responses = mock.Mock(
            return_value=self.responses(False, False)
        )
        with self.assertRaisesRegex(RuntimeError, "phase=warmup"):
            impl.run()
        impl._curl_server_responses.assert_called_once()

    def test_failed_profile_is_not_hidden_by_successful_measure(self):
        impl = self.make_impl(profile_runs=1)
        impl._curl_server_responses = mock.Mock(
            side_effect=[self.responses(True, True), self.responses(True, False)]
        )
        with self.assertRaisesRegex(RuntimeError, "phase=profile"):
            impl.run()

    def test_successful_all_phases_preserve_measurement_counts(self):
        impl = self.make_impl(warmup_runs=1, measure_runs=2, profile_runs=1)
        impl._curl_server_responses = mock.Mock(return_value=self.responses(True, True))
        metric = impl.run()
        self.assertEqual(metric.total_requests, 4)
        self.assertEqual(metric.success_requests, 4)
        self.assertEqual(impl._curl_server_responses.call_count, 4)

    def test_guard_defaults_off(self):
        os.environ.pop("PERF_REQUIRE_ALL_REQUESTS_SUCCESS")
        impl = self.make_impl(warmup_runs=1, profile_runs=1)
        impl._curl_server_responses = mock.Mock(
            return_value=self.responses(False, False)
        )
        self.assertEqual(impl.run().fail_requests, 2)
        self.assertEqual(impl._curl_server_responses.call_count, 3)

    def test_disabled_grid_warmup_does_not_require_requests(self):
        impl = self.make_impl(measure_runs=0)
        impl._curl_server_responses = mock.Mock()
        self.assertEqual(impl.run().total_requests, 0)
        impl._curl_server_responses.assert_not_called()


if __name__ == "__main__":
    unittest.main()
