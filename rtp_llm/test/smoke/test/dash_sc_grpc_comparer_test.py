import unittest
from unittest import mock

from smoke.common_def import Tracer
from smoke.dash_sc_grpc_comparer import DashScGrpcComparer


class _Outcome:
    @staticmethod
    def ok():
        return {
            "accumulated": [1, 2, 3],
            "parameters": {},
            "error_message": None,
            "prompt_token_num": None,
            "prompt_cached": None,
            "finish_reason": None,
        }

    @staticmethod
    def transient(code, message="grammar validation overloaded"):
        return {
            "accumulated": [],
            "parameters": {"status_code": code, "status_message": message},
            "error_message": message,
            "prompt_token_num": None,
            "prompt_cached": None,
            "finish_reason": None,
        }


class DashScTransientRetryTest(unittest.TestCase):
    """The DashSc channel shares the Dash error-spec/retry policy."""

    def setUp(self):
        self.comparer = DashScGrpcComparer(None, "", {}, Tracer(), False)
        self.calls = 0
        # no real sleeping between attempts
        self._sleep = mock.patch(
            "smoke.dash_sc_grpc_comparer.DASH_RETRY_INTERVAL_S", 0.0
        )
        self._sleep.start()
        self.addCleanup(self._sleep.stop)

    def _respond_with(self, outcomes):
        def fake_infer_once(query_info, request, expected, grpc_addr, deadline=None):
            self.calls += 1
            return outcomes[min(self.calls - 1, len(outcomes) - 1)]

        self.comparer._infer_once = fake_infer_once

    def _expected(self, **kwargs):
        return self.comparer.format_result(kwargs)

    def test_transient_errors_are_retried_until_success(self):
        self._respond_with(
            [_Outcome.transient(503), _Outcome.transient(429), _Outcome.ok()]
        )
        outcome, attempts = self.comparer._infer_with_transient_retry(
            None, None, self._expected(), "127.0.0.1:1"
        )
        self.assertEqual(self.calls, 3)
        self.assertEqual(attempts, 3)
        self.assertIsNone(outcome["error_message"])

    def test_error_expectation_disables_retry(self):
        self._respond_with([_Outcome.transient(504)])
        _, attempts = self.comparer._infer_with_transient_retry(
            None,
            None,
            self._expected(expected_status_code=504),
            "127.0.0.1:1",
        )
        self.assertEqual(self.calls, 1)
        self.assertEqual(attempts, 1)

    def test_cold_cache_golden_disables_retry(self):
        self.comparer.qr_info = {
            "_reuse_cache_enabled": True,
            "result": {"aux_info": {"reuse_len": 0, "local_reuse_len": 0}},
        }
        self._respond_with([_Outcome.transient(503)])
        _, attempts = self.comparer._infer_with_transient_retry(
            None, None, self._expected(), "127.0.0.1:1"
        )
        self.assertEqual(self.calls, 1)
        self.assertEqual(attempts, 1)

    def test_partial_tokens_disable_retry(self):
        # A rejection after visible output is observable to the caller; the Dash
        # channel refuses to retry in that case and DashSc must match.
        outcome = _Outcome.transient(503)
        outcome["accumulated"] = [1, 2, 3]
        self._respond_with([outcome])
        _, attempts = self.comparer._infer_with_transient_retry(
            None, None, self._expected(), "127.0.0.1:1"
        )
        self.assertEqual(self.calls, 1)
        self.assertEqual(attempts, 1)

    def test_non_transient_error_is_not_retried(self):
        self._respond_with([_Outcome.transient(400, "invalid parameter")])
        _, attempts = self.comparer._infer_with_transient_retry(
            None, None, self._expected(), "127.0.0.1:1"
        )
        self.assertEqual(self.calls, 1)
        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()