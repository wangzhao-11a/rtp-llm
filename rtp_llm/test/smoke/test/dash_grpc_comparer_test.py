import json
import unittest

from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.dash_error_policy import (
    DASH_MAX_ATTEMPTS,
    DASH_TRANSIENT_STATUS_CODES,
    should_retry_transient,
)
from smoke.dash_grpc_comparer import (
    DashGrpcComparer,
    DashSmokeResponse,
    _dash_error_spec,
)

from rtp_llm.dash_sc.proto import predict_v2_pb2

ERROR_SPEC = {
    "status_code": 504,
    "status_name": "GatewayTimeout",
    "status_message": "grammar validation timed out",
}


def _response(params):
    response = predict_v2_pb2.ModelStreamInferResponse()
    for name, value in params.items():
        parameter = response.infer_response.parameters[name]
        if isinstance(value, int):
            parameter.int64_param = value
        else:
            parameter.string_param = value
    return response


class DashErrorSpecTest(unittest.TestCase):
    def test_error_spec_is_parsed_from_parameters(self):
        spec = _dash_error_spec(
            _response(
                {
                    "error_msg": json.dumps(ERROR_SPEC),
                    "status_code": 504,
                    "status_name": "GatewayTimeout",
                    "status_message": "grammar validation timed out",
                }
            )
        )
        self.assertEqual(spec, ERROR_SPEC)

    def test_error_msg_alone_is_enough(self):
        spec = _dash_error_spec(_response({"error_msg": json.dumps(ERROR_SPEC)}))
        self.assertEqual(spec, ERROR_SPEC)

    def test_missing_or_malformed_error_msg_yields_none(self):
        self.assertIsNone(_dash_error_spec(_response({})))
        self.assertIsNone(_dash_error_spec(_response({"error_msg": "{not json"})))


class DashTransientRetryTest(unittest.TestCase):
    def setUp(self):
        self.comparer = DashGrpcComparer(None, "", {}, Tracer(), False)
        self.comparer._DASH_RETRY_INTERVAL_S = 0
        self.calls = 0

    def _respond_with(self, responses):
        def fake_infer_once(query_info, tokenizer, request):
            self.calls += 1
            return responses[self.calls - 1]

        self.comparer._infer_once = fake_infer_once

    def test_transient_error_specs_are_retried(self):
        self._respond_with(
            [
                DashSmokeResponse(response="", finish_reason=1000, dash_error_code=429),
                DashSmokeResponse(response="", finish_reason=1000, dash_error_code=503),
                DashSmokeResponse(response="ok", generated_ids=[1]),
            ]
        )
        actual = self.comparer._infer_with_transient_retry(None, None, None)
        self.assertEqual(self.calls, 3)
        self.assertEqual(actual.response, "ok")

    def test_persistent_transient_spec_stops_after_bounded_attempts(self):
        self._respond_with(
            [
                DashSmokeResponse(response="", finish_reason=1000, dash_error_code=504)
                for _ in range(3)
            ]
        )
        actual = self.comparer._infer_with_transient_retry(None, None, None)
        self.assertEqual(self.calls, 3)
        self.assertEqual(actual.dash_error_code, 504)

    def test_non_transient_spec_is_not_retried(self):
        self._respond_with(
            [
                DashSmokeResponse(
                    response="",
                    finish_reason=1000,
                    dash_error_code=400,
                    dash_status_name="InvalidParameter",
                )
            ]
        )
        actual = self.comparer._infer_with_transient_retry(None, None, None)
        self.assertEqual(self.calls, 1)
        self.assertEqual(actual.dash_error_code, 400)


class DashErrorSpecVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.comparer = DashGrpcComparer(None, "", {}, Tracer(), False)

    def test_error_spec_without_tokens_reports_visit_failed(self):
        actual = DashSmokeResponse(
            response="",
            finish_reason=1000,
            dash_error_code=504,
            dash_status_name="GatewayTimeout",
            dash_status_message="grammar validation timed out",
        )
        with self.assertRaises(SmokeException) as raised:
            self.comparer.compare_result(DashSmokeResponse(response=""), actual)
        self.assertEqual(raised.exception.error_status, QueryStatus.VISIT_FAILED)
        self.assertIn("status_code=504", raised.exception.message)
        self.assertIn("GatewayTimeout", raised.exception.message)


if __name__ == "__main__":
    unittest.main()


class SharedTransientRetryPolicyTest(unittest.TestCase):
    """The Dash and DashSc channels must share one retry decision."""

    def _decide(self, **overrides):
        kwargs = dict(
            retryable=True,
            status_code=504,
            attempts=1,
            deadline=float("inf"),
        )
        kwargs.update(overrides)
        return should_retry_transient(**kwargs)

    def test_retries_transient_status_within_budget(self):
        self.assertTrue(self._decide())
        self.assertIn(504, DASH_TRANSIENT_STATUS_CODES)

    def test_never_retries_non_transient_status(self):
        self.assertFalse(self._decide(status_code=400))
        self.assertFalse(self._decide(status_code=None))

    def test_case_level_veto_and_bounds(self):
        self.assertFalse(self._decide(retryable=False))
        self.assertFalse(self._decide(attempts=DASH_MAX_ATTEMPTS))
        self.assertFalse(self._decide(deadline=0.0))
