"""Offline tests for the proposal boundary and saved request receipt."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from .decode_search import (
    MODEL, OPTIONS, attempt_dir, make_higher_cap_request, parse_proposal, read_json,
    retry_dir, submit, verify_model_access,
)


def response(proposal, finish="stop"):
    return json.dumps(dict(choices=[dict(finish_reason=finish,
                                        message=dict(content=json.dumps(proposal)))],
                           usage=dict(completion_tokens=13))).encode()


class DecodeSearchTest(unittest.TestCase):
    proposal = dict(config_id="n32-k32-w2-s1", hypothesis="Fewer warps may reduce overhead.")

    def test_accepts_allowed_complete_proposal(self):
        self.assertEqual(parse_proposal(response(self.proposal), OPTIONS), self.proposal)

    def test_rejects_unknown_or_unavailable_config(self):
        for config_id, options in (("unknown", OPTIONS), (self.proposal["config_id"], {})):
            with self.subTest(config_id=config_id), self.assertRaises(ValueError):
                parse_proposal(response(dict(self.proposal, config_id=config_id)), options)

    def test_rejects_truncated_response_or_extra_code(self):
        with self.assertRaises(ValueError):
            parse_proposal(response(self.proposal, finish="length"), OPTIONS)
        with self.assertRaises(ValueError):
            parse_proposal(response(dict(self.proposal, code="print(1)")), OPTIONS)

    def test_mocked_request_saves_wire_bytes_and_proposal(self):
        with TemporaryDirectory() as directory, patch(
            "stage30.decode_search.urllib.request.build_opener"
        ) as build, redirect_stdout(io.StringIO()):
            opener = build.return_value
            reply = opener.open.return_value.__enter__.return_value
            reply.headers = {"x-request-id": "mock-id"}
            reply.read.return_value = response(self.proposal)
            attempt = Path(directory) / "round-1"
            submit(attempt, {"model": "kimi-k3"}, OPTIONS, "test-key")
            self.assertEqual(opener.open.call_count, 1)
            self.assertEqual(opener.open.call_args.args[0].data, (attempt / "request.json").read_bytes())
            self.assertEqual(read_json(attempt / "proposal.json"), self.proposal)
            self.assertEqual(read_json(attempt / "api.json")["status"], "accepted")

    def test_catalog_requires_configured_model(self):
        with patch("stage30.decode_search.urllib.request.build_opener") as build:
            opener = build.return_value
            reply = opener.open.return_value.__enter__.return_value
            reply.read.return_value = json.dumps(dict(data=[dict(id=MODEL)])).encode()
            verify_model_access("test-key")
            self.assertEqual(opener.open.call_args.args[0].get_method(), "GET")
            reply.read.return_value = b'{"data":[]}'
            with self.assertRaises(ValueError):
                verify_model_access("test-key")

    def test_one_time_retry_preserves_original_404(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            original = run / "round-1"
            original.mkdir()
            saved = original / "api.json"
            saved.write_text('{"status":"http_error","http_status":404}')
            with patch("stage30.decode_search.RUN", run):
                retry = retry_dir(1)
                self.assertEqual(retry, run / "round-1-retry")
                retry.mkdir()
                self.assertEqual(attempt_dir(1), retry)
                with self.assertRaises(ValueError):
                    retry_dir(1)
                self.assertEqual(read_json(saved)["http_status"], 404)

    def test_http_error_saves_provider_code(self):
        with TemporaryDirectory() as directory, patch(
            "stage30.decode_search.urllib.request.build_opener"
        ) as build, redirect_stdout(io.StringIO()):
            build.return_value.open.side_effect = HTTPError(
                "https://api.inco.ai/v1/chat/completions", 404, "not found",
                {"x-request-id": "mock-error"}, io.BytesIO(b'{"error":{"code":"model_not_found"}}'),
            )
            attempt = Path(directory) / "attempt"
            submit(attempt, {"model": MODEL}, OPTIONS, "test-key")
            receipt = read_json(attempt / "api.json")
            self.assertEqual(receipt["error_code"], "model_not_found")
            self.assertEqual(receipt["request_id"], "mock-error")

    def test_higher_cap_changes_only_token_budget(self):
        original = dict(model=MODEL, max_tokens=8192, messages=[{"content": "Exact prompt"}], stream=False)
        with TemporaryDirectory() as directory:
            run = Path(directory)
            with patch("stage30.decode_search.RUN", run), patch(
                "stage30.decode_search.make_request", return_value=(original, OPTIONS)
            ) as rebuild:
                for number in (2, 3):
                    with self.subTest(round=number):
                        attempt = run / f"round-{number}"
                        attempt.mkdir()
                        (attempt / "request.json").write_text(json.dumps(original))
                        receipt = attempt / "api.json"
                        receipt.write_text('{"status":"invalid_proposal","finish_reason":"length"}')
                        payload, allowed = make_higher_cap_request(number)
                        rebuild.assert_called_with(number)
                        self.assertEqual(payload, dict(original, max_tokens=16384))
                        self.assertEqual(allowed, OPTIONS)
                        self.assertEqual(read_json(attempt / "request.json"), original)
                        receipt.write_text('{"status":"accepted","finish_reason":"stop"}')
                        with self.assertRaises(ValueError):
                            make_higher_cap_request(number)


if __name__ == "__main__":
    unittest.main()
