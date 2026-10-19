import socket
import threading
import time
import unittest

from smoke.multi_inst_case_runner import (
    _dump_manager_logs,
    _extract_int_arg,
    _wait_tcp_port_ready,
)


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ExtractIntArgTest(unittest.TestCase):
    def test_flag_and_equals_forms(self):
        args = "--tp_size 2 --worker_info_port_num=9"
        self.assertEqual(_extract_int_arg(args, "--tp_size", 1), 2)
        self.assertEqual(_extract_int_arg(args, "--worker_info_port_num", 1), 9)

    def test_default_when_absent(self):
        self.assertEqual(_extract_int_arg("--tp_size 2", "--missing", 7), 7)
        self.assertEqual(_extract_int_arg("", "--tp_size", 1), 1)


class WaitTcpPortReadyTest(unittest.TestCase):
    def test_open_port_returns_promptly(self):
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            started = time.monotonic()
            _wait_tcp_port_ready("127.0.0.1", port, 5.0, "open port")
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            server.close()

    def test_closed_port_times_out(self):
        port = _free_port()
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as raised:
            _wait_tcp_port_ready("127.0.0.1", port, 0.5, "closed port")
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("closed port", str(raised.exception))

    def test_port_opened_late_is_waited_for(self):
        port = _free_port()
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        def open_later():
            time.sleep(0.4)
            server.bind(("127.0.0.1", port))
            server.listen(1)

        opener = threading.Thread(target=open_later, daemon=True)
        opener.start()
        try:
            started = time.monotonic()
            _wait_tcp_port_ready("127.0.0.1", port, 5.0, "late port")
            self.assertGreaterEqual(time.monotonic() - started, 0.3)
        finally:
            opener.join(timeout=1)
            server.close()


if __name__ == "__main__":
    unittest.main()


class _FakeManager:
    def __init__(self, explode=False):
        self.explode = explode
        self.dumped = 0

    def print_process_log(self, max_lines: int = 0):
        self.dumped += 1
        if self.explode:
            raise RuntimeError("no log file")


class DumpManagerLogsTest(unittest.TestCase):
    def test_dumps_every_manager_and_skips_none(self):
        first, second = _FakeManager(), _FakeManager()
        _dump_manager_logs([first, None, second], max_lines=7)
        self.assertEqual(first.dumped, 1)
        self.assertEqual(second.dumped, 1)

    def test_swallows_dump_failures(self):
        broken = _FakeManager(explode=True)
        ok = _FakeManager()
        _dump_manager_logs([broken, ok])  # must not raise
        self.assertEqual(broken.dumped, 1)
        self.assertEqual(ok.dumped, 1)
