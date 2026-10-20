import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtp_llm.test.utils.device_resource import (
    _log_test_placement,
    _run_test_command,
    _watchdog_seconds,
    get_cuda_info,
    main,
    python_code,
)


class CudaDriverProbeTest(unittest.TestCase):
    def probe(self, info, stderr="", code=0):
        return subprocess.CompletedProcess(
            args=[], returncode=code, stdout=json.dumps(info), stderr=stderr
        )

    def test_healthy_cuda_uses_configured_libraries(self):
        path = "/usr/local/cuda/compat:/usr/local/nvidia/lib64"
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=self.probe({"A10": 1}),
            ) as run:
                self.assertEqual(get_cuda_info(), ("A10", 1))
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], path)
                self.assertEqual(run.call_count, 1)

    def test_803_retries_native_and_preserves_other_libraries(self):
        for compat in (
            "/usr/local/cuda/compat",
            "/usr/local/cuda-13.2/compat/",
            "/usr/local/cuda/compat/lib",
            "/usr/local/cuda-13.2/compat/lib64",
        ):
            with self.subTest(compat=compat):
                path = f"{compat}:/opt/conda310/lib:/usr/local/nvidia/lib64"
                native_path = "/opt/conda310/lib:/usr/local/nvidia/lib64"
                with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
                    with patch(
                        "rtp_llm.test.utils.device_resource.subprocess.run",
                        side_effect=[
                            self.probe(
                                {}, "CUDA initialization: Error 803: driver mismatch"
                            ),
                            self.probe({"A10": 2}),
                        ],
                    ) as run:
                        self.assertEqual(get_cuda_info(), ("A10", 2))
                        self.assertEqual(os.environ["LD_LIBRARY_PATH"], native_path)
                        self.assertEqual(
                            run.call_args.kwargs["env"]["LD_LIBRARY_PATH"], native_path
                        )

    def test_803_with_both_drivers_broken_is_not_treated_as_cpu(self):
        path = "/usr/local/cuda/compat:/usr/local/nvidia/lib64"
        for native in (self.probe({}), self.probe({}, "no driver", code=1)):
            with self.subTest(native=native):
                with patch.dict(os.environ, {"LD_LIBRARY_PATH": path}):
                    with patch(
                        "rtp_llm.test.utils.device_resource.subprocess.run",
                        side_effect=[
                            self.probe(
                                {}, "CUDA initialization: Error 803: driver mismatch"
                            ),
                            native,
                        ],
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "both configured compat and native"
                        ):
                            get_cuda_info()
                        self.assertEqual(os.environ["LD_LIBRARY_PATH"], path)

    def test_803_without_compat_reports_the_worker_error(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.run",
                return_value=self.probe(
                    {}, "CUDA initialization: Error 803: driver mismatch"
                ),
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "CUDA driver mismatch"):
                    get_cuda_info()
                self.assertEqual(run.call_count, 1)

    def test_real_cpu_worker_still_runs_cpu_tests(self):
        with patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=self.probe({}),
        ):
            self.assertIsNone(get_cuda_info())


class CudaDriverDiagnosticsTest(unittest.TestCase):
    def run_probe(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.version.cuda = "13.2"
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"torch": torch}), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            exec(python_code, {})
        self.assertEqual(json.loads(stdout.getvalue()), {})
        return json.loads(stderr.getvalue().split("CUDA probe diagnostics: ", 1)[1])

    def test_realpath_error_preserves_structured_probe_output(self):
        with patch("os.path.realpath", side_effect=OSError("broken cuda symlink")):
            details = self.run_probe()
        self.assertEqual(details["driver_details_error"], "broken cuda symlink")
        self.assertEqual(details["torch_cuda"], "13.2")

    def test_deleted_libcuda_keeps_full_mapped_path(self):
        path = "/usr/local/cuda/compat/libcuda.so.1 (deleted)"
        maps = f"7f00-7f10 r-xp 00000000 08:01 123 {path}\n"
        with patch("os.path.realpath", return_value="/usr/local/cuda-13.2"), patch(
            "builtins.open", side_effect=[io.StringIO("driver 550"), io.StringIO(maps)]
        ):
            details = self.run_probe()
        self.assertEqual(details["libcuda"], [path])


class GpuLockEntryTest(unittest.TestCase):
    @staticmethod
    def _fake_popen(returncode):
        proc = MagicMock()
        proc.returncode = returncode
        proc.poll.return_value = returncode
        return proc

    def test_cpu_target_does_not_probe_or_lock_gpus(self):
        for world_size in ("1", "4"):
            with self.subTest(world_size=world_size):
                env = {
                    "GPU_COUNT": "0",
                    "WORLD_SIZE": world_size,
                    "CUDA_VISIBLE_DEVICES": "1,2",
                    "HIP_VISIBLE_DEVICES": "3",
                }
                with patch.dict(os.environ, env):
                    with patch.object(sys, "argv", ["gpu_lock", "cpu-test", "arg"]):
                        with patch(
                            "rtp_llm.test.utils.device_resource.get_cuda_info"
                        ) as probe, patch(
                            "rtp_llm.test.utils.device_resource.DeviceResource"
                        ) as resource, patch(
                            "rtp_llm.test.utils.device_resource.subprocess.Popen",
                            return_value=self._fake_popen(0),
                        ) as popen:
                            self.assertEqual(main(), 0)
                            probe.assert_not_called()
                            resource.assert_not_called()
                            popen.assert_called_once_with(
                                ["cpu-test", "arg"], start_new_session=True
                            )
                            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "")

    def test_cpu_target_preserves_command_failure(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}):
            with patch(
                "rtp_llm.test.utils.device_resource.subprocess.Popen",
                return_value=self._fake_popen(23),
            ):
                self.assertEqual(main(), 23)

    def test_gpu_target_still_probes_and_locks_requested_devices(self):
        jit_setup = MagicMock()
        env = {"GPU_COUNT": "1", "WORLD_SIZE": "2"}
        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {"jit_sys_path_setup": jit_setup}):
                with patch(
                    "rtp_llm.test.utils.device_resource.get_cuda_info",
                    return_value=("NVIDIA A10", 4),
                ) as probe, patch(
                    "rtp_llm.test.utils.device_resource.DeviceResource"
                ) as resource, patch(
                    "rtp_llm.test.utils.device_resource._nvidia_smi",
                    return_value=None,
                ), patch(
                    "rtp_llm.test.utils.device_resource.subprocess.Popen",
                    return_value=self._fake_popen(17),
                ):
                    resource.return_value.__enter__.return_value.gpu_ids = ["1", "3"]
                    self.assertEqual(main(), 17)
                    probe.assert_called_once_with()
                    resource.assert_called_once_with(2)
                    jit_setup.setup_jit_cache.assert_called_once_with()
                    resource.return_value.__exit__.assert_called_once()
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,3")


class RuntimeWatchdogTest(unittest.TestCase):
    def test_budget_defaults_to_1800_and_rejects_garbage(self):
        cases = (
            ("60", 60),
            ("0", 0),
            ("-5", -5),
            ("not-a-number", 1800),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": value}):
                    self.assertEqual(_watchdog_seconds(), expected)

    def test_missing_budget_variable_uses_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_TEST_WATCHDOG_SECONDS", None)
            self.assertEqual(_watchdog_seconds(), 1800)

    def test_placement_line_names_host_gpus_and_uuids(self):
        completed = subprocess.CompletedProcess(
            [], 0, stdout="GPU-aaa\nGPU-bbb\n", stderr=""
        )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi",
            return_value="/usr/bin/nvidia-smi",
        ), patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            return_value=completed,
        ) as run, patch(
            "rtp_llm.test.utils.device_resource.logging.info"
        ) as log:
            _log_test_placement()
            self.assertEqual(
                run.call_args.args[0][-2:],
                ["--query-gpu=uuid", "--format=csv,noheader"],
            )
            self.assertEqual(run.call_args.args[0][1:3], ["-i", "0,1"])
            line = log.call_args.args[0]
            self.assertIn("host=", line)
            self.assertIn("gpus=0,1", line)
            self.assertIn("uuids=GPU-aaa,GPU-bbb", line)

    def test_placement_degrades_when_nvidia_smi_is_missing(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), patch(
            "rtp_llm.test.utils.device_resource._nvidia_smi", return_value=None
        ):
            _log_test_placement()  # must not raise

    def test_hung_test_is_killed_and_reports_exit_code_70(self):
        hung = [sys.executable, "-c", "import time; time.sleep(30)"]
        with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": "1"}), patch(
            "rtp_llm.test.utils.device_resource._WATCHDOG_SIGNAL_GRACE_SECONDS", 0.2
        ), patch(
            "rtp_llm.test.utils.device_resource._log_test_placement"
        ), patch.object(
            sys, "argv", ["gpu_lock"] + hung
        ):
            self.assertEqual(_run_test_command(), 70)

    def test_watchdog_kills_the_whole_process_group(self):
        # Both levels ignore SIGUSR1, so only the group SIGKILL can stop them:
        # the workers a hung test spawned must not outlive the wrapper.
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "grandchild.pid"
            grandchild = (
                "import signal, time;"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN);"
                "time.sleep(60)"
            )
            child = (
                "import signal, subprocess, sys, time;"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN);"
                f"p = subprocess.Popen([sys.executable, '-c', {grandchild!r}]);"
                f"open({str(pid_file)!r}, 'w').write(str(p.pid));"
                "time.sleep(60)"
            )
            with patch.dict(os.environ, {"RTP_TEST_WATCHDOG_SECONDS": "1"}), patch(
                "rtp_llm.test.utils.device_resource._WATCHDOG_SIGNAL_GRACE_SECONDS", 0.5
            ), patch(
                "rtp_llm.test.utils.device_resource._log_test_placement"
            ), patch.object(
                sys, "argv", ["gpu_lock", sys.executable, "-c", child]
            ):
                self.assertEqual(_run_test_command(), 70)
            grandchild_pid = int(pid_file.read_text())
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail(f"grandchild {grandchild_pid} survived the watchdog")


if __name__ == "__main__":
    unittest.main()
