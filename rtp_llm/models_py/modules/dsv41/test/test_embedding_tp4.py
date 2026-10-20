"""Real TP4 embedding lookup and graph replay for target/draft token layouts."""

import faulthandler
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from rtp_llm.models_py.distributed import collective_torch as collective
    from rtp_llm.models_py.modules.dsv41.utils import V41Embedding
    from rtp_llm.test.utils import hang_watchdog

    # This case has hung silently in CI with four cards held. Bound each rank
    # and leave a stack dump if it stops making progress; the gpu_lock wrapper
    # kills the whole process group at a similar budget from the outside.
    hang_watchdog.install()
    faulthandler.dump_traceback_later(60, repeat=False)
    hang_watchdog.mark(f"worker{rank}:set_device")
    torch.cuda.set_device(rank)
    hang_watchdog.mark(f"worker{rank}:init_process_group")
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    hang_watchdog.mark(f"worker{rank}:new_group")
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    collective._group_map[collective.Group.TP] = dist.group.WORLD
    collective._group_map[collective.Group.DP_AND_TP] = dist.group.WORLD
    collective._parallelism_config = SimpleNamespace(tp_size=4, dp_size=1, world_size=4)
    collective._initialized = True
    try:
        hang_watchdog.mark(f"worker{rank}:build_weight")
        weight = (
            torch.arange(64 * 128, device="cuda", dtype=torch.float32)
            .reshape(64, 128)
            .to(torch.bfloat16)
        )
        embedding = V41Embedding(
            None,
            SimpleNamespace(get_attn_tp_size=lambda: 4),
            weight.chunk(4, dim=-1)[rank].contiguous(),
        )
        for shape in ((4,), (2, 3), (2, 1, 3)):
            hang_watchdog.mark(f"worker{rank}:shape{shape}")
            ids = torch.arange(6 if len(shape) > 1 else 4, device="cuda").reshape(shape)
            reference = torch.nn.functional.embedding(ids, weight)
            torch.testing.assert_close(embedding(ids), reference, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    embedding(ids)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier(group=control)
            hang_watchdog.mark(f"worker{rank}:capture_start")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = embedding(ids)
            dist.barrier(group=control)
            ids.add_(7)
            hang_watchdog.mark(f"worker{rank}:replay")
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                actual, torch.nn.functional.embedding(ids, weight), rtol=0, atol=0
            )
            dist.barrier(group=control)
            graph.reset()
            del graph
        dist.barrier(group=control)
    finally:
        collective._group_map.clear()
        collective._initialized = False
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


class EmbeddingTP4Test(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
    def test_full_hidden_width_and_live_graph_token_ids(self):
        # Parent-side watchdog too: if mp.spawn/join hangs before the workers
        # report, this prints the parent stacks instead of waiting forever.
        hang_watchdog.install()
        # The BUILD declares GPU_COUNT=4; make a mis-declared or mis-pinned run
        # fail loudly here instead of surfacing later as a mysterious hang.
        visible = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(visible)]
        hang_watchdog.mark(f"parent:visible_devices={visible} names={names}")
        self.assertEqual(
            visible, 4, "GPU_COUNT=4 requires the wrapper to pin four cards"
        )
        hang_watchdog.mark("parent:spawn")
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(
                _worker,
                args=(str(Path(temporary) / "rendezvous"),),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
