"""Validate startup model warmup completion without importing the serving engine."""


async def consume_startup_warmup(output_stream, expected_input_len: int):
    """Consume RPC frames and require valid, single-token terminal outputs.

    RPC frame count is transport metadata, not the scheduler's chunk count.
    Stream errors propagate unchanged so startup cannot mark the health gate ready.
    """
    frame_count = 0
    terminal = None
    async for outputs in output_stream:
        frame_count += 1
        terminal = outputs

    if frame_count == 0:
        raise RuntimeError("startup warmup returned no RPC frames")
    terminal_outputs = getattr(terminal, "generate_outputs", None)
    if not terminal_outputs:
        raise RuntimeError("startup warmup returned empty terminal outputs")

    for output in terminal_outputs:
        if not getattr(output, "finished", False):
            raise RuntimeError("startup warmup terminal output is unfinished")
        aux = getattr(output, "aux_info", None)
        if aux is None:
            raise RuntimeError("startup warmup terminal output has no auxiliary info")
        if getattr(aux, "input_len", None) != expected_input_len:
            raise RuntimeError(
                "startup warmup terminal input length does not match request"
            )
        if getattr(aux, "output_len", None) != 1:
            raise RuntimeError("startup warmup terminal output length is not one")
        if getattr(aux, "reuse_len", None) != 0:
            raise RuntimeError("startup warmup unexpectedly reused cached input")
        if getattr(aux, "pd_sep", None) is not False:
            raise RuntimeError("startup warmup unexpectedly used PD separation")
        output_ids = getattr(output, "output_ids", None)
        if output_ids is None or output_ids.numel() != 1:
            raise RuntimeError("startup warmup terminal output tensor size is not one")

    return frame_count, terminal_outputs[0].aux_info
