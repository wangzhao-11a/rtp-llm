py_test(
    name = "async_log_handler_test",
    srcs = ["async_log_handler_test.py"],
    deps = [
        "//rtp_llm/access_logger:access_logger",
    ],
)

py_test(
    name = "access_logger_test",
    srcs = ["access_logger_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:structure",
        "//rtp_llm:testlib",
        "//rtp_llm/access_logger:access_logger",
    ],
    # The test builds MMProcessEngine and imports CUDA-backed ops; on a worker
    # without an NVIDIA driver the import/assertion flakes. Route it to the
    # same L20 pool the sibling MM/api-server tests use.
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)
