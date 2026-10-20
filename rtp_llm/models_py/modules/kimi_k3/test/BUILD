load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")

py_test(
    name = "fused_shared_jit_warmup_test",
    srcs = ["test_fused_shared_jit_warmup.py"],
    main = "test_fused_shared_jit_warmup.py",
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "native_mla_prefill_backend_test",
    srcs = ["test_native_mla_prefill_backend.py"],
    main = "test_native_mla_prefill_backend.py",
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "bounded_projection_jit_warmup_test",
    srcs = ["test_bounded_projection_jit_warmup.py"],
    main = "test_bounded_projection_jit_warmup.py",
    data = [
        "//rtp_llm/models_py:bounded_projection_jit_test_sources",
    ],
)

py_test(
    name = "bounded_projection_jit_gpu_test",
    srcs = ["test_bounded_projection_jit_gpu.py"],
    main = "test_bounded_projection_jit_gpu.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:einops",
        "//rtp_llm:transformers",
        "//rtp_llm:triton",
        cuda13_requirement("flash-linear-attention"),
        cuda13_requirement("nvidia-cutlass-dsl"),
        cuda13_requirement("backports-strenum"),
    ],
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)
