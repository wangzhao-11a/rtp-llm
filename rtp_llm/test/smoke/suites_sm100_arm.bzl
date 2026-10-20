load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

# SM100_ARM (aarch64) smoke suites.
#
# Merged from suites_cuda13.bzl (ARM half) and suites_sm100.bzl so the ARM lane
# has one file: smoke_cuda13_arm (DeepSeek-V4 production regression + V4.1
# CP2/DP2), smoke_sm100_dense / smoke_sm100_moe / smoke_sm100_eval (Qwen3 and
# Qwen3.5 dense + MoE + agentic benchmark).  The x86/SM103 suites live in
# suites_sm103.bzl.
def sm100_arm_suites():
    native.test_suite(
        name = "smoke_cuda13_arm",
        tests = [
            ":deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm",
            # DSpark speculative decode with CP page-RR. This is the blocking
            # regression for publishing both target and draft SWA_KV before decode
            # dispatch; missing draft publication fails with EC_FAILED_LOAD_BUFFER.
            # Both roles pin --moe_strategy mega_moe_se (FP8 shared expert fused
            # into the routed kernel, the only mega-moe form deep_gemm 2.6.1
            # exports), taking over the SE coverage from the retired dedicated
            # mega_moe_se case on this same CP2/DP2 page-RR topology.
            smoke_test(
                name="smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_async_xgrammar_json_sm100_arm.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 69632 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 1 --enable_fp32_lm_head 0",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --decode_capture_config '1,2,4,8,16' --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 30000 --reserver_runtime_mem_mb 10240 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2 --think_mode 1 --enable_fp32_lm_head 0",
                },
                envs={
                    "prefill": [
                        "ENABLE_DSPARK=1",
                        "SP_TYPE=dspark",
                        "SP_MODEL_TYPE=deepseek_v4_dspark",
                        "SP_CHECKPOINT_PATH=/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark",
                        "SP_ACT_TYPE=bf16",
                        "GEN_NUM_PER_CIRCLE=3",
                        "MODEL_EVAL_VARIANT=fp8_kv_cache",
                        "DSV4_BF16_VLLM=0",
                        "DSV4_COMPRESSOR_FAST=1",
                        "DSV4_COMPRESSOR_METADATA_TRITON=0",
                        "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single",
                        "DSV4_FIXED_POOL_USE_MEMORY=0",
                        "DSV4_TRAP_INVALID_KV_ACCESS=0",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DSV4_CHUNK_TOKENS=12288",
                        "ENABLE_GPU_PREFIX_TREE=1",
                        "ENABLE_DSV4_STATE_BLOCK_INDEPENDENT_EVICTION=0",
                        "ENABLE_LEGACY_MEMORY_CONNECTOR_FALLBACK=0",
                        "CP_FORCE_SINGLE_PREFILL=0",
                        "PREFILL_CP_KV_CACHE_SHARDED=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                        "ENABLE_LAYER_MICRO_BATCH=0",
                        "LINEAR_STEP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS=300",
                        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:1024",
                        "FRONTEND_SERVER_COUNT=1",
                        "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS=500",
                        "THINK_MODE=1",
                        "THINK_START_TAG=<think>",
                        "THINK_END_TAG=</think>",
                    ],
                    "decode": [
                        "ENABLE_DSPARK=1",
                        "SP_TYPE=dspark",
                        "SP_MODEL_TYPE=deepseek_v4_dspark",
                        "SP_CHECKPOINT_PATH=/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark",
                        "SP_ACT_TYPE=bf16",
                        "GEN_NUM_PER_CIRCLE=3",
                        "MODEL_EVAL_VARIANT=fp8_kv_cache",
                        "DSV4_BF16_VLLM=0",
                        "DSV4_COMPRESSOR_FAST=1",
                        "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single",
                        "OMP_NUM_THREADS=8",
                        "RTP_LLM_STREAM_ASYNC=1",
                        "RTP_LLM_DROP_BROAD_SYNC=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                        "RTP_LLM_MTP_ASYNC_PREPARE=0",
                        "ENABLE_LAYER_MICRO_BATCH=0",
                        "LINEAR_STEP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "DG_JIT_CACHE_DIR=__TEST_TMPDIR__/deep_gemm_pd_cache",
                        "THINK_MODE=1",
                        "THINK_START_TAG=<think>",
                        "THINK_END_TAG=</think>",
                    ],
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
        ],
        tags = ["manual"],
    )
    native.test_suite(
        name = "smoke_sm100_dense",
        tests = [
            smoke_test(
                name = "dense_tp1_sm100",
                task_info = "data/model/qwen3/q_r_l20a_fp4_tp1_py.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "dense_tp2_sm100",
                task_info = "data/model/qwen3/q_r_l20a_fp4_tp2_py.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "fp8_attention_sm100",
                # Separate golden from the H20 suite's q_r_block_fp8.json: the same
                # prompt yields a different completion on SM100, so one shared file
                # cannot satisfy both architectures.
                task_info = "data/model/qwen3/q_r_block_fp8_sm100_arm.json",
                # Bound KV allocation directly instead of assuming a nearly empty
                # 192-GiB card and reserving the rest as runtime memory.
                smoke_args = "--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 --kv_cache_mem_mb 8192 --reserver_runtime_mem_mb 8192 --warm_up 0",
                gpu_type = ["SM100_ARM_CU13"],
            ),
        ],
    )
    native.test_suite(
        name = "smoke_sm100_moe",
        tests = [
            # REMOVED: moe_deepep_ll_dp2_sm100 (FP8_PER_BLOCK + DeepEP LL + DP2)
            # Reason: DeepEP LL path already covered by 5 NVFP4 LL cases; UE8M0 scale path
            # covered by moe_deepep_normal_tp2_sm100; H20 has moe_deepep_ll_tp2 for LL+FP8.
            smoke_test(
                name = "moe_deepep_normal_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_py_tp2_sm100.json",
                smoke_args = "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            # REMOVED: moe_nvfp4_deepep_ll_dp2_sm100 (NVFP4 + DeepEP LL + DP2, no CudaGraph)
            # Reason: moe_nvfp4_deepep_ll_cudagraph_dp2_sm100 is its CudaGraph superset
            # (same config + enable_cuda_graph), per Graph覆盖原则.
            smoke_test(
                name = "moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_ll_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_ll_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_normal_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "moe_nvfp4_deepep_normal_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "moe_nvfp4_online_deepep_ll_tp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_nvfp4_online_py_tp2_ll_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1 --quantization MODELOPT_FP4",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "moe_nvfp4_online_deepep_normal_dp2_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_nvfp4_online_py_dp2_normal_sm100_arm.json",
                smoke_args = "--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0 --quantization MODELOPT_FP4",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info = "data/model/qwen35/q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 22000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            smoke_test(
                name = "next_moe_nvfp4_cudagraph_tp2_sm100",
                task_info = "data/model/qwen35/q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json",
                smoke_args = "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64",
                gpu_type = ["SM100_ARM_CU13"],
            ),
            # REMOVED: moe_masked_cudagraph_tp2_sm100 (Masked MoE + FP8_PER_BLOCK + CudaGraph + TP2)
            # Reason: shares task_info with H20 moe_masked_fp8_tp2 (q_r_30b_py_masked_without_deepep_tp2.json),
            # only differs by enable_cuda_graph and mem size; SM100 DeepGemm UE8M0 masked path
            # already covered by moe_deepep_normal_tp2_sm100 (same FP8_PER_BLOCK quantization).
        ],
    )
    native.test_suite(
        name = "smoke_sm100_eval",
        tests = [
            smoke_test(
                name = "qwen3_moe_tau2_bench_sm100",
                task_info = "data/model/qwen3_moe/q_r_30b_tau2_bench_tp2_sm100.json",
                smoke_args = "--warm_up 0 --act_type BF16 --max_seq_len 16384 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type = ["SM100_ARM_CU13"],
                data = [
                    "data/model/qwen3_moe/passing_tasks.json",
                    "data/model/qwen3_moe/run_tau2_bench.py",
                ],
            ),
        ],
    )
