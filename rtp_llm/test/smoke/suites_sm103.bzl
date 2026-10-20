load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")
load("@arch_config//:arch_select.bzl", "cuda13_x86_test_gpu")

# SM103 (x86_64) smoke suites.
#
# Split out of suites_cuda13.bzl: the SM103 pool carries the DeepSeek-V4-Pro
# checkpoint, which exceeds SM100_ARM memory at EP=4.
def sm103_suites():
    native.test_suite(
        name = "smoke_cuda13_x86",
        tests = [
            # CRC write/read regression with real HOST reuse; DEVICE/DISK/remote
            # reuse are disabled so the second query must reload HOST blocks.
            # A final multi-token request also exercises DSpark decode in this
            # same CP page-RR, 2P + 2D server, without depending on grammar pins.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103.json",
                smoke_args = {
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 1024 --memory_cache_sync_timeout_ms 120000 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1", "DG_JIT_CPP_STANDARD=20", "LOG_LEVEL=DEBUG"],
                },
                gpu_type = [cuda13_x86_test_gpu()],
                sleep_time_qr = 10,
            ),

            # Long HOST prefixes exercise large CRC transfer batches,
            # alongside the short-prefix cases.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_large_batch_crc_memory_sm103",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_large_batch_crc_memory_sm103.json",
                smoke_args = {
                    "prefill": "--memory_cache_max_descriptors_per_transfer_batch 32 --load_method fastsafetensors --max_seq_len 12288 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 1024 --memory_cache_sync_timeout_ms 120000 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0",
                    "decode": "--memory_cache_max_descriptors_per_transfer_batch 32 --load_method fastsafetensors --max_seq_len 12288 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1", "DG_JIT_CPP_STANDARD=20", "LOG_LEVEL=DEBUG"],
                },
                gpu_type = [cuda13_x86_test_gpu()],
                sleep_time_qr = 10,
            ),

            # Only DeepSeek-V4-Pro case. Deliberately plain — no CUDA graph, no PD
            # separation, no speculative decode, no page-RR. The advanced feature
            # matrix is covered on Flash above; this exists so a Pro-specific
            # regression (weight layout, expert count, rope/scaling config) cannot
            # ship unnoticed.
            #
            # Single-role CP=4 + EP=4 all-gather prefill topology, 4 GPUs. Runs on
            # SM103 rather than SM100_ARM: Pro is ~865GB, so EP=4 needs ~216GB/rank,
            # over SM100_ARM's ~186GB but inside SM103's ~288GB.
            smoke_test(
                name="v4_pro_cp4_ep4_basic_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_pro_cp4_sm100_arm.json",
                smoke_args="--load_method scratch --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 4 --dp_size 1 --ep_size 4 --moe_strategy mega_moe_se --world_size 4 --seq_size_per_block 256 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --concurrency_limit 1 --max_context_batch_size 1 --reserver_runtime_mem_mb 20480",
                envs=["DG_JIT_CPP_STANDARD=20"],
                gpu_type=[cuda13_x86_test_gpu()],
            ),
        ],
        tags = ["manual"],
    )
