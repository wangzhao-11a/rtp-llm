#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

namespace rtp_llm {
namespace {

EngineInitParams decodePerfParams() {
    EngineInitParams params;
    params.pd_sep_config.role_type                   = RoleType::DECODE;
    params.kv_cache_config.test_block_num            = 2400;
    params.runtime_config.use_batch_decode_scheduler = true;
    return params;
}

TEST(RemoteRpcServiceImplTest, LocalDecodeRequiresExactOptIn) {
    const auto params = decodePerfParams();
    EXPECT_TRUE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, "1"));
    const char* disabled_values[] = {nullptr, "", "0", "true", "01", "1 "};
    for (const char* value : disabled_values) {
        EXPECT_FALSE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, value));
    }
}

TEST(RemoteRpcServiceImplTest, LocalDecodeRequiresExplicitPerfConfiguration) {
    auto params                           = decodePerfParams();
    params.kv_cache_config.test_block_num = 0;
    EXPECT_FALSE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, "1"));
    params.kv_cache_config.test_block_num = -1;
    EXPECT_FALSE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, "1"));
    params.kv_cache_config.test_block_num            = 2400;
    params.runtime_config.use_batch_decode_scheduler = false;
    EXPECT_FALSE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, "1"));
}

TEST(RemoteRpcServiceImplTest, OtherRolesNeverEnableLocalDecodeRouting) {
    auto params = decodePerfParams();
    for (RoleType role : {RoleType::PDFUSION, RoleType::PREFILL, RoleType::VIT, RoleType::FRONTEND}) {
        params.pd_sep_config.role_type = role;
        EXPECT_FALSE(RemoteRpcServiceImpl::allowPerfLocalRequests(params, "1"));
    }
}

TEST(RemoteRpcServiceImplTest, DecodeStillRejectsRegularRequestsByDefault) {
    RemoteRpcServiceImpl service;
    const auto           status = service.GenerateStreamCall(nullptr, nullptr, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "server not implement GenerateStreamCall");
}

TEST(RemoteRpcServiceImplTest, LocalDecodeRetainsInitializationGuard) {
    RemoteRpcServiceImpl service;
    service.perf_local_requests_ = true;
    EXPECT_EQ(service.GenerateStreamCall(nullptr, nullptr, nullptr).error_code(), grpc::StatusCode::UNAVAILABLE);

    // A server object can exist while its engine is still initializing.
    service.local_server_ = std::make_shared<LocalRpcServer>();
    EXPECT_EQ(service.GenerateStreamCall(nullptr, nullptr, nullptr).error_code(), grpc::StatusCode::UNAVAILABLE);
}

}  // namespace
}  // namespace rtp_llm
