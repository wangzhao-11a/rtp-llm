#include "rtp_llm/cpp/engine_base/schedulers/PrefillChunk.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

void check(bool condition) {
    if (!condition) {
        std::fprintf(stderr, "Prefill chunk boundary regression failed\n");
        std::abort();
    }
}

}  // namespace

int main() {
    using rtp_llm::combineChunkAlignment;
    using rtp_llm::computeChunkGrant;

    // K3: Page-RR over eight 2048-token pages, with one LINEAR checkpoint per stripe.
    const auto alignment = combineChunkAlignment(2048, 8 * 2048);
    check(alignment == 16384);
    check(computeChunkGrant(65536 - 2048, 1, 131072, alignment) == 49152);
    check(computeChunkGrant(65536 - 2048, 2, 131072, alignment) == 16384);
    check(computeChunkGrant(2048, 1, 131072, alignment) == 0);
    check(computeChunkGrant(2048, 1, 123, alignment, 65536) == 123);
    check(computeChunkGrant(65536, 1, 131072, alignment, 2048) == 63488);

    // Heterogeneous LINEAR groups must reach a checkpoint in every group.
    check(combineChunkAlignment(alignment, 6144 * 8) == 49152);
    check(combineChunkAlignment(alignment, 2048 * 16) == 32768);
    check(combineChunkAlignment(0, 2048) == 0);
    check(combineChunkAlignment(2048, 0) == 0);
    check(combineChunkAlignment(std::numeric_limits<int>::max(), 2) == 0);

    // Enumerate legal end positions independently of the rounding calculation.
    // Verify maximal progress, final tails and the per-row budget together.
    for (int64_t boundary = 1; boundary <= 12; ++boundary) {
        for (int64_t prefix = 0; prefix <= 24; ++prefix) {
            for (int64_t rows = 1; rows <= 4; ++rows) {
                for (int64_t budget = 0; budget <= 48; ++budget) {
                    for (int64_t remaining = 1; remaining <= 32; ++remaining) {
                        int64_t expected = 0;
                        for (int64_t length = 1; length <= remaining && length * rows <= budget; ++length) {
                            if (length == remaining || (prefix + length) % boundary == 0) {
                                expected = length;
                            }
                        }
                        check(computeChunkGrant(budget, rows, remaining, boundary, prefix) == expected);
                    }
                }
            }
        }
    }

    check(computeChunkGrant(16, 0, 32, 4) == 0);
    check(computeChunkGrant(-1, 1, 32, 4) == 0);
    check(computeChunkGrant(16, 1, 32, 4, -1) == 0);
    check(computeChunkGrant(16, 1, 32, 0) == 0);
    const auto maximum = std::numeric_limits<int64_t>::max();
    check(computeChunkGrant(maximum, 1, maximum, 16384, maximum) == maximum);
    check(computeChunkGrant(65536, 1, maximum, 16384, maximum) == 49153);
    std::puts("Prefill chunk boundary checks passed");
    return 0;
}
