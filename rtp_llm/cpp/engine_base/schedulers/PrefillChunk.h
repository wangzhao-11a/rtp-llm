#pragma once

#include <cstdint>
#include <limits>
#include <numeric>

namespace rtp_llm {

// Stream positions and chunk lengths must fit the stream's int token indices.
inline int64_t combineChunkAlignment(int64_t alignment, int64_t tokens) {
    if (alignment <= 0 || tokens <= 0) {
        return 0;
    }
    const auto factor = alignment / std::gcd(alignment, tokens);
    if (factor > std::numeric_limits<int>::max() / tokens) {
        return 0;
    }
    return factor * tokens;
}

inline int64_t
computeChunkGrant(int64_t budget, int64_t rows, int64_t remaining, int64_t alignment, int64_t prefix = 0) {
    if (budget <= 0 || rows <= 0 || remaining <= 0 || alignment <= 0 || prefix < 0) {
        return 0;
    }

    const int64_t max_len = budget / rows;
    if (remaining <= max_len) {
        return remaining;
    }

    // Align the absolute end position. A loaded prefix need not itself be a
    // multiple of the common alignment of all cache groups.
    const int64_t first_boundary = (alignment - prefix % alignment) % alignment;
    if (max_len < first_boundary) {
        return 0;
    }
    return first_boundary + ((max_len - first_boundary) / alignment) * alignment;
}

}  // namespace rtp_llm
