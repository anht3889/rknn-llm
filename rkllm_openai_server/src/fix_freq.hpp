#ifndef RKLLM_OPENAI_SERVER_FIX_FREQ_HPP
#define RKLLM_OPENAI_SERVER_FIX_FREQ_HPP

#include <string>

namespace rkllm_openai {

/**
 * Apply fix_freq settings for the given Rockchip platform (same logic as
 * scripts/fix_freq_<platform>.sh): disable CPU idle state1, set NPU/CPU/GPU/DDR
 * to userspace governor and fix max frequencies for stable inference.
 * Requires root (or sufficient capability) for sysfs/debugfs writes.
 * @param platform One of "rk3588", "rk3576", "rv1126b", "rk3562".
 * @param verbose If true, log each write to stderr.
 * @return true if at least the NPU fix was applied (or platform unsupported);
 *         false only on internal error. Individual write failures are logged.
 */
bool apply_fix_freq(const std::string& platform, bool verbose = false);

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_FIX_FREQ_HPP
