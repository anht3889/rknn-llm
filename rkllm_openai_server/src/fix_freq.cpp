#include "fix_freq.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace rkllm_openai {

static bool write_sysfs(const std::string& path, const std::string& value, bool verbose) {
    std::ofstream f(path);
    if (!f) {
        if (verbose)
            std::cerr << "[fix_freq] skip (no access): " << path << "\n";
        return false;
    }
    f << value;
    if (!f) {
        if (verbose)
            std::cerr << "[fix_freq] write failed: " << path << "\n";
        return false;
    }
    if (verbose)
        std::cerr << "[fix_freq] " << path << " = " << value << "\n";
    return true;
}

struct SysfsOp {
    const char* path;
    const char* value;
};

static void apply_ops(const SysfsOp* ops, size_t n, bool verbose) {
    for (size_t i = 0; i < n; ++i)
        write_sysfs(ops[i].path, ops[i].value, verbose);
}

/* RK3588: matches scripts/fix_freq_rk3588.sh */
static const SysfsOp rk3588_ops[] = {
    { "/sys/devices/system/cpu/cpu0/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu1/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu2/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu3/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu4/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu5/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu6/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu7/cpuidle/state1/disable", "1" },
    { "/sys/class/devfreq/fdab0000.npu/governor", "userspace" },
    { "/sys/class/devfreq/fdab0000.npu/userspace/set_freq", "1000000000" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed", "1800000" },
    { "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed", "2352000" },
    { "/sys/devices/system/cpu/cpufreq/policy6/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed", "2352000" },
    { "/sys/class/devfreq/fb000000.gpu/governor", "userspace" },
    { "/sys/class/devfreq/fb000000.gpu/userspace/set_freq", "1000000000" },
    { "/sys/class/devfreq/dmc/governor", "userspace" },
    { "/sys/class/devfreq/dmc/userspace/set_freq", "2112000000" },
};

/* RK3576: matches scripts/fix_freq_rk3576.sh */
static const SysfsOp rk3576_ops[] = {
    { "/sys/devices/system/cpu/cpu0/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu1/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu2/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu3/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu4/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu5/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu6/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu7/cpuidle/state1/disable", "1" },
    { "/sys/class/devfreq/27700000.npu/governor", "userspace" },
    { "/sys/class/devfreq/27700000.npu/userspace/set_freq", "1000000000" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed", "2208000" },
    { "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed", "2304000" },
    { "/sys/class/devfreq/27800000.gpu/governor", "userspace" },
    { "/sys/class/devfreq/27800000.gpu/userspace/set_freq", "950000000" },
    { "/sys/class/devfreq/dmc/governor", "userspace" },
    { "/sys/class/devfreq/dmc/userspace/set_freq", "2112000000" },
};

/* RV1126B: matches scripts/fix_freq_rv1126b.sh */
static const SysfsOp rv1126b_ops[] = {
    { "/sys/devices/system/cpu/cpu0/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu1/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu2/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu3/cpuidle/state1/disable", "1" },
    { "/sys/class/devfreq/22000000.npu/governor", "userspace" },
    { "/sys/class/devfreq/22000000.npu/userspace/set_freq", "950000000" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed", "1608000" },
    { "/sys/class/devfreq/dmc/governor", "userspace" },
    { "/sys/class/devfreq/dmc/userspace/set_freq", "1332000000" },
};

/* RK3562: matches scripts/fix_freq_rk3562.sh (NPU uses debugfs, CPU path differs) */
static const SysfsOp rk3562_ops[] = {
    { "/sys/devices/system/cpu/cpu0/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu1/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu2/cpuidle/state1/disable", "1" },
    { "/sys/devices/system/cpu/cpu3/cpuidle/state1/disable", "1" },
    { "/sys/kernel/debug/rknpu/freq", "1000000000" },
    { "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "userspace" },
    { "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed", "2016000" },
    { "/sys/class/devfreq/dmc/governor", "userspace" },
    { "/sys/class/devfreq/dmc/userspace/set_freq", "2112000000" },
};

bool apply_fix_freq(const std::string& platform, bool verbose) {
    const SysfsOp* ops = nullptr;
    size_t n = 0;
    if (platform == "rk3588") {
        ops = rk3588_ops;
        n = sizeof(rk3588_ops) / sizeof(rk3588_ops[0]);
    } else if (platform == "rk3576") {
        ops = rk3576_ops;
        n = sizeof(rk3576_ops) / sizeof(rk3576_ops[0]);
    } else if (platform == "rv1126b") {
        ops = rv1126b_ops;
        n = sizeof(rv1126b_ops) / sizeof(rv1126b_ops[0]);
    } else if (platform == "rk3562") {
        ops = rk3562_ops;
        n = sizeof(rk3562_ops) / sizeof(rk3562_ops[0]);
    } else {
        if (verbose)
            std::cerr << "[fix_freq] unsupported platform: " << platform << "\n";
        return true;  /* no-op, not failure */
    }
    if (verbose)
        std::cerr << "[fix_freq] applying for platform " << platform << " (" << n << " ops)\n";
    apply_ops(ops, n, verbose);
    return true;
}

}  // namespace rkllm_openai
