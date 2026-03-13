#include "wav_writer.hpp"
#include <algorithm>
#include <cstring>

namespace rkllm_openai {

std::vector<uint8_t> wav_bytes_from_float_mono(const float* samples, size_t num_samples, int sample_rate) {
    const size_t data_bytes = num_samples * 2u;  // 16-bit
    const size_t total = 44 + data_bytes;
    std::vector<uint8_t> out(total, 0);
    uint8_t* p = out.data();

    auto write16 = [&p](uint16_t v) {
        p[0] = static_cast<uint8_t>(v & 0xff);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
        p += 2;
    };
    auto write32 = [&p](uint32_t v) {
        p[0] = static_cast<uint8_t>(v & 0xff);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
        p[2] = static_cast<uint8_t>((v >> 16) & 0xff);
        p[3] = static_cast<uint8_t>((v >> 24) & 0xff);
        p += 4;
    };

    memcpy(p, "RIFF", 4); p += 4;
    write32(36 + static_cast<uint32_t>(data_bytes));
    memcpy(p, "WAVE", 4); p += 4;
    memcpy(p, "fmt ", 4); p += 4;
    write32(16);
    write16(1);   // PCM
    write16(1);   // mono
    write32(static_cast<uint32_t>(sample_rate));
    write32(static_cast<uint32_t>(sample_rate * 2));
    write16(2);
    write16(16);
    memcpy(p, "data", 4); p += 4;
    write32(static_cast<uint32_t>(data_bytes));

    for (size_t i = 0; i < num_samples; i++) {
        float s = std::max(-1.f, std::min(1.f, samples[i]));
        int16_t v = static_cast<int16_t>(s * 32767.f);
        p[0] = static_cast<uint8_t>(v & 0xff);
        p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
        p += 2;
    }
    return out;
}

}  // namespace rkllm_openai
