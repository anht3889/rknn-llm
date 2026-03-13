#ifndef RKLLM_OPENAI_SERVER_WAV_WRITER_HPP
#define RKLLM_OPENAI_SERVER_WAV_WRITER_HPP

#include <cstdint>
#include <vector>

namespace rkllm_openai {

/** Write 16-bit mono WAV from float samples (-1..1). Sample rate from config. */
std::vector<uint8_t> wav_bytes_from_float_mono(const float* samples, size_t num_samples, int sample_rate);

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_WAV_WRITER_HPP
