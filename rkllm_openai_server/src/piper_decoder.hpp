#ifndef RKLLM_OPENAI_SERVER_PIPER_DECODER_HPP
#define RKLLM_OPENAI_SERVER_PIPER_DECODER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace rkllm_openai {

/**
 * Piper RKNN decoder: z [1,C,T], y_mask [1,1,T] -> audio [1,1,S].
 * Chunk size is fixed by the model (static input dim).
 */
class PiperDecoder {
public:
    PiperDecoder() = default;
    ~PiperDecoder();

    bool init(const std::string& rknn_path);
    void release();

    /** Expected chunk time dimension (decoder input size). */
    size_t get_chunk_time() const { return chunk_time_; }

    /** Run decoder on one chunk. z size = 1*C*chunk_time, y_mask size = 1*1*chunk_time. Out audio (float) length = 1*1*samples. */
    bool run_chunk(const float* z, const float* y_mask, std::vector<float>& out_audio);

private:
    void* ctx_ = nullptr;  // rknn_context
    size_t chunk_time_ = 0;
    size_t z_channels_ = 0;
    size_t out_samples_ = 0;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_PIPER_DECODER_HPP
