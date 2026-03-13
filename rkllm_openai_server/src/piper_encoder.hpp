#ifndef RKLLM_OPENAI_SERVER_PIPER_ENCODER_HPP
#define RKLLM_OPENAI_SERVER_PIPER_ENCODER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace rkllm_openai {

/**
 * Piper ONNX encoder: phoneme_ids -> z, y_mask.
 * Inputs: phoneme_ids [1, N], scales [noise_scale, length_scale, noise_w_scale], optional speaker_id.
 * Outputs: z [1, C, T], y_mask [1, 1, T].
 */
class PiperEncoder {
public:
    PiperEncoder() = default;
    ~PiperEncoder();

    bool init(const std::string& onnx_path);
    void release();

    /** Run encoder. phoneme_ids length = N. scales = [noise_scale, length_scale, noise_w_scale]. speaker_id < 0 = no speaker. */
    bool run(const int64_t* phoneme_ids, size_t num_phonemes,
             float noise_scale, float length_scale, float noise_w_scale,
             int64_t speaker_id,
             std::vector<float>& out_z, std::vector<float>& out_y_mask);

    /** Output dims: z (1, C, T), y_mask (1, 1, T). T = encoder output time steps. */
    size_t get_z_channels() const { return z_channels_; }
    size_t get_z_time() const { return z_time_; }

private:
    void* session_ = nullptr;  // Ort::Session* when ONNX available
    void* env_ = nullptr;       // Ort::Env*
    size_t z_channels_ = 0;
    size_t z_time_ = 0;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_PIPER_ENCODER_HPP
