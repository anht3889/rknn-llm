#ifndef RKLLM_OPENAI_SERVER_PIPER_CONFIG_HPP
#define RKLLM_OPENAI_SERVER_PIPER_CONFIG_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rkllm_openai {

/** Piper voice/config from piper.json. */
struct PiperConfig {
    int sample_rate = 22050;
    int hop_length = 256;
    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    float noise_w_scale = 0.8f;
    int num_speakers = 1;
    std::string espeak_voice = "en-us";
    /** Phoneme character (UTF-8) -> id. Value in JSON can be [id] or id. */
    std::unordered_map<std::string, int64_t> phoneme_id_map;
    /** Speaker name -> id */
    std::unordered_map<std::string, int64_t> speaker_id_map;

    /** Load from piper.json path. Returns false on error. */
    bool load_from_file(const std::string& path);
    /** Load from JSON string. */
    bool load_from_string(const std::string& json_str);

    /** Get phoneme id for a single character/symbol. Returns -1 if unknown. */
    int64_t get_phoneme_id(const std::string& symbol) const;
    /** Get speaker id by name. Returns -1 if unknown. */
    int64_t get_speaker_id(const std::string& name) const;
};

/** Find exactly one .onnx, one .rknn, one .json in dir. Returns (onnx_path, rknn_path, json_path) or empty strings. */
void find_piper_model_files(const std::string& dir,
                            std::string& out_onnx,
                            std::string& out_rknn,
                            std::string& out_json);

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_PIPER_CONFIG_HPP
