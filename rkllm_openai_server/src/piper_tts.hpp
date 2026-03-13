#ifndef RKLLM_OPENAI_SERVER_PIPER_TTS_HPP
#define RKLLM_OPENAI_SERVER_PIPER_TTS_HPP

#include "piper_config.hpp"
#include "piper_encoder.hpp"
#include "piper_decoder.hpp"
#include "piper_phonemizer.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace rkllm_openai {

/**
 * Native C++ Piper TTS: text -> phonemes -> encoder -> decoder -> WAV.
 * Only WAV output is supported in C++ (no mp3/opus).
 */
class PiperTts {
public:
    PiperTts() = default;
    ~PiperTts();

    /** Load from model directory (must contain .onnx, .rknn, .json). Returns false on error. */
    bool init(const std::string& model_dir);
    void release();

    bool is_initialized() const { return initialized_; }

    /**
     * Synthesize text to WAV bytes.
     * voice: speaker name (optional; use empty for default).
     * speed: > 0 to scale length (e.g. 1.0 = normal); applied via length_scale = 1/speed.
     * Returns WAV bytes on success; empty on error.
     */
    std::vector<uint8_t> synthesize(const std::string& text, const std::string& voice, float speed);

private:
    bool initialized_ = false;
    std::string model_dir_;
    PiperConfig config_;
    PiperEncoder encoder_;
    PiperDecoder decoder_;
    PiperPhonemizer phonemizer_;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_PIPER_TTS_HPP
