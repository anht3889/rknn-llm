#ifndef RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP
#define RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace rkllm_openai {

class PiperTts;

/**
 * Handler for OpenAI-style POST /v1/audio/speech (Piper TTS only).
 * When enabled, uses native C++ Piper (if native_tts is set) or invokes the Python runner.
 */
class TtsHandler {
public:
    /** Disabled state: handle_speech returns empty and error message. */
    TtsHandler();

    /**
     * Enable TTS: native_tts (if non-null and initialized) or runner_argv.
     * runner_argv: e.g. {"python3", "-m", "rkllama.scripts.piper_tts_cli"} when not using native.
     */
    TtsHandler(std::string tts_model_path, std::vector<std::string> runner_argv, PiperTts* native_tts = nullptr);

    /** Returns true if TTS is enabled (native Piper or model path + runner). */
    bool enabled() const;

    /**
     * Handle POST body (JSON): input, model?, voice?, response_format?, speed?.
     * Returns (audio_bytes, content_type) on success; on error content_type is empty and audio_bytes may contain error JSON.
     */
    std::pair<std::vector<uint8_t>, std::string> handle_speech(const std::string& body) const;

private:
    std::string tts_model_path_;
    std::vector<std::string> runner_argv_;
    PiperTts* native_tts_ = nullptr;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP
