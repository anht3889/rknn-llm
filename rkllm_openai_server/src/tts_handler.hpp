#ifndef RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP
#define RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP

#include <string>
#include <vector>

namespace rkllm_openai {

/**
 * Handler for OpenAI-style POST /v1/audio/speech (Piper TTS only).
 * When enabled, invokes the Piper TTS runner script (e.g. rkllama) and returns audio.
 */
class TtsHandler {
public:
    /** Disabled state: handle_speech returns empty and error message. */
    TtsHandler();

    /**
     * Enable TTS with the given Piper model path and runner argv.
     * runner_argv: e.g. {"python3", "-m", "rkllama.scripts.piper_tts_cli"} or {"python3", "/path/to/piper_tts_cli.py"}.
     * The handler spawns the runner with JSON on stdin; expects "Content-Type: ...\n" then raw audio on stdout.
     */
    TtsHandler(std::string tts_model_path, std::vector<std::string> runner_argv);

    /** Returns true if TTS is enabled (model path and runner set). */
    bool enabled() const;

    /**
     * Handle POST body (JSON): input, model?, voice?, response_format?, speed?.
     * Returns (audio_bytes, content_type) on success; on error content_type is empty and audio_bytes may contain error JSON.
     */
    std::pair<std::vector<uint8_t>, std::string> handle_speech(const std::string& body) const;

private:
    std::string tts_model_path_;
    std::vector<std::string> runner_argv_;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_TTS_HANDLER_HPP
