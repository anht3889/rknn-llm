#include "piper_phonemizer.hpp"
#include <cstring>
#include <sstream>

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_ESPEAK)
extern "C" {
#include <espeak-ng/speak_lib.h>
}
#endif

namespace rkllm_openai {

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_ESPEAK)

static bool g_espeak_initialized = false;

static bool ensure_espeak_init(const std::string& data_path) {
    if (g_espeak_initialized) return true;
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path.empty() ? nullptr : data_path.c_str(), 0);
    if (rate <= 0) return false;
    g_espeak_initialized = true;
    return true;
}

/** Advance to next UTF-8 codepoint; return length in bytes (1-4) or 0 at end. */
static int next_utf8(const char* p, size_t len, std::string& out_cp) {
    if (len == 0) return 0;
    unsigned char c = static_cast<unsigned char>(p[0]);
    int n = 1;
    if (c >= 0xf0) n = 4;
    else if (c >= 0xe0) n = 3;
    else if (c >= 0xc0) n = 2;
    if (static_cast<size_t>(n) > len) return 0;
    out_cp.assign(p, n);
    return n;
}

/** Map phoneme string to IDs: split by | then each token by UTF-8 codepoint, lookup in config. */
static void phoneme_string_to_ids(const PiperConfig& config, const std::string& phoneme_str, std::vector<int64_t>& out_ids) {
    int64_t bos = config.get_phoneme_id("^");
    int64_t eos = config.get_phoneme_id("$");
    if (bos >= 0) out_ids.push_back(bos);

    std::istringstream iss(phoneme_str);
    std::string token;
    while (std::getline(iss, token, '|')) {
        for (size_t i = 0; i < token.size(); ) {
            std::string sym;
            int n = next_utf8(token.c_str() + i, token.size() - i, sym);
            if (n <= 0) break;
            int64_t id = config.get_phoneme_id(sym);
            if (id >= 0) out_ids.push_back(id);
            i += static_cast<size_t>(n);
        }
    }

    if (eos >= 0) out_ids.push_back(eos);
}

bool PiperPhonemizer::phonemize(const PiperConfig& config, const std::string& text, std::vector<int64_t>& out_ids) {
    if (!ensure_espeak_init(espeak_data_path_)) return false;
    if (text.empty()) return true;

    espeak_SetVoiceByName(const_cast<char*>(config.espeak_voice.c_str()));
#ifdef espeakPHONEMES_UTF8
    int phoneme_flags = espeakPHONEMES_IPA | espeakPHONEMES_UTF8;
#else
    int phoneme_flags = espeakPHONEMES_IPA;
#endif
    const char* text_ptr = text.c_str();
    const char* phonemes = espeak_TextToPhonemes(reinterpret_cast<const void**>(&text_ptr), 0, phoneme_flags);
    if (!phonemes) return false;

    std::string phoneme_str(phonemes);
    phoneme_string_to_ids(config, phoneme_str, out_ids);
    return true;
}

#else  // no espeak

bool PiperPhonemizer::phonemize(const PiperConfig& config, const std::string& text, std::vector<int64_t>& out_ids) {
    (void)config;
    (void)text;
    (void)out_ids;
    return false;  // Build without espeak-ng: phonemization not available
}

#endif
}  // namespace rkllm_openai
