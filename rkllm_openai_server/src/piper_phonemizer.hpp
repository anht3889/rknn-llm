#ifndef RKLLM_OPENAI_SERVER_PIPER_PHONEMIZER_HPP
#define RKLLM_OPENAI_SERVER_PIPER_PHONEMIZER_HPP

#include "piper_config.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace rkllm_openai {

/**
 * Convert text to Piper phoneme IDs using config's phoneme_id_map.
 * When built with espeak-ng: use espeak_TextToPhonemes and map to IDs.
 * Otherwise: stub returns false (phonemization not available).
 */
class PiperPhonemizer {
public:
    PiperPhonemizer() = default;

    /** Set espeak data path (optional, e.g. /usr/share/espeak-ng-data). */
    void set_espeak_data_path(const std::string& path) { espeak_data_path_ = path; }

    /**
     * Convert text to phoneme IDs. Prepends BOS (^) and appends EOS ($) if in config.
     * Returns true on success; ids are appended to out_ids.
     */
    bool phonemize(const PiperConfig& config, const std::string& text, std::vector<int64_t>& out_ids);

private:
    std::string espeak_data_path_;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_PIPER_PHONEMIZER_HPP
