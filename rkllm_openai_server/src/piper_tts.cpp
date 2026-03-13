#include "piper_tts.hpp"
#include "wav_writer.hpp"
#include <algorithm>
#include <cstring>

namespace rkllm_openai {

PiperTts::~PiperTts() {
    release();
}

bool PiperTts::init(const std::string& model_dir) {
    if (initialized_) return true;
    std::string onnx_path, rknn_path, json_path;
    find_piper_model_files(model_dir, onnx_path, rknn_path, json_path);
    if (onnx_path.empty() || rknn_path.empty() || json_path.empty())
        return false;
    if (!config_.load_from_file(json_path))
        return false;
    if (!encoder_.init(onnx_path))
        return false;
    if (!decoder_.init(rknn_path))
        return false;
    model_dir_ = model_dir;
    initialized_ = true;
    return true;
}

void PiperTts::release() {
    encoder_.release();
    decoder_.release();
    initialized_ = false;
    model_dir_.clear();
}

std::vector<uint8_t> PiperTts::synthesize(const std::string& text, const std::string& voice, float speed) {
    last_error_.clear();
    if (!initialized_ || text.empty()) {
        last_error_ = "TTS not initialized or empty input.";
        return {};
    }

    std::vector<int64_t> phoneme_ids;
    if (!phonemizer_.phonemize(config_, text, phoneme_ids) || phoneme_ids.empty()) {
        last_error_ = "Phonemization failed. Install espeak-ng (libespeak-ng-dev) and rebuild with ENABLE_TTS=ON, or use Python TTS (--tts_runner).";
        return {};
    }

    float length_scale = config_.length_scale;
    if (speed > 0.f) length_scale = 1.f / speed;
    int64_t speaker_id = -1;
    if (!voice.empty()) {
        speaker_id = config_.get_speaker_id(voice);
        if (speaker_id < 0 && config_.num_speakers > 1) speaker_id = 0;
    }

    std::vector<float> z, y_mask;
    if (!encoder_.run(phoneme_ids.data(), phoneme_ids.size(),
                      config_.noise_scale, length_scale, config_.noise_w_scale,
                      speaker_id, z, y_mask)) {
        last_error_ = "ONNX encoder failed (check model and input).";
        return {};
    }

    size_t chunk_time = decoder_.get_chunk_time();
    if (chunk_time == 0) {
        last_error_ = "Decoder chunk size is zero.";
        return {};
    }
    size_t C = encoder_.get_z_channels();
    size_t T = encoder_.get_z_time();
    if (C == 0 || T == 0 || z.size() != C * T || y_mask.size() != T) {
        last_error_ = "Encoder output shape mismatch.";
        return {};
    }

    const size_t step = chunk_time;
    std::vector<float> audio_all;
    for (size_t start = 0; start < T; start += step) {
        size_t end = std::min(start + chunk_time, T);
        size_t cur_len = end - start;
        bool need_pad = (cur_len < chunk_time);

        std::vector<float> z_chunk(1 * C * chunk_time, 0.f);
        std::vector<float> y_chunk(1 * 1 * chunk_time, 0.f);
        memcpy(z_chunk.data(), z.data() + start * C, cur_len * C * sizeof(float));
        memcpy(y_chunk.data(), y_mask.data() + start, cur_len * sizeof(float));

        std::vector<float> out_chunk;
        if (!decoder_.run_chunk(z_chunk.data(), y_chunk.data(), out_chunk)) {
            last_error_ = "RKNN decoder failed.";
            return {};
        }

        if (need_pad) {
            size_t out_total = out_chunk.size();
            size_t keep = static_cast<size_t>(out_total * static_cast<double>(cur_len) / static_cast<double>(chunk_time));
            if (keep < out_total) out_chunk.resize(keep);
        }
        audio_all.insert(audio_all.end(), out_chunk.begin(), out_chunk.end());
    }

    return wav_bytes_from_float_mono(audio_all.data(), audio_all.size(), config_.sample_rate);
}

}  // namespace rkllm_openai
