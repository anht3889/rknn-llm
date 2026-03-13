#ifndef RKLLM_OPENAI_SERVER_IMAGE_ENCODER_HPP
#define RKLLM_OPENAI_SERVER_IMAGE_ENCODER_HPP

#include "rkllm_backend.hpp"
#include <string>
#include <vector>
#include <optional>
#include <cstddef>

namespace rkllm_openai {

/**
 * Wraps the RKNN image encoder (vision encoder model) to produce image embeddings
 * for multimodal LLM input. Preprocesses input image bytes (JPEG/PNG) and runs
 * the encoder. Used when ENABLE_MULTIMODAL is on and --encoder_model_path is set.
 */
class ImageEncoder {
public:
    ImageEncoder() = default;
    ~ImageEncoder();

    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /** Initialize encoder from RKNN model path. core_num: NPU cores (e.g. 1). Returns true on success. */
    bool init(const std::string& model_path, int core_num = 1);

    bool is_initialized() const { return initialized_; }

    /**
     * Encode image bytes (JPEG or PNG) to embedding + MultimodalInput.
     * Preprocessing: decode, BGR->RGB, expand2square(127.5), resize to model size.
     * Returns nullopt on failure.
     */
    std::optional<MultimodalInputResult> encode(const std::vector<uint8_t>& image_bytes);

private:
    bool initialized_ = false;
    void* ctx_ = nullptr;  /* opaque: rknn_app_context_t */
};

}  // namespace rkllm_openai

#endif
