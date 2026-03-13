/* Image encoder for multimodal: requires ENABLE_MULTIMODAL, OpenCV, RKNN, and image_enc from multimodal_model_demo. */
#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL

#include "image_encoder.hpp"
#include "image_enc.h"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <vector>

namespace rkllm_openai {

static cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int w = img.cols, h = img.rows;
    if (w == h) return img.clone();
    int size = std::max(w, h);
    cv::Mat result(size, size, img.type(), background_color);
    int x = (size - w) / 2, y = (size - h) / 2;
    img.copyTo(result(cv::Rect(x, y, w, h)));
    return result;
}

ImageEncoder::~ImageEncoder() {
    if (ctx_) {
        release_imgenc(static_cast<rknn_app_context_t*>(ctx_));
        ctx_ = nullptr;
    }
    initialized_ = false;
}

bool ImageEncoder::init(const std::string& model_path, int core_num) {
    if (initialized_) return true;
    auto* ctx = new rknn_app_context_t();
    std::memset(ctx, 0, sizeof(rknn_app_context_t));
    if (init_imgenc(model_path.c_str(), ctx, core_num) != 0) {
        delete ctx;
        return false;
    }
    ctx_ = ctx;
    initialized_ = true;
    return true;
}

std::optional<MultimodalInputResult> ImageEncoder::encode(const std::vector<uint8_t>& image_bytes) {
    if (!ctx_ || image_bytes.empty()) return std::nullopt;
    rknn_app_context_t* ctx = static_cast<rknn_app_context_t*>(ctx_);

    cv::Mat img = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
    if (img.empty()) return std::nullopt;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Scalar bg(127.5, 127.5, 127.5);
    cv::Mat square = expand2square(img, bg);
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(ctx->model_width, ctx->model_height), 0, 0, cv::INTER_LINEAR);

    size_t n_embed = ctx->model_image_token * ctx->model_embed_size;
    if (ctx->io_num.n_output > 1) {
        n_embed = ctx->model_image_token * ctx->io_num.n_output * ctx->model_embed_size;
    }
    std::vector<float> out_vec(n_embed, 0.f);

    if (run_imgenc(ctx, resized.data, out_vec.data()) != 0)
        return std::nullopt;

    MultimodalInput multi;
    multi.n_image_tokens = static_cast<size_t>(ctx->model_image_token);
    multi.n_image = 1;
    multi.image_width = static_cast<size_t>(ctx->model_width);
    multi.image_height = static_cast<size_t>(ctx->model_height);

    MultimodalInputResult result(std::move(out_vec), multi);
    result.second.image_embed = result.first.data();
    return result;
}

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_ENABLE_MULTIMODAL
