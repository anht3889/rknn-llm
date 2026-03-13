#include "piper_decoder.hpp"

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_RKNN)
#include "rknn_api.h"
#include <cstdint>
#include <cstring>
#endif

namespace rkllm_openai {

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_RKNN)

PiperDecoder::~PiperDecoder() {
    release();
}

bool PiperDecoder::init(const std::string& rknn_path) {
    if (ctx_) return true;
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, const_cast<char*>(rknn_path.c_str()), 0, 0, nullptr);
    if (ret != 0) return false;

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        rknn_destroy(ctx);
        return false;
    }

    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret != 0) {
        rknn_destroy(ctx);
        return false;
    }
    // Input 0: z [1, C, T] -> dims[2] = T
    chunk_time_ = static_cast<size_t>(input_attr.dims[2]);
    z_channels_ = static_cast<size_t>(input_attr.dims[1]);

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (ret != 0) {
        rknn_destroy(ctx);
        return false;
    }
    out_samples_ = 1;
    for (int i = 0; i < out_attr.n_dims; i++)
        if (out_attr.dims[i] > 1) out_samples_ *= static_cast<size_t>(out_attr.dims[i]);

    ctx_ = reinterpret_cast<void*>(static_cast<uintptr_t>(ctx));
    return true;
}

void PiperDecoder::release() {
    if (ctx_) {
        rknn_destroy(reinterpret_cast<rknn_context>(reinterpret_cast<uintptr_t>(ctx_)));
        ctx_ = nullptr;
    }
    chunk_time_ = 0;
    z_channels_ = 0;
    out_samples_ = 0;
}

bool PiperDecoder::run_chunk(const float* z, const float* y_mask, std::vector<float>& out_audio) {
    rknn_context ctx = reinterpret_cast<rknn_context>(reinterpret_cast<uintptr_t>(ctx_));
    if (!ctx || !z || !y_mask) return false;

    size_t z_size = 1 * z_channels_ * chunk_time_ * sizeof(float);
    size_t y_size = 1 * 1 * chunk_time_ * sizeof(float);

    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NCHW;
    inputs[0].size = static_cast<uint32_t>(z_size);
    inputs[0].buf = const_cast<float*>(z);
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = RKNN_TENSOR_NCHW;
    inputs[1].size = static_cast<uint32_t>(y_size);
    inputs[1].buf = const_cast<float*>(y_mask);

    int ret = rknn_inputs_set(ctx, 2, inputs);
    if (ret != 0) return false;
    ret = rknn_run(ctx, nullptr);
    if (ret != 0) return false;

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret != 0) return false;

    out_audio.resize(out_samples_);
    memcpy(out_audio.data(), outputs[0].buf, out_samples_ * sizeof(float));
    rknn_outputs_release(ctx, 1, outputs);
    return true;
}

#else

PiperDecoder::~PiperDecoder() {}
bool PiperDecoder::init(const std::string&) { return false; }
void PiperDecoder::release() {}
bool PiperDecoder::run_chunk(const float*, const float*, std::vector<float>&) { return false; }

#endif
}  // namespace rkllm_openai
