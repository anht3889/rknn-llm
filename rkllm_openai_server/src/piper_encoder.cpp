#include "piper_encoder.hpp"

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_ONNX)
#include <onnxruntime_cxx_api.h>
#include <vector>
#endif

namespace rkllm_openai {

#if defined(RKLLM_OPENAI_ENABLE_TTS) && defined(RKLLM_OPENAI_TTS_USE_ONNX)

namespace {
using namespace Ort;
}

PiperEncoder::~PiperEncoder() {
    release();
}

bool PiperEncoder::init(const std::string& onnx_path) {
    if (session_) return true;
    try {
        env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "piper_encoder");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = new Ort::Session(*static_cast<Ort::Env*>(env_), onnx_path.c_str(), opts);
        return true;
    } catch (...) {
        if (session_) {
            delete static_cast<Ort::Session*>(session_);
            session_ = nullptr;
        }
        if (env_) {
            delete static_cast<Ort::Env*>(env_);
            env_ = nullptr;
        }
        return false;
    }
}

void PiperEncoder::release() {
    if (session_) {
        delete static_cast<Ort::Session*>(session_);
        session_ = nullptr;
    }
    if (env_) {
        delete static_cast<Ort::Env*>(env_);
        env_ = nullptr;
    }
    z_channels_ = 0;
    z_time_ = 0;
}

bool PiperEncoder::run(const int64_t* phoneme_ids, size_t num_phonemes,
                       float noise_scale, float length_scale, float noise_w_scale,
                       int64_t speaker_id,
                       std::vector<float>& out_z, std::vector<float>& out_y_mask) {
    auto* sess = static_cast<Ort::Session*>(session_);
    if (!sess || num_phonemes == 0) return false;

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(num_phonemes)};
    std::vector<int64_t> input_ids(phoneme_ids, phoneme_ids + num_phonemes);
    std::vector<int64_t> input_lengths_data = {static_cast<int64_t>(num_phonemes)};
    std::vector<float> scales_data = {noise_scale, length_scale, noise_w_scale};

    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_tensors;

    input_names.push_back("input");
    input_tensors.push_back(Ort::Value::CreateTensor(mem_info, input_ids.data(), num_phonemes * sizeof(int64_t), input_shape.data(), input_shape.size()));

    input_names.push_back("input_lengths");
    input_tensors.push_back(Ort::Value::CreateTensor(mem_info, input_lengths_data.data(), input_lengths_data.size() * sizeof(int64_t), std::vector<int64_t>{1}.data(), 1));

    input_names.push_back("scales");
    input_tensors.push_back(Ort::Value::CreateTensor(mem_info, scales_data.data(), scales_data.size() * sizeof(float), std::vector<int64_t>{3}.data(), 1));

    if (speaker_id >= 0) {
        std::vector<int64_t> sid_data = {speaker_id};
        input_names.push_back("sid");
        input_tensors.push_back(Ort::Value::CreateTensor(mem_info, sid_data.data(), sizeof(int64_t), std::vector<int64_t>{1}.data(), 1));
    }

    const char* output_names[] = {"z", "y_mask"};
    auto output_tensors = sess->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names, 2);

    auto& z_tensor = output_tensors[0];
    auto& y_mask_tensor = output_tensors[1];
    auto z_shape = z_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto y_shape = y_mask_tensor.GetTensorTypeAndShapeInfo().GetShape();
    float* z_data = z_tensor.GetTensorMutableData<float>();
    float* y_data = y_mask_tensor.GetTensorMutableData<float>();
    size_t z_elems = 1;
    for (auto d : z_shape) z_elems *= static_cast<size_t>(d);
    size_t y_elems = 1;
    for (auto d : y_shape) y_elems *= static_cast<size_t>(d);

    z_channels_ = (z_shape.size() >= 2) ? static_cast<size_t>(z_shape[1]) : 0;
    z_time_ = (z_shape.size() >= 3) ? static_cast<size_t>(z_shape[2]) : 0;

    out_z.assign(z_data, z_data + z_elems);
    out_y_mask.assign(y_data, y_data + y_elems);
    return true;
}

#else  // !RKLLM_OPENAI_ENABLE_TTS || !RKLLM_OPENAI_TTS_USE_ONNX

PiperEncoder::~PiperEncoder() {}
bool PiperEncoder::init(const std::string&) { return false; }
void PiperEncoder::release() {}
bool PiperEncoder::run(const int64_t*, size_t, float, float, float, int64_t, std::vector<float>&, std::vector<float>&) { return false; }

#endif
}  // namespace rkllm_openai
