#include "rkllm_backend.hpp"
#include "rkllm.h"
#include <cstring>

namespace rkllm_openai {

int RKLLMBackend::static_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* self = static_cast<RKLLMBackend*>(userdata);
    if (!self) return 0;
    if (result) {
        if (result->text)
            self->push_chunk(result->text);
        if (state == RKLLM_RUN_FINISH || state == RKLLM_RUN_ERROR) {
            int prefill = 0, completion = 0;
            float prefill_ms = 0.f, generate_ms = 0.f;
            if (state == RKLLM_RUN_FINISH) {
                prefill = result->perf.prefill_tokens;
                completion = result->perf.generate_tokens;
                prefill_ms = result->perf.prefill_time_ms;
                generate_ms = result->perf.generate_time_ms;
            }
            self->set_finished(static_cast<int>(state), prefill, completion, prefill_ms, generate_ms);
        }
    }
    return 0;
}

void RKLLMBackend::push_chunk(const char* text) {
    if (!text) return;
    std::lock_guard<std::mutex> lock(run_mutex_);
    chunk_queue_.push(std::string(text));
}

void RKLLMBackend::set_finished(int state, int prefill, int completion, float prefill_ms, float generate_ms) {
    call_state_.store(state);
    prefill_tokens_ = prefill;
    completion_tokens_ = completion;
    prefill_time_ms_ = prefill_ms;
    generate_time_ms_ = generate_ms;
}

bool RKLLMBackend::init(const std::string& model_path,
                        const std::string& platform,
                        int max_context_len,
                        int max_new_tokens,
                        float temperature,
                        float top_p,
                        const std::string& prompt_cache_path,
                        const std::string& img_start,
                        const std::string& img_end,
                        const std::string& img_content) {
    model_path_ = model_path;
    platform_ = platform;

    /* Match working llm_demo exactly: use createDefaultParam() and only override same fields. */
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path_.c_str();
    param.top_k = 1;
    param.top_p = top_p;
    param.temperature = temperature;
    param.repeat_penalty = 1.1f;
    param.frequency_penalty = 0.0f;
    param.presence_penalty = 0.0f;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = img_start.empty() ? 0 : 1;  /* 1 for multimodal (vision) */
    param.extend_param.embed_flash = 1;
    if (!img_start.empty()) param.img_start = img_start.c_str();
    if (!img_end.empty()) param.img_end = img_end.c_str();
    if (!img_content.empty()) param.img_content = img_content.c_str();

    int ret = rkllm_init(&handle_, &param, static_callback);
    if (ret != 0)
        return false;
    if (!prompt_cache_path.empty() && rkllm_load_prompt_cache(handle_, prompt_cache_path.c_str()) != 0)
        return false;
    return true;
}

RKLLMBackend::~RKLLMBackend() {
    if (handle_) {
        rkllm_destroy(handle_);
        handle_ = nullptr;
    }
}

bool RKLLMBackend::is_busy() const {
    if (!handle_) return false;
    return rkllm_is_running(handle_) == 0;
}

void RKLLMBackend::abort() {
    if (handle_)
        rkllm_abort(handle_);
}

RunResult RKLLMBackend::run(const std::string& prompt,
                           const std::string& role,
                           bool enable_thinking,
                           const std::string* tools_json,
                           const std::string* system_prompt,
                           const MultimodalInput* multimodal) {
    RunResult out;
    if (!handle_) {
        out.error = true;
        return out;
    }

    std::lock_guard<std::mutex> serial_lock(serialize_mutex_);

    {
        std::lock_guard<std::mutex> lock(run_mutex_);
        while (!chunk_queue_.empty()) chunk_queue_.pop();
    }
    call_state_.store(-1);
    prefill_tokens_ = 0;
    completion_tokens_ = 0;
    prefill_time_ms_ = 0.f;
    generate_time_ms_ = 0.f;

    if (tools_json && !tools_json->empty() && system_prompt) {
        rkllm_set_function_tools(handle_, system_prompt->c_str(),
                                 tools_json->c_str(), "tool_response");
    }

    RKLLMInput rkllm_input;
    std::memset(&rkllm_input, 0, sizeof(rkllm_input));
    rkllm_input.role = role.empty() ? "user" : role.c_str();
    rkllm_input.enable_thinking = enable_thinking;
    if (multimodal && multimodal->image_embed && multimodal->n_image_tokens > 0) {
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.multimodal_input.prompt = const_cast<char*>(prompt.c_str());
        rkllm_input.multimodal_input.image_embed = const_cast<float*>(multimodal->image_embed);
        rkllm_input.multimodal_input.n_image_tokens = multimodal->n_image_tokens;
        rkllm_input.multimodal_input.n_image = multimodal->n_image > 0 ? multimodal->n_image : 1;
        rkllm_input.multimodal_input.image_width = multimodal->image_width;
        rkllm_input.multimodal_input.image_height = multimodal->image_height;
    } else {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.prompt_input = prompt.c_str();
    }

    RKLLMInferParam rkllm_infer_params;
    std::memset(&rkllm_infer_params, 0, sizeof(rkllm_infer_params));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    rkllm_infer_params.lora_params = nullptr;
    rkllm_infer_params.prompt_cache_params = nullptr;

    int ret = rkllm_run(handle_, &rkllm_input, &rkllm_infer_params, this);

    std::string accumulated;
    {
        std::lock_guard<std::mutex> lock(run_mutex_);
        while (!chunk_queue_.empty()) {
            accumulated += chunk_queue_.front();
            chunk_queue_.pop();
        }
    }
    out.content = accumulated;

    out.prefill_tokens = prefill_tokens_;
    out.completion_tokens = completion_tokens_;
    out.prefill_time_ms = prefill_time_ms_;
    out.generate_time_ms = generate_time_ms_;
    out.error = (ret != 0 || call_state_.load() == static_cast<int>(RKLLM_RUN_ERROR));
    return out;
}

}  // namespace rkllm_openai
