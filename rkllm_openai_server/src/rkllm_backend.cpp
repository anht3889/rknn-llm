#include "rkllm_backend.hpp"
#include "rkllm.h"
#include <cstring>
#include <thread>

namespace rkllm_openai {

int RKLLMBackend::static_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* self = static_cast<RKLLMBackend*>(userdata);
    if (!self) return 0;
    if (result) {
        if (result->text)
            self->push_chunk(result->text);
        if (state == RKLLM_RUN_FINISH || state == RKLLM_RUN_ERROR) {
            int prefill = 0, completion = 0;
            if (state == RKLLM_RUN_FINISH) {
                prefill = result->perf.prefill_tokens;
                completion = result->perf.generate_tokens;
            }
            self->set_finished(static_cast<int>(state), prefill, completion);
        }
    }
    return 0;
}

void RKLLMBackend::push_chunk(const char* text) {
    if (!text) return;
    std::lock_guard<std::mutex> lock(run_mutex_);
    stream_queue_.push(std::string(text));
    stream_cv_.notify_one();
}

void RKLLMBackend::set_finished(int state, int prefill, int completion) {
    call_state_.store(state);
    prefill_tokens_ = prefill;
    completion_tokens_ = completion;
    run_finished_.store(true);
    stream_cv_.notify_one();
}

bool RKLLMBackend::init(const std::string& model_path,
                        const std::string& platform,
                        int max_context_len,
                        int max_new_tokens,
                        float temperature,
                        float top_p) {
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
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;

    int ret = rkllm_init(&handle_, &param, static_callback);
    return ret == 0;
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
                           bool stream,
                           std::function<void(const std::string&)> on_stream_chunk) {
    RunResult out;
    if (!handle_) {
        out.error = true;
        return out;
    }

    {
        std::lock_guard<std::mutex> lock(run_mutex_);
        while (!stream_queue_.empty()) stream_queue_.pop();
    }
    call_state_.store(-1);
    run_finished_.store(false);
    prefill_tokens_ = 0;
    completion_tokens_ = 0;

    if (tools_json && !tools_json->empty() && system_prompt) {
        rkllm_set_function_tools(handle_, system_prompt->c_str(),
                                 tools_json->c_str(), "tool_response");
    }

    RKLLMInput rkllm_input;
    std::memset(&rkllm_input, 0, sizeof(rkllm_input));
    rkllm_input.role = role.empty() ? "user" : role.c_str();
    rkllm_input.enable_thinking = enable_thinking;
    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.prompt_input = prompt.c_str();

    RKLLMInferParam rkllm_infer_params;
    std::memset(&rkllm_infer_params, 0, sizeof(rkllm_infer_params));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    rkllm_infer_params.lora_params = nullptr;
    rkllm_infer_params.prompt_cache_params = nullptr;

    if (!stream) {
        int ret = rkllm_run(handle_, &rkllm_input, &rkllm_infer_params, this);
        std::string accumulated;
        {
            std::lock_guard<std::mutex> lock(run_mutex_);
            while (!stream_queue_.empty()) {
                accumulated += stream_queue_.front();
                stream_queue_.pop();
            }
        }
        out.content = accumulated;
        out.prefill_tokens = prefill_tokens_;
        out.completion_tokens = completion_tokens_;
        out.error = (ret != 0 || call_state_.load() == static_cast<int>(RKLLM_RUN_ERROR));
        return out;
    }

    std::thread th([this, rkllm_input, rkllm_infer_params]() {
        rkllm_run(handle_, const_cast<RKLLMInput*>(&rkllm_input),
                  const_cast<RKLLMInferParam*>(&rkllm_infer_params), this);
    });

    while (!run_finished_.load() || !stream_queue_.empty()) {
        std::string chunk;
        {
            std::unique_lock<std::mutex> lock(run_mutex_);
            stream_cv_.wait_for(lock, std::chrono::milliseconds(50),
                                [this]() { return run_finished_.load() || !stream_queue_.empty(); });
            if (!stream_queue_.empty()) {
                chunk = stream_queue_.front();
                stream_queue_.pop();
            }
        }
        if (!chunk.empty() && on_stream_chunk)
            on_stream_chunk(chunk);
    }
    if (th.joinable()) th.join();

    out.prefill_tokens = prefill_tokens_;
    out.completion_tokens = completion_tokens_;
    out.error = (call_state_.load() == static_cast<int>(RKLLM_RUN_ERROR));
    return out;
}

}  // namespace rkllm_openai
