#ifndef RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP
#define RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP

#include <cstddef>
#include "rkllm.h"
#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <memory>

namespace rkllm_openai {

struct RunResult {
    std::string content;
    int prefill_tokens = 0;
    int completion_tokens = 0;
    bool error = false;
};

class RKLLMBackend {
public:
    RKLLMBackend() = default;
    ~RKLLMBackend();

    RKLLMBackend(const RKLLMBackend&) = delete;
    RKLLMBackend& operator=(const RKLLMBackend&) = delete;

    bool init(const std::string& model_path,
              const std::string& platform,
              int max_context_len = 4096,
              int max_new_tokens = 4096,
              float temperature = 0.8f,
              float top_p = 0.9f);

    bool is_busy() const;
    RunResult run(const std::string& prompt,
                 const std::string& role,
                 bool enable_thinking,
                 const std::string* tools_json,
                 const std::string* system_prompt,
                 bool stream,
                 std::function<void(const std::string&)> on_stream_chunk);

    void abort();

private:
    LLMHandle handle_ = nullptr;
    std::string model_path_;
    std::string platform_;

    mutable std::mutex run_mutex_;
    std::queue<std::string> stream_queue_;
    std::condition_variable stream_cv_;
    std::atomic<int> call_state_{ -1 };
    std::atomic<bool> run_finished_{ false };
    int prefill_tokens_ = 0;
    int completion_tokens_ = 0;

    static int static_callback(RKLLMResult* result, void* userdata, LLMCallState state);
    void push_chunk(const char* text);
    void set_finished(int state, int prefill, int completion);
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP
