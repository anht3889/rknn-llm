#ifndef RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP
#define RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP

#include <cstddef>
#include "rkllm.h"
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <mutex>
#include <queue>
#include <memory>
#include <thread>
#include <condition_variable>

namespace rkllm_openai {

/** Optional multimodal input (image embeddings). When non-null in run(), use RKLLM_INPUT_MULTIMODAL. */
struct MultimodalInput {
    const float* image_embed = nullptr;
    size_t n_image_tokens = 0;
    size_t n_image = 0;
    size_t image_width = 0;
    size_t image_height = 0;
};

/** Result of encoding an image: embedding vector + MultimodalInput (image_embed points into the vector). */
using MultimodalInputResult = std::pair<std::vector<float>, MultimodalInput>;

struct RunResult {
    std::string content;
    int prefill_tokens = 0;
    int completion_tokens = 0;
    float prefill_time_ms = 0.f;
    float generate_time_ms = 0.f;
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
              float top_p = 0.9f,
              const std::string& prompt_cache_path = {},
              const std::string& img_start = {},
              const std::string& img_end = {},
              const std::string& img_content = {});

    bool is_busy() const;
    RunResult run(const std::string& prompt,
                 const std::string& role,
                 bool enable_thinking,
                 const std::string* tools_json,
                 const std::string* system_prompt,
                 const MultimodalInput* multimodal = nullptr);

    /** Start inference in a background thread for streaming. Call pop_stream_chunk() until done. */
    bool run_streaming_start(const std::string& prompt,
                             const std::string& role,
                             bool enable_thinking,
                             const std::string* tools_json,
                             const std::string* system_prompt,
                             const MultimodalInput* multimodal = nullptr);

    /** Block until next token chunk or stream end. Returns true if chunk is valid; false if stream finished (check out.error). */
    bool pop_stream_chunk(std::string& chunk, RunResult& out);

    void abort();

private:
    LLMHandle handle_ = nullptr;
    std::string model_path_;
    std::string platform_;

    std::mutex serialize_mutex_;  /* one inference at a time */
    mutable std::mutex run_mutex_;
    std::condition_variable stream_cv_;
    std::queue<std::string> chunk_queue_;  /* filled by callback during rkllm_run */
    std::atomic<int> call_state_{ -1 };
    int prefill_tokens_ = 0;
    int completion_tokens_ = 0;
    float prefill_time_ms_ = 0.f;
    float generate_time_ms_ = 0.f;

    bool streaming_mode_ = false;
    bool stream_finished_ = false;
    std::thread run_thread_;

    static int static_callback(RKLLMResult* result, void* userdata, LLMCallState state);
    void push_chunk(const char* text);
    void set_finished(int state, int prefill, int completion, float prefill_ms, float generate_ms);
    void run_streaming_thread(const std::string& prompt,
                              const std::string& role,
                              bool enable_thinking,
                              const std::string* tools_json,
                              const std::string* system_prompt,
                              const MultimodalInput* multimodal);
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_RKLLM_BACKEND_HPP
