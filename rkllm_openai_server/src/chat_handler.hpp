#ifndef RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
#define RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP

#include "rkllm_backend.hpp"
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <nlohmann/json_fwd.hpp>

namespace rkllm_openai {

/** Optional image encoder: (image bytes) -> (embedding + MultimodalInput) or nullopt if disabled/failed. */
using EncodeImageFn = std::function<std::optional<MultimodalInputResult>(const std::vector<uint8_t>&)>;

class ChatHandler {
public:
    ChatHandler(RKLLMBackend* backend, bool debug = false, EncodeImageFn encode_image = nullptr);

    std::string handle_chat_completions(const std::string& body);

private:
    RKLLMBackend* backend_;
    bool debug_;
    EncodeImageFn encode_image_;

    nlohmann::json parse_messages_and_run(const nlohmann::json& data);
    void log_debug_stats(const RunResult& r) const;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
