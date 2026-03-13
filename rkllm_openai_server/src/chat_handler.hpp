#ifndef RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
#define RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP

#include "rkllm_backend.hpp"
#include <string>
#include <nlohmann/json_fwd.hpp>

namespace rkllm_openai {

class ChatHandler {
public:
    ChatHandler(RKLLMBackend* backend, bool debug = false);

    std::string handle_chat_completions(const std::string& body,
                                        bool stream,
                                        std::function<void(const std::string&)> stream_write);

private:
    RKLLMBackend* backend_;
    bool debug_;

    nlohmann::json parse_messages_and_run(const nlohmann::json& data,
                                          bool stream,
                                          std::function<void(const std::string&)> stream_write);
    void log_debug_stats(const RunResult& r) const;
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
