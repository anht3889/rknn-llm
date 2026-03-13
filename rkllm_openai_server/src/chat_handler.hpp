#ifndef RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
#define RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP

#include "rkllm_backend.hpp"
#include <string>
#include <nlohmann/json_fwd.hpp>

namespace rkllm_openai {

class ChatHandler {
public:
    explicit ChatHandler(RKLLMBackend* backend);

    std::string handle_chat_completions(const std::string& body,
                                        bool stream,
                                        std::function<void(const std::string&)> stream_write);

private:
    RKLLMBackend* backend_;

    nlohmann::json parse_messages_and_run(const nlohmann::json& data,
                                          bool stream,
                                          std::function<void(const std::string&)> stream_write);
};

}  // namespace rkllm_openai

#endif  // RKLLM_OPENAI_SERVER_CHAT_HANDLER_HPP
