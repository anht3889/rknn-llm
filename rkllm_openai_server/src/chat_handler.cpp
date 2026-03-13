#include "chat_handler.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <chrono>

namespace rkllm_openai {

using json = nlohmann::json;

ChatHandler::ChatHandler(RKLLMBackend* backend) : backend_(backend) {}

std::string ChatHandler::handle_chat_completions(const std::string& body,
                                                 bool stream,
                                                 std::function<void(const std::string&)> stream_write) {
    json data;
    try {
        data = json::parse(body);
    } catch (const json::exception&) {
        json err;
        err["error"] = {{"message", "Invalid JSON"}, {"type", "invalid_request_error"}};
        return err.dump();
    }

    if (!data.contains("messages") || !data["messages"].is_array()) {
        json err;
        err["error"] = {{"message", "Missing or invalid 'messages' array"}, {"type", "invalid_request_error"}};
        return err.dump();
    }

    if (data.contains("stream") && data["stream"].is_boolean())
        stream = data["stream"].get<bool>();

    json result = parse_messages_and_run(data, stream, stream_write);
    if (result.contains("error"))
        return result.dump();
    if (stream)
        return {};
    return result.dump();
}

json ChatHandler::parse_messages_and_run(const json& data,
                                         bool stream,
                                         std::function<void(const std::string&)> stream_write) {
    const json& messages = data["messages"];
    bool enable_thinking = data.value("enable_thinking", false);
    std::string model_id = data.value("model", "rkllm");
    const json* tools = data.contains("tools") && !data["tools"].is_null() ? &data["tools"] : nullptr;
    std::string tools_json = tools ? tools->dump() : "";

    std::string system_prompt;
    std::string prompt;
    std::string role = "user";

    for (const auto& msg : messages) {
        if (!msg.is_object() || !msg.contains("role") || !msg.contains("content"))
            continue;
        std::string r = msg["role"].get<std::string>();
        std::string content = msg["content"].is_string() ? msg["content"].get<std::string>() : "";
        if (r == "system") {
            system_prompt = content;
            continue;
        }
        if (r == "assistant")
            continue;
        if (r == "user") {
            prompt = content;
            role = "user";
            continue;
        }
        if (r == "tool") {
            prompt = content;
            role = "tool";
            continue;
        }
    }

    if (prompt.empty()) {
        json err;
        err["error"] = {{"message", "No user or tool message content"}, {"type", "invalid_request_error"}};
        return err;
    }

    /* Serialization is done inside backend->run() (one inference at a time). */

    const std::string* tools_ptr = tools_json.empty() ? nullptr : &tools_json;
    const std::string* sys_ptr = system_prompt.empty() ? nullptr : &system_prompt;

    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();
    std::string id = "chatcmpl-" + std::to_string(created);

    if (!stream) {
        RunResult run_result = backend_->run(prompt, role, enable_thinking, tools_ptr, sys_ptr, false, nullptr);
        if (run_result.error) {
            json err;
            err["error"] = {{"message", "Inference failed"}, {"type", "server_error"}};
            return err;
        }
        json resp;
        resp["id"] = id;
        resp["object"] = "chat.completion";
        resp["created"] = created;
        resp["model"] = model_id;
        resp["choices"] = json::array({
            {{"index", 0},
             {"message", {{"role", "assistant"}, {"content", run_result.content}}},
             {"finish_reason", "stop"}}
        });
        resp["usage"] = {
            {"prompt_tokens", run_result.prefill_tokens},
            {"completion_tokens", run_result.completion_tokens},
            {"total_tokens", run_result.prefill_tokens + run_result.completion_tokens}
        };
        return resp;
    }

    std::string full_content;
    RunResult run_result = backend_->run(prompt, role, enable_thinking, tools_ptr, sys_ptr, true,
        [&full_content, &stream_write, &id, &model_id, created](const std::string& chunk) {
            full_content += chunk;
            json chunk_obj;
            chunk_obj["id"] = id;
            chunk_obj["object"] = "chat.completion.chunk";
            chunk_obj["created"] = created;
            chunk_obj["model"] = model_id;
            chunk_obj["choices"] = json::array({
                {{"index", 0},
                 {"delta", {{"content", chunk}}},
                 {"finish_reason", json::value_t::null}}
            });
            if (stream_write)
                stream_write(chunk_obj.dump() + "\n");
        });

    json finish_obj;
    finish_obj["id"] = id;
    finish_obj["object"] = "chat.completion.chunk";
    finish_obj["created"] = created;
    finish_obj["model"] = model_id;
    finish_obj["choices"] = json::array({
        {{"index", 0},
         {"delta", {}},
         {"finish_reason", "stop"}}
    });
    finish_obj["usage"] = {
        {"prompt_tokens", run_result.prefill_tokens},
        {"completion_tokens", run_result.completion_tokens},
        {"total_tokens", run_result.prefill_tokens + run_result.completion_tokens}
    };
    if (stream_write)
        stream_write(finish_obj.dump() + "\n");

    return json::object();
}

}  // namespace rkllm_openai
