#include "chat_handler.hpp"
#include "base64.hpp"
#include "httplib.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace rkllm_openai {

using json = nlohmann::json;

ChatHandler::ChatHandler(RKLLMBackend* backend, bool debug, EncodeImageFn encode_image)
    : backend_(backend), debug_(debug), encode_image_(std::move(encode_image)) {}

void ChatHandler::log_debug_stats(const RunResult& r) const {
    if (!debug_) return;
    float prefill_tps = (r.prefill_time_ms > 0.f) ? (r.prefill_tokens / (r.prefill_time_ms / 1000.f)) : 0.f;
    float gen_tps = (r.generate_time_ms > 0.f) ? (r.completion_tokens / (r.generate_time_ms / 1000.f)) : 0.f;
    std::cerr << "[rkllm_debug] prefill_tokens=" << r.prefill_tokens
              << " prefill_time_ms=" << std::fixed << std::setprecision(2) << r.prefill_time_ms
              << " prefill_speed=" << std::setprecision(1) << prefill_tps << " tok/s"
              << " | generate_tokens=" << r.completion_tokens
              << " generate_time_ms=" << std::setprecision(2) << r.generate_time_ms
              << " generate_speed=" << std::setprecision(1) << gen_tps << " tok/s"
              << std::endl;
}

std::string ChatHandler::handle_chat_completions(const std::string& body, void* res) {
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

    bool stream = data.contains("stream") && data["stream"].is_boolean() && data["stream"].get<bool>();
    auto* response = static_cast<httplib::Response*>(res);

    if (stream && !response) {
        json err;
        err["error"] = {{"message", "Streaming requires a response object."}, {"type", "invalid_request_error"}};
        return err.dump();
    }

    if (stream && response) {
        /* Streaming path: parse same as non-stream, then run_streaming_start + chunked SSE. */
        json parse_result = parse_messages_and_run(data);
        if (parse_result.contains("error"))
            return parse_result.dump();

        /* We need to run streaming: parse_messages_and_run already ran backend_->run() once.
         * So we need a variant that only parses and returns (prompt, role, ...) without running.
         * For simplicity we add a dedicated streaming flow that re-parses. */
        const json& messages = data["messages"];
        bool enable_thinking = data.value("enable_thinking", false);
        std::string model_id = data.value("model", "rkllm");
        const json* tools = data.contains("tools") && !data["tools"].is_null() ? &data["tools"] : nullptr;
        std::string tools_json = tools ? tools->dump() : "";
        std::string system_prompt;
        std::string prompt;
        std::string role = "user";
        std::vector<std::vector<uint8_t>> last_images;

        for (const auto& msg : messages) {
            if (!msg.is_object() || !msg.contains("role") || !msg.contains("content")) continue;
            std::string r = msg["role"].get<std::string>();
            if (r == "system") {
                if (msg["content"].is_string()) system_prompt = msg["content"].get<std::string>();
                continue;
            }
            if (r == "assistant") continue;
            if (r == "user") {
                parse_content(msg["content"], prompt, last_images);
                role = "user";
                continue;
            }
            if (r == "tool") {
                if (msg["content"].is_string()) prompt = msg["content"].get<std::string>();
                last_images.clear();
                role = "tool";
                continue;
            }
        }
        if (prompt.empty()) {
            json err;
            err["error"] = {{"message", "No user or tool message content"}, {"type", "invalid_request_error"}};
            return err.dump();
        }
        std::optional<MultimodalInputResult> multimodal_result;
        if (!last_images.empty()) {
            if (!encode_image_) {
                json err;
                err["error"] = {{"message", "Multimodal (image) not configured."}, {"type", "invalid_request_error"}};
                return err.dump();
            }
            multimodal_result = encode_image_(last_images[0]);
            if (!multimodal_result) {
                json err;
                err["error"] = {{"message", "Image encoding failed."}, {"type", "server_error"}};
                return err.dump();
            }
        }
        const std::string* tools_ptr = tools_json.empty() ? nullptr : &tools_json;
        const std::string* sys_ptr = system_prompt.empty() ? nullptr : &system_prompt;
        const MultimodalInput* multimodal_ptr = multimodal_result ? &multimodal_result->second : nullptr;

        if (!backend_->run_streaming_start(prompt, role, enable_thinking, tools_ptr, sys_ptr, multimodal_ptr)) {
            json err;
            err["error"] = {{"message", "Failed to start streaming inference."}, {"type", "server_error"}};
            return err.dump();
        }

        auto now = std::chrono::system_clock::now().time_since_epoch();
        int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();
        std::string id = "chatcmpl-" + std::to_string(created);

        response->set_header("Content-Type", "text/event-stream");
        response->set_header("Cache-Control", "no-cache");
        response->set_header("Access-Control-Allow-Origin", "*");
        response->set_chunked_content_provider(
            "text/event-stream",
            [this, backend = backend_, model_id, created, id](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::string chunk;
                RunResult final_result;
                if (backend->pop_stream_chunk(chunk, final_result)) {
                    json delta = {{"content", chunk}, {"role", "assistant"}};
                    json choice = {{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", nullptr}};
                    json line = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                                {"model", model_id}, {"choices", json::array({choice})}};
                    std::string line_str = "data: " + line.dump() + "\n\n";
                    if (!sink.write(line_str.data(), line_str.size())) return false;
                    return true;
                }
                /* Stream finished: send final chunk with usage and [DONE]. */
                json usage = {
                    {"prompt_tokens", final_result.prefill_tokens},
                    {"completion_tokens", final_result.completion_tokens},
                    {"total_tokens", final_result.prefill_tokens + final_result.completion_tokens}
                };
                json final_chunk = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                                   {"model", model_id}, {"choices", json::array()}, {"usage", std::move(usage)}};
                std::string final_str = "data: " + final_chunk.dump() + "\n\ndata: [DONE]\n\n";
                if (sink.write(final_str.data(), final_str.size())) { /* ignore */ }
                if (debug_) log_debug_stats(final_result);
                sink.done();
                return false;
            });

        return "";
    }

    json result = parse_messages_and_run(data);
    if (result.contains("error"))
        return result.dump();
    return result.dump();
}

/* Parse OpenAI-style content: string or array of { type: "text"|"image_url", text?: string, image_url?: { url: "data:image/...;base64,..." } }. */
static void parse_content(const json& content, std::string& prompt_out, std::vector<std::vector<uint8_t>>& images_out) {
    prompt_out.clear();
    images_out.clear();
    if (content.is_string()) {
        prompt_out = content.get<std::string>();
        return;
    }
    if (!content.is_array())
        return;
    for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type"))
            continue;
        std::string type = part["type"].get<std::string>();
        if (type == "text" && part.contains("text") && part["text"].is_string()) {
            prompt_out += part["text"].get<std::string>();
        } else if (type == "image_url" && part.contains("image_url") && part["image_url"].is_object()) {
            const auto& img = part["image_url"];
            std::string url = img.contains("url") && img["url"].is_string() ? img["url"].get<std::string>() : "";
            /* data:image/jpeg;base64,<data> or data:image/png;base64,<data> */
            const std::string prefix = "data:";
            size_t comma = url.find(',');
            if (url.size() > prefix.size() && url.compare(0, prefix.size(), prefix) == 0 && comma != std::string::npos) {
                std::string b64 = url.substr(comma + 1);
                std::vector<uint8_t> decoded = rkllm_openai::base64_decode(b64);
                if (!decoded.empty())
                    images_out.push_back(std::move(decoded));
            }
            prompt_out += "<image>";  /* placeholder; model uses img_start/img_end/img_content */
        }
    }
}

json ChatHandler::parse_messages_and_run(const json& data) {
    const json& messages = data["messages"];
    bool enable_thinking = data.value("enable_thinking", false);
    std::string model_id = data.value("model", "rkllm");
    const json* tools = data.contains("tools") && !data["tools"].is_null() ? &data["tools"] : nullptr;
    std::string tools_json = tools ? tools->dump() : "";

    std::string system_prompt;
    std::string prompt;
    std::string role = "user";
    std::vector<std::vector<uint8_t>> last_images;

    for (const auto& msg : messages) {
        if (!msg.is_object() || !msg.contains("role") || !msg.contains("content"))
            continue;
        std::string r = msg["role"].get<std::string>();
        if (r == "system") {
            if (msg["content"].is_string())
                system_prompt = msg["content"].get<std::string>();
            continue;
        }
        if (r == "assistant")
            continue;
        if (r == "user") {
            parse_content(msg["content"], prompt, last_images);
            role = "user";
            continue;
        }
        if (r == "tool") {
            if (msg["content"].is_string())
                prompt = msg["content"].get<std::string>();
            last_images.clear();
            role = "tool";
            continue;
        }
    }

    if (prompt.empty()) {
        json err;
        err["error"] = {{"message", "No user or tool message content"}, {"type", "invalid_request_error"}};
        return err;
    }

    std::optional<MultimodalInputResult> multimodal_result;
    if (!last_images.empty()) {
        if (!encode_image_) {
            json err;
            err["error"] = {{"message", "Multimodal (image) not configured. Start server with --encoder_model_path and a vision LLM (--img_start/--img_end/--img_content)."}, {"type", "invalid_request_error"}};
            return err;
        }
        /* Use first image only (RKLLM multimodal demo uses n_image=1). */
        multimodal_result = encode_image_(last_images[0]);
        if (!multimodal_result) {
            json err;
            err["error"] = {{"message", "Image encoding failed."}, {"type", "server_error"}};
            return err;
        }
    }

    /* Serialization is done inside backend->run() (one inference at a time). */

    const std::string* tools_ptr = tools_json.empty() ? nullptr : &tools_json;
    const std::string* sys_ptr = system_prompt.empty() ? nullptr : &system_prompt;

    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();
    std::string id = "chatcmpl-" + std::to_string(created);

    const MultimodalInput* multimodal_ptr = multimodal_result ? &multimodal_result->second : nullptr;

    RunResult run_result = backend_->run(prompt, role, enable_thinking, tools_ptr, sys_ptr, multimodal_ptr);
    if (debug_) log_debug_stats(run_result);
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

}  // namespace rkllm_openai
