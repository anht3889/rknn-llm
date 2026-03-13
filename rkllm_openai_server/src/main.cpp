#include "chat_handler.hpp"
#include "rkllm_backend.hpp"
#include "fix_freq.hpp"
#include "httplib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

static std::string model_id = "rkllm";

/**
 * Auto-detect Rockchip platform by reading /proc/device-tree/compatible.
 * Returns platform string (e.g. "rk3588") or empty if not detected or not supported.
 */
static std::string detect_rockchip_platform() {
    std::ifstream f("/proc/device-tree/compatible", std::ios::binary);
    if (!f)
        return {};
    std::vector<char> buf(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());
    f.close();
    if (buf.empty())
        return {};
    buf.push_back('\0');
    const char* p = buf.data();
    while (p < buf.data() + buf.size() - 1) {
        size_t len = std::strlen(p);
        if (len == 0) {
            ++p;
            continue;
        }
        if (strcmp(p, "rockchip,rk3588") == 0) return "rk3588";
        if (strcmp(p, "rockchip,rk3576") == 0) return "rk3576";
        if (strcmp(p, "rockchip,rv1126") == 0) return "rv1126b";
        if (strcmp(p, "rockchip,rk3562") == 0) return "rk3562";
        p += len + 1;
    }
    return {};
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string platform = "auto";
    std::string host = "0.0.0.0";
    int port = 8080;
    bool debug = false;
    int max_context_len = 4096;
    int max_new_tokens = 4096;
    std::string prompt_cache_path;
    bool fix_freq = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) {
            platform = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_context_len") == 0 && i + 1 < argc) {
            max_context_len = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_new_tokens") == 0 && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt_cache") == 0 && i + 1 < argc) {
            prompt_cache_path = argv[++i];
        } else if (strcmp(argv[i], "--no-fix-freq") == 0) {
            fix_freq = false;
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " --model_path <path> [--platform auto|rk3588|rk3576|rv1126b|rk3562] [--host 0.0.0.0] [--port 8080]\n"
                      << "       [--max_context_len 4096] [--max_new_tokens 4096] [--prompt_cache <path>]\n"
                      << "       [--no-fix-freq] [--debug]\n"
                      << "       Platform default: auto. Use --no-fix-freq to skip NPU/CPU/GPU/DDR frequency fix (requires root).\n";
            return 0;
        }
    }

    if (platform == "auto") {
        platform = detect_rockchip_platform();
        if (platform.empty()) {
            platform = "rk3588";
            std::cerr << "Platform auto-detect failed (not Rockchip or /proc/device-tree missing). Using default: rk3588\n";
        } else {
            std::cout << "Detected platform: " << platform << "\n";
        }
    }

    if (fix_freq) {
        std::cout << "Applying fix_freq for " << platform << " (NPU/CPU/GPU/DDR to max freq)...\n";
        rkllm_openai::apply_fix_freq(platform, debug);
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required.\n";
        return 1;
    }

    rkllm_openai::RKLLMBackend backend;
    if (!backend.init(model_path, platform, max_context_len, max_new_tokens, 0.8f, 0.9f, prompt_cache_path)) {
        std::cerr << "Error: RKLLM init failed.\n";
        return 1;
    }
    std::cout << "RKLLM init success.\n";

    rkllm_openai::ChatHandler chat_handler(&backend, debug);

    httplib::Server svr;

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "application/json");
        res.set_content(
            "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id + "\",\"object\":\"model\"}]}",
            "application/json");
    });

    svr.Post("/v1/chat/completions", [&chat_handler](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Content-Type", "application/json");
        res.set_header("Access-Control-Allow-Origin", "*");

        bool stream = false;
        if (req.has_param("stream"))
            stream = (req.get_param_value("stream") == "true" || req.get_param_value("stream") == "1");
        if (req.body.find("\"stream\":true") != std::string::npos)
            stream = true;

        std::string streamed;
        auto stream_write = [&streamed](const std::string& chunk) {
            streamed += chunk;
        };

        std::string out = chat_handler.handle_chat_completions(req.body, stream, stream_write);

        if (stream && streamed.size() > 0) {
            res.set_content(streamed, "application/x-ndjson");
        } else if (!out.empty()) {
            if (out.find("\"error\"") != std::string::npos && out.find("\"status\":503") != std::string::npos)
                res.status = 503;
            res.set_content(out, "application/json");
        } else {
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"Internal error\",\"type\":\"server_error\"}}", "application/json");
        }
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404)
            res.set_content("{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}", "application/json");
    });

    std::cout << "Listening on " << host << ":" << port << "\n";
    std::cout << "  GET  /v1/models\n";
    std::cout << "  POST /v1/chat/completions\n";

    if (!svr.bind_to_port(host.c_str(), port)) {
        std::cerr << "Error: bind failed.\n";
        return 1;
    }
    svr.listen_after_bind();

    return 0;
}
