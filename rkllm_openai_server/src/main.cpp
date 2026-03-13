#include "chat_handler.hpp"
#include "rkllm_backend.hpp"
#include "httplib.h"
#include <iostream>
#include <string>
#include <cstring>

static std::string model_id = "rkllm";

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string platform = "rk3588";
    std::string host = "0.0.0.0";
    int port = 8080;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) {
            platform = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " --model_path <path> [--platform rk3588|rk3576] [--host 0.0.0.0] [--port 8080] [--debug]\n";
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required.\n";
        return 1;
    }

    rkllm_openai::RKLLMBackend backend;
    if (!backend.init(model_path, platform)) {
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
