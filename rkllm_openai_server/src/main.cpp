#include "chat_handler.hpp"
#include "rkllm_backend.hpp"
#include "fix_freq.hpp"
#include "tts_handler.hpp"
#if defined(RKLLM_OPENAI_ENABLE_TTS) && RKLLM_OPENAI_ENABLE_TTS
#include "piper_tts.hpp"
#endif
#include "httplib.h"
#include <nlohmann/json.hpp>
#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL
#include "image_encoder.hpp"
#endif
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <sstream>

static std::string model_id = "rkllm";

/** Split a command string into argv (by spaces). */
static std::vector<std::string> split_runner_cmd(const std::string& cmd) {
    std::vector<std::string> out;
    std::istringstream iss(cmd);
    std::string s;
    while (iss >> s)
        out.push_back(std::move(s));
    return out;
}

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
    std::string encoder_model_path;
    std::string img_start, img_end, img_content;
    int encoder_core_num = 1;
    std::string tts_model_path;
    std::string tts_runner = "python3 -m rkllama.scripts.piper_tts_cli";

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
#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL
        } else if (strcmp(argv[i], "--encoder_model_path") == 0 && i + 1 < argc) {
            encoder_model_path = argv[++i];
        } else if (strcmp(argv[i], "--img_start") == 0 && i + 1 < argc) {
            img_start = argv[++i];
        } else if (strcmp(argv[i], "--img_end") == 0 && i + 1 < argc) {
            img_end = argv[++i];
        } else if (strcmp(argv[i], "--img_content") == 0 && i + 1 < argc) {
            img_content = argv[++i];
        } else if (strcmp(argv[i], "--encoder_core_num") == 0 && i + 1 < argc) {
            encoder_core_num = std::stoi(argv[++i]);
#endif
        } else if (strcmp(argv[i], "--tts_model_path") == 0 && i + 1 < argc) {
            tts_model_path = argv[++i];
        } else if (strcmp(argv[i], "--tts_runner") == 0 && i + 1 < argc) {
            tts_runner = argv[++i];
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " --model_path <path> [--platform auto|rk3588|rk3576|rv1126b|rk3562] [--host 0.0.0.0] [--port 8080]\n"
                      << "       [--max_context_len 4096] [--max_new_tokens 4096] [--prompt_cache <path>]\n"
#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL
                      << "       [--encoder_model_path <path>] [--img_start <s>] [--img_end <s>] [--img_content <s>] [--encoder_core_num 1]\n"
#endif
                      << "       [--tts_model_path <path>] [--tts_runner <cmd>] [--no-fix-freq] [--debug]\n"
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

#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL
    if (!encoder_model_path.empty() && img_start.empty()) {
        img_start = "<|vision_start|>";
        img_end = "<|vision_end|>";
        img_content = "<|image_pad|>";
    }
#endif

    rkllm_openai::RKLLMBackend backend;
    if (!backend.init(model_path, platform, max_context_len, max_new_tokens, 0.8f, 0.9f, prompt_cache_path, img_start, img_end, img_content)) {
        std::cerr << "Error: RKLLM init failed.\n";
        return 1;
    }
    std::cout << "RKLLM init success.\n";

    rkllm_openai::EncodeImageFn encode_image;
#if defined(RKLLM_OPENAI_ENABLE_MULTIMODAL) && RKLLM_OPENAI_ENABLE_MULTIMODAL
    std::unique_ptr<rkllm_openai::ImageEncoder> image_encoder;
    if (!encoder_model_path.empty()) {
        image_encoder = std::make_unique<rkllm_openai::ImageEncoder>();
        if (!image_encoder->init(encoder_model_path, encoder_core_num)) {
            std::cerr << "Error: Image encoder init failed (--encoder_model_path " << encoder_model_path << ").\n";
            return 1;
        }
        std::cout << "Image encoder init success (multimodal enabled).\n";
        encode_image = [&image_encoder](const std::vector<uint8_t>& bytes) { return image_encoder->encode(bytes); };
    }
#endif

    rkllm_openai::ChatHandler chat_handler(&backend, debug, encode_image);

#if defined(RKLLM_OPENAI_ENABLE_TTS) && RKLLM_OPENAI_ENABLE_TTS
    std::unique_ptr<rkllm_openai::PiperTts> piper_tts;
    rkllm_openai::PiperTts* piper_tts_ptr = nullptr;
    if (!tts_model_path.empty()) {
        piper_tts = std::make_unique<rkllm_openai::PiperTts>();
        if (piper_tts->init(tts_model_path)) {
            piper_tts_ptr = piper_tts.get();
            std::cout << "Piper TTS (native C++) enabled: " << tts_model_path << "\n";
        } else {
            std::cerr << "Warning: Native Piper TTS init failed for " << tts_model_path << "; use --tts_runner for Python fallback.\n";
        }
    }
#endif
    std::vector<std::string> tts_runner_argv = split_runner_cmd(tts_runner);
    rkllm_openai::TtsHandler tts_handler(tts_model_path, std::move(tts_runner_argv)
#if defined(RKLLM_OPENAI_ENABLE_TTS) && RKLLM_OPENAI_ENABLE_TTS
                                         , piper_tts_ptr
#endif
    );

    httplib::Server svr;
    /* Allow long write timeout for streaming (token-by-token) responses. */
    svr.set_write_timeout(300, 0);

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "application/json");
        res.set_content(
            "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_id + "\",\"object\":\"model\"}]}",
            "application/json");
    });

    svr.Post("/v1/chat/completions", [&chat_handler](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        std::string out = chat_handler.handle_chat_completions(req.body, &res);
        if (!out.empty()) {
            res.set_header("Content-Type", "application/json");
            if (out.find("\"error\"") != std::string::npos && out.find("\"status\":503") != std::string::npos)
                res.status = 503;
            res.set_content(out, "application/json");
        }
        /* else: streaming was set up (Content-Type and body via chunked provider) */
    });

    svr.Post("/v1/audio/speech", [&tts_handler](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        auto [audio, content_type] = tts_handler.handle_speech(req.body);
        if (content_type.empty()) {
            res.set_header("Content-Type", "application/json");
            try {
                auto err = nlohmann::json::parse(std::string(audio.begin(), audio.end()));
                if (err.contains("error") && err["error"].contains("type") && err["error"]["type"] == "server_error")
                    res.status = 500;
                else
                    res.status = 400;
            } catch (...) {
                res.status = 500;
            }
            res.set_content(std::string(audio.begin(), audio.end()), "application/json");
        } else {
            res.set_header("Content-Type", content_type);
            res.set_content(std::string(audio.begin(), audio.end()), content_type);
        }
    });

    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 404)
            res.set_content("{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}", "application/json");
    });

    std::cout << "Listening on " << host << ":" << port << "\n";
    std::cout << "  GET  /v1/models\n";
    std::cout << "  POST /v1/chat/completions\n";
    if (tts_handler.enabled())
        std::cout << "  POST /v1/audio/speech (Piper TTS)\n";

    if (!svr.bind_to_port(host.c_str(), port)) {
        std::cerr << "Error: bind failed.\n";
        return 1;
    }
    svr.listen_after_bind();

    return 0;
}
