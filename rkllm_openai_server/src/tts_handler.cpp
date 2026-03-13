#include "tts_handler.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

namespace rkllm_openai {

using json = nlohmann::json;

TtsHandler::TtsHandler() : tts_model_path_(), runner_argv_() {}

TtsHandler::TtsHandler(std::string tts_model_path, std::vector<std::string> runner_argv)
    : tts_model_path_(std::move(tts_model_path)), runner_argv_(std::move(runner_argv)) {}

bool TtsHandler::enabled() const {
    return !tts_model_path_.empty() && !runner_argv_.empty();
}

std::pair<std::vector<uint8_t>, std::string> TtsHandler::handle_speech(const std::string& body) const {
    if (!enabled()) {
        json err;
        err["error"] = {{"message", "TTS not configured. Start server with --tts_model_path and --tts_runner."}, {"type", "invalid_request_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    json req;
    try {
        req = json::parse(body);
    } catch (const json::exception&) {
        json err;
        err["error"] = {{"message", "Invalid JSON"}, {"type", "invalid_request_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    std::string input;
    if (req.contains("input") && req["input"].is_string())
        input = req["input"].get<std::string>();
    if (input.empty()) {
        json err;
        err["error"] = {{"message", "Missing or empty 'input' (text to speak)"}, {"type", "invalid_request_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    std::string voice;
    if (req.contains("voice") && req["voice"].is_string())
        voice = req["voice"].get<std::string>();

    std::string response_format = "wav";
    if (req.contains("response_format")) {
        if (req["response_format"].is_string())
            response_format = req["response_format"].get<std::string>();
        else if (req["response_format"].is_object() && req["response_format"].contains("type") && req["response_format"]["type"].is_string())
            response_format = req["response_format"]["type"].get<std::string>();
    }

    double speed = 0.0;
    if (req.contains("speed") && req["speed"].is_number())
        speed = req["speed"].get<double>();

    json runner_req;
    runner_req["model_path"] = tts_model_path_;
    runner_req["input"] = input;
    if (!voice.empty()) runner_req["voice"] = voice;
    runner_req["response_format"] = response_format;
    runner_req["stream_format"] = "audio";
    if (speed > 0.0) runner_req["speed"] = speed;

    std::string runner_req_str = runner_req.dump();

    int stdin_pipe[2];
    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        json err;
        err["error"] = {{"message", "Failed to create pipes for TTS runner"}, {"type", "server_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        json err;
        err["error"] = {{"message", "Failed to fork TTS runner"}, {"type", "server_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    if (pid == 0) {
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        std::vector<char*> argv_ptrs;
        for (auto& s : runner_argv_)
            argv_ptrs.push_back(const_cast<char*>(s.c_str()));
        argv_ptrs.push_back(nullptr);

        execvp(runner_argv_[0].c_str(), argv_ptrs.data());
        _exit(127);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    size_t written = 0;
    while (written < runner_req_str.size()) {
        ssize_t n = write(stdin_pipe[1], runner_req_str.data() + written, runner_req_str.size() - written);
        if (n <= 0) break;
        written += static_cast<size_t>(n);
    }
    close(stdin_pipe[1]);

    std::string content_type;
    std::vector<uint8_t> audio;
    {
        std::string first_line;
        char c;
        while (read(stdout_pipe[0], &c, 1) == 1 && c != '\n')
            first_line += c;
        const std::string prefix = "Content-Type: ";
        if (first_line.size() >= prefix.size() && first_line.compare(0, prefix.size(), prefix) == 0)
            content_type = first_line.substr(prefix.size());
        while (true) {
            std::array<uint8_t, 4096> buf;
            ssize_t n = read(stdout_pipe[0], buf.data(), buf.size());
            if (n <= 0) break;
            audio.insert(audio.end(), buf.data(), buf.data() + static_cast<size_t>(n));
        }
    }
    close(stdout_pipe[0]);

    std::string stderr_out;
    {
        std::array<char, 256> buf;
        ssize_t n;
        while ((n = read(stderr_pipe[0], buf.data(), buf.size())) > 0)
            stderr_out.append(buf.data(), static_cast<size_t>(n));
    }
    close(stderr_pipe[0]);

    int wstatus = 0;
    waitpid(pid, &wstatus, 0);

    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
        if (!stderr_out.empty()) {
            std::vector<uint8_t> err_bytes(stderr_out.begin(), stderr_out.end());
            return {err_bytes, ""};
        }
        json err;
        err["error"] = {{"message", "TTS runner failed"}, {"type", "server_error"}};
        return {std::vector<uint8_t>(err.dump().begin(), err.dump().end()), ""};
    }

    return {audio, content_type};
}

}  // namespace rkllm_openai
