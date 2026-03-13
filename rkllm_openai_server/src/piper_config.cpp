#include "piper_config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

namespace rkllm_openai {

using json = nlohmann::json;

static void load_phoneme_id_map(const json& j, std::unordered_map<std::string, int64_t>& out) {
    out.clear();
    if (!j.is_object()) return;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string key = it.key();
        const auto& v = it.value();
        int64_t id = -1;
        if (v.is_array() && !v.empty() && v[0].is_number_integer())
            id = v[0].get<int64_t>();
        else if (v.is_number_integer())
            id = v.get<int64_t>();
        if (id >= 0)
            out[key] = id;
    }
}

static void load_speaker_id_map(const json& j, std::unordered_map<std::string, int64_t>& out) {
    out.clear();
    if (!j.is_object()) return;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string key = it.key();
        const auto& v = it.value();
        if (v.is_number_integer())
            out[key] = v.get<int64_t>();
    }
}

bool PiperConfig::load_from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    return load_from_string(ss.str());
}

bool PiperConfig::load_from_string(const std::string& json_str) {
    try {
        json j = json::parse(json_str);

        if (j.contains("audio") && j["audio"].is_object()) {
            const auto& a = j["audio"];
            if (a.contains("sample_rate") && a["sample_rate"].is_number_integer())
                sample_rate = a["sample_rate"].get<int>();
            if (a.contains("hop_length") && a["hop_length"].is_number_integer())
                hop_length = a["hop_length"].get<int>();
        }
        if (j.contains("inference") && j["inference"].is_object()) {
            const auto& inf = j["inference"];
            if (inf.contains("noise_scale") && inf["noise_scale"].is_number())
                noise_scale = inf["noise_scale"].get<float>();
            if (inf.contains("length_scale") && inf["length_scale"].is_number())
                length_scale = inf["length_scale"].get<float>();
            if (inf.contains("noise_w") && inf["noise_w"].is_number())
                noise_w_scale = inf["noise_w"].get<float>();
            else if (inf.contains("noise_w_scale") && inf["noise_w_scale"].is_number())
                noise_w_scale = inf["noise_w_scale"].get<float>();
        }
        if (j.contains("num_speakers") && j["num_speakers"].is_number_integer())
            num_speakers = j["num_speakers"].get<int>();
        if (j.contains("espeak") && j["espeak"].is_object() && j["espeak"].contains("voice") && j["espeak"]["voice"].is_string())
            espeak_voice = j["espeak"]["voice"].get<std::string>();
        if (j.contains("phoneme_id_map") && j["phoneme_id_map"].is_object())
            load_phoneme_id_map(j["phoneme_id_map"], phoneme_id_map);
        if (j.contains("speaker_id_map") && j["speaker_id_map"].is_object())
            load_speaker_id_map(j["speaker_id_map"], speaker_id_map);

        return true;
    } catch (...) {
        return false;
    }
}

int64_t PiperConfig::get_phoneme_id(const std::string& symbol) const {
    auto it = phoneme_id_map.find(symbol);
    if (it != phoneme_id_map.end()) return it->second;
    return -1;
}

int64_t PiperConfig::get_speaker_id(const std::string& name) const {
    auto it = speaker_id_map.find(name);
    if (it != speaker_id_map.end()) return it->second;
    return -1;
}

void find_piper_model_files(const std::string& dir,
                            std::string& out_onnx,
                            std::string& out_rknn,
                            std::string& out_json) {
    out_onnx.clear();
    out_rknn.clear();
    out_json.clear();
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        size_t dot = name.rfind('.');
        std::string ext = (dot != std::string::npos) ? name.substr(dot) : "";
        if (ext == ".onnx") out_onnx = path;
        else if (ext == ".rknn") out_rknn = path;
        else if (ext == ".json") out_json = path;
    }
    closedir(d);
}

}  // namespace rkllm_openai
