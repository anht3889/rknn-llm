#ifndef RKLLM_OPENAI_SERVER_BASE64_HPP
#define RKLLM_OPENAI_SERVER_BASE64_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace rkllm_openai {

/** Decode base64 string to bytes. Returns empty vector on invalid input. */
std::vector<uint8_t> base64_decode(const std::string& in);

}  // namespace rkllm_openai

#endif
