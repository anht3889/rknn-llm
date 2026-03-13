# RKLLM OpenAI-like inference API server (C++)

A C++ HTTP server that exposes an OpenAI-compatible chat completions API using the RKLLM runtime. It mirrors the behavior of the Python Flask server in `examples/rkllm_server_demo` with a single model and optional streaming.

## Requirements

- CMake 3.14+
- C++17 compiler
- RKLLM runtime: `rkllm-runtime/Linux/librkllm_api` (include and `aarch64/librkllmrt.so` for Linux aarch64)
- Build is intended for **Linux** (native or cross-compiled for aarch64). The RKLLM library is not available for macOS/Windows.

## Build

Compilation has been verified; linking requires the RKLLM shared library for your target (e.g. `librkllmrt.so` for Linux aarch64). On a host with a different OS/arch (e.g. macOS), link will fail with "unknown file type" until you cross-compile or build on the target device.

From the repo root or from this directory:

```bash
cd rkllm_openai_server
mkdir build && cd build
cmake -DCMAKE_SYSTEM_NAME=Linux ..
make
```

For cross-compilation to aarch64 (e.g. from x86_64):

```bash
cmake -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
      ..
make
```

Ensure `librkllmrt.so` is present at `rkllm-runtime/Linux/librkllm_api/aarch64/` (or the appropriate arch for your target). Copy it next to the binary or set `LD_LIBRARY_PATH` when running.

## Run

```bash
./rkllm_openai_server --model_path /path/to/model.rkllm [--platform auto] [--host 0.0.0.0] [--port 8080]
```

On a Rockchip board (e.g. RK3588), platform is auto-detected by default; you can omit `--platform` or pass `--platform rk3588` explicitly. Before loading the model, the server applies **fix_freq** for the detected platform (same behavior as the Python Flask server running `sudo bash fix_freq_<platform>.sh`). Run with `sudo` if you want fix_freq to succeed; otherwise use `--no-fix-freq` to skip it.

Options:

- `--model_path` (required): Absolute path to the converted RKLLM model on the device.
- `--platform`: One of `auto`, `rk3588`, `rk3576`, `rv1126b`, `rk3562`. Default: `auto`. When `auto`, the platform is detected from `/proc/device-tree/compatible` (e.g. on a Rockchip RK3588 board you can omit `--platform`).
- `--host`: Bind address. Default: `0.0.0.0`.
- `--port`: Port. Default: `8080`.
- `--max_context_len`: Maximum context length (tokens). Default: `4096`. Lower values reduce KV cache memory and can improve prefill/generate speed on constrained devices.
- `--max_new_tokens`: Maximum new tokens per response. Default: `4096`. Lower if you don't need long replies.
- `--prompt_cache`: Path to a pre-built prompt cache file. If set, the runtime loads it after init to skip re-prefill for the cached prefix (faster first token when using the same system/template).
- `--no-fix-freq`: Skip applying fix_freq (do not lock NPU/CPU/GPU/DDR to max frequency). By default, the server applies the same fix as `scripts/fix_freq_<platform>.sh` using the detected or given platform; this requires root (e.g. run with `sudo`).
- `--debug`: Log prefill/generate token counts and speeds (tokens/s) to stderr for each request.

When built with **multimodal** support (`-DENABLE_MULTIMODAL=ON`, see below):

- `--encoder_model_path`: Path to the RKNN vision encoder model (e.g. from `examples/multimodal_model_demo`). Enables image input in chat completions.
- `--img_start`, `--img_end`, `--img_content`: Vision prompt tokens for the LLM (default `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>` when encoder is set).
- `--encoder_core_num`: NPU core count for the encoder (default `1`).

## Multimodal (vision) build

To support image inputs (like `examples/multimodal_model_demo`), build with OpenCV, RKNN runtime, and the demo’s image encoder:

```bash
cd rkllm_openai_server/build
cmake -DCMAKE_SYSTEM_NAME=Linux -DENABLE_MULTIMODAL=ON ..
make
```

Requirements for `ENABLE_MULTIMODAL=ON`:

- The repo must contain `examples/multimodal_model_demo/deploy` (with `src/image_enc.cc` and `3rdparty/librknnrt`, `3rdparty/opencv`).
- Run on the board with the encoder RKNN model and a vision-capable RKLLM model (e.g. Qwen2-VL, InternVL).

Run with an encoder and vision model:

```bash
./rkllm_openai_server --model_path /path/to/vision_llm.rkllm --encoder_model_path /path/to/encoder.rknn
```

Then send a message with image content (see API below).

## API

### GET /v1/models

Returns a fixed model list for compatibility with OpenAI-style clients.

Example:

```bash
curl http://localhost:8080/v1/models
```

### POST /v1/chat/completions

Request body (JSON):

- `model` (optional): Echoed in the response; no effect on inference.
- `messages`: Array of `{ "role": "user"|"system"|"assistant"|"tool", "content": "..." }`. The last `user` or `tool` message is used as the prompt; `system` is used as system prompt (e.g. for function tools). For **multimodal** (when the server is started with `--encoder_model_path`), `content` may be an array of parts: `{ "type": "text", "text": "..." }` and `{ "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }`. Text and image parts are combined; the image is encoded by the vision encoder and sent to the LLM with the text (one image per request).
- `stream` (optional): `true` for newline-delimited JSON stream; `false` or omit for a single JSON response.
- `enable_thinking` (optional): Enable thinking mode (e.g. Qwen3).
- `tools` (optional): JSON array of tool definitions for function calling (same format as OpenAI tools).

Response (non-stream): OpenAI-style `chat.completion` with `choices[].message` and `usage`.

Response (stream): Newline-delimited JSON lines, each a `chat.completion.chunk` with `choices[].delta.content` or final chunk with `finish_reason: "stop"` and `usage`.

Example (non-stream):

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rkllm","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

Example (stream):

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rkllm","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Python client

You can use the existing `examples/rkllm_server_demo/chat_api_flask.py` by changing the URL to `http://<board_ip>:8080/v1/chat/completions` and the request format to the same `model` / `messages` / `stream` JSON (it is already compatible).

## Concurrency

Only one inference runs at a time (same as the Flask demo). Concurrent requests to `/v1/chat/completions` will receive 503 when the server is busy.

## Memory and speed optimization

- **Context and generation length**: Use `--max_context_len` and `--max_new_tokens` to match your workload. Smaller values reduce KV cache allocation and often improve both prefill and decode speed on memory-limited boards (e.g. RK3588).
- **Prompt cache**: For repeated system prompts or fixed prefixes, build a prompt cache once (e.g. with the C++ demo or a one-off run that saves cache via RKLLM API) and pass it with `--prompt_cache <path>`. The server will load it at startup and skip re-prefill for the cached prefix, improving time-to-first-token.
- **Streaming**: The server uses a condition-variable wait (no fixed 50 ms polling) so streamed chunks are delivered as soon as the runtime produces them, improving perceived latency.
- **Runtime settings**: The backend already sets `embed_flash = 1` (embeddings from flash) and `keep_history = 0` (no cross-request KV reuse), which keeps memory use and behavior predictable. Further tuning (e.g. `n_batch`, CPU affinity) is available in the RKLLM API if you extend the server.

## Troubleshooting

**Segmentation fault at startup**

1. Get a backtrace to see where it crashes:
   ```bash
   gdb --args ./rkllm_openai_server --model_path /path/to/model.rkllm --platform rk3588
   run
   # after crash:
   bt
   quit
   ```
2. Ensure `librkllmrt.so` is the correct build for your board and is findable (`LD_LIBRARY_PATH` or same directory as the binary).
3. Confirm the model path is a valid `.rkllm` file and the process has read access.
4. Try running the C++ demo first to confirm the runtime works: `examples/rkllm_api_demo/deploy`.
