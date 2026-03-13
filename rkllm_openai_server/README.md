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
./rkllm_openai_server --model_path /path/to/model.rkllm --platform rk3588 [--host 0.0.0.0] [--port 8080]
```

Options:

- `--model_path` (required): Absolute path to the converted RKLLM model on the device.
- `--platform`: One of `rk3588`, `rk3576`, `rv1126b`, `rk3562`. Default: `rk3588`.
- `--host`: Bind address. Default: `0.0.0.0`.
- `--port`: Port. Default: `8080`.

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
- `messages`: Array of `{ "role": "user"|"system"|"assistant"|"tool", "content": "..." }`. The last `user` or `tool` message is used as the prompt; `system` is used as system prompt (e.g. for function tools).
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
