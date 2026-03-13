cmake_minimum_required(VERSION 3.14)
project(rkllm_openai_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target lib arch for RKLLM (same as rkllm_api_demo)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(TARGET_LIB_ARCH aarch64)
    else()
        set(TARGET_LIB_ARCH armhf)
    endif()
    if(CMAKE_C_COMPILER MATCHES "uclibc")
        set(TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
    endif()
endif()

# Vendored header-only deps: 3rdparty/nlohmann (json), 3rdparty/httplib.h
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
if(NOT EXISTS "${THIRDPARTY_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "3rdparty/nlohmann/json.hpp not found. Ensure 3rdparty is populated.")
endif()

# RKLLM runtime (path relative to project root: rkllm_openai_server/../rkllm-runtime)
set(RKLLM_API_PATH "${CMAKE_SOURCE_DIR}/../rkllm-runtime/${CMAKE_SYSTEM_NAME}/librkllm_api")
if(NOT EXISTS "${RKLLM_API_PATH}/include/rkllm.h")
    message(FATAL_ERROR "RKLLM API not found at ${RKLLM_API_PATH}. Build on Linux or cross-compile for Linux (e.g. -DCMAKE_SYSTEM_NAME=Linux). Ensure rkllm-runtime is present.")
endif()
include_directories(
    ${RKLLM_API_PATH}/include
    ${THIRDPARTY_DIR}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librkllmrt.so)
    find_package(OpenMP REQUIRED)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/${TARGET_LIB_ARCH}/librkllmrt.so)
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
endif()

add_executable(rkllm_openai_server
    src/main.cpp
    src/rkllm_backend.cpp
    src/chat_handler.cpp
    src/fix_freq.cpp
)

target_link_libraries(rkllm_openai_server PRIVATE ${RKLLM_RT_LIB})
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(rkllm_openai_server PRIVATE log OpenMP::OpenMP_CXX)
endif()

# Install
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/openai_server_${CMAKE_SYSTEM_NAME}_${TARGET_LIB_ARCH})
install(TARGETS rkllm_openai_server DESTINATION .)
install(PROGRAMS ${RKLLM_RT_LIB} DESTINATION .)
