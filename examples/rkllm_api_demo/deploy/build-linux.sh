#!/bin/bash
# Debug / Release / RelWithDebInfo
# On aarch64 Linux (e.g. Rock 5T), uses system gcc/g++ for native build.
# On x86_64, uses cross-compiler if found; set C_COMPILER/CXX_COMPILER to override.
set -e
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

TARGET_ARCH=aarch64
TARGET_PLATFORM=linux
if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi

# Native build on aarch64: use system compiler
HOST_ARCH=$(uname -m 2>/dev/null || true)
if [[ "${HOST_ARCH}" == "aarch64" ]] || [[ "${HOST_ARCH}" == "arm64" ]]; then
    C_COMPILER=${C_COMPILER:-gcc}
    CXX_COMPILER=${CXX_COMPILER:-g++}
else
    # Cross-compile from x86_64: use toolchain if not set
    GCC_COMPILER_PATH=${GCC_COMPILER_PATH:-$HOME/opts/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu}
    C_COMPILER=${C_COMPILER:-${GCC_COMPILER_PATH}-gcc}
    CXX_COMPILER=${CXX_COMPILER:-${GCC_COMPILER_PATH}-g++}
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j4
make install