cmake_minimum_required(VERSION 3.22)
project(diablogym CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 上游引擎源码位置(铁律 #3:引擎住临时目录,由 bootstrap.sh 提供,不进项目文件夹)
if(NOT DEFINED DEVILUTIONX_SRC)
  set(DEVILUTIONX_SRC "$ENV{TMPDIR}/alphadiablo-dev/devilutionX")
endif()
if(NOT EXISTS "${DEVILUTIONX_SRC}/Source/diablo.cpp")
  message(FATAL_ERROR "找不到 DevilutionX 源码: ${DEVILUTIONX_SRC}\n先运行 ../bootstrap.sh 或传 -DDEVILUTIONX_SRC=...")
endif()

# BUILD_TESTING=ON 是为了拿到测试基建里的共享库目标 libdevilutionx_so
# (整个引擎打包成一个 .dylib,全局变量单实例 —— pybind11 嵌入的正确姿势)
set(BUILD_TESTING ON CACHE BOOL "" FORCE)
add_subdirectory("${DEVILUTIONX_SRC}" engine EXCLUDE_FROM_ALL)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_diablogym src/diablogym.cpp)
target_link_libraries(_diablogym PRIVATE libdevilutionx_so)
target_include_directories(_diablogym PRIVATE "${DEVILUTIONX_SRC}/Source")
