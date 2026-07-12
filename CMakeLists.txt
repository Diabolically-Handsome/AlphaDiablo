cmake_minimum_required(VERSION 3.22)

# Configure the deployment target before any engine/dependency targets exist.
# A clean local build defaults to the actual host OS instead of advertising an
# untested historical minimum.  A caller may deliberately request an older
# target, but build.sh will then audit every linked non-system dylib and reject
# dependencies that require a newer OS.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  execute_process(
    COMMAND /usr/bin/sw_vers -productVersion
    OUTPUT_VARIABLE _ALPHADIABLO_HOST_MACOS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ALPHADIABLO_SW_VERS_RESULT)
  if(NOT _ALPHADIABLO_SW_VERS_RESULT EQUAL 0
      OR NOT _ALPHADIABLO_HOST_MACOS MATCHES "^[0-9]+\\.[0-9]+(\\.[0-9]+)?$")
    message(FATAL_ERROR
      "无法确定宿主 macOS 版本；请显式传 -DCMAKE_OSX_DEPLOYMENT_TARGET=<version>")
  endif()
  set(CMAKE_OSX_DEPLOYMENT_TARGET "${_ALPHADIABLO_HOST_MACOS}" CACHE STRING
      "Minimum macOS version for the bridge and embedded engine" FORCE)
  set(ALPHADIABLO_DEPLOYMENT_SOURCE "host-default" CACHE STRING
      "Origin of CMAKE_OSX_DEPLOYMENT_TARGET" FORCE)
endif()

project(diablogym CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 上游引擎源码位置(铁律 #3:引擎住临时目录,由 bootstrap.sh 提供,不进项目文件夹)
if(NOT DEFINED DEVILUTIONX_SRC)
  if(DEFINED ENV{TMPDIR} AND NOT "$ENV{TMPDIR}" STREQUAL "")
    set(_ALPHADIABLO_TMP "$ENV{TMPDIR}")
  else()
    set(_ALPHADIABLO_TMP "/tmp")
  endif()
  set(DEVILUTIONX_SRC "${_ALPHADIABLO_TMP}/alphadiablo-dev/devilutionX")
endif()
if(NOT EXISTS "${DEVILUTIONX_SRC}/Source/diablo.cpp")
  message(FATAL_ERROR "找不到 DevilutionX 源码: ${DEVILUTIONX_SRC}\n先运行 ../bootstrap.sh 或传 -DDEVILUTIONX_SRC=...")
endif()

# BUILD_TESTING=ON 是为了拿到测试基建里的共享库目标 libdevilutionx_so
# (整个引擎打包成一个 .dylib,全局变量单实例 —— pybind11 嵌入的正确姿势)
set(BUILD_TESTING ON CACHE BOOL "" FORCE)
add_subdirectory("${DEVILUTIONX_SRC}" engine EXCLUDE_FROM_ALL)

# DevilutionX intentionally lowers CMAKE_OSX_DEPLOYMENT_TARGET to 10.13 near
# the end of its subdirectory (for its standalone compatibility policy).  The
# test-only shared library used by this embedder is created in that scope, so
# without an explicit override it can still advertise an older minimum than
# both its object files and Homebrew dependencies.  Pin the two upstream final
# binaries built by build.sh back to this project's audited target.
if(APPLE)
  foreach(_ALPHADIABLO_ENGINE_TARGET IN ITEMS libdevilutionx_so devilutionx)
    if(TARGET ${_ALPHADIABLO_ENGINE_TARGET})
      target_compile_options(${_ALPHADIABLO_ENGINE_TARGET} PRIVATE
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
      target_link_options(${_ALPHADIABLO_ENGINE_TARGET} PRIVATE
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
  endforeach()
endif()

# pybind11 3 still falls back to the removed/deprecated FindPythonInterp path
# unless modern FindPython mode is selected explicitly.  That legacy path uses
# the unrelated PYTHON_EXECUTABLE cache key and can silently retain headers and
# an extension suffix from an old interpreter even when build.sh passes a new
# venv via Python_EXECUTABLE.
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
# Remove legacy pybind11/FindPythonLibsNew cache entries.  They are not inputs
# to modern FindPython and retaining them makes a switched venv look as if its
# stale interpreter and headers were still authoritative.
foreach(_ALPHADIABLO_LEGACY_PYTHON_VAR IN ITEMS
        PYTHON_EXECUTABLE PYTHON_INCLUDE_DIR PYTHON_LIBRARY)
  unset(${_ALPHADIABLO_LEGACY_PYTHON_VAR})
  unset(${_ALPHADIABLO_LEGACY_PYTHON_VAR} CACHE)
endforeach()
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(ALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE "" CACHE FILEPATH
    "Interpreter build.sh requires FindPython to use")
set(ALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX "" CACHE STRING
    "sysconfig EXT_SUFFIX required for the bridge")
set(ALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR "" CACHE PATH
    "sysconfig include directory required for the bridge")

if(ALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE)
  cmake_path(ABSOLUTE_PATH ALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE
             NORMALIZE OUTPUT_VARIABLE _ALPHADIABLO_EXPECTED_PYTHON)
  cmake_path(ABSOLUTE_PATH Python_EXECUTABLE
             NORMALIZE OUTPUT_VARIABLE _ALPHADIABLO_FOUND_PYTHON)
  if(NOT _ALPHADIABLO_FOUND_PYTHON STREQUAL _ALPHADIABLO_EXPECTED_PYTHON)
    message(FATAL_ERROR
      "FindPython 解释器漂移: found=${_ALPHADIABLO_FOUND_PYTHON}, "
      "expected=${_ALPHADIABLO_EXPECTED_PYTHON}")
  endif()
endif()

if(ALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR)
  cmake_path(ABSOLUTE_PATH ALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR
             NORMALIZE OUTPUT_VARIABLE _ALPHADIABLO_EXPECTED_INCLUDE)
  set(_ALPHADIABLO_NORMALIZED_INCLUDES)
  foreach(_ALPHADIABLO_INCLUDE IN LISTS Python_INCLUDE_DIRS)
    cmake_path(ABSOLUTE_PATH _ALPHADIABLO_INCLUDE
               NORMALIZE OUTPUT_VARIABLE _ALPHADIABLO_NORMALIZED_INCLUDE)
    list(APPEND _ALPHADIABLO_NORMALIZED_INCLUDES
         "${_ALPHADIABLO_NORMALIZED_INCLUDE}")
  endforeach()
  list(FIND _ALPHADIABLO_NORMALIZED_INCLUDES
       "${_ALPHADIABLO_EXPECTED_INCLUDE}" _ALPHADIABLO_INCLUDE_INDEX)
  if(_ALPHADIABLO_INCLUDE_INDEX EQUAL -1)
    message(FATAL_ERROR
      "FindPython headers 漂移: found=${_ALPHADIABLO_NORMALIZED_INCLUDES}, "
      "expected=${_ALPHADIABLO_EXPECTED_INCLUDE}")
  endif()
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
  OUTPUT_VARIABLE _ALPHADIABLO_FOUND_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _ALPHADIABLO_SUFFIX_RESULT)
if(NOT _ALPHADIABLO_SUFFIX_RESULT EQUAL 0 OR NOT _ALPHADIABLO_FOUND_EXT_SUFFIX)
  message(FATAL_ERROR
    "无法从 FindPython 解释器读取 EXT_SUFFIX: ${Python_EXECUTABLE}")
endif()
if(ALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX
    AND NOT _ALPHADIABLO_FOUND_EXT_SUFFIX
            STREQUAL ALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX)
  message(FATAL_ERROR
    "Python EXT_SUFFIX 漂移: found=${_ALPHADIABLO_FOUND_EXT_SUFFIX}, "
    "expected=${ALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX}")
endif()

pybind11_add_module(_diablogym src/diablogym.cpp)
# Make the target filename an explicit property of the selected interpreter;
# build.sh also checks this exact file after linking, so a stale wrong-ABI cache
# cannot be mistaken for a successful build.
set_target_properties(_diablogym PROPERTIES
  SUFFIX "${_ALPHADIABLO_FOUND_EXT_SUFFIX}")
target_link_libraries(_diablogym PRIVATE libdevilutionx_so)
target_include_directories(_diablogym PRIVATE "${DEVILUTIONX_SRC}/Source")

file(WRITE "${CMAKE_BINARY_DIR}/alphadiablo-python-build.txt"
  "executable=${Python_EXECUTABLE}\n"
  "include_dirs=${Python_INCLUDE_DIRS}\n"
  "ext_suffix=${_ALPHADIABLO_FOUND_EXT_SUFFIX}\n"
  "deployment_target=${CMAKE_OSX_DEPLOYMENT_TARGET}\n"
  "deployment_source=${ALPHADIABLO_DEPLOYMENT_SOURCE}\n")
