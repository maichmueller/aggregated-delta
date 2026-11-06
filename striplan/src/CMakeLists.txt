cmake_minimum_required(VERSION 3.24)

project(striplan_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STRIPLAN_BUILD_BENCHMARKS "Build C++ benchmarks." OFF)
option(STRIPLAN_GENERATE_STUBS "Generate nanobind stubs in dev mode." ON)
option(STRIPLAN_EXPORT_LOCAL_CONAN_RECIPES "Export local Conan recipes before dependency resolution." ON)
option(STRIPLAN_FREE_THREADED "Enable nanobind FREE_THREADED mode when available." ON)

if (STRIPLAN_EXPORT_LOCAL_CONAN_RECIPES AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_repo_root "${CMAKE_CURRENT_LIST_DIR}/..")
    if (DEFINED Python_EXECUTABLE AND EXISTS "${Python_EXECUTABLE}")
        set(_python_cmd "${Python_EXECUTABLE}")
    else ()
        find_program(_python_cmd NAMES python3 python)
    endif ()
    if (NOT _python_cmd)
        message(FATAL_ERROR "Could not locate Python to run conan_export.py")
    endif ()

    set(_conan_cmd "")
    if (DEFINED CONAN_COMMAND AND NOT CONAN_COMMAND STREQUAL "")
        set(_conan_cmd "${CONAN_COMMAND}")
    endif ()
    if (_conan_cmd STREQUAL "")
        set(_conan_cmd "$ENV{CONAN_COMMAND}")
    endif ()
    if (_conan_cmd STREQUAL "")
        set(_conan_cmd "$ENV{CONAN_CMD}")
    endif ()
    if (_conan_cmd STREQUAL "")
        find_program(_conan_cmd NAMES conan)
    endif ()
    if (NOT _conan_cmd)
        set(_conan_cmd "conan")
    endif ()

    execute_process(
        COMMAND "${_python_cmd}" "${_repo_root}/conan_export.py" "--conan_cmd" "${_conan_cmd}"
        WORKING_DIRECTORY "${_repo_root}"
        RESULT_VARIABLE _export_result
    )
    if (NOT _export_result EQUAL 0)
        message(FATAL_ERROR "Failed to export local Conan recipes with ${_conan_cmd}.")
    endif ()
endif ()

if (NOT DEFINED Python_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX})
    set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
endif ()
find_package(
    Python 3.10
    REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.Embed
)

find_package(nanobind CONFIG REQUIRED)
if (NOT COMMAND nanobind_build_library)
    if (DEFINED nanobind_BUILD_DIRS_RELEASE AND EXISTS "${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_RELEASE}/nanobind-config.cmake")
    elseif (DEFINED nanobind_BUILD_DIRS_DEBUG AND EXISTS "${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
        include("${nanobind_BUILD_DIRS_DEBUG}/nanobind-config.cmake")
    endif ()
endif ()

find_package(Boost CONFIG REQUIRED COMPONENTS iostreams)
find_package(fmt REQUIRED)
find_package(ZLIB REQUIRED)
find_package(loki REQUIRED)
find_package(nauty REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(phmap CONFIG REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(strong_type REQUIRED)
find_package(range-v3 REQUIRED)
find_package(dlpack REQUIRED)
find_package(valla CONFIG REQUIRED)
if (STRIPLAN_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    find_package(argparse REQUIRED)
endif ()

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ConanFallbacks.cmake")
nanobind_build_library(nanobind)

if (TARGET valla::valla AND NOT TARGET valla::core)
    add_library(valla::core ALIAS valla::valla)
endif ()
if (TARGET loki::loki AND NOT TARGET loki::parsers)
    add_library(loki::parsers ALIAS loki::loki)
endif ()

if (NOT NAUTY_INCLUDE_DIR)
    if (DEFINED nauty_INCLUDE_DIRS_RELEASE)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_RELEASE}")
    elseif (DEFINED nauty_INCLUDE_DIRS_DEBUG)
        set(NAUTY_INCLUDE_DIR "${nauty_INCLUDE_DIRS_DEBUG}")
    endif ()
endif ()
if (NOT NAUTY_LIBRARY)
    if (DEFINED nauty_LIB_DIRS_RELEASE)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_RELEASE}" NO_DEFAULT_PATH)
    elseif (DEFINED nauty_LIB_DIRS_DEBUG)
        find_library(NAUTY_LIBRARY NAMES nauty PATHS "${nauty_LIB_DIRS_DEBUG}" NO_DEFAULT_PATH)
    endif ()
endif ()

if (NOT DEFINED mimir_DIR)
    if (DEFINED ENV{MIMIR_CMAKE_DIR} AND NOT "$ENV{MIMIR_CMAKE_DIR}" STREQUAL "")
        set(mimir_DIR "$ENV{MIMIR_CMAKE_DIR}" CACHE PATH "Path to mimir CMake package")
    else ()
        execute_process(
            COMMAND "${Python_EXECUTABLE}" "-c" "import pymimir; print(pymimir.get_cmake_dir())"
            RESULT_VARIABLE _mimir_result
            OUTPUT_VARIABLE _mimir_dir
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (_mimir_result EQUAL 0 AND
            (EXISTS "${_mimir_dir}/mimir-config.cmake" OR EXISTS "${_mimir_dir}/mimirConfig.cmake"))
            set(mimir_DIR "${_mimir_dir}" CACHE PATH "Path to mimir CMake package")
        endif ()
    endif ()
endif ()

find_package(mimir REQUIRED)

set(STRIPLAN_RPATH_MODE "dev" CACHE STRING "RPATH mode: dev or wheel")
set_property(CACHE STRIPLAN_RPATH_MODE PROPERTY STRINGS dev wheel)

function(_append_rpath_from_target out_var target_name)
    if (NOT TARGET "${target_name}")
        return()
    endif ()
    foreach (prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG INTERFACE_LINK_DIRECTORIES)
        get_target_property(_loc "${target_name}" "${prop}")
        if (_loc)
            foreach (_entry IN LISTS _loc)
                if (_entry MATCHES "\\$<")
                    list(APPEND "${out_var}" "${_entry}")
                elseif (IS_DIRECTORY "${_entry}")
                    list(APPEND "${out_var}" "${_entry}")
                elseif (EXISTS "${_entry}")
                    get_filename_component(_dir "${_entry}" DIRECTORY)
                    list(APPEND "${out_var}" "${_dir}")
                endif ()
            endforeach ()
        endif ()
    endforeach ()
    set("${out_var}" "${${out_var}}" PARENT_SCOPE)
endfunction()

add_library(
    striplan_core
    STATIC
    "_core/striplan/core/flat_batch_encoding.cpp"
    "_core/striplan/core/iw_expansion.cpp"
    "_core/striplan/core/relational_rw_return.cpp"
    "_core/striplan/core/rw_return_propagator.cpp"
    "_core/striplan/core/transition_tree_encoding.cpp"
    "_core/striplan/core/extended_encoded_batch.cpp"
)
set_target_properties(striplan_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(striplan_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/_core")
target_link_libraries(
    striplan_core
    PUBLIC
    Boost::iostreams
    fmt::fmt
    ZLIB::ZLIB
    loki::loki
    nauty::nauty
    mimir::core
    absl::node_hash_set
    phmap
    unordered_dense::unordered_dense
    TBB::tbb
    nanobind
    rollbear::strong_type
    range-v3::range-v3
    dlpack::dlpack
)
target_compile_definitions(
    striplan_core
    PUBLIC
    BOOST_MPL_LIMIT_LIST_SIZE=30
    BOOST_MPL_LIMIT_VECTOR_SIZE=30
    BOOST_VARIANT_LIMIT_TYPES=30
    BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
)
target_compile_definitions(
    striplan_core
    PRIVATE
    NB_DOMAIN=pymimir_abi_domain
)
conan_add_package_include_fallback(striplan_core abseil)
conan_add_package_include_fallback(striplan_core onetbb)

if (STRIPLAN_FREE_THREADED)
    set(_free_threaded "FREE_THREADED")
endif ()

nanobind_add_module(
    striplan_native_module
    ${_free_threaded}
    STABLE_ABI
    NB_DOMAIN pymimir_abi_domain
    "_core/striplan/init_flat_batch_encoding.cpp"
    "_core/striplan/module.cpp"
    "_core/striplan/init_iw_expansion.cpp"
    "_core/striplan/init_relational_rw_return.cpp"
    "_core/striplan/init_transition_tree_encoding.cpp"
    "_core/striplan/init_extended_encoded_batch.cpp"
)
set_target_properties(
    striplan_native_module
    PROPERTIES
    OUTPUT_NAME "_iw_encoding_backend"
)
target_link_libraries(striplan_native_module PRIVATE striplan_core mimir::core)

set(_rpaths "")
if (APPLE)
    list(APPEND _rpaths "@loader_path")
elseif (UNIX)
    list(APPEND _rpaths "$ORIGIN")
endif ()
if (STRIPLAN_RPATH_MODE STREQUAL "wheel")
    if (APPLE)
        list(APPEND _rpaths "@loader_path/../pymimir/lib")
    elseif (UNIX)
        list(APPEND _rpaths "$ORIGIN/../pymimir/lib")
    endif ()
endif ()
if (STRIPLAN_RPATH_MODE STREQUAL "dev")
    _append_rpath_from_target(_rpaths mimir::core)
    _append_rpath_from_target(_rpaths nanobind)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c"
                "import pathlib, pymimir; print((pathlib.Path(pymimir.__file__).resolve().parent / 'lib').resolve())"
        RESULT_VARIABLE _pymimir_lib_result
        OUTPUT_VARIABLE _pymimir_lib_dir
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (_pymimir_lib_result EQUAL 0 AND IS_DIRECTORY "${_pymimir_lib_dir}")
        list(APPEND _rpaths "${_pymimir_lib_dir}")
    endif ()
endif ()
list(REMOVE_DUPLICATES _rpaths)

set_target_properties(
    striplan_native_module
    PROPERTIES
    BUILD_RPATH "${_rpaths}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${_rpaths}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)
if (STRIPLAN_RPATH_MODE STREQUAL "dev")
    set_target_properties(
        striplan_native_module
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/striplan"
    )
    if (TARGET nanobind)
        set_target_properties(
            nanobind
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/striplan"
        )
    endif ()
endif ()

install(TARGETS striplan_native_module LIBRARY DESTINATION "striplan")
if (TARGET nanobind)
    install(TARGETS nanobind LIBRARY DESTINATION "striplan")
endif ()

if (STRIPLAN_GENERATE_STUBS)
    set(_stub_output "striplan/_iw_encoding_backend.pyi")
    if (STRIPLAN_RPATH_MODE STREQUAL "dev")
        string(PREPEND _stub_output "${CMAKE_SOURCE_DIR}/src/")
        nanobind_add_stub(
            striplan_native_module_stubs
            MODULE striplan._iw_encoding_backend
            OUTPUT "${_stub_output}"
            PYTHON_PATH "${CMAKE_SOURCE_DIR}/src"
            DEPENDS striplan_native_module
        )
    else ()
        nanobind_add_stub(
            striplan_native_module_stubs
            MODULE striplan._iw_encoding_backend
            OUTPUT "${_stub_output}"
            DEPENDS striplan_native_module
            INSTALL_TIME
        )
    endif ()
endif ()
