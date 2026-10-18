cmake_minimum_required(VERSION 3.20)
project(edge_vision_accelerator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive- /EHsc)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -pthread)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Core C++ Static Library
add_library(eva_core STATIC
    cpp/src/accelerator.cpp
)
target_include_directories(eva_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

# Optional Python Bindings with Pybind11
option(BUILD_PYTHON_BINDINGS "Build Python C++ PyBind11 extension" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        # Try finding pybind11 via cmake or submodule
        find_package(pybind11 CONFIG QUIET)
        if(NOT pybind11_FOUND)
            # Fetch or use fallback
            message(STATUS "pybind11 CMake package not found. Looking for pybind11 include dirs.")
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m pybind11 --includes
                OUTPUT_VARIABLE PYBIND11_INC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(PYBIND11_INC)
                string(REPLACE "-I" "" PYBIND11_INC_DIRS "${PYBIND11_INC}")
                separate_arguments(PYBIND11_INC_DIRS)
                include_directories(${PYBIND11_INC_DIRS})
                add_library(_eva_cpp MODULE
                    cpp/bindings/eva_bindings.cpp
                    cpp/src/accelerator.cpp
                )
                target_link_libraries(_eva_cpp PRIVATE eva_core Python3::Python)
                set_target_properties(_eva_cpp PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")
            endif()
        else()
            pybind11_add_module(_eva_cpp
                cpp/bindings/eva_bindings.cpp
                cpp/src/accelerator.cpp
            )
            target_link_libraries(_eva_cpp PRIVATE eva_core)
        endif()
    endif()
endif()

# Optional Native Tests
option(BUILD_TESTS "Build native C++ tests" ON)
if(BUILD_TESTS)
    add_executable(eva_test_suite
        tests/cpp_tests.cpp
        cpp/src/accelerator.cpp
    )
    target_link_libraries(eva_test_suite PRIVATE eva_core)
endif()
