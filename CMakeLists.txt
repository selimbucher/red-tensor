cmake_minimum_required(VERSION 3.21)
project(RedTensor LANGUAGES CXX HIP)

# Dependency Management
include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12  # Pin a specific version for stability
)

# Download Doctest
FetchContent_MakeAvailable(doctest)

# Setup Core Library
file(GLOB_RECURSE SRC_FILES "src/*.hip" "src/*.cpp")
add_library(core_ml ${SRC_FILES})
target_include_directories(core_ml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Setup Tests
file(GLOB TEST_SOURCES "tests/*.hip")

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    
    # Link the engine and the downloaded doctest library
    target_link_libraries(${test_name} PRIVATE core_ml doctest::doctest)

    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    
    set_target_properties(${test_name} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()