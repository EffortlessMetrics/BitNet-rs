cmake_minimum_required(VERSION 3.14)
project(bitnet_cpp_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Find required packages
# In a real implementation, this would find the actual BitNet C++ library
# find_package(BitNet REQUIRED)

# Create executable
add_executable(bitnet_example main.cpp)

# Link libraries
# In a real implementation, this would link against BitNet C++ libraries
# target_link_libraries(bitnet_example BitNet::BitNet)

# For this example, we'll just link standard libraries
target_link_libraries(bitnet_example)

# Set output directory
set_target_properties(bitnet_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation (optional)
install(TARGETS bitnet_example
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

# Custom target for running the example
add_custom_target(run
    COMMAND bitnet_example
    DEPENDS bitnet_example
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running BitNet C++ example"
)

# Custom target for cleaning
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)
