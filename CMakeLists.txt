cmake_minimum_required(VERSION 3.10)

# Set compiler before project() command
if(APPLE)
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

project(3DHoPD)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# Set OpenMP configuration for macOS
if(APPLE)
    # Add LLVM directory to the search path
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/llvm" ${CMAKE_PREFIX_PATH})
    
    # OpenMP settings
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

    # Include directories for OpenMP
    include_directories("/opt/homebrew/opt/libomp/include")
endif()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)  # For std::this_thread

# PCL configuration - updated with required components
set(PCL_DIR "/usr/lib/cmake/pcl")
find_package(PCL 1.14 REQUIRED COMPONENTS 
    common 
    io 
    visualization 
    features 
    filters 
    registration 
    kdtree 
    search
)

# Boost configuration
find_package(Boost REQUIRED COMPONENTS 
    thread 
    system 
    filesystem 
    chrono 
    date_time 
    atomic
)

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    include
    /opt/homebrew/include  # Additional macOS paths
)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${Boost_LIBRARY_DIRS}
    /opt/homebrew/lib
)

# Compiler flags
add_definitions(${PCL_DEFINITIONS})
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Main executable with all dependencies
add_executable(3DHoPD src/3DHoPD.cpp)
target_link_libraries(3DHoPD
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
    Boost::thread
    Boost::system
    Boost::filesystem
    Boost::chrono
    Threads::Threads
)

# Robustness test executable
add_executable(3DHoPD_robustness_to_rotations src/3DHoPD_robustness_to_rotations.cpp)
target_link_libraries(3DHoPD_robustness_to_rotations
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
    Boost::thread
    Boost::system
)

# Installation directives (optional)
install(TARGETS 3DHoPD 3DHoPD_robustness_to_rotations
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

