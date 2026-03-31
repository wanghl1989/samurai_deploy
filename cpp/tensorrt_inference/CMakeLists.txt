cmake_minimum_required(VERSION 3.16)
project(sam2_tracker)

# Generate compile_commands.json for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -O0 -ggdb -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wno-deprecated-declarations -O2")

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(OpenMP REQUIRED)

# add_library(tensorrt_cpp_api SHARED
#             src/engine.cpp)

add_executable(${PROJECT_NAME}
            main.cpp
            src/engine.cpp
            src/sam2_tracker.cpp
            src/kalman_filter.cpp
            )
# message("-- TensorRT include dirs: ${TensorRT_INCLUDE_DIRS}")
# message("-- TensorRT libraries: ${TensorRT_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PRIVATE
                            include
                            ${OpenCV_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIRS}
                            /usr/include/spdlog
                            )

target_link_libraries(${PROJECT_NAME} PRIVATE
                        ${OpenCV_LIBS}
                        nvinfer
                        nvonnxparser
                        # onnxruntime
                        ${CUDA_LIBRARIES}
                        # OpenMP::OpenMP_CXX
                        spdlog::spdlog
                        )
target_compile_definitions(${PROJECT_NAME} PRIVATE 
                            SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
                        )
