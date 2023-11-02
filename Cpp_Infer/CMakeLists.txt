cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

project(LightTrack)

find_package(OpenCV REQUIRED)
set(OpenVINO_DIR /usr/lib/x86_64-linux-gnu/cmake/openvino2022.3.0)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
    "/usr/include/eigen3"
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(LightTrack main.cpp Tracker.cpp)

target_link_libraries(
    LightTrack
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${OpenCV_LIBS}
)

