cmake_minimum_required(VERSION 3.0)

project(cclient_project)

aux_source_directory(./src src)

set(CMAKE_BUILD_TYPE DEBUG)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /workspace/install/include)

add_executable(cclient ${src})

find_library(grpc_path grpcclient /workspace/install/lib)
find_library(http_path httpclient /workspace/install/lib)

target_link_libraries(cclient ${OpenCV_LIBS}  ${grpc_path} ${http_path})
