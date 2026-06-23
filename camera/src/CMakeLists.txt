cmake_minimum_required(VERSION 3.10)

project(videoTest)

set(OpenCV_DIR "/home/firefly/opencv411/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(videoTest
    main.cc
)

# 链接所需的库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}   # OpenCV库
)