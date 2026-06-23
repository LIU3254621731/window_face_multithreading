# 设置CMake最低版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称
project(Face_Mesh_v2)




# 设置OpenCV路径

set(OpenCV_DIR "C:\\Users\\32546\\Downloads\\Programs\\opencv\\build")
find_package(OpenCV REQUIRED)


# 打印确认路径
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

# 设置运行时库路径
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")


# 设置Eigen的路径
set(Eigen3_DIR "C:\\Users\\32546\\vcpkg\\packages\\eigen3_x64-windows\\share\\eigen3" )
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_Include})


# 收集当前目录下所有的.cc源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#设置iir滤波器的路径   
add_subdirectory(iir1-master)

# 创建可执行文件
add_executable(${PROJECT_NAME}
    main.cc
    camera.cc
    image_processor.cc
    pos_algorithm.cc
    yunet.cc
    SignalProcessor.cc
    filter.cc
)

# 链接所需的库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}   # OpenCV库
    Eigen3::Eigen    # Eigen库
    iir::iir_static  # iir库
)

# 配置线程库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

# 设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}        # 当前目录
    ${OpenCV_INCLUDE_DIRS}            # OpenCV头文件目录
    ${EIGEN3_INCLUDE_DIR}             # Eigen头文件目录
)

# 安装目标文件
install(TARGETS ${PROJECT_NAME} DESTINATION .)
# 安装测试图像
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/test.jpg DESTINATION model)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/RetinaFace.rknn DESTINATION model)

# 安装shell脚本
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run.sh DESTINATION .)
# 设置shell脚本的执行权限
install(CODE "execute_process(COMMAND chmod +x ${CMAKE_INSTALL_PREFIX}/run.sh)")



