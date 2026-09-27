# cmake_minimum_required(VERSION 3.16)

# project(vision_training LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_executable(hello src/main.cpp)

# target_compile_options(hello PRIVATE -Wall -Wextra)

cmake_minimum_required(VERSION 3.16)
project(vision_math_demo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_executable(demo src/task1_image/main.cpp)
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(tmp src/task1_image/banz.cpp)
target_include_directories(tmp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tmp PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(t2 src/task2_fit/main.cpp)
target_include_directories(t2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(t2 PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(t3 src/task3_windmill/main.cpp)
target_include_directories(t3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(t3 PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)