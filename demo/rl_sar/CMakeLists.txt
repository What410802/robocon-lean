# rl_sar: builds only the node that runs an RL policy in simulation (rl_sim) plus a small self-test.
#
# Upstream rl_sar e5c2f41 has 344 lines, most of them ROS 1 / real-robot SDK branches; this is a rewrite
# of its ROS 2 + rl_sim branch - see UPSTREAM.md §2 and the root MIGRATION_NOTES.md.

cmake_minimum_required(VERSION 3.16)
project(rl_sar VERSION 3.0.0 LANGUAGES CXX)

# The 2.14 torch wheel headers require C++20 (ATen/ATen.h hard-errors on
# older standards); the rest of this package is plain C++17 code.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Same as upstream: policies and yaml are looked up as <source dir>/policy/<robot>/... (rl_sdk.cpp).
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(quadruped_ros2 REQUIRED) # messages: MitCommand / MotorState (@20261005_ros2)

# libtorch comes from the PyPI torch wheel (the same 2.14.1 cpu build the eval
# judge runs), not from a conda libtorch package: conda libtorch 2.14 cannot
# coexist with mujoco 3.11 in one environment.  Ask the environment's python
# where the wheel is, then point find_package() at its CMake config.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_LOOKUP_RESULT)
if(NOT TORCH_LOOKUP_RESULT EQUAL 0 OR TORCH_PYTHON_DIR STREQUAL "")
    message(FATAL_ERROR "cannot locate the torch python package; no PyPI torch in this environment (pixi install -e demo)")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_DIR}")
find_package(Torch REQUIRED)          # stock TorchConfig.cmake shipped inside the wheel
# The wheel's TorchConfig/Caffe2 config quotes the main libraries by absolute path
# but still lists plain names (c10); give the linker the wheel's lib dir too.
link_directories("${TORCH_PYTHON_DIR}/lib")
find_package(TBB REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# framework layer: observation assembly / inference / action scaling / yaml / FSM (verbatim upstream)
add_library(rl_sdk STATIC
    library/core/rl_sdk/rl_sdk.cpp
    library/core/observation_buffer/observation_buffer.cpp
)
target_include_directories(rl_sdk PUBLIC
    library/core/rl_sdk
    library/core/observation_buffer
    library/core/fsm
    library/core/loop
)
target_link_libraries(rl_sdk PUBLIC ${TORCH_LIBRARIES} TBB::tbb yaml-cpp Threads::Threads)

# adaptation layer: the ROS 2 node that talks to @20261005_ros2's sim node
add_executable(rl_sim src/rl_sim.cpp)
target_include_directories(rl_sim PRIVATE include policy)
target_link_libraries(rl_sim rl_sdk)
ament_target_dependencies(rl_sim rclcpp sensor_msgs geometry_msgs std_srvs quadruped_ros2)

install(TARGETS rl_sim DESTINATION lib/${PROJECT_NAME})
# The wheel's libtorch lives under site-packages/torch/lib.  Let CMake put every
# linked library directory on the installed binary's rpath, so `ros2 run`
# works without an LD_LIBRARY_PATH dance.
set_target_properties(rl_sim PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Header-only self-tests: no ROS graph, no sim, no gamepad. They live in test/ because they are not
# part of the node. Built unconditionally so `ros2 run rl_sar <name>` always works; the CTest
# registration is what makes `colcon test --packages-select rl_sar` run them too.
#   input_map_selftest     - key map / intents (see include/input_map.hpp)
#   command_limit_selftest - command-side limiting (see include/command_limit.hpp)
foreach(selftest IN ITEMS input_map_selftest command_limit_selftest)
  add_executable(${selftest} test/${selftest}.cpp)
  target_include_directories(${selftest} PRIVATE include)
  install(TARGETS ${selftest} DESTINATION lib/${PROJECT_NAME})
endforeach()
include(CTest)
if(BUILD_TESTING)
  add_test(NAME input_map_selftest COMMAND input_map_selftest)
  add_test(NAME command_limit_selftest COMMAND command_limit_selftest)
endif()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
