cmake_minimum_required(VERSION 3.16)

project(quadruped_ros2)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For IDE IntelliSense

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# MuJoCo / glfw 由 pixi 环境（`pixi run …`，见仓库根 pixi.toml）提供：它们的 CMake 配置
# 就在 $CONDA_PREFIX 下。CMake 默认不会去搜 conda 前缀，所以显式加进来——用环境变量而不是写死
# 路径（换机器 / 换环境名都对）。只链 mujoco::mujoco：窗口是自己用 GLFW + mjv/mjr 开的
# （include/quadruped_ros2/viewer.hpp），不再需要官方界面那个 libmujoco_simulate。
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
# mujocoConfig does find_dependency(OpenGL); pin the same legacy-GL choice CMake already makes by
# default so CMake 4.x does not emit the CMP0072 dev warning and the link set cannot flip later.
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(mujoco REQUIRED)
# 截图诊断（viewer.hpp 的 SavePpm）直接用 glReadPixels/glPixelStorei，所以要显式链 OpenGL
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

# ① 自定义消息（任务第 1 项）：控制器 → 仿真 的 MIT 五参数，仿真 → 控制器 的电机反馈，
#    控制器 → 手柄 的回程状态。
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/MitCommand.msg"
    "msg/MotorState.msg"
    "msg/ControlStatus.msg"
    DEPENDENCIES std_msgs
)

# ②a 控制器节点（任务第 2 项，必须 C++）：控制律见 include/quadruped_ros2/control.hpp，
#     移植自 @20260927_motor/cpp/essential_core/src/state.h。
add_executable(controller_node src/controller_node.cpp)
target_include_directories(controller_node PRIVATE include)
ament_target_dependencies(controller_node rclcpp std_msgs sensor_msgs std_srvs)

# ②b 仿真节点（任务第 2 项）：MuJoCo + 电机模型，窗口自建（GLFW + mjv/mjr）。
add_executable(sim_node src/sim_node.cpp)
target_include_directories(sim_node PRIVATE include)
# 这条不是给编译器用的（conda g++ 内置就搜 $CONDA_PREFIX/include），而是给**读编译数据库的工具**：
# MuJoCo / GLFW 的头文件在 $CONDA_PREFIX/include 下，而"编译器的内置搜索路径"不会出现在
# compile_commands.json 里，于是 clangd 会报 'mujoco/mujoco.h' file not found（实测 68 个错误）。
# 注意两点：① 用 target_compile_options 直接传 -I——CMake 会把 target_include_directories 里
# "编译器已内置的目录"过滤掉，那样写等于没写；② 修复后源文件在 clangd 下 0 错误，
# 但 clangd 对"头文件"推断编译命令时仍可能丢这条 -I，头文件若还报错就给 clangd 传
# `--query-driver=**/bin/*`（glob 要匹配编译数据库里出现的 …/bin/c++）。
if(DEFINED ENV{CONDA_PREFIX})
    target_compile_options(sim_node PRIVATE "-I$ENV{CONDA_PREFIX}/include")
endif()
ament_target_dependencies(sim_node rclcpp std_msgs sensor_msgs std_srvs)

# 自己生成的消息类型支持（Humble 的推荐写法；rosidl_target_interfaces 已弃用）
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(controller_node "${cpp_typesupport_target}")
target_link_libraries(sim_node
    "${cpp_typesupport_target}"
    mujoco::mujoco
    glfw
    OpenGL::GL
    Threads::Threads
)

install(TARGETS
    controller_node
    sim_node
    DESTINATION lib/${PROJECT_NAME})

# ③ 手柄节点（任务第 3 项，Python）：按可执行文件装进 lib/<包名>/，ros2 run 才能起它。
#    带 RENAME 时 colcon 的 --symlink-install 会退化成"拷贝"（见 ament_cmake_symlink_install_programs
#    的 unsupported_keywords）——所以改了 joy_node.py 要重新 colcon build，这点与 C++ 侧一样。
install(PROGRAMS
    scripts/joy_node.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME joy_node)

# ④ launch（任务第 4 项）
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)

# 说明：本包不启用模板自带的 ament_lint 测试块 —— uncrustify 按 ROS 风格（2 空格）校验，
# 与本目录的 4 空格 / 100 列 / Allman 风格（.clang-format）冲突。

ament_package()
