# 12 个电机的反馈（**按执行器顺序**，与 MitCommand 同一套顺序）+ 仿真时间。
#
# 方向：仿真节点 → 控制器节点。字段与任务书给的一致：{q, dq, ddq, tau, cur}。
# imu 不在本条消息里：它用 ROS 2 自带的 sensor_msgs/Imu，单独发在 /imu。
std_msgs/Header header

# 仿真时间 [s]（= MuJoCo 的 data.time）。
# 控制器用它当**起身斜坡的时基**，而不是用 header.stamp（那是发消息时的墙上时间）：
# 这样仿真全速跑（realtime:=false，远快于 1x）时，"1 s 站完"仍然是 1 s 仿真时间，
# 回归脚本不必等真实时间，量出来的轨迹也与实时跑的一致。
float64 sim_time

# 关节角 [rad]
float64[] q

# 关节角速度 [rad/s]
float64[] dq

# 关节角加速度 [rad/s^2]。MuJoCo 里**读得到**（data.qacc，取执行器所在自由度的分量）：
# 实测起身过程峰值 7.2 rad/s²、趴卧/站住静止时 ~1e-5（数值零，是约束求解的残差）。
float64[] ddq

# 实际输出力矩（关节侧）[N·m]（= MuJoCo 的 actuator_force，已经过电机侧限幅）
float64[] tau

# 电机电流 [A]：**恒 0 = 未测量**（培训方口径"读不到就置 0"）。
# MuJoCo 的执行器是纯力矩源，模型里没有电流环也没有电流传感器；按 cur = tau/(减速比×转矩常数)
# 折算看似有量纲，但转矩常数 Kt 无出处（讲义与手册都没给 GO-M8010-6 的值），算出来的数与实机
# 电流没有可验证的对应关系，所以不编。字段保留是因为任务书的接口就是这个五元组。
# 将来真要做，需要的是"电机电气模型 + 实测标定数据"。见 docs/ros2-nodes.md §1.1。
float64[] cur
