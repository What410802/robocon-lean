# MIT 指令：12 个关节（**按执行器顺序** FL/FR/RR/RL × hip/thigh/calf）的 5 个参数。
#
# 方向：控制器节点 → 仿真节点。控制器**只发参数**，力矩由仿真侧的电机自己算：
#   tau_joint[i] = tau[i] + kp[i] * (q[i] - q_now[i]) + kd[i] * (w[i] - dq_now[i])
# 这与实机分工一致：上层发 MIT 五参数，驱动板的 FOC 电流环把它变成力矩（讲义 §1.2 / §2.3）。
#
# 单位（全是**关节侧**，不是转子侧）：kp [N·m/rad]、kd [N·m·s/rad]、q [rad]、w [rad/s]、tau [N·m]。
# 阻尼模式就是 kp = 0、q = w = tau = 0、kd = kd_damp ⇒ τ = −kd·dq。
std_msgs/Header header

float64[] kp
float64[] kd
float64[] q
float64[] w
float64[] tau
