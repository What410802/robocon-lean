# 深度相机帧（仿真节点 → 策略侧）：一帧 64x48 的公制**平面**深度。
#
# 这是 plum 深度策略在 sim 侧的输入契约，与 eval 的 policy_runner.render_depth 同源：
# depth[i] = 沿相机光轴的平面距离 [m]（不是欧氏距离），行主序（height 行 x width 列）。
# 无命中（天空 / 超出远端）就是转换结果里的远端值（本场景 ≈150 m），近端约 0.03 m；
# 消费者自己按训练契约归一化（clamp(min_depth, cutoff) / cutoff，cutoff=3.0 m）。
std_msgs/Header header

# 渲染这一帧时 MuJoCo 的 data.time [s]。与同一步 /motor_state 的 sim_time 相同：
# 本节点先发深度、再发反馈（消费端因此可以"先看到深度，再被反馈回调驱动"）。
float64 sim_time

# 图像尺寸。固定 64x48（训练/评测契约 depth.json 的 width/height），放这里是为了消息自描述。
uint32 width
uint32 height

# metric planar depth [m], row-major (height rows x width cols)
float32[] depth
