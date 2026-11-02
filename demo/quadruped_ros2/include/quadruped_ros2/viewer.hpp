/**
 * @file viewer.hpp
 * @brief 自己开的 MuJoCo 窗口：GLFW + `mjv`/`mjr`，不用官方 `Simulate` 界面。
 *
 * **为什么不用官方界面**（2026-10-06 实测，见 UPSTREAM.md §2）：官方 `Simulate`
 * 是"物理线程 + UI 线程共享一把递归锁"的结构，UI 线程每帧要重绘两套 mjUI 面板、每个鼠标
 * 事件都要对面板做命中测试；实测**鼠标一移动帧率就从 56.9 fps 掉到 43.1 fps**（p95 41.8 ms），
 * 关掉面板也只能回到 46.0 fps。本文件这套是**单线程、完全无锁**（见下），同一负载下 56.0 fps。
 *
 * **线程安全**：本类**只在主线程用**（GLFW 也要求"谁建窗口谁用"）。节点里 `mjModel`/`mjData`
 * 归主线程独占（物理、渲染、复位都在主线程），ROS 执行器那条线程只碰 `SimNode` 自己的
 * 命令行槽位（它自己有一把小锁），所以这里不需要任何锁——这比"到处加锁"更安全，
 * 因为根本不存在共享可变状态，也就没有任何跨线程的收工标志：退出只有"用户点掉窗口"与
 * 节点的 `quit_requested`（后者由节点自己轮询）两条路，窗口这边只负责 `Poll()` 返回 false。
 *
 * 相机操作与上次任务（`@20260927_motor/cpp/src/viewer.h`）一致：左键转、右键平移（Shift 换轴）、
 * 中键/滚轮缩放；鼠标位移先除以**窗口（逻辑）高度**再交给 `mjv_moveCamera`，与官方
 * `platform_ui_adapter.cc` + `simulate.cc` 的两次换算等价（转动灵敏度与屏幕缩放无关）。
 */

#pragma once

#include <mujoco/mujoco.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace quadruped::viewer
{

/// 一次按键事件：键码 + 修饰键（`Ctrl+R` 这类要区分开，所以修饰键必须一起传）。
struct KeyEvent
{
    int key = 0;
    int mods = 0;
};

/// 自建窗口：事件轮询、按键队列、渲染与 HUD。
///
/// **按键映射是数据，不是 `switch`**：显示开关的快捷键直接从库里的 `mjVISSTRING` /
/// `mjRNDSTRING` 两张表生成（31 + 11 项，"改键"= 改 `kRemap` 一行），另外挂了三个动作键
/// （`F6` frame 循环、`F7` label 循环、`Home` 回默认视角）。官方界面那 12 个框架键是
/// `simulate.cc` 里硬编码的 `case`，改不动；我们的表可以随便改，见
/// UPSTREAM.md §2。
///
/// 翻这些 bit **不需要我们实现功能**：`opt_.flags[]` / `scn_.flags[]` / `opt_.frame` /
/// `opt_.label` 都是 `mjv_updateScene` 与 `mjr_render`（都在 libmujoco 里）的入参，
/// 与官方界面共用同一套实现；我们只是"把开关交给库里那两个函数"。
class Window
{
  public:
    /// 默认多重采样数（官方 `simulate/glfw_adapter.cc:53` 也是 4）。
    /// 必须声明在构造函数之前：默认参数里的类内名字按声明顺序查找。
    static constexpr int kDefaultMsaa = 4;
    /// 默认超采样倍率：离屏按 `倍率 × framebuffer` 渲染再缩回窗口（1 = 不超采样）
    static constexpr int kDefaultRenderScale = 2;

    Window(const mjModel* m, const char* title, int width, int height, bool shadow = true,
           int msaa = kDefaultMsaa, int render_scale = kDefaultRenderScale)
        : m_(m), width_(width), height_(height)
    {
        if (!glfwInit())
        {
            mju_error("glfwInit 失败");
        }
        glfwWindowHint(GLFW_VISIBLE, 1);
        glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
        // **抗锯齿**：默认帧缓冲的多重采样。官方适配器建窗时请求 4
        // （`simulate/glfw_adapter.cc:53`），我们这套自建窗口（沿用上次任务的 viewer.h）
        // 原本一个采样都没请求 —— 放大观察时部件边缘就是明显锯齿（用户 2026-10-06 实测反馈）。
        // 只有建窗**之前**设才有用；实际拿到多少用 `glfwGetWindowAttrib(GLFW_SAMPLES)` 回读。
        glfwWindowHint(GLFW_SAMPLES, msaa);
        win_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (win_ == nullptr)
        {
            mju_error("创建窗口失败（有显示服务吗？只要数字请用 viewer:=false）");
        }
        glfwMakeContextCurrent(win_);
        // 垂直同步：60 fps 上限，省得空转烧 CPU/GPU；帧率是否够用另说，见文件头。
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(win_, this);
        glfwSetKeyCallback(win_, OnKey);
        glfwSetMouseButtonCallback(win_, OnMouseButton);
        glfwSetCursorPosCallback(win_, OnCursorPos);
        glfwSetScrollCallback(win_, OnScroll);

        mjv_defaultCamera(&cam_);
        mjv_defaultOption(&opt_);
        mjv_defaultScene(&scn_);
        mjr_defaultContext(&con_);
        mjv_makeScene(m, &scn_, kMaxGeom);
        mjr_makeContext(m, &con_, mjFONTSCALE_150);
        mjr_setBuffer(mjFB_WINDOW, &con_); // 画到窗口上（不是离屏）
        viewport_ = mjr_maxViewport(&con_);

        // 阴影开关是**渲染标志**（`mjvScene.flags[mjRND_SHADOW]`）：`mjr_render` 每帧读它，
        // 所以运行中随时改都生效（官方界面面板里那个勾选就是它）。
        // 注意与 `mjVisual.quality.shadowsize`（**贴图尺寸**）不同：那个只在建立 GL 上下文
        // （`mjr_makeContext`）时被消费，运行中改无效——官方 `simulate/` 里这个函数
        // 也只在加载模型时调一次。见 UPSTREAM.md §2。
        scn_.flags[mjRND_SHADOW] = shadow ? 1 : 0;

        ResetCamera(); // 与 Python 侧 render_preview.py 的 iso 视角一致，便于和截图对照
        BuildBindings();
        const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        const int got_msaa = glfwGetWindowAttrib(win_, GLFW_SAMPLES);
        // **抗锯齿怎么落地**：先看窗口的默认 framebuffer 有没有拿到 MSAA（官方面板也走它）。
        // 本机实测 Mesa 三条平台路径（wayland / x11 / auto）请求 4× 都只拿到 0×，所以默认再走
        // "离屏渲染 → `mjr_blitBuffer` 缩回窗口"这条路：离屏缓冲的多重采样数在建上下文时由
        // `mjVisual.quality.offsamples` 决定（`SimLoop` 会先把它设成 `viewer_msaa`），而且离屏可以
        // 按 `render_scale` 倍尺寸渲染再缩下去（超采样），比窗口 MSAA 更彻底。
        // 见 UPSTREAM.md §2。
        use_offscreen_ = (render_scale > 1) || (msaa > 0 && got_msaa == 0);
        offscreen_scale_ = use_offscreen_ ? (render_scale > 0 ? render_scale : 1) : 1;
        std::printf("窗口：%dx%d（初始 framebuffer %dx%d，缩放随合成器变）；显示器 %dx%d @ %d Hz；"
                    "MSAA 请求 %d×、实得 %d×%s；渲染路径 %s%s\n",
                    width_, height_, viewport_.width, viewport_.height,
                    mode != nullptr ? mode->width : 0, mode != nullptr ? mode->height : 0,
                    mode != nullptr ? mode->refreshRate : 0, msaa, got_msaa,
                    got_msaa == 0 ? "（窗口拿不到 MSAA）" : "",
                    use_offscreen_ ? "离屏渲染→缩放 blit" : "直接画到窗口",
                    use_offscreen_ ? (offscreen_scale_ > 1 ? "，超采样" : "，离屏 MSAA") : "");
        std::printf(
            "鼠标：左键转、右键平移（Shift 换轴）、中键/滚轮缩放；"
            "显示开关快捷键 %d 个（与官方面板同一套，见 UPSTREAM.md §2）；"
            "F6/F7 循环 frame/label、Home 回默认视角\n",
            static_cast<int>(bindings_.size()) - 3);
    }

    ~Window()
    {
        if (win_ != nullptr)
        {
            mjr_freeContext(&con_);
            mjv_freeScene(&scn_);
            glfwDestroyWindow(win_);
            // 不调 glfwTerminate()：本机驱动上收尾会崩（@20260923_mujoco 的教训），
            // 进程退出时由系统回收。
        }
    }

    // 不可拷贝、不可移动：持有 GLFWwindow* 与 mjrContext，浅拷贝会 double free。
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// 取一次事件；返回 false = 窗口该关了（用户点掉窗口，或节点请求退出）。
    bool Poll()
    {
        glfwPollEvents();
        return win_ != nullptr && glfwWindowShouldClose(win_) == 0;
    }

    /// 取一个"本帧按下"的键（没有则返回 false）；修饰键一起给出，`Ctrl+R` 这类才分得清
    bool TakeKey(KeyEvent* event)
    {
        if (keys_.empty())
        {
            return false;
        }
        *event = keys_.front();
        keys_.erase(keys_.begin());
        return true;
    }

    /// 处理"显示开关"类按键：命中就翻对应 bit / 走对应动作，返回 true = 已消化，
    /// 上层（节点）不用再管。映射见 `kBindings`（由库表生成）。
    bool HandleKey(const KeyEvent& event)
    {
        if ((event.mods & GLFW_MOD_CONTROL) != 0)
        {
            return false; // 带 Ctrl 的都是节点动作（Ctrl+R 复位、Ctrl+Q 退出）
        }
        bool handled = false;
        for (const Binding& b : bindings_)
        {
            if (b.key != event.key)
            {
                continue;
            }
            handled = true;
            switch (b.action)
            {
            case Action::kVisFlag:
                opt_.flags[b.index] = opt_.flags[b.index] ? 0 : 1;
                break;
            case Action::kRndFlag:
                scn_.flags[b.index] = scn_.flags[b.index] ? 0 : 1;
                break;
            case Action::kCycleFrame:
                opt_.frame = (opt_.frame + 1) % mjNFRAME;
                break;
            case Action::kCycleLabel:
                opt_.label = (opt_.label + 1) % mjNLABEL;
                break;
            case Action::kResetCamera:
                ResetCamera();
                break;
            }
        }
        return handled;
    }

    /// 当前开着的显示项（快捷键字符串拼起来，给 HUD 用）
    std::string FlagsSummary() const
    {
        std::string out;
        for (const Binding& b : bindings_)
        {
            const int on = b.action == Action::kVisFlag   ? opt_.flags[b.index]
                           : b.action == Action::kRndFlag ? scn_.flags[b.index]
                                                          : 0;
            if (on != 0)
            {
                if (!out.empty())
                {
                    out += ' ';
                }
                out += b.shortcut;
            }
        }
        return out.empty() ? "-" : out;
    }

    /// 画一帧：更新场景 → 渲染 → 两段 HUD → 交换缓冲（内部等垂直同步）。
    /// `hud_left` / `hud_right` **必须是 ASCII**：MuJoCo 内置字体只覆盖 ASCII，
    /// 中文/希腊字母/°/· 会被画成实心块（上次任务实测，每字节"墨迹"95~122 像素）。
    void Draw(const mjModel* m, mjData* d, const char* hud_left, const char* hud_right)
    {
        // 视口每帧按**实际 framebuffer** 重取：窗口缩放/最大化后 Framebuffer 会变
        // （Wayland 下 compositor 随时会改），只在构造时取一次会把画面留在左下角。
        int fb_w = 0;
        int fb_h = 0;
        glfwGetFramebufferSize(win_, &fb_w, &fb_h);
        if (fb_w <= 0 || fb_h <= 0)
        {
            return; // 最小化时是 0×0，别画（也别除零）
        }
        viewport_ = {0, 0, fb_w, fb_h};

        // 跟随 base：qpos[0..2] 是 free joint 的 base 位置（black/blackW 都是模型第一个 joint）。
        if (follow_base_ && m->nq >= 3)
        {
            cam_.lookat[0] = d->qpos[0];
            cam_.lookat[1] = d->qpos[1];
            cam_.lookat[2] = d->qpos[2];
        }
        mjv_updateScene(m, d, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
        if (!use_offscreen_)
        {
            mjr_render(viewport_, &scn_, &con_);
        }
        else
        {
            // 离屏尺寸 = framebuffer × 超采样倍率；变了就重建（窗口/缩放变化时）
            const int ow = viewport_.width * offscreen_scale_;
            const int oh = viewport_.height * offscreen_scale_;
            if (ow != off_w_ || oh != off_h_)
            {
                mjr_resizeOffscreen(ow, oh, &con_);
                off_w_ = ow;
                off_h_ = oh;
            }
            const mjrRect off{0, 0, off_w_, off_h_};
            mjr_setBuffer(mjFB_OFFSCREEN, &con_);
            mjr_render(off, &scn_, &con_);
            // **blit 要在"离屏仍是当前 buffer"时调用**：它的语义是"从当前 framebuffer 的 src
            // 拷到另一个（窗口）的 dst"（mujoco.h:915）。先切回窗口再 blit 会拷反方向 → 画面全黑。
            // 尺寸不同 → GL_LINEAR 插值缩回窗口（同一条注释）。
            mjr_blitBuffer(off, viewport_, 1, 0, &con_);
            mjr_setBuffer(mjFB_WINDOW, &con_);
        }
        WarnIfNotAscii(hud_left, hud_right);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_, hud_left, nullptr, &con_);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPRIGHT, viewport_, hud_right, nullptr, &con_);
        glfwSwapBuffers(win_);
    }

    /// 阴影开关（运行中随时可改：只是每帧渲染标志）
    void SetShadow(bool on) { scn_.flags[mjRND_SHADOW] = on ? 1 : 0; }
    bool ShadowEnabled() const { return scn_.flags[mjRND_SHADOW] != 0; }

    /// 当前**窗口（逻辑）高度**：鼠标位移按它归一化（见文件头）
    int WindowHeight() const
    {
        int w = 0;
        int h = 0;
        glfwGetWindowSize(win_, &w, &h);
        if (h > 0)
        {
            return h;
        }
        return viewport_.height > 0 ? viewport_.height : height_;
    }

  public:
    /// 相机是否跟随机器人 base：每帧把 free camera 的 lookat 设成 base 位置（与评审/录制视频
    /// 脚本里"cam.lookat=base"的跟随方式一致）。只动 lookat，方位角/仰角/距离仍由鼠标控制。
    void SetFollowBase(bool on) { follow_base_ = on; }
    bool FollowBase() const { return follow_base_; }

  private:
    static constexpr int kMaxGeom = 2000; ///< 场景 geom 上限（够这个狗 + 地面用）

    /// 一个绑定做什么
    enum class Action
    {
        kVisFlag,    ///< 翻 `mjvOption.flags[index]`（画什么）
        kRndFlag,    ///< 翻 `mjvScene.flags[index]`（怎么画）
        kCycleFrame, ///< `F6`：frame 可视化循环
        kCycleLabel, ///< `F7`：label 可视化循环
        kResetCamera ///< `Home`：回默认 iso 视角
    };

    /// 一条按键映射：键码 + 官方表里的快捷键字符（HUD 显示它）+ 动作 + 下标
    struct Binding
    {
        int key;
        char shortcut;
        Action action;
        int index;
    };

    /// **改键表**：`{官方快捷键, 换成哪个 GLFW 键}`，空 = 全按官方来。
    /// 想改键就在这里加一行（例如把"反射"从 `R` 挪到 `F9`：`{'R', GLFW_KEY_F9}`），
    /// 渲染与节点代码都不用动。官方那 12 个框架键是硬编码的 `case`，改不了；
    /// 我们这张表是数据，见 UPSTREAM.md §2。
    struct Remap
    {
        char official;
        int key;
    };
    static constexpr std::array<Remap, 0> kRemap{};

    /// 官方快捷键字符 → GLFW 键码（只用得到字母和这几个符号）
    static int KeyFromChar(char c)
    {
        if (c >= 'A' && c <= 'Z')
        {
            return GLFW_KEY_A + (c - 'A');
        }
        switch (c)
        {
        case '\'':
            return GLFW_KEY_APOSTROPHE;
        case ',':
            return GLFW_KEY_COMMA;
        case ';':
            return GLFW_KEY_SEMICOLON;
        case '/':
            return GLFW_KEY_SLASH;
        case '`':
            return GLFW_KEY_GRAVE_ACCENT;
        case '\\':
            return GLFW_KEY_BACKSLASH;
        default:
            return 0;
        }
    }

    static int RemappedKey(char official)
    {
        for (const Remap& r : kRemap)
        {
            if (r.official == official)
            {
                return r.key;
            }
        }
        return KeyFromChar(official);
    }

    /// 建按键表：**直接从库里的两张表读**，所以永远与所用 MuJoCo 版本一致
    void BuildBindings()
    {
        for (int i = 0; i < mjNVISFLAG; ++i)
        {
            if (mjVISSTRING[i][2][0] != 0)
            {
                const int key = RemappedKey(mjVISSTRING[i][2][0]);
                if (key != 0)
                {
                    bindings_.push_back({key, mjVISSTRING[i][2][0], Action::kVisFlag, i});
                }
            }
        }
        for (int i = 0; i < mjNRNDFLAG; ++i)
        {
            if (mjRNDSTRING[i][2][0] != 0)
            {
                const int key = RemappedKey(mjRNDSTRING[i][2][0]);
                if (key != 0)
                {
                    bindings_.push_back({key, mjRNDSTRING[i][2][0], Action::kRndFlag, i});
                }
            }
        }
        // 动作键（官方分别在 F6 / F7 / Esc；Esc 在我们这儿是退出，所以视角复位用 Home）
        bindings_.push_back({GLFW_KEY_F6, ' ', Action::kCycleFrame, -1});
        bindings_.push_back({GLFW_KEY_F7, ' ', Action::kCycleLabel, -1});
        bindings_.push_back({GLFW_KEY_HOME, ' ', Action::kResetCamera, -1});
    }

    /// 默认 iso 视角（与 Python 侧 render_preview.py 一致，便于和截图对照）
    void ResetCamera()
    {
        cam_.type = mjCAMERA_FREE;
        cam_.azimuth = 135.0;
        cam_.elevation = -20.0;
        cam_.distance = 2.0;
        cam_.lookat[0] = 0.0;
        cam_.lookat[1] = 0.0;
        cam_.lookat[2] = 0.15;
    }

    static Window* Self(GLFWwindow* w) { return static_cast<Window*>(glfwGetWindowUserPointer(w)); }

    static bool IsAscii(const char* text)
    {
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p != '\0'; ++p)
        {
            if (*p >= 0x80)
            {
                return false;
            }
        }
        return true;
    }

    void WarnIfNotAscii(const char* left, const char* right)
    {
        if (ascii_ok_)
        {
            return;
        }
        if (IsAscii(left) && IsAscii(right))
        {
            ascii_ok_ = true;
            return;
        }
        std::printf("警告：HUD 里有非 ASCII 字符，MuJoCo 内置字体画不出来（会变成实心块）。\n"
                    "      HUD 一律用 ASCII，中文留给终端日志。\n");
        ascii_ok_ = true;
    }

    static void OnKey(GLFWwindow* w, int key, int /*scancode*/, int action, int mods)
    {
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
        {
            Self(w)->keys_.push_back(KeyEvent{key, mods});
        }
    }

    static void OnMouseButton(GLFWwindow* w, int button, int action, int mods)
    {
        Window* s = Self(w);
        s->dragging_ = (action == GLFW_PRESS);
        s->button_ = button;
        s->shift_ = (mods & GLFW_MOD_SHIFT) != 0;
        glfwGetCursorPos(w, &s->last_x_, &s->last_y_);
    }

    static void OnCursorPos(GLFWwindow* w, double x, double y)
    {
        Window* s = Self(w);
        if (!s->dragging_)
        {
            return;
        }
        const double h = static_cast<double>(s->WindowHeight());
        const double dx = (x - s->last_x_) / h;
        const double dy = (y - s->last_y_) / h; // GLFW 的 y 向下，官方两次取负相消 → 这里不取负
        s->last_x_ = x;
        s->last_y_ = y;
        int action = mjMOUSE_ROTATE_V;
        if (s->button_ == GLFW_MOUSE_BUTTON_RIGHT)
        {
            action = s->shift_ ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        }
        else if (s->button_ == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            action = mjMOUSE_ZOOM;
        }
        else
        {
            action = s->shift_ ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        }
        mjv_moveCamera(s->m_, action, dx, dy, &s->cam_);
    }

    static void OnScroll(GLFWwindow* w, double /*xoffset*/, double yoffset)
    {
        Window* s = Self(w);
        mjv_moveCamera(s->m_, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, &s->cam_); // 5% 窗口高度
    }

    const mjModel* m_ = nullptr;
    GLFWwindow* win_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;
    mjrRect viewport_{};
    bool use_offscreen_ = false;
    int offscreen_scale_ = 1;
    int off_w_ = 0;
    int off_h_ = 0;
    std::vector<KeyEvent> keys_;
    std::vector<Binding> bindings_;
    bool follow_base_ = false;
    bool dragging_ = false;
    bool shift_ = false;
    bool ascii_ok_ = false;
    int button_ = GLFW_MOUSE_BUTTON_LEFT;
    double last_x_ = 0.0;
    double last_y_ = 0.0;
};

} // namespace quadruped::viewer
