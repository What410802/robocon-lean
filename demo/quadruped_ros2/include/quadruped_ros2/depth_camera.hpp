/**
 * @file depth_camera.hpp
 * @brief 深度相机：EGL 离屏渲染 64x48 公制平面深度（plum 深度策略在 sim 侧的输入）。
 *
 * 数值契约（"看着差不多"不行：训练/评测喂给策略的就是这些数，差一个 ULP 也可能让动作分叉）：
 *   * 相机 = 场景 XML 里已有的 `depth_cam`（`demo/robots/blackW/blackW_description.xml`，
 *     挂在 d435_link 上），`mjCAMERA_FIXED`；固定 64x48，与 policy 目录的 `depth.json` 一致；
 *   * **只画 geom group 0**（地形/障碍物）：blackW 的视觉网格是 group 1、碰撞体是 group 3，
 *     所以看不见自己的狗——训练/评测（`visible_geom_groups=[0]`）就是这么渲染的；
 *   * 渲染参数逐项照抄 `mujoco/rendering/classic/renderer.py` 的 depth 路径：场景标志
 *     `mjRND_SEGMENT + mjRND_IDCOLOR`（分段渲染在远处精度更好）、`mjrContext.readDepthMap =
 *     mjDEPTH_ZEROFAR`、`mjr_readPixels(None, depth, rect, ctx)`；然后按 renderer.py 的公式把
 *     reverse-Z 的 NDC 换成米（**系数用 float32 算、除法用 float64**），最后**上下翻转**
 *     （GL 的行序是自下而上）。eval 侧用同一场景同一公式渲染，逐位对拍差异应为 0。
 *
 * GL 上下文：有窗口时**复用 GLFW 已经 make current 的上下文**（同一线程，没必要再开一个）；
 * 无窗口（headless demo / `--depth-self-test`）时用 EGL 自己建一个离屏上下文。EGL display 的
 * 选法与 mujoco-python 的 `mujoco/egl/__init__.py` 一致：先枚举 `EGLDeviceEXT`，对每个设备
 * `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, ...)` + `eglInitialize`，取第一个成功的；
 * **不要先试 `eglGetDisplay(EGL_DEFAULT_DISPLAY)`**——本机实测它会落到另一套驱动（Mesa）上，
 * 渲染数值就与 eval 的 Python 端对不上了（设备扩展不可用时才兜底走它）。
 *
 * 线程：与 `mjModel`/`mjData` 同一条主线程使用，无锁。
 */
#pragma once

#include <mujoco/mujoco.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace quadruped::depth
{

/// 深度相机。用法：`Init()` 一次 → 每帧 `Render()`。任何一步失败都只是返回 false 并给出
/// `error()`，调用方（sim_node）降级为"不发 /depth_cam"，仿真与盲走策略不受影响。
class DepthCamera
{
  public:
    /// 训练/评测契约的固定分辨率（`depth.json` 的 width/height）
    static constexpr int kWidth = 64;
    static constexpr int kHeight = 48;

    DepthCamera() = default;
    ~DepthCamera() { Close(); }
    DepthCamera(const DepthCamera&) = delete;
    DepthCamera& operator=(const DepthCamera&) = delete;

    /// 初始化。`gl_context_already_current = true` 表示调用线程上已经有一个 current 的 GL
    /// 上下文（窗口模式）：直接复用它、完全不碰 EGL。失败返回 false，原因见 `error()`。
    bool Init(mjModel* m, bool gl_context_already_current, const char* camera_name = "depth_cam");

    /// 渲染一帧：写 `kWidth*kHeight` 个 float32（行主序、单位 m、已上下翻转，与 Python 逐位一致）。
    bool Render(const mjModel* m, const mjData* d, std::vector<float>* depth);

    const std::string& error() const { return error_; }

  private:
    /// mjvScene 容量。取 mujoco.renderer 的默认 max_geom=10000，保证与 eval 侧同一口径
    /// （本场景远用不到，但"容量不同导致截断顺序不同"这种事不该留给运气）。
    static constexpr int kMaxGeom = 10000;

    bool InitEgl();
    void Close();

    bool initialized_ = false;
    bool owns_egl_ = false; ///< EGL display/context/surface 是否由本类创建（窗口复用时 false）
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLContext egl_context_ = EGL_NO_CONTEXT;

    int camera_id_ = -1;
    float c_coef_ = 0.0f; ///< reverse-Z 反投影系数（推导见 Init；与 renderer.py 逐位同式）
    float d_coef_ = 0.0f;

    mjvScene scene_{};
    mjvOption option_{};
    mjvCamera camera_{};
    mjrContext context_{};
    mjrRect rect_{0, 0, kWidth, kHeight};
    bool scene_made_ = false;
    bool context_made_ = false;
    std::vector<float> raw_; ///< glReadPixels 的原始 NDC（自下而上），每帧复用

    std::string error_;
};

/// 与 `mujoco/egl/__init__.py::create_initialized_egl_device_display` 相同的选法（理由见文件头）。
inline EGLDisplay EglDeviceDisplay()
{
    const auto query_devices =
        reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    const auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (query_devices != nullptr && get_platform_display != nullptr)
    {
        EGLDeviceEXT devices[16];
        EGLint count = 0;
        if (query_devices(16, devices, &count))
        {
            for (EGLint i = 0; i < count; ++i)
            {
                EGLDisplay display =
                    get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                if (display == EGL_NO_DISPLAY || eglGetError() != EGL_SUCCESS)
                {
                    continue;
                }
                EGLint major = 0;
                EGLint minor = 0;
                if (eglInitialize(display, &major, &minor) && eglGetError() == EGL_SUCCESS)
                {
                    return display;
                }
            }
        }
    }
    // 设备扩展不可用时的兜底：与 mujoco-python 不同，可能落到 Mesa/软件渲染上，
    // 所以只在设备枚举失败时走这里（数值对拍以设备路径为准）。
    EGLDisplay fallback = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0;
    EGLint minor = 0;
    if (fallback == EGL_NO_DISPLAY || !eglInitialize(fallback, &major, &minor))
    {
        return EGL_NO_DISPLAY;
    }
    return fallback;
}

inline bool DepthCamera::InitEgl()
{
    egl_display_ = EglDeviceDisplay();
    if (egl_display_ == EGL_NO_DISPLAY)
    {
        error_ = "无法初始化 EGL display（没有可用的 EGL 设备/驱动？）";
        return false;
    }
    owns_egl_ = true; // 从这里开始 Close() 负责销毁 display

    // 配置属性与 mujoco.egl 的 EGL_ATTRIBUTES 完全一致（选到哪个 config 不影响离屏 FBO）。
    const EGLint config_attrs[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                   EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
                                   EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER, EGL_SURFACE_TYPE,
                                   EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    if (!eglChooseConfig(egl_display_, config_attrs, &config, 1, &num_configs) || num_configs < 1)
    {
        error_ = "EGL 找不到匹配的 framebuffer 配置";
        return false;
    }
    if (!eglBindAPI(EGL_OPENGL_API))
    {
        error_ = "eglBindAPI(EGL_OPENGL_API) 失败";
        return false;
    }
    egl_context_ = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, nullptr);
    if (egl_context_ == EGL_NO_CONTEXT)
    {
        error_ = "eglCreateContext 失败";
        return false;
    }
    // 与 mujoco.egl 相同：优先 surfaceless（同一套 pixel format，数值才与 eval 对得上）；
    // 驱动不支持时退到 64x48 的 pbuffer —— 真正的渲染目标是 mjr 自己的离屏 FBO
    // （尺寸 = 模型 vis.global.offwidth/offheight，默认 640x480，放 64x48 的矩形绰绰有余）。
    if (!eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context_))
    {
        const EGLint pbuffer_attrs[] = {EGL_WIDTH, kWidth, EGL_HEIGHT, kHeight, EGL_NONE};
        egl_surface_ = eglCreatePbufferSurface(egl_display_, config, pbuffer_attrs);
        if (egl_surface_ == EGL_NO_SURFACE ||
            !eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
        {
            error_ = "EGL 上下文无法 make current";
            return false;
        }
    }
    return true;
}

inline bool DepthCamera::Init(mjModel* m, bool gl_context_already_current,
                              const char* camera_name)
{
    if (initialized_)
    {
        return true;
    }
    if (m == nullptr)
    {
        error_ = "mjModel 为空";
        return false;
    }
    error_.clear();

    camera_id_ = mj_name2id(m, mjOBJ_CAMERA, camera_name);
    if (camera_id_ < 0)
    {
        error_ = std::string("模型里没有名为 \"") + camera_name + "\" 的相机";
        return false;
    }

    // 视锥近/远端 = vis.map.znear/zfar * stat.extent（米），与 renderer.py 同一口径。
    // 反投影：标准 GL 透视矩阵 c0 = -(f+n)/(f-n)、d0 = -2fn/(f-n)；reverse-Z 再做
    // c = -c0/2 - 1/2、d = -d0/2，于是 metric = d/(ndc + c)（ndc=1 → 近，ndc=0 → 远）。
    const float znear = static_cast<float>(m->vis.map.znear * m->stat.extent);
    const float zfar = static_cast<float>(m->vis.map.zfar * m->stat.extent);
    const float c = -(zfar + znear) / (zfar - znear);
    const float d = -(2.0f * zfar * znear) / (zfar - znear);
    c_coef_ = -0.5f * c - 0.5f;
    d_coef_ = -0.5f * d;

    if (!gl_context_already_current && !InitEgl())
    {
        Close(); // 清掉可能已经建了一半的 EGL 资源
        return false;
    }

    // 场景/选项：只画 geom group 0（地形/障碍物），见文件头。
    mjv_defaultScene(&scene_);
    mjv_makeScene(m, &scene_, kMaxGeom);
    scene_made_ = true;
    mjv_defaultOption(&option_);
    for (int i = 0; i < mjNGROUP; ++i)
    {
        option_.geomgroup[i] = 0;
    }
    option_.geomgroup[0] = 1;

    mjv_defaultCamera(&camera_);
    camera_.type = mjCAMERA_FIXED;
    camera_.fixedcamid = camera_id_;

    // 自己的 mjr 上下文（与窗口/其他渲染器互不干扰）；fontscale 用 Python Renderer 的默认值。
    mjr_defaultContext(&context_);
    mjr_makeContext(m, &context_, mjFONTSCALE_150);
    context_made_ = true;
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    context_.readDepthMap = mjDEPTH_ZEROFAR; // 默认是 ZERONEAR；不改这里换算全错

    raw_.assign(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    initialized_ = true;
    return true;
}

inline bool DepthCamera::Render(const mjModel* m, const mjData* d, std::vector<float>* depth)
{
    if (!initialized_ || m == nullptr || d == nullptr || depth == nullptr)
    {
        return false;
    }
    depth->resize(static_cast<std::size_t>(kWidth) * kHeight);

    // mjv_updateScene 的签名是 mjData*（MuJoCo 的历史原因；它只读 data，不改），这里去掉 const。
    mjv_updateScene(m, const_cast<mjData*>(d), &option_, nullptr, &camera_, mjCAT_ALL, &scene_);

    // 深度路径的两个场景标志（与 mujoco.Renderer.enable_depth_rendering 内部一样，在 render
    // 之前设）：mjRND_SEGMENT 让分段渲染走"远处精度更好"的那条路，IDCOLOR 是它的配套开关。
    scene_.flags[mjRND_SEGMENT] = 1;
    scene_.flags[mjRND_IDCOLOR] = 1;

    // 窗口模式下本渲染器与 viewer 共享同一个 GL 上下文：先清掉别的渲染器可能留下的 GL error
    // （调用有上限，避免上下文丢失时死循环），再把本类的离屏缓冲设成当前 buffer。
    for (int i = 0; i < 8 && mjr_getError() != 0; ++i)
    {
    }
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    mjr_render(rect_, &scene_, &context_);
    mjr_readPixels(nullptr, raw_.data(), rect_, &context_);
    if (!owns_egl_)
    {
        mjr_setBuffer(mjFB_WINDOW, &context_); // 把默认帧缓冲还给窗口（viewer 下一帧直接画它）
    }
    // 失败检测：headless（EGL 自建上下文）路径的 GL error 干净，可以直接判 render 失败；
    // 窗口模式复用 GLFW 上下文时，本机 GL 栈会给 mjr_render 留一个固定的 0x502 噪音
    // （repo 自己的 viewer 每帧也一样，实测是既有现象），不能据此判失败，只做下面的数值自检。
    if (owns_egl_ && mjr_getError() != 0)
    {
        error_ = "深度渲染的 GL 调用报错";
        return false;
    }

    // reverse-Z NDC → 米：系数是 float32、除法用 float64（与 renderer.py 的 astype 顺序一致）；
    // 同时把 GL 的自下而上行序翻成消息契约的自上而下（row 0 = 图像顶部）。
    // 非有限值 = readPixels/换算没给出可用深度（失败就返回 false，调用方跳过这一帧）。
    bool all_finite = true;
    for (int y = 0; y < kHeight; ++y)
    {
        for (int x = 0; x < kWidth; ++x)
        {
            const float ndc = raw_[static_cast<std::size_t>(y) * kWidth + x];
            const double metres = static_cast<double>(d_coef_) /
                                  (static_cast<double>(ndc) + static_cast<double>(c_coef_));
            all_finite = all_finite && std::isfinite(metres);
            (*depth)[static_cast<std::size_t>(kHeight - 1 - y) * kWidth + x] =
                static_cast<float>(metres);
        }
    }
    if (!all_finite)
    {
        error_ = "深度渲染结果里有非有限值";
        return false;
    }
    return true;
}

inline void DepthCamera::Close()
{
    if (context_made_)
    {
        if (owns_egl_ && egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT)
        {
            // mjr_freeContext 会删 GL 对象：必须在创建它的上下文 current 时调用。
            eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
        }
        mjr_freeContext(&context_);
        context_made_ = false;
    }
    if (scene_made_)
    {
        mjv_freeScene(&scene_);
        scene_made_ = false;
    }
    if (owns_egl_ && egl_display_ != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_surface_ != EGL_NO_SURFACE)
        {
            eglDestroySurface(egl_display_, egl_surface_);
        }
        if (egl_context_ != EGL_NO_CONTEXT)
        {
            eglDestroyContext(egl_display_, egl_context_);
        }
        eglTerminate(egl_display_);
        eglReleaseThread();
        egl_surface_ = EGL_NO_SURFACE;
        egl_context_ = EGL_NO_CONTEXT;
        egl_display_ = EGL_NO_DISPLAY;
    }
    initialized_ = false;
}

} // namespace quadruped::depth
