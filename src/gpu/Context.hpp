// SPDX-License-Identifier: Apache-2.0
// GPU context: Vulkan device selection (prefers devices with shaderFloat64),
// Kompute manager ownership, an algorithm cache and the kernel-dispatch
// helpers.
//
// Kompute's default Manager(index) path creates the logical device *without*
// any enabled features, which would forbid double precision. We therefore
// create the Vulkan instance / physical device / logical device ourselves and
// hand them over via the non-owning Manager( instance, pdev, device ) ctor.
#pragma once

// ---------------------------------------------------------------------------
// ABI pin. vk::DispatchLoaderBase (vulkan-hpp) contains a 16-byte debug-only
// guard (vkHeaderVersion + m_valid) when NDEBUG is *not* defined, so
// sizeof(kp::Manager) — which embeds a DispatchLoaderDynamic — depends on the
// NDEBUG state of the including TU. libkompute.a is always built with NDEBUG
// (see the Release pin in xmake.lua). To keep the layouts identical in every
// build mode we compile the Kompute headers with NDEBUG pinned, then restore
// the ambient state (assert() stays active in the rest of our code).
#pragma push_macro("NDEBUG")
#undef NDEBUG
#define NDEBUG
#include <kompute/Kompute.hpp>
#pragma pop_macro("NDEBUG")
// ---------------------------------------------------------------------------

#include <array>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SpvCode.hpp"

namespace kpalg {

/// Maximum number of int32 kernel parameters per dispatch.
inline constexpr std::size_t kMaxParams = 16;

/// Non-copyable, thread-unsafe singleton owning the Vulkan/Kompute context.
class Context final {
 public:
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  [[nodiscard]] static Context& instance() {
    static Context ctx;
    return ctx;
  }

  /// True when the selected device supports native/shader float64.
  [[nodiscard]] bool has_f64() const noexcept { return fp64_; }

  /// A scalar type usable by the GPU kernels on this device.
  using DefaultScalar = double;

  [[nodiscard]] const std::string& device_name() const noexcept {
    return device_name_;
  }

  [[nodiscard]] kp::Manager& kompute() noexcept { return *manager_; }

  [[nodiscard]] std::shared_ptr<kp::Sequence>& sequence() noexcept {
    return sequence_;
  }

  template <class T>
  [[nodiscard]] std::shared_ptr<kp::TensorT<T>> tensor(
      std::vector<T> data,
      kp::Tensor::TensorTypes type = kp::Tensor::TensorTypes::eDevice) {
    return manager_->tensorT<T>(std::move(data), type);
  }

  /// One recorded kernel dispatch.
  struct Step {
    spv::Kernel kernel;
    std::vector<std::shared_ptr<kp::Tensor>> bindings;
    kp::Workgroup wg;
    std::vector<std::int32_t> params;
  };

  /// Record-then-submit a single kernel dispatch in one queue submission.
  ///
  /// `bindings` are the kernel's SSBOs in binding order; the shared int32
  /// parameter buffer is appended automatically as the last binding.
  /// `uploads` are synced host->device and `downloads` device->host within
  /// the same submission, so no extra round-trips are needed.
  void dispatch(
      spv::Kernel kernel,
      const std::vector<std::shared_ptr<kp::Tensor>>& bindings,
      const kp::Workgroup& wg, std::span<const std::int32_t> params,
      std::vector<std::shared_ptr<kp::Tensor>> uploads = {},
      std::vector<std::shared_ptr<kp::Tensor>> downloads = {}) {
    if (params.size() > kMaxParams) {
      throw std::invalid_argument("kpalg::dispatch: too many kernel params");
    }
    // A zero-size dispatch is skipped entirely: it computes nothing, and
    // drivers are within their rights to behave arbitrarily for empty
    // launches. Any pending downloads still go out.
    if (wg[0] == 0 || wg[1] == 0 || wg[2] == 0) {
      download(std::move(downloads));
      return;
    }
    Step step{kernel, bindings, wg, {params.begin(), params.end()}};
    dispatch_seq({std::move(step)}, std::move(uploads), std::move(downloads));
  }

  /// Submit a chain of dependent dispatches as ONE queue submission.
  ///
  /// Kompute records no ShaderWrite->ShaderRead barrier between consecutive
  /// OpAlgoDispatch ops, so dependent kernels (pivot search -> row swap ->
  ... ) would race. We record an explicit memory barrier after every
  /// dispatch, and each step gets its own parameter buffer uploaded just
  /// before its dispatch. Steps whose workgroup is zero-sized are skipped.
  void dispatch_seq(
      const std::vector<Step>& steps,
      std::vector<std::shared_ptr<kp::Tensor>> uploads = {},
      std::vector<std::shared_ptr<kp::Tensor>> downloads = {}) {
    auto& seq = sequence_;
    if (!uploads.empty()) seq->record<kp::OpTensorSyncDevice>(std::move(uploads));

    for (std::size_t i = 0; i < steps.size(); ++i) {
      const Step& st = steps[i];
      if (st.wg[0] == 0 || st.wg[1] == 0 || st.wg[2] == 0) continue;
      check_wg(st.wg);
      auto params = params_slot(i);
      fill_params(*params, st.params);

      auto algo = get_algo(st.kernel, st.wg, st.bindings, params);
      seq->record<kp::OpTensorSyncDevice>({params});
      seq->record<kp::OpAlgoDispatch>(algo);
      // make the writes of this dispatch visible to the next one
      std::vector<std::shared_ptr<kp::Tensor>> touched = st.bindings;
      touched.push_back(params);
      seq->record<kp::OpMemoryBarrier>(
          touched, vk::AccessFlagBits::eShaderWrite,
          vk::AccessFlagBits::eShaderRead,
          vk::PipelineStageFlagBits::eComputeShader,
          vk::PipelineStageFlagBits::eComputeShader);
    }
    if (!downloads.empty()) seq->record<kp::OpTensorSyncLocal>(std::move(downloads));
    seq->eval();
  }

  /// Upload host data of `tensors` to the device (one submission).
  void upload(std::vector<std::shared_ptr<kp::Tensor>> tensors) {
    if (!tensors.empty()) {
      sequence_->eval<kp::OpTensorSyncDevice>(std::move(tensors));
    }
  }

  /// Download device data of `tensors` back to their host vectors.
  void download(std::vector<std::shared_ptr<kp::Tensor>> tensors) {
    if (!tensors.empty()) {
      sequence_->eval<kp::OpTensorSyncLocal>(std::move(tensors));
    }
  }

 private:
  Context() {
    // ---- instance -------------------------------------------------------
    vk::ApplicationInfo app{};
    app.setPApplicationName("kpalg")
        .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
        .setPEngineName("kpalg")
        .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
        .setApiVersion(VK_API_VERSION_1_1);
    instance_ = vk::createInstanceUnique(vk::InstanceCreateInfo{}.setPApplicationInfo(&app));

    // ---- physical device selection --------------------------------------
    const std::vector<vk::PhysicalDevice> devices =
        instance_->enumeratePhysicalDevices();
    if (devices.empty()) {
      throw std::runtime_error(
          "kpalg: no Vulkan-capable device found (is a driver installed?)");
    }

    std::size_t best = devices.size();
    int best_score = -1;
    for (std::size_t i = 0; i < devices.size(); ++i) {
      const vk::PhysicalDevice& dev = devices[i];
      const auto props = dev.getProperties();
      const bool has_compute =
          std::ranges::any_of(dev.getQueueFamilyProperties(), [](const auto& q) {
            return static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eCompute);
          });
      if (!has_compute) continue;
      const bool f64 = static_cast<bool>(dev.getFeatures().shaderFloat64);
      int score = f64 ? 4 : 0;
      switch (props.deviceType) {
        case vk::PhysicalDeviceType::eDiscreteGpu: score += 2; break;
        case vk::PhysicalDeviceType::eIntegratedGpu: score += 1; break;
        default: break;
      }
      if (score > best_score) {
        best_score = score;
        best = i;
        fp64_ = f64;
      }
    }
    if (best == devices.size()) {
      throw std::runtime_error("kpalg: no Vulkan device with a compute queue");
    }

    const vk::PhysicalDevice& pdev = devices[best];
    const auto props = pdev.getProperties();
    device_name_ = std::string(props.deviceName.data());
    limits_ = props.limits;

    // ---- logical device (compute queue, fp64 when supported) -------------
    std::uint32_t compute_family = 0;
    bool found = false;
    const auto families = pdev.getQueueFamilyProperties();
    for (std::uint32_t f = 0; f < families.size(); ++f) {
      if (families[f].queueFlags & vk::QueueFlagBits::eCompute) {
        compute_family = f;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("kpalg: selected device has no compute queue");
    }

    const float priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_info{};
    queue_info.setQueueFamilyIndex(compute_family).setQueueCount(1).setPQueuePriorities(&priority);

    vk::PhysicalDeviceFeatures features{};
    if (fp64_) features.setShaderFloat64(VK_TRUE);  // only enable if supported

    device_ = pdev.createDeviceUnique(
        vk::DeviceCreateInfo{}
            .setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&queue_info)
            .setPEnabledFeatures(&features));

    // ---- hand over to Kompute (Kompute must NOT own our handles) --------
    // Wrap the raw handles in non-owning shared_ptrs: the deleter only
    // releases the tiny handle-wrapper object, never the Vulkan resource
    // (vk::UniqueInstance / vk::UniqueDevice below keep ownership).
    manager_ = std::make_shared<kp::Manager>(
        std::shared_ptr<vk::Instance>(new vk::Instance(instance_.get()),
                                      [](vk::Instance* p) noexcept { delete p; }),
        std::make_shared<vk::PhysicalDevice>(pdev),
        std::shared_ptr<vk::Device>(new vk::Device(device_.get()),
                                    [](vk::Device* p) noexcept { delete p; }));

    sequence_ = manager_->sequence();
    params_pool_.push_back(manager_->tensorT<std::int32_t>(
        std::vector<std::int32_t>(kMaxParams, 0),
        kp::Tensor::TensorTypes::eDevice));
  }

  // ---- algorithm cache ----------------------------------------------------
  // Kompute's Manager::algorithm compiles the SPIR-V module and builds a
  // pipeline + descriptor sets on EVERY call, and destroying the returned
  // Algorithm tears them down again — thousands of create/destroy cycles per
  // factorization. That is both brutally slow and a relentless hammering of
  // the driver. We cache Algorithms keyed by (kernel, fp64, workgroup, bound
  // tensors). Entries pin their tensors (the descriptor sets reference raw
  // buffers), and the whole cache is dropped once it grows past kAlgoCacheMax
  // entries to bound pinned memory.
  struct AlgoKey {
    std::uint8_t kernel;
    bool f64;
    std::array<std::uint32_t, 3> wg;
    std::vector<const void*> tensors;  // includes the params buffer, last
    bool operator==(const AlgoKey& o) const noexcept {
      return kernel == o.kernel && f64 == o.f64 && wg == o.wg &&
             tensors == o.tensors;
    }
  };
  struct AlgoKeyHash {
    [[nodiscard]] std::size_t operator()(const AlgoKey& k) const noexcept {
      std::size_t h = std::hash<std::uint64_t>{}(
          (static_cast<std::uint64_t>(k.kernel) << 1) |
          static_cast<std::uint64_t>(k.f64));
      for (std::uint32_t w : k.wg) {
        h = h * 0x9E3779B97F4A7C15ull ^ std::hash<std::uint32_t>{}(w);
      }
      for (const void* p : k.tensors) {
        h = h * 0x9E3779B97F4A7C15ull ^ std::hash<const void*>{}(p);
      }
      return h;
    }
  };
  struct AlgoEntry {
    std::shared_ptr<kp::Algorithm> algo;
    std::vector<std::shared_ptr<kp::Tensor>> pinned;  // keeps buffers alive
  };
  static constexpr std::size_t kAlgoCacheMax = 128;
  std::unordered_map<AlgoKey, AlgoEntry, AlgoKeyHash> algo_cache_;

  [[nodiscard]] std::shared_ptr<kp::Algorithm> get_algo(
      spv::Kernel kernel, const kp::Workgroup& wg,
      const std::vector<std::shared_ptr<kp::Tensor>>& bindings,
      const std::shared_ptr<kp::TensorT<std::int32_t>>& params) {
    AlgoKey key{static_cast<std::uint8_t>(kernel), fp64_, {wg[0], wg[1], wg[2]}, {}};
    key.tensors.reserve(bindings.size() + 1);
    for (const auto& t : bindings) key.tensors.push_back(t.get());
    key.tensors.push_back(params.get());

    if (auto it = algo_cache_.find(key); it != algo_cache_.end()) {
      return it->second.algo;
    }
    std::vector<std::shared_ptr<kp::Tensor>> all = bindings;
    all.push_back(params);
    auto algo = manager_->algorithm(all, spv::code(kernel, fp64_), wg);
    if (algo_cache_.size() >= kAlgoCacheMax) {
      algo_cache_.clear();
    }
    return algo_cache_.emplace(std::move(key),
                               AlgoEntry{algo, std::move(all)})
        .first->second.algo;
  }

  // ---- parameter buffers ----------------------------------------------------
  // Position i of a dispatch chain always reuses the same int32 buffer, so
  // repeated calls with identical chains hit the algorithm cache. Chain
  // positions stay sequential in one command buffer, so each dispatch sees
  // its own parameter upload.
  std::vector<std::shared_ptr<kp::TensorT<std::int32_t>>> params_pool_;

  [[nodiscard]] std::shared_ptr<kp::TensorT<std::int32_t>> params_slot(
      std::size_t i) {
    while (params_pool_.size() <= i) {
      params_pool_.push_back(manager_->tensorT<std::int32_t>(
          std::vector<std::int32_t>(kMaxParams, 0),
          kp::Tensor::TensorTypes::eDevice));
    }
    return params_pool_[i];
  }

  static void fill_params(kp::TensorT<std::int32_t>& t,
                          std::span<const std::int32_t> v) {
    auto p = t.vector();  // staging buffer is always mapped
    std::ranges::fill(p, std::int32_t{0});
    std::ranges::copy(v, p.begin());
    t.setData(p);
  }

  // ---- driver-safety checks -------------------------------------------------
  /// Reject workgroup counts beyond the device limits instead of letting the
  /// driver hit undefined behaviour / device loss.
  void check_wg(const kp::Workgroup& wg) const {
    const auto& c = limits_.maxComputeWorkGroupCount;
    if (wg[0] > c[0] || wg[1] > c[1] || wg[2] > c[2]) {
      throw std::overflow_error(
          "kpalg: workgroup count exceeds device limits");
    }
  }

  vk::UniqueInstance instance_;
  vk::UniqueDevice device_;
  std::shared_ptr<kp::Manager> manager_;
  std::shared_ptr<kp::Sequence> sequence_;
  vk::PhysicalDeviceLimits limits_{};
  bool fp64_ = false;
  std::string device_name_;
};

/// Checked narrowing to the u32 used inside shaders.
[[nodiscard]] inline std::uint32_t u32(std::size_t v) {
  if (v > static_cast<std::size_t>(UINT32_MAX)) {
    throw std::overflow_error("kpalg: index does not fit into uint32");
  }
  return static_cast<std::uint32_t>(v);
}

}  // namespace kpalg
