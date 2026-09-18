// greedy_color_cells 单元测试：颜色类划分的完备性与独立性。
#include "test_util.hpp"

#include <algorithm>
#include <vector>

namespace {

// 复刻 main 的棱柱布局：k_segs 段，每段 prism 拆 3 个四面体
std::vector<std::array<size_t, 4>> prism_cells(int k_segs) {
  constexpr int tets[3][4] = {{0, 1, 2, 4}, {0, 2, 4, 5}, {0, 4, 5, 3}};
  std::vector<std::array<size_t, 4>> cells;
  for (int seg = 0; seg < k_segs; ++seg)
    for (const auto &t : tets) {
      const int b = 3 * seg, n = 3 * (seg + 1);
      auto vidx = [&](int c) {
        return size_t(c < 3 ? b + c : n + (c - 3));
      };
      cells.push_back({vidx(t[0]), vidx(t[1]), vidx(t[2]), vidx(t[3])});
    }
  return cells;
}

void check_coloring(const std::vector<std::array<size_t, 4>> &cells) {
  const auto classes = xpbd::FP_XPBD::greedy_color_cells<4>(cells);

  // 相邻段共享截面顶点 ⇒ 至少两种颜色
  CHECK(classes.size() > 1);

  // (a) 颜色类并集 == 全部 cell 恰好一次
  std::vector<int> seen(cells.size(), 0);
  for (const auto &cls : classes)
    for (auto c : cls) {
      CHECK_MSG(c < cells.size() && seen[c] == 0, "cell {} 越界或重复染色", c);
      if (c < cells.size())
        ++seen[c];
    }
  CHECK_MSG(std::ranges::all_of(seen, [](int s) { return s == 1; }),
            "并非每个 cell 恰好着色一次");

  // (b) 同色类内任意两 cell 顶点集不相交
  for (const auto &cls : classes)
    for (size_t i = 0; i < cls.size(); ++i)
      for (size_t j = i + 1; j < cls.size(); ++j) {
        const auto &a = cells[cls[i]];
        const auto &b = cells[cls[j]];
        const bool disjoint = std::ranges::none_of(a, [&](size_t v) {
          return std::ranges::find(b, v) != b.end();
        });
        CHECK_MSG(disjoint, "同色 cell {} 与 {} 共享顶点", cls[i], cls[j]);
      }
}

} // namespace

XTEST_SUITE(coloring) {
  check_coloring(prism_cells(2));  // 2 段 prism
  check_coloring(prism_cells(10)); // 主 main 的 K=10 布局
}
