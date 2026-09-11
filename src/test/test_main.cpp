// 测试入口：注册的所有套件统一执行，任一断言失败返回非 0。
#include "test_util.hpp"

#include <print>

int main() { return xtest::run_all(); }
