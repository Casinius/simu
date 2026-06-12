#include "tire.hpp"
#include <cstdio>
int main(int argc, char **argv) {
  PacejkaParameters p;
  PacejkaTire tire(p);
  tire.set_normal_load(10000);
  tire.set_road_mu(1.2);
  tire.set_slip(0.1, 0.2);
  std::printf("%f",tire.compute().Fx);
    return 0;
}
