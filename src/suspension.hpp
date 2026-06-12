#pragma once
#include "unit.hpp"
class SuspensionComponent {
  Length_Unit rest_length;                // m
  Force_D_Length_Unit stiffness;          // N/m
  Force_D_Length_Accel_Unit bump_damping; // N/(m/s)
  double rebound_damping;
  Force_D_Length_Unit anti_roll_stiffness; // N/m (连接对侧)

  Length_Unit current_length;
  Length_Unit prev_length;

public:
  Force_Unit compute_force(double opposite_compression, Time_Unit dt) {
    Length_Unit compression = rest_length - current_length;
    Speed_Unit speed = (current_length - prev_length) / dt;
    double damping = (speed > 0) ? bump_damping : rebound_damping;
    Force_Unit spring_force = compression * stiffness;
    Force_Unit damping_force = speed * damping;
    Force_Unit antiroll_force =
        anti_roll_stiffness * (compression - opposite_compression);
    Force_Unit total_force = spring_force + damping_force + antiroll_force;
    // 不钳位到非负，允许拉伸力
    return total_force;
  }
};