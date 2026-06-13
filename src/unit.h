#include <units.h>
#pragma once
// 角速度 (rad/s)
using radians_per_second = units::compound_unit<
    units::angle::radians,
    units::inverse<units::time::seconds>
>;
using RSpeed_Unit = units::unit_t<units::angular_velocity::rpm, double>;

// 角加速度 (rad/s²)
using radians_per_second_squared = units::compound_unit<
    units::angle::radians,
    units::inverse<units::squared<units::time::seconds>>
>;
using RAccelSpeed_Unit = units::unit_t<radians_per_second_squared, double>;

// 转动惯量 (kg·m²)
using kilogram_meter_squared = units::compound_unit<
    units::mass::kilogram,
    units::squared<units::length::meter>
>;
using Inertia_Unit = units::unit_t<kilogram_meter_squared, double>;

// 阻尼 (N·m·s/rad) 或 (N·m)/(rad/s) ，此处简化：扭矩/角速度
using newton_meter_per_rad_per_second = units::compound_unit<
    units::torque::newton_meter,
    units::inverse<radians_per_second>
>;
using Damp_Unit = units::unit_t<newton_meter_per_rad_per_second, double>;

// 扭矩 × 长度 (N·m²) - 很少用，按需定义
using newton_meter_squared = units::compound_unit<
    units::torque::newton_meter,
    units::length::meter
>;
using Force_D_Length_Unit = units::unit_t<newton_meter_squared, double>;

// 扭矩 × 长度 × 角加速度 (N·m²·rad/s²)
using Force_D_Length_Accel_Unit = units::unit_t<
    units::compound_unit<newton_meter_squared, radians_per_second_squared>
>;

using LinearForce_Unit = units::force::newton_t;

using Torque_Unit = units::unit_t<units::torque::newton_meter, double>;
using Time_Unit = units::time::second_t;
using Length_Unit = units::length::meter_t;
using Speed_Unit = units::velocity::meters_per_second_t;