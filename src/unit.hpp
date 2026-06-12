#pragma once
using Time_Unit = double;
using Force_Unit = double;
using RAccelSpeed_Unit = double;
using Speed_Unit = double;
using RSpeed_Unit = double;
using Inertia_Unit = double;
using Length_Unit = double;
using Force_D_Length_Unit = decltype(Force_Unit(1) / Length_Unit(1));
using Force_D_Length_Accel_Unit =
    decltype(Force_D_Length_Unit(1) * Time_Unit(1));