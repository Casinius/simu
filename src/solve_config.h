#include "solve.h"
#include "solve.cxx"
#include "boost/math/cstdfloat/cstdfloat_types.hpp"
#pragma once
using Scalar = double;
template class RK45Solver<Scalar>;
template class BDF2Solver<Scalar>;
template class IRK2Solver<Scalar>;
template class AdaptiveBDF2Solver<Scalar>;
