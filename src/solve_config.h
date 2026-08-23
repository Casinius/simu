#include "solve.h"
#include "solve.cxx"
#pragma once
using Scalar = double;
template class RK45Solver<Scalar>;
template class BDF2Solver<Scalar>;
template class IRK2Solver<Scalar>;
template class SemiImplicitEulerSolver<Scalar>;
template class VerletSolver<Scalar>;