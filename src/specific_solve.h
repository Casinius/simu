
// #include "solve_config.h"
#include "solve.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>

#ifndef SPAR_SOLVER_H
#define SPAR_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/LU>
#include <vector>
/**
 * 抽象求解器基类
 * 所有具体求解器必须实现 solve() 方法
 */
template <typename Scalar>
using State = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template<class Scalar,size_t n_element> struct ring_array : std::array<Scalar, n_element> {
    size_t default_set_position=0;
    void push_back(Scalar val){
        this->at(default_set_position) = val;
        if(default_set_position<n_element)
        this->default_set_position += 1;
        else
         this->default_set_position=0;
    }
};

struct TimedSolver{
    
};
#endif
