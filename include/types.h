#pragma once

#include <functional>
#include <vector>

namespace nlshade {

using fitness_function = std::function<double(const double *, std::size_t)>;

struct optimization_result {
  const double best_fitness;
  std::vector<double> best_par;
  const unsigned generation_num;
  const unsigned func_eval;
};
} // namespace nlshade
