#include "functions.h"
#include "nlshade/nlshade.h"
#include "types.h"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iostream>
#include <vector>

auto print_optim_result(const nlshade::optimization_result &result) {
  fmt::print("---\n");
  fmt::print("Best fitness function = {}\nTotal generation number = {}\nTotal "
             "fitness function evals = {}\n",
             result.best_fitness, result.generation_num, result.func_eval);
  fmt::print("Best params = {}\n", result.best_par);
  fmt::print("---\n");
}

int main() {

  const auto dim = 10;
  const auto population_size = 30 * dim;
  const auto memory_size = 20 * dim;
  const auto archive_size = 2.1;
  const auto max_fevals = 100000;

  nlshade::Optimizer optim{nlshade::sphere_function,
                           population_size,
                           dim,
                           memory_size,
                           archive_size,
                           max_fevals};
  const auto results = optim.optimize();

  print_optim_result(results);

  return 0;
}
