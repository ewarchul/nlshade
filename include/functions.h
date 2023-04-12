#pragma once

#include <cstddef>

namespace nlshade {

inline auto sphere_function(const double *input, std::size_t n) -> double {
  double result{0};
  for (std::size_t i = 0; i < n; ++i) {
    result += input[i] * input[i];
  }
  return result;
}
} // namespace nlshade
