#include "util.h"
#include <random>

namespace nlshade::detail {

int IntRandom(int target) {
  if (target == 0)
    return 0;
  return uni_int(generator_uni_i) % target;
}

double Random(double minimal, double maximal) {
  return uni_real(generator_uni_r) * (maximal - minimal) + minimal;
}

double NormRand(double mu, double sigma) {
  return norm_dist(generator_norm) * sigma + mu;
}

double CachyRand(double mu, double sigma) {
  return cachy_dist(generator_cachy) * sigma + mu;
}

void qSort1(double *Mass, int low, int high) {
  int i = low;
  int j = high;
  double x = Mass[(low + high) >> 1];
  do {
    while (Mass[i] < x)
      ++i;
    while (Mass[j] > x)
      --j;
    if (i <= j) {
      double temp = Mass[i];
      Mass[i] = Mass[j];
      Mass[j] = temp;
      i++;
      j--;
    }
  } while (i <= j);
  if (low < j)
    qSort1(Mass, low, j);
  if (i < high)
    qSort1(Mass, i, high);
}

void qSort2int(double *Mass, int *Mass2, int low, int high) {
  int i = low;
  int j = high;
  double x = Mass[(low + high) >> 1];
  do {
    while (Mass[i] < x)
      ++i;
    while (Mass[j] > x)
      --j;
    if (i <= j) {
      double temp = Mass[i];
      Mass[i] = Mass[j];
      Mass[j] = temp;
      int temp2 = Mass2[i];
      Mass2[i] = Mass2[j];
      Mass2[j] = temp2;
      i++;
      j--;
    }
  } while (i <= j);
  if (low < j)
    qSort2int(Mass, Mass2, low, j);
  if (i < high)
    qSort2int(Mass, Mass2, i, high);
}

void GenerateNextRandUnif(const int num, const int Range, int *Rands,
                          const int Prohib) {
  for (int j = 0; j != 25; j++) {
    bool generateagain = false;
    Rands[num] = IntRandom(Range);
    for (int i = 0; i != num; i++)
      if (Rands[i] == Rands[num])
        generateagain = true;
    if (!generateagain)
      break;
  }
}

void GenerateNextRandUnifOnlyArch(const int num, const int Range,
                                  const int Range2, int *Rands,
                                  const int Prohib) {
  for (int j = 0; j != 25; j++) {
    bool generateagain = false;
    Rands[num] = IntRandom(Range2) + Range;
    for (int i = 0; i != num; i++)
      if (Rands[i] == Rands[num])
        generateagain = true;
    if (!generateagain)
      break;
  }
}

bool CheckGenerated(const int num, int *Rands, const int Prohib) {
  if (Rands[num] == Prohib)
    return false;
  for (int j = 0; j != num; j++)
    if (Rands[j] == Rands[num])
      return false;
  return true;
}

void FindLimits(double *Ind, double *Parent, int CurNVars, double CurLeft,
                double CurRight) {
  for (int j = 0; j < CurNVars; j++) {
    if (Ind[j] < CurLeft)
      Ind[j] = Random(CurLeft, CurRight);
    if (Ind[j] > CurRight)
      Ind[j] = Random(CurLeft, CurRight);
  }
}
} // namespace nlshade
