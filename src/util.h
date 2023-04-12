#pragma once

#include <random>

namespace nlshade::detail {

static unsigned default_seed = 0;

static std::mt19937 generator_uni_i(default_seed);
static std::uniform_int_distribution<int> uni_int(0, 32768);

static std::mt19937 generator_uni_r(default_seed + 100);
static std::uniform_real_distribution<double> uni_real(0.0, 1.0);

static std::mt19937 generator_norm(default_seed + 200);
static std::normal_distribution<double> norm_dist(0.0, 1.0);

static std::mt19937 generator_cachy(default_seed + 300);
static std::cauchy_distribution<double> cachy_dist(0.0, 1.0);

int IntRandom(int target);

double Random(double minimal, double maximal);

double NormRand(double mu, double sigma);

double CachyRand(double mu, double sigma);

void qSort1(double *Mass, int low, int high);

void qSort2int(double *Mass, int *Mass2, int low, int high);

void GenerateNextRandUnif(const int num, const int Range, int *Rands,
                          const int Prohib);

void GenerateNextRandUnifOnlyArch(const int num, const int Range,
                                  const int Range2, int *Rands,
                                  const int Prohib);

bool CheckGenerated(const int num, int *Rands, const int Prohib);

void FindLimits(double *Ind, double *Parent, int CurNVars, double CurLeft,
                double CurRight);
} // namespace nlshade
