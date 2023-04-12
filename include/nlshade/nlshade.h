#pragma once

#include "types.h"

#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace nlshade {

class Optimizer {
public:
  Optimizer(fitness_function &&func, unsigned population_size, unsigned dim,
            int memory_size, double archive_size, int max_fevals,
            double lower_bounds = -100, double upper_bounds = 100,
            std::optional<unsigned> seed = std::nullopt);

  ~Optimizer();

  [[nodiscard]] optimization_result optimize();

private:
  void Initialize();
  optimization_result MainCycle();

  void FindNSaveBest(bool init, int ChosenOne);
  inline double GetValue(const int index, const int NInds, const int j);
  void CopyToArchive(double *RefusedParent);
  void SaveSuccessCrF(double Cr, double F, double FitD);
  void UpdateMemoryCrF();
  double MeanWL_general(double *Vector, double *TempWeights,
                        double g_p, double g_m);
  void RemoveWorst(int NInds, int NewNInds);

  fitness_function func_;

  unsigned population_size_;
  unsigned dim_;
  int memory_size_;
  double archive_size_;
  int max_fevals_;

  double lower_bounds_;
  double upper_bounds_;

  double best_so_far_{std::numeric_limits<double>::max()};
  bool best_so_far_initialized_{false};
  unsigned fe_num_{0};

  unsigned seed_;
  std::mt19937 random_generator_;

  std::vector<double> FitTemp3;
  bool FitNotCalculated;
  int Int_ArchiveSizeParam;
  int MemorySize;
  int MemoryIter;
  int SuccessFilled;
  int MemoryCurrentIndex;
  unsigned NVars;
  unsigned NInds;
  int NIndsMax;
  int NIndsMin;
  int besti;
  unsigned Generation;
  int ArchiveSize;
  int CurrentArchiveSize;
  double F;
  double Cr;
  double bestfit;
  double ArchiveSizeParam;
  int *Rands;
  int *Indexes;
  int *BackIndexes;
  double *Weights;
  double *Donor;
  double *Trial;
  double *FitMass;
  double *FitMassTemp;
  double *FitMassCopy;
  double *BestInd;
  double *tempSuccessCr;
  double *tempSuccessF;
  double *FGenerated;
  double *CrGenerated;
  double *MemoryCr;
  double *MemoryF;
  double *FitDelta;
  double *ArchUsages;
  double **Popul;
  double **PopulTemp;
  double **Archive;
};
} // namespace nlshade
