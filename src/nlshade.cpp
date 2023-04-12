#include "nlshade/nlshade.h"
#include "util.h"

#include <cmath>


namespace nlshade {

Optimizer::Optimizer(fitness_function &&func, unsigned population_size,
                     unsigned dim, int memory_size, double archive_size,
                     int max_fevals, double lower_bounds, double upper_bounds,
                     std::optional<unsigned> seed)
    : func_{std::move(func)}, population_size_{population_size}, dim_{dim},
      memory_size_{memory_size}, archive_size_{archive_size},
      max_fevals_{max_fevals}, lower_bounds_{lower_bounds}, upper_bounds_{
                                                                upper_bounds} {
  Initialize();
  if (seed.has_value()) {
    random_generator_.seed(seed_);
  } else {
    std::random_device rd;
    random_generator_.seed(rd());
  }
};

optimization_result Optimizer::optimize() { return MainCycle(); }

Optimizer::~Optimizer() {
  delete Donor;
  delete Trial;
  delete Rands;
  for (int i = 0; i != NIndsMax; i++) {
    delete Popul[i];
    delete PopulTemp[i];
  }
  for (int i = 0; i != NIndsMax * Int_ArchiveSizeParam; i++)
    delete Archive[i];
  delete ArchUsages;
  delete Archive;
  delete Popul;
  delete PopulTemp;
  delete FitMass;
  delete FitMassTemp;
  delete FitMassCopy;
  delete BestInd;
  delete Indexes;
  delete BackIndexes;
  delete tempSuccessCr;
  delete tempSuccessF;
  delete FGenerated;
  delete CrGenerated;
  delete FitDelta;
  delete MemoryCr;
  delete MemoryF;
  delete Weights;
}

void Optimizer::Initialize() {
  FitNotCalculated = true;
  NInds = population_size_;
  NIndsMax = static_cast<int>(NInds);
  NIndsMin = 4;
  NVars = dim_;
  Cr = 0.2;
  F = 0.2;
  besti = 0;
  Generation = 0;
  CurrentArchiveSize = 0;
  ArchiveSizeParam = archive_size_;
  Int_ArchiveSizeParam = static_cast<int>(std::ceil(archive_size_));
  ArchiveSize = static_cast<int>(NIndsMax) * static_cast<int>(archive_size_);
  Popul = new double *[static_cast<std::size_t>(NIndsMax)];
  for (auto i = 0; i != NIndsMax; i++)
    Popul[i] = new double[NVars];
  PopulTemp = new double *[static_cast<std::size_t>(NIndsMax)];
  for (auto i = 0; i != NIndsMax; i++)
    PopulTemp[i] = new double[NVars];
  Archive =
      new double *[static_cast<std::size_t>(NIndsMax * Int_ArchiveSizeParam)];
  for (auto i = 0;
       i != NIndsMax * Int_ArchiveSizeParam; i++)
    Archive[i] = new double[NVars];
  FitMass = new double[static_cast<std::size_t>(NIndsMax)];
  FitMassTemp = new double[static_cast<std::size_t>(NIndsMax)];
  FitMassCopy = new double[static_cast<std::size_t>(NIndsMax)];
  Indexes = new int[static_cast<std::size_t>(NIndsMax)];
  BackIndexes = new int[static_cast<std::size_t>(NIndsMax)];
  BestInd = new double[NVars];
  for (auto i = 0; i < NIndsMax; i++)
    for (std::size_t j = 0; j < NVars; j++)
      Popul[i][j] = detail::Random(lower_bounds_, upper_bounds_);
  Donor = new double[NVars];
  Trial = new double[NVars];
  Rands = new int[static_cast<std::size_t>(NIndsMax)];
  tempSuccessCr = new double[static_cast<std::size_t>(NIndsMax)];
  tempSuccessF = new double[static_cast<std::size_t>(NIndsMax)];
  FitDelta = new double[static_cast<std::size_t>(NIndsMax)];
  FGenerated = new double[static_cast<std::size_t>(NIndsMax)];
  CrGenerated = new double[static_cast<std::size_t>(NIndsMax)];
  for (auto i = 0; i != NIndsMax; i++) {
    tempSuccessCr[i] = 0;
    tempSuccessF[i] = 0;
  }
  MemorySize = memory_size_;
  MemoryIter = 0;
  SuccessFilled = 0;
  ArchUsages = new double[static_cast<std::size_t>(NIndsMax)];
  Weights = new double[static_cast<std::size_t>(NIndsMax)];
  MemoryCr = new double[static_cast<std::size_t>(MemorySize)];
  MemoryF = new double[static_cast<std::size_t>(MemorySize)];
  for (int i = 0; i != MemorySize; i++) {
    MemoryCr[i] = 0.2;
    MemoryF[i] = 0.2;
  }
}

void Optimizer::SaveSuccessCrF(double Cr, double F, double FitD) {
  tempSuccessCr[SuccessFilled] = Cr;
  tempSuccessF[SuccessFilled] = F;
  FitDelta[SuccessFilled] = FitD;
  SuccessFilled++;
}

void Optimizer::UpdateMemoryCrF() {
  if (SuccessFilled != 0) {
    MemoryCr[MemoryIter] =
        MeanWL_general(tempSuccessCr, FitDelta, 2, 1);
    MemoryF[MemoryIter] =
        MeanWL_general(tempSuccessF, FitDelta, 2, 1);
    MemoryIter++;
    if (MemoryIter >= MemorySize)
      MemoryIter = 0;
  } else {
    MemoryF[MemoryIter] = 0.5;
    MemoryCr[MemoryIter] = 0.5;
  }
}

double Optimizer::MeanWL_general(double *Vector, double *TempWeights,
                                 double g_p, double g_m) {
  double SumWeight = 0;
  double SumSquare = 0;
  double Sum = 0;
  for (int i = 0; i != SuccessFilled; i++)
    SumWeight += TempWeights[i];
  for (int i = 0; i != SuccessFilled; i++)
    Weights[i] = TempWeights[i] / SumWeight;
  for (int i = 0; i != SuccessFilled; i++)
    SumSquare += Weights[i] * pow(Vector[i], g_p);
  for (int i = 0; i != SuccessFilled; i++)
    Sum += Weights[i] * pow(Vector[i], g_p - g_m);
  if (fabs(Sum) > 0.000001)
    return SumSquare / Sum;
  else
    return 0.5;
}

void Optimizer::CopyToArchive(double *RefusedParent) {
  if (CurrentArchiveSize < ArchiveSize) {
    for (std::size_t i = 0; i != NVars; i++)
      Archive[CurrentArchiveSize][i] = RefusedParent[i];
    CurrentArchiveSize++;
  } else if (ArchiveSize > 0) {
    int RandomNum = detail::IntRandom(ArchiveSize);
    for (std::size_t i = 0; i != NVars; i++)
      Archive[RandomNum][i] = RefusedParent[i];
  }
}

void Optimizer::FindNSaveBest(bool init, int ChosenOne) {
  if (FitMass[ChosenOne] <= bestfit || init) {
    bestfit = FitMass[ChosenOne];
    besti = ChosenOne;
    for (std::size_t j = 0; j != NVars; j++)
      BestInd[j] = Popul[besti][j];
  }
  if (bestfit < best_so_far_)
    best_so_far_ = bestfit;
}

void Optimizer::RemoveWorst(int NInds, int NewNInds) {
  int PointsToRemove = NInds - NewNInds;
  for (int L = 0; L != PointsToRemove; L++) {
    double WorstFit = FitMass[0];
    int WorstNum = 0;
    for (int i = 1; i != NInds; i++) {
      if (FitMass[i] > WorstFit) {
        WorstFit = FitMass[i];
        WorstNum = i;
      }
    }
    for (auto i = WorstNum; i != NInds - 1; i++) {
      for (std::size_t j = 0; j != NVars; j++)
        Popul[i][j] = Popul[i + 1][j];
      FitMass[i] = FitMass[i + 1];
    }
  }
}

inline double Optimizer::GetValue(const int index, const int NInds,
                                  const int j) {
  if (index < NInds)
    return Popul[index][j];
  return Archive[index - NInds][j];
}

optimization_result Optimizer::MainCycle() {
  double ArchSuccess;
  double NoArchSuccess;
  double NArchUsages;
  double ArchProbs = 0.5;
  for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
    FitMass[TheChosenOne] = func_(Popul[TheChosenOne], dim_);
    fe_num_++;
    FindNSaveBest(TheChosenOne == 0, TheChosenOne);
    if (!best_so_far_initialized_ || bestfit < best_so_far_) {
      best_so_far_ = bestfit;
      best_so_far_initialized_ = true;
    }
  }
  do {
    double minfit = FitMass[0];
    double maxfit = FitMass[0];
    for (int i = 0; i != NInds; i++) {
      FitMassCopy[i] = FitMass[i];
      Indexes[i] = i;
      if (FitMass[i] >= maxfit)
        maxfit = FitMass[i];
      if (FitMass[i] <= minfit)
        minfit = FitMass[i];
    }
    if (minfit != maxfit)
      detail::qSort2int(FitMassCopy, Indexes, 0, NInds - 1);
    for (std::size_t i = 0; i != NInds; i++)
      for (std::size_t j = 0; j != NInds; j++)
        if (i == Indexes[j]) {
          BackIndexes[i] = j;
          break;
        }
    FitTemp3.resize(NInds);
    for (int i = 0; i != NInds; i++)
      FitTemp3[i] = exp(-double(i) / (double)NInds);
    std::discrete_distribution<int> ComponentSelector3(FitTemp3.begin(),
                                                       FitTemp3.end());
    int psizeval =
        std::max(2.0, NInds * (0.2 / (double)max_fevals_ * (double)fe_num_ + 0.2));
    int CrossExponential = 0;
    if (detail::Random(0, 1) < 0.5)
      CrossExponential = 1;
    for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
      MemoryCurrentIndex = detail::IntRandom(MemorySize);
      Cr = std::min(1.0, std::max(0.0, detail::NormRand(MemoryCr[MemoryCurrentIndex], 0.1)));
      do {
        F = detail::CachyRand(MemoryF[MemoryCurrentIndex], 0.1);
      } while (F <= 0);
      FGenerated[TheChosenOne] = std::min(F, 1.0);
      CrGenerated[TheChosenOne] = Cr;
    }
    detail::qSort1(CrGenerated, 0, NInds - 1);

    for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
      Rands[0] = Indexes[detail::IntRandom(psizeval)];
      for (int i = 0; i != 25 && !detail::CheckGenerated(0, Rands, TheChosenOne); i++)
        Rands[0] = Indexes[detail::IntRandom(psizeval)];
      detail::GenerateNextRandUnif(1, NInds, Rands, TheChosenOne);
      if (detail::Random(0, 1) > ArchProbs || CurrentArchiveSize == 0) {
        Rands[2] = Indexes[ComponentSelector3(random_generator_)];
        for (int i = 0; i != 25 && !detail::CheckGenerated(2, Rands, TheChosenOne); i++)
          Rands[2] = Indexes[ComponentSelector3(random_generator_)];
        ArchUsages[TheChosenOne] = 0;
      } else {
        detail::GenerateNextRandUnifOnlyArch(2, NInds, CurrentArchiveSize, Rands,
                                     TheChosenOne);
        ArchUsages[TheChosenOne] = 1;
      }
      for (int j = 0; j != NVars; j++)
        Donor[j] = Popul[TheChosenOne][j] +
                   FGenerated[TheChosenOne] *
                       (GetValue(Rands[0], NInds, j) - Popul[TheChosenOne][j]) +
                   FGenerated[TheChosenOne] * (GetValue(Rands[1], NInds, j) -
                                               GetValue(Rands[2], NInds, j));

      int WillCrossover = detail::IntRandom(NVars);
      Cr = CrGenerated[BackIndexes[TheChosenOne]];
      double CrToUse = 0;
      if (fe_num_ > 0.5 * max_fevals_)
        CrToUse = (double(fe_num_) / double(max_fevals_) - 0.5) * 2;
      if (CrossExponential == 0) {
        for (int j = 0; j != NVars; j++) {
          if (detail::Random(0, 1) < CrToUse || WillCrossover == j)
            PopulTemp[TheChosenOne][j] = Donor[j];
          else
            PopulTemp[TheChosenOne][j] = Popul[TheChosenOne][j];
        }
      } else {
        int StartLoc = detail::IntRandom(NVars);
        int L = StartLoc + 1;
        while (detail::Random(0, 1) < Cr && L < NVars)
          L++;
        for (int j = 0; j != NVars; j++)
          PopulTemp[TheChosenOne][j] = Popul[TheChosenOne][j];
        for (int j = StartLoc; j != L; j++)
          PopulTemp[TheChosenOne][j] = Donor[j];
      }
      detail::FindLimits(PopulTemp[TheChosenOne], Popul[TheChosenOne], NVars,
                 lower_bounds_, upper_bounds_);

      FitMassTemp[TheChosenOne] = func_(PopulTemp[TheChosenOne], dim_);

      fe_num_++;
      if (FitMassTemp[TheChosenOne] <= best_so_far_)
        best_so_far_ = FitMassTemp[TheChosenOne];

      if (FitMassTemp[TheChosenOne] < FitMass[TheChosenOne])
        SaveSuccessCrF(Cr, F,
                       fabs(FitMass[TheChosenOne] - FitMassTemp[TheChosenOne]));
      FindNSaveBest(false, TheChosenOne);
    }
    ArchSuccess = 0;
    NoArchSuccess = 0;
    NArchUsages = 0;
    for (int TheChosenOne = 0; TheChosenOne != NInds; TheChosenOne++) {
      if (FitMassTemp[TheChosenOne] <= FitMass[TheChosenOne]) {
        if (ArchUsages[TheChosenOne] == 1) {
          ArchSuccess += (FitMass[TheChosenOne] - FitMassTemp[TheChosenOne]) /
                         FitMass[TheChosenOne];
          NArchUsages += 1;
        } else
          NoArchSuccess += (FitMass[TheChosenOne] - FitMassTemp[TheChosenOne]) /
                           FitMass[TheChosenOne];
        CopyToArchive(Popul[TheChosenOne]);
        for (int j = 0; j != NVars; j++)
          Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
        FitMass[TheChosenOne] = FitMassTemp[TheChosenOne];
      }
    }
    if (NArchUsages != 0) {
      ArchSuccess = ArchSuccess / NArchUsages;
      NoArchSuccess = NoArchSuccess / (NInds - NArchUsages);
      ArchProbs = ArchSuccess / (ArchSuccess + NoArchSuccess);
      ArchProbs = std::max(0.1, std::min(0.9, ArchProbs));
      if (ArchSuccess == 0)
        ArchProbs = 0.5;
    } else
      ArchProbs = 0.5;
    int newNInds =
        std::round((NIndsMin - NIndsMax) *
                       std::pow((double(fe_num_) / double(max_fevals_)),
                                (1.0 - double(fe_num_) / double(max_fevals_))) +
                   NIndsMax);
    if (newNInds < NIndsMin)
      newNInds = NIndsMin;
    if (newNInds > NIndsMax)
      newNInds = NIndsMax;
    int newArchSize =
        std::round((NIndsMin - NIndsMax) *
                       std::pow((double(fe_num_) / double(max_fevals_)),
                                (1.0 - double(fe_num_) / double(max_fevals_))) +
                   NIndsMax) *
        ArchiveSizeParam;
    if (newArchSize < NIndsMin)
      newArchSize = NIndsMin;
    ArchiveSize = newArchSize;
    if (CurrentArchiveSize >= ArchiveSize)
      CurrentArchiveSize = ArchiveSize;
    RemoveWorst(NInds, newNInds);
    NInds = newNInds;
    UpdateMemoryCrF();
    SuccessFilled = 0;
    Generation++;
  } while (fe_num_ < max_fevals_);

  return {.best_fitness = bestfit,
          .best_par = std::vector<double>(BestInd, BestInd + dim_),
          .generation_num = Generation,
          .func_eval = fe_num_};
}
} // namespace nlshade
