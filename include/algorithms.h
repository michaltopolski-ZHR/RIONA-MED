#pragma once

#include "dataset.h"
#include "distance.h"

#include <string>
#include <vector>

bool SatisfiesGRule(const Dataset& ds,
                    const Stats& stats,
                    const DistanceConfig& cfg,
                    const Instance& cand,
                    const Instance& tst,
                    const Instance& trn);

bool IsConsistentGRule(const Dataset& ds,
                       const Stats& stats,
                       const DistanceConfig& cfg,
                       const Instance& tst,
                       const Instance& trn,
                       const std::vector<int>& verifySet);

std::vector<Neighbor> ComputeNeighbors(const Dataset& ds,
                                       const Stats& stats,
                                       const DistanceConfig& cfg,
                                       const Instance& tst,
                                       const std::vector<int>& candidates,
                                       int k);

std::string ChooseClass(const Dataset& ds,
                        const std::vector<int>& supportCounts,
                        const std::vector<int>& classSizes,
                        bool normalized);

std::vector<int> ComputeClassSizes(const Dataset& ds, const std::vector<int>& indices);

ClassificationResult ClassifyKPlusNN(const Dataset& ds,
                                     const DistanceConfig& cfg,
                                     const Stats& baseStats,
                                     const std::vector<int>& trainingIdx,
                                     int tstIdx,
                                     int k,
                                     int nLocal);

ClassificationResult ClassifyRIA(const Dataset& ds,
                                 const DistanceConfig& cfg,
                                 const Stats& stats,
                                 const std::vector<int>& trainingIdx,
                                 int tstIdx,
                                 int kForReport);

ClassificationResult ClassifyRIONA(const Dataset& ds,
                                   const DistanceConfig& cfg,
                                   const Stats& stats,
                                   const std::vector<int>& trainingIdx,
                                   int tstIdx,
                                   int k);