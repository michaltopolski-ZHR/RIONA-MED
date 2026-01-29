#include "algorithms.h"

#include <algorithm>

bool SatisfiesGRule(const Dataset& ds,
                    const Stats& stats,
                    const DistanceConfig& cfg,
                    const Instance& cand,
                    const Instance& tst,
                    const Instance& trn) {
    const size_t m = ds.types.size();
    for (size_t a = 0; a < m; ++a) {
        if (ds.types[a] == AttrType::Numeric) {
            const auto& vTst = tst.attrs[a];
            const auto& vTrn = trn.attrs[a];
            const auto& vCand = cand.attrs[a];

            // Missing values => attribute does not constrain the rule.
            if (vTst.missing || vTrn.missing || vCand.missing) {
                continue;
            }

            double lo = std::min(vTst.num, vTrn.num);
            double hi = std::max(vTst.num, vTrn.num);
            if (vCand.num < lo || vCand.num > hi) {
                return false;
            }
        } else {
            const auto& vTst = tst.attrs[a];
            const auto& vTrn = trn.attrs[a];
            const auto& vCand = cand.attrs[a];

            // Missing values => attribute does not constrain the rule.
            if (vTst.missing || vTrn.missing || vCand.missing) {
                continue;
            }

            const auto& ns = stats.nomStats[a];
            double r = NominalDistance(ns, vTst.raw, vTrn.raw, cfg);
            double d = NominalDistance(ns, vTst.raw, vCand.raw, cfg);
            if (d > r + 1e-12) {
                return false;
            }
        }
    }
    return true;
}

bool IsConsistentGRule(const Dataset& ds,
                       const Stats& stats,
                       const DistanceConfig& cfg,
                       const Instance& tst,
                       const Instance& trn,
                       const std::vector<int>& verifySet) {
    const std::string& decision = trn.decision;
    for (int idx : verifySet) {
        const auto& cand = ds.rows[idx];
        if (cand.decision != decision &&
            SatisfiesGRule(ds, stats, cfg, cand, tst, trn)) {
            return false;
        }
    }
    return true;
}

std::vector<Neighbor> ComputeNeighbors(const Dataset& ds,
                                       const Stats& stats,
                                       const DistanceConfig& cfg,
                                       const Instance& tst,
                                       const std::vector<int>& candidates,
                                       int k) {
    std::vector<Neighbor> neighbors;
    neighbors.reserve(candidates.size());

    for (int idx : candidates) {
        const auto& trn = ds.rows[idx];
        Neighbor nb;
        nb.index = idx;
        nb.dist = InstanceDistance(ds, stats, cfg, tst, trn);
        neighbors.push_back(nb);
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](const Neighbor& a, const Neighbor& b) {
                  if (a.dist != b.dist) return a.dist < b.dist;
                  return a.index < b.index;
              });

    if (k > (int)neighbors.size()) {
        k = static_cast<int>(neighbors.size());
    }
    neighbors.resize(k);
    return neighbors;
}

std::string ChooseClass(const Dataset& ds,
                        const std::vector<int>& supportCounts,
                        const std::vector<int>& classSizes,
                        bool normalized) {
    double bestScore = -1.0;
    int bestIdx = 0;

    for (size_t i = 0; i < supportCounts.size(); ++i) {
        double score = 0.0;
        if (normalized) {
            if (classSizes[i] > 0) {
                score = static_cast<double>(supportCounts[i]) / classSizes[i];
            } else {
                score = 0.0;
            }
        } else {
            score = static_cast<double>(supportCounts[i]);
        }

        if (score > bestScore) {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        } else if (score == bestScore) {
            // tie-breaker: stable ordering by class label
            if (ds.decisionValues[i] < ds.decisionValues[bestIdx]) {
                bestIdx = static_cast<int>(i);
            }
        }
    }

    return ds.decisionValues[bestIdx];
}

std::vector<int> ComputeClassSizes(const Dataset& ds, const std::vector<int>& indices) {
    std::vector<int> sizes(ds.decisionValues.size(), 0);
    for (int idx : indices) {
        const auto& inst = ds.rows[idx];
        int cls = ds.decisionIndex.at(inst.decision);
        sizes[cls] += 1;
    }
    return sizes;
}

ClassificationResult ClassifyKPlusNN(const Dataset& ds,
                                     const DistanceConfig& cfg,
                                     const Stats& baseStats,
                                     const std::vector<int>& trainingIdx,
                                     int tstIdx,
                                     int k,
                                     int nLocal) {
    const auto& tst = ds.rows[tstIdx];

    // Step 1: pick N(x, nLocal) using base (global or local) distance.
    if (nLocal < k) {
        nLocal = k;
    }
    if (nLocal > (int)trainingIdx.size()) {
        nLocal = static_cast<int>(trainingIdx.size());
    }

    std::vector<Neighbor> neighborsN = ComputeNeighbors(ds, baseStats, cfg, tst, trainingIdx, nLocal);
    std::vector<int> nIdx;
    nIdx.reserve(neighborsN.size());
    for (const auto& nb : neighborsN) {
        nIdx.push_back(nb.index);
    }

    // Step 2: induce local SVDM on N(x, nLocal)
    Stats localStats = ComputeStats(ds, nIdx, cfg);

    // Step 3: choose k nearest neighbors using local SVDM.
    std::vector<Neighbor> neighborsK = ComputeNeighbors(ds, localStats, cfg, tst, nIdx, k);

    // Support counts for standard/normalized decisions.
    std::vector<int> support(ds.decisionValues.size(), 0);
    for (const auto& nb : neighborsK) {
        int cls = ds.decisionIndex.at(ds.rows[nb.index].decision);
        support[cls] += 1;
    }
    std::vector<int> classSizes = ComputeClassSizes(ds, trainingIdx);

    ClassificationResult res;
    res.predictedStandard = ChooseClass(ds, support, classSizes, false);
    res.predictedNormalized = ChooseClass(ds, support, classSizes, true);
    res.knnList = std::move(neighborsK);
    return res;
}

ClassificationResult ClassifyRIA(const Dataset& ds,
                                 const DistanceConfig& cfg,
                                 const Stats& stats,
                                 const std::vector<int>& trainingIdx,
                                 int tstIdx,
                                 int kForReport) {
    const auto& tst = ds.rows[tstIdx];

    std::vector<int> support(ds.decisionValues.size(), 0);

    // For each training example: check if g-rule is consistent with the whole training set.
    for (int idx : trainingIdx) {
        const auto& trn = ds.rows[idx];
        if (IsConsistentGRule(ds, stats, cfg, tst, trn, trainingIdx)) {
            int cls = ds.decisionIndex.at(trn.decision);
            support[cls] += 1;
        }
    }

    std::vector<int> classSizes = ComputeClassSizes(ds, trainingIdx);

    ClassificationResult res;
    res.predictedStandard = ChooseClass(ds, support, classSizes, false);
    res.predictedNormalized = ChooseClass(ds, support, classSizes, true);

    // For the kNN output file we still provide k nearest neighbors.
    res.knnList = ComputeNeighbors(ds, stats, cfg, tst, trainingIdx, kForReport);
    return res;
}

ClassificationResult ClassifyRIONA(const Dataset& ds,
                                   const DistanceConfig& cfg,
                                   const Stats& stats,
                                   const std::vector<int>& trainingIdx,
                                   int tstIdx,
                                   int k) {
    const auto& tst = ds.rows[tstIdx];

    // Neighborhood N(tst, k)
    std::vector<Neighbor> neighbors = ComputeNeighbors(ds, stats, cfg, tst, trainingIdx, k);
    std::vector<int> nIdx;
    nIdx.reserve(neighbors.size());
    for (const auto& nb : neighbors) {
        nIdx.push_back(nb.index);
    }

    std::vector<int> support(ds.decisionValues.size(), 0);

    // For each neighbor, check g-rule consistency with the neighborhood.
    for (int idx : nIdx) {
        const auto& trn = ds.rows[idx];
        if (IsConsistentGRule(ds, stats, cfg, tst, trn, nIdx)) {
            int cls = ds.decisionIndex.at(trn.decision);
            support[cls] += 1;
        }
    }

    std::vector<int> classSizes = ComputeClassSizes(ds, trainingIdx);

    ClassificationResult res;
    res.predictedStandard = ChooseClass(ds, support, classSizes, false);
    res.predictedNormalized = ChooseClass(ds, support, classSizes, true);
    res.knnList = std::move(neighbors);
    return res;
}