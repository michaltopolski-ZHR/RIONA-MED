#include "distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

Stats ComputeStats(const Dataset& ds,
                   const std::vector<int>& indices,
                   const DistanceConfig& distCfg) {
    const size_t m = ds.types.size();
    Stats stats;
    stats.numStats.resize(m);
    stats.nomStats.resize(m);

    // ---- Numeric stats: min, max, range ----
    for (size_t a = 0; a < m; ++a) {
        if (ds.types[a] != AttrType::Numeric) {
            continue;
        }
        NumericStat ns;
        ns.min = std::numeric_limits<double>::infinity();
        ns.max = -std::numeric_limits<double>::infinity();
        ns.hasValue = false;
        for (int idx : indices) {
            const auto& v = ds.rows[idx].attrs[a];
            if (v.missing) {
                continue;
            }
            ns.hasValue = true;
            ns.min = std::min(ns.min, v.num);
            ns.max = std::max(ns.max, v.num);
        }
        if (!ns.hasValue) {
            ns.min = ns.max = ns.range = 0.0;
        } else {
            ns.range = ns.max - ns.min;
        }
        stats.numStats[a] = ns;
    }

    // ---- Nominal stats: SVDM distance matrices ----
    // NOTE: attribute weights w_i are effectively 1.0 (no selection),
    // because the task does not specify a concrete weighting procedure.
    const size_t d = ds.decisionValues.size();
    for (size_t a = 0; a < m; ++a) {
        if (ds.types[a] != AttrType::Nominal) {
            continue;
        }

        // counts[value][class] and totals per value
        std::unordered_map<std::string, std::vector<int>> counts;
        std::unordered_map<std::string, int> totals;

        for (int idx : indices) {
            const auto& inst = ds.rows[idx];
            const auto& v = inst.attrs[a];
            if (v.missing) {
                continue;
            }
            auto& vec = counts[v.raw];
            if (vec.empty()) {
                vec.resize(d, 0);
            }
            int cls = ds.decisionIndex.at(inst.decision);
            vec[cls] += 1;
            totals[v.raw] += 1;
        }

        NominalStat ns;
        ns.values.reserve(counts.size());
        for (const auto& kv : counts) {
            ns.index[kv.first] = static_cast<int>(ns.values.size());
            ns.values.push_back(kv.first);
        }

        const size_t vcount = ns.values.size();
        ns.dist.assign(vcount, std::vector<double>(vcount, 0.0));

        for (size_t i = 0; i < vcount; ++i) {
            const std::string& valI = ns.values[i];
            for (size_t j = i; j < vcount; ++j) {
                const std::string& valJ = ns.values[j];
                double sum = 0.0;
                int totalI = totals[valI];
                int totalJ = totals[valJ];

                for (size_t c = 0; c < d; ++c) {
                    double pi = (totalI == 0) ? 0.0 : (double)counts[valI][c] / (double)totalI;
                    double pj = (totalJ == 0) ? 0.0 : (double)counts[valJ][c] / (double)totalJ;
                    sum += std::abs(pi - pj);
                }

                if (distCfg.svdmPrime) {
                    sum *= 0.5; // normalize to [0,1]
                }

                ns.dist[i][j] = ns.dist[j][i] = sum;
            }
        }

        stats.nomStats[a] = std::move(ns);
    }

    return stats;
}

// ---------------------------------------
// Distance calculation (global metric)
// ---------------------------------------

double NominalDistance(const NominalStat& ns,
                       const std::string& a,
                       const std::string& b,
                       const DistanceConfig& cfg) {
    auto itA = ns.index.find(a);
    auto itB = ns.index.find(b);
    if (itA == ns.index.end() || itB == ns.index.end()) {
        return cfg.missingNominal;
    }
    return ns.dist[itA->second][itB->second];
}

double InstanceDistance(const Dataset& ds,
                        const Stats& stats,
                        const DistanceConfig& cfg,
                        const Instance& x,
                        const Instance& y) {
    double sum = 0.0;
    const size_t m = ds.types.size();
    for (size_t a = 0; a < m; ++a) {
        if (ds.types[a] == AttrType::Numeric) {
            const auto& vx = x.attrs[a];
            const auto& vy = y.attrs[a];
            if (vx.missing || vy.missing) {
                sum += cfg.missingNumeric;
                continue;
            }
            const auto& ns = stats.numStats[a];
            if (!ns.hasValue || ns.range == 0.0) {
                sum += 0.0;
            } else {
                sum += std::abs(vx.num - vy.num) / ns.range;
            }
        } else {
            const auto& vx = x.attrs[a];
            const auto& vy = y.attrs[a];
            if (vx.missing || vy.missing) {
                sum += cfg.missingNominal;
                continue;
            }
            sum += NominalDistance(stats.nomStats[a], vx.raw, vy.raw, cfg);
        }
    }
    return sum;
}