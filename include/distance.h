#pragma once

#include "dataset.h"

Stats ComputeStats(const Dataset& ds, const std::vector<int>& indices, const DistanceConfig& distCfg);
double NominalDistance(const NominalStat& ns, const std::string& a, const std::string& b, const DistanceConfig& cfg);
double InstanceDistance(const Dataset& ds, const Stats& stats, const DistanceConfig& cfg, const Instance& x, const Instance& y);