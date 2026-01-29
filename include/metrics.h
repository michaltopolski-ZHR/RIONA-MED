#pragma once

#include "dataset.h"

#include <vector>

std::vector<std::vector<int>> InitMatrix(size_t d);
std::vector<MetricsPerClass> ComputeMetrics(const std::vector<std::vector<int>>& conf);
MetricsPerClass ComputeBalanced(const std::vector<MetricsPerClass>& perClass);