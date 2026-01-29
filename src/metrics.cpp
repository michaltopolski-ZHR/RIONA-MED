#include "metrics.h"

std::vector<std::vector<int>> InitMatrix(size_t d) {
    return std::vector<std::vector<int>>(d, std::vector<int>(d, 0));
}

std::vector<MetricsPerClass> ComputeMetrics(const std::vector<std::vector<int>>& conf) {
    const size_t d = conf.size();
    std::vector<MetricsPerClass> out(d);

    for (size_t i = 0; i < d; ++i) {
        int tp = conf[i][i];
        int fp = 0;
        int fn = 0;
        for (size_t r = 0; r < d; ++r) {
            if (r != i) {
                fp += conf[r][i];
                fn += conf[i][r];
            }
        }
        double precision = (tp + fp == 0) ? 0.0 : (double)tp / (double)(tp + fp);
        double recall = (tp + fn == 0) ? 0.0 : (double)tp / (double)(tp + fn);
        double f1 = (precision + recall == 0.0) ? 0.0 : (2.0 * precision * recall) / (precision + recall);
        out[i] = {precision, recall, f1};
    }
    return out;
}

MetricsPerClass ComputeBalanced(const std::vector<MetricsPerClass>& perClass) {
    MetricsPerClass bal;
    if (perClass.empty()) {
        return bal;
    }
    for (const auto& m : perClass) {
        bal.precision += m.precision;
        bal.recall += m.recall;
        bal.f1 += m.f1;
    }
    double d = static_cast<double>(perClass.size());
    bal.precision /= d;
    bal.recall /= d;
    bal.f1 /= d;
    return bal;
}