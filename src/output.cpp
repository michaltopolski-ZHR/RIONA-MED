#include "output.h"

#include <fstream>

void WriteOutFile(const std::string& path,
                  const Dataset& ds,
                  const std::vector<std::string>& predStd,
                  const std::vector<std::string>& predNorm,
                  const std::string& missingToken) {
    std::ofstream out(path);
    for (size_t i = 0; i < ds.rows.size(); ++i) {
        const auto& inst = ds.rows[i];
        out << inst.id;
        for (const auto& attr : inst.attrs) {
            out << ",";
            if (attr.missing) {
                out << missingToken;
            } else {
                out << attr.raw;
            }
        }
        out << "," << inst.decision
            << "," << predStd[i]
            << "," << predNorm[i] << "\n";
    }
}

void WriteKnnFile(const std::string& path,
                  const std::vector<std::vector<Neighbor>>& knnLists) {
    std::ofstream out(path);
    for (size_t i = 0; i < knnLists.size(); ++i) {
        const auto& list = knnLists[i];
        out << (i + 1) << "," << list.size();
        for (const auto& nb : list) {
            out << ",(" << (nb.index + 1) << "," << nb.dist << ")";
        }
        out << "\n";
    }
}

void WriteStatFile(const std::string& path,
                   const Dataset& ds,
                   const Stats& globalStats,
                   const std::string& inputFile,
                   const std::string& algo,
                   const std::string& mode,
                   const std::string& svdmLabel,
                   int k,
                   double timeReadMs,
                   double timePrepMs,
                   double timeClassifyMs,
                   double timeWriteMs,
                   double timeTotalMs,
                   const std::vector<std::vector<int>>& confStd,
                   const std::vector<std::vector<int>>& confNorm) {
    std::ofstream out(path);

    out << "InputFile: " << inputFile << "\n";
    out << "Attributes: " << ds.types.size() << "\n";
    out << "Objects: " << ds.rows.size() << "\n";
    out << "Algorithm: " << algo << "\n";
    out << "Mode: " << mode << "\n";
    out << "k: " << k << "\n";
    out << "NominalDistance: " << svdmLabel << "\n";
    out << "Times(ms): read=" << timeReadMs
        << ", preprocess=" << timePrepMs
        << ", classify=" << timeClassifyMs
        << ", write=" << timeWriteMs
        << ", total=" << timeTotalMs << "\n";

    out << "d (number of classes): " << ds.decisionValues.size() << "\n";
    out << "ClassCounts:";
    for (const auto& v : ds.decisionValues) {
        int count = 0;
        for (const auto& inst : ds.rows) {
            if (inst.decision == v) {
                count++;
            }
        }
        out << " " << v << "=" << count;
    }
    out << "\n";

    if (mode == "l") {
        out << "Note: Local mode recomputes statistics per test object.\n";
        out << "Global stats below are provided for reference.\n";
    }

    // Numeric stats
    out << "NumericStats (global):\n";
    for (size_t a = 0; a < ds.types.size(); ++a) {
        if (ds.types[a] != AttrType::Numeric) {
            continue;
        }
        const auto& ns = globalStats.numStats[a];
        out << "  attr[" << a << "]: min=" << ns.min << ", max=" << ns.max << ", range=" << ns.range << "\n";
    }

    // Nominal SVDM matrices
    out << "NominalSVDM (global):\n";
    for (size_t a = 0; a < ds.types.size(); ++a) {
        if (ds.types[a] != AttrType::Nominal) {
            continue;
        }
        const auto& ns = globalStats.nomStats[a];
        out << "  attr[" << a << "] values:";
        for (const auto& v : ns.values) {
            out << " " << v;
        }
        out << "\n";
        for (size_t i = 0; i < ns.values.size(); ++i) {
            out << "    " << ns.values[i] << ":";
            for (size_t j = 0; j < ns.values.size(); ++j) {
                out << " " << ns.dist[i][j];
            }
            out << "\n";
        }
    }

    // Confusion matrices
    out << "ConfusionMatrix Standard (rows=true, cols=pred):\n";
    out << "  labels:";
    for (const auto& v : ds.decisionValues) {
        out << " " << v;
    }
    out << "\n";
    for (size_t i = 0; i < confStd.size(); ++i) {
        out << "  " << ds.decisionValues[i] << ":";
        for (size_t j = 0; j < confStd.size(); ++j) {
            out << " " << confStd[i][j];
        }
        out << "\n";
    }

    out << "ConfusionMatrix Normalized (rows=true, cols=pred):\n";
    out << "  labels:";
    for (const auto& v : ds.decisionValues) {
        out << " " << v;
    }
    out << "\n";
    for (size_t i = 0; i < confNorm.size(); ++i) {
        out << "  " << ds.decisionValues[i] << ":";
        for (size_t j = 0; j < confNorm.size(); ++j) {
            out << " " << confNorm[i][j];
        }
        out << "\n";
    }

    // Our metrics
    auto metricsStd = ComputeMetrics(confStd);
    auto metricsNorm = ComputeMetrics(confNorm);
    auto balStd = ComputeBalanced(metricsStd);
    auto balNorm = ComputeBalanced(metricsNorm);

    out << "PerClassMetrics (standard / normalized):\n";
    for (size_t i = 0; i < ds.decisionValues.size(); ++i) {
        out << "  " << ds.decisionValues[i]
            << " Precision=" << metricsStd[i].precision
            << " Recall=" << metricsStd[i].recall
            << " F1=" << metricsStd[i].f1
            << " | NPrecision=" << metricsNorm[i].precision
            << " NRecall=" << metricsNorm[i].recall
            << " NF1=" << metricsNorm[i].f1 << "\n";
    }

    out << "BalancedMetrics:\n";
    out << "  Bal_Precision=" << balStd.precision
        << " Bal_Recall=" << balStd.recall
        << " Bal_F1=" << balStd.f1 << "\n";
    out << "  NBal_Precision=" << balNorm.precision
        << " NBal_Recall=" << balNorm.recall
        << " NBal_F1=" << balNorm.f1 << "\n";
}