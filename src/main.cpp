#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "algorithms.h"
#include "arff_reader.h"
#include "dataset.h"
#include "distance.h"
#include "metrics.h"
#include "output.h"
#include "util.h"

// Parse attribute types string (e.g., "n,c,n" or "ncn").
static std::vector<AttrType> ParseTypes(const std::string& spec) {
    std::vector<AttrType> types;
    for (char c : spec) {
        if (c == 'n' || c == 'N') {
            types.push_back(AttrType::Numeric);
        } else if (c == 'c' || c == 'C' || c == 's' || c == 'S') {
            types.push_back(AttrType::Nominal);
        } else {
            // ignore
        }
    }
    return types;
}

static void BuildTypeIndices(Dataset& ds) {
    ds.numericIdx.clear();
    ds.nominalIdx.clear();
    for (size_t i = 0; i < ds.types.size(); ++i) {
        if (ds.types[i] == AttrType::Numeric) {
            ds.numericIdx.push_back((int)i);
        } else {
            ds.nominalIdx.push_back((int)i);
        }
    }
}

static std::string SanitizePathPart(const std::string& s) {
    std::string out = s;
    for (char& ch : out) {
        if (ch == ' ' || ch == ':' || ch == '*' || ch == '?' || ch == '"' ||
            ch == '<' || ch == '>' || ch == '|' || ch == '\\' || ch == '/') {
            ch = '_';
        }
    }
    return out;
}

static void PrintUsage() {
    std::cout
        << "Usage: riona.exe --input <file.arff> [--types <spec>] [options]\n"
        << "Options:\n"
        << "  --types <spec>                Optional override types (e.g., n,c,n)\n"
        << "  --algo riona|ria|knn|all      Algorithm (default: all)\n"
        << "  --mode g|l|both               Distance stats mode (default: g)\n"
        << "  --svdm svdm|svdmprime         Nominal distance (default: svdm)\n"
        << "  --k 1,3,log                   k values (default: 1,3,log2(n))\n"
        << "  --n <int>                     n for k+NN local neighborhood (default: n-1)\n"
        << "  --missing <token>             Missing value token (default: ?)\n"
        << "  --outdir <dir>                Output directory (default: .)\n";
}

int main(int argc, char** argv) {
    Config cfg;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            cfg.inputFile = argv[++i];
        } else if (arg == "--types" && i + 1 < argc) {
            cfg.typesSpec = argv[++i];
        } else if (arg == "--algo" && i + 1 < argc) {
            cfg.algo = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (arg == "--svdm" && i + 1 < argc) {
            cfg.svdm = argv[++i];
        } else if (arg == "--k" && i + 1 < argc) {
            std::string kSpec = argv[++i];
            std::stringstream ss(kSpec);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token = Trim(token);
                if (token == "log" || token == "log2") {
                    // placeholder handled later when n is known
                    cfg.kValues.push_back(-1);
                } else if (!token.empty()) {
                    cfg.kValues.push_back(std::stoi(token));
                }
            }
        } else if (arg == "--n" && i + 1 < argc) {
            cfg.nForKPlusNN = std::stoi(argv[++i]);
        } else if (arg == "--missing" && i + 1 < argc) {
            cfg.missingToken = argv[++i];
        } else if (arg == "--outdir" && i + 1 < argc) {
            cfg.outDir = argv[++i];
        } else if (arg == "--help") {
            PrintUsage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            PrintUsage();
            return 1;
        }
    }

    if (cfg.inputFile.empty()) {
        PrintUsage();
        return 1;
    }

    // Prepare distance config
    DistanceConfig distCfg;
    if (cfg.svdm == "svdmprime" || cfg.svdm == "svdm'" || cfg.svdm == "svdmp") {
        distCfg.svdmPrime = true;
        distCfg.missingNominal = 1.0;
    } else {
        distCfg.svdmPrime = false;
        distCfg.missingNominal = 2.0;
    }
    distCfg.missingNumeric = 1.0;

    // Read dataset (ARFF)
    Dataset ds;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tReadStart = t0;

    std::string err;
    ArffReader reader;
    if (!reader.Read(cfg.inputFile, cfg, ds, err)) {
        std::cerr << err << "\n";
        return 1;
    }
    auto tReadEnd = std::chrono::high_resolution_clock::now();

    if (ds.rows.size() < 2) {
        std::cerr << "Dataset must contain at least 2 objects for leave-one-out.\n";
        return 1;
    }

    // Optional override of attribute types
    if (!cfg.typesSpec.empty()) {
        ds.types = ParseTypes(cfg.typesSpec);
        if (ds.types.size() != ds.rows[0].attrs.size()) {
            std::cerr << "Types count does not match number of attributes.\n";
            return 1;
        }
    }
    BuildTypeIndices(ds);

    // Convert numeric values to doubles for numeric attributes
    for (auto& inst : ds.rows) {
        for (size_t a = 0; a < ds.types.size(); ++a) {
            if (ds.types[a] != AttrType::Numeric) {
                continue;
            }
            auto& v = inst.attrs[a];
            if (v.missing) {
                continue;
            }
            try {
                v.num = std::stod(v.raw);
            } catch (...) {
                v.missing = true;
            }
        }
    }

    // Build decision label mapping
    for (const auto& inst : ds.rows) {
        if (ds.decisionIndex.find(inst.decision) == ds.decisionIndex.end()) {
            int idx = static_cast<int>(ds.decisionValues.size());
            ds.decisionValues.push_back(inst.decision);
            ds.decisionIndex[inst.decision] = idx;
        }
    }

    // Compute global stats on the full dataset (used in global mode and for reporting)
    std::vector<int> allIndices(ds.rows.size());
    for (size_t i = 0; i < ds.rows.size(); ++i) {
        allIndices[i] = static_cast<int>(i);
    }
    auto tPrepStart = std::chrono::high_resolution_clock::now();
    Stats globalStats = ComputeStats(ds, allIndices, distCfg);
    auto tPrepEnd = std::chrono::high_resolution_clock::now();

    // Prepare k values
    if (cfg.kValues.empty()) {
        cfg.kValues.push_back(1);
        cfg.kValues.push_back(3);
        cfg.kValues.push_back(-1); // log2(n)
    }

    // Expand k list (resolve log2)
    std::vector<int> kList;
    int nAll = static_cast<int>(ds.rows.size());
    for (int k : cfg.kValues) {
        if (k == -1) {
            int kval = (int)std::floor(std::log2(std::max(1, nAll)));
            kList.push_back(std::max(1, kval));
        } else {
            kList.push_back(std::max(1, k));
        }
    }
    // unique + sort
    std::sort(kList.begin(), kList.end());
    kList.erase(std::unique(kList.begin(), kList.end()), kList.end());

    // Determine which algorithms to run
    std::vector<std::string> algos;
    if (cfg.algo == "all") {
        algos = {"RIONA", "RIA", "KNN"};
    } else if (cfg.algo == "riona") {
        algos = {"RIONA"};
    } else if (cfg.algo == "ria") {
        algos = {"RIA"};
    } else if (cfg.algo == "knn") {
        algos = {"KNN"};
    } else {
        std::cerr << "Unknown algorithm: " << cfg.algo << "\n";
        return 1;
    }

    std::vector<std::string> modes;
    if (cfg.mode == "both") {
        modes = {"g", "l"};
    } else if (cfg.mode == "g" || cfg.mode == "l") {
        modes = {cfg.mode};
    } else {
        std::cerr << "Unknown mode: " << cfg.mode << "\n";
        return 1;
    }

    // Run experiments
    for (const auto& algo : algos) {
        for (const auto& mode : modes) {
            for (int k : kList) {
                int maxK = static_cast<int>(ds.rows.size()) - 1;
                int kEff = std::min(k, maxK);
                if (kEff < 1) {
                    continue;
                }
                // Prepare output buffers
                std::vector<std::string> predStd(ds.rows.size());
                std::vector<std::string> predNorm(ds.rows.size());
                std::vector<std::vector<Neighbor>> knnLists(ds.rows.size());

                auto tClassifyStart = std::chrono::high_resolution_clock::now();

                std::vector<std::vector<int>> confStd = InitMatrix(ds.decisionValues.size());
                std::vector<std::vector<int>> confNorm = InitMatrix(ds.decisionValues.size());

                for (size_t i = 0; i < ds.rows.size(); ++i) {
                    // Build training index list for leave-one-out
                    std::vector<int> trainingIdx;
                    trainingIdx.reserve(ds.rows.size() - 1);
                    for (size_t j = 0; j < ds.rows.size(); ++j) {
                        if (j == i) continue;
                        trainingIdx.push_back(static_cast<int>(j));
                    }

                    // Choose base stats: global or local
                    Stats baseStats = (mode == "g") ? globalStats : ComputeStats(ds, trainingIdx, distCfg);

                    ClassificationResult res;
                    if (algo == "RIONA") {
                        res = ClassifyRIONA(ds, distCfg, baseStats, trainingIdx, (int)i, kEff);
                    } else if (algo == "RIA") {
                        res = ClassifyRIA(ds, distCfg, baseStats, trainingIdx, (int)i, kEff);
                    } else { // KNN => k+NN
                        int nLocal = (cfg.nForKPlusNN < 0) ? (int)trainingIdx.size() : cfg.nForKPlusNN;
                        if (nLocal < kEff) {
                            nLocal = kEff;
                        }
                        res = ClassifyKPlusNN(ds, distCfg, baseStats, trainingIdx, (int)i, kEff, nLocal);
                    }

                    predStd[i] = res.predictedStandard;
                    predNorm[i] = res.predictedNormalized;
                    knnLists[i] = std::move(res.knnList);

                    int trueIdx = ds.decisionIndex.at(ds.rows[i].decision);
                    int predStdIdx = ds.decisionIndex.at(predStd[i]);
                    int predNormIdx = ds.decisionIndex.at(predNorm[i]);
                    confStd[trueIdx][predStdIdx] += 1;
                    confNorm[trueIdx][predNormIdx] += 1;
                }

                auto tClassifyEnd = std::chrono::high_resolution_clock::now();
                auto tWriteStart = tClassifyEnd;

                // Build output filenames
                std::string inputBase = cfg.inputFile;
                size_t slash = inputBase.find_last_of("/\\");
                if (slash != std::string::npos) {
                    inputBase = inputBase.substr(slash + 1);
                }
                size_t dot = inputBase.find_last_of('.');
                if (dot != std::string::npos) {
                    inputBase = inputBase.substr(0, dot);
                }

                std::string svdmLabel = distCfg.svdmPrime ? "SVDMprime" : "SVDM";
                int D = static_cast<int>(ds.types.size());
                int R = static_cast<int>(ds.rows.size());

                std::stringstream suffix;
                suffix << algo << "_" << inputBase
                       << "_D" << D
                       << "_R" << R
                       << "_k" << kEff
                       << "_" << svdmLabel
                       << "_" << mode;

                std::string baseFolderName = SanitizePathPart(inputBase);
                std::filesystem::path baseDir = std::filesystem::path(cfg.outDir) / baseFolderName;
                std::filesystem::create_directories(baseDir);

                std::string expFolderName = "EXP_" + SanitizePathPart(suffix.str());
                std::filesystem::path expDir = baseDir / expFolderName;
                std::filesystem::create_directories(expDir);

                std::string outFile = (expDir / ("OUT_" + suffix.str() + ".csv")).string();
                std::string statFile = (expDir / ("STAT_" + suffix.str() + ".txt")).string();
                std::string knnFile = (expDir / ("kNN_" + suffix.str() + ".csv")).string();

                WriteOutFile(outFile, ds, predStd, predNorm, cfg.missingToken);
                WriteKnnFile(knnFile, knnLists);

                auto tWriteEnd = std::chrono::high_resolution_clock::now();
                double timeReadMs = std::chrono::duration<double, std::milli>(tReadEnd - tReadStart).count();
                double timePrepMs = std::chrono::duration<double, std::milli>(tPrepEnd - tPrepStart).count();
                double timeClassifyMs = std::chrono::duration<double, std::milli>(tClassifyEnd - tClassifyStart).count();
                double timeWriteMs = std::chrono::duration<double, std::milli>(tWriteEnd - tWriteStart).count();
                double timeTotalMs = timeReadMs + timePrepMs + timeClassifyMs + timeWriteMs;

                WriteStatFile(statFile,
                              ds,
                              globalStats,
                              cfg.inputFile,
                              algo,
                              mode,
                              svdmLabel,
                              kEff,
                              timeReadMs,
                              timePrepMs,
                              timeClassifyMs,
                              timeWriteMs,
                              timeTotalMs,
                              confStd,
                              confNorm);
            }
        }
    }

    std::cout << "Done.\n";
    return 0;
}
