#pragma once

#include <string>
#include <unordered_map>
#include <vector>

enum class AttrType { Numeric, Nominal };

// Represents a single attribute value (numeric or nominal) plus missing flag.
struct AttributeValue {
    bool missing = false;
    double num = 0.0;          // valid when numeric and not missing
    std::string raw;           // original token (for output)
};

// Represents a single data instance (row).
struct Instance {
    int id = 0;                                // 1-based id
    std::vector<AttributeValue> attrs;         // conditional attributes only
    std::string decision;                      // decision/class value
};

// Encapsulates dataset along with attribute types and class label mapping.
struct Dataset {
    std::vector<Instance> rows;
    std::vector<AttrType> types;               // size = number of conditional attributes
    std::vector<int> numericIdx;               // indices of numeric attributes
    std::vector<int> nominalIdx;               // indices of nominal attributes
    std::vector<std::string> decisionValues;   // unique decision values
    std::unordered_map<std::string, int> decisionIndex;
};

// Statistics for numeric attributes (min/max/range).
struct NumericStat {
    double min = 0.0;
    double max = 0.0;
    double range = 0.0;
    bool hasValue = false;
};

// Statistics for nominal attributes (value list + SVDM distance matrix).
struct NominalStat {
    std::vector<std::string> values;                         // index -> value
    std::unordered_map<std::string, int> index;              // value -> index
    std::vector<std::vector<double>> dist;                   // SVDM distance matrix
};

// Preprocessing result used for distance calculations.
struct Stats {
    std::vector<NumericStat> numStats;                       // size = attributes
    std::vector<NominalStat> nomStats;                       // size = attributes
};

// Settings describing how distances are computed.
struct DistanceConfig {
    bool svdmPrime = false;          // true => SVDM' (normalized), false => SVDM
    double missingNominal = 2.0;     // distance when nominal value missing
    double missingNumeric = 1.0;     // distance when numeric value missing
};

// Neighbour used in kNN lists.
struct Neighbor {
    int index = -1;      // index in dataset rows
    double dist = 0.0;
};

// CLI configuration
struct Config {
    std::string inputFile;
    std::string typesSpec;
    std::string algo = "all";          // riona | ria | knn | all
    std::string mode = "g";            // g | l | both
    std::string svdm = "svdm";         // svdm | svdmprime
    std::string missingToken = "?";
    std::string outDir = ".";
    std::vector<int> kValues;          // if empty => auto
    int nForKPlusNN = -1;              // if -1 => use training size
};

// Classification output per instance
struct ClassificationResult {
    std::string predictedStandard;
    std::string predictedNormalized;
    std::vector<Neighbor> knnList;  // neighbors used in the algorithm
};

// Metrics per class
struct MetricsPerClass {
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};