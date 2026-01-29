#pragma once

#include "dataset.h"
#include "metrics.h"

#include <string>
#include <vector>

void WriteOutFile(const std::string& path,
                  const Dataset& ds,
                  const std::vector<std::string>& predStd,
                  const std::vector<std::string>& predNorm,
                  const std::string& missingToken);

void WriteKnnFile(const std::string& path,
                  const std::vector<std::vector<Neighbor>>& knnLists);

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
                   const std::vector<std::vector<int>>& confNorm);