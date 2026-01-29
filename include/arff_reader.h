#pragma once

#include "dataset.h"

#include <string>

class ArffReader {
public:
    bool Read(const std::string& path, const Config& cfg, Dataset& ds, std::string& err) const;
};