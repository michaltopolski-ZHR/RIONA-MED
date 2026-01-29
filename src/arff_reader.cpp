#include "arff_reader.h"

#include "util.h"

#include <fstream>

struct AttributeDef {
    std::string name;
    AttrType type = AttrType::Nominal;
};

static bool ParseAttributeLine(const std::string& line, AttributeDef& def, std::string& err) {
    std::string rest = Trim(line);
    if (!StartsWithNoCase(rest, "@attribute")) {
        err = "Invalid @attribute line.";
        return false;
    }
    rest = Trim(rest.substr(std::string("@attribute").size()));
    if (rest.empty()) {
        err = "Invalid @attribute line (missing name/type).";
        return false;
    }

    // Parse attribute name (possibly quoted).
    std::string name;
    size_t pos = 0;
    if (rest[pos] == '\'' || rest[pos] == '"') {
        char q = rest[pos++];
        size_t end = rest.find(q, pos);
        if (end == std::string::npos) {
            err = "Invalid @attribute name (unterminated quote).";
            return false;
        }
        name = rest.substr(pos, end - pos);
        pos = end + 1;
    } else {
        size_t end = rest.find_first_of(" \t");
        if (end == std::string::npos) {
            err = "Invalid @attribute line (missing type).";
            return false;
        }
        name = rest.substr(0, end);
        pos = end;
    }

    std::string typeStr = Trim(rest.substr(pos));
    if (typeStr.empty()) {
        err = "Invalid @attribute line (missing type).";
        return false;
    }

    std::string typeLower = ToLower(typeStr);
    AttrType type = AttrType::Nominal;
    if (!typeStr.empty() && typeStr[0] == '{') {
        type = AttrType::Nominal;
    } else if (StartsWithNoCase(typeLower, "numeric") ||
               StartsWithNoCase(typeLower, "real") ||
               StartsWithNoCase(typeLower, "integer")) {
        type = AttrType::Numeric;
    } else if (StartsWithNoCase(typeLower, "string") ||
               StartsWithNoCase(typeLower, "nominal") ||
               StartsWithNoCase(typeLower, "date")) {
        type = AttrType::Nominal;
    } else {
        type = AttrType::Nominal;
    }

    def.name = name;
    def.type = type;
    return true;
}

static std::vector<std::string> SplitDataLine(const std::string& line) {
    if (line.find(',') != std::string::npos) {
        return SplitCsvLike(line);
    }
    return SplitByWhitespace(line);
}

bool ArffReader::Read(const std::string& path, const Config& cfg, Dataset& ds, std::string& err) const {
    std::ifstream in(path);
    if (!in) {
        err = "Cannot open input file: " + path;
        return false;
    }

    std::vector<AttributeDef> attrs;
    std::string line;
    bool inData = false;
    int idCounter = 1;

    while (std::getline(in, line)) {
        std::string trimmed = Trim(line);
        if (trimmed.empty() || IsCommentLine(trimmed)) {
            continue;
        }

        if (!inData) {
            if (StartsWithNoCase(trimmed, "@relation")) {
                continue;
            }
            if (StartsWithNoCase(trimmed, "@attribute")) {
                AttributeDef def;
                if (!ParseAttributeLine(trimmed, def, err)) {
                    return false;
                }
                attrs.push_back(def);
                continue;
            }
            if (StartsWithNoCase(trimmed, "@data")) {
                inData = true;
                continue;
            }
            continue;
        }

        // Data section
        if (IsCommentLine(trimmed)) {
            continue;
        }

        std::vector<std::string> tokens = SplitDataLine(trimmed);
        if (tokens.empty()) {
            continue;
        }
        if (tokens.size() != attrs.size()) {
            err = "Invalid data line: number of values does not match attributes.";
            return false;
        }

        Instance inst;
        inst.id = idCounter++;
        inst.attrs.resize(attrs.size() - 1);

        for (size_t i = 0; i + 1 < attrs.size(); ++i) {
            AttributeValue val;
            val.raw = Trim(tokens[i]);
            if (val.raw.empty() || val.raw == cfg.missingToken || val.raw == "?") {
                val.missing = true;
            }
            inst.attrs[i] = val;
        }

        inst.decision = Trim(tokens.back());
        ds.rows.push_back(std::move(inst));
    }

    if (attrs.size() < 2) {
        err = "ARFF file must define at least 2 attributes (including decision).";
        return false;
    }
    if (ds.rows.empty()) {
        err = "Dataset is empty.";
        return false;
    }

    // Last attribute is treated as decision/class.
    ds.types.clear();
    ds.types.reserve(attrs.size() - 1);
    for (size_t i = 0; i + 1 < attrs.size(); ++i) {
        ds.types.push_back(attrs[i].type);
    }
    return true;
}