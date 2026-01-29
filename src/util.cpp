#include "util.h"

#include <cctype>
#include <sstream>

std::string Trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::vector<std::string> SplitByWhitespace(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string token;
    while (ss >> token) {
        out.push_back(Trim(token));
    }
    return out;
}

std::string ToLower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

bool StartsWithNoCase(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) {
        return false;
    }
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

bool IsCommentLine(const std::string& s) {
    if (s.empty()) {
        return true;
    }
    return s[0] == '%' || s[0] == '#';
}

std::vector<std::string> SplitCsvLike(const std::string& line) {
    std::vector<std::string> tokens;
    std::string cur;
    bool inQuote = false;
    char quoteChar = '\0';

    for (char ch : line) {
        if (inQuote) {
            if (ch == quoteChar) {
                inQuote = false;
            } else {
                cur.push_back(ch);
            }
        } else {
            if (ch == '"' || ch == '\'') {
                inQuote = true;
                quoteChar = ch;
            } else if (ch == ',') {
                tokens.push_back(Trim(cur));
                cur.clear();
            } else {
                cur.push_back(ch);
            }
        }
    }
    tokens.push_back(Trim(cur));
    return tokens;
}