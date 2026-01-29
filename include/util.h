#pragma once

#include <string>
#include <vector>

std::string Trim(const std::string& s);
std::vector<std::string> SplitByWhitespace(const std::string& line);
std::string ToLower(const std::string& s);
bool StartsWithNoCase(const std::string& s, const std::string& prefix);
bool IsCommentLine(const std::string& s);
std::vector<std::string> SplitCsvLike(const std::string& line);