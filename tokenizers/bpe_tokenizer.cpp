/*!
 *  Copyright (c) 2023 by Contributors
 * \file bpe_tokenizer.cpp
 * \brief Compact native BPE tokenizer
 */
#include <tokenizers/tokenizer_cache_util.h>
#include <tokenizers/tokenizers_cpp.h>

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace tokenizers {
namespace {

using json = nlohmann::json;

constexpr uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kInvalidRank = std::numeric_limits<uint32_t>::max();
constexpr char kCacheName[] = "BPE";
constexpr cache_util::CacheKind kCacheKind = cache_util::CacheKind::BPE;

using cache_util::AppendHeader;
using cache_util::AppendU32;
using cache_util::ReadBytes;
using cache_util::ReadHeader;
using cache_util::ReadU32;

enum class BPEVariant : uint32_t {
  ByteLevel = 1,
  SpaceReplacement = 2,
};

struct TokenEntry {
  uint32_t offset = 0;
  uint32_t length = 0;
  uint32_t id = 0;
};

struct MergeEntry {
  uint32_t left = 0;
  uint32_t right = 0;
  uint32_t rank = 0;
  uint32_t merged = 0;
};

uint64_t PairKey(uint32_t left, uint32_t right) {
  return (static_cast<uint64_t>(left) << 32) | right;
}

void AppendUtf8(std::string &out, uint32_t cp) {
  if (cp <= 0x7f) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

uint32_t DecodeUtf8At(const std::string &text, size_t offset, size_t &next) {
  const unsigned char c = static_cast<unsigned char>(text[offset]);
  if (c < 0x80) {
    next = offset + 1;
    return c;
  }
  if ((c & 0xe0) == 0xc0 && offset + 1 < text.size()) {
    next = offset + 2;
    return ((c & 0x1f) << 6) |
           (static_cast<unsigned char>(text[offset + 1]) & 0x3f);
  }
  if ((c & 0xf0) == 0xe0 && offset + 2 < text.size()) {
    next = offset + 3;
    return ((c & 0x0f) << 12) |
           ((static_cast<unsigned char>(text[offset + 1]) & 0x3f) << 6) |
           (static_cast<unsigned char>(text[offset + 2]) & 0x3f);
  }
  if ((c & 0xf8) == 0xf0 && offset + 3 < text.size()) {
    next = offset + 4;
    return ((c & 0x07) << 18) |
           ((static_cast<unsigned char>(text[offset + 1]) & 0x3f) << 12) |
           ((static_cast<unsigned char>(text[offset + 2]) & 0x3f) << 6) |
           (static_cast<unsigned char>(text[offset + 3]) & 0x3f);
  }

  next = offset + 1;
  return c;
}

bool IsAsciiWhitespace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' ||
         cp == '\v';
}

bool IsNewline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

bool IsAsciiLetter(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

bool IsNumber(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool IsLetterOrNumber(uint32_t cp) {
  if (IsAsciiLetter(cp) || IsNumber(cp)) {
    return true;
  }

  return (cp >= 0x00c0 && cp <= 0x02af) || (cp >= 0x0370 && cp <= 0x052f) ||
         (cp >= 0x0590 && cp <= 0x08ff) || (cp >= 0x0900 && cp <= 0x0d7f) ||
         (cp >= 0x1100 && cp <= 0x11ff) || (cp >= 0x3040 && cp <= 0x30ff) ||
         (cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xac00 && cp <= 0xd7af) ||
         (cp >= 0xff10 && cp <= 0xff19) || (cp >= 0xff21 && cp <= 0xff3a) ||
         (cp >= 0xff41 && cp <= 0xff5a);
}

bool StartsWithInsensitive(const std::string &text, size_t offset,
                           const char *pattern) {
  for (size_t i = 0; pattern[i] != '\0'; ++i) {
    if (offset + i >= text.size()) {
      return false;
    }
    const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
    const unsigned char rhs = static_cast<unsigned char>(pattern[i]);
    if (std::tolower(lhs) != std::tolower(rhs)) {
      return false;
    }
  }
  return true;
}

size_t ConsumeLetters(const std::string &text, size_t offset) {
  size_t i = offset;
  while (i < text.size()) {
    size_t next = i;
    const uint32_t cp = DecodeUtf8At(text, i, next);
    if (!IsLetterOrNumber(cp) || IsNumber(cp)) {
      break;
    }
    i = next;
  }
  return i;
}

size_t ConsumeNumbers(const std::string &text, size_t offset,
                      uint32_t max_digits) {
  size_t i = offset;
  uint32_t count = 0;
  while (i < text.size() && count < max_digits) {
    size_t next = i;
    const uint32_t cp = DecodeUtf8At(text, i, next);
    if (!IsNumber(cp)) {
      break;
    }
    i = next;
    ++count;
  }
  return i;
}

std::vector<std::string> GPTSplit(const std::string &text,
                                  uint32_t max_digit_group_size) {
  std::vector<std::string> pieces;
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == '\'') {
      const char *suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
      bool matched = false;
      for (const char *suffix : suffixes) {
        if (StartsWithInsensitive(text, i, suffix)) {
          const size_t end = i + std::strlen(suffix);
          pieces.push_back(text.substr(i, end - i));
          i = end;
          matched = true;
          break;
        }
      }
      if (matched) {
        continue;
      }
    }

    size_t next = i;
    uint32_t cp = DecodeUtf8At(text, i, next);

    if (!IsLetterOrNumber(cp) && !IsNewline(cp)) {
      if (next < text.size()) {
        size_t next2 = next;
        const uint32_t cp2 = DecodeUtf8At(text, next, next2);
        if (IsLetterOrNumber(cp2) && !IsNumber(cp2)) {
          const size_t end = ConsumeLetters(text, next);
          pieces.push_back(text.substr(i, end - i));
          i = end;
          continue;
        }
      }
    }

    if (IsLetterOrNumber(cp) && !IsNumber(cp)) {
      const size_t end = ConsumeLetters(text, i);
      pieces.push_back(text.substr(i, end - i));
      i = end;
      continue;
    }

    if (IsNumber(cp)) {
      const size_t end = ConsumeNumbers(text, i, max_digit_group_size);
      pieces.push_back(text.substr(i, end - i));
      i = end;
      continue;
    }

    if (cp == ' ' && next < text.size()) {
      size_t after_space = next;
      const uint32_t cp2 = DecodeUtf8At(text, next, after_space);
      if (!IsAsciiWhitespace(cp2) && !IsLetterOrNumber(cp2)) {
        size_t end = next;
        while (end < text.size()) {
          size_t n = end;
          const uint32_t c = DecodeUtf8At(text, end, n);
          if (IsAsciiWhitespace(c) || IsLetterOrNumber(c)) {
            break;
          }
          end = n;
        }
        while (end < text.size()) {
          size_t n = end;
          const uint32_t c = DecodeUtf8At(text, end, n);
          if (!IsNewline(c)) {
            break;
          }
          end = n;
        }
        pieces.push_back(text.substr(i, end - i));
        i = end;
        continue;
      }
    }

    if (!IsAsciiWhitespace(cp) && !IsLetterOrNumber(cp)) {
      size_t end = i;
      while (end < text.size()) {
        size_t n = end;
        const uint32_t c = DecodeUtf8At(text, end, n);
        if (IsAsciiWhitespace(c) || IsLetterOrNumber(c)) {
          break;
        }
        end = n;
      }
      while (end < text.size()) {
        size_t n = end;
        const uint32_t c = DecodeUtf8At(text, end, n);
        if (!IsNewline(c)) {
          break;
        }
        end = n;
      }
      pieces.push_back(text.substr(i, end - i));
      i = end;
      continue;
    }

    size_t end = i;
    while (end < text.size()) {
      size_t n = end;
      const uint32_t c = DecodeUtf8At(text, end, n);
      if (!IsAsciiWhitespace(c)) {
        break;
      }
      end = n;
    }
    const size_t last_newline = text.find_last_of("\r\n", end - 1);
    if (last_newline != std::string::npos && last_newline >= i) {
      pieces.push_back(text.substr(i, last_newline + 1 - i));
      i = last_newline + 1;
    } else if (end < text.size() && end - i > 1) {
      pieces.push_back(text.substr(i, end - i - 1));
      i = end - 1;
    } else {
      pieces.push_back(text.substr(i, end - i));
      i = end;
    }
  }

  return pieces;
}

class CompactBPETokenizer : public Tokenizer {
public:
  explicit CompactBPETokenizer(const std::string &json_blob) {
    LoadJSON(json::parse(json_blob));
  }

  explicit CompactBPETokenizer(const std::string &cache_blob, bool) {
    LoadCache(cache_blob);
  }

  std::vector<int32_t> Encode(const std::string &text) final {
    return Encode(text, false);
  }

  std::vector<int32_t> Encode(const std::string &text,
                              bool add_special_tokens) final {
    std::vector<int32_t> ids;
    if (add_special_tokens) {
      for (uint32_t id : prefix_ids_) {
        ids.push_back(static_cast<int32_t>(id));
      }
    }

    std::string pending;
    size_t i = 0;
    while (i < text.size()) {
      uint32_t special_id = kInvalidId;
      size_t special_len = 0;
      if (FindSpecialToken(text, i, special_id, special_len)) {
        EncodeOrdinary(pending, ids);
        pending.clear();
        ids.push_back(static_cast<int32_t>(special_id));
        i += special_len;
      } else {
        pending.push_back(text[i++]);
      }
    }
    EncodeOrdinary(pending, ids);

    return ids;
  }

  std::string Decode(const std::vector<int32_t> &ids) final {
    std::string text;
    for (int32_t raw_id : ids) {
      if (raw_id < 0 ||
          static_cast<size_t>(raw_id) + 1 >= token_offsets_.size()) {
        continue;
      }

      const std::string token = IdToToken(raw_id);
      if (variant_ == BPEVariant::ByteLevel) {
        DecodeByteLevelToken(token, text);
      } else {
        DecodeSpaceReplacementToken(token, text);
      }
    }
    return text;
  }

  size_t GetVocabSize() final { return GetVocabSizeInternal(); }

  std::string IdToToken(int32_t token_id) final {
    if (token_id < 0 ||
        static_cast<size_t>(token_id) + 1 >= token_offsets_.size()) {
      return "";
    }

    const uint32_t id = static_cast<uint32_t>(token_id);
    const uint32_t begin = token_offsets_[id];
    const uint32_t end = token_offsets_[id + 1];
    if (begin > end || end > token_bytes_.size()) {
      return "";
    }
    return token_bytes_.substr(begin, end - begin);
  }

  int32_t TokenToId(const std::string &token) final {
    const uint32_t id = LookupToken(token);
    return id == kInvalidId ? -1 : static_cast<int32_t>(id);
  }

  std::string SerializeToCache() const final {
    std::string out;
    AppendHeader(out, kCacheKind);
    AppendU32(out, static_cast<uint32_t>(variant_));
    AppendU32(out, digit_group_size_);
    AppendU32(out, unk_id_);
    AppendU32(out, static_cast<uint32_t>(GetVocabSizeInternal()));
    AppendU32(out, static_cast<uint32_t>(token_bytes_.size()));
    AppendU32(out, static_cast<uint32_t>(token_entries_.size()));
    AppendU32(out, static_cast<uint32_t>(merges_.size()));
    AppendU32(out, static_cast<uint32_t>(special_ids_.size()));
    AppendU32(out, static_cast<uint32_t>(prefix_ids_.size()));

    for (uint32_t offset : token_offsets_) {
      AppendU32(out, offset);
    }
    out += token_bytes_;
    for (const auto &entry : token_entries_) {
      AppendU32(out, entry.offset);
      AppendU32(out, entry.length);
      AppendU32(out, entry.id);
    }
    for (const auto &merge : merges_) {
      AppendU32(out, merge.left);
      AppendU32(out, merge.right);
      AppendU32(out, merge.rank);
      AppendU32(out, merge.merged);
    }
    for (uint32_t id : special_ids_) {
      AppendU32(out, id);
    }
    for (uint32_t id : prefix_ids_) {
      AppendU32(out, id);
    }
    return out;
  }

private:
  void LoadJSON(const json &tokenizer_json) {
    if (!tokenizer_json.contains("model") ||
        !tokenizer_json["model"].is_object()) {
      throw std::runtime_error("BPE tokenizer.json has no model object");
    }

    const json &model = tokenizer_json["model"];
    if (!model.contains("type") || !model["type"].is_string() ||
        model["type"].get<std::string>() != "BPE") {
      throw std::runtime_error("tokenizer.json is not a BPE tokenizer");
    }

    DetectVariant(tokenizer_json);
    LoadTokens(tokenizer_json);
    LoadMerges(model);
    LoadSpecialTokens(tokenizer_json);
    LoadPostProcessor(tokenizer_json);
    BuildByteMaps();
    Validate();
  }

  void DetectVariant(const json &tokenizer_json) {
    const json &model = tokenizer_json["model"];
    const bool byte_fallback = model.contains("byte_fallback") &&
                               model["byte_fallback"].is_boolean() &&
                               model["byte_fallback"].get<bool>();

    if (!byte_fallback && tokenizer_json.contains("pre_tokenizer") &&
        tokenizer_json["pre_tokenizer"].is_object() &&
        tokenizer_json["pre_tokenizer"].contains("type") &&
        tokenizer_json["pre_tokenizer"]["type"].get<std::string>() ==
          "Sequence") {
      variant_ = BPEVariant::ByteLevel;
      digit_group_size_ = 1;
      const std::string pre = tokenizer_json["pre_tokenizer"].dump();
      if (pre.find("\\\\p{N}{1,3}") != std::string::npos) {
        digit_group_size_ = 3;
      }
      return;
    }

    if (byte_fallback && tokenizer_json.contains("normalizer") &&
        tokenizer_json["normalizer"].is_object() &&
        tokenizer_json["normalizer"].contains("type") &&
        tokenizer_json["normalizer"]["type"].get<std::string>() == "Replace") {
      variant_ = BPEVariant::SpaceReplacement;
      digit_group_size_ = 1;
      return;
    }

    throw std::runtime_error("Unsupported BPE tokenizer variant");
  }

  void LoadTokens(const json &tokenizer_json) {
    const json &model = tokenizer_json["model"];
    if (!model.contains("vocab") || !model["vocab"].is_object()) {
      throw std::runtime_error("BPE tokenizer.json has no model.vocab");
    }

    size_t max_id = 0;
    auto visit_token = [&](const std::string &, const json &value) {
      if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::runtime_error("BPE vocab id must be an integer");
      }
      const int64_t id = value.get<int64_t>();
      if (id < 0) {
        throw std::runtime_error("BPE vocab id must be non-negative");
      }
      max_id = std::max(max_id, static_cast<size_t>(id));
    };

    for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
      visit_token(it.key(), it.value());
    }
    if (tokenizer_json.contains("added_tokens") &&
        tokenizer_json["added_tokens"].is_array()) {
      for (const auto &token : tokenizer_json["added_tokens"]) {
        if (token.is_object() && token.contains("id")) {
          visit_token("", token["id"]);
        }
      }
    }

    std::vector<std::string> tokens(max_id + 1);
    for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
      tokens[static_cast<size_t>(it.value().get<int64_t>())] = it.key();
    }
    if (tokenizer_json.contains("added_tokens") &&
        tokenizer_json["added_tokens"].is_array()) {
      for (const auto &token : tokenizer_json["added_tokens"]) {
        if (!token.is_object() || !token.contains("id") ||
            !token.contains("content") || !token["content"].is_string()) {
          continue;
        }
        tokens[static_cast<size_t>(token["id"].get<int64_t>())] =
          token["content"].get<std::string>();
      }
    }

    token_offsets_.clear();
    token_bytes_.clear();
    token_offsets_.reserve(tokens.size() + 1);
    token_offsets_.push_back(0);
    for (const std::string &token : tokens) {
      token_bytes_ += token;
      token_offsets_.push_back(static_cast<uint32_t>(token_bytes_.size()));
    }

    token_entries_.clear();
    token_entries_.reserve(tokens.size());
    for (uint32_t id = 0; id < tokens.size(); ++id) {
      token_entries_.push_back(TokenEntry{
        token_offsets_[id], token_offsets_[id + 1] - token_offsets_[id], id});
    }
    SortTokenEntries();

    if (model.contains("unk_token") && model["unk_token"].is_string()) {
      unk_id_ = LookupToken(model["unk_token"].get<std::string>());
    }
  }

  void LoadMerges(const json &model) {
    if (!model.contains("merges") || !model["merges"].is_array()) {
      throw std::runtime_error("BPE tokenizer.json has no model.merges");
    }

    merges_.clear();
    uint32_t rank = 0;
    for (const auto &merge : model["merges"]) {
      std::string left;
      std::string right;
      if (merge.is_array() && merge.size() == 2 && merge[0].is_string() &&
          merge[1].is_string()) {
        left = merge[0].get<std::string>();
        right = merge[1].get<std::string>();
      } else if (merge.is_string()) {
        const std::string line = merge.get<std::string>();
        const size_t space = line.find(' ');
        if (space == std::string::npos) {
          throw std::runtime_error("Invalid BPE merge line");
        }
        left = line.substr(0, space);
        right = line.substr(space + 1);
      } else {
        throw std::runtime_error("Invalid BPE merge entry");
      }

      const uint32_t left_id = LookupToken(left);
      const uint32_t right_id = LookupToken(right);
      const uint32_t merged_id = LookupToken(left + right);
      if (left_id != kInvalidId && right_id != kInvalidId &&
          merged_id != kInvalidId) {
        merges_.push_back(MergeEntry{left_id, right_id, rank, merged_id});
      }
      ++rank;
    }

    std::sort(
      merges_.begin(), merges_.end(), [](const auto &lhs, const auto &rhs) {
        return PairKey(lhs.left, lhs.right) < PairKey(rhs.left, rhs.right);
      });
  }

  void LoadSpecialTokens(const json &tokenizer_json) {
    special_ids_.clear();
    if (!tokenizer_json.contains("added_tokens") ||
        !tokenizer_json["added_tokens"].is_array()) {
      return;
    }

    for (const auto &token : tokenizer_json["added_tokens"]) {
      if (!token.is_object() || !token.contains("id")) {
        continue;
      }
      special_ids_.push_back(static_cast<uint32_t>(token["id"].get<int64_t>()));
    }

    SortSpecialIds();
  }

  void LoadPostProcessor(const json &tokenizer_json) {
    prefix_ids_.clear();
    if (!tokenizer_json.contains("post_processor")) {
      return;
    }
    const json *processor =
      FindTemplateProcessor(tokenizer_json["post_processor"]);
    if (processor == nullptr || !processor->contains("single") ||
        !(*processor)["single"].is_array()) {
      return;
    }

    for (const auto &item : (*processor)["single"]) {
      if (item.contains("Sequence")) {
        break;
      }
      if (!item.contains("SpecialToken")) {
        continue;
      }
      const json &special = item["SpecialToken"];
      if (!special.contains("id") || !special["id"].is_string()) {
        continue;
      }
      const uint32_t id = LookupToken(special["id"].get<std::string>());
      if (id != kInvalidId) {
        prefix_ids_.push_back(id);
      }
    }
  }

  const json *FindTemplateProcessor(const json &processor) const {
    if (!processor.is_object() || !processor.contains("type")) {
      return nullptr;
    }
    if (processor["type"].is_string() &&
        processor["type"].get<std::string>() == "TemplateProcessing") {
      return &processor;
    }
    if (processor.contains("processors") &&
        processor["processors"].is_array()) {
      for (const auto &item : processor["processors"]) {
        const json *found = FindTemplateProcessor(item);
        if (found != nullptr) {
          return found;
        }
      }
    }
    return nullptr;
  }

  void LoadCache(const std::string &blob) {
    size_t offset = ReadHeader(blob, kCacheKind, kCacheName);
    variant_ = static_cast<BPEVariant>(ReadU32(blob, offset, kCacheName));
    digit_group_size_ = ReadU32(blob, offset, kCacheName);
    unk_id_ = ReadU32(blob, offset, kCacheName);
    const uint32_t vocab_size = ReadU32(blob, offset, kCacheName);
    const uint32_t token_bytes_size = ReadU32(blob, offset, kCacheName);
    const uint32_t token_entry_count = ReadU32(blob, offset, kCacheName);
    const uint32_t merge_count = ReadU32(blob, offset, kCacheName);
    const uint32_t special_count = ReadU32(blob, offset, kCacheName);
    const uint32_t prefix_count = ReadU32(blob, offset, kCacheName);

    token_offsets_.resize(static_cast<size_t>(vocab_size) + 1);
    for (uint32_t &value : token_offsets_) {
      value = ReadU32(blob, offset, kCacheName);
    }
    token_bytes_ = ReadBytes(blob, offset, token_bytes_size, kCacheName);

    token_entries_.resize(token_entry_count);
    for (auto &entry : token_entries_) {
      entry.offset = ReadU32(blob, offset, kCacheName);
      entry.length = ReadU32(blob, offset, kCacheName);
      entry.id = ReadU32(blob, offset, kCacheName);
    }

    merges_.resize(merge_count);
    for (auto &merge : merges_) {
      merge.left = ReadU32(blob, offset, kCacheName);
      merge.right = ReadU32(blob, offset, kCacheName);
      merge.rank = ReadU32(blob, offset, kCacheName);
      merge.merged = ReadU32(blob, offset, kCacheName);
    }

    special_ids_.resize(special_count);
    for (uint32_t &id : special_ids_) {
      id = ReadU32(blob, offset, kCacheName);
    }
    prefix_ids_.resize(prefix_count);
    for (uint32_t &id : prefix_ids_) {
      id = ReadU32(blob, offset, kCacheName);
    }

    if (offset != blob.size()) {
      throw std::runtime_error("Invalid BPE cache: trailing bytes");
    }

    BuildByteMaps();
    SortSpecialIds();
    Validate();
  }

  void Validate() const {
    if (variant_ != BPEVariant::ByteLevel &&
        variant_ != BPEVariant::SpaceReplacement) {
      throw std::runtime_error("Invalid BPE cache: bad variant");
    }
    if (token_offsets_.empty() || token_offsets_.front() != 0 ||
        static_cast<size_t>(token_offsets_.back()) != token_bytes_.size()) {
      throw std::runtime_error("Invalid BPE cache: bad token offsets");
    }
    for (size_t i = 1; i < token_offsets_.size(); ++i) {
      if (token_offsets_[i - 1] > token_offsets_[i]) {
        throw std::runtime_error("Invalid BPE cache: unsorted offsets");
      }
    }
  }

  size_t GetVocabSizeInternal() const {
    return token_offsets_.empty() ? 0 : token_offsets_.size() - 1;
  }

  void SortTokenEntries() {
    std::sort(
      token_entries_.begin(), token_entries_.end(),
      [&](const auto &lhs, const auto &rhs) {
        const std::string_view l(token_bytes_.data() + lhs.offset, lhs.length);
        const std::string_view r(token_bytes_.data() + rhs.offset, rhs.length);
        return l < r;
      });
  }

  void SortSpecialIds() {
    std::sort(special_ids_.begin(), special_ids_.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                const uint32_t lhs_len =
                  token_offsets_[lhs + 1] - token_offsets_[lhs];
                const uint32_t rhs_len =
                  token_offsets_[rhs + 1] - token_offsets_[rhs];
                if (lhs_len != rhs_len) {
                  return lhs_len > rhs_len;
                }
                return lhs < rhs;
              });
  }

  uint32_t LookupToken(const std::string &token) const {
    auto it =
      std::lower_bound(token_entries_.begin(), token_entries_.end(), token,
                       [&](const TokenEntry &entry, const std::string &value) {
                         const std::string_view lhs(
                           token_bytes_.data() + entry.offset, entry.length);
                         return lhs < value;
                       });
    if (it == token_entries_.end()) {
      return kInvalidId;
    }
    const std::string_view found(token_bytes_.data() + it->offset, it->length);
    return found == token ? it->id : kInvalidId;
  }

  const MergeEntry *LookupMerge(uint32_t left, uint32_t right) const {
    const uint64_t key = PairKey(left, right);
    auto it =
      std::lower_bound(merges_.begin(), merges_.end(), key,
                       [](const MergeEntry &entry, uint64_t value) {
                         return PairKey(entry.left, entry.right) < value;
                       });
    if (it == merges_.end() || PairKey(it->left, it->right) != key) {
      return nullptr;
    }
    return &(*it);
  }

  bool FindSpecialToken(const std::string &text, size_t offset, uint32_t &id,
                        size_t &len) const {
    for (uint32_t special_id : special_ids_) {
      const uint32_t begin = token_offsets_[special_id];
      const uint32_t end = token_offsets_[special_id + 1];
      const size_t token_len = end - begin;
      if (offset + token_len <= text.size() &&
          text.compare(offset, token_len, token_bytes_, begin, token_len) ==
            0) {
        id = special_id;
        len = token_len;
        return true;
      }
    }
    return false;
  }

  void EncodeOrdinary(const std::string &text,
                      std::vector<int32_t> &ids) const {
    if (text.empty()) {
      return;
    }

    if (variant_ == BPEVariant::ByteLevel) {
      for (const std::string &piece : GPTSplit(text, digit_group_size_)) {
        EncodeBPE(ByteLevelEncode(piece), ids);
      }
    } else {
      EncodeBPE(SpaceReplacementInitialTokens(text), ids);
    }
  }

  void EncodeBPE(const std::vector<uint32_t> &initial,
                 std::vector<int32_t> &ids) const {
    if (initial.empty()) {
      return;
    }

    std::vector<uint32_t> word = initial;
    while (word.size() > 1) {
      uint32_t best_rank = kInvalidRank;
      uint32_t best_left = kInvalidId;
      uint32_t best_right = kInvalidId;
      uint32_t best_merged = kInvalidId;

      for (size_t i = 0; i + 1 < word.size(); ++i) {
        const MergeEntry *merge = LookupMerge(word[i], word[i + 1]);
        if (merge != nullptr && merge->rank < best_rank) {
          best_rank = merge->rank;
          best_left = merge->left;
          best_right = merge->right;
          best_merged = merge->merged;
        }
      }

      if (best_rank == kInvalidRank) {
        break;
      }

      std::vector<uint32_t> merged;
      merged.reserve(word.size());
      for (size_t i = 0; i < word.size();) {
        if (i + 1 < word.size() && word[i] == best_left &&
            word[i + 1] == best_right) {
          merged.push_back(best_merged);
          i += 2;
        } else {
          merged.push_back(word[i++]);
        }
      }
      word.swap(merged);
    }

    for (uint32_t id : word) {
      ids.push_back(static_cast<int32_t>(id));
    }
  }

  std::vector<uint32_t> ByteLevelEncode(const std::string &piece) const {
    std::vector<uint32_t> ids;
    ids.reserve(piece.size());
    for (unsigned char byte : piece) {
      const uint32_t id = byte_token_ids_[byte];
      if (id == kInvalidId) {
        throw std::runtime_error("BPE byte token is missing from vocab");
      }
      ids.push_back(id);
    }
    return ids;
  }

  std::vector<uint32_t>
  SpaceReplacementInitialTokens(const std::string &text) const {
    std::string normalized;
    normalized.reserve(text.size());
    for (char c : text) {
      if (c == ' ') {
        normalized += "\xE2\x96\x81";
      } else {
        normalized.push_back(c);
      }
    }

    std::vector<uint32_t> ids;
    for (size_t i = 0; i < normalized.size();) {
      size_t next = i;
      DecodeUtf8At(normalized, i, next);
      const std::string token = normalized.substr(i, next - i);
      const uint32_t id = LookupToken(token);
      if (id != kInvalidId) {
        ids.push_back(id);
      } else if (unk_id_ != kInvalidId) {
        for (size_t j = i; j < next; ++j) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "<0x%02X>",
                        static_cast<unsigned char>(normalized[j]));
          const uint32_t byte_id = LookupToken(buf);
          ids.push_back(byte_id == kInvalidId ? unk_id_ : byte_id);
        }
      }
      i = next;
    }
    return ids;
  }

  void BuildByteMaps() {
    std::fill(std::begin(byte_token_ids_), std::end(byte_token_ids_),
              kInvalidId);
    byte_decoder_.clear();

    std::vector<uint32_t> bytes;
    for (uint32_t b = 33; b <= 126; ++b) {
      bytes.push_back(b);
    }
    for (uint32_t b = 161; b <= 172; ++b) {
      bytes.push_back(b);
    }
    for (uint32_t b = 174; b <= 255; ++b) {
      bytes.push_back(b);
    }

    bool present[256] = {};
    for (uint32_t b : bytes) {
      present[b] = true;
    }

    uint32_t n = 0;
    std::vector<uint32_t> codepoints = bytes;
    for (uint32_t b = 0; b < 256; ++b) {
      if (!present[b]) {
        bytes.push_back(b);
        codepoints.push_back(256 + n++);
      }
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
      std::string token;
      AppendUtf8(token, codepoints[i]);
      const uint32_t id = LookupToken(token);
      if (id != kInvalidId) {
        byte_token_ids_[bytes[i]] = id;
      }
      byte_decoder_[codepoints[i]] = static_cast<unsigned char>(bytes[i]);
    }
  }

  void DecodeByteLevelToken(const std::string &token, std::string &out) const {
    for (size_t i = 0; i < token.size();) {
      size_t next = i;
      const uint32_t cp = DecodeUtf8At(token, i, next);
      auto it = byte_decoder_.find(cp);
      if (it != byte_decoder_.end()) {
        out.push_back(static_cast<char>(it->second));
      } else {
        out += token.substr(i, next - i);
      }
      i = next;
    }
  }

  void DecodeSpaceReplacementToken(const std::string &token,
                                   std::string &out) const {
    for (size_t i = 0; i < token.size();) {
      if (token.compare(i, 3, "\xE2\x96\x81") == 0) {
        out.push_back(' ');
        i += 3;
      } else if (i + 6 <= token.size() && token.compare(i, 3, "<0x") == 0 &&
                 token[i + 5] == '>') {
        const std::string hex = token.substr(i + 3, 2);
        char *end = nullptr;
        const long value = std::strtol(hex.c_str(), &end, 16);
        if (end != nullptr && *end == '\0') {
          out.push_back(static_cast<char>(value));
          i += 6;
        } else {
          out.push_back(token[i++]);
        }
      } else {
        out.push_back(token[i++]);
      }
    }
  }

  BPEVariant variant_ = BPEVariant::ByteLevel;
  uint32_t digit_group_size_ = 1;
  uint32_t unk_id_ = kInvalidId;
  std::vector<uint32_t> token_offsets_;
  std::string token_bytes_;
  std::vector<TokenEntry> token_entries_;
  std::vector<MergeEntry> merges_;
  std::vector<uint32_t> special_ids_;
  std::vector<uint32_t> prefix_ids_;
  uint32_t byte_token_ids_[256];
  std::unordered_map<uint32_t, unsigned char> byte_decoder_;
};

} // namespace

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobBPEJSON(const std::string &json_blob) {
  return std::make_unique<CompactBPETokenizer>(json_blob);
}

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobBPECache(const std::string &cache_blob) {
  return std::make_unique<CompactBPETokenizer>(cache_blob, true);
}

} // namespace tokenizers
