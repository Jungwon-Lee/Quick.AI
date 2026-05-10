/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizer_cache_util.h
 * \brief Common helpers for compact tokenizer cache blobs
 */
#ifndef TOKENIZER_CACHE_UTIL_H_
#define TOKENIZER_CACHE_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tokenizers {
namespace cache_util {

enum class CacheKind : uint32_t {
  WordPiece = 1,
  BPE = 2,
};

constexpr char kCacheMagic[] = {'Q', 'A', 'I', 'T', 'O', 'K', 'C', 'H'};
constexpr size_t kCacheMagicSize = sizeof(kCacheMagic);
constexpr uint32_t kCacheVersion = 1;

inline std::string InvalidCacheMessage(const char *cache_name,
                                       const char *reason) {
  return std::string("Invalid ") + cache_name + " cache: " + reason;
}

inline void AppendU32(std::string &out, uint32_t value) {
  out.push_back(static_cast<char>(value & 0xff));
  out.push_back(static_cast<char>((value >> 8) & 0xff));
  out.push_back(static_cast<char>((value >> 16) & 0xff));
  out.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void CheckMagic(const std::string &blob, const char *magic,
                       size_t magic_size, const char *cache_name) {
  if (blob.size() < magic_size ||
      std::memcmp(blob.data(), magic, magic_size) != 0) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "bad magic"));
  }
}

inline void AppendHeader(std::string &out, CacheKind kind) {
  out.append(kCacheMagic, kCacheMagicSize);
  AppendU32(out, kCacheVersion);
  AppendU32(out, static_cast<uint32_t>(kind));
}

inline uint32_t ReadU32(const std::string &blob, size_t &offset,
                        const char *cache_name) {
  if (offset + 4 > blob.size()) {
    throw std::runtime_error(InvalidCacheMessage(cache_name, "truncated u32"));
  }

  const unsigned char *p =
    reinterpret_cast<const unsigned char *>(blob.data() + offset);
  offset += 4;
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

inline std::string ReadBytes(const std::string &blob, size_t &offset,
                             size_t len, const char *cache_name) {
  if (offset + len > blob.size()) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "truncated bytes"));
  }

  std::string out(blob.data() + offset, len);
  offset += len;
  return out;
}

inline size_t ReadHeader(const std::string &blob, CacheKind expected_kind,
                         const char *cache_name) {
  CheckMagic(blob, kCacheMagic, kCacheMagicSize, cache_name);

  size_t offset = kCacheMagicSize;
  const uint32_t version = ReadU32(blob, offset, cache_name);
  if (version != kCacheVersion) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "unsupported version"));
  }

  const auto kind = static_cast<CacheKind>(ReadU32(blob, offset, cache_name));
  if (kind != expected_kind) {
    throw std::runtime_error(
      InvalidCacheMessage(cache_name, "wrong tokenizer kind"));
  }

  return offset;
}

} // namespace cache_util
} // namespace tokenizers

#endif // TOKENIZER_CACHE_UTIL_H_
