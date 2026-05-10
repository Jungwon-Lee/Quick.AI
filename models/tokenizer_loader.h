// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   tokenizer_loader.h
 * @brief  Tokenizer loading helpers for Quick.AI models
 */
#ifndef __TOKENIZER_LOADER_H__
#define __TOKENIZER_LOADER_H__

#include <memory>

#include "json.hpp"
#include <tokenizers/tokenizers_cpp.h>

namespace quick_dot_ai {

std::unique_ptr<tokenizers::Tokenizer> LoadTokenizer(nlohmann::json &nntr_cfg);

} // namespace quick_dot_ai

#endif // __TOKENIZER_LOADER_H__
