// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_cached_slim_causallm.h
 * @brief  SmallThinker cached slim causal language model.
 */

#ifndef __SMALLTHINKER_CACHED_SLIM_CAUSAL_LM_H__
#define __SMALLTHINKER_CACHED_SLIM_CAUSAL_LM_H__

#include <smallthinker_slim_causallm.h>

namespace quick_dot_ai {

/**
 * @brief SmallThinker cached slim CausalLM class with expert cache
 */
class SmallThinkerCachedSlimCausalLM : public SmallThinkerSlimCausalLM {
public:
  static constexpr const char *architectures =
    "SmallThinkerCachedSlimForCausalLM";

  SmallThinkerCachedSlimCausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerSlimCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerCachedSlimCausalLM() = default;

protected:
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;
};

} // namespace quick_dot_ai

#endif /* __SMALLTHINKER_CACHED_SLIM_CAUSAL_LM_H__ */
