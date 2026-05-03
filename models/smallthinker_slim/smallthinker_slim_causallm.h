// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_slim_causallm.h
 * @brief  SmallThinker slim causal language model.
 */

#ifndef __SMALLTHINKER_SLIM_CAUSAL_LM_H__
#define __SMALLTHINKER_SLIM_CAUSAL_LM_H__

#include <smallthinker_causallm.h>

namespace quick_dot_ai {

/**
 * @brief SmallThinker slim CausalLM class with on-the-fly expert loading
 */
class SmallThinkerSlimCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerSlimForCausalLM";

  SmallThinkerSlimCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~SmallThinkerSlimCausalLM() = default;

protected:
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;
};

} // namespace quick_dot_ai

#endif /* __SMALLTHINKER_SLIM_CAUSAL_LM_H__ */
