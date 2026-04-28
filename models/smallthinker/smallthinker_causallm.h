// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_causallm.h
 * @date   28 April 2026
 * @brief  SmallThinker causal language model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#ifndef __SMALLTHINKER_CAUSAL_LM_H__
#define __SMALLTHINKER_CAUSAL_LM_H__

#include <causal_lm.h>

namespace quick_dot_ai {

/**
 * @brief SmallThinkerCausalLM class
 * @note  Implements SmallThinker's MoE decoder structure.
 */
class SmallThinkerCausalLM : public CausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerForCausalLM";

  SmallThinkerCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~SmallThinkerCausalLM() = default;

protected:
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void constructModel() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

private:
  static json &normalizeConfig(json &cfg);

  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  bool ROUTER_APPLY_SOFTMAX;
  std::string router_input_name_;
};

} // namespace quick_dot_ai

#endif /* __SMALLTHINKER_CAUSAL_LM_H__ */
