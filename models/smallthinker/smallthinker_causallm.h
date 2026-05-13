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

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void constructModel() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

  virtual const char *getMoELayerType() const { return "smallthinker_moe"; }

  virtual bool usesMoECache() const { return false; }

  static json &normalizeConfig(json &cfg);

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
  unsigned int MOE_CACHE_SIZE;
  bool ROUTER_APPLY_SOFTMAX;
  std::string router_input_name_;
  std::vector<bool> rope_layout_;
  std::vector<bool> sliding_window_layout_;
};

/**
 * @brief SmallThinkerSlimCausalLM class
 * @note  Uses virtual expert weights for on-demand MoE loading.
 */
class SmallThinkerSlimCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures = "SmallThinkerSlimForCausalLM";

  SmallThinkerSlimCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_slim";
  }

  void registerCustomLayers() override;
};

/**
 * @brief SmallThinkerCachedSlimCausalLM class
 * @note  Keeps a bounded LRU cache of active virtual expert weights.
 */
class SmallThinkerCachedSlimCausalLM : public SmallThinkerCausalLM {
public:
  static constexpr const char *architectures =
    "SmallThinkerCachedSlimForCausalLM";

  SmallThinkerCachedSlimCausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
    Transformer(normalizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    SmallThinkerCausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~SmallThinkerCachedSlimCausalLM() = default;

protected:
  const char *getMoELayerType() const override {
    return "smallthinker_moe_cached_slim";
  }

  bool usesMoECache() const override { return true; }

  void registerCustomLayers() override;
};

} // namespace quick_dot_ai

#endif /* __SMALLTHINKER_CAUSAL_LM_H__ */
