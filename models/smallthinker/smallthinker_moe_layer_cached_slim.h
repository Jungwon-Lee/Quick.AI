// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_cached_slim.h
 * @date   12 May 2026
 * @brief  SmallThinker MoE layer with an on-demand expert cache.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__
#ifdef __cplusplus

#include <list>
#include <mutex>
#include <smallthinker_moe_layer.h>
#include <unordered_map>

namespace quick_dot_ai {

namespace props {

/**
 * @brief Number of virtual experts to keep active in the layer cache.
 */
class MoECacheSize : public nntrainer::PositiveIntegerProperty {
public:
  MoECacheSize(unsigned int value = 16) { set(value); }
  static constexpr const char *key =
    "moe_cache_size";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

} // namespace props

/**
 * @class   SmallThinkerCachedSlimMoELayer
 * @brief   SmallThinker MoE layer with bounded LRU cached virtual experts.
 */
class SmallThinkerCachedSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of SmallThinker cached slim MoE layer.
   */
  SmallThinkerCachedSlimMoELayer();

  /**
   * @brief Destructor of SmallThinker cached slim MoE layer.
   */
  ~SmallThinkerCachedSlimMoELayer() = default;

  SmallThinkerCachedSlimMoELayer(SmallThinkerCachedSlimMoELayer &&rhs) = delete;
  SmallThinkerCachedSlimMoELayer &
  operator=(SmallThinkerCachedSlimMoELayer &&rhs) = delete;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SmallThinkerCachedSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type =
    "smallthinker_moe_cached_slim"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token */
  unsigned int cache_size;       /**< max active virtual experts */
  bool router_apply_softmax;     /**< whether router uses softmax or sigmoid */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             props::MoERouterApplySoftmax, props::MoECacheSize>
    moe_props;

  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;

  std::list<int> loaded_expert_deque;
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::vector<bool> need_load;
  std::mutex cache_mutex;

  unsigned int gate_idx;
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  void route_and_compute(nntrainer::RunLayerContext &context,
                         nntrainer::Tensor &input,
                         nntrainer::Tensor &router_input,
                         nntrainer::Tensor &output,
                         nntrainer::Tensor &router_logits,
                         nntrainer::Tensor &expert_mask,
                         unsigned int hidden_size, unsigned int total_tokens);

  void activate_or_touch_expert(nntrainer::RunLayerContext &context,
                                unsigned int expert_idx);

  void evict_experts(nntrainer::RunLayerContext &context);

  void deactivate_expert(nntrainer::RunLayerContext &context,
                         unsigned int expert_idx);

  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};

} // namespace quick_dot_ai

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__ */
