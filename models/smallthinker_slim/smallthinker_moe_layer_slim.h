// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_slim.h
 * @brief  SmallThinker MoE layers with on-the-fly expert loading.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_SLIM_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <list>
#include <mutex>
#include <smallthinker_moe_layer.h>
#include <unordered_map>

namespace quick_dot_ai {

/**
 * @class   SmallThinkerSlimMoELayer
 * @brief   SmallThinker MoE layer with virtual expert tensors.
 */
class SmallThinkerSlimMoELayer : public nntrainer::LayerImpl {
public:
  SmallThinkerSlimMoELayer();

  ~SmallThinkerSlimMoELayer() = default;

  SmallThinkerSlimMoELayer(SmallThinkerSlimMoELayer &&rhs) = delete;

  SmallThinkerSlimMoELayer &operator=(SmallThinkerSlimMoELayer &&rhs) = delete;

  void finalize(nntrainer::InitLayerContext &context) override;

  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  void calcDerivative(nntrainer::RunLayerContext &context) override;

  void calcGradient(nntrainer::RunLayerContext &context) override;

  void setProperty(const std::vector<std::string> &values) override;

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override { return layer_type; };

  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "smallthinker_moe_slim";

protected:
  explicit SmallThinkerSlimMoELayer(unsigned int cache_capacity,
                                    const char *layer_type);

private:
  unsigned int num_experts;
  unsigned int topk;
  unsigned int cache_capacity;
  bool router_apply_softmax;
  const char *layer_type;
  nntrainer::ActiFunc acti_func;
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             props::MoERouterApplySoftmax>
    moe_props;

  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  std::vector<bool> need_load;
  std::list<int> loaded_experts;
  std::unordered_map<int, std::list<int>::iterator> loaded_expert_iters;
  std::mutex cache_mutex;

  unsigned int gate_idx;
  unsigned int router_logits_idx;

  void run_moe(nntrainer::RunLayerContext &context, nntrainer::Tensor &input,
               nntrainer::Tensor &router_input, nntrainer::Tensor &output,
               nntrainer::Tensor &router_logits);

  void route_tokens(
    nntrainer::RunLayerContext &context, nntrainer::Tensor &router_input,
    nntrainer::Tensor &router_logits, unsigned int total_tokens,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments);

  void activate_expert(nntrainer::RunLayerContext &context,
                       unsigned int expert_idx);

  void release_experts(nntrainer::RunLayerContext &context);

  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  inline void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  bool compute_expert_forward_sparse_generation(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};

} // namespace quick_dot_ai

#endif /* __cplusplus */
#endif /* __SMALLTHINKER_MOE_LAYER_SLIM_H__ */
