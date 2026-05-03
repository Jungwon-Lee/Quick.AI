/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <node_exporter.h>
#include <omp.h>
#include <smallthinker_moe_layer_slim.h>
#include <stdexcept>

namespace quick_dot_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr float SPARSE_DENSITY_THRESHOLD = 0.4f;

SmallThinkerSlimMoELayer::SmallThinkerSlimMoELayer() :
  SmallThinkerSlimMoELayer(0, SmallThinkerSlimMoELayer::type) {}

SmallThinkerSlimMoELayer::SmallThinkerSlimMoELayer(unsigned int cache_capacity_,
                                                   const char *layer_type_) :
  LayerImpl(),
  num_experts(0),
  topk(0),
  cache_capacity(cache_capacity_),
  router_apply_softmax(true),
  layer_type(layer_type_),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation(),
            props::MoERouterApplySoftmax()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  need_load({}),
  loaded_experts({}),
  loaded_expert_iters({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()) {}

void SmallThinkerSlimMoELayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "SmallThinker slim MoE layer requires expert input and router input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  router_apply_softmax =
    std::get<props::MoERouterApplySoftmax>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);
  need_load.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, true));
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));
    need_load.push_back(true);
  }

  const unsigned total_tokens = in_dim.batch() * in_dim.height();
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void SmallThinkerSlimMoELayer::route_tokens(
  nntrainer::RunLayerContext &context, nntrainer::Tensor &router_input,
  nntrainer::Tensor &router_logits, unsigned int total_tokens,
  std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments) {

  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  router_input.dot(gate_weights, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  if (router_apply_softmax) {
    topk_values.apply(nntrainer::ActiFunc::softmax<float>, topk_values);
  } else {
    float *values_data = topk_values.getData<float>();
    for (unsigned int i = 0; i < total_tokens; ++i) {
      float sum = 0.0f;
      for (unsigned int k = 0; k < topk; ++k) {
        float &value = values_data[i * topk + k];
        value = 1.0f / (1.0f + std::exp(-value));
        sum += value;
      }
      for (unsigned int k = 0; k < topk; ++k) {
        values_data[i * topk + k] /= sum;
      }
    }
  }

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned int expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }
}

void SmallThinkerSlimMoELayer::activate_expert(
  nntrainer::RunLayerContext &context, unsigned int expert_idx) {

  if (cache_capacity == 0) {
    context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
    context.getWeight(expert_up_proj_indices[expert_idx]).activate();
    context.getWeight(expert_down_proj_indices[expert_idx]).activate();
    return;
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  if (need_load[expert_idx]) {
    context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
    context.getWeight(expert_up_proj_indices[expert_idx]).activate();
    context.getWeight(expert_down_proj_indices[expert_idx]).activate();
    need_load[expert_idx] = false;
  } else {
    loaded_experts.erase(loaded_expert_iters[expert_idx]);
  }
  loaded_experts.push_back(static_cast<int>(expert_idx));
  loaded_expert_iters[expert_idx] = --loaded_experts.end();
}

void SmallThinkerSlimMoELayer::release_experts(
  nntrainer::RunLayerContext &context) {

  if (cache_capacity == 0) {
    for (unsigned int i = 0; i < num_experts; ++i) {
      if (!need_load[i]) {
        context.getWeight(expert_gate_proj_indices[i]).deactivate();
        context.getWeight(expert_up_proj_indices[i]).deactivate();
        context.getWeight(expert_down_proj_indices[i]).deactivate();
        need_load[i] = true;
      }
    }
    return;
  }

  while (loaded_experts.size() > cache_capacity) {
    int expert_idx;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      expert_idx = loaded_experts.front();
      loaded_experts.pop_front();
      loaded_expert_iters.erase(expert_idx);
      need_load[expert_idx] = true;
    }
    context.getWeight(expert_gate_proj_indices[expert_idx]).deactivate();
    context.getWeight(expert_up_proj_indices[expert_idx]).deactivate();
    context.getWeight(expert_down_proj_indices[expert_idx]).deactivate();
  }
}

void SmallThinkerSlimMoELayer::run_moe(nntrainer::RunLayerContext &context,
                                       nntrainer::Tensor &input,
                                       nntrainer::Tensor &router_input,
                                       nntrainer::Tensor &output,
                                       nntrainer::Tensor &router_logits) {

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  route_tokens(context, router_input, router_logits, total_tokens,
               expert_assignments);

  if (total_tokens == 1) {
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      activate_expert(context, expert_idx);
      bool used_sparse = compute_expert_forward_sparse_generation(
        input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      if (!used_sparse) {
        compute_expert_forward(
          input, output, assignments,
          context.getWeight(expert_gate_proj_indices[expert_idx]),
          context.getWeight(expert_up_proj_indices[expert_idx]),
          context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
      }
      if (cache_capacity == 0) {
        context.getWeight(expert_gate_proj_indices[expert_idx]).deactivate();
        context.getWeight(expert_up_proj_indices[expert_idx]).deactivate();
        context.getWeight(expert_down_proj_indices[expert_idx]).deactivate();
      }
    }
    release_experts(context);
    output.reshape({batch_size, 1, seq_len, hidden_size});
    input.reshape({batch_size, 1, seq_len, hidden_size});
    router_input.reshape({batch_size, 1, seq_len, hidden_size});
    return;
  }

  std::vector<nntrainer::Tensor> expert_outputs(num_experts);
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
       ++expert_idx) {
    if (!expert_assignments[expert_idx].empty()) {
      expert_outputs[expert_idx] = nntrainer::Tensor(
        total_tokens, 1, 1, hidden_size, output.getTensorType());
      expert_outputs[expert_idx].setZero();
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
       ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    activate_expert(context, expert_idx);
    compute_expert_forward_no_critical(
      input, expert_outputs[expert_idx], assignments,
      context.getWeight(expert_gate_proj_indices[expert_idx]),
      context.getWeight(expert_up_proj_indices[expert_idx]),
      context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
    if (cache_capacity == 0) {
      context.getWeight(expert_gate_proj_indices[expert_idx]).deactivate();
      context.getWeight(expert_up_proj_indices[expert_idx]).deactivate();
      context.getWeight(expert_down_proj_indices[expert_idx]).deactivate();
    }
  }

  release_experts(context);

  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
       ++expert_idx) {
    if (!expert_assignments[expert_idx].empty()) {
      output.add_i(expert_outputs[expert_idx]);
    }
  }

  output.reshape({batch_size, 1, seq_len, hidden_size});
  input.reshape({batch_size, 1, seq_len, hidden_size});
  router_input.reshape({batch_size, 1, seq_len, hidden_size});
}

void SmallThinkerSlimMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                          bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input = context.getInput(1);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);

  run_moe(context, input, router_input, output, router_logits);
}

void SmallThinkerSlimMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_input_ = context.getInput(1);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);
  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto router_input = router_input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    run_moe(context, input, router_input, output, router_logits);
  }
}

inline void SmallThinkerSlimMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();
  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());
  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);
    token_input.dot(gate_proj, gate_out);
    acti_func.run_fn(gate_out, acti_out);
    token_input.dot(up_proj, up_out);
    acti_out.multiply_i(up_out);

    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);
    token_expert_output.multiply_i(weight);
    nntrainer::Tensor token_output = expert_output.getSharedDataTensor(
      token_output_dim, token_idx * hidden_size, true);
    token_output.add_i(token_expert_output);
  }

  output.add_i(expert_output);
}

inline void SmallThinkerSlimMoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();
  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_idx * hidden_size, true);

    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);
    token_input.dot(gate_proj, gate_out);
    acti_func.run_fn(gate_out, acti_out);
    token_input.dot(up_proj, up_out);
    acti_out.multiply_i(up_out);

    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);
    token_expert_output.multiply_i(weight);
    nntrainer::Tensor token_output = expert_output.getSharedDataTensor(
      token_output_dim, token_idx * hidden_size, true);
    token_output.add_i(token_expert_output);
  }
}

bool SmallThinkerSlimMoELayer::compute_expert_forward_sparse_generation(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  if (token_assignments.size() != 1 ||
      input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      gate_proj.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      up_proj.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      down_proj.getDataType() != ml::train::TensorDim::DataType::FP32) {
    return false;
  }

  const unsigned intermediate_size = gate_proj.width();
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  const unsigned token_idx = token_assignments[0].first;
  const float router_weight = token_assignments[0].second;
  nntrainer::Tensor token_input =
    input.getSharedDataTensor(token_input_dim, token_idx * hidden_size, true);
  nntrainer::Tensor gate_out(intermediate_dim);
  token_input.dot(gate_proj, gate_out);

  const float *gate_data = gate_out.getData<float>();
  std::vector<unsigned int> active_indices;
  active_indices.reserve(intermediate_size);
  for (unsigned int j = 0; j < intermediate_size; ++j) {
    if (gate_data[j] > 0.0f) {
      active_indices.push_back(j);
    }
  }

  if (active_indices.empty())
    return true;

  const float density =
    static_cast<float>(active_indices.size()) / intermediate_size;
  if (density > SPARSE_DENSITY_THRESHOLD)
    return false;

  std::vector<float> output_buffer(hidden_size, 0.0f);
  const float *input_data = token_input.getData<float>();

  for (unsigned int j : active_indices) {
    float up_value = 0.0f;
    for (unsigned int h = 0; h < hidden_size; ++h) {
      up_value += input_data[h] * up_proj.getValue<float>(0, 0, h, j);
    }

    const float hidden_value = gate_data[j] * up_value * router_weight;
    for (unsigned int h = 0; h < hidden_size; ++h) {
      output_buffer[h] += hidden_value * down_proj.getValue<float>(0, 0, j, h);
    }
  }

  nntrainer::Tensor token_expert_output(token_output_dim);
  float *expert_data = token_expert_output.getData<float>();
  std::copy(output_buffer.begin(), output_buffer.end(), expert_data);
  nntrainer::Tensor token_output =
    output.getSharedDataTensor(token_output_dim, token_idx * hidden_size, true);
  token_output.add_i(token_expert_output);
  return true;
}

void SmallThinkerSlimMoELayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void SmallThinkerSlimMoELayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void SmallThinkerSlimMoELayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void SmallThinkerSlimMoELayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace quick_dot_ai
