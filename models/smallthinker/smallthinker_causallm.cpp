/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   smallthinker_causallm.cpp
 * @date   28 April 2026
 * @brief  SmallThinker causal language model.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 */

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <smallthinker_causallm.h>
#include <smallthinker_moe_layer.h>
#include <stdexcept>

namespace quick_dot_ai {

namespace {

std::vector<bool> parseLayerLayout(const json &cfg, const char *key,
                                   int num_layers, bool default_value) {
  std::vector<bool> layout(num_layers, default_value);
  if (!cfg.contains(key) || !cfg[key].is_array())
    return layout;

  const int limit = std::min<int>(num_layers, cfg[key].size());
  for (int i = 0; i < limit; ++i) {
    const json &value = cfg[key][i];
    if (value.is_boolean())
      layout[i] = value.get<bool>();
    else if (value.is_number_integer())
      layout[i] = value.get<int>() != 0;
  }
  return layout;
}

} // namespace

json &SmallThinkerCausalLM::normalizeConfig(json &cfg) {
  if (!cfg.contains("intermediate_size") &&
      cfg.contains("moe_ffn_hidden_size")) {
    cfg["intermediate_size"] = cfg["moe_ffn_hidden_size"];
  }
  if (!cfg.contains("sliding_window") && cfg.contains("sliding_window_size")) {
    cfg["sliding_window"] = cfg["sliding_window_size"];
  }
  if (!cfg.contains("tie_word_embeddings") ||
      !cfg["tie_word_embeddings"].is_boolean()) {
    cfg["tie_word_embeddings"] = true;
  }
  if (cfg.contains("sliding_window_pattern") &&
      !cfg["sliding_window_pattern"].is_number_unsigned()) {
    cfg["sliding_window_pattern"] = 1;
  }

  return cfg;
}

void SmallThinkerCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("sliding_window_layout") &&
      cfg["sliding_window_layout"].is_array() &&
      std::none_of(cfg["sliding_window_layout"].begin(),
                   cfg["sliding_window_layout"].end(), [](const json &enabled) {
                     return enabled.is_number_integer() &&
                            enabled.get<int>() != 0;
                   })) {
    SLIDING_WINDOW = UINT_MAX;
  }

  try {
    NUM_EXPERTS = cfg["moe_num_primary_experts"];
    NUM_EXPERTS_PER_TOK = cfg["moe_num_active_primary_experts"];
    INTERMEDIATE_SIZE = cfg["moe_ffn_hidden_size"];
    ROUTER_APPLY_SOFTMAX =
      cfg.contains("moe_primary_router_apply_softmax")
        ? cfg["moe_primary_router_apply_softmax"].get<bool>()
        : false;
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "SmallThinker: required MoE config keys are missing");
  }

  rope_layout_ = parseLayerLayout(cfg, "rope_layout", NUM_LAYERS, true);
  sliding_window_layout_ =
    parseLayerLayout(cfg, "sliding_window_layout", NUM_LAYERS, false);
}

void SmallThinkerCausalLM::constructModel() {

  std::vector<LayerHandle> layers;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  layers.push_back(createLayer(
    "embedding_layer",
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));

  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoderBlock(0, "embedding0");
    else
      transformer = createTransformerDecoderBlock(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}

std::vector<LayerHandle>
SmallThinkerCausalLM::createTransformerDecoderBlock(const int layer_id,
                                                    std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  router_input_name_ = input_name;
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());
  router_input_name_.clear();

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle>
SmallThinkerCausalLM::createAttention(const int layer_id, int seq_len,
                                      int n_heads, int head_dim,
                                      std::string query_name,
                                      std::string key_name,
                                      std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones")}));

  const bool use_sliding_window =
    layer_id < static_cast<int>(sliding_window_layout_.size())
      ? sliding_window_layout_[layer_id]
      : false;
  const bool use_rope = layer_id < static_cast<int>(rope_layout_.size())
                          ? rope_layout_[layer_id]
                          : true;

  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", use_sliding_window ? SLIDING_WINDOW : UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("use_rope", use_rope ? "true" : "false"),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("input_layers", A), withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle>
SmallThinkerCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                std::string input_name) {

  std::vector<LayerHandle> layers;
  const std::string router_input =
    router_input_name_.empty() ? input_name : router_input_name_;

  layers.push_back(createLayer(
    "smallthinker_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("input_layers", {input_name, router_input}),
     withKey("unit", hidden_dim), withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "relu"),
     withKey("moe_router_apply_softmax",
             ROUTER_APPLY_SOFTMAX ? "true" : "false")}));

  return layers;
}

void SmallThinkerCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<quick_dot_ai::SmallThinkerMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace quick_dot_ai
