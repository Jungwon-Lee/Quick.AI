// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_causallm.cpp
 * @brief  LFM2 causal language model.
 */

#include <lfm2_causallm.h>
#include <lfm2_short_conv_layer.h>
#include <reshaped_rms_norm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>

namespace quick_dot_ai {

Lfm2Transformer::Lfm2Transformer(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg) {}

void Lfm2Transformer::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("rope_parameters") &&
      cfg["rope_parameters"].contains("rope_theta")) {
    ROPE_THETA = cfg["rope_parameters"]["rope_theta"].get<unsigned int>();
  }
  if (cfg.contains("norm_eps"))
    NORM_EPS = cfg["norm_eps"].get<float>();
  if (cfg.contains("tie_embedding"))
    TIE_WORD_EMBEDDINGS = cfg["tie_embedding"].get<bool>();
  if (cfg.contains("conv_L_cache"))
    conv_l_cache = cfg["conv_L_cache"].get<unsigned int>();
  if (cfg.contains("layer_types"))
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();

  if (cfg.value("block_auto_adjust_ff_dim", false)) {
    int adjusted = static_cast<int>(2 * INTERMEDIATE_SIZE / 3);
    if (cfg.contains("block_ffn_dim_multiplier") &&
        !cfg["block_ffn_dim_multiplier"].is_null()) {
      adjusted = static_cast<int>(
        cfg["block_ffn_dim_multiplier"].get<float>() * adjusted);
    }
    const int multiple = cfg.value("block_multiple_of", 256);
    INTERMEDIATE_SIZE = multiple * ((adjusted + multiple - 1) / multiple);
  }
}

std::vector<LayerHandle>
Lfm2Transformer::createTransformerDecoderBlock(const int layer_id,
                                               std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_operator_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  const std::string operator_input =
    "layer" + std::to_string(layer_id) + "_operator_norm";
  const bool is_attention =
    layer_types.empty() || layer_types[layer_id] == "full_attention";

  if (is_attention) {
    auto att_layer = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                     operator_input, operator_input,
                                     operator_input);
    layers.insert(layers.end(), att_layer.begin(), att_layer.end());
  } else {
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_conv_in"),
       withKey("unit", 3 * DIM), withKey("disable_bias", "true"),
       withKey("input_layers", operator_input),
       withKey("weight_initializer", "ones")}));
    layers.push_back(createLayer(
      "lfm2_short_conv",
      {withKey("name", "layer" + std::to_string(layer_id) + "_conv"),
       withKey("l_cache", std::to_string(conv_l_cache)),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_conv_in")}));
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_conv_out"),
       withKey("unit", DIM), withKey("disable_bias", "true"),
       withKey("input_layers", "layer" + std::to_string(layer_id) + "_conv"),
       withKey("weight_initializer", "ones")}));
  }

  const std::string operator_output =
    is_attention ? "layer" + std::to_string(layer_id) + "_attention_out"
                 : "layer" + std::to_string(layer_id) + "_conv_out";

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_operator_add"),
     withKey("input_layers", input_name + "," + operator_output)}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_operator_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_operator_add,layer" +
                               std::to_string(layer_id) + "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle> Lfm2Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_layernorm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_layernorm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("input_layers", Q),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("input_layers", K),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("input_layers", {Q_norm, K_norm, V})}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A),
     withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle> Lfm2Transformer::createMlp(const int layer_id, int dim,
                                                    int hidden_dim,
                                                    std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_w1"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_w3"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));
  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_w1," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_w3")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void Lfm2Transformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<quick_dot_ai::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<quick_dot_ai::Lfm2ShortConvLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Lfm2CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Lfm2Transformer::registerCustomLayers();
}

void Lfm2CausalLM::setupParameters(json &cfg, json &generation_cfg,
                                   json &nntr_cfg) {
  Lfm2Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
}

} // namespace quick_dot_ai
