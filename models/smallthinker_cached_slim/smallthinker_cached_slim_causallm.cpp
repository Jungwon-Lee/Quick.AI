/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <smallthinker_cached_slim_causallm.h>
#include <smallthinker_moe_layer_cached_slim.h>

#include <iostream>

namespace quick_dot_ai {

std::vector<LayerHandle> SmallThinkerCachedSlimCausalLM::createMlp(
  const int layer_id, int dim, int hidden_dim, std::string input_name) {

  std::vector<LayerHandle> layers;
  const std::string router_input =
    router_input_name_.empty() ? input_name : router_input_name_;

  layers.push_back(createLayer(
    "smallthinker_moe_cached_slim",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("input_layers", {input_name, router_input}),
     withKey("unit", hidden_dim), withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "relu"),
     withKey("moe_router_apply_softmax",
             ROUTER_APPLY_SOFTMAX ? "true" : "false")}));

  return layers;
}

void SmallThinkerCachedSlimCausalLM::registerCustomLayers() {
  SmallThinkerSlimCausalLM::registerCustomLayers();

  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<quick_dot_ai::SmallThinkerCachedSlimMoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace quick_dot_ai
