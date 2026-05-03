// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   smallthinker_moe_layer_cached_slim.h
 * @brief  SmallThinker MoE layer with virtual expert tensors and LRU cache.
 */

#ifndef __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__
#define __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__

#include <smallthinker_moe_layer_slim.h>

namespace quick_dot_ai {

class SmallThinkerCachedSlimMoELayer : public SmallThinkerSlimMoELayer {
public:
  SmallThinkerCachedSlimMoELayer();

  static constexpr const char *type = "smallthinker_moe_cached_slim";
};

} // namespace quick_dot_ai

#endif /* __SMALLTHINKER_MOE_LAYER_CACHED_SLIM_H__ */
