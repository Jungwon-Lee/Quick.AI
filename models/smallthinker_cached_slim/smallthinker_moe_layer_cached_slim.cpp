/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <smallthinker_moe_layer_cached_slim.h>

namespace quick_dot_ai {

SmallThinkerCachedSlimMoELayer::SmallThinkerCachedSlimMoELayer() :
  SmallThinkerSlimMoELayer(32, SmallThinkerCachedSlimMoELayer::type) {}

} // namespace quick_dot_ai
