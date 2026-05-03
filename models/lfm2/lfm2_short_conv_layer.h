// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_short_conv_layer.h
 * @brief  LFM2 gated short convolution core layer.
 */

#ifndef __LFM2_SHORT_CONV_LAYER_H__
#define __LFM2_SHORT_CONV_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace quick_dot_ai {

namespace props {

class Lfm2ConvLCache : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "l_cache";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief LFM2 short convolution core.
 *
 * The input is the output of HF's in_proj with shape (..., 3 * hidden_size):
 * B, C, x are split along the feature axis, B and x are multiplied, a causal
 * depthwise convolution is applied, and the result is gated by C.
 */
WIN_EXPORT class Lfm2ShortConvLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Lfm2ShortConvLayer();
  WIN_EXPORT ~Lfm2ShortConvLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override {
    return Lfm2ShortConvLayer::type;
  }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "lfm2_short_conv";

private:
  std::array<unsigned int, 1> wt_idx;
  std::array<unsigned int, 1> tensor_idx;
  std::tuple<props::Lfm2ConvLCache> conv_props;
  unsigned int hidden_size = 0;
  unsigned int l_cache = 3;
};

} // namespace quick_dot_ai

#endif /* __LFM2_SHORT_CONV_LAYER_H__ */
