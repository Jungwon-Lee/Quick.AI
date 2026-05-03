// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lfm2_short_conv_layer.cpp
 * @brief  LFM2 gated short convolution core layer.
 */

#include <lfm2_short_conv_layer.h>

#include <algorithm>
#include <cstring>
#include <nntrainer_error.h>

namespace quick_dot_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum Lfm2ShortConvWeights { conv_weight };
enum Lfm2ShortConvTensors { conv_state };

Lfm2ShortConvLayer::Lfm2ShortConvLayer() :
  Layer(), wt_idx({0}), tensor_idx({0}), conv_props(props::Lfm2ConvLCache()) {}

void Lfm2ShortConvLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "LFM2 short convolution layer takes one projected input";

  const auto &input_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(input_dim.width() % 3 != 0, std::invalid_argument)
    << "LFM2 short convolution input width must be 3 * hidden_size";

  hidden_size = input_dim.width() / 3;
  l_cache = std::get<props::Lfm2ConvLCache>(conv_props).get();

  std::vector<nntrainer::TensorDim> output_dims(1, input_dim);
  output_dims[0].width(hidden_size);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

  nntrainer::TensorDim weight_dim(
    1, 1, hidden_size, l_cache,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     ml::train::TensorDim::DataType::FP32));
  wt_idx[Lfm2ShortConvWeights::conv_weight] = context.requestWeight(
    weight_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv_weight", false);

  nntrainer::TensorDim state_dim(
    input_dim.batch(), 1, l_cache, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     ml::train::TensorDim::DataType::FP32));
  tensor_idx[Lfm2ShortConvTensors::conv_state] = context.requestTensor(
    state_dim, "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void Lfm2ShortConvLayer::exportTo(nntrainer::Exporter &exporter,
                                  const ml::train::ExportMethods &method) const {
  exporter.saveResult(conv_props, method, this);
}

void Lfm2ShortConvLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[lfm2_short_conv] Unknown Layer Properties count "
    << std::to_string(values.size());
}

void Lfm2ShortConvLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

void Lfm2ShortConvLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight =
    context.getWeight(wt_idx[Lfm2ShortConvWeights::conv_weight]);
  nntrainer::Tensor &state =
    context.getTensor(tensor_idx[Lfm2ShortConvTensors::conv_state]);

  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "LFM2 short convolution currently supports FP32 activations only";

  const unsigned int step = to - from;
  const unsigned int batch = input.batch();
  const float *in = input.getData<float>();
  float *out = output.getData<float>();
  const float *w = weight.getData<float>();
  float *cache = state.getData<float>();
  std::vector<float> bx(static_cast<size_t>(batch) * step * hidden_size);

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int h = 0; h < step; ++h) {
      for (unsigned int d = 0; d < hidden_size; ++d) {
        const size_t in_base = input.getIndex(b, 0, h, 0);
        const float gate_b = in[in_base + d];
        const float gate_c = in[in_base + hidden_size + d];
        const float x = in[in_base + 2 * hidden_size + d];
        bx[(static_cast<size_t>(b) * step + h) * hidden_size + d] =
          gate_b * x;

        float conv = 0.0f;
        for (unsigned int k = 0; k < l_cache; ++k) {
          const int rel = static_cast<int>(h) + static_cast<int>(k) -
                          static_cast<int>(l_cache) + 1;
          float src = 0.0f;
          if (rel >= 0) {
            src = bx[(static_cast<size_t>(b) * step +
                      static_cast<unsigned int>(rel)) *
                       hidden_size +
                     d];
          } else if (from != 0) {
            const unsigned int cache_h =
              static_cast<unsigned int>(static_cast<int>(l_cache) + rel);
            src = cache[state.getIndex(b, 0, cache_h, d)];
          }
          conv += src * w[weight.getIndex(0, 0, d, k)];
        }
        out[output.getIndex(b, 0, h, d)] = gate_c * conv;
      }
    }
  }

  for (unsigned int b = 0; b < batch; ++b) {
    std::vector<float> new_cache(l_cache * hidden_size, 0.0f);
    for (unsigned int cache_h = 0; cache_h < l_cache; ++cache_h) {
      const int rel = static_cast<int>(step) - static_cast<int>(l_cache) +
                      static_cast<int>(cache_h);
      for (unsigned int d = 0; d < hidden_size; ++d) {
        if (rel >= 0) {
          new_cache[cache_h * hidden_size + d] =
            bx[(static_cast<size_t>(b) * step + static_cast<unsigned int>(rel)) *
                 hidden_size +
               d];
        } else if (from != 0) {
          const unsigned int old_h =
            static_cast<unsigned int>(static_cast<int>(l_cache) + rel);
          new_cache[cache_h * hidden_size + d] =
            cache[state.getIndex(b, 0, old_h, d)];
        }
      }
    }
    for (unsigned int cache_h = 0; cache_h < l_cache; ++cache_h) {
      for (unsigned int d = 0; d < hidden_size; ++d) {
        cache[state.getIndex(b, 0, cache_h, d)] =
          new_cache[cache_h * hidden_size + d];
      }
    }
  }
}

void Lfm2ShortConvLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto input_dim = input_dimensions[0];
  auto output_dim = input_dim;
  output_dim.width(input_dim.width() / 3);
  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

void Lfm2ShortConvLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

} // namespace quick_dot_ai
