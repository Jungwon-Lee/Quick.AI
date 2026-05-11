// SPDX-License-Identifier: Apache-2.0
/**
 * @file   chat_template.h
 * @brief  Hugging Face chat template adapter for OpenAI-style chat inputs.
 */
#ifndef __CHAT_TEMPLATE_H__
#define __CHAT_TEMPLATE_H__

#include "json.hpp"

#include <memory>
#include <string>

namespace quick_dot_ai {

class ChatTemplate {
public:
  enum class Builtin { FunctionGemma };

  struct Options {
    enum class GenerationPromptMode { Auto, Always, Never };
    enum class DeveloperRolePolicy { Auto, Preserve, MergeIntoSystem };

    GenerationPromptMode generation_prompt = GenerationPromptMode::Auto;
    DeveloperRolePolicy developer_role_policy = DeveloperRolePolicy::Auto;
    bool continue_final_message = false;
    std::string template_name;
  };

  static bool Exists(const std::string &model_path);
  static ChatTemplate Load(const std::string &model_path);
  static ChatTemplate LoadBuiltin(Builtin builtin);

  ChatTemplate(ChatTemplate &&) noexcept;
  ChatTemplate &operator=(ChatTemplate &&) noexcept;
  ChatTemplate(const ChatTemplate &) = delete;
  ChatTemplate &operator=(const ChatTemplate &) = delete;
  ~ChatTemplate();

  std::string apply(const nlohmann::json &request) const;
  std::string apply(const nlohmann::json &request,
                    const Options &options) const;

  const std::string &sourcePath() const;

private:
  struct Impl;

  explicit ChatTemplate(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace quick_dot_ai

#endif // __CHAT_TEMPLATE_H__
