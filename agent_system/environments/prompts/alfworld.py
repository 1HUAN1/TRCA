# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions are listed below. Each line is one complete valid action:
{admissible_actions}

Now choose the single best action for this step.
First output one <think>...</think> block with brief English reasoning.
Then output exactly one <action>...</action> block.
The text inside <action> must be copied verbatim from exactly one line in the admissible actions list.
Do not invent, translate, summarize, or modify the action.
Do not output Chinese, quotes, bullets, numbering, an Action: prefix, multiple actions, or any text after </action>.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions are listed below. Each line is one complete valid action:
{admissible_actions}

Now choose the single best action for this step.
First output one <think>...</think> block with brief English reasoning.
Then output exactly one <action>...</action> block.
The text inside <action> must be copied verbatim from exactly one line in the admissible actions list.
Do not invent, translate, summarize, or modify the action.
Do not output Chinese, quotes, bullets, numbering, an Action: prefix, multiple actions, or any text after </action>.
"""

ALFWORLD_ACTION_ONLY_TAGGED_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

Output exactly one line in this format:
<action>one admissible action copied verbatim</action>

Rules:
The action must be copied verbatim from the admissible actions list, including spaces.
Do not output <think>, explanations, Chinese, multiple actions, or text after </action>.
"""

ALFWORLD_ACTION_ONLY_TAGGED_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Recent history: {action_history}
Current observation: {current_observation}
Your admissible actions are: [{admissible_actions}].

Output exactly one line in this format:
<action>one admissible action copied verbatim</action>

Rules:
The action must be copied verbatim from the admissible actions list, including spaces.
Do not output <think>, explanations, Chinese, multiple actions, or text after </action>.
"""

# Backward-compatible names for the previous action_only tagged prompt.
ALFWORLD_ACTION_ONLY_TEMPLATE_NO_HIS = ALFWORLD_ACTION_ONLY_TAGGED_TEMPLATE_NO_HIS
ALFWORLD_ACTION_ONLY_TEMPLATE = ALFWORLD_ACTION_ONLY_TAGGED_TEMPLATE

ALFWORLD_ACTION_ONLY_BARE_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions are listed below. Each line is one complete valid action:
{admissible_actions}

Output exactly one line copied from the admissible actions list above.
The whole response must be one complete listed action.
If the action you want is not listed, do not write it; choose one listed action instead.
Do not change object names, numbers, prepositions, or word order.
Do not use XML tags, quotes, bullets, numbering, punctuation, or an Action: prefix.
Do not write explanations, multiple actions, Chinese, or any text before or after the action.
"""

ALFWORLD_ACTION_ONLY_BARE_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Recent history: {action_history}
Current observation: {current_observation}
Your admissible actions are listed below. Each line is one complete valid action:
{admissible_actions}

Output exactly one line copied from the admissible actions list above.
The whole response must be one complete listed action.
If the action you want is not listed, do not write it; choose one listed action instead.
Do not change object names, numbers, prepositions, or word order.
Do not use XML tags, quotes, bullets, numbering, punctuation, or an Action: prefix.
Do not write explanations, multiple actions, Chinese, or any text before or after the action.
"""
