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

SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_MULTIHOP_BRIDGE_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

You have {remaining_actions} action(s) left, including the current action.
For a compositional question, solve one relation at a time: first use a short, focused query to identify the bridge entity, then use that concrete entity together with the still-missing relation in the next query. Do not merely repeat the full question or an earlier query. A direct answer or a single search is allowed when the evidence is already sufficient; do not force two searches.

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
Inside <think>, never write literal action tags. After </think>, emit exactly one complete action block and no additional action block.
After completing your reasoning, choose exactly one of the following actions (do not perform both):
(1) If evidence for a required relation is missing and more than one action remains, call the search engine using: <search> your focused query </search>.
(2) If the evidence is sufficient, provide the final answer using: <answer> shortest answer at the granularity requested </answer>.
The <answer> block must contain only the answer, with no explanation, full-sentence restatement, or unrequested title/date detail.
If exactly one action remains, do not search because a last-action search cannot be observed; return the best supported final answer now.
"""

SEARCH_MULTIHOP_BRIDGE_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

You have {remaining_actions} action(s) left, including the current action.
For a compositional question, solve one relation at a time: first use a short, focused query to identify the bridge entity, then use that concrete entity together with the still-missing relation in the next query. Do not merely repeat the full question or an earlier query. A direct answer or a single search is allowed when the evidence is already sufficient; do not force two searches.

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
Inside <think>, never write literal action tags. After </think>, emit exactly one complete action block and no additional action block.
After completing your reasoning, choose exactly one of the following actions (do not perform both):
(1) If evidence for a required relation is missing and more than one action remains, call the search engine using: <search> your focused query </search>.
(2) If the evidence is sufficient, provide the final answer using: <answer> shortest answer at the granularity requested </answer>.
The <answer> block must contain only the answer, with no explanation, full-sentence restatement, or unrequested title/date detail.
If exactly one action remains, do not search because a last-action search cannot be observed; return the best supported final answer now.
"""


