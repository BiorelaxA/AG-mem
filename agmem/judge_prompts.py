"""Benchmark-derived Judge templates, originally fetched on 2026-10-27.

The LoCoMo template locally adapts time equivalence and additional-answer
handling to the Task Agent's answer contract (2026-11-04). LongMemEval text
is unchanged. Reproducibility uses the recorded content SHA256 rather than
a moving upstream branch. Reader, Writer and Task Agent prompts live in
prompts.py.

LoCoMo Refined attribution notice:
This benchmark is a modified version of the LoCoMo release from Snap Research:
https://github.com/snap-research/locomo
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
(ACL 2024 / arXiv:2402.17753).
LoCoMo Refined modifies the benchmark data and judge/evaluator behavior:
https://github.com/mem-eval-suite/LoCoMo_refined
The reproduced Refined prompt is licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/
https://github.com/mem-eval-suite/LoCoMo_refined/blob/main/LICENSE.txt
This copy adapts the Refined prompt's time and additional-answer rules;
using it on this project's LoCoMo data does not turn that data into Refined.

LongMemEval templates: MIT License, Copyright (c) 2024 Di Wu.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

LOCOMO_REFINED = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.
Core principle — Inclusion + Non-contradiction
- Judge semantic coverage of the gold answer, not exact wording, fact order, answer length or the number of list items.
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT. Ignore additional details, reasons and list items that do not contradict the gold.
- Mark WRONG only when required gold content is missing, its meaning is changed, or the generated answer contradicts it. Additional information alone is not a reason to mark WRONG.
TIME (semantic equivalence; aligned with Task Agent time answers)
- Accept an absolute date, an equivalent relative expression, or both together when they refer to the same event time. The Task Agent is asked to provide both forms when the evidence supports them; this must not itself cause a WRONG label.
- Allow calendar arithmetic when an explicit, unambiguous anchor is available in the question, gold answer or generated answer. Anchor expressions inside a historical session to that session's stated date. For questions asking how many days/weeks/months ago, use the stated current date. Never invent a missing anchor or use the computer's current date.
- Compare the event time with the gold at the precision required by the question. Do not require identical surface granularity or reject a matching answer merely because it also contains a compatible date, date range, timestamp or explanatory anchor. A clearly identified session/anchor date is not a competing event date.
- Preserve uncertainty: do not replace "last week" or "a few days before" with an unsupported exact day. Accept an equivalent date range when the stated anchor determines it. Reject conflicting event dates, incorrect conversions, or an exact date that replaces an unresolved interval.
- When relative and absolute forms are both given, they must be mutually consistent. Treat equivalent date formats, punctuation and harmless wording differences as equivalent.
    - Example: gold = "July 2023"; generated = "Last month relative to 19 August 2023 (July 2023)" → CORRECT.
    - Example: gold = "yesterday, relative to 25 May 2023"; generated = "24 May 2023 (yesterday relative to 25 May 2023)" → CORRECT.
    - Example: gold = "a few days before 24 May 2023"; generated = "24 May 2023 (a few days before 24 May 2023)" → WRONG because the event dates conflict.
Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Ignore extra non-contradictory items; the generated list does not have to match the gold list exactly.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → CORRECT when D does not contradict the gold.
    - Example: gold = A, B, C ; gen = A, B, D → WRONG because C is missing.
- A gold element may be paraphrased, elaborated or split into finer details (e.g., C → C, C′) without penalty.
Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":
```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""

LONGMEMEVAL_FACT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LONGMEMEVAL_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days.\nIf the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LONGMEMEVAL_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LONGMEMEVAL_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

LONGMEMEVAL_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

LONGMEMEVAL_TEMPLATES = {
    "single-session-user": LONGMEMEVAL_FACT,
    "single-session-assistant": LONGMEMEVAL_FACT,
    "multi-session": LONGMEMEVAL_FACT,
    "temporal-reasoning": LONGMEMEVAL_TEMPORAL,
    "knowledge-update": LONGMEMEVAL_UPDATE,
    "single-session-preference": LONGMEMEVAL_PREFERENCE,
    "abstention": LONGMEMEVAL_ABSTENTION,
}

UPSTREAM = {
    "locomo_refined": {
        "url": "https://github.com/mem-eval-suite/LoCoMo_refined/blob/main/src/llm_judge.py",
        "license": "CC-BY-NC-4.0", "retrieved": "2026-10-27",
        "local_adaptation": "task_aligned_time_and_gold_inclusion_v1",
    },
    "longmemeval_official": {
        "url": "https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py",
        "license": "MIT", "retrieved": "2026-10-27",
    },
}
