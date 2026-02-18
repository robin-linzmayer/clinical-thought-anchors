og_categories = """
1. Initializing: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
2. Deduction: Deriving conclusions based on current approach and assumptions.
3. Adding Knowledge: Incorporating external knowledge to refine reasoning.
4. Example Testing: Generating examples or scenarios to test hypotheses.
5. Uncertainty Estimation: Explicitly stating confidence or uncertainty regarding reasoning.
6. Backtracking: Abandoning the current approach and exploring alternative strategies.
7. Unknown: The chunk does not fit into any of the above categories.
"""

et_categories = """
1. Initializing: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
2. Deduction: Deriving conclusions based on current approach and assumptions.
3. Adding Knowledge: Incorporating external knowledge to refine reasoning.
4. Example Testing: Generating examples or scenarios to test hypotheses.
5. Uncertainty Estimation: Explicitly stating confidence or uncertainty regarding reasoning.
6. Backtracking: Abandoning the current approach and exploring alternative strategies.
7. Comparative Analysis: Comparing multiple approaches or options side by side.
8. Question Posing: Asking self-directed questions to guide reasoning.
9. Summary: Summarizing or recapping thinking, often near the end of reasoning.
10. Metaphorical Thinking: Using metaphors or analogies to understand the problem.
11. Final Answer: Providing the final answer to the problem.
12. Unknown: The chunk does not fit into any of the above categories.

If you're unsure, then it's better to assign it to "Unknown".
"""

etm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Initialization: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
3. Planning and Strategy: Laying out an approach or plan before solving.
4. Recall: Recalling math facts, definitions, or formulas from memory.
5. Substitution: Rewriting or re-encoding expressions (e.g., defining variables, changing forms).
6. Intermediate Step: Performing intermediate calculations or algebra.
7. Verification: Re-checking previous steps for correctness or coherence.
8. Alternative Strategy and Backtracking: Proposing a second method, reframing the problem, or abandoning the current approach.
9. Summary and Preparation: Summarizing reasoning just before giving the final answer.
10. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
11. Unknown: The chunk does not fit into any of the above categories.
"""

pm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Initialization: Rephrasing the given task and stating initial thoughts at the beginning of reasoning.
3. Planning and Strategy: Laying out an approach or plan before solving.
4. Recall: Recalling math facts, definitions, or formulas from memory.
6. Intermediate Step: Performing intermediate calculations or algebra.
7. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
8. Unknown: The chunk does not fit into any of the above categories.

If you're unsure, then it's better to assign it to "Unknown".
"""

sm_categories = """
1. Prompt: The initial prompt that the model received. It is up until the ocurrence of the first <think> token. Everything after should be one of the below categories.
2. Intermediate Step: Performing intermediate calculations, i.e., intermediate steps, that are later used to derive the final answer.
3. Final Answer: Stating or formatting the final boxed answer. If "Final Answer" is assigned to a chunk, it should be the first category in the list.
4. Unknown: The chunk does not fit into any of the above categories.

We call "Intermediate Step" chunks with calculations or computations that are later directly used to derive the final answer. 
If a chunk doesn't contain any calculations or computations, then it's not an "Intermediate Step".
You need to assess if "Intermediate Step" chunks are causally important for the final answer or not.
If you're unsure, then it's better to assign it to "Unknown".
"""

ss_categories = """
1. Prompt:
   This is the model’s initial prompt (e.g., task description, problem statement) before the reasoning begins.
   This chunk **must always be labeled Prompt if it appears**, and typically ends at the first <think> token.

2. Initialization:
   Rephrasing the problem, unpacking definitions, or stating general intentions at the start of reasoning.
   No equations or plan details yet. Just framing and initial engagement.

3. Planning and Strategy:
   Laying out a specific plan, method, or approach the model intends to follow.
   Often includes conditionals (“I’ll try X first...”) or descriptions of potential steps.

4. Recall or Memory or Fact Retrieval:
   Recalling known math facts, identities, theorems, formulas, definitions, or number facts (e.g., “3^5 = 243”).
   Includes explicit retrieval from memory, not derivation.

5. Substitution or Rewriting:
   Rewriting or representing an expression in a more useful form (e.g., “243 = 3^5” or “Let x = ...”).
   Includes algebraic rearrangement, defining variables, or transforming the expression’s structure.

6. Intermediate Step:
   Performing actual computations, algebraic manipulations, simplifications, or solving sub-parts of the problem.
   These steps **directly contribute to the final answer** (e.g., computing 6000 / 750 = 8).

7. Verification or Consistency Check:
   Reviewing or re-computing a previous result to confirm correctness.
   Often introduced with phrases like “let me double check” or “just to be sure…”

8. Alternative Strategy or Backtracking:
   Proposing a new approach, re-analyzing earlier assumptions, or expressing doubt in the current direction.
   Includes backtracking, starting over, or switching methods (“Alternatively, I could...”)

9. Comparison or Meta-Analysis:
   Comparing multiple solution paths, strategies, or results side-by-side.
   Includes explicit assessment of which approach is more efficient, elegant, or robust.

10. Uncertainty or Confidence Estimation:
    Explicitly expressing confidence or uncertainty in the reasoning or result.
    E.g., “I’m not sure if this is right...” or “I feel confident in this answer.”

11. Final Answer Preparation or Summary:
    A short chunk that leads directly into the answer — summarizing key steps, restating conclusions, or rephrasing in final form before answer emission.

12. Final Answer:
    The explicit answer — boxed result, final numerical or symbolic output.
    If present, this chunk must include the boxed or final answer format.

13. Metaphorical or Conceptual Thinking:
    Using analogy, metaphor, or conceptual reframing to understand the problem (e.g., “Like in a handshake problem…”)
    Rare but important for open-ended or transfer tasks.

14. Question Posing or Self-Directed Inquiry:
    The model asks itself questions to guide reasoning (“What does the exponent mean?”, “Is this number a square?”).
    Different from Planning — it’s introspective rather than directive.

15. Unknown:
    The chunk does not fit into any of the above categories.
    Use this when the chunk is ambiguous, off-topic, irrelevant, or purely stylistic.
    
If you're unsure, then it's better to assign it to "Unknown".
"""

dr_categories = """
1. `case_representation`:
    Summarizing, reframing, or abstracting the patient presentation into a structured clinical description.

2. `knowledge_activation`:
    Recalling disease patterns, epidemiology, risk factors, or clinical associations without yet applying them to the specific case.

3. `hypothesis_generation`:
    Proposing one or more candidate diagnoses as possible explanations.

4. `evidence_evaluation`:
    Linking case findings to hypotheses OR contrasting hypotheses against findings; weighing support and contradictions.

5. `reconsideration_uncertainty`:
    Expressing doubt, backtracking, cycling between hypotheses, or recognizing unresolved ambiguity.

6. `diagnostic_verification`:
    Explicitly checking that the proposed diagnosis accounts for all key findings or confirming that no major contradictions remain before commitment.

7. `diagnostic_commitment`:
    Selecting and justifying the most likely diagnosis based on prior evaluation.

8. `final_answer_emission`:
    Explicit statement of the final diagnosis independent of reasoning justification.

9. `unknown`:
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

"""

CATEGORIES = {
    "og": og_categories,
    "et": et_categories,
    "etm": etm_categories,
    "pm": pm_categories,
    "sm": sm_categories,
    "ss": ss_categories,
    "dr": dr_categories,
}

DAG_PROMPT_MATH = """
You are an expert in interpreting how language models solve math problems using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, broken into discrete text chunks, and label each chunk with:

1. **function_tags**: One or more labels that describe what this chunk is *doing* functionally in the reasoning process.

2. **depends_on**: A list of earlier chunk indices that this chunk directly depends on — meaning it uses information, results, or logic introduced in those earlier chunks.

This annotation will be used to build a dependency graph and perform causal analysis, so please be precise and conservative: only mark a chunk as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per chunk if appropriate):

1. `problem_setup`:
    Parsing or rephrasing the problem (initial reading or comprehension).

2. `plan_generation`:
    Stating or deciding on a plan of action (often meta-reasoning).

3. `fact_retrieval`:
    Recalling facts, formulas, problem details (without immediate computation).

4. `active_computation`:
    Performing algebra, calculations, manipulations toward the answer.

5. `result_consolidation`:
    Aggregating intermediate results, summarizing, or preparing final answer.

6. `uncertainty_management`:
    Expressing confusion, re-evaluating, proposing alternative plans (includes backtracking).

7. `final_answer_emission`:
    Explicit statement of the final boxed answer or earlier chunks that contain the final answer.

8. `self_checking`:
    Verifying previous steps, Pythagorean checking, re-confirmations.

9. `unknown`:
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

---

{depends_on_instructions}

---

### Output Format:

{output_format}

Here is the math problem:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{full_chunked_text}

Now label each chunk with function tags and dependencies.
"""

DAG_PROMPT_DIAGNOSTIC = """
You are an expert in interpreting how language models perform clinical diagnostic reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace for a medical diagnosis question, broken into discrete text chunks, and label each chunk with:

1. **function_tags**: One or more labels that describe what this chunk is *doing* functionally in the diagnostic reasoning process.

2. **depends_on**: A list of earlier chunk indices that this chunk directly depends on — meaning it uses information, findings, or reasoning introduced in those earlier chunks.

This annotation will be used to build a dependency graph and perform causal analysis of diagnostic reasoning, so please be precise and conservative: only mark a chunk as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per chunk if appropriate):

1. `case_representation`:
    Summarizing, reframing, or abstracting the patient presentation into a structured clinical description.

2. `knowledge_activation`:
    Recalling disease patterns, epidemiology, risk factors, or clinical associations without yet applying them to the specific case.

3. `hypothesis_generation`:
    Proposing one or more candidate diagnoses as possible explanations.

4. `evidence_evaluation`:
    Linking specific case findings to hypotheses OR contrasting hypotheses against findings; weighing support and contradictions.

5. `reconsideration_uncertainty`:
    Expressing doubt, backtracking, cycling between hypotheses, or recognizing unresolved ambiguity.

6. `diagnostic_verification`:
    Explicitly checking that the proposed diagnosis accounts for all key findings or confirming that no major contradictions remain before commitment.

7. `diagnostic_commitment`:
    Selecting and justifying the most likely diagnosis based on prior evaluation.

8. `final_answer_emission`:
    Explicit statement of the final diagnosis independent of reasoning justification.

9. `unknown`:
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

---

{depends_on_instructions}

---

### Output Format:

{output_format}

Here is the clinical vignette:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{full_chunked_text}

Now label each chunk with function tags and dependencies.
"""

_DEPENDS_ON_INSTRUCTIONS = """### depends_on Instructions:

For each chunk, include a list of earlier chunk indices that the reasoning in this chunk *uses*. For example:
- If Chunk 9 evaluates evidence based on a hypothesis in Chunk 4 and a recalled fact in Chunk 5, then `depends_on: [4, 5]`
- If Chunk 24 reconsiders based on findings discussed in Chunk 23, then `depends_on: [23]`
- If there's no clear dependency (e.g. a general knowledge recall), use an empty list: `[]`
- If Chunk 13 evaluates evidence using information from Chunk 11, which in turn uses information from Chunk 7, then `depends_on: [11, 7]`

Important Notes:
- Make sure to include all dependencies for each chunk.
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies.
- Try to be as comprehensive as possible.
- Make sure there is always a path from earlier chunks (e.g. case_representation and/or evidence_evaluation) to the final answer."""

_OUTPUT_FORMAT = """Return a single dictionary with one entry per chunk, where each entry has:
- the chunk index (as the key, converted to a string),
- a dictionary with:
    - `"function_tags"`: list of tag strings
    - `"depends_on"`: list of chunk indices, converted to strings

Here's the expected format:

```language=json
{{
    "0": {{
    "function_tags": ["case_representation"],
    "depends_on": []
    }},
    "3": {{
    "function_tags": ["knowledge_activation"],
    "depends_on": []
    }},
    "5": {{
    "function_tags": ["hypothesis_generation"],
    "depends_on": ["0", "3"]
    }},
    "8": {{
    "function_tags": ["evidence_evaluation"],
    "depends_on": ["0", "5"]
    }},
    "12": {{
    "function_tags": ["diagnostic_commitment"],
    "depends_on": ["5", "8"]
    }},
    "13": {{
    "function_tags": ["final_answer_emission"],
    "depends_on": ["12"]
    }}
}}
```"""

DAG_PROMPTS = {
    "math": DAG_PROMPT_MATH,
    "diagnostic": DAG_PROMPT_DIAGNOSTIC,
}

# Backward compatibility: DAG_PROMPT defaults to math
DAG_PROMPT = DAG_PROMPT_MATH


def get_dag_prompt(domain="math"):
    """Get the DAG prompt for the specified domain.

    Args:
        domain: "math" or "diagnostic"

    Returns:
        Prompt template string with {problem_text} and {full_chunked_text} placeholders.
    """
    template = DAG_PROMPTS[domain]
    return template.format(
        depends_on_instructions=_DEPENDS_ON_INSTRUCTIONS,
        output_format=_OUTPUT_FORMAT,
        problem_text="{problem_text}",
        full_chunked_text="{full_chunked_text}",
    )