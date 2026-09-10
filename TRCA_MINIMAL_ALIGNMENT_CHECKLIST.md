# TRCA minimal-alignment checklist

This checklist separates method-level requirements from acceptable engineering differences. The goal is a small, auditable training path rather than a rewrite of `verl-agent`.

## P0 - Method principles (implemented in this copy)

1. **Shared Rubric library generated once through an API**
   - Code: `trca/api_generate.py`, `trca/schema.py`, `examples/trca_trainer/generate_rubrics.sh`.
   - One library per benchmark, exactly five Evidence, five Invalidity, and five Execution operators.
   - Training-time evaluation is deterministic; no online LLM judge.

2. **Observable, task-bound Boolean judgments**
   - Code: `trca/binding.py`, `trca/facts.py`, `trca/evaluator.py`.
   - Operators may read only the paper's canonical allowlist.
   - Missing or unknown inputs return zero even inside `not`/`falsy`; outcome and score fields are not exported.

3. **Foundational and Breakthrough rewards**
   - Code: `trca/rewards.py`, `reward_methods/mixer.py`.
   - Foundational uses signed category-normalized item contributions.
   - Breakthrough rewards first coverage of positive Evidence/Execution only and is causal within each trajectory.
   - Fusion is `(1-lambda) * rF + lambda * rB`, default `lambda=0.8`; no legacy terminal anchor in this path.

4. **Paper-level return and Advantage**
   - Code: `gigpo/core_trca.py`, `verl/trainer/ppo/ray_trainer.py`.
   - Return combines binary terminal outcome with TRCA transition reward at `gamma=0.95`.
   - Final Advantage is the unweighted sum of episode-relative and context-relative normalized terms.
   - Context equality uses the complete normalized observable state, avoiding false equivalence from heuristic summaries.

5. **Action-only optimization**
   - Code: `trca/action_mask.py`, `agent_system/multi_turn_rollout/rollout_loop.py`, `verl/trainer/ppo/ray_trainer.py`.
   - Supports `<action>`, SearchQA `<search>`, and SearchQA `<answer>` with tokenizer offsets.
   - Think blocks, tags, padding, malformed blocks, and non-action response tokens are excluded.
   - The action mask is propagated into the multi-turn actor loss mask, covering policy, KL, and entropy losses.

## P1 - Required before a public reproducibility claim

1. **Generate and review real Rubric artifacts**
   - Set `TRCA_RUBRIC_API_URL`, `TRCA_RUBRIC_API_KEY`, and `TRCA_RUBRIC_MODEL`.
   - Run `bash examples/trca_trainer/generate_rubrics.sh`.
   - Manually inspect all 45 operators against the paper and real environment transitions; commit the approved JSON and generation metadata.

2. **Run one real smoke update per benchmark**
   - Verify rollout -> facts -> judgments -> both rewards -> return -> both Advantages -> actor update.
   - Record nonzero action-token rate, operator hit rates, comparable-context group count, finite losses, and a confirmed parameter update.

3. **Validate each environment adapter with captured transitions**
   - Add small redacted fixtures for ALFWorld, WebShop, and SearchQA.
   - For every approved operator, include positive, negative, and missing-evidence examples.

4. **Resolve data/runtime paths**
   - The source-only copy omits model weights, datasets, WebShop indexes, and retriever data.
   - Document public download/preparation steps and remove private defaults from the release launchers or expose them only as environment-variable examples.

5. **Re-run the claimed results**
   - The new TRCA path differs materially from the old DCSR/HBRR/FCSR path. Do not attach old metrics to the new implementation without re-running them.

## P2 - Open-source polish

1. Add the final paper link/citation and author/repository metadata.
2. Add a clean environment lock or tested installation matrix.
3. Add CI for the focused CPU suite and shell syntax checks.
4. Run a secret/path/license scan and retain upstream attribution.
5. Publish resolved configs and the exact Rubric artifact hashes used for each result.

## Acceptable differences

GPU count, batch micro-sizing, local data placement, logging backend, checkpoint frequency, and equivalent implementation details may differ when documented. Rubric observability, missing-evidence semantics, two reward definitions, reward fusion, binary terminal signal, two-part Advantage, context comparability, and action-only optimization are method-level constraints and should not drift.

## Current verification

- Focused CPU suite: 34 passed; two dependency deprecation warnings.
- Shell syntax: TRCA generation/launch scripts and the patched ALFWorld launcher pass `bash -n`.
- Real Qwen2.5 tokenizer probes correctly credit action/search/answer content, including newline-delimited actions, and exclude action-like text inside `<think>`.
- Not performed: paid API generation, GPU training, benchmark evaluation, or result reproduction.
