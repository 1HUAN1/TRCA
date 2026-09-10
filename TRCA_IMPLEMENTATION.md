# Minimal paper-aligned TRCA path

This copy implements the method in *TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents*. It preserves the original FCSR/GiGPO code for comparison and adds an independent `reward_method=trca`, `algorithm.adv_estimator=trca` path.

## Method mapping

- `trca/api_generate.py`: one API call per benchmark creates one shared 15-operator library (5 Evidence, 5 Invalidity, 5 Execution). The response is validated against a safe JSON condition DSL.
- `trca/binding.py`: deterministic task-conditioned binding. SearchQA ground-truth answers and aliases are explicitly removed.
- `trca/facts.py` and `trca/evaluator.py`: allowlisted canonical observable facts and deterministic Boolean evaluation. Missing/unknown evidence returns 0, including under `not` and `falsy`; no online LLM judge is used.
- `trca/rewards.py`: Foundational reward, first-coverage Breakthrough reward, and `(1-lambda) rF + lambda rB`. Invalidity is negative in Foundational and excluded from Breakthrough.
- `trca/advantage.py`: discounted return over the binary terminal outcome plus `trca_reward`, episode-relative outcome advantage, context-relative return advantage, and their unweighted sum. Exact normalized observable state is used as the conservative context key.
- `trca/action_mask.py`: only content inside complete `<action>`, `<search>`, or `<answer>` spans is trainable/credited. Think blocks, tags, prompts, observations, retrieved content, padding, and malformed/unclosed spans are zero-masked. The same mask is copied into the multi-turn actor loss mask.

The TRCA path does not add a terminal anchor to rubric reward, does not clip negative Invalidity reward, and does not separately mix environment and rubric advantages.

## 1. Generate rubric libraries once via API

The endpoint must implement the standard chat-completions request/response shape. It may be a hosted provider or an internal compatible gateway.

```bash
cd /root/data/zh/TRCA_code
export TRCA_RUBRIC_API_URL=https://your-api.example/v1
export TRCA_RUBRIC_API_KEY=YOUR_KEY
export TRCA_RUBRIC_MODEL=YOUR_MODEL
bash examples/trca_trainer/generate_rubrics.sh
```

Each JSON file records the model, endpoint, timestamp, and one-call generation setting. Training fails fast if the requested library is absent or not exactly 5 operators per category.

## 2. Launch paper defaults

```bash
bash examples/trca_trainer/run_paper_trca.sh alfworld
bash examples/trca_trainer/run_paper_trca.sh webshop
bash examples/trca_trainer/run_paper_trca.sh searchqa
```

The wrapper applies the paper defaults: group size 8, learning rate `8e-7`, PPO mini-batch 256, micro-batch 32/GPU, gamma 0.95, clip 0.20/0.28, KL 0.012, entropy 0.001, temperature 1.0, response length 512, and TRCA lambda 0.8. Hardware-specific values can be appended as Hydra overrides. Reducing micro-batch size for memory changes an engineering parameter from the paper but not the TRCA algorithm.

Category budgets default to 1.0 each because the paper defines symbolic budgets but does not list benchmark-specific numeric values in the core hyperparameter table. They are explicit launcher arguments and can be changed only if the experimental recipe supplies the omitted values.

## Verification boundary

The repository contains no generated rubric JSON by default. The API variables above must be supplied and the three generated libraries must be reviewed before training. The copied release also excludes checkpoints, logs, datasets, model weights, caches, and private data symlinks; provide their paths through the existing launcher environment variables.

CPU checks cover the reward equations, first-coverage behavior, unknown-evidence semantics, all three action tag protocols, binary terminal scaling, conservative context grouping, and legacy FCSR regressions. A full GPU policy update and benchmark reproduction are still required before claiming result-level reproducibility.
