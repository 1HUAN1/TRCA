<div align="center">

# TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-release%20candidate-orange.svg)](#release-status)

</div>

## Introduction
<img width="2883" height="1362" alt="image" src="https://github.com/user-attachments/assets/b73e2e56-16d3-4868-9309-0774ba988571" />


TRCA adopts a global Rubric framework consisting of three dimensions: **Evidence**, **Invalidity**, and **Execution**. Before policy training, we use **GPT-5.6-sol** to instantiate this framework into a benchmark-specific Rubric library for each of ALFWorld, WebShop, and SearchQA.

For each benchmark, the model generates **15 reusable Rubric operators**, comprising five operators for each dimension. Generation is performed offline, once per benchmark, and the resulting library is fixed and shared across all task instances throughout training.

During training, a deterministic task binder instantiates the benchmark-level operators using task-specific information extracted from the task instruction and observable environment metadata. Each bound operator then evaluates individual action-induced transitions through deterministic Boolean conditions.

Based on these Rubric judgments, TRCA constructs two complementary reward components:

- **Foundational Rubric Reward (FRR)** aggregates signed scores across Evidence, Invalidity, and Execution to assess local transition quality, rewarding beneficial actions while penalizing invalid or redundant behavior.
- **Breakthrough Rubric Reward (BRR)** tracks the cumulative coverage of positive Evidence and Execution conditions, rewarding newly established task progress when these conditions are satisfied for the first time within a trajectory.

The two components are fused into a transition-level reward signal. Together with terminal task outcomes, these rewards capture local step quality, incremental progress, and final task completion, providing fine-grained step-level advantages for policy optimization.

**No LLM-based Rubric generation or judgment is performed during trajectory collection or policy optimization.**

**TRCA** provides transition-level credit for long-horizon LLM agents while preserving the original sparse task outcome. It uses one shared, benchmark-level Rubric library whose operators are generated once before training, grounded to each task by a deterministic binder, and evaluated during training without an online LLM judge.

TRCA has two complementary reward components:

- **Foundational Rubric Reward** supplies signed local feedback from Evidence, Invalidity, and Execution judgments.
- **Breakthrough Rubric Reward** rewards the first coverage of positive Evidence and Execution conditions within a trajectory.

The two rewards are fused as

```text
r_TRCA = (1 - lambda) * r_F + lambda * r_B
```

The training return combines the binary terminal outcome with transition-level TRCA reward. The final transition Advantage is the sum of an episode-relative outcome term and a context-relative return term, and is assigned only to executable action tokens.

This repository is a source-only release built on `verl-agent`. It keeps the upstream algorithms for comparison and adds an independent `reward_method=trca` and `algorithm.adv_estimator=trca` path.

## Method at a Glance

```text
offline API generation
        |
        v
shared 15-operator Rubric library
        |
task instruction + observable transition
        |
        v
deterministic task binding and Boolean judgments
        |
        +--> Foundational Reward
        `--> first-coverage Breakthrough Reward
                    |
binary terminal outcome + fused TRCA reward
                    |
        completion-aware return
                    |
episode-relative + context-relative Advantage
                    |
      action-token-only policy update
```

The default mixing coefficient is `lambda=0.8`, and the default discount factor is `gamma=0.95`.

## Repository Structure

```text
TRCA_code/
|-- README.md
|-- README_UPSTREAM.md               # original verl-agent documentation
|-- TRCA_IMPLEMENTATION.md           # method mapping and commands
|-- TRCA_MINIMAL_ALIGNMENT_CHECKLIST.md
|-- trca/
|   |-- api_generate.py              # one-time API Rubric generation
|   |-- schema.py                    # safe operator schema and allowlists
|   |-- binding.py                   # deterministic task binding
|   |-- facts.py                     # observable transition adapters
|   |-- evaluator.py                 # deterministic Boolean evaluation
|   |-- rewards.py                   # Foundational/Breakthrough/fusion
|   |-- action_mask.py               # action/search/answer token masks
|   `-- advantage.py                 # return and two-part Advantage
|-- gigpo/                           # retained GiGPO/FCSR-GiGPO baselines
|-- reward_methods/                  # reward routing and rollout adapter
|-- agent_system/                    # multi-turn rollout and environments
|-- verl/                            # trainer integration
|-- examples/
|   `-- trca_trainer/
|       |-- generate_rubrics.sh
|       `-- run_paper_trca.sh
`-- tests/
    `-- test_trca_paper.py
```

Checkpoints, logs, model weights, datasets, caches, experiment outputs, private data symlinks, and platform-specific runtime binaries are intentionally not included.

## Setup

### 1. Create the environment

```bash
git clone https://github.com/1HUAN1/TRCA.git
cd TRCA

conda create -n trca python=3.10 -y
conda activate trca
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

GPU training additionally requires a CUDA/PyTorch/vLLM stack compatible with your hardware. See [`README_UPSTREAM.md`](README_UPSTREAM.md) for the underlying `verl-agent` environment details.

### 2. Prepare benchmark assets

Provide public or local paths for the assets used by the selected environment:

- ALFWorld task data;
- WebShop product data and search indexes;
- SearchQA training/evaluation data and retriever assets;
- a local or mounted Hugging Face policy model.

The benchmark launchers expose paths through environment variables such as `MODEL_PATH`, `DATA_DIR`, `TRAIN_FILE`, `VAL_FILE`, `RETRIEVE_DATA_ROOT`, and `PYTHON_BIN`. Do not commit credentials, private datasets, or checkpoints.

## Generate the Rubric Libraries

The paper uses one API-generated shared Rubric library per benchmark. Generation happens once, before policy training.

```bash
export TRCA_RUBRIC_API_URL=https://your-openai-compatible-endpoint/v1
export TRCA_RUBRIC_API_KEY=YOUR_API_KEY
export TRCA_RUBRIC_MODEL=YOUR_MODEL_NAME

bash examples/trca_trainer/generate_rubrics.sh
```

The command writes:

```text
trca_rubrics/alfworld.json
trca_rubrics/webshop.json
trca_rubrics/searchqa.json
```

Each library must contain exactly five Evidence, five Invalidity, and five Execution operators. The generated JSON is validated against a non-executable condition DSL and a canonical fact/binding allowlist. Missing or unknown evidence always yields a zero judgment, including under negation.

Review all generated operators against real, observable environment transitions before using them in a reported experiment.

## Training

Run the paper-aligned path with one command per benchmark:

```bash
bash examples/trca_trainer/run_paper_trca.sh alfworld
bash examples/trca_trainer/run_paper_trca.sh webshop
bash examples/trca_trainer/run_paper_trca.sh searchqa
```

Paths and engineering parameters can be overridden without editing the scripts:

```bash
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
bash examples/trca_trainer/run_paper_trca.sh alfworld \
  trainer.n_gpus_per_node=4
```

The wrapper applies the paper-level defaults for `gamma`, `lambda`, rollout group size, learning rate, PPO clipping, KL coefficient, entropy coefficient, temperature, and maximum response length. Hardware-dependent micro-batching and local asset paths may be changed and should be reported with the experiment.

## Action and Token Conventions

TRCA assigns a transition Advantage only to executable content inside a complete action block:

```text
<think>private reasoning</think>
<action>environment command</action>
```

SearchQA retains its native executable protocols:

```text
<search>focused query</search>
<answer>short final answer</answer>
```

Reasoning, tags, prompts, observations, retrieved content, malformed blocks, and padding are excluded from the policy/KL/entropy loss mask. The mask is constructed with tokenizer offsets rather than assuming that a tag has the same tokenization in every context.

## Verification

Run the focused CPU suite:

```bash
python -m pytest -q \
  tests/test_trca_paper.py \
  tests/test_fcsr_gigpo.py \
  tests/test_fcsr_mixer.py \
  tests/test_dcsr_reward.py \
  tests/test_hbrr_reward.py
```

The checked release candidate passes 34 focused tests. The tests cover:

- the 5/5/5 Rubric schema and mocked one-call API generation;
- safe missing-evidence semantics and canonical input allowlists;
- Foundational, first-coverage Breakthrough, and fusion equations;
- binary terminal return and two-part Advantage;
- `<action>`, `<search>`, and `<answer>` token masks;
- compatibility with the retained legacy reward paths.

Real Qwen2.5 tokenizer probes also cover newline-delimited actions and action-like text inside `<think>`.


## Citation

If you use this code, please cite:

```bibtex
@misc{zhang2026trcatransitionwiserubriccredit,
      title={TRCA: Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents}, 
      author={Huan Zhang and Mingju Chen and Dongxu Zhou and Can Lv and Heng Chang and Sen Cui and Faguo Wu and Shiji Zhou},
      year={2026},
      eprint={2608.16156},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.16156}, 
}
```

Add the public paper URL, identifier, and final venue metadata when they become available.

## Acknowledgments

This repository builds on [`verl-agent`](https://github.com/langfengQ/verl-agent) and [veRL](https://github.com/volcengine/verl). The upstream README is preserved in [`README_UPSTREAM.md`](README_UPSTREAM.md), and the upstream Apache 2.0 license and notices remain in [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt).
