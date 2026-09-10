from verl.trainer.ppo.ray_trainer import validation_success_metric_weight


def test_validation_success_mean_is_weighted_by_unique_episodes():
    full_sources = ["nq"] * 512
    full_uids = [f"full-{index}" for index in range(512)]
    tail_sources = ["nq"] * 26
    tail_uids = [f"tail-{index}" for index in range(26)]

    full_weight = validation_success_metric_weight("nq_success_rate", full_sources, full_uids)
    tail_weight = validation_success_metric_weight("nq_success_rate", tail_sources, tail_uids)
    assert full_weight == 512
    assert tail_weight == 26

    weighted = (0.5 * full_weight + 1.0 * tail_weight) / (full_weight + tail_weight)
    assert weighted != (0.5 + 1.0) / 2
    assert abs(weighted - (282.0 / 538.0)) < 1e-12


def test_validation_success_weight_deduplicates_expanded_steps_and_filters_source():
    sources = ["nq", "nq", "hotpotqa", "hotpotqa", "hotpotqa"]
    traj_uids = ["nq-1", "nq-1", "hp-1", "hp-1", "hp-2"]

    assert validation_success_metric_weight("success_rate", sources, traj_uids) == 3
    assert validation_success_metric_weight("nq_success_rate", sources, traj_uids) == 1
    assert validation_success_metric_weight("hotpotqa_success_rate", sources, traj_uids) == 2
