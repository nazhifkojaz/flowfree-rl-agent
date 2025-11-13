"""Unit tests for configuration dataclasses."""

import pytest

from rl.solver.trainers.config import (
    CurriculumConfig,
    EvaluationConfig,
    RolloutConfig,
    TrainingConfig,
)


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = TrainingConfig(
            episodes=100,
            batch_size=32,
            buffer_size=10000,
            gamma=0.99,
            lr=1e-4,
        )
        assert config.episodes == 100
        assert config.batch_size == 32
        assert config.gamma == 0.99

    def test_negative_episodes(self):
        """Test that negative episodes raises error."""
        with pytest.raises(ValueError, match="episodes must be positive"):
            TrainingConfig(episodes=-1)

    def test_negative_batch_size(self):
        """Test that negative batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_buffer_smaller_than_batch(self):
        """Test that buffer_size < batch_size raises error."""
        with pytest.raises(ValueError, match="buffer_size.*must be >= batch_size"):
            TrainingConfig(batch_size=128, buffer_size=64)

    def test_invalid_gamma(self):
        """Test that gamma outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="gamma must be in"):
            TrainingConfig(gamma=1.5)
        with pytest.raises(ValueError, match="gamma must be in"):
            TrainingConfig(gamma=-0.1)

    def test_negative_lr(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="lr must be positive"):
            TrainingConfig(lr=-1e-4)

    def test_invalid_epsilon_schedule(self):
        """Test that invalid epsilon schedule raises error."""
        with pytest.raises(ValueError, match="epsilon_schedule must be"):
            TrainingConfig(epsilon_schedule="invalid")

    def test_invalid_expert_ratio(self):
        """Test that expert_sample_ratio outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="expert_sample_ratio must be in"):
            TrainingConfig(expert_buffer_size=1000, expert_sample_ratio=1.5)

    def test_negative_gradient_accumulation(self):
        """Test that negative gradient accumulation raises error."""
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
            TrainingConfig(gradient_accumulation_steps=-1)


class TestEvaluationConfig:
    """Tests for EvaluationConfig validation."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = EvaluationConfig(
            eval_interval=50,
            eval_episodes=5,
            eval_epsilon=0.0,
        )
        assert config.eval_interval == 50
        assert config.eval_episodes == 5

    def test_negative_eval_interval(self):
        """Test that negative eval_interval raises error."""
        with pytest.raises(ValueError, match="eval_interval must be non-negative"):
            EvaluationConfig(eval_interval=-1)

    def test_zero_eval_episodes(self):
        """Test that zero eval_episodes raises error."""
        with pytest.raises(ValueError, match="eval_episodes must be positive"):
            EvaluationConfig(eval_episodes=0)

    def test_invalid_eval_epsilon(self):
        """Test that eval_epsilon outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="eval_epsilon must be in"):
            EvaluationConfig(eval_epsilon=1.5)

    def test_negative_validation_limit(self):
        """Test that negative validation_limit raises error."""
        with pytest.raises(ValueError, match="validation_limit must be positive if set"):
            EvaluationConfig(validation_limit=-1)


class TestCurriculumConfig:
    """Tests for CurriculumConfig validation."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = CurriculumConfig(
            curriculum_six_prob_start=0.5,
            curriculum_six_prob_end=0.85,
            curriculum_six_prob_episodes=1500,
        )
        assert config.curriculum_six_prob_start == 0.5
        assert config.curriculum_six_prob_end == 0.85

    def test_invalid_prob_start(self):
        """Test that prob_start outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="curriculum_six_prob_start must be in"):
            CurriculumConfig(curriculum_six_prob_start=1.5)

    def test_invalid_prob_end(self):
        """Test that prob_end outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="curriculum_six_prob_end must be in"):
            CurriculumConfig(curriculum_six_prob_end=-0.1)

    def test_negative_episodes(self):
        """Test that negative episodes raises error."""
        with pytest.raises(ValueError, match="curriculum_six_prob_episodes must be non-negative"):
            CurriculumConfig(curriculum_six_prob_episodes=-1)

    def test_negative_warmup(self):
        """Test that negative warmup raises error."""
        with pytest.raises(ValueError, match="penalty_warmup must be non-negative"):
            CurriculumConfig(penalty_warmup=-1)


class TestRolloutConfig:
    """Tests for RolloutConfig validation."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = RolloutConfig(
            record_rollouts=True,
            rollout_frequency=10,
            rollout_max=100,
        )
        assert config.record_rollouts is True
        assert config.rollout_frequency == 10

    def test_negative_frequency(self):
        """Test that negative frequency raises error."""
        with pytest.raises(ValueError, match="rollout_frequency must be non-negative"):
            RolloutConfig(rollout_frequency=-1)

    def test_negative_max(self):
        """Test that negative rollout_max raises error."""
        with pytest.raises(ValueError, match="rollout_max must be non-negative if set"):
            RolloutConfig(rollout_max=-1)

    def test_negative_gif_duration(self):
        """Test that negative gif_duration raises error."""
        with pytest.raises(ValueError, match="rollout_gif_duration must be positive"):
            RolloutConfig(rollout_gif_duration=-1)
