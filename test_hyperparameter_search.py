"""Tests for sampled training hyperparameters."""

from unittest import TestCase

from hyperparameter_search import (
    BATCH_SIZE_OPTIONS,
    SEARCH_EPOCHS,
    WEIGHT_DECAY_OPTIONS,
    random_configurations,
)


class HyperparameterSearchTests(TestCase):
    """Check that quick trials sample each tuned parameter."""

    def test_trials_sample_tuned_parameters_and_keep_one_search_epoch(self) -> None:
        configurations = random_configurations(seed=42, trials=5)

        self.assertEqual(len(configurations), 5)
        self.assertEqual(
            len(
                {
                    (
                        config.learning_rate,
                        config.batch_size,
                        config.weight_decay,
                    )
                    for config in configurations
                }
            ),
            5,
        )
        self.assertGreater(
            len({config.learning_rate for config in configurations}), 1
        )
        for config in configurations:
            self.assertGreaterEqual(config.learning_rate, 1e-5)
            self.assertLessEqual(config.learning_rate, 1e-3)
            self.assertIn(config.batch_size, BATCH_SIZE_OPTIONS)
            self.assertIn(config.weight_decay, WEIGHT_DECAY_OPTIONS)
            self.assertEqual(config.epochs, SEARCH_EPOCHS)
