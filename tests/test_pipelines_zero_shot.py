import unittest

from transformers.pipelines import Pipeline

from .test_pipelines_common import CustomInputPipelineCommonMixin


class ZeroShotClassificationPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "zero-shot-classification"
    small_models = [
        "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
    ]  # Models tested without the @slow decorator
    large_models = ["roberta-large-mnli"]  # Models tested with the @slow decorator

    def _test_scores_sum_to_one(self, result):
        sum = 0.0
        for score in result["scores"]:
            sum += score
        self.assertAlmostEqual(sum, 1.0)

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"sequence", "labels", "scores"}
        valid_mono_inputs = [
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics"]},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics, public health"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics", "public health"]},
            {"sequences": ["Who are you voting for in 2020?"], "candidate_labels": "politics"},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "This text is about {}",
            },
        ]
        valid_multi_input = {
            "sequences": ["Who are you voting for in 2020?", "What is the capital of Spain?"],
            "candidate_labels": "politics",
        }
        invalid_inputs = [
            {"sequences": None, "candidate_labels": "politics"},
            {"sequences": "", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": None},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ""},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": None,
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "",
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "Template without formatting syntax.",
            },
        ]
        self.assertIsNotNone(nlp)

        for mono_input in valid_mono_inputs:
            mono_result = nlp(**mono_input)
            self.assertIsInstance(mono_result, dict)
            if len(mono_result["labels"]) > 1:
                self._test_scores_sum_to_one(mono_result)

            for key in output_keys:
                self.assertIn(key, mono_result)

        multi_result = nlp(**valid_multi_input)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)
        self.assertEqual(len(multi_result), len(valid_multi_input["sequences"]))

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

            if len(result["labels"]) > 1:
                self._test_scores_sum_to_one(result)

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, **bad_input)

        if nlp.model.name_or_path in self.large_models:
            # We also check the outputs for the large models
            inputs = [
                {
                    "sequences": "Who are you voting for in 2020?",
                    "candidate_labels": ["politics", "public health", "science"],
                },
                {
                    "sequences": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                    "candidate_labels": ["machine learning", "statistics", "translation", "vision"],
                    "multi_class": True,
                },
            ]

            expected_outputs = [
                {
                    "sequence": "Who are you voting for in 2020?",
                    "labels": ["politics", "public health", "science"],
                    "scores": [0.975, 0.015, 0.008],
                },
                {
                    "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                    "labels": ["translation", "machine learning", "vision", "statistics"],
                    "scores": [0.817, 0.712, 0.018, 0.017],
                },
            ]

            for input, expected_output in zip(inputs, expected_outputs):
                output = nlp(**input)
                for key in output:
                    if key == "scores":
                        for output_score, expected_score in zip(output[key], expected_output[key]):
                            self.assertAlmostEqual(output_score, expected_score, places=2)
                    else:
                        self.assertEqual(output[key], expected_output[key])
