"""CPU checks: python -m unittest discover -s LC-Rec-backbone/tests -v"""

import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment import AlignedQwen3ForCausalLM
from collator import AlignmentCollator, Collator_DecoderOnly_manual
from data import SeqRecDataset
from utils import parse_dataset_args, parse_global_args, parse_test_args, parse_train_args


def make_args(*extra):
    parser = argparse.ArgumentParser()
    for configure in (parse_global_args, parse_dataset_args, parse_train_args, parse_test_args):
        configure(parser)
    parser.add_argument("--data_file", default=".parquet")
    return parser.parse_args(list(extra))


def tiny_config(vocab_size=32):
    return Qwen3Config(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        eos_token_id=1, pad_token_id=1, use_cache=False, attn_implementation="eager",
    )


def wrap_lora(base, alignment=True):
    prepare_model_for_kbit_training(base)
    modules = ["embed_tokens", "lm_head"]
    if alignment:
        base.initialize_alignment(3, 0.4)
        modules.append("hidden_to_item_emb")
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], modules_to_save=modules,
    ))


class DataAndCollatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "Beauty"
        self.folder.mkdir()
        codes = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        np.save(self.folder / "Beauty_t5_rqvae.npy", codes)
        for split in ("train", "valid", "test"):
            pd.DataFrame({"history": [[1, 2]], "target": [3]}).to_parquet(self.folder / f"{split}.parquet")
        self.args = make_args(
            "--dataset", "Beauty", "--data_path", str(self.root),
            "--mse_loss_weight", "0.4", "--item_emb_dim", "3", "--only_train_response",
        )
        self.latents = np.eye(3, dtype=np.float32)
        self.quantized = self.latents + 0.25
        for suffix, embeddings in (
            ("encoder_latent", self.latents), ("quantized_latent", self.quantized)
        ):
            pd.DataFrame({
                "ItemID": [1, 2, 3], "embedding": list(embeddings),
            }).to_parquet(self.folder / f"item_emb_rqvae_{suffix}.parquet")

    def tokenizer(self, dataset):
        backend = Tokenizer(models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "What": 2, "would": 3}, unk_token="<unk>"
        ))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>",
            pad_token="<eos>", padding_side="left", model_max_length=128,
        )
        tokenizer.add_tokens(dataset.get_new_tokens())
        return tokenizer

    def test_current_next_and_both_embedding_sources(self):
        for source, embeddings in (("latent", self.latents), ("quantized", self.quantized)):
            for item, expected in (("current", [0, 1]), ("next", [1, 2])):
                with self.subTest(source=source, item=item):
                    self.args.align_target = source
                    self.args.align_item = item
                    dataset = SeqRecDataset(self.args, "train")
                    self.assertEqual(len(dataset), 2)
                    for index, item_index in enumerate(expected):
                        np.testing.assert_array_equal(dataset[index]["target_item_emb"], embeddings[item_index])
                    valid = SeqRecDataset(self.args, "valid")
                    self.assertEqual(len(valid), 1)
                    np.testing.assert_array_equal(valid[0]["target_item_emb"], embeddings[expected[-1]])

    def test_zero_weight_and_test_split_need_no_embedding_file(self):
        self.args.item_emb_path = str(self.root / "does-not-exist.parquet")
        self.args.mse_loss_weight = 0
        for split in ("train", "valid", "test"):
            dataset = SeqRecDataset(self.args, split)
            self.assertIsNone(dataset.item_embeddings)
            self.assertEqual(set(dataset[0]), {"input_ids", "labels"})
        self.args.mse_loss_weight = 1
        self.assertNotIn("target_item_emb", SeqRecDataset(self.args, "test")[0])
        with self.assertRaises(FileNotFoundError):
            SeqRecDataset(self.args, "train")

    def test_empty_history_fallback_and_vector_mean(self):
        pd.DataFrame({"history": [[]], "target": [3]}).to_parquet(self.folder / "valid.parquet")
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        path = self.root / "multi-vector.parquet"
        pd.DataFrame({"ItemID": [3], "embedding": [vectors]}).to_parquet(path)
        self.args.item_emb_path = str(path)
        dataset = SeqRecDataset(self.args, "valid")
        np.testing.assert_array_equal(dataset[0]["target_item_emb"], [0.5, 0.5, 0.0])

    def test_positions_and_original_ce_labels_with_padding(self):
        dataset = SeqRecDataset(self.args)
        tokenizer = self.tokenizer(dataset)
        batch = [dataset[0], dataset[1]]
        for side in ("left", "right"):
            tokenizer.padding_side = side
            original = Collator_DecoderOnly_manual(self.args, tokenizer)(batch)
            actual = AlignmentCollator(self.args, tokenizer)(batch)
            for key in original:
                torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
            for row, sample in enumerate(batch):
                selected = actual["input_ids"][row, actual["alignment_positions"][row]].tolist()
                response = tokenizer.encode(sample["labels"], add_special_tokens=False)
                self.assertEqual(selected, [tokenizer.convert_tokens_to_ids("|start_of_answer|")] + response[:3])
                self.assertEqual(
                    actual["labels"][row][actual["labels"][row] != -100].tolist(), response
                )

    def test_rejects_truncated_target(self):
        dataset = SeqRecDataset(self.args)
        tokenizer = self.tokenizer(dataset)
        sample = dataset[0]
        tokenizer.model_max_length = len(tokenizer.encode(sample["input_ids"])) + 3
        with self.assertRaisesRegex(ValueError, "truncated"):
            AlignmentCollator(self.args, tokenizer)([sample])

    def test_trainer_train_eval_and_best_checkpoint(self):
        dataset = SeqRecDataset(self.args)
        tokenizer = self.tokenizer(dataset)
        model = wrap_lora(AlignedQwen3ForCausalLM(tiny_config(len(tokenizer))))
        trainer = Trainer(
            model=model, train_dataset=dataset, eval_dataset=SeqRecDataset(self.args, "valid"),
            data_collator=AlignmentCollator(self.args, tokenizer),
            args=TrainingArguments(
                output_dir=str(self.root / "trainer"), use_cpu=True,
                max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                per_device_eval_batch_size=1, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                save_strategy="steps", eval_strategy="steps", save_steps=1, eval_steps=1,
                load_best_model_at_end=True, report_to="none", disable_tqdm=True,
                save_safetensors=False, label_names=["labels"],
            ),
        )
        self.assertFalse(trainer.model_accepts_loss_kwargs)
        result = trainer.train()
        self.assertTrue(np.isfinite(result.training_loss))
        self.assertTrue(np.isfinite(trainer.evaluate()["eval_loss"]))
        saved = torch.load(self.root / "trainer" / "checkpoint-2" / "adapter_model.bin", weights_only=True)
        self.assertTrue(any("hidden_to_item_emb" in key for key in saved))


class AlignmentModelTests(unittest.TestCase):
    def batch(self):
        return {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1], [1, 2, 3, 4, 5, 6, 7, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, -100, 5, 6, 7, 8, -100], [-100, -100, -100, 4, 5, 6, 7, -100]]),
            "alignment_positions": torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]]),
            "target_item_emb": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        }

    def test_exact_mean_cosine_and_gradient_positions(self):
        torch.manual_seed(42)
        model = AlignedQwen3ForCausalLM(tiny_config())
        model.initialize_alignment(3, 0.4)
        hidden = torch.randn(2, 10, 16, requires_grad=True)
        positions = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
        targets = torch.randn(2, 3, requires_grad=True)
        actual = model.alignment_loss(hidden, positions, targets)
        pooled = torch.stack([hidden[0, 2:6].mean(0), hidden[1, 5:9].mean(0)])
        expected = 1 - torch.nn.functional.cosine_similarity(model.hidden_to_item_emb(pooled), targets, dim=-1).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(targets.grad)
        selected_mask = torch.zeros(2, 10, dtype=torch.bool).scatter_(1, positions, True)
        self.assertGreater(hidden.grad[selected_mask].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[~selected_mask].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in model.hidden_to_item_emb.parameters()))

    def test_combined_loss_and_baseline_without_alignment_inputs(self):
        torch.manual_seed(42)
        config = tiny_config()
        base = AutoModelForCausalLM.from_config(config)
        aligned = AlignedQwen3ForCausalLM(copy.deepcopy(config))
        aligned.load_state_dict(base.state_dict())
        aligned.initialize_alignment(3, 0.4)
        base.eval()
        aligned.eval()
        batch = self.batch()
        original_inputs = {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}
        original = base(**original_inputs)
        unchanged = aligned(**original_inputs)
        torch.testing.assert_close(unchanged.loss, original.loss, rtol=0, atol=0)
        torch.testing.assert_close(unchanged.logits, original.logits, rtol=0, atol=0)
        active = aligned(**batch)
        self.assertIsNone(active.hidden_states)
        self.assertNotIn("hidden_states", active)
        losses = aligned.last_loss_dict
        torch.testing.assert_close(active.loss, original.loss + 0.4 * losses["align"])
        active.loss.backward()
        self.assertIsNotNone(aligned.hidden_to_item_emb[0].weight.grad)

    def test_gradient_accumulation_preserves_alignment_weight(self):
        torch.manual_seed(42)
        prototype = AlignedQwen3ForCausalLM(tiny_config())
        prototype.initialize_alignment(3, 0.4)
        initial_state = copy.deepcopy(prototype.state_dict())
        batch = self.batch()
        dataset = [{key: value[row].clone() for key, value in batch.items()} for row in (0, 1, 0, 1)]
        trained = []
        with tempfile.TemporaryDirectory() as temp:
            for batch_size, accumulation in ((4, 1), (2, 2)):
                model = AlignedQwen3ForCausalLM(tiny_config())
                model.initialize_alignment(3, 0.4)
                model.load_state_dict(initial_state)
                trainer = Trainer(
                    model=model, train_dataset=dataset,
                    optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
                    args=TrainingArguments(
                        output_dir=str(Path(temp) / str(accumulation)), use_cpu=True,
                        max_steps=1, per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=accumulation, max_grad_norm=0,
                        save_strategy="no", report_to="none", disable_tqdm=True,
                    ),
                )
                trainer.train()
                trained.append(copy.deepcopy(model.state_dict()))
        for name in trained[0]:
            torch.testing.assert_close(trained[0][name], trained[1][name], rtol=1e-5, atol=1e-7)

    def test_peft_checkpoint_roundtrip_and_plain_inference(self):
        torch.manual_seed(42)
        base = AlignedQwen3ForCausalLM(tiny_config())
        initial_state = copy.deepcopy(base.state_dict())
        model = wrap_lora(base)
        batch = self.batch()
        model(**batch).loss.backward()
        for name, param in model.named_parameters():
            if "hidden_to_item_emb.modules_to_save.default" in name:
                self.assertIsNotNone(param.grad)
            if param.requires_grad and param.grad is not None:
                param.data.add_(param.grad, alpha=-0.01)
        model.eval()
        expected = model(**batch)
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp, safe_serialization=False)
            state = torch.load(Path(temp) / "adapter_model.bin", weights_only=True)
            self.assertTrue(any("hidden_to_item_emb" in key for key in state))
            fresh = AlignedQwen3ForCausalLM(tiny_config())
            fresh.load_state_dict(initial_state)
            restored = wrap_lora(fresh)
            result = set_peft_model_state_dict(restored, copy.deepcopy(state))
            self.assertFalse(result.unexpected_keys)
            restored.eval()
            torch.testing.assert_close(restored(**batch).loss, expected.loss)
            for name, param in restored.named_parameters():
                if "hidden_to_item_emb.modules_to_save.default" in name:
                    torch.testing.assert_close(param, dict(model.named_parameters())[name])

            # Existing test entry points load the recommendation weights without
            # needing the training-only projection head or any item embedding file.
            plain = AutoModelForCausalLM.from_config(tiny_config())
            plain.load_state_dict(initial_state)
            inference_model = wrap_lora(plain, alignment=False)
            result = set_peft_model_state_dict(inference_model, copy.deepcopy(state))
            self.assertTrue(all("hidden_to_item_emb" in key for key in result.unexpected_keys))
            inference_model.eval()
            inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
            torch.testing.assert_close(inference_model(**inputs).logits, expected.logits)


if __name__ == "__main__":
    unittest.main()
