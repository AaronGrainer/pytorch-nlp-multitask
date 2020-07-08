import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import transformers
import nlp

import dataclasses
from packaging import version
from torch.utils.data.dataloader import DataLoader
from transformers.trainer import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, TrainOutput
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict, Optional
from tqdm.auto import tqdm, trange

from google.cloud import storage

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_name = "roberta-base"
max_length = 128

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


def get_args():
	args_parser = argparse.ArgumentParser()

	args_parser.add_argument(
		"--batch-size",
		help="Batch size for each training and evaluation step",
		type=int,
		default=8
	)

	args_parser.add_argument(
		"--job-dir",
		help="GCS location to export models")
	
	return args_parser.parse_args()


class MultitaskModel(transformers.PreTrainedModel):
	def __init__(self, encoder, taskmodels_dict):
		"""
		Setting MultitaskModel up as a PretrainedModel allows us
		to take better advantage of Trainer features
		"""
		super().__init__(transformers.PretrainedConfig())

		self.encoder = encoder
		self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

	@classmethod
	def create(cls, model_name, model_type_dict, model_config_dict):
		"""
		This creates a MultitaskModel using the model class and config objects
		from single-task models.

		We do this by creating each single-task model, and having them share
		the same encoder transformer.
		"""
		shared_encoder = None
		taskmodels_dict = {}
		for task_name, model_type in model_type_dict.items():
			model = model_type.from_pretrained(
				model_name,
				config=model_config_dict[task_name],
			)
			if shared_encoder is None:
				shared_encoder = getattr(
					model, cls.get_encoder_attr_name(model))
			else:
				setattr(model, cls.get_encoder_attr_name(
					model), shared_encoder)
			taskmodels_dict[task_name] = model
		return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

	@classmethod
	def get_encoder_attr_name(cls, model):
		"""
		The encoder transformer is named differently in each model "architecture".
		This method lets us get the name of the encoder attribute
		"""
		model_class_name = model.__class__.__name__
		if model_class_name.startswith("Bert"):
			return "bert"
		elif model_class_name.startswith("Roberta"):
			return "roberta"
		elif model_class_name.startswith("Albert"):
			return "albert"
		else:
			raise KeyError(f"Add support for new model {model_class_name}")

	def forward(self, task_name, **kwargs):
		return self.taskmodels_dict[task_name](**kwargs)


def convert_to_stsb_features(example_batch):
	inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
	features = tokenizer.batch_encode_plus(
			inputs, max_length=max_length, pad_to_max_length=True)
	features["labels"] = example_batch["label"]
	return features


def convert_to_rte_features(example_batch):
	inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
	features = tokenizer.batch_encode_plus(
			inputs, max_length=max_length, pad_to_max_length=True)
	features["labels"] = example_batch["label"]
	return features


def convert_to_commonsense_qa_features(example_batch):
	num_examples = len(example_batch["question"])
	num_choices = len(example_batch["choices"][0]["text"])
	features = {}
	for example_i in range(num_examples):
		choices_inputs = tokenizer.batch_encode_plus(
			list(zip(
				[example_batch["question"][example_i]] * num_choices,
				example_batch["choices"][example_i]["text"],
			)),
			max_length=max_length, pad_to_max_length=True,
		)
		for k, v in choices_inputs.items():
			if k not in features:
				features[k] = []
			features[k].append(v)
	labels2id = {char: i for i, char in enumerate("ABCDE")}
	# Dummy answers for test
	if example_batch["answerKey"][0]:
		features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
	else:
		features["labels"] = [0] * num_examples
	return features


class NLPDataCollator(DataCollator):
	"""
	Extending the existing DataCollator to work with NLP dataset batches
	"""

	def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
		first = features[0]
		if isinstance(first, dict):
			# NLP data sets current works presents features as lists of dictionary
			# (one per example), so we  will adapt the collate_batch logic for that
			if "labels" in first and first["labels"] is not None:
				if first["labels"].dtype == torch.int64:
					labels = torch.tensor([f["labels"]
											for f in features], dtype=torch.long)
				else:
					labels = torch.tensor([f["labels"]
											for f in features], dtype=torch.float)
				batch = {"labels": labels}
			for k, v in first.items():
				if k != "labels" and v is not None and not isinstance(v, str):
					batch[k] = torch.stack([f[k] for f in features])
			return batch
		else:
		  # Otherwise, revert to using the default collate_batch
		  return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):
	"""
	This is a hack. The Trainer is going call .to(device) on every input
	value, but we need to pass in an additional `task_name` string.
	This prevents it from throwing an error
	"""

	def to(self, device):
		return self


class DataLoaderWithTaskname:
	"""
	Wrapper around a DataLoader to also yield a task name
	"""

	def __init__(self, task_name, data_loader):
		self.task_name = task_name
		self.data_loader = data_loader

		self.batch_size = data_loader.batch_size
		self.dataset = data_loader.dataset

	def __len__(self):
		return len(self.data_loader)

	def __iter__(self):
		for batch in self.data_loader:
			batch["task_name"] = StrIgnoreDevice(self.task_name)
			yield batch


class MultitaskDataloader:
	"""
	Data loader that combines and samples from multiple single-task
	data loaders.
	"""

	def __init__(self, dataloader_dict):
		self.dataloader_dict = dataloader_dict
		self.num_batches_dict = {
			task_name: len(dataloader)
			for task_name, dataloader in self.dataloader_dict.items()
		}
		self.task_name_list = list(self.dataloader_dict)
		self.dataset = [None] * sum(
			len(dataloader.dataset)
			for dataloader in self.dataloader_dict.values()
		)

	def __len__(self):
		return sum(self.num_batches_dict.values())

	def __iter__(self):
		"""
		For each batch, sample a task, and yield a batch from the respective
		task Dataloader.

		We use size-proportional sampling, but you could easily modify this
		to sample from some-other distribution.
		"""
		task_choice_list = []
		for i, task_name in enumerate(self.task_name_list):
			task_choice_list += [i] * self.num_batches_dict[task_name]
		task_choice_list = np.array(task_choice_list)
		np.random.shuffle(task_choice_list)
		dataloader_iter_dict = {
			task_name: iter(dataloader)
			for task_name, dataloader in self.dataloader_dict.items()
		}
		for task_choice in task_choice_list:
			task_name = self.task_name_list[task_choice]
			yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):
	def get_single_train_dataloader(self, task_name, train_dataset):
		"""
		Create a single-task data loader that also yields task names
		"""
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")
		train_sampler = (
			RandomSampler(train_dataset)
			if self.args.local_rank == -1
			else DistributedSampler(train_dataset)
		)

		data_loader = DataLoaderWithTaskname(
			task_name=task_name,
			data_loader=DataLoader(
				train_dataset,
				batch_size=self.args.train_batch_size,
				sampler=train_sampler,
				collate_fn=self.data_collator.collate_batch,
			),
		)

		return data_loader

	def get_train_dataloader(self):
		"""
		Returns a MultitaskDataloader, which is not actually a Dataloader
		but an iterable that returns a generator that samples from each 
		task Dataloader
		"""
		return MultitaskDataloader({
			task_name: self.get_single_train_dataloader(
				task_name, task_dataset)
			for task_name, task_dataset in self.train_dataset.items()
		})

	def train(self, model_path: Optional[str] = None):
		"""
		Main training entry point.

		Args:
				model_path (:obj:`str`, `optional`):
						Local path to the model if the model to train has been instantiated from a local path. If present,
						training will resume from the optimizer/scheduler states loaded here.
		"""
		train_dataloader = self.get_train_dataloader()
		if self.args.max_steps > 0:
			t_total = self.args.max_steps
			num_train_epochs = (
					self.args.max_steps // (len(train_dataloader) //
																	self.args.gradient_accumulation_steps) + 1
			)
		else:
			t_total = int(len(train_dataloader) //
										self.args.gradient_accumulation_steps * self.args.num_train_epochs)
			num_train_epochs = self.args.num_train_epochs

		optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

		# Check if saved optimizer or scheduler states exist
		if (
			model_path is not None
			and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
			and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
		):
			# Load in optimizer and scheduler states
			optimizer.load_state_dict(
					torch.load(os.path.join(model_path, "optimizer.pt"),
											map_location=self.args.device)
			)
			scheduler.load_state_dict(torch.load(
				os.path.join(model_path, "scheduler.pt")))

		model = self.model

		# multi-gpu training (should be after apex fp16 initialization)
		if self.args.n_gpu > 1:
			model = torch.nn.DataParallel(model)

		# Distributed training (should be after apex fp16 initialization)
		if self.args.local_rank != -1:
			model = torch.nn.parallel.DistributedDataParallel(
					model,
					device_ids=[self.args.local_rank],
					output_device=self.args.local_rank,
					find_unused_parameters=True,
			)

		if self.tb_writer is not None:
			self.tb_writer.add_text("args", self.args.to_json_string())
			self.tb_writer.add_hparams(
				self.args.to_sanitized_dict(), metric_dict={})

		# Train!
		total_train_batch_size = (
				self.args.train_batch_size
				* self.args.gradient_accumulation_steps
				* (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
		)
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", self.num_examples(train_dataloader))
		logger.info("  Num Epochs = %d", num_train_epochs)
		logger.info("  Instantaneous batch size per device = %d",
								self.args.per_device_train_batch_size)
		logger.info(
				"  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
		logger.info("  Gradient Accumulation steps = %d",
								self.args.gradient_accumulation_steps)
		logger.info("  Total optimization steps = %d", t_total)

		self.global_step = 0
		self.epoch = 0
		epochs_trained = 0
		steps_trained_in_current_epoch = 0
		# Check if continuing training from a checkpoint
		if model_path is not None:
			# set global_step to global_step of last saved checkpoint from model path
			try:
				self.global_step = int(model_path.split("-")[-1].split("/")[0])
				epochs_trained = self.global_step // (
						len(train_dataloader) // self.args.gradient_accumulation_steps)
				steps_trained_in_current_epoch = self.global_step % (
						len(train_dataloader) // self.args.gradient_accumulation_steps
				)

				logger.info(
						"  Continuing training from checkpoint, will skip to saved global_step")
				logger.info(
						"  Continuing training from epoch %d", epochs_trained)
				logger.info(
						"  Continuing training from global step %d", self.global_step)
				logger.info(
						"  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
			except ValueError:
				self.global_step = 0
				logger.info("  Starting fine-tuning.")

		tr_loss = 0.0
		logging_loss = 0.0
		model.zero_grad()
		train_iterator = trange(
				epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
		)
		for epoch in train_iterator:
			if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
				train_dataloader.sampler.set_epoch(epoch)

			epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

			# Reset the past mems state at the beginning of each epoch if necessary.
			if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
				self._past = None

			for step, inputs in enumerate(epoch_iterator):

				# Skip past any already trained steps if resuming training
				if steps_trained_in_current_epoch > 0:
					steps_trained_in_current_epoch -= 1
					continue

				tr_loss += self._training_step(model, inputs, optimizer)

				if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
					# last step in epoch but step is always smaller than gradient_accumulation_steps
					len(epoch_iterator) <= self.args.gradient_accumulation_steps
					and (step + 1) == len(epoch_iterator)
				):
					torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

					optimizer.step()

					scheduler.step()
					model.zero_grad()
					self.global_step += 1
					self.epoch = epoch + (step + 1) / len(epoch_iterator)

					if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
							self.global_step == 1 and self.args.logging_first_step
					):
						logs: Dict[str, float] = {}
						logs["loss"] = (tr_loss - logging_loss) / \
								self.args.logging_steps
						# backward compatibility for pytorch schedulers
						logs["learning_rate"] = (
								scheduler.get_last_lr()[0]
								if version.parse(torch.__version__) >= version.parse("1.4")
								else scheduler.get_lr()[0]
						)
						logging_loss = tr_loss

						self._log(logs)

					if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
						self.evaluate()

					if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
						# In all cases (even distributed/parallel), self.model is always a reference
						# to the model we want to save.
						if hasattr(model, "module"):
							assert model.module is self.model
						else:
							assert model is self.model
						# Save model checkpoint
						output_dir = os.path.join(
								self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

						temp_dir = "{}".format("/".join(output_dir.split("gs://")[1].split("/")[1:])) if "gs://" in output_dir else output_dir

						self.save_model(temp_dir)

						if self.is_world_master():
							self._rotate_checkpoints()

						if self.is_world_master():
							torch.save(optimizer.state_dict(), os.path.join(
								temp_dir, "optimizer.pt"))
							torch.save(scheduler.state_dict(), os.path.join(
								temp_dir, "scheduler.pt"))

						if "gs://" in output_dir:
							self.save_gcs(output_dir, temp_dir)

				if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
					epoch_iterator.close()
					break
			if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
				train_iterator.close()
				break

		if self.tb_writer:
			self.tb_writer.close()
		if self.args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of training
			delattr(self, "_past")

		logger.info(
				"\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
		return TrainOutput(self.global_step, tr_loss / self.global_step)

	def save_model(self, temp_dir: Optional[str] = None):
		"""
		Will save the model, so you can reload it using :obj:`from_pretrained()`.

		Will only save from the world_master process (unless in TPUs).
		"""

		if self.is_world_master():
			self._save(temp_dir)

	def _save(self, temp_dir: Optional[str] = None):
		os.makedirs(temp_dir, exist_ok=True)
		logger.info("Saving model checkpoint to %s", temp_dir)
		# Save a trained model and configuration using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		if not isinstance(self.model, PreTrainedModel):
			raise ValueError("Trainer.model appears to not be a PreTrainedModel")
		self.model.save_pretrained(temp_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(self.args, os.path.join(temp_dir, "training_args.bin"))

	def save_gcs(self, output_dir: Optional[str] = None, temp_dir: Optional[str] = None):
		output_dir = output_dir if output_dir is not None else self.args.overwrite_output_dir

		bucket_name = output_dir.lstrip("gs://").split("/")[0]
		bucket_path = output_dir.split("gs://{}/".format(bucket_name))[1]

		for filename in ["config.json", "optimizer.pt", "scheduler.pt", "pytorch_model.bin", "training_args.bin"]:
			bucket = storage.Client().bucket(bucket_name)
			blob = bucket.blob("{}/{}".format(bucket_path, filename))
			blob.upload_from_filename("{}/{}".format(temp_dir, filename))


def main():
	args = get_args()

	dataset_dict = {
		"stsb": nlp.load_dataset('glue', name="stsb"),
		"rte": nlp.load_dataset('glue', name="rte"),
		"commonsense_qa": nlp.load_dataset('commonsense_qa'),
	}

	for task_name, dataset in dataset_dict.items():
		print(task_name)
		print(dataset_dict[task_name]["train"][0])
		print()

	multitask_model = MultitaskModel.create(
		model_name=model_name,
		model_type_dict={
			"stsb": transformers.AutoModelForSequenceClassification,
			"rte": transformers.AutoModelForSequenceClassification,
			"commonsense_qa": transformers.AutoModelForMultipleChoice,
		},
		model_config_dict={
			"stsb": transformers.AutoConfig.from_pretrained(model_name, num_labels=1),
			"rte": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
			"commonsense_qa": transformers.AutoConfig.from_pretrained(model_name),
		}
	)

	if model_name.startswith("roberta-"):
		print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
		print(multitask_model.taskmodels_dict["stsb"].roberta.embeddings.word_embeddings.weight.data_ptr())
		print(multitask_model.taskmodels_dict["rte"].roberta.embeddings.word_embeddings.weight.data_ptr())
		print(multitask_model.taskmodels_dict["commonsense_qa"].roberta.embeddings.word_embeddings.weight.data_ptr())

	convert_func_dict = {
		"stsb": convert_to_stsb_features,
		"rte": convert_to_rte_features,
		"commonsense_qa": convert_to_commonsense_qa_features,
	}

	columns_dict = {
		"stsb": ['input_ids', 'attention_mask', 'labels'],
		"rte": ['input_ids', 'attention_mask', 'labels'],
		"commonsense_qa": ['input_ids', 'attention_mask', 'labels'],
	}

	features_dict = {}
	for task_name, dataset in dataset_dict.items():
		features_dict[task_name] = {}
		for phase, phase_dataset in dataset.items():
			features_dict[task_name][phase] = phase_dataset.map(
				convert_func_dict[task_name],
				batched=True,
				load_from_cache_file=False,
			)
			print(task_name, phase, len(phase_dataset),
							len(features_dict[task_name][phase]))
			features_dict[task_name][phase].set_format(
				type="torch",
				columns=columns_dict[task_name],
			)
			print(task_name, phase, len(phase_dataset),
							len(features_dict[task_name][phase]))

	train_dataset = {
		task_name: dataset["train"] for task_name, dataset in features_dict.items()
	}
	trainer = MultitaskTrainer(
		model=multitask_model,
		args=transformers.TrainingArguments(
			output_dir=args.job_dir,
			overwrite_output_dir=True,
			learning_rate=1e-5,
			do_train=True,
			num_train_epochs=3,
			per_device_train_batch_size=args.batch_size,
			save_steps=3000,
		),
		data_collator=NLPDataCollator(),
		train_dataset=train_dataset,
	)
	trainer.train()

	preds_dict = {}
	for task_name in ["rte", "stsb", "commonsense_qa"]:
		eval_dataloader = DataLoaderWithTaskname(
			task_name,
			trainer.get_eval_dataloader(
				eval_dataset=features_dict[task_name]["validation"])
		)
		print(eval_dataloader.data_loader.collate_fn)
		preds_dict[task_name] = trainer._prediction_loop(
			eval_dataloader,
			description=f"Validation: {task_name}",
		)

	# Evalute RTE
	nlp.load_metric('glue', name="rte").compute(
		np.argmax(preds_dict["rte"].predictions, axis=1),
		preds_dict["rte"].label_ids,
	)

	# Evalute STS-B
	nlp.load_metric('glue', name="stsb").compute(
		preds_dict["stsb"].predictions.flatten(),
		preds_dict["stsb"].label_ids,
	)

	# Evalute Commonsense QA
	np.mean(
		np.argmax(preds_dict["commonsense_qa"].predictions, axis=1)
		== preds_dict["commonsense_qa"].label_ids
	)


if __name__ == "__main__":
	main()
