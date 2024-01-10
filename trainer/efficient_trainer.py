# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer,default_data_collator
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
from transformers.trainer_utils import (EvalPrediction, EvalLoopOutput, TrainOutput,seed_worker,
                                        has_length, speed_metrics, denumpify_detensorize)
from transformers.utils import logging
from transformers.training_args import TrainingArguments
from utils.deepspeed_utils import is_deepspeed_zero3_enabled, deepspeed_init
from args import AdditionalArguments
import random
import datasets
from transformers.utils import is_datasets_available
import deepspeed
import utils.lora_utils as lora
from transformers.trainer_callback import TrainerState
import mlflow
import torch.nn as nn
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
mlflow.autolog()

logger = logging.get_logger(__name__)

class EfficientTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: TrainingArguments = None,
            additional_args: AdditionalArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            l0_module=None,
            use_lora: bool = False,
            lora_train_bias: str = "none",
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.additional_args = additional_args
        
        self.l0_module = l0_module
        self.prepruning_finetune_steps = 0
        self.start_prune = False
        self.zs = None
        self.l0_optimizer = None
        self.lagrangian_optimizer = None
        self.global_step = 0
        self.start_saving_best = False # if self.additional_args.pruning_type is None else False

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)
        self.use_lora = use_lora
        self.lora_train_bias = lora_train_bias
        if self.use_lora:
            logger.info("LoRA enabled.")
        if self.additional_args.do_iterative_distill:
            from utils.distill_utils import IterativeDistillManager
            self.iter_manager = IterativeDistillManager()

    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.optimizer is None and self.use_lora:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
            ]
            log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if build_l0_optimizer and self.l0_module is not None:
            l0_params = [{
                "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                "weight_decay": 0.0,
                "lr": self.additional_args.reg_learning_rate
            }]
            log_params(l0_params, "l0 reg params")
            self.l0_optimizer = AdamW(l0_params,
                                        betas=(self.args.adam_beta1,
                                                self.args.adam_beta2),
                                        eps=self.args.adam_epsilon, )
            lagrangian_params = [{
                "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                "weight_decay": 0.0,
                "lr": -self.additional_args.reg_learning_rate
            }]
            log_params(lagrangian_params, "l0 reg lagrangian params")
            self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                betas=(self.args.adam_beta1,
                                                        self.args.adam_beta2),
                                                eps=self.args.adam_epsilon)
        if self.lr_scheduler is None:
            if self.additional_args.scheduler_type == "linear" and self.optimizer is not None:
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None
        return

    def train(self,resume_from_checkpoint):
        self._memory_tracker.start()
        args = self.args
        train_dataloader = self.get_train_dataloader()
        trial=None
        self._hp_search_setup(trial)
        self.is_in_train = True
        self._train_batch_size = self.args.train_batch_size

        import datetime
        now = datetime.datetime.now()
        self.args.output_dir = os.path.join(self.args.output_dir, 'Compresso-{}-s{}-lr{}-reglr{}-warmup{}/{}-{}-{}-{}-{}'.format(
            self.additional_args.task_name,
            self.additional_args.target_sparsity*100,
            self.args.learning_rate,
            self.additional_args.reg_learning_rate,
            self.additional_args.lagrangian_warmup_epochs,
            now.year, now.month, now.day, now.hour, now.minute
        ))
        logger.info(f"Output dir: {self.args.output_dir}")
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        logger.info("building folder finish")

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
        
        #add deepspeed https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        if self.use_lora:
            import utils.lora_utils as lora
            total_params = sum([p.ds_numel if hasattr(p,'ds_numel') else p.numel() for p in self.model.parameters()])
            logger.info(f"Number of total parameters: {total_params}")
            trainable_params = sum([p.ds_numel if hasattr(p,'ds_numel') else p.numel() for p in self.model.parameters() if p.requires_grad])
            logger.info(f"Number of trainable parameters before applying LoRA: {trainable_params}")
            lora.mark_only_lora_as_trainable(self.model, bias=self.lora_train_bias)
            lora.set_lora_as_float32(self.model)
            trainable_params = sum([p.ds_numel if hasattr(p,'ds_numel') else p.numel() for p in self.model.parameters() if p.requires_grad])
            logger.info(f"Number of trainable parameters after applying LoRA: {trainable_params}")
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        model = self._wrap_model(self.model_wrapped)
        self.model_wrapped = model
        # import copy
        # self.teacher_model = copy.deepcopy(model)

        # teacher model for distillation
        if self.additional_args.do_distill:
            pass

        # the value of self.prepruning_finetune_steps is zero if finetune
        if self.additional_args.pretrained_pruned_model is None:
            self.prepruning_finetune_steps = 0
            # self.prepruning_finetune_steps = len_dataloader * self.additional_args.prepruning_finetune_epochs
        if self.l0_module is not None:
            lagrangian_warmup_steps = self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        self.t_total = max_steps
        if self.additional_args.uniform:
            self.l0_module.prepare_uniform_lambda()
        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)
            
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
        logger.info("  Total optimization steps = %d", self.t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        ce_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_ce_loss_scalar = 0.0
        logging_lag_loss_scalar = 0.0

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(
            np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        # training
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))): #! 20 epoch
            epoch_start = time.time()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader
            print("training on dataset with pruning prompt")

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration",
                              disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):
                if self.l0_module is not None and self.global_step == self.prepruning_finetune_steps: 
                    self.start_prune = True
                    lr_steps = self.t_total - self.global_step
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    logger.info("Starting l0 regularization!")
                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    self.fill_inputs_with_zs(zs, inputs) #! use the zs

                # self.zs is not None when finetune
                if self.zs is not None:
                    self.fill_inputs_with_zs(self.zs, inputs)

                loss_terms = self.training_step(model, self.additional_args.do_distill, inputs)
                if loss_terms is None:
                    print('inf detected, skip')
                    continue
                tr_loss_step = loss_terms["loss"]
                ce_loss_step = loss_terms["ce_loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]

                tr_loss += tr_loss_step
                ce_loss += ce_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                    if self.optimizer is not None:
                        if self.deepspeed:
                            self.deepspeed.step()
                        else:
                            self.optimizer.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None and not self.deepspeed:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        ce_loss_scalar = ce_loss.item()
                        lag_loss_scalar = lag_loss.item()

                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["ce_loss"] = (
                            ce_loss_scalar - logging_ce_loss_scalar) / self.args.logging_steps
                        logs["lag_loss"] = (
                            lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logs["lambda_1"] = self.l0_module.lambda_1.mean().item() if self.l0_module is not None else None
                        logs["lambda_2"] = self.l0_module.lambda_2.mean().item() if self.l0_module is not None else None
                        if self.additional_args.uniform:
                            logs["lambda_3"] = self.l0_module.lambda_3.mean().item() if self.l0_module is not None else None
                            logs["lambda_4"] = self.l0_module.lambda_3.mean().item() if self.l0_module is not None else None
                        logs["expected_sparsity"] = loss_terms["expected_sparsity"]
                        logs["target_sparsity"] = loss_terms["target_sparsity"]
                        logging_loss_scalar = tr_loss_scalar
                        logging_ce_loss_scalar = ce_loss_scalar
                        logging_lag_loss_scalar = lag_loss_scalar

                        # self.log(logs)
                        if self.args.local_rank <= 0:
                            for k, v in logs.items():
                                try:
                                    mlflow.log_metric(k, v, step=self.state.global_step)
                                except:
                                    pass

                        # try:
                        if self.l0_module is not None:
                            self.l0_module.eval()
                            zs = self.l0_module.forward(training=False)
                            pruned_model_size_info = self.l0_module.calculate_model_size(zs)
                        else:
                            pruned_model_size_info = {}
                        # except:
                        #     pruned_model_size_info = {}

                        if self.args.local_rank <= 0:
                            for k, v in pruned_model_size_info.items():
                                try:
                                    mlflow.log_metric(k, v, step=self.state.global_step)
                                except:
                                    pass

                        logger.info(f"{logs}, {pruned_model_size_info}")

                epoch_pbar.update(1)
                torch.cuda.empty_cache()
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

                # save on specific steps
                if self.global_step % 500 == 0:
                    self.save_checkpoint(model, f"step_{self.global_step}")

            epoch_end = time.time()
            torch.cuda.empty_cache()

            self.save_checkpoint(model, f"epoch{epoch}")

            logger.info(f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
            
        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, None)

    def save_checkpoint(self, model, name):
        if self.args.local_rank <= 0:
            epoch_output_dir = '{}/{}'.format(self.args.output_dir, name)
            print("Epoch folder: ", epoch_output_dir)
            if not os.path.exists(epoch_output_dir):
                os.makedirs(epoch_output_dir)
            if self.use_lora:
                lora_weights = {}
                for n, m in model.named_parameters():
                    if 'lora_' in n:
                        gather = lora.should_gather(m)
                        with gather:
                            lora_weights[n.replace('module.','')] = m.data
                torch.save(lora_weights,'{}/lora_weights.pt'.format(epoch_output_dir))
            self.save_model_mask(epoch_output_dir)
    
    def save_model_mask(self, output_dir: Optional[str] = None):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.zs == None and self.l0_module is not None:
            torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))
            zs = self.l0_module.forward(training=False)
            torch.save(zs, os.path.join(output_dir, "zs.pt"))
        elif self.zs is not None:
            torch.save(self.zs, os.path.join(output_dir, "zs.pt"))

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        layer_loss = 0
        if self.additional_args.do_layer_distill: #! only do layer distill
            mlp_z = None
            head_layer_z = None
            # logger.info(f"zs={zs}")
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            teacher_layer_output = teacher_outputs['hidden_states'][1:]
            student_layer_output = student_outputs['hidden_states'][1:]
            # teacher_layer_output = teacher_outputs[3][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            # student_layer_output = student_outputs[3][1:] 

            # distilliting existing layers
            layer_losses = []
            if self.additional_args.layer_distill_version == 2:
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    # s_layer_o = self.model.layer_transformation(s_layer_o)
                    t_layer_o = t_layer_o.to(torch.float32)
                    s_layer_o = s_layer_o.to(torch.float32)
                    s_layer_non_zero = s_layer_o[s_layer_o != 0]
                    t_layer_non_zero = t_layer_o[s_layer_o != 0]
                    l = mse_loss(t_layer_non_zero, s_layer_non_zero).to(torch.float16)
                    layer_losses.append(l)
                    if mlp_z is None or mlp_z[layer_num] > 0:
                        layer_loss += l
                # print("Layer distill loss", layer_losses)

            # distilling layers with a minimal distance
            elif self.additional_args.layer_distill_version > 2:
                l = []
                if self.additional_args.layer_distill_version > 4:
                    specified_teacher_layers = [i for i in range(len(teacher_layer_output))]
                    if self.additional_args.layer_distill_version ==5:
                        specified_teacher_layers = sorted(random.sample(specified_teacher_layers, 4))
                    elif self.additional_args.layer_distill_version ==6:
                        result_layers_T= []
                        skip_window = len(specified_teacher_layers)//4
                        for i in range(0, len(specified_teacher_layers), skip_window):
                            result_layers_T.append(random.sample(specified_teacher_layers[i:i+skip_window], 1)[0])
                        specified_teacher_layers = result_layers_T
                    specified_teacher_layers[0] = max(2, specified_teacher_layers[0])
                else:
                    specified_teacher_layers = [2, 5, 8, 11]
                # logger.info(f"sampled teacher layers: {specified_teacher_layers}")
                # transformed_s_layer_o = [self.model.layer_transformation(
                    # s_layer_o) for s_layer_o in student_layer_output]
                transformed_s_layer_o = student_layer_output
                specified_teacher_layer_reps = [
                    teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

                device = transformed_s_layer_o[0].device
                for t_layer_o in specified_teacher_layer_reps:
                    for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
                        l.append(mse_loss(t_layer_o, s_layer_o))
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0
                    existing_layers = existing_layers.to(layerwiseloss.device)

                #! no ordering restriction specified
                if self.additional_args.layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                #! added the ordering restriction -> to choose the min loss in 4 student layers
                elif self.additional_args.layer_distill_version in (3, 4, 5, 6):
                    last_aligned_layer = 12
                    alignment = []
                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{self.additional_args.layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(len(specified_teacher_layers)).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                if self.global_step % 100 == 0:
                    logger.info(f"v{self.additional_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
            return layer_loss
        else:
            return None
          
    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)
        distill_loss = layer_loss

        input=F.log_softmax(student_outputs[1] / self.additional_args.distill_temp, dim=-1)
        target=F.softmax(teacher_outputs[1] / self.additional_args.distill_temp, dim=-1)

        ce_distill_loss = F.kl_div(
            input=input, #! logits: [32,3]
            target=target, #! distill_temp: 2.0
            reduction="batchmean") * (self.additional_args.distill_temp ** 2)

        loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
        if distill_loss is not None:
            loss += self.additional_args.distill_loss_alpha * distill_loss

        return distill_loss, ce_distill_loss, loss

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def training_step(self, model: torch.nn.Module, distill: bool, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.l0_module is not None:
            self.l0_module.train()
        inputs = self._prepare_inputs(inputs)

        distill_loss = None
        distill_ce_loss = None
        if distill:
            # ----------step 1: backup student lora weights----------
            student_lora_weights = {}
            for n, m in model.named_parameters():
                if 'lora_' in n:
                    student_lora_weights[n.replace('module.','')] = m.data.detach().clone()
            # ----------step 2: get teacher model and get teacher output----------
            with torch.no_grad():
                model.eval()
                teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                       "output_attentions", "output_hidden_states", "return_dict"]
                teacher_inputs = {key: inputs[key]
                                  for key in teacher_inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                teacher_zs, teacher_lora = None, None
                if self.additional_args.do_iterative_distill:
                    teacher_zs_teacher_lora = self.iter_manager.check_use_iterative_distill(self.t_total - self.global_step)
                    if teacher_zs_teacher_lora is not None:
                        teacher_zs, teacher_lora = teacher_zs_teacher_lora
                    if teacher_zs is not None:
                        self.fill_inputs_with_zs(teacher_zs, teacher_inputs)
                # update model by teacher lora_weights
                if teacher_lora is not None:
                    for n, m in model.named_parameters():
                        if 'lora_' in n:
                            m.data = teacher_lora[n.replace('module.','')].clone().to(m.device)
                else:
                    for n, m in model.named_parameters():
                        if 'lora_' in n:
                            m.data *= 0
                teacher_outputs = model(**teacher_inputs)

            # ----------step 3: restore model to student----------
            for n, m in model.named_parameters():
                if 'lora_' in n:
                    m.data = student_lora_weights[n.replace('module.','')].to(m.device)
            model.train()
            self.shortens_inputs(inputs)
            student_outputs = model(**inputs) #! get the two outputs

            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                teacher_outputs, student_outputs, zs)
        else:
            loss = self.compute_loss(model, inputs)
        if torch.isinf(loss).any():
            return None

        ce_loss = loss.clone()
        lagrangian_loss = None
        expected_sparsity = None
        target_sparsity = None
        
        if self.start_prune :
            if self.additional_args.uniform:
                lagrangian_loss, expected_sparsity, target_sparsity = self.l0_module.layerwise_lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps, self.t_total)
            else:
                lagrangian_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)
            loss += lagrangian_loss
            if self.additional_args.do_iterative_distill:
                self.iter_manager.update(expected_sparsity.item(), self.l0_module, self.t_total - self.global_step)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.deepspeed:
            loss = self.deepspeed.backward(loss)  #10-22G * number of GPU
        else:
            loss.backward()
        return {"loss": loss.detach(),
                "ce_loss": ce_loss.detach(),
                "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
                "expected_sparsity": expected_sparsity.item() if expected_sparsity is not None else 0.0,
                "target_sparsity": target_sparsity if target_sparsity is not None else 0.0,
                "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
                "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key].to(inputs["input_ids"].device)
        if self.l0_module is not None:
            inputs["block_layer_start"] = self.l0_module.block_layer_start
            inputs["block_layer_end"] = self.l0_module.block_layer_end