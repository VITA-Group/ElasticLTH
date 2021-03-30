# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
import warnings

import torch
from datasets.base import DataLoader
import datasets.registry
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.base import Model, DataParallel, DistributedDataParallel
import models.registry
from platforms.platform import get_platform
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger

try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True


def train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None,
    suffix: str = ''
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch, suffix)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), labels)
            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

    get_platform().barrier()


def accumulate(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    data_order_seed: int = None,
    suffix: str = ''
):

    """Accumulate the gradient for one training epoch.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * data_order_seed: The RNG seed for data shuffling.
    """

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model = apex.amp.initialize(model, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    train_loader.shuffle(data_order_seed)

    for it, (examples, labels) in enumerate(train_loader):

        examples = examples.to(device=get_platform().torch_device)
        labels = labels.to(device=get_platform().torch_device)

        model.eval()
        loss = model.loss_criterion(model(examples), labels)
        if training_hparams.apex_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    get_platform().barrier()


def grasp(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    parameter_list: list,
    train_loader: DataLoader,
    data_order_seed: int = None,
    suffix: str = ''
):

    """For the implementation of GraSP.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * data_order_seed: The RNG seed for data shuffling.
    """

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model = apex.amp.initialize(model, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    train_loader.shuffle(data_order_seed)

    # First gradient without computational graph
    stopped_grads = 0
    for it, (examples, labels) in enumerate(train_loader):

        examples = examples.to(device=get_platform().torch_device)
        labels = labels.to(device=get_platform().torch_device)

        model.eval()
        output = model(examples) / 200.0  # temp = 200
        loss = model.loss_criterion(output, labels)
        # if training_hparams.apex_fp16:
        #     with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        grads = torch.autograd.grad(loss, parameter_list, create_graph=False)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads

    train_loader.shuffle(None if data_order_seed is None else (data_order_seed + 1))
    # Second gradient vector with computational graph
    for it, (examples, labels) in enumerate(train_loader):

        examples = examples.to(device=get_platform().torch_device)
        labels = labels.to(device=get_platform().torch_device)

        model.eval()
        output = model(examples) / 200.0  # temp = 200
        loss = model.loss_criterion(output, labels)
        # if training_hparams.apex_fp16:
        #     with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        grads = torch.autograd.grad(loss, parameter_list, create_graph=True)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

    get_platform().barrier()




def distill(
    training_hparams: hparams.TrainingHparams,
    distill_hparams: hparams.DistillHparams,
    student: Model,
    teacher: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * distll_hparams: The knowledge distillation hyperparameters whose schema is specified in hparams.py.
      * student: The student model to train. Must be a models.base.Model
      * teacher: The teacher model to distill the knowledge. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    student.to(get_platform().torch_device)
    teacher.to(get_platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, student)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    if distill_hparams.alpha_mse > 0.0:
        mse_loss_fct = nn.MSELoss(reduction='sum')
    if distill_hparams.alpha_cos > 0.0:
        cos_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        (student, teacher), step_optimizer = apex.amp.initialize(
            [student, teacher], optimizer, loss_scale='dynamic', verbosity=0
        )

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        student = DistributedDataParallel(student, device_ids=[get_platform().rank])
        teacher = DistributedDataParallel(teacher, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        student = DataParallel(student)
        teacher = DataParallel(teacher)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, student, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, student, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            loss = 0.0
            step_optimizer.zero_grad()
            student.train()
            teacher.eval()

            student_outputs = student(examples)
            with torch.no_grad():
                teacher_outputs = teacher(examples)

            s_logits = student_outputs
            t_logits = teacher_outputs

            # KL Divergence loss for the knowledge distillation
            loss_ce = ce_loss_fct(
                F.log_softmax(s_logits / distill_hparams.temperature, dim=-1),
                F.softmax(t_logits / distill_hparams.temperature, dim=-1),
            ) * distill_hparams.temperature**2
            loss += distill_hparams.alpha_ce * loss_ce

            if distill_hparams.alpha_cls > 0.0:
                loss_cls = student.loss_criterion(student_outputs, labels)
                loss += distill_hparams.alpha_cls * loss_cls

            if distill_hparams.alpha_mse > 0.0:
                loss_mse = mse_loss_fct(s_logits, t_logits) / s_logits.size(0)
                loss += distill_hparams.alpha_mse * loss_mse

            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

    get_platform().barrier()


def standard_train(
  model: Model,
  output_location: str,
  dataset_hparams: hparams.DatasetHparams,
  training_hparams: hparams.TrainingHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True,
  suffix: str = ''
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
        get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch, suffix=suffix)
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step, suffix=suffix)


def accumulate_gradient(
  training_hparams: hparams.TrainingHparams,
  model: Model,
  dataset_hparams: hparams.DatasetHparams,
  data_order_seed: int = None,
  verbose: bool = True,
  suffix: str = ''
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    # train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    # if (models.registry.exists(output_location, train_end_step) and
    #     get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    # callbacks = standard_callbacks.standard_callbacks(
    #     training_hparams, train_loader, test_loader, start_step=start_step,
    #     verbose=verbose, evaluate_every_epoch=evaluate_every_epoch, suffix=suffix)
    accumulate(training_hparams, model, train_loader, data_order_seed, suffix=suffix)


def run_grasp(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    parameter_list: list,
    dataset_hparams: hparams.DatasetHparams,
    data_order_seed: int = None,
    verbose: bool = True,
    suffix: str = ''
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    # train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    # if (models.registry.exists(output_location, train_end_step) and
    #     get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    # callbacks = standard_callbacks.standard_callbacks(
    #     training_hparams, train_loader, test_loader, start_step=start_step,
    #     verbose=verbose, evaluate_every_epoch=evaluate_every_epoch, suffix=suffix)
    grasp(training_hparams, model, parameter_list, train_loader, data_order_seed, suffix=suffix)


def distill_train(
  student: Model,
  teacher: Model,
  output_location: str,
  dataset_hparams: hparams.DatasetHparams,
  training_hparams: hparams.TrainingHparams,
  distill_hparams: hparams.DistillHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True,
  suffix: str = ""
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
        get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch, suffix=suffix)
    distill(training_hparams, distill_hparams, student, teacher,
            train_loader, output_location, callbacks, start_step=start_step)
