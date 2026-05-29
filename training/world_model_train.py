"""
Dual-stream world model training entry point.

Wires the dataset, the DDP rollout-aware sampler, and the
WanDualStreamWorldModelModule into an accelerate training loop with swanlab
logging. Each episode is trained as 1+ autoregressive rollout chunks; the
sampler keeps the rollout count identical across DDP ranks at every step.
"""

import torch, os, json
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import swanlab

from world_model_module import WanDualStreamWorldModelModule, parse_start_epoch
from dataset import RoboTwinWorldModelDataset
from sampler import RolloutAwareSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    return {
        "rgb_video": batch[0]["rgb_video"],
        "flow_video": batch[0]["flow_video"],
        "reference_image": batch[0]["reference_image"],
        "prompt": batch[0]["prompt"],
        "num_rollouts": batch[0]["num_rollouts"],
    }


def launch_training_with_swanlab(
    dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    start_epoch: int = 0,
    args=None,
):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters
    save_every_n_epochs = getattr(args, "save_every_n_epochs", 1)

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=find_unused_parameters
        )],
    )

    sampler = RolloutAwareSampler(
        rollout_counts=dataset.rollout_counts,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        bucket_oversample=args.bucket_oversample,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    if accelerator.is_main_process:
        training_config = {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "start_epoch": start_epoch,
            "flow_loss_weight": args.flow_loss_weight,
            "max_stride": args.max_stride,
            "max_rollouts": args.max_rollouts,
            "bucket_oversample": args.bucket_oversample,
            "output_path": args.output_path,
            "mode": "dual_stream_world_model",
        }
        swanlab.init(project="wan-dual-stream-world-robotwin", config=training_config)

        os.makedirs(model_logger.output_path, exist_ok=True)
        config_path = os.path.join(model_logger.output_path, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)

    global_step = start_epoch * len(dataloader)
    for epoch_id in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch_id)

        for data in tqdm(dataloader, desc=f"Epoch {epoch_id}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss_dict = model(data)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                if accelerator.is_main_process:
                    log_data = {
                        "loss": loss.item(),
                        "loss_rgb": loss_dict["loss_rgb"].item(),
                        "loss_flow": loss_dict["loss_flow"].item(),
                        "epoch": epoch_id,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "num_rollouts": data["num_rollouts"],
                    }
                    swanlab.log(log_data, step=global_step)
                global_step += 1

        if save_steps is None:
            is_first = (epoch_id == start_epoch)
            is_last = (epoch_id == num_epochs - 1)
            should_save = (epoch_id + 1) % save_every_n_epochs == 0 or is_first or is_last
            if should_save:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
    if accelerator.is_main_process:
        swanlab.finish()


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--variant", type=str, nargs="+",
        default=["aloha-agilex_clean_50"],
        help="One or more variant subdir names under each task directory "
             "(<data_root>/<task>/<variant>/). All listed variants are "
             "scanned per task and their episodes are concatenated into a "
             "single flat sample list. Tasks that don't have a given variant "
             "on disk are silently skipped, so it's safe to pass a variant "
             "that is still being generated for some tasks. "
             "Example: --variant aloha-agilex_clean_50 new_clean",
    )
    parser.add_argument("--camera", type=str, default="head_camera")
    parser.add_argument(
        "--task_names", type=str, nargs="+", default=None,
        help="Subset of task names to train on (default: all RoboTwin tasks).",
    )
    parser.add_argument("--flow_method", type=str, default="raft", choices=["raft", "farneback"])
    parser.add_argument("--flow_device", type=str, default="cuda")
    parser.add_argument("--flow_max_magnitude", type=float, default=25.0)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--flow_loss_weight", type=float, default=0.5)
    parser.add_argument("--ref_aug_strength", type=float, default=0.1)
    parser.add_argument("--fp32_modulation", action="store_true", default=False)
    parser.add_argument("--save_every_n_epochs", type=int, default=20)
    parser.add_argument("--low_res_data_root", type=str, required=True)
    parser.add_argument("--max_stride", type=int, default=3)
    parser.add_argument("--max_rollouts", type=int, default=2)
    parser.add_argument("--bucket_oversample", type=int, default=3)
    args = parser.parse_args()

    dataset = RoboTwinWorldModelDataset(
        data_root=args.dataset_base_path,
        low_res_data_root=args.low_res_data_root,
        variant=args.variant,
        camera=args.camera,
        task_names=args.task_names,
        size=tuple(args.size),
        chunk_output_frames=args.num_frames,
        max_stride=args.max_stride,
        max_rollouts=args.max_rollouts,
        flow_method=args.flow_method,
        flow_device=args.flow_device,
        flow_max_magnitude=args.flow_max_magnitude,
    )

    model = WanDualStreamWorldModelModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        resume_checkpoint=args.resume_checkpoint,
        flow_loss_weight=args.flow_loss_weight,
        ref_aug_strength=args.ref_aug_strength,
        fp32_modulation=args.fp32_modulation,
        chunk_output_frames=args.num_frames,
    )

    start_epoch = parse_start_epoch(args.resume_checkpoint)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launch_training_with_swanlab(
        dataset, model, model_logger, start_epoch=start_epoch, args=args
    )
