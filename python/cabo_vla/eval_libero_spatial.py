import argparse
import json
from math import isfinite
from pathlib import Path

import av
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from cabo_vla.data.dataloader import Batch
from cabo_vla.training_configs.q35vla import BACKBONE_REPO, DATASET_REPO, load_vla_checkpoint


def quantiles(stats: dict, feature: str):
    low = np.asarray(stats[feature]["q01"], dtype=np.float32)
    span = np.asarray(stats[feature]["q99"], dtype=np.float32) - low
    return low, np.where(span == 0, 1, span)


CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def record_frame(obs: dict) -> np.ndarray:
    return np.concatenate([obs[camera][::-1, ::-1] for camera in CAMERAS], axis=1)


def save_rollout(path: Path, frames: list[np.ndarray]):
    temporary = path.with_suffix(".tmp")
    with av.open(str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}) as video:
        stream = video.add_stream("libx264", rate=20)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    temporary.replace(path)


def ensemble_action(plans: list[tuple[int, np.ndarray]], step: int, decay: float, replan_steps: int) -> np.ndarray:
    current = [(step - start, plan[step - start]) for start, plan in plans]
    ages, actions = zip(*current, strict=True)
    # Keep the newest gripper command while smoothing the six arm controls.
    action = actions[-1].copy()
    action[:6] = np.average(np.asarray(actions)[:, :6], axis=0, weights=decay ** ((np.asarray(ages) - ages[-1]) / replan_steps))
    return action


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Cabo Qwen3.5 VLA checkpoint on LIBERO Spatial.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--video-dir", type=Path, help="Record an MP4 of both camera views for each trial")
    parser.add_argument("--output", type=Path, help="Write rollout results as JSON")
    parser.add_argument("--stats", type=Path, help="LeRobot LIBERO meta/stats.json")
    parser.add_argument("--tokenizer", type=Path, help="Qwen3.5-0.8B tokenizer.json")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--temporal-decay", type=float, help="Weight multiplier per replan interval for overlapping arm actions, in (0, 1]")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.tasks, args.trials, args.max_steps, args.replan_steps, args.denoise_steps) <= 0:
        parser.error("Task, trial, and step counts must be positive")

    if args.temporal_decay is not None and (not isfinite(args.temporal_decay) or not 0 < args.temporal_decay <= 1):
        parser.error("--temporal-decay must be in (0, 1]")

    try:
        from libero.libero import benchmark, get_libero_path  # ty: ignore[unresolved-import]
        from libero.libero.envs import OffScreenRenderEnv  # ty: ignore[unresolved-import]
        from robosuite.utils.transform_utils import quat2axisangle  # ty: ignore[unresolved-import]
    except ImportError as error:
        parser.error(f"LIBERO evaluation requires the Linux hf-libero package: {error}")

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if args.tasks > suite.n_tasks:
        parser.error(f"LIBERO Spatial has only {suite.n_tasks} tasks")
    cfg, model = load_vla_checkpoint(args.checkpoint)
    if args.replan_steps > cfg.action_chunk_size:
        parser.error("Replan interval must fit the action chunk")
    if cfg.pad_token_id is None:
        parser.error("The checkpoint must provide a pad token ID")
    tokenizer = Tokenizer.from_file(str(args.tokenizer or hf_hub_download(BACKBONE_REPO, "tokenizer.json")))
    stats_file = args.stats or Path(hf_hub_download(DATASET_REPO, "meta/stats.json", repo_type="dataset"))
    stats = json.loads(stats_file.read_text())
    state_low, state_span = quantiles(stats, "observation.state")
    action_low, action_span = quantiles(stats, "action")
    pixel_low, pixel_high = cfg.observation_range

    @nnx.jit
    def predict(model, batch, key):
        context = model.encode_batch(batch)
        noise = jax.random.normal(key, batch.action.shape, dtype=jnp.float32)
        dt = 1 / args.denoise_steps

        def denoise(i, actions):
            t = jnp.full((actions.shape[0],), 1 - i * dt, dtype=jnp.float32)
            return actions - dt * model.run_action_head(context, actions, t).astype(jnp.float32)

        return jax.lax.fori_loop(0, args.denoise_steps, denoise, noise)[0]

    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(args.seed)
    results = []
    for task_id in range(args.tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if args.trials > len(initial_states):
            parser.error(f"Task {task_id} has only {len(initial_states)} initial states")
        ids = tokenizer.encode(task.language, add_special_tokens=False).ids[:64]
        instruction = jnp.asarray([ids + [cfg.pad_token_id] * (64 - len(ids))], dtype=jnp.int32)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        successes = 0
        try:
            for trial in range(args.trials):
                env.reset()
                obs = env.set_init_state(initial_states[trial])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                frames = [record_frame(obs)] if args.video_dir else None
                plans: list[tuple[int, np.ndarray]] = []
                success = False
                for step in range(args.max_steps):
                    if step % args.replan_steps == 0:
                        # Match the OpenPI LIBERO camera orientation and state representation.
                        pixels = np.stack([obs[camera][::-1, ::-1] for camera in CAMERAS])
                        pixels = pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255
                        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]]).astype(
                            np.float32
                        )
                        batch = Batch(
                            observations=jnp.asarray((pixel_low + pixels * (pixel_high - pixel_low))[None]),
                            instruction=instruction,
                            action=jnp.zeros((1, cfg.action_chunk_size, cfg.action_size), dtype=jnp.float32),
                            prio=jnp.asarray((2 * (state - state_low) / state_span - 1)[None]),
                            timestamp=jnp.zeros((1,), dtype=jnp.float32),
                        )
                        key, plan_key = jax.random.split(key)
                        normalized = np.asarray(predict(model, batch, plan_key))
                        plan = np.clip(action_low + (normalized + 1) * action_span / 2, -1, 1)
                        if args.temporal_decay is not None:
                            plans.append((step, plan))
                    if args.temporal_decay is not None:
                        plans = [(start, chunk) for start, chunk in plans if step - start < len(chunk)]
                        action = ensemble_action(plans, step, args.temporal_decay, args.replan_steps)
                    else:
                        action = plan[step % args.replan_steps]
                    obs, _, done, _ = env.step(action.tolist())
                    if frames is not None:
                        frames.append(record_frame(obs))
                    if done:
                        success = True
                        break
                if args.video_dir and frames is not None:
                    outcome = "success" if success else "failure"
                    save_rollout(args.video_dir / f"task-{task_id:02d}-trial-{trial + 1:02d}-{outcome}.mp4", frames)
                successes += int(success)
                print(f"Task {task_id}, trial {trial + 1}: {'success' if success else 'failure'}", flush=True)
        finally:
            env.close()
        results.append({"task_id": task_id, "task": task.language, "successes": successes, "trials": args.trials})
    total = args.tasks * args.trials
    successes = sum(result["successes"] for result in results)
    summary = json.dumps(
        {
            "suite": "libero_spatial",
            "successes": successes,
            "trials": total,
            "success_rate": successes / total,
            "temporal_decay": args.temporal_decay,
            "tasks": results,
        },
        indent=2,
    )
    if args.output:
        args.output.write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
