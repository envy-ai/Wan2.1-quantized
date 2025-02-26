import argparse
import gradio as gr
import os
import sys
from datetime import datetime
import warnings
import torch, random
import torch.distributed as dist
from PIL import Image
import json  # For saving settings as JSON

# Suppress warnings (as in your original script)
warnings.filterwarnings('ignore')

# --- Import your wan modules and functions ---
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# --- Global pipeline variables ---
global_wan_t2v = None
global_wan_t2v_config = None
global_wan_i2v = None
global_wan_i2v_config = None

# --- Your original script code (unchanged parts) ---
EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

def resize_with_mode(img, target_size, mode):
    if mode == "none":
        return img
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    if mode == "crop":
        scale = max(target_w / orig_w, target_h / orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        left = (new_size[0] - target_w) // 2
        top = (new_size[1] - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    elif mode == "pad":
        scale = min(target_w / orig_w, target_h / orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        resized_img = img.resize(new_size, Image.LANCZOS)
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        left = (target_w - new_size[0]) // 2
        top = (target_h - new_size[1]) // 2
        new_img.paste(resized_img, (left, top))
        img = new_img
    else:
        raise ValueError("Unsupported resize mode: " + mode)
    return img

def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50
    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832", "720*480", "480*720"]:
            args.sample_shift = 3.0
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _init_logging(rank):
    import logging
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    import logging
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    logging.info("Starting generation process...")

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, "The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=args.ring_size, ulysses_degree=args.ulysses_size)

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task, device=rank)
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    pipeline_config = (args.task, args.ckpt_dir, args.offload_model, args.t5_fsdp, args.dit_fsdp, args.t5_cpu)

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.use_prompt_extend:
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                input_prompt = prompt_output.prompt if prompt_output.status else args.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
        logging.info("Using WanT2V pipeline...")
        global global_wan_t2v, global_wan_t2v_config
        if global_wan_t2v is None or global_wan_t2v_config != pipeline_config:
            print("Initializing new WanT2V pipeline...")
            global_wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
            global_wan_t2v_config = pipeline_config
        wan_t2v = global_wan_t2v
        logging.info("Generating video/image...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            n_prompt=args.n_prompt  # Pass the negative prompt here
        )
    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        if isinstance(args.image, str):
            img = Image.open(args.image).convert("RGB")
        else:
            img = args.image.convert("RGB")
        if args.use_prompt_extend:
            if rank == 0:
                prompt_output = prompt_expander(args.prompt, tar_lang=args.prompt_extend_target_lang, image=img, seed=args.base_seed)
                input_prompt = prompt_output.prompt if prompt_output.status else args.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
        target_size = SIZE_CONFIGS[args.size]
        img = resize_with_mode(img, target_size, args.resize_mode)
        logging.info("Using WanI2V pipeline...")
        global global_wan_i2v, global_wan_i2v_config
        if global_wan_i2v is None or global_wan_i2v_config != pipeline_config:
            print("Initializing new WanI2V pipeline...")
            global_wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
            global_wan_i2v_config = pipeline_config
        wan_i2v = global_wan_i2v
        logging.info("Generating video...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            n_prompt=args.n_prompt  # Pass the negative prompt here as well
        )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = "".join([c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in args.save_file])
        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(tensor=video.squeeze(1)[None],
                        save_file=args.save_file,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(tensor=video[None],
                        save_file=args.save_file,
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
        logging.info(f"Output saved to {args.save_file}")
        # Save settings to a JSON file with the same base name.
        json_filename = os.path.splitext(args.save_file)[0] + ".json"
        # Remove the image from args so it can be serialized.
        args_copy = args.copy()
        args_copy.image = None
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(vars(args_copy), f, ensure_ascii=False, indent=4)
    return args.save_file

def run_wan_pipeline(
    image,                   # 1. Input Image (for i2v tasks)
    prompt,                  # 2. Prompt
    n_prompt,                # 3. Negative Prompt
    resize_mode,             # 4. Image Resize Mode
    task,                    # 5. Task
    size,                    # 6. Size
    frame_num,               # 7. Frame Num
    ckpt_dir,                # 8. Checkpoint Directory
    offload_model,           # 9. Offload Model
    save_file,               # 10. Save File (optional)
    base_seed,               # 11. Base Seed
    sample_solver,           # 12. Sample Solver
    sample_steps,            # 13. Sample Steps
    sample_shift,            # 14. Sample Shift
    sample_guide_scale,      # 15. Sample Guide Scale
    ulysses_size,            # 16. Ulysses Size
    ring_size,               # 17. Ring Size
    t5_fsdp,                 # 18. T5 FSDP
    t5_cpu,                  # 19. T5 CPU
    dit_fsdp,                # 20. DiT FSDP
    use_prompt_extend,       # 21. Use Prompt Extend
    prompt_extend_method,    # 22. Prompt Extend Method
    prompt_extend_model,     # 23. Prompt Extend Model (optional)
    prompt_extend_target_lang,# 24. Prompt Extend Target Language
    progress=gr.Progress(track_tqdm=True)  # 25. Progress bar
):
    print("Starting Wan pipeline generation via Gradio...")
    progress(0.1)
    args = argparse.Namespace(
        task=task,
        size=size,
        frame_num=int(frame_num) if frame_num is not None else None,
        ckpt_dir=ckpt_dir if ckpt_dir.strip() != "" else None,
        offload_model=offload_model,
        ulysses_size=int(ulysses_size),
        ring_size=int(ring_size),
        t5_fsdp=t5_fsdp,
        t5_cpu=t5_cpu,
        dit_fsdp=dit_fsdp,
        save_file=save_file if save_file.strip() != "" else None,
        prompt=prompt if prompt.strip() != "" else None,
        n_prompt=n_prompt if n_prompt.strip() != "" else None,
        use_prompt_extend=use_prompt_extend,
        prompt_extend_method=prompt_extend_method,
        prompt_extend_model=prompt_extend_model if prompt_extend_model.strip() != "" else None,
        prompt_extend_target_lang=prompt_extend_target_lang,
        base_seed=int(base_seed),
        image=image if image is not None else None,
        sample_solver=sample_solver,
        sample_steps=int(sample_steps) if sample_steps is not None and str(sample_steps).strip() != "" else None,
        sample_shift=float(sample_shift) if sample_shift is not None and str(sample_shift).strip() != "" else None,
        sample_guide_scale=float(sample_guide_scale),
        resize_mode=resize_mode,
    )
    progress(0.20)
    print("Validating arguments...")
    _validate_args(args)
    progress(0.30)
    print("Arguments validated. Generating output...")
    generate(args)
    progress(1)
    print(f"Generation complete. Output saved to {args.save_file}")
    status = f"Generation complete. Output saved to {args.save_file}"
    if "t2i" in task:
        return None, args.save_file, status
    else:
        return args.save_file, None, status

def update_image_visibility(task):
    # Hide the input image widget if task is a t2v task; show it if task is an i2v task.
    if task in ["t2v-1.3B", "t2v-14B"]:
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def update_output_visibility(task):
    # If task outputs an image (contains "t2i"), hide the video output widget.
    # Otherwise, hide the image output widget.
    if "t2i" in task:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Wan Video GUI</h1>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image (for i2v tasks)", type="pil")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter prompt here", lines=2)
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter negative prompt",
                lines=2,
                value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            )
            with gr.Row(equal_height=True):
                with gr.Column(min_width="200px"):
                    resize_mode = gr.Dropdown(label="Image Resize Mode", choices=["none", "crop", "pad"], value="crop")
                    task = gr.Dropdown(label="Task", choices=["t2v-1.3B", "t2v-14B", "t2i-14B", "i2v-14B"], value="i2v-14B")
                    size = gr.Dropdown(label="Size", choices=list(SIZE_CONFIGS.keys()), value="1280*720")
                    frame_num = gr.Number(label="Frame Num", value=81)
                with gr.Column(min_width="200px"):
                    ckpt_dir = gr.Dropdown(
                        label="Checkpoint Directory",
                        choices=["Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P", "Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"],
                        value="Wan2.1-I2V-14B-480P",
                    )
                    offload_model = gr.Checkbox(label="Offload Model", value=True)
                    save_file = gr.Textbox(label="Save File (optional)", placeholder="Optional save file name", lines=1)
                    base_seed = gr.Number(label="Base Seed", value=-1)
                with gr.Column(min_width="200px"):
                    sample_solver = gr.Dropdown(label="Sample Solver", choices=["unipc", "dpm++"], value="unipc")
                    sample_steps = gr.Number(label="Sample Steps", value=50)
                    sample_shift = gr.Number(label="Sample Shift", value=5.0)
                    sample_guide_scale = gr.Number(label="Sample Guide Scale", value=5.0)
                    ulysses_size = gr.Number(label="Ulysses Size", value=1, visible=False)
                    ring_size = gr.Number(label="Ring Size", value=1, visible=False)
                    t5_fsdp = gr.Checkbox(label="T5 FSDP", value=False, visible=False)
                    t5_cpu = gr.Checkbox(label="T5 CPU", value=False, visible=False)
                    dit_fsdp = gr.Checkbox(label="DiT FSDP", value=False, visible=False)
                    use_prompt_extend = gr.Checkbox(label="Use Prompt Extend", value=False, visible=False)
                    prompt_extend_method = gr.Dropdown(label="Prompt Extend Method", choices=["dashscope", "local_qwen"], value="local_qwen", visible=False)
                    prompt_extend_model = gr.Textbox(label="Prompt Extend Model (optional)", placeholder="Optional prompt extend model", lines=1, visible=False)
                    prompt_extend_target_lang = gr.Dropdown(label="Prompt Extend Target Language", choices=["ch", "en"], value="ch", visible=False)
        with gr.Column():
            output_video = gr.Video(label="Output Video")
            output_image = gr.Image(label="Output Image")
            status = gr.Textbox(label="Status Message")
    # Update input image visibility based on task
    task.change(fn=update_image_visibility, inputs=task, outputs=image_input)
    # Update output widgets visibility based on task
    task.change(fn=update_output_visibility, inputs=task, outputs=[output_video, output_image])
    
    demo_button = gr.Button("Run Wan Pipeline")
    demo_button.click(
        run_wan_pipeline,
        inputs=[
            image_input, prompt_input, negative_prompt, resize_mode, task, size, frame_num, ckpt_dir,
            offload_model, save_file, base_seed, sample_solver, sample_steps,
            sample_shift, sample_guide_scale, ulysses_size, ring_size, t5_fsdp,
            t5_cpu, dit_fsdp, use_prompt_extend, prompt_extend_method,
            prompt_extend_model, prompt_extend_target_lang,
        ],
        outputs=[output_video, output_image, status],
    )

demo.launch(server_name="0.0.0.0")
