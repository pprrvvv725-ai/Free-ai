import torch
from diffusers import AnimateDiffPipeline, ControlNetModel
from diffusers.utils import export_to_video
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16
)

pipe = AnimateDiffPipeline.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

def animate(image_path, pose_dir):
    image = Image.open(image_path).convert("RGB")
    poses = sorted([os.path.join(pose_dir, f) for f in os.listdir(pose_dir)])

    frames = pipe(
        prompt="realistic human, smooth full body motion",
        image=image,
        control_image=[Image.open(p) for p in poses],
        num_frames=len(poses),
        guidance_scale=7.5
    ).frames

    export_to_video(frames, "output.mp4", fps=12)
    print("🎥 Video created")

if __name__ == "__main__":
    animate("image.png", "poses")
