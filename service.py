from __future__ import annotations
import typing as t

import bentoml
from PIL.Image import Image

SD2_MODEL_ID = 'stabilityai/stable-diffusion-2'
SD2_UPSCALER_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

DEFAULT_SIZE = 512

@bentoml.service(
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
        # we can also specify GPU memory requirement:
        # "memory": "16Gi",
    },
    traffic={"timeout": 600},
)
class SD2Upscaler:

    def __init__(self) -> None:
        import torch
        import diffusers

        # Load model into pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
            SD2_UPSCALER_MODEL_ID, use_safetensors=True
        )
        self.pipe.to(self.device)

    @bentoml.api
    def upscale(self, image: Image, prompt: str, negative_prompt: t.Optional[str] = None) -> Image:
        image = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )[0][0]
        image.format = "png"
        return image


@bentoml.service(
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
        # we can also specify GPU memory requirement:
        # "memory": "16Gi",
    },
    traffic={"timeout": 600},
)
class StableDiffusion2:
    upscaler_service: SD2Upscaler = bentoml.depends(SD2Upscaler)

    def __init__(self) -> None:
        import torch
        import diffusers

        # Load model into pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.txt2img_pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            SD2_MODEL_ID, use_safetensors=True
        )
        self.img2img_pipe = diffusers.StableDiffusionImg2ImgPipeline(
            **self.txt2img_pipe.components
        )
        self.txt2img_pipe.to(self.device)
        self.img2img_pipe.to(self.device)

    @bentoml.api
    def txt2img(
            self,
            prompt: str = "photo a majestic sunrise in the mountains, best quality, 4k",
            negative_prompt: t.Optional[str] = None,
            height: int = DEFAULT_SIZE,
            width: int = DEFAULT_SIZE,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            upscale: bool = True,
    ) -> Image:

        res = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        image = res[0][0]
        if upscale:
            low_res_img = image
            low_res_img.format = "png"
            image = self.upscaler_service.upscale(
                image=low_res_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
        return image

    # @bentoml.api
    # def img2img(self, image: Image, input_data: t.Dict[str, t.Any] = sample_img2img_input) -> Image:
    #     upscale = input_data.pop("upscale")
    #     input_data["image"] = image
    #     res = self.img2img_pipe(**input_data)
    #     images = res[0]
    #     if upscale:
    #         prompt = input_data["prompt"]
    #         negative_prompt = input_data.get("negative_prompt")
    #         low_res_img = images[0]
    #         res = self.upscaler_model_pipeline(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             image=low_res_img
    #         )
    #         images = res[0]
    #     return images[0]
