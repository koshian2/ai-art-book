from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch
from torch import FloatTensor
from typing import Any, Callable, Dict, List, Optional, Union
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

# Diffusers==0.16.1
class LatentCouplePipeline(StableDiffusionPipeline):
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, 
                       prompt_embeds: FloatTensor | None = None, negative_prompt_embeds: FloatTensor | None = None):
        negative_prompt = [negative_prompt for i in range(len(prompt))]
        prompt_embeds = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds)
        uncond_prompt, cond_prompt = prompt_embeds.chunk(2) # 1+3
        prompt_embeds = torch.cat([uncond_prompt[:1], cond_prompt], dim=0)
        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        mask: FloatTensor = None, # 追加
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = torch.cat([latents] * prompt_embeds.shape[0]) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                # if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # ここを追加
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text_all = noise_pred[0:1], noise_pred[1:]
                    mask_normed = mask / (torch.sum(mask, dim=0, keepdims=True) + 1e-5)
                    noise_pred_direction = torch.sum((noise_pred_text_all - noise_pred_uncond) * mask_normed, dim=0, keepdim=True)
                    noise_pred = noise_pred_uncond + guidance_scale * noise_pred_direction

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main_latent_couple(width=960, height=512):
    device = "cuda"    
    pipe = LatentCouplePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompts = [
        "two girls standing in a lavender field in the countryside and having fun, best quality, extremely detailed",
        "a girl enjoying the scent of flowers in a lavender field, 1girl, standing, beautiful girl with long blonde hair like a fairy tale princess, blue eyes, white dress, sandals, best quality, extremely detailed",
        "a girl taking a photo in a lavender field, 1girl, standing, healthy girl, wheat-colored skin tan, large eyes, colorful floral shirt, short cut hair, black hair, denim shorts, best quality, extremely detailed"
    ]
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    # 全体、左1/3、右1/3+αでマスクを作る
    all_mask = torch.ones((1, 4, height//8, width//8), dtype=torch.float16)
    left_mask, right_mask = all_mask.clone(), all_mask.clone()
    left_mask[:, :, :, width//24:] = 0 # 横1/3～をオフ
    right_mask[:, :, :, :width//24] = 0 # 左1/3をオフ
    right_mask[:, :, :, width//11:] = 0 # 右1/3+αをオフ
    masks = torch.cat([all_mask, left_mask*1.5, right_mask*1.5], dim=0).to(device)

    # 乱数は1個で初期化
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, mask=masks, 
                 latents=latent, num_inference_steps=50).images[0]
    image.save(f"{CACHE_DIR}/output/17_latent_couple.jpg", quality=92)

if __name__ == "__main__":
    with stub.run():
        main_latent_couple.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/17_latent_couple.jpg output', shell=True)
