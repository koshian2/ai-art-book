def change_cloth(width=512, height=960):
    device = "cuda:1"
    models = [
        "NoCrypt/SomethingV2_2",
        "H:/diffusion_models/diffusers/merge_Counterfeit-V3.0_orangemix"
    ]
    model_names = ["SomethingV2_2", "merge"]
    
    fig = plt.figure(figsize=(18, 14))
    cloths = ["", "yukata", "basketball uniform", "track and field uniform", "bikini"]
    weights = [1.0, 1.1, 1.8, 1.55, 1.95]
    for i, model in enumerate(models):
        for j, (cloth, weight) in enumerate(zip(cloths, weights)):
            pipe = DiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16,
                custom_pipeline="lpw_stable_diffusion")
            pipe = load_safetensors_lora(pipe, "H:/diffusion_models/lora/lumine1-000008.safetensors", alpha=0.3)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_vae_tiling()
            if i != 0:
                pipe.safety_checker = lambda images, **kwargs: (images, False)
            pipe.to(device)

            prompt = "luminedef, luminernd, "
            prompt += f"({cloth}:{weight}), " if cloth != "" else ""
            prompt += "1girl, look at viewer, best quality, extremely detailed"
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
            generator = torch.Generator(device).manual_seed(1234)
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, max_embeddings_multiples=3,
                        generator=generator, width=width, height=height, num_inference_steps=30).images[0]
            
            ax = fig.add_subplot(2, 5, 5*i+j+1)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{model_names[i]} {cloth if cloth != '' else 'None'}")

    plt.show()