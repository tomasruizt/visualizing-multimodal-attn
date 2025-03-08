from pathlib import Path

import numpy as np
from getAttentionLib import (
    get_response,
    get_vqa_balanced_pairs,
    guassian_noising_activation_patching,
    image_symmetric_token_replacement,
    load_pg2_model_and_processor,
    unique_vqa_imgs,
)


# either "gn" for gaussian noising or "str" for symmetric token replacement
shortname: str = "gn"
assert shortname in ["gn", "str"]

sigma = 0.0045
# xs_std = 3 * sigma  # from compute_img_tokens_embeddings_std(n_vqa_samples=100) = 10 * xs_std
gn_std = 10 * sigma

if __name__ == "__main__":
    model, processor = load_pg2_model_and_processor(compile=True)
    n_img_tokens = 256

    n_vqa_samples = 100
    for idx, (healthy_row, unhealthy_row) in enumerate(
        get_vqa_balanced_pairs(n_vqa_samples)
    ):
        tgt_dir = Path(f"vqa_patching/{shortname}/{idx:03d}")
        if tgt_dir.exists():
            print(f"Skipping {tgt_dir} because it already exists")
            continue

        healthy_text = f"<image>Answer en {healthy_row['question']}"
        print(f"{healthy_text=}")

        healthy_img = healthy_row["image"].convert("RGB")
        hinputs = processor(
            text=healthy_text,
            images=healthy_img,
            return_tensors="pt",
        ).to(model.device)
        houtputs = model.generate(**hinputs, max_new_tokens=1, do_sample=False)
        healthy_tok_str = processor.decode(houtputs[0, -1])
        print(f"{healthy_tok_str=}")

        unhealthy_img = unhealthy_row["image"].convert("RGB")
        uinputs = processor(
            text=healthy_text,
            images=unhealthy_img,
            return_tensors="pt",
        ).to(model.device)
        uoutputs = model.generate(**uinputs, max_new_tokens=1, do_sample=False)
        unhealthy_tok_str = processor.decode(uoutputs[0, -1])
        print(f"{unhealthy_tok_str=}")

        if shortname == "str":
            image_symmetric_token_replacement(
                model=model,
                processor=processor,
                text=healthy_text,
                healthy_img_alias=healthy_row["image_id"],
                healthy_img=healthy_img,
                healthy_tok_str=healthy_tok_str,
                unhealthy_img_alias=unhealthy_row["image_id"],
                unhealthy_img=unhealthy_img,
                unhealthy_tok_str=unhealthy_tok_str,
                tgt_directory=tgt_dir,
            )
        elif shortname == "gn":
            guassian_noising_activation_patching(
                model=model,
                processor=processor,
                text=healthy_text,
                healthy_img_alias=healthy_row["image_id"],
                healthy_img=healthy_img,
                healthy_tok_str=healthy_tok_str,
                tgt_directory=tgt_dir,
                n_img_tokens=n_img_tokens,
                gaussian_noise_std=gn_std,
            )
        else:
            raise ValueError(f"Unknown activation patching type: {shortname}")
