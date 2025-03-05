from pathlib import Path
from getAttentionLib import (
    get_response,
    image_symmetric_token_replacement,
    load_pg2_model_and_processor,
    unique_vqa_imgs,
)


if __name__ == "__main__":
    model, processor = load_pg2_model_and_processor(compile=True)

    n_vqa_samples = 40
    rows = list(unique_vqa_imgs(n_vqa_samples + 1))
    for idx, (healthy_row, unhealthy_row) in enumerate(zip(rows[:-1], rows[1:])):
        tgt_dir = f"vqa_patching/{idx:03d}"
        if Path(tgt_dir).exists():
            print(f"Skipping {tgt_dir} because it already exists")
            continue

        healthy_text = f"<image>Answer en {healthy_row['question']}"

        hinputs = processor(
            text=healthy_text, images=healthy_row["image"], return_tensors="pt"
        ).to(model.device)
        houtputs = model.generate(**hinputs, max_new_tokens=1, do_sample=False)
        healthy_tok_str = processor.decode(houtputs[0, -1])
        print(f"{healthy_tok_str=}")

        uinputs = processor(
            text=healthy_text, images=unhealthy_row["image"], return_tensors="pt"
        ).to(model.device)
        uoutputs = model.generate(**uinputs, max_new_tokens=1, do_sample=False)
        unhealthy_tok_str = processor.decode(uoutputs[0, -1])
        print(f"{unhealthy_tok_str=}")

        image_symmetric_token_replacement(
            model=model,
            processor=processor,
            text=healthy_text,
            healthy_img_alias=healthy_row["image_id"],
            healthy_img=healthy_row["image"],
            healthy_tok_str=healthy_tok_str,
            unhealthy_img_alias=unhealthy_row["image_id"],
            unhealthy_img=unhealthy_row["image"],
            unhealthy_tok_str=unhealthy_tok_str,
            tgt_directory=tgt_dir,
        )
