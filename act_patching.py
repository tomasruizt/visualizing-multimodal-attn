from pathlib import Path
from getAttentionLib import (
    ActivationPatchingResult,
    get_decoder_layer_outputs,
    paligemma_merge_text_and_image,
    patch_all_activations,
)
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from PIL import Image

torch.set_grad_enabled(False)  # avoid blowing up mem
device = "cuda"

model_id = "google/paligemma2-3b-pt-224"
model = (
    PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    .to(device)
    .eval()
)
processor = PaliGemmaProcessor.from_pretrained(model_id)

text = "<image>Answer en what is the frisbee's color?"
n_img_tokens = 256

f2_img_path = "imgs/frisbee2.png"
f2_image = Image.open(f2_img_path)
f2_inputs = processor(text=text, images=f2_image, return_tensors="pt").to(model.device)
f2_embeds = paligemma_merge_text_and_image(model, f2_inputs)
f2_activations, _ = get_decoder_layer_outputs(model, f2_embeds)

purple_token = processor.tokenizer.encode("purple")
assert processor.tokenizer.decode(purple_token) == "purple"

f1_img_path = "imgs/frisbee.jpg"
f1_image = Image.open(f1_img_path)
f1_inputs = processor(text=text, images=f1_image, return_tensors="pt").to(model.device)
f1_embeds = paligemma_merge_text_and_image(model, f1_inputs)

blue_token = processor.tokenizer.encode("blue")
assert processor.tokenizer.decode(blue_token) == "blue"

patching_result: ActivationPatchingResult = patch_all_activations(
    model=model,
    healthy_activations=f2_activations,
    unhealthy_embeds=f1_embeds,
    healthy_response_tok_idx=purple_token,
    unhealthy_response_tok_idx=blue_token,
)

patching_result.save(
    directory=Path("rev_frisbee_pr"),
    health_response_tok="blue",
    unhealthy_response_tok="purple",
    corruption_type="symmetric_token_replacement",
    corruption_img_alias=f1_img_path,
    healthy_img_alias=f2_img_path,
    prompt=text,
)
