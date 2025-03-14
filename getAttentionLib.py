from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
from PIL import Image
import PIL.Image
from tqdm import tqdm
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3ForConditionalGeneration,
)

from typing import Any, Iterable, Literal

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from circuitsvis.attention import attention_heads
from datasets import load_dataset
from torch import nn
from transformers import (
    PaliGemmaForConditionalGeneration,
    BatchFeature,
    PaliGemmaProcessor,
)
import plotly.graph_objects as go
import pandas as pd


@dataclass
class SaveInputOutputHook:
    module: nn.Module | None = None
    input: Any | None = None
    output: Any | None = None

    def hook(self, module, input, output):
        assert self.module is None
        assert self.input is None
        assert self.output is None
        self.module = module
        self.input = input
        self.output = output


@dataclass
class RestoreActivationHook:
    """In the forward pass, restores the activations of a given layer and token idx"""

    healthy_activations: torch.Tensor
    layer_idx: int
    token_idx: int

    def hook(self, module, input, output):
        if len(output) == 2:
            out, cache = output
        else:
            out = output[0]
        desired = self.healthy_activations[self.layer_idx, self.token_idx]
        out[0, self.token_idx, :] = desired
        if len(output) == 2:
            return out, cache
        else:
            return (out,)


def get_img_grid_sizes(model, inputs):
    if isinstance(model, Idefics3ForConditionalGeneration):
        vision_outputs = model.model.vision_model(
            inputs.pixel_values[0].to(dtype=torch.bfloat16)
        )
        n_img_tokens = vision_outputs.last_hidden_state.shape[1]
        # IDEFICS uses 14x14 patches for 224x224 images
        grid_side_len = int(np.sqrt(n_img_tokens))
        return n_img_tokens, grid_side_len

    else:
        batch_size_, n_img_tokens, hidden_dim_ = model.get_image_features(
            inputs.pixel_values
        ).shape
    grid_side_len = n_img_tokens**0.5
    # print(grid_side_len)
    assert grid_side_len.is_integer()
    grid_side_len = int(grid_side_len)
    return n_img_tokens, grid_side_len


def dump_attn(
    attn_weights: torch.Tensor,
    layer_idx: int,
    name: str,
    tokens: list[str],
    img_path: str,
    grid_side_len: int,
) -> None:
    """
    attn_weights: Torch tensor of shape [1, n_heads, n_tokens, n_tokens]
    name: Prefix for the html file name
    tokens: All tokens in string format, len=n_tokens
    """
    assert len(attn_weights.shape) == 4
    assert attn_weights.shape[-1] == len(tokens)
    assert attn_weights.shape[-2] == len(tokens)

    viz_data = dict(
        attention=attn_weights[0].round(decimals=6).cpu().tolist(),
        tokens=tokens,
        image=img_path,
        image_grid_dims=[grid_side_len, grid_side_len],
        image_tokens_start=0,
        max_value=attn_weights.max().item() / 100,  # deprecate
        min_value=attn_weights.min().item(),  # deprecate
    )

    html = attention_heads(**viz_data)
    html_str = str(html)
    print("Rendered HTML")
    tgt_file = f"{name}_layer_{layer_idx}_attention_heads.html"
    with open(tgt_file, "w") as f:
        f.write(html_str)
    print(f"Wrote to file='{tgt_file}'")


def get_response(
    model, processor, text: str, image: PIL.Image.Image, max_new_tokens: int = 100
) -> tuple[list[str], str]:
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    inputs_tokens = [processor.decode(id) for id in inputs.input_ids[0]]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response: str = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return inputs_tokens, response


def compute_attn_sums(attns: torch.Tensor, n_img_tokens: int) -> torch.Tensor:
    """attns is a tensor of shape [1, n_heads, n_tokens, n_tokens]"""
    if len(attns.shape) == 4:
        # remove batch dimension
        assert attns.shape[0] == 1, f"attns.shape: {attns.shape}"
        attns = attns[0]
    assert len(attns.shape) == 3, f"attns.shape: {attns.shape}"
    # print("full attns.shape:", attns.shape)

    bos_token = n_img_tokens
    i2i_attn = attns[:, :n_img_tokens, :n_img_tokens]  # images tokens
    i2b_attn = attns[:, :n_img_tokens, bos_token : bos_token + 1]  # bos token
    i2t_attn = attns[:, :n_img_tokens, bos_token + 1 : -1]  # text tokens
    i2f_attn = attns[:, :n_img_tokens, -1:]  # final token

    b2i_attn = attns[:, bos_token : bos_token + 1, :n_img_tokens]
    b2b_attn = attns[:, bos_token : bos_token + 1, bos_token : bos_token + 1]
    b2t_attn = attns[:, bos_token : bos_token + 1, bos_token + 1 : -1]
    b2f_attn = attns[:, bos_token : bos_token + 1, -1:]

    t2i_attn = attns[:, bos_token + 1 :, :n_img_tokens]
    t2b_attn = attns[:, bos_token + 1 :, bos_token : bos_token + 1]
    t2t_attn = attns[:, bos_token + 1 :, bos_token + 1 : -1]
    t2f_attn = attns[:, bos_token + 1 :, -1:]

    f2i_attn = attns[:, -1:, :n_img_tokens]
    f2b_attn = attns[:, -1:, bos_token : bos_token + 1]
    f2t_attn = attns[:, -1:, bos_token + 1 : -1]
    f2f_attn = attns[:, -1:, -1:]

    attn_sums = torch.tensor(
        [
            [
                i2i_attn.sum(dim=2).mean(),
                i2b_attn.sum(dim=2).mean(),
                i2t_attn.sum(dim=2).mean(),
                i2f_attn.sum(dim=2).mean(),
            ],
            [
                b2i_attn.sum(dim=2).mean(),
                b2b_attn.sum(dim=2).mean(),
                b2t_attn.sum(dim=2).mean(),
                b2f_attn.sum(dim=2).mean(),
            ],
            [
                t2i_attn.sum(dim=2).mean(),
                t2b_attn.sum(dim=2).mean(),
                t2t_attn.sum(dim=2).mean(),
                t2f_attn.sum(dim=2).mean(),
            ],
            [
                f2i_attn.sum(dim=2).mean(),
                f2b_attn.sum(dim=2).mean(),
                f2t_attn.sum(dim=2).mean(),
                f2f_attn.sum(dim=2).mean(),
            ],
        ]
    )
    return attn_sums


def plot_attn_sums(
    attn_sums: torch.Tensor,
    show_colorbar: bool = True,
    title: str = "",
    show_ylabel: bool = True,
    show_yticks: bool = True,
    stds: torch.Tensor | None = None,
    color_threshold: float = 0.5,
    cmap: str = "Blues",
    **imshow_kwargs,
):
    plt.imshow(attn_sums, cmap=cmap, **imshow_kwargs)
    if show_colorbar:
        plt.colorbar()
    names = ["img tokens", "<bos> token", "text tokens", "final token"]
    shortnames = ["img \ntokens", "<bos>\ntoken", "text\ntokens", "final\ntoken"]
    plt.xticks(ticks=[0, 1, 2, 3], labels=shortnames)
    plt.xlabel("Source token(s)")
    if not show_yticks:
        names = [""] * len(names)
    plt.yticks(ticks=[0, 1, 2, 3], labels=names)
    if show_ylabel:
        plt.ylabel("Destination token(s)")
    if title != "":
        plt.title(title)
    for (i, j), val in np.ndenumerate(attn_sums):
        if stds is None:
            text = f"{val:.3f}"
        else:
            text = f"{val:.2f}\n± {stds[i, j]:.2f}"
        plt.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            color="black" if val < color_threshold else "white",
        )
    plt.tight_layout()
    return plt.gcf()


def compute_mult_attn_sums(
    model, model_kwargs, layers: list[int], n_img_tokens: int
) -> torch.Tensor:
    attns = torch.stack(model(**model_kwargs, output_attentions=True).attentions)
    mult_attn_sums = torch.stack(
        [compute_attn_sums(attns[l], n_img_tokens) for l in layers]
    )
    mult_attn_sums = mult_attn_sums.float()
    return mult_attn_sums


def plot_mult_attn_sums(
    model,
    model_kwargs,
    layers: list[int],
    mult_attn_sums=None,
    stds=None,
    n_img_tokens=None,
    figsize=(8, 4),
    **kwargs,
) -> plt.Figure:
    if mult_attn_sums is None:
        mult_attn_sums = compute_mult_attn_sums(
            model, model_kwargs, layers, n_img_tokens
        )

    fig = plt.figure(figsize=figsize)
    for i, attn_sums in enumerate(mult_attn_sums):
        is_first = i == 0
        plt.subplot(1, len(layers), i + 1)
        fig = plot_attn_sums(
            attn_sums,
            show_colorbar=False,
            show_ylabel=is_first,
            show_yticks=is_first,
            title=f"Layer {layers[i]}",
            stds=stds[i] if stds is not None else None,
            **kwargs,
        )
    fig.tight_layout()
    return fig


def load_vqa_ds(split: str | None = None):
    avoid_timeout = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}}
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=split,
        storage_options=avoid_timeout,
        trust_remote_code=True,
    )
    return ds


def load_vqa_samples() -> list[dict[str, Any]]:
    root = Path("./vqa-samples")
    assert root.exists()
    samples = []
    for i, dir_path in enumerate(root.glob("*")):
        data = {}
        data["image"] = PIL.Image.open(dir_path / "image.png")
        data["question"] = Path(dir_path / "question.txt").read_text()
        data["answer"] = Path(dir_path / "answer.txt").read_text()
        data["image_id"] = str(i)
        samples.append(data)
    return samples


def plot_images_grid(
    images: list[PIL.Image.Image],
    texts: list[str],
    nrows: int,
    ncols: int,
    figsize: tuple[int, int] = (15, 15),
) -> plt.Figure:
    """
    Plot a grid of images with text below each image.

    Args:
        images: List of PIL images
        texts: List of strings to display below each image
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object
    """
    assert len(images) == len(texts), "Number of images must match number of texts"
    assert len(images) <= nrows * ncols, "Grid size too small for number of images"

    # Create figure with extra vertical space for text
    fig = plt.figure(figsize=figsize)

    for idx, (img, text) in enumerate(zip(images, texts)):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.imshow(img)
        ax.axis("off")

        ax.set_title(
            text,
            wrap=True,
            pad=10,  # Add padding between image and text
            fontsize="small",  # Reduce font size
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
        )

    # Adjust layout to prevent text overlap
    plt.tight_layout(h_pad=1.5, w_pad=1.5)  # Increase spacing between subplots
    return fig


def plot_region_attn_progression(model, inputs, mult_attn_sums=None):
    if mult_attn_sums is None:
        mult_attn_sums = compute_mult_attn_sums(
            model, inputs, layers=list(range(len(model.language_model.model.layers)))
        )

    names = ["img tokens", "<bos> token", "text tokens"]
    fig = plt.figure(figsize=(12, 4))
    ylims = (-0.1, 1.1)

    titles = ["Dest: img tokens", "Dest: <bos> token", "Dest: text tokens"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.stackplot(
            range(len(mult_attn_sums)),
            mult_attn_sums[:, i, :].T,
            labels=names,
        )
        plt.title(titles[i])
        plt.ylim(ylims)
        if i == 1:
            plt.legend()
        plt.grid()
        if i == 0:
            plt.ylabel("Attention fraction")
        plt.xlabel("Layer")

    plt.tight_layout()
    return fig


def paligemma_merge_text_and_image(
    self: PaliGemmaForConditionalGeneration, inputs: BatchFeature
) -> torch.Tensor:
    """Return input_embeds with image features in the image token positions."""

    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values

    inputs_embeds = self.get_input_embeddings()(input_ids)
    image_features = self.get_image_features(pixel_values)

    special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
        inputs_embeds.device
    )
    if inputs_embeds[special_image_mask].numel() != image_features.numel():
        image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
        raise ValueError(
            f"Number of images does not match number of special image tokens in the input text. "
            f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
            "tokens from image embeddings."
        )
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    return inputs_embeds


def gaussian_noising(
    input_embeds: torch.Tensor, num_img_tokens: int, std: float
) -> torch.Tensor:
    """Introduce noise in the image tokens. They are assumed to be at the begginning."""
    new_embeds = input_embeds.clone()
    noise = torch.randn_like(new_embeds[:, :num_img_tokens, :]) * std
    new_embeds[:, :num_img_tokens, :] += noise
    assert new_embeds.shape == input_embeds.shape
    return new_embeds


def compute_img_tokens_embeddings_std(
    model, processor, n_vqa_samples: int, num_img_tokens: int
):
    embeds = []
    for row in unique_vqa_imgs(n_vqa_samples=n_vqa_samples):
        text = f"<image>Answer en {row['question']}"
        inputs = processor(
            text=text,
            images=row["image"].convert("RGB"),
            return_tensors="pt",
        ).to(model.device)
        inputs_embeds = paligemma_merge_text_and_image(model, inputs)
        embeds.append(inputs_embeds[:, :num_img_tokens, :])
    embeds = torch.cat(embeds)
    assert embeds.shape == (
        n_vqa_samples,
        num_img_tokens,
        model.config.projection_dim,  # 2304
    )
    return embeds.std().item()


def get_decoder_layer_outputs(model, inputs_embeds):
    """
    Cannot use outputs.hidden_states because they already have a normalization applied.
    After restoring the activations, the model would normalize them again (this is not an idempotent operation).
    """
    hooks, outputs = run_with_hooks(model, inputs_embeds)
    activations = torch.stack([h.output[0][0] for h in hooks])
    return activations, outputs


def run_with_hooks(
    model, inputs_embeds, ctor=SaveInputOutputHook, modules=None
) -> tuple[list, Any]:
    handles = []
    hooks = []

    if modules is None:
        modules = model.language_model.model.layers

    for module in modules:
        hook = ctor()
        hooks.append(hook)
        handle = module.register_forward_hook(hook.hook)
        handles.append(handle)

    try:
        outputs = model(inputs_embeds=inputs_embeds)
    finally:
        for handle in handles:
            handle.remove()
    return hooks, outputs


def forward_with_patched_activation(
    model, unhealthy_embeds, hook: RestoreActivationHook
):
    layer = model.language_model.model.layers[hook.layer_idx]
    handle = layer.register_forward_hook(hook.hook)
    try:
        outputs = model(inputs_embeds=unhealthy_embeds)
    finally:
        handle.remove()
    return outputs


def tok_prob(outputs, token_idx: int):
    logits = outputs.logits[0, -1, :]
    return logits.softmax(dim=-1)[token_idx]


@dataclass
class ActivationPatchingResult:
    healthy_tok_response_probs: torch.Tensor
    healthy_tok_response_logits: torch.Tensor
    unhealthy_tok_response_probs: torch.Tensor
    unhealthy_tok_response_logits: torch.Tensor
    metadata: dict[str, Any] | None = None

    def get_logit_diff_str(self) -> torch.Tensor:
        """logit diffs for symmetric token replacement"""
        unhealthy_logit_diff = (
            self.metadata["unhealthy_run_healthy_tok_logit"]
            - self.metadata["unhealthy_run_unhealthy_tok_logit"]
        )
        patched_logit_diff = (
            self.healthy_tok_response_logits - self.unhealthy_tok_response_logits
        )
        denominator = abs(self.logit_diff_denominator_str())
        if denominator == 0.0:
            raise ZeroDivisionError(
                "Denominator is 0. Healthy and unhealthy logit diffs are same"
            )

        normalized = (patched_logit_diff - unhealthy_logit_diff) / denominator
        return normalized

    def logit_diff_denominator_str(self) -> float:
        """without abs()"""
        unhealthy_logit_diff = (
            self.metadata["unhealthy_run_healthy_tok_logit"]
            - self.metadata["unhealthy_run_unhealthy_tok_logit"]
        )
        healthy_logit_diff = (
            self.metadata["healthy_run_healthy_tok_logit"]
            - self.metadata["healthy_run_unhealthy_tok_logit"]
        )
        return healthy_logit_diff - unhealthy_logit_diff

    def get_logit_diff_gn(self) -> torch.Tensor:
        """logit diffs for gaussian noising"""
        nomin = (
            self.healthy_tok_response_logits
            - self.metadata["unhealthy_run_healthy_tok_logit"]
        )
        denom = abs(self.logit_diff_denominator_gn())
        if denom == 0.0:
            raise ZeroDivisionError(
                "Denominator is 0. Healthy and unhealthy logit diffs are same"
            )
        return nomin / denom

    def logit_diff_denominator_gn(self) -> float:
        """without abs()"""
        return (
            self.metadata["healthy_run_healthy_tok_logit"]
            - self.metadata["unhealthy_run_healthy_tok_logit"]
        )

    def save(
        self,
        directory: Path | str,
        health_response_tok: str,
        unhealthy_response_tok: str,
        corruption_type: Literal["gaussian_noising", "symmetric_token_replacement"],
        healthy_img_alias: str,
        prompt: str,
        token_strings: list[str],
        unhealthy_run_unhealthy_tok_logit: float,
        unhealthy_run_healthy_tok_logit: float,
        corruption_img_alias: str | None = None,
        healthy_run_unhealthy_tok_logit: float | None = None,
        healthy_run_healthy_tok_logit: float | None = None,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "healthy_response_tok": health_response_tok,
            "unhealthy_response_tok": unhealthy_response_tok,
            "corruption_type": corruption_type,
            "healthy_img_alias": healthy_img_alias,
            "prompt": prompt,
            "corruption_img_alias": corruption_img_alias,
            "unhealthy_run_unhealthy_tok_logit": unhealthy_run_unhealthy_tok_logit,
            "unhealthy_run_healthy_tok_logit": unhealthy_run_healthy_tok_logit,
            "token_strings": token_strings,
            "healthy_run_unhealthy_tok_logit": healthy_run_unhealthy_tok_logit,
            "healthy_run_healthy_tok_logit": healthy_run_healthy_tok_logit,
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        hprobs = directory / "healthy_tok_response_probs.pt"
        hlogits = directory / "healthy_tok_response_logits.pt"
        uprobs = directory / "unhealthy_tok_response_probs.pt"
        ulogits = directory / "unhealthy_tok_response_logits.pt"
        torch.save(self.healthy_tok_response_probs, hprobs)
        torch.save(self.healthy_tok_response_logits, hlogits)
        torch.save(self.unhealthy_tok_response_probs, uprobs)
        torch.save(self.unhealthy_tok_response_logits, ulogits)
        print(f"Saved to {directory}")

    @classmethod
    def load(cls, directory: Path | str):
        directory = Path(directory)
        hprobs = torch.load(directory / "healthy_tok_response_probs.pt")
        hlogits = torch.load(directory / "healthy_tok_response_logits.pt")
        uprobs = torch.load(directory / "unhealthy_tok_response_probs.pt")
        ulogits = torch.load(directory / "unhealthy_tok_response_logits.pt")
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        return cls(
            healthy_tok_response_probs=hprobs,
            healthy_tok_response_logits=hlogits,
            unhealthy_tok_response_probs=uprobs,
            unhealthy_tok_response_logits=ulogits,
            metadata=metadata,
        )


def patch_all_activations(
    model,
    healthy_activations,
    unhealthy_embeds,
    healthy_response_tok_idx: int,
    unhealthy_response_tok_idx: int,
):
    n_layers = len(model.language_model.model.layers)
    n_tokens = healthy_activations.shape[1]

    healthy_tok_response_probs = torch.zeros(n_layers, n_tokens)
    healthy_tok_response_logits = torch.zeros(n_layers, n_tokens)
    unhealthy_tok_response_probs = torch.zeros(n_layers, n_tokens)
    unhealthy_tok_response_logits = torch.zeros(n_layers, n_tokens)

    progress_bar = tqdm(total=n_layers * n_tokens)

    for layer_idx in range(n_layers):
        for token_idx in range(n_tokens):
            restore_hook = RestoreActivationHook(
                healthy_activations, layer_idx=layer_idx, token_idx=token_idx
            )
            outputs = forward_with_patched_activation(
                model, unhealthy_embeds, restore_hook
            )

            logits = outputs.logits[0, -1, :]
            hlogits = logits[healthy_response_tok_idx]
            healthy_tok_response_logits[layer_idx, token_idx] = hlogits
            ulogits = logits[unhealthy_response_tok_idx]
            unhealthy_tok_response_logits[layer_idx, token_idx] = ulogits

            probs = logits.softmax(dim=-1)
            hprob = probs[healthy_response_tok_idx]
            healthy_tok_response_probs[layer_idx, token_idx] = hprob
            uprob = probs[unhealthy_response_tok_idx]
            unhealthy_tok_response_probs[layer_idx, token_idx] = uprob

            progress_bar.update(1)

    result = ActivationPatchingResult(
        healthy_tok_response_probs=healthy_tok_response_probs,
        healthy_tok_response_logits=healthy_tok_response_logits,
        unhealthy_tok_response_probs=unhealthy_tok_response_probs,
        unhealthy_tok_response_logits=unhealthy_tok_response_logits,
    )
    return result


def plot_pooled_probs_plotly(
    pooled_probs: torch.Tensor, inputs_tokens: list[str], healthy_response_tok_name: str
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pooled_probs.T,
            colorscale="Blues",
            colorbar=dict(title=f"Prob ({healthy_response_tok_name})"),
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="layer",
            tickmode="array",
            tickvals=list(range(len(pooled_probs.T))),
            ticktext=["img_tokens"] + inputs_tokens[256:],
        ),
        xaxis=dict(title="token"),
        width=600,
        height=400,
    )

    return fig


def plot_pooled_probs_plt(
    pooled_probs: torch.Tensor,
    inputs_tokens: list[str],
    cbar_label: str,
    n_img_tokens: int,
    cmax=None,
    cmin=None,
    cmap="Blues",
    ax=None,
    show_ylabel=True,
    figsize=(6, 3),
    fraction=0.03,
):
    if cmax is None:
        cmax = pooled_probs.max().item()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pooled_probs.T, cmap=cmap)
    if show_ylabel:
        ax.set_ylabel("Token")
    ax.set_xlabel("Layer")
    ax.set_xticks(
        ticks=range(len(pooled_probs))[::2],
        labels=list(range(1, len(pooled_probs) + 1))[::2],
    )
    ax.set_yticks(
        ticks=range(len(pooled_probs.T)),
        labels=["img_tokens"] + inputs_tokens[n_img_tokens:],
    )
    cbar = plt.colorbar(im, ax=ax, fraction=fraction)
    cbar.set_label(cbar_label)
    cbar.mappable.set_clim(cmin, cmax)
    plt.tight_layout()
    return ax.figure


import ipywidgets as widgets
from IPython.display import display, clear_output


def plot_and_browse_img_token_in_probs(probs: torch.Tensor, cmax=0.7):
    assert len(probs.shape) == 2
    # Create a slider widget
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=probs.shape[0] - 1,  # num_layers
        step=1,
        description="Layer:",
        continuous_update=False,
    )

    # Function to update the plot based on the slider value
    def update_plot(change):
        layer = change["new"]
        clear_output(wait=True)
        display(slider)
        layer_probs = probs[layer, :256].reshape(16, 16)
        plot_img_probs(layer_probs, title=f"Layer {layer}", cmax=cmax)
        plt.show()

    # Attach the update function to the slider
    slider.observe(update_plot, names="value")

    # Initial plot
    update_plot({"new": 0})


def plot_img_probs(
    probs: torch.Tensor,
    title: str,
    cmax=None,
    cmin=None,
    cmap="Blues",
    img=None,
    ax=None,
):
    h, w = probs.shape
    if ax is None:
        _, ax = plt.subplots()

    # Display image if provided
    if img is not None:
        ax.imshow(img, alpha=1.0)
        img_height, img_width = (
            img.shape[:2] if len(img.shape) >= 2 else (img.shape[0], 1)
        )
        extent = [0, img_width, img_height, 0]
        im = ax.imshow(probs, cmap=cmap, alpha=0.8, extent=extent)

        # Set ticks (1 to 16)
        ax.set_xticks(np.linspace(0, img_width, w + 1)[:-1])
        ax.set_yticks(np.linspace(0, img_height, h + 1)[:-1])
    else:
        im = ax.imshow(probs, cmap=cmap)
        ax.set_xticks(range(0, w, 2))
        ax.set_yticks(range(0, h, 2))

    # Set tick labels (1 to h/w)
    ax.set_xticklabels(range(1, w + 1, 2))
    ax.set_yticklabels(range(1, h + 1, 2))

    # Add minor ticks for grid
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    # Hide minor ticks for cleaner appearance
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add colorbar and title
    cbar = plt.colorbar(im)
    cbar.mappable.set_clim(cmin, cmax)
    ax.set_title(title)

    ax.set_ylabel("Image patch")
    ax.set_xlabel("Image patch")


def maxpool_img_tokens(probs: torch.Tensor, n_img_tokens: int) -> torch.Tensor:
    imgs_max_pool = probs[:, :n_img_tokens].max(dim=1)[0][:, None]
    pooled = torch.hstack([imgs_max_pool, probs[:, n_img_tokens:]])
    return pooled


def avgpool_img_tokens(probs: torch.Tensor) -> torch.Tensor:
    imgs_avg_pool = probs[:, :256].mean(dim=1)[:, None]
    pooled = torch.hstack([imgs_avg_pool, probs[:, 256:]])
    return pooled


def maxabspool_img_tokens(probs: torch.Tensor, n_img_tokens: int) -> torch.Tensor:
    imgs_max_pool = maxabs_reduction(probs[:, :n_img_tokens], dim=1)[:, None]
    pooled = torch.hstack([imgs_max_pool, probs[:, n_img_tokens:]])
    return pooled


def maxabs_reduction(xs: torch.Tensor, dim: int) -> torch.Tensor:
    """Returns the max abs values in dim, regardless of sign"""
    _, idxs = xs.abs().max(dim=dim, keepdim=True)
    return xs.gather(dim=dim, index=idxs).squeeze(dim=dim)


def plot_fx_norms_progressions(
    max_norms: torch.Tensor, avg_norms: torch.Tensor, sharey=False
):
    """
    Inputs are the outputs of aggregate_layer_norms()
    Plot the norm progression for each token type over the layers
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=sharey, figsize=(7, 3))

    ax1.plot(max_norms.cpu(), marker="x", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Norm Value")
    ax1.set_title("Max Norms")

    ax2.plot(avg_norms.cpu(), marker="x", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_title("Average Norms")
    ax2.legend(["Image Tokens", "BOS Token", "Text Tokens", "Final Token"])

    ax1.grid(True)
    ax2.grid(True)
    plt.tight_layout()

    return fig


def aggregate_layer_norms(
    all_fx_norms: torch.Tensor,
    n_img_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_norms = []
    for layer_idx in range(all_fx_norms.shape[0]):
        across_heads = all_fx_norms[layer_idx].max(dim=0).values
        of_img_toks = across_heads[:n_img_tokens].max()
        of_bos_tok = across_heads[n_img_tokens]
        of_text_toks = across_heads[n_img_tokens + 1 : -1].max()
        of_final_tok = across_heads[-1]
        layer_avg_norms = torch.stack(
            [of_img_toks, of_bos_tok, of_text_toks, of_final_tok]
        )
        max_norms.append(layer_avg_norms)
    max_norms = torch.stack(max_norms)

    avg_norms = []
    for layer_idx in range(all_fx_norms.shape[0]):
        across_heads = all_fx_norms[layer_idx].mean(dim=0)
        of_img_toks = across_heads[:n_img_tokens].mean()
        of_bos_tok = across_heads[n_img_tokens]
        of_text_toks = across_heads[n_img_tokens + 1 : -1].mean()
        of_final_tok = across_heads[-1]
        layer_avg_norms = torch.stack(
            [of_img_toks, of_bos_tok, of_text_toks, of_final_tok]
        )
        avg_norms.append(layer_avg_norms)
    avg_norms = torch.stack(avg_norms)

    return max_norms, avg_norms


VQARow = dict[str, Any]


def unique_vqa_imgs(n_vqa_samples: int) -> Iterable[VQARow]:
    ds = load_vqa_ds(split="train")
    seen_imgs = set()
    progress_bar = tqdm(total=n_vqa_samples)

    for row in ds:
        if row["image_id"] not in seen_imgs:
            seen_imgs.add(row["image_id"])
            yield row
            progress_bar.update(1)

        if len(seen_imgs) >= n_vqa_samples:
            progress_bar.close()
            return

    progress_bar.close()


def get_vqa_balanced_pairs(n_vqa_samples: int) -> Iterable[tuple[VQARow, VQARow]]:
    dataset = load_vqa_ds(split="train")

    # wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
    file = (
        Path(__file__).parent
        / "vqa_dataset/v2_mscoco_train2014_complementary_pairs.json"
    )
    with open(file, "r") as f:
        pairs = json.load(f)

    qids = pd.Series(dataset["question_id"])
    n_yielded = 0
    for pair in pairs:
        indices = [idx for qid in pair for idx in qids[qids == qid].index]
        assert len(indices) == 2

        rows = dataset.select(indices)
        row1 = rows[0]
        row2 = rows[1]

        if row1["multiple_choice_answer"] == row2["multiple_choice_answer"]:
            print(f"Skipping pair {pair} because the dataset answers are the same")
            continue

        yield row1, row2
        n_yielded += 1

        if n_yielded >= n_vqa_samples:
            return


def compute_mult_attn_sums_over_vqa(
    model,
    processor,
    n_vqa_samples: int,
    layers: list[int],
    n_img_tokens: int,
    get_responses: bool = True,
) -> torch.Tensor:
    attens_tensor = []
    responses = []
    imgs = []

    for row in unique_vqa_imgs(n_vqa_samples):
        text = f"<image>Answer en {row['question']}"
        img = row["image"].convert("RGB")
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)

        if get_responses:
            response = get_response(model, processor, text, img)[1]
            responses.append(response)

        imgs.append(img)

        mult_attn_sums = compute_mult_attn_sums(
            model, inputs, layers=layers, n_img_tokens=n_img_tokens
        )
        attens_tensor.append(mult_attn_sums)

        if len(imgs) >= n_vqa_samples:
            break

    stacked_attens = torch.stack(attens_tensor)
    assert stacked_attens.shape == (n_vqa_samples, len(layers), 4, 4)
    return stacked_attens, imgs, responses


def compute_mult_attn_sums_over_noisy_vqa(
    model,
    processor,
    n_vqa_samples: int,
    layers: list[int],
    n_img_tokens: int,
    std: float,
):
    stacked_attens = []
    for row in unique_vqa_imgs(n_vqa_samples=n_vqa_samples):
        text = f"<image>Answer en {row['question']}"
        inputs = processor(
            text=text,
            images=row["image"].convert("RGB"),
            return_tensors="pt",
        ).to(model.device)

        inputs_embeds = paligemma_merge_text_and_image(model, inputs)
        gn_inputs_embeds = gaussian_noising(
            inputs_embeds, num_img_tokens=n_img_tokens, std=std
        )
        mult_attn_sums = compute_mult_attn_sums(
            model,
            {"inputs_embeds": gn_inputs_embeds},
            layers=layers,
            n_img_tokens=n_img_tokens,
        )
        stacked_attens.append(mult_attn_sums)
    stacked_attens = torch.stack(stacked_attens)
    return stacked_attens


def plot_img_and_text_probs_side_by_side(
    probs: torch.Tensor,
    n_img_tokens: int,
    token_strings: list[str],
    token_str: str,
    cmax=None,
    is_probabilities: bool = True,
    reduction: str = "mean",  # or "absmax"
    cmin_cmax: tuple[float, float] | None = None,
):
    if reduction == "absmax":
        pooled_probs = maxabspool_img_tokens(probs, n_img_tokens=n_img_tokens)
        probs_by_img = maxabs_reduction(probs[:, :n_img_tokens], dim=0).reshape(16, 16)
    elif reduction == "mean":
        pooled_probs = avgpool_img_tokens(probs)
        probs_by_img = probs[:, :n_img_tokens].mean(dim=0).reshape(16, 16)
    elif reduction == "max":
        pooled_probs = maxpool_img_tokens(probs, n_img_tokens=n_img_tokens)
        probs_by_img = probs[:, :n_img_tokens].max(dim=0)[0].reshape(16, 16)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # Create a side-by-side plot with img probs on the left and pooled probs on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    # Left plot: image probabilities
    plt.sca(ax1)

    if is_probabilities:
        title = f"Max Prob ({token_str}) over all layers"
        cmin = 0
        cmap = "Blues"
    else:
        title = "Agg LD over all layers"
        cmax = probs.abs().max()
        cmin = -cmax
        cmap = "RdBu"

    if cmin_cmax is None:
        cmax = cmax * 0.3 if not is_probabilities else cmax
        cmin = cmin * 0.3 if not is_probabilities else cmin
    else:
        cmin, cmax = cmin_cmax

    plot_img_probs(
        probs=probs_by_img,
        title=title,
        # img=np.array(image),
        ax=ax1,
        cmax=cmax,
        cmin=cmin,
        cmap=cmap,
    )
    # plt.grid(True, which="minor")

    # Right plot: pooled probabilities
    plt.sca(ax2)
    if is_probabilities:
        cbar_label = f"Prob ({token_str})"
    else:
        cbar_label = "Logit Diff"
    plot_pooled_probs_plt(
        pooled_probs,
        token_strings,
        cbar_label=cbar_label,
        ax=ax2,
        show_ylabel=False,
        cmax=cmax,
        cmin=cmin,
        cmap=cmap,
        n_img_tokens=n_img_tokens,
    )
    # No need to close pooled_fig since we're directly plotting to ax2

    plt.tight_layout()
    return fig


def plot_metric_with_std_over_layers(
    metric, ylabel: str, ax=None, label=None, marker="o"
):
    """metric is a tensor of shape (n_examples, n_layers)"""

    if ax is None:
        fig = plt.figure(figsize=(4, 2.5))
        ax = plt.gca()
    else:
        fig = ax.figure

    ax.plot(metric.mean(dim=0), marker=marker, label=label, alpha=0.8)
    # Plot mean with standard deviation bands
    means = metric.mean(dim=0)
    stds = metric.std(dim=0)
    x = range(len(means))

    ax.fill_between(x, means - stds, means + stds, alpha=0.3)
    ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(range(1, len(x) + 1)[::2])
    ax.grid()
    plt.tight_layout()

    return fig


def load_pg2_model_and_processor(
    compile=False, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
):
    torch.set_grad_enabled(False)  # avoid blowing up memory
    model_id = "google/paligemma2-3b-pt-224"
    model = (
        PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        .to("cuda")
        .eval()
    )
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    return model, processor


def image_symmetric_token_replacement(
    model,
    processor,
    text: str,
    healthy_img_alias: Path | str,
    healthy_tok_str: str,
    unhealthy_img_alias: Path | str,
    unhealthy_tok_str: str,
    tgt_directory: Path | str,
    healthy_img: Image.Image | None = None,
    unhealthy_img: Image.Image | None = None,
):
    """Runs STR patching using by using the unhealthy image as the corruption image"""

    if healthy_img is None:
        healthy_img = Image.open(healthy_img_alias)
    if unhealthy_img is None:
        unhealthy_img = Image.open(unhealthy_img_alias)

    healthy_tok_idx = processor.tokenizer.encode(healthy_tok_str)
    assert processor.tokenizer.decode(healthy_tok_idx) == healthy_tok_str
    healthy_inputs = processor(text=text, images=healthy_img, return_tensors="pt").to(
        model.device
    )
    healthy_embeds = paligemma_merge_text_and_image(model, healthy_inputs)
    healthy_activations, _ = get_decoder_layer_outputs(model, healthy_embeds)

    unhealthy_tok_idx = processor.tokenizer.encode(unhealthy_tok_str)
    assert processor.tokenizer.decode(unhealthy_tok_idx) == unhealthy_tok_str
    unhealthy_inputs = processor(
        text=text, images=unhealthy_img, return_tensors="pt"
    ).to(model.device)
    unhealthy_embeds = paligemma_merge_text_and_image(model, unhealthy_inputs)

    if healthy_tok_idx == unhealthy_tok_idx:
        print(
            "Skipping activation patching because the healthy and unhealthy response tokens are the same"
        )
        return

    patching_result: ActivationPatchingResult = patch_all_activations(
        model=model,
        healthy_activations=healthy_activations,
        unhealthy_embeds=unhealthy_embeds,
        healthy_response_tok_idx=healthy_tok_idx,
        unhealthy_response_tok_idx=unhealthy_tok_idx,
    )

    unhealthy_outputs = model(**unhealthy_inputs)
    unhealthy_logits = unhealthy_outputs.logits[0, -1, :]

    healthy_outputs = model(**healthy_inputs)
    healthy_logits = healthy_outputs.logits[0, -1, :]

    token_strings = processor.tokenizer.convert_ids_to_tokens(
        healthy_inputs.input_ids[0]
    )

    patching_result.save(
        directory=tgt_directory,
        health_response_tok=healthy_tok_str,
        unhealthy_response_tok=unhealthy_tok_str,
        corruption_type="symmetric_token_replacement",
        corruption_img_alias=unhealthy_img_alias,
        healthy_img_alias=healthy_img_alias,
        prompt=text,
        token_strings=token_strings,
        unhealthy_run_unhealthy_tok_logit=unhealthy_logits[unhealthy_tok_idx].item(),
        unhealthy_run_healthy_tok_logit=unhealthy_logits[healthy_tok_idx].item(),
        healthy_run_unhealthy_tok_logit=healthy_logits[unhealthy_tok_idx].item(),
        healthy_run_healthy_tok_logit=healthy_logits[healthy_tok_idx].item(),
    )


def guassian_noising_activation_patching(
    model,
    processor,
    text: str,
    healthy_img_alias: Path | str,
    healthy_tok_str: str,
    tgt_directory: Path | str,
    n_img_tokens: int,
    gaussian_noise_std: float,
    healthy_img: Image.Image | None = None,
):
    if healthy_img is None:
        healthy_img = Image.open(healthy_img_alias)

    healthy_inputs = processor(text=text, images=healthy_img, return_tensors="pt").to(
        model.device
    )
    healthy_embeds = paligemma_merge_text_and_image(model, healthy_inputs)
    healthy_activation, _ = get_decoder_layer_outputs(model, healthy_embeds)

    unhealthy_embeds = gaussian_noising(
        healthy_embeds,
        num_img_tokens=n_img_tokens,
        std=gaussian_noise_std,
    )

    healthy_tok_idx = processor.tokenizer.encode(healthy_tok_str)
    unhealthy_tok_idx = healthy_tok_idx

    unhealthy_outputs = model(inputs_embeds=unhealthy_embeds)
    unhealthy_logits = unhealthy_outputs.logits[0, -1, :]

    healthy_outputs = model(**healthy_inputs)
    healthy_logits = healthy_outputs.logits[0, -1, :]

    token_strings = processor.tokenizer.convert_ids_to_tokens(
        healthy_inputs.input_ids[0]
    )

    patching_result: ActivationPatchingResult = patch_all_activations(
        model=model,
        healthy_activations=healthy_activation,
        unhealthy_embeds=unhealthy_embeds,
        healthy_response_tok_idx=healthy_tok_idx,
        unhealthy_response_tok_idx=unhealthy_tok_idx,
    )

    patching_result.save(
        directory=tgt_directory,
        health_response_tok=healthy_tok_str,
        unhealthy_response_tok=healthy_tok_str,
        corruption_type="gaussian_noising",
        corruption_img_alias=None,
        healthy_img_alias=healthy_img_alias,
        prompt=text,
        token_strings=token_strings,
        unhealthy_run_unhealthy_tok_logit=unhealthy_logits[unhealthy_tok_idx].item(),
        unhealthy_run_healthy_tok_logit=unhealthy_logits[healthy_tok_idx].item(),
        healthy_run_unhealthy_tok_logit=healthy_logits[unhealthy_tok_idx].item(),
        healthy_run_healthy_tok_logit=healthy_logits[healthy_tok_idx].item(),
    )


def group_logits_diffs(
    logits_diffs: torch.Tensor, n_img_tokens: int, reduction: str
) -> torch.Tensor:
    """Logits diffs is a tensor of shape (n_layers, n_tokens)"""
    if reduction == "absmax":
        img_tokens = maxabs_reduction(logits_diffs[:, :n_img_tokens], dim=1)
        text_tokens = maxabs_reduction(logits_diffs[:, n_img_tokens + 1 : -1], dim=1)
    elif reduction == "mean":
        img_tokens = logits_diffs[:, :n_img_tokens].mean(dim=1)
        text_tokens = logits_diffs[:, n_img_tokens + 1 : -1].mean(dim=1)
    elif reduction == "max":
        img_tokens = logits_diffs[:, :n_img_tokens].max(dim=1)[0]
        text_tokens = logits_diffs[:, n_img_tokens + 1 : -1].max(dim=1)[0]
    elif reduction == "absmean":
        img_tokens = logits_diffs[:, :n_img_tokens].abs().mean(dim=1)
        text_tokens = logits_diffs[:, n_img_tokens + 1 : -1].abs().mean(dim=1)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    bos_tokens = logits_diffs[:, n_img_tokens]
    final_token = logits_diffs[:, -1]

    out = torch.stack([img_tokens, bos_tokens, text_tokens, final_token])
    return out  # shape (4, n_layers)


def square_img(img: Image.Image) -> np.ndarray:
    return np.array(img.resize((224, 224)))


def plot_squared_imgs(
    healthy_img: Image.Image,
    unhealthy_img: Image.Image,
    title1="Clean Image",
    title2="Corrupted Image",
):
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    fig.subplots_adjust(wspace=0)
    axes[0].imshow(square_img(healthy_img))
    axes[0].set_title(title1)
    axes[0].set_xlabel("Clean Image", fontsize=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(square_img(unhealthy_img))
    axes[1].set_title(title2)
    axes[1].set_xlabel("Corrupted Image", fontsize=12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    # fig.tight_layout()
    return fig


def dump_activation_patching_for_sample(
    pr: ActivationPatchingResult,
    dataset: dict[str, Any],
    n_img_tokens: int,
    tgt_dir: Path | str,
):
    """dataset maps image_id to row of VQA dataset"""

    tgt_dir = Path(tgt_dir)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    hvqa_row = dataset[pr.metadata["healthy_img_alias"]]
    uvqa_row = dataset[pr.metadata["corruption_img_alias"]]
    fig1 = plot_squared_imgs(
        hvqa_row["image"].convert("RGB"), uvqa_row["image"].convert("RGB")
    )
    fig1.tight_layout(pad=0.5)
    fig1.savefig(tgt_dir / "healthy_and_unhealthy_imgs.png")

    pooled = maxpool_img_tokens(pr.get_logit_diff_str(), n_img_tokens=n_img_tokens)
    fig2 = plot_pooled_probs_plt(
        pooled_probs=pooled,
        inputs_tokens=pr.metadata["token_strings"],
        cbar_label="Logit Diff",
        n_img_tokens=n_img_tokens,
        cmap="RdBu",
        cmax=(cmax := pooled.abs().max()),
        cmin=-cmax,
    )
    fig2.tight_layout(pad=0.5)
    fig2.savefig(tgt_dir / "pooled_probs.png")


def load_and_group_logit_diffs(
    root: str | Path, is_gaussian_noising: bool, n_img_tokens: int, reduction: str
):
    root = Path(root)
    grouped_logit_diffs = []
    for p in [root / p for p in os.listdir(root)]:
        pr = ActivationPatchingResult.load(p)
        try:
            if is_gaussian_noising:
                ld = pr.get_logit_diff_gn()
            else:
                ld = pr.get_logit_diff_str()
            cld = group_logits_diffs(ld, n_img_tokens=n_img_tokens, reduction=reduction)
            grouped_logit_diffs.append(cld)
        except ZeroDivisionError:
            print(f"Skipping {p} because denominator is 0")
            continue
    grouped_logit_diffs = torch.stack(grouped_logit_diffs)
    return grouped_logit_diffs
