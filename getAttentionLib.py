from dataclasses import dataclass
from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration

from typing import Any

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from circuitsvis.attention import attention_heads
from datasets import load_dataset
from torch import nn
from transformers import PaliGemmaForConditionalGeneration, BatchFeature
from tqdm import trange
import plotly.graph_objects as go


@dataclass
class State:
    outputs: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None

    def hook(self, module, input, output):
        # print("State.hook() called")
        output_tensor, attn_weights, cache = output
        self.outputs = output_tensor
        self.attn_weights = attn_weights


@dataclass
class Hook:
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
        out, cache = output
        desired = self.healthy_activations[self.layer_idx, self.token_idx]
        out[0, self.token_idx, :] = desired
        return out, cache


def get_img_grid_sizes(model, inputs):
    
    if isinstance(model, Idefics3ForConditionalGeneration):
        vision_outputs = model.model.vision_model(inputs.pixel_values[0].to(dtype=torch.bfloat16))
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


def get_attention(model, model_kwargs: dict, layer_idx: int) -> tuple[State, int]:
    all_layers = model.language_model.model.layers
    # print("num layers:", len(all_layers))
    attn_layer = all_layers[layer_idx].self_attn
    state = State()
    # print("state is empty:", state.outputs is None)
    hook_handle = attn_layer.register_forward_hook(state.hook)
    outputs = model(**model_kwargs, output_attentions=True)
    n_output_tokens = outputs[0].shape[1]
    # print("state: is empty:", state.outputs is None)
    hook_handle.remove()
    return state, n_output_tokens

def get_attention_smol_lvm(model, inputs, layer_idx: int) -> tuple[State, int]:
    attention_maps = []
    
    def attention_hook(module, input, output):
        attention_maps.append(output[0])  # attention weights are typically the second output
    
    # Register the hook on the specific layer's attention
    hook = model.model.text_model.layers[layer_idx].self_attn.register_forward_hook(attention_hook)
    
    # Forward pass
    outputs = model(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.get('attention_mask', None),
        output_hidden_states=True,
        return_dict=True
    )
    
    # Remove the hook
    hook.remove()
    
    # Get the attention weights from our hook
    attention_weights = attention_maps[0] if attention_maps else None
    
    class State:
        def __init__(self, outputs, attn_weights):
            self.outputs = outputs
            self.attn_weights = attn_weights
    
    state = State(outputs.hidden_states[-1], attention_weights)
    return state, inputs.input_ids.shape[1]

def dump_attn(
    state: State,
    layer_idx: int,
    name: str,
    tokens: list[str],
    img_path: str,
    grid_side_len: int,
) -> None:
    viz_data = dict(
        attention=state.attn_weights[0].round(decimals=6).cpu().tolist(),
        tokens=tokens,
        image=img_path,
        image_grid_dims=[grid_side_len, grid_side_len],
        image_tokens_start=0,
        max_value=state.attn_weights.max().item() / 100,
        min_value=state.attn_weights.min().item(),
    )

    html = attention_heads(**viz_data)
    html_str = str(html)
    print("Rendered HTML")
    tgt_file = f"{name}_layer_{layer_idx}_attention_heads.html"
    with open(tgt_file, "w") as f:
        f.write(html_str)
    print(f"Wrote to file='{tgt_file}'")


def get_response(
    model, processor, text: str, image: PIL.Image.Image
) -> tuple[list[str], str]:
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    inputs_tokens = [processor.decode(id) for id in inputs.input_ids[0]]
    outputs = model.generate(**inputs, max_new_tokens=100)
    response: str = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return inputs_tokens, response


def compute_attn_sums(state: State, n_img_tokens: int) -> torch.Tensor:
    attns = state.attn_weights.float()
    if len(attns.shape) == 4:
        attns = attns[0]
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
):
    plt.imshow(attn_sums, cmap="Blues")
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
            color="black" if val < 0.5 else "white",
        )
    plt.tight_layout()
    return plt.gcf()


def compute_mult_attn_sums(
    model, model_kwargs, layers: list[int], n_img_tokens=None
) -> list[torch.Tensor]:
    if n_img_tokens is None:
        n_img_tokens, _ = get_img_grid_sizes(model, model_kwargs)
    mult_attn_sums = []
    for layer in layers:
        if isinstance(model, Idefics3ForConditionalGeneration):
            state, _ = get_attention_smol_lvm(model, model_kwargs, layer_idx=layer)
        else:
            state, _ = get_attention(model, model_kwargs, layer_idx=layer)
        mult_attn_sums.append(compute_attn_sums(state, n_img_tokens))
    return mult_attn_sums


def plot_mult_attn_sums(
    model,
    model_kwargs,
    layers: list[int],
    mult_attn_sums=None,
    stds=None,
    n_img_tokens=None,
) -> plt.Figure:
    if mult_attn_sums is None:
        mult_attn_sums = compute_mult_attn_sums(
            model, model_kwargs, layers, n_img_tokens
        )

    plt.figure(figsize=(12, 4))
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
        )
    return fig


def load_vqa_ds(split: str | None = None):
    avoid_timeout = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}}
    ds = load_dataset("HuggingFaceM4/VQAv2", split=split, storage_options=avoid_timeout)
    return ds


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
            torch.stack(mult_attn_sums)[:, i, :].T,
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


def gaussian_noising(input_embeds: torch.Tensor, num_img_tokens: int) -> torch.Tensor:
    """Introduce noise in the image tokens. They are assumed to be at the begginning."""
    new_embeds = input_embeds.clone()
    noise = torch.randn_like(new_embeds[:, :num_img_tokens, :]) * 3**0.5  # Var=3
    # noise = torch.zeros_like(new_embeds[:, :num_img_tokens, :])
    new_embeds[:, :num_img_tokens, :] += noise
    return new_embeds


def get_activations(model, inputs_embeds):
    handles = []
    hooks = []

    for layer in model.language_model.model.layers:
        hook = Hook()
        hooks.append(hook)
        handle = layer.register_forward_hook(hook.hook)
        handles.append(handle)

    try:
        outputs = model(inputs_embeds=inputs_embeds)
    finally:
        for handle in handles:
            handle.remove()

    activations = torch.stack([h.output[0][0] for h in hooks])
    return activations, outputs


def restore_activation(model, unhealthy_embeds, hook: RestoreActivationHook):
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


def loop_over_restore_all_activations(
    model, healthy_activations, unhealthy_embeds, healthy_response_tok_idx: int
):
    n_layers = len(model.language_model.model.layers)
    n_tokens = healthy_activations.shape[1]
    correct_tok_probs = torch.zeros(n_layers, n_tokens)

    for layer_idx in trange(n_layers):
        for token_idx in trange(n_tokens, leave=False):
            restore_hook = RestoreActivationHook(
                healthy_activations, layer_idx=layer_idx, token_idx=token_idx
            )
            outputs = restore_activation(model, unhealthy_embeds, restore_hook)
            prob = tok_prob(outputs, healthy_response_tok_idx)
            correct_tok_probs[layer_idx, token_idx] = prob

    return correct_tok_probs


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
    pooled_probs: torch.Tensor, inputs_tokens: list[str], healthy_response_tok_name: str
):
    plt.imshow(pooled_probs.T, cmap="Blues")
    plt.ylabel("layer")
    plt.xlabel("token")
    plt.yticks(
        ticks=range(len(pooled_probs.T)),
        labels=["img_tokens"] + inputs_tokens[256:],
    )
    cbar = plt.colorbar()
    cbar.set_label(f"Prob ({healthy_response_tok_name})")
    plt.tight_layout()
    return plt.gcf()


import ipywidgets as widgets
from IPython.display import display, clear_output


def plot_and_browse_img_token_in_probs(probs: torch.Tensor):
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
        plt.imshow(probs[layer, :256].reshape(16, 16), cmap="Blues")
        plt.colorbar().mappable.set_clim(0, 0.7)
        plt.title(f"Layer {layer}")
        plt.show()

    # Attach the update function to the slider
    slider.observe(update_plot, names="value")

    # Initial plot
    update_plot({"new": 0})


def maxpool_img_tokens(probs: torch.Tensor) -> torch.Tensor:
    imgs_max_pool = probs[:, :256].max(dim=1)[0][:, None]
    pooled = torch.hstack([imgs_max_pool, probs[:, 256:]])
    return pooled
