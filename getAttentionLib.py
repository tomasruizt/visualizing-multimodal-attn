from dataclasses import dataclass

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from circuitsvis.attention import attention_heads
from datasets import load_dataset


@dataclass
class State:
    outputs: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None

    def hook(self, module, input, output):
        # print("State.hook() called")
        output_tensor, attn_weights, cache = output
        self.outputs = output_tensor
        self.attn_weights = attn_weights


def get_img_grid_sizes(model, inputs):
    batch_size_, n_img_tokens, hidden_dim_ = model.get_image_features(
        inputs.pixel_values
    ).shape
    grid_side_len = n_img_tokens**0.5
    # print(grid_side_len)
    assert grid_side_len.is_integer()
    grid_side_len = int(grid_side_len)
    return n_img_tokens, grid_side_len


def get_attention(model, inputs, layer_idx: int) -> tuple[State, int]:
    all_layers = model.language_model.model.layers
    # print("num layers:", len(all_layers))
    attn_layer = all_layers[layer_idx].self_attn
    state = State()
    # print("state is empty:", state.outputs is None)
    hook_handle = attn_layer.register_forward_hook(state.hook)
    outputs = model(**inputs, output_attentions=True)
    n_output_tokens = outputs[0].shape[1]
    # print("state: is empty:", state.outputs is None)
    hook_handle.remove()
    return state, n_output_tokens


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
    attns = state.attn_weights[0].float()
    # print("full attns.shape:", attns.shape)

    bos_token = n_img_tokens
    i2i_attn = attns[:, :n_img_tokens, :n_img_tokens]
    i2b_attn = attns[:, :n_img_tokens, bos_token : bos_token + 1]
    i2t_attn = attns[:, :n_img_tokens, bos_token + 1 :]

    b2i_attn = attns[:, bos_token : bos_token + 1, :n_img_tokens]
    b2b_attn = attns[:, bos_token : bos_token + 1, bos_token : bos_token + 1]
    b2t_attn = attns[:, bos_token : bos_token + 1, bos_token + 1 :]

    t2i_attn = attns[:, bos_token + 1 :, :n_img_tokens]
    t2t_attn = attns[:, bos_token + 1 :, n_img_tokens:]
    t2b_attn = attns[:, bos_token + 1 :, bos_token : bos_token + 1]
    t2t_attn = attns[:, bos_token + 1 :, bos_token + 1 :]

    attn_sums = torch.tensor(
        [
            [
                i2i_attn.sum(dim=2).mean(),
                i2b_attn.sum(dim=2).mean(),
                i2t_attn.sum(dim=2).mean(),
            ],
            [
                b2i_attn.sum(dim=2).mean(),
                b2b_attn.sum(dim=2).mean(),
                b2t_attn.sum(dim=2).mean(),
            ],
            [
                t2i_attn.sum(dim=2).mean(),
                t2b_attn.sum(dim=2).mean(),
                t2t_attn.sum(dim=2).mean(),
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
    plt.imshow(attn_sums, cmap="viridis")
    if show_colorbar:
        plt.colorbar()
    names = ["img tokens", "<bos> token", "text tokens"]
    plt.xticks(ticks=[0, 1, 2], labels=names)
    plt.xlabel("Source token(s)")
    if not show_yticks:
        names = [""] * len(names)
    plt.yticks(ticks=[0, 1, 2], labels=names)
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
            color="white" if val < 0.5 else "black",
        )
    plt.tight_layout()
    return plt.gcf()


def compute_mult_attn_sums(model, inputs, layers: list[int]) -> list[torch.Tensor]:
    n_img_tokens, _ = get_img_grid_sizes(model, inputs)
    mult_attn_sums = []
    for layer in layers:
        state, _ = get_attention(model, inputs, layer_idx=layer)
        mult_attn_sums.append(compute_attn_sums(state, n_img_tokens))
    return mult_attn_sums


def plot_mult_attn_sums(
    model, inputs, layers: list[int], mult_attn_sums=None, stds=None
) -> plt.Figure:
    if mult_attn_sums is None:
        mult_attn_sums = compute_mult_attn_sums(model, inputs, layers)

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
