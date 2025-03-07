from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from getAttentionLib import (
    ActivationPatchingResult,
    load_vqa_ds,
    plot_img_and_text_probs_side_by_side,
    plot_squared_imgs,
)

st.set_page_config(layout="wide")


@st.cache_data
def get_dataset_cached():
    dataset = load_vqa_ds(split="train")
    img_ids = pd.Series(dataset["image_id"])
    return dataset, img_ids


dataset, img_ids = get_dataset_cached()

root = Path("vqa_patching")

# folder1 = "str/*"
folder1 = "gn_sigma_xs"
cases1: list[str] = sorted(f.name for f in root.glob(f"{folder1}/*"))
st.write(f"num {folder1} cases:", len(cases1))
folder2 = "gn"
cases2: list[str] = sorted(f.name for f in root.glob(f"{folder2}/*"))
st.write(f"num {folder2} cases:", len(cases2))
joint_cases = sorted(set(cases1) & set(cases2))
st.write("num joint cases:", len(joint_cases))

if len(joint_cases) == 0:
    st.error("No joint cases found")
    st.stop()

cmax = st.slider("cmax", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
case = st.radio("Select a case", joint_cases, horizontal=True)

pr1 = ActivationPatchingResult.load(root / folder1 / case)
pr2 = ActivationPatchingResult.load(root / folder2 / case)

c1, c2 = st.columns(2)

with c1:
    st.write("**Prompt:**", pr1.metadata["prompt"])
    st.write("**Healthy response token:**", pr1.metadata["healthy_response_tok"])
    st.write("**Unhealthy response token:**", pr1.metadata["unhealthy_response_tok"])


def find_matching_rows(dataset, img_ids, healthy_img_id, corruption_img_id=None):
    matching_img_ids = []
    matching_img_ids.extend(img_ids[img_ids == healthy_img_id].index.values)
    matching_img_ids.extend(img_ids[img_ids == corruption_img_id].index.values)
    rows = dataset.select(matching_img_ids).to_pandas()
    short_question = pr1.metadata["prompt"].replace("<image>Answer en", "").strip()
    rows = rows.query("question == @short_question")
    clean_row = rows.query("image_id == @healthy_img_id").iloc[0]
    if corruption_img_id is not None:
        corrupted_row = rows.query("image_id == @corruption_img_id").iloc[0]
    else:
        corrupted_row = None
    return clean_row, corrupted_row


hvqa_row, uvqa_row = find_matching_rows(
    dataset,
    img_ids,
    pr1.metadata["healthy_img_alias"],
    pr1.metadata["corruption_img_alias"],
)

if pr1.metadata["corruption_img_alias"] is not None:
    with c1:
        st.write("**Answer To Clean Image:**", hvqa_row["multiple_choice_answer"])
        st.write("**Answer To Corrupt Image:**", uvqa_row["multiple_choice_answer"])
    fig = plot_squared_imgs(
        Image.open(hvqa_row["image"]["path"]).convert("RGB"),
        Image.open(uvqa_row["image"]["path"]).convert("RGB"),
    )
    with c2:
        st.pyplot(fig)
else:
    with c1:
        st.write(
            "**Dataset answer for clean image:**", hvqa_row["multiple_choice_answer"]
        )
    with c2:
        st.image(hvqa_row["image"]["path"])


kwargs = dict(
    n_img_tokens=256,
    token_strings=pr1.metadata["token_strings"],
    token_str=pr1.metadata["healthy_response_tok"],
    is_probabilities=False,
    cmin_cmax=(-cmax, cmax),
)


c1, c2 = st.columns(2)

logits_diff1 = pr1.get_logit_diff_str() if "str" in folder1 else pr1.get_logit_diff_gn()
logits_diff2 = pr2.get_logit_diff_str() if "str" in folder2 else pr2.get_logit_diff_gn()


with c1:
    st.write(f"### Reduction: mean, Folder: {folder1}")
    fig = plot_img_and_text_probs_side_by_side(logits_diff1, reduction="mean", **kwargs)
    st.write(fig)
    st.write(f"### Reduction: mean, Folder: {folder2}")
    fig = plot_img_and_text_probs_side_by_side(logits_diff2, reduction="mean", **kwargs)
    st.write(fig)

with c2:
    st.write(f"### Reduction: absmax, Folder: {folder1}")
    fig = plot_img_and_text_probs_side_by_side(
        logits_diff1, **kwargs, reduction="absmax"
    )
    st.write(fig)
    st.write(f"### Reduction: absmax, Folder: {folder2}")
    fig = plot_img_and_text_probs_side_by_side(
        logits_diff2, **kwargs, reduction="absmax"
    )
    st.write(fig)
