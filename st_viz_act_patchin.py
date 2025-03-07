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

# method = st.radio("Select a method", ["str", "gn"])
method = "str"
is_gaussian_noising = method == "gn"

cases_str: list[str] = sorted(f.name for f in root.glob("str/*"))
st.write("num str cases:", len(cases_str))
cases_gn: list[str] = sorted(f.name for f in root.glob("gn/*"))
st.write("num gn cases:", len(cases_gn))
joint_cases = sorted(set(cases_str) & set(cases_gn))
st.write("num joint cases:", len(joint_cases))

case = st.selectbox("Select a case", joint_cases)

str_pr = ActivationPatchingResult.load(root / "str" / case)
gn_pr = ActivationPatchingResult.load(root / "gn" / case)

c1, c2 = st.columns(2)

with c1:
    st.write("**Prompt:**", str_pr.metadata["prompt"])
    st.write("**Healthy response token:**", str_pr.metadata["healthy_response_tok"])
    st.write("**Unhealthy response token:**", str_pr.metadata["unhealthy_response_tok"])

if str_pr.metadata["corruption_img_alias"] is not None:
    matching_img_ids = []
    matching_img_ids.extend(
        img_ids[img_ids == str_pr.metadata["healthy_img_alias"]].index.values
    )
    matching_img_ids.extend(
        img_ids[img_ids == str_pr.metadata["corruption_img_alias"]].index.values
    )
    rows = dataset.select(matching_img_ids).to_pandas()
    short_question = str_pr.metadata["prompt"].replace("<image>Answer en", "").strip()
    rows = rows.query("question == @short_question")
    hvqa_row = rows.query("image_id == @str_pr.metadata['healthy_img_alias']").iloc[0]
    uvqa_row = rows.query("image_id == @str_pr.metadata['corruption_img_alias']").iloc[
        0
    ]
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
    row = dataset.select(
        [img_ids[img_ids == str_pr.metadata["healthy_img_alias"]].index.values[0]]
    )[0]
    fig = plt.imshow(row["image"])
    st.write(fig)


kwargs = dict(
    n_img_tokens=256,
    token_strings=str_pr.metadata["token_strings"],
    token_str=str_pr.metadata["healthy_response_tok"],
    is_probabilities=False,
    cmin_cmax=(-1, 1),
)


c1, c2 = st.columns(2)

with c1:
    st.write("### Reduction: mean, Method: IR")
    fig = plot_img_and_text_probs_side_by_side(
        str_pr.get_logit_diff_str(), reduction="mean", **kwargs
    )
    st.write(fig)
    st.write("### Reduction: mean, Method: GN")
    fig = plot_img_and_text_probs_side_by_side(
        gn_pr.get_logit_diff_gn(), reduction="mean", **kwargs
    )
    st.write(fig)

with c2:
    st.write("### Reduction: absmax, Method: IR")
    fig = plot_img_and_text_probs_side_by_side(
        str_pr.get_logit_diff_str(), **kwargs, reduction="absmax"
    )
    st.write(fig)
    st.write("### Reduction: absmax, Method: GN")
    fig = plot_img_and_text_probs_side_by_side(
        gn_pr.get_logit_diff_gn(), **kwargs, reduction="absmax"
    )
    st.write(fig)
