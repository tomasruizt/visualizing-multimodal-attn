from getAttentionLib import (
    image_symmetric_token_replacement,
    load_pg2_model_and_processor,
)


if __name__ == "__main__":
    model, processor = load_pg2_model_and_processor()

    text = "<image>Answer en what is the frisbee's color?"
    n_img_tokens = 256

    healthy_img_path = "imgs/frisbee2.png"
    healthy_tok_str = "blue"

    unhealthy_img_path = "imgs/frisbee.jpg"
    unhealthy_tok_str = "purple"

    image_symmetric_token_replacement(
        model=model,
        processor=processor,
        text=text,
        healthy_img_alias=healthy_img_path,
        healthy_tok_str=healthy_tok_str,
        unhealthy_img_alias=unhealthy_img_path,
        unhealthy_tok_str=unhealthy_tok_str,
        tgt_directory="rev_frisbee_pr2",
    )
