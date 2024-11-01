import argparse
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import vec2text
import os
os.environ['HF_HOME'] = '/work/dagarwal_umass_edu/HF_HOME'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", type=str, default="sentence-transformers/gtr-t5-base"
    )
    parser.add_argument(
        "--corrector", type=str, default="gtr-base"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="sentence-transformers/gtr-t5-base"
    )
    parser.add_argument(
        "--inversion_steps", type=int, default=20
    )
    args = parser.parse_args()
    return args


def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings


if __name__ == "__main__":
    args = parse_args()

    encoder = AutoModel.from_pretrained(args.encoder).encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    corrector = vec2text.load_pretrained_corrector(args.corrector)

    while _input := input("Enter text: ") != "exit":
        embeddings = get_gtr_embeddings([_input], encoder, tokenizer)
        inverted = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=args.inversion_steps
        )
        print(inverted)
        print("")
