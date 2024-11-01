import argparse
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import vec2text
import os
os.environ['HF_HOME'] = '/work/dagarwal_umass_edu/HF_HOME'
os.environ['TRANSFORMERS_CACHE'] = '/work/dagarwal_umass_edu/HF_HOME'


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
    parser.add_argument(
        "--sequence_beam_width", type=int, default=1
    )
    parser.add_argument(
        "--interpolate", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--interpolate_alpha", type=float, default=0.5
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

    while _input := input("Enter text: "):
        if _input == "exit":
            break
        
        inputs = [_input]
        
        if args.interpolate:
            _input2 = input("Enter text 2: ")
            inputs.append(_input2)
        
        embeddings = get_gtr_embeddings(inputs, encoder, tokenizer)
        
        if args.interpolate:
            if args.interpolate_alpha != -1:
                embeddings = torch.lerp(input=embeddings[0], end=embeddings[1], weight=args.interpolate_alpha)[None, :]
            else:
                # Interpolate at all alpha values
                for alpha in torch.linspace(0, 1, 11):
                    interpolated = torch.lerp(input=embeddings[0], end=embeddings[1], weight=alpha)[None, :]
                    inverted = vec2text.invert_embeddings(
                        embeddings=interpolated.cuda(),
                        corrector=corrector,
                        num_steps=args.inversion_steps,
                        sequence_beam_width=args.sequence_beam_width,
                    )
                    print(f"Alpha: {alpha:.1f}")
                    print(inverted)
                    print("")
                continue
        
        inverted = vec2text.invert_embeddings(
            embeddings=embeddings.cuda(),
            corrector=corrector,
            num_steps=args.inversion_steps,
            sequence_beam_width=args.sequence_beam_width,
        )
        print(inverted)
        print("")
