import pandas as pd
import torch
import sys
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from open_flamingo import create_model_and_transforms


def load_model_and_processor(ckpt_path):
    # replace GenerationMixin to modify attention mask handling
    from transformers.generation.utils import GenerationMixin
    from open_flamingo import _aki_update_model_kwargs_for_generation
    GenerationMixin._update_model_kwargs_for_generation = _aki_update_model_kwargs_for_generation
    
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="siglip-so400m-patch14-384",
        clip_vision_encoder_pretrained="google",
        lang_encoder_path='microsoft/Phi-3.5-mini-instruct',
        tokenizer_path='microsoft/Phi-3.5-mini-instruct',
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    msd = checkpoint.pop("model_state_dict")
    msd = {k.replace("module.", ""): v for k, v in msd.items()}

    if 'vision_tokenizer.latents' in msd.keys():
        msd_current = model.state_dict()
        if msd_current['vision_tokenizer.latents'].shape != msd['vision_tokenizer.latents'].shape:
            msd["vision_tokenizer.latents"] = msd_current['vision_tokenizer.latents'] # Random re-init.

    results = model.load_state_dict(msd, strict=False)
    torch.cuda.empty_cache()
    model.eval().cuda()
    print("Model initialization is done.")
    return model, image_processor, tokenizer


def optionize_choices(choices: list) -> str:
    """
    Convert a list of choices into a formatted string.
    
    Args:
    choices (list): A list of choices to convert.
    
    Returns:
    str: The formatted string of choices.
    """
    options = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join([f"{options[i]}. {choice}" for i, choice in enumerate(choices)])


def apply_prompt_template(query: str) -> str:
    SYSTEM_BASE = "A chat between a curious user and an artificial intelligence assistant."
    SYSTEM_DETAIL = "The assistant gives helpful, detailed, and polite answers to the user's questions."
    SYSTEM_MESSAGE = SYSTEM_BASE + " " + SYSTEM_DETAIL
    SYSTEM_MESSAGE_ROLE = '<|system|>' + '\n' + SYSTEM_MESSAGE + '<|end|>\n'

    s = (
        f'<s> {SYSTEM_MESSAGE_ROLE}'
        f'<|user|>\n<image>\n{query}<|end|>\n<|assistant|>\n'
    )
    return s


def process_input(image: Image.Image, text_input: str) -> str:
    """
    Processes the input image and text prompt to generate a response from the AKI model.
    
    Args:
    image (PIL.Image): The input image.
    text_input (str): The text prompt to accompany the image.
    
    Returns:
    str: The generated text from the model.
    """
    
    # tokenize text input with the chat template
    prompt = apply_prompt_template(text_input)
    lang_x = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)

    # print("Prompt:", prompt)
    
    # Preprocess inputs for the model
    vision_x = []
    vision_x.append(image_processor(image).unsqueeze(0))

    vision_x = torch.cat(vision_x, dim=0) if len(vision_x) > 1 else vision_x[0]
    vision_x = vision_x.unsqueeze(1).unsqueeze(0).cuda()

    generation_kwargs = {
        'max_new_tokens': 256,
        'temperature': 0.0,
        'do_sample': False,
    }
    
    # Generate the model's output based on the inputs
    output = model.generate(
        vision_x=vision_x.cuda(),
        lang_x=lang_x['input_ids'].cuda(),
        attention_mask=lang_x['attention_mask'].cuda(),
        **generation_kwargs
    )
    
    # Decode the generated output into readable text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


# Define a function to calculate accuracy for a given source
def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy


def compute_scores(df):
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    # Print the results
    print(f"CV-Bench Accuracy: {combined_accuracy:.4f}")
    print()
    print(f"Type Accuracies:")
    print(f"2D Accuracy: {accuracy_2d:.4f}")
    print(f"3D Accuracy: {accuracy_3d:.4f}")
    print()
    print(f"Source Accuracies:")
    print(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}")
    print(f"COCO Accuracy: {accuracy_2d_coco:.4f}")
    print(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}")

    # write the results to a file
    with open("cv_bench_results.txt", "w") as f:
        f.write(f"CV-Bench Accuracy: {combined_accuracy:.4f}\n")
        f.write("\nType Accuracies:\n")
        f.write(f"2D Accuracy: {accuracy_2d:.4f}\n")
        f.write(f"3D Accuracy: {accuracy_3d:.4f}\n")
        f.write("\nSource Accuracies:\n")
        f.write(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}\n")
        f.write(f"COCO Accuracy: {accuracy_2d_coco:.4f}\n")
        f.write(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}\n")


if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    model, image_processor, tokenizer = load_model_and_processor(ckpt_path)

    # generate predictions
    cv_bench = load_dataset("nyu-visionx/CV-Bench")
    questions, image_paths = [], []
    sources, correct_or_incorrect = [], []
    predictions, answers = [], []
    cnt = 0
    for sample in tqdm(cv_bench['test']):
        text_input = f"Answer with the option's letter from the given choices directly. {sample['question']}\nOptions:\n{optionize_choices(sample['choices'])}\n"
        questions.append(text_input), image_paths.append(sample['filename'])
        prediction = process_input(sample["image"], text_input)

        # post-process: remove () from the ground truth
        answer = sample['answer'].replace("(", "").replace(")", "")
        predictions.append(prediction), answers.append(answer)
    
        # Check if the prediction is correct
        correct_or_incorrect.append(int(prediction == answer))
        sources.append(sample['source'])

    results = pd.DataFrame({'question': questions, 'path': image_paths, 'answer': answers, 'prediction': predictions, 'source': sources, 'result': correct_or_incorrect})
    results.to_csv("cv_bench_results.csv", index=False)
    compute_scores(results)