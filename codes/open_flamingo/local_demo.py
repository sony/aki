import gradio as gr
import sys
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from src.aki import AKI
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def load_model_and_processor(ckpt_path, config):
    # replace GenerationMixin to modify attention mask handling
    from transformers.generation.utils import GenerationMixin
    from open_flamingo import _aki_update_model_kwargs_for_generation
    GenerationMixin._update_model_kwargs_for_generation = _aki_update_model_kwargs_for_generation
    
    n_px = getattr(config, "n_px", 384)
    norm_mean = getattr(config, "norm_mean", 0.5)
    norm_std = getattr(config, "norm_std", 0.5)

    # replace GenerationMixin to modify attention mask handling
    from transformers.generation.utils import GenerationMixin
    from open_flamingo import _aki_update_model_kwargs_for_generation
    GenerationMixin._update_model_kwargs_for_generation = _aki_update_model_kwargs_for_generation
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AKI.from_pretrained(ckpt_path, tokenizer=tokenizer)
    image_processor = Compose([
        Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True),
        Lambda(lambda x: x.convert('RGB')),
        ToTensor(),
        Normalize(mean=(norm_mean, norm_mean, norm_mean), std=(norm_std, norm_std, norm_std))
    ])

    model.eval().cuda()
    print("Model initialization is done.")
    return model, image_processor, tokenizer

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

# Function to process the input for the Gradio interface
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

    print("Prompt:", prompt)
    
    # Preprocess inputs for the model
    vision_x = image_processor(image)[None, None, None, ...].cuda()

    generation_kwargs = {
        'max_new_tokens': 256,
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

# Main execution
if __name__ == "__main__":
    model_path = "Sony/AKI-4B-phi-3.5-mini"
    # Load model, image_processor, tokenizer
    config = AutoConfig.from_pretrained(model_path)
    model, image_processor, tokenizer = load_model_and_processor(model_path, config=config)

    # Define and set up the Gradio interface
    demo = gr.Interface(
        fn=process_input,  # Function that handles the inputs
        inputs=[
            gr.Image(type="pil", label="Input Image"),  # Image input
            gr.Textbox(label="Text Prompt")              # Text input box for prompt
        ], 
        outputs=gr.Textbox(label="Generated Output"),    # Output box for generated text
        title="Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs",  # Title for UI
        description="Upload an image and provide a text prompt. Our AKI model will generate a response based on the given inputs."
    )

    # Launch the interface
    demo.launch(share=False)