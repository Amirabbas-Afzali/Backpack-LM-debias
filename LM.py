import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_text(model, input_ids, max_length=50, temperature=1.0, top_k=50):
    """
    Generate text using the provided language model.
    
    Args:
        model: The language model.
        input_ids: The input token IDs to start the generation.
        max_length: The maximum length of the generated sequence.
        temperature: Controls the randomness of predictions by scaling the logits before applying softmax.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.

    Returns:
        generated_sequence: The generated token IDs.
    """
    # Initialize the generated sequence with the input_ids
    generated_sequence = input_ids

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass through the model
            torch_out = model(generated_sequence)

            # Extract logits for the last token in the sequence
            logits = torch_out.logits[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)

            # Stop generation if end of sequence token is produced
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated_sequence



model_id = "stanfordnlp/backpack-gpt2"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True)
model.to(device)

model.eval()

input_text = "Hi, What are you doing?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate text
output_ids = generate_text(model, input_ids, max_length=50, temperature=1.0, top_k=50)

# Decode the generated sequence
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:\n\n", output_text)