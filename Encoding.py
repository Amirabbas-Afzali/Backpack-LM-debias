import torch
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
from modeling_backpack_gpt2 import BackpackGPT2LMHeadModel
from configuration_backpack_gpt2 import BackpackGPT2Config
import json

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load JSON file
with open('/mnt/d/EE/Term6/Projects/LLM/BP/config.json', 'r') as file:
    conf = json.load(file)

config = BackpackGPT2Config(**conf) 
model = BackpackGPT2LMHeadModel(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

checkpoint_path = '/mnt/d/EE/Term6/Projects/LLM/BP/pytorch_model.bin'
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Filter the external state_dict to retain only the keys present in the model's state_dict
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()} 

model.load_state_dict(state_dict=filtered_state_dict)
model.to(device)

print('Model is Loaded!')
model.eval()


input_text = "Hi, What are you doing?"

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# input = torch.randint(0, 50264, (1, 512), dtype=torch.long)
torch_out = model(
    input_ids,
    position_ids=None,
)

out = torch.nn.functional.softmax(torch_out.logits, dim=-1) 

print("Output shapes: ",out.shape) 
print("Senses shapes: ",torch_out.senses.shape)


