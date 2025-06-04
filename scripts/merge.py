from peft import PeftConfig, PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

peft_model_id = "Asanshay/websight-7B"
config = PeftConfig.from_pretrained(peft_model_id)

base_model = AutoModelForVision2Seq.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)
websight_model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer = AutoProcessor.from_pretrained(config.base_model_name_or_path)

from huggingface_hub import login

# Log in to Hugging Face
login()

# Push the merged model to the Hub
websight_model.push_to_hub("tanvirb/websight-7B")
tokenizer.push_to_hub("tanvirb/websight-7B")