from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG

args = DEFAULT_CONFIG
model = TransformerModel(args)
load_openai_pretrained_model(model)
