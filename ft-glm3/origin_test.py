from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

response, history = model.chat(tokenizer, "请续写歌词，第一句为：春天花会开", history=[])
print(response)

response, history = model.chat(tokenizer, "请续写歌词，第一句为：帘外芭蕉惹骤雨", history=[])
print(response)

response, history = model.chat(tokenizer, "请续写歌词，第一句为：如果离开以后", history=[])
print(response)