from transformers import AutoTokenizer, AutoModel

from peft import LoraConfig, get_peft_model, TaskType


def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    return get_peft_model(model, config)


"""
s: 130001, 130004,
tar: 130005
"""

if __name__ == '__main__':
    checkpoint = "THUDM/chatglm-6b"
    revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
    model = AutoModel.from_pretrained("../weight/chatglm-6b", revision=revision, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("../weight/chatglm-6b", revision=revision, trust_remote_code=True)

    # sava
    # tokenizer.save_pretrained('../权重/chatglm-6b_3084e')
    # model.save_pretrained('../权重/chatglm-6b_3084e')
    print(model)
    print(tokenizer)
    print(type(model))

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    print("done ! ")

    model = load_lora_config(model)
