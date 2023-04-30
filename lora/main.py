import json
import os
from typing import Dict, List, Union, Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
max_src_length = 256
max_dst_length = 64
prefix = ""


def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=['query_key_value']  # ['dense', 'dense_h_to_4h', 'dense_4h_to_h', 'query_key_value']
    )
    # ['dense','dense_h_to_4h','dense_4h_to_h','query_key_value']
    return get_peft_model(model, config)


def create_inputs_and_labels(tokenizer, examples: Dict[str, Union[str, List]], prompt_column="prompt",
                             response_column="response",
                             history_column="history", ignore_pad_token_for_loss=True):
    """
    将json格式的数据转换为input_ids 和 label 的格式
    Args:
        examples: json格式的数据
        prompt_column:
        response_column:
        history_column:
        ignore_pad_token_for_loss: 是否将label中的pad计算loss，默认不计算算pad的loss

    Returns: 字典：{"input_ids":...,"labels":...}

    """
    max_seq_length = max_src_length + max_dst_length
    model_inputs = {}

    query, answer = examples[prompt_column], examples[response_column]
    prompt = ""
    history = examples[history_column]
    for turn_idx, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
    prompt += "[Round {}]\n问：{}".format(len(history), query)
    answer_prefix = "\n答："

    answer_prefix_ids = tokenizer.encode(
        answer_prefix,
        add_special_tokens=True
    )
    answer_prefix_len = len(answer_prefix_ids)
    special_tokens_num = 2
    prompt_ids = tokenizer.encode(
        prompt,
        max_length=max_src_length - (answer_prefix_len - special_tokens_num),
        truncation=True,
        add_special_tokens=False
    )

    prompt_ids = prompt_ids + answer_prefix_ids
    answer_ids = tokenizer.encode(
        answer,
        max_length=max_dst_length,
        truncation=True,
        add_special_tokens=False
    )

    input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]

    context_length = input_ids.index(tokenizer.bos_token_id)
    mask_position = context_length - 1
    labels = [-100] * context_length + input_ids[mask_position + 1:]

    # labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]

    model_inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long, device=device)

    return model_inputs


def get_attention_mask(tokenizer, input_ids, device):
    seq = input_ids.tolist()
    context_len = seq.index(tokenizer.bos_token_id)
    seq_len = len(seq)
    attention_mask = torch.ones((seq_len, seq_len), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_len] = 1
    attention_mask.unsqueeze_(0)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(tokenizer, input_ids, device,
                     position_encoding_2d=True):
    seq = input_ids.tolist()
    context_len = seq.index(tokenizer.bos_token_id)
    seq_len = len(seq)

    mask = tokenizer.mask_token_id
    gmask = tokenizer.gmask_token_id

    mask_token = mask if mask in seq else gmask
    use_gmask = False if mask in seq else gmask

    mask_position = seq.index(mask_token)

    if position_encoding_2d:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(context_len, dtype=torch.long, device=device),
            torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position

    return position_ids


class MyDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer

        # Todo fix answer_ids 以5开始
        # a = preprocess_function_train(tokenizer=tokenizer, examples=item_data)
        model_inputs = create_inputs_and_labels(tokenizer=tokenizer, examples=item_data)
        input_ids = model_inputs["input_ids"]
        labels = model_inputs["labels"]

        attention_mask = get_attention_mask(tokenizer, input_ids, device)
        position_ids = get_position_ids(tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []

    for obj in batch:
        input_ids.append(obj['input_ids'])
        labels.append(obj['labels'])
        attention_mask.append(obj['attention_mask'])
        position_ids.append(obj['position_ids'])

    return {
        'input_ids': torch.stack(input_ids).to(device),
        'attention_mask': torch.stack(attention_mask).to(device),
        'labels': torch.stack(labels).to(device),
        'position_ids': torch.stack(position_ids).to(device)
    }


def test():
    local_weight = "../weight/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(local_weight, trust_remote_code=True)
    test_data = [{"prompt": "在看考公务员\n你找工作没，松松", "response": "聊天嘛\n瑞瑞", "history": []},
                 {"prompt": "在看考公务员\n你找工作没，松松", "response": "聊天嘛\n瑞瑞", "history": []}]

    dataset = MyDataset(test_data, tokenizer)
    train_data = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
    for item in train_data:
        print(item['input_ids'].shape)
        print(item['labels'].shape)
        print(item)

    # print("inputs: \n", input_ids)
    # print("\nlabels: \n", labels)
    # print("\nposition_ids: \n", position_ids.tolist())
    # print("\nattention_mask: \n", attention_mask.tolist())


def save_tuned_parameters(model, path):
    saved_params = {
        k: v.to(device)
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def load_train_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        objs = [json.loads(l.rstrip("\n")) for l in f.readlines()]

    a = [{"prompt": "你好", "response": "你好，我是一个聊天机器人，由yacai的微信的聊天记录训练而来。", "history": []},
         {"prompt": "您好", "response": "您好，我是一款聊天机器人，我可以陪您聊天、解答问题。", "history": []},
         {"prompt": "在吗", "response": "您好，我是一款聊天机器人，我可以陪您聊天、解答问题。", "history": []},
         {"prompt": "有人吗", "response": "您好，我是一款聊天机器人，我可以陪您聊天、解答问题,根据微信的聊天记录训练而来。", "history": []},
         {"prompt": "请问你能干什么", "response": "您好，我是一款聊天机器人，我可以陪您聊天、解答问题。", "history": []},
         {"prompt": "你好", "response": "你好，我是一个聊天机器人，由yacai的微信的聊天记录训练而来。", "history": []},
         {"prompt": "您好", "response": "您好，有啥问题吗", "history": []},
         {"prompt": "在吗", "response": "有什么我可以帮您的吗？我是一个聊天机器人，由微信的聊天记录训练而来。", "history": []},
         {"prompt": "有人吗", "response": "您好，我是一名智能机器人，可以陪您聊天、猜谜语、讲笑话等。", "history": []},
         {"prompt": "你是机器人吗", "response": "是的，我是一个聊天机器人，可以和您聊天、解答问题、提供帮助。", "history": []},
         {"prompt": "请问你能干什么", "response": "您好，我是一名智能聊天机器人，可以回答您的问题、谈天说地。", "history": []},
         {"prompt": "请问可以跟你聊天吗", "response": "当然可以，我是一款AI聊天机器人，经过不断学习和优化，可以进行人性化的对话。", "history": []},
         {"prompt": "你能理解中文吗", "response": "是的，我能够理解中文，并且可以模拟人类的语言行为与您进行交流。", "history": []},
         {"prompt": "你是什么", "response": "我是一个聊天机器人，通过人工智能技术和大量数据的训练，可以进行智能对话。", "history": []},
         {"prompt": "嗨", "response": "嗨，我是一款聊天机器人，通过yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "您好呀", "response": "您好呀，我是一个聊天机器人，基于yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "你好啊", "response": "你好啊，我是一名聊天机器人，根据yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "您好呢", "response": "您好呢，我是一台聊天机器人，通过yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "嘿", "response": "嘿，我是一个聊天机器人，由yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "你好哇", "response": "你好哇，我是一名聊天机器人，受过yacai的微信聊天记录的训练。", "history": []},
         {"prompt": "你好嘞", "response": "你好嘞，我是一款聊天机器人，基于yacai的微信聊天记录进行了训练。", "history": []},
         {"prompt": "您好啦", "response": "您好啦，我是一台聊天机器人，由yacai的微信聊天记录进行了训练。", "history": []},
         ]
    objs.extend(a)
    print(objs[-10:])

    return objs


def train():
    local_weight = "../weight/chatglm-6b"
    checkpoint = "THUDM/chatglm-6b"
    revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
    model = AutoModel.from_pretrained(local_weight, trust_remote_code=True)
    # print(f"model quantize to 8 bit ")
    # model = model.quantize(8)
    tokenizer = AutoTokenizer.from_pretrained(local_weight, trust_remote_code=True)

    model = load_lora_config(model)
    model = model.half().to(device)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./output",
        fp16=False,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        num_train_epochs=5,
        save_steps=1,  # 设置为 1，表示每个 epoch 结束时保存一次模型的权重
        save_strategy="epoch",  # 设置为 "epoch"，表示按 epoch 保存模型的权重
        save_total_limit=5,

        evaluation_strategy="no",  # 禁用评估
        eval_steps=1,

        # max_steps=3000,
        logging_strategy="steps",  # 设置 logging_strategy 为 "steps"
        logging_steps=100,  # 设置 logging_steps 为打印损失的频率

        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,

        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
        dataloader_pin_memory=False,
    )

    class ModifiedTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False):
            return model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                labels=inputs["labels"]
            ).loss

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            def save_tunable_parameters(model, path):
                saved_params = {
                    k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
                }
                # saved_params = model.state_dict()
                torch.save(saved_params, path)
                print(f"#### 保存 lora 权重:{path}")

            save_tunable_parameters(
                self.model, os.path.join(output_dir, "chatglm-lora.pt")
            )

    test_data = [{"prompt": "在看考公务员\n你找工作没，松松", "response": "聊天嘛\n瑞瑞", "history": []},
                 {"prompt": "在看考公务员\n你找工作没，松松", "response": "聊天嘛\n瑞瑞", "history": []}]
    train_data = load_train_data("./WechatData/train.json")
    train_dataset = MyDataset(train_data, tokenizer=tokenizer)
    eval_dataset = MyDataset(train_data, tokenizer=tokenizer)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=train_dataset,  # 将 eval_dataset 参数设置为 None
        args=training_args,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )

    trainer.train()

    save_tuned_parameters(model, os.path.join("./output", "chatglm-6b-lora.pt"))
    print("done!!!")


def eval():
    checkpoint = "THUDM/chatglm-6b"
    revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
    model = AutoModel.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)

    model = load_lora_config(model)
    model.load_state_dict(torch.load(f"./output/chatglm-6b-lora.pt"), strict=False)

    model.half().cuda().eval()
    response, history = model.chat(tokenizer, "AI探险家的颜值如何？", history=[])
    print(response)


"""
s: 130001, 130004,
tar: 130005
"""

if __name__ == '__main__':
    train()

    eval()
