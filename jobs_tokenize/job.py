#!/usr/bin/env python
# coding: utf-8

# In[114]:


# In[ ]:
from google.cloud import secretmanager
from google.api_core import exceptions


def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """
    Hämtar en secret från Secret Manager och returnerar den som sträng.
    Exempel: get_secret("my-project", "my-secret")
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    try:
        response = client.access_secret_version(request={"name": name})
    except exceptions.NotFound:
        raise RuntimeError(f"Secret eller version hittades inte: {name}")
    except Exception as e:
        raise RuntimeError(f"Fel vid åtkomst av secret: {e}")

    # payload kan vara bytes — här dekodar vi till sträng
    payload_bytes = response.payload.data
    return payload_bytes.decode("utf-8")


project = "dan-data-eng-20-ce03"
secret = "hf-token"
value = get_secret(project, secret)
print("Secret value:", value)

import os

os.environ["HF_TOKEN"] = value


print(value)


import argparse


def get_dataset_shard():
    parser = argparse.ArgumentParser(description="Process audio dataset shard")
    parser.add_argument(
        "--dataset-shard",
        type=str,
        required=True,
        help="Dataset shard name (e.g., dataset-000000)",
    )
    args = parser.parse_args()
    return args.dataset_shard


dataset_shard = get_dataset_shard()


# In[ ]:


import locale

locale.getpreferredencoding = lambda do_setlocale=False: "UTF-8"
import torch
from snac import SNAC


model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model.to("cuda")


import torchaudio.transforms as T
from datasets import Audio


def tokenise_audio(
    waveform,
):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    # resample_transform = T.Resample(orig_freq=24000, new_freq=24000)
    # waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to("cuda")
    # generate the codes from snac
    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

    return all_codes


# In[ ]:


from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

df = load_dataset(
    "cubbk/audio_swedish_2_dataset_cleaned",
    split="train",
    streaming=True,
    data_dir="8sidor_text_speech_dataset",
    data_files=f"{dataset_shard}.tar",
)

tokenised_df_arr = []

for i, dataset_item in tqdm(enumerate(df)):
    wf = np.asarray(dataset_item["wav"]["array"], dtype=np.float32)
    codes_list = tokenise_audio(wf)
    tokenised_df_arr.append({"codes_list": codes_list, "txt": dataset_item["txt"]})

tokenized_df = pd.DataFrame(tokenised_df_arr)
print(tokenized_df)


# In[ ]:


from transformers import AutoTokenizer
import os

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
num_proc = os.cpu_count() - 2

# Fix: Use boolean indexing instead of .filter() for pandas DataFrame
ds = tokenized_df[tokenized_df["codes_list"].notna()]
ds = ds[ds["codes_list"].apply(lambda x: len(x) > 0)]

# Convert to datasets.Dataset for the map operations
from datasets import Dataset

ds = Dataset.from_pandas(ds)


# @title Create Input Ids
def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i : i + 7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example


ds = ds.map(remove_duplicate_frames, num_proc=num_proc)

tok_info = """*** HERE you can modify the text prompt
i.e. if you wanted a multispeaker model like canopylabs/orpheus-3b-0.1-ft, you can pass:
f"{example["source"]}:  {example["text"]}", as is passed.
"""
print(tok_info)


def create_input_ids(example):
    text_ids = tokenizer.encode(example["txt"], add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example


ds = ds.map(create_input_ids, num_proc=num_proc, remove_columns=["txt", "codes_list"])

ds


# In[ ]:


columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

ds = ds.remove_columns(columns_to_remove)

ds

# ds.push_to_hub(
#     "cubbk/audio_swedish_2_dataset_cleaned",  # Your existing dataset repo
#     config_name="8sidor_tokenized",  # Optional: create a separate config
#     split="train",
#     commit_message="Add tokenized parquet file",
#     create_pr=True  # Set to True if you want to create a pull request instead
# )

df_to_save = ds.to_pandas()
df_to_save.to_parquet(f"{dataset_shard}_tokenized.parquet")


# In[ ]:


def upload_to_gcs():
    os.system(
        f"gsutil cp {dataset_shard}_tokenized.parquet gs://audio_swedish_2/tokenized/{dataset_shard}_tokenized.parquet"
    )


upload_to_gcs()
