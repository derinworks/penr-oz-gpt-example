import gzip
from io import BytesIO
import json
import math
import os
import time
from typing import Tuple
from requests import Response
import requests
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# load any env file
load_dotenv()

# Prepare Prediction server config
prediction_server_url = os.environ.get("PREDICTION_SERVER_URL", "http://127.0.0.1:8000")
print(f"{prediction_server_url=}")
model_request = {
    "model_id": "gpt-example"
}

def make_training_data(source_data: list[int]) -> Tuple[list[list[int]], list[list[int]]]:
    end = len(source_data) - block_size
    return ([source_data[i:i + block_size] for i in tqdm(range(end), desc="input")],
            [source_data[i + 1:i + block_size + 1] for i in tqdm(range(end), desc="target")])

def request_prediction_progress(delay_secs = 0, timeout_secs = 1, log_prefix = "") -> Response:
    # keep requesting until condition met or times out
    for _ in range(timeout_secs // max(1, delay_secs)):
        if delay_secs > 0: # wait for progress to build up
            time.sleep(delay_secs)
        # check progress
        progress_resp = requests.get(f"{prediction_server_url}/progress/", params=model_request)
        progress_status, progress_body = progress_resp.status_code, progress_resp.json()
        log.info(f"{log_prefix}{progress_status=}")
        if progress_resp.status_code == 200:
            if len(progress_body["progress"]) > 0: # log info about progress
                costs = [progress["cost"] for progress in progress_body["progress"]]
                cost = sum(costs) / len(costs)
                avg_cost = progress_body["average_cost"]
                print(f"{cost=}")
                print(f"{avg_cost=}")
                last_epoch = progress_body["progress"][-1]["epoch"]
                print(f"{last_epoch=}")
            model_status = progress_body["status"]
            print(f"{model_status=}")
            if model_status == "Training":
                continue # checking
        else: # barf possible error body
            log.error(f"{progress_body=}")
            if progress_resp.status_code != 404: # any error besides not found
                continue # checking
        return progress_resp # done
    # timed out
    raise TimeoutError(f"{log_prefix} took too long")

def make_prediction(input_data: list[list[int]]) -> list[list[float]]:
    prediction_request = model_request | {
        "input": input_data,
    }
    resp = requests.post(f"{prediction_server_url}/output/", json=prediction_request)

    if resp.status_code == 200:
        return resp.json()['output']

    raise RuntimeError(f"Failed to receive a good prediction: {resp.status_code} - {resp.json()}")

def compress_with_progress(data: dict, chunk_size: int = 32 * 2**20) -> bytes:
    # Compress JSON with progress tracking
    encoder = json.JSONEncoder(separators=(",", ":"), ensure_ascii=False)
    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
        bar = tqdm(unit="B", unit_scale=True, desc="Compressing")
        buffer = bytearray()
        bytes_written = 0
        for part in encoder.iterencode(data):
            buffer.extend(part.encode("utf-8"))
            while len(buffer) >= chunk_size:
                gz.write(buffer[:chunk_size])
                bytes_written += chunk_size
                del buffer[:chunk_size]
                bar.update(chunk_size)
        if buffer:
            gz.write(buffer)
        bar.close()
    # show compressed size and return buffer
    compressed_bytes = buf.getvalue()
    compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
    log.info(f"✅ Final compressed size: {compressed_size_mb:.2f} MB")
    return compressed_bytes

def calculate_cost(eval_epochs: int, eval_batch_size: int, input_data: list[list[int]], target: list[list[int]]) -> float:
    num_eval_items = len(input_data)
    log.info(f"Evaluate cost for data of size {num_eval_items} to average over {eval_epochs} epochs "
             f"with batch size {eval_batch_size}")

    cost_request = model_request | {
        "input": input_data,
        "target": target,
        "epochs": eval_epochs,
        "batch_size": eval_batch_size,
    }

    if compress_request:
        log.info(f"Compressing evaluation request ...")
        compressed_cost_request = compress_with_progress(cost_request)
        log.info(f"Compressed cost request for data of size {num_eval_items}")
        cost_resp = requests.post(f"{prediction_server_url}/evaluate/", data=compressed_cost_request,
                              headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})
    else:
        cost_resp = requests.post(f"{prediction_server_url}/evaluate/", json=cost_request)

    if cost_resp.status_code == 200:
        cost = cost_resp.json()['cost']
        return cost
    else:
        raise RuntimeError(f"Failed to calculate cost: {cost_resp.status_code} - {cost_resp.json()}")

def run_training(training_epochs: int, train_batch_size: int, input_data: list[list[int]], target: list[list[int]]):
    # Prepare training request parameters
    num_train_items = len(input_data)
    training_model_request = model_request | {
        "epochs": training_epochs,
        "batch_size": train_batch_size,
    }

    # Prepare training request
    training_request = training_model_request | {
        "input": input_data,
        "target": target,
    }
    log.info(f"Prepared training data of size {num_train_items} to run for {training_epochs} epochs "
             f"with batch size {train_batch_size}")

    # Submit training request to prediction service
    if compress_request:
        log.info(f"Compressing training request ...")
        compressed_training_request = compress_with_progress(training_request)
        log.info(f"Compressed training request for data of size {num_train_items}")
        training_resp = requests.put(f"{prediction_server_url}/train/", data=compressed_training_request,
                                     headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})
    else:
        training_resp = requests.put(f"{prediction_server_url}/train/", json=training_request)
    log.info(f"Submitted: {training_resp.status_code} - {training_resp.json()}")
    # check progress
    delay_secs = 15 * scale_factor
    print(f"{delay_secs=}")
    timeout_secs = max(training_epochs, 15 * 60 * scale_factor)
    print(f"{timeout_secs=}")
    request_prediction_progress(delay_secs, timeout_secs, "Training...")
    # mark end of training request
    print(f"###### Finished Training Round ########")

def generate(input_context: list[list[int]], max_new_tokens: int) -> list[int]:
    print(f"Generating up to {max_new_tokens} new tokens")

    generate_request = model_request | {
        "input": input_context,
        "block_size": block_size,
        "max_new_tokens": max_new_tokens,
    }
    resp = requests.post(f"{prediction_server_url}/generate/", json=generate_request)

    if resp.status_code == 200:
        tokens = resp.json()['tokens']
        return tokens
    else:
        raise RuntimeError(f"Failed to generate tokens: {resp.status_code} - {resp.json()}")

if __name__ == "__main__":
    # User selection
    user_selection = (input('Choose (S) generate samples or (T) perform training: (default: S)') or "S").upper()
    print(f"{user_selection=}")
    # Scale factor
    scale_factor = int(input('Choose scale up: (default: 1)') or 1)
    print(f"{scale_factor=}")

    # Configure block context size
    block_size = 64 * scale_factor
    print(f"{block_size=}")

    # Read example in
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    # Extract vocabulary
    vocabulary = sorted(list(set(example)))
    vocab_size = len(vocabulary)
    print(f"{vocab_size=}")
    s2i = {s: i for i, s in enumerate(vocabulary)}
    print(f"{s2i=}")

    # Create prediction model if not already
    model_resp = request_prediction_progress(log_prefix="Checking...")
    if model_resp.status_code == 404:
        # Device selection
        device_selection = input('Choose device: (default: cpu)') or 'cpu'
        print(f"{device_selection=}")
        # embedding depth number of dimensions
        embed_depth = 96 * scale_factor
        print(f"{embed_depth=}")
        # number of attention heads
        attn_heads = max(2, int(1.5 * scale_factor))
        print(f"{attn_heads=}")
        # number of transformer layers
        tran_layers = max(2, int(1.5 * scale_factor))
        print(f"{tran_layers=}")
        # drop out ratio
        dropout = 0.05 * scale_factor
        print(f"{dropout=}")
        # learning rate
        learning_rate = 3e-4
        print(f"{learning_rate=}")
        # init parameters config
        init_w = {"normal": {"mean": 0.0, "std": 0.02}}
        print(f"{init_w=}")
        init_proj_w = {"normal": {"mean": 0.0, "std": 0.02 / math.sqrt(2 * tran_layers)}}
        init_b = {"zeros": {}}
        print(f"{init_b=}")
        # create model
        create_model_request = model_request | {
            "layers":
                [{"summation": [
                    {"embedding": {"num_embeddings": vocab_size, "embedding_dim": embed_depth}} | init_w,
                    {"position": {"num_embeddings": block_size, "embedding_dim": embed_depth}} | init_w]},
                 {"dropout": {"p": dropout}}] +
                [{"residual": [
                    {"sequential": [
                        {"layernorm": {"normalized_shape": embed_depth}},
                        {"linear": {"in_features": embed_depth, "out_features": 3 * embed_depth}} | init_w | init_b,
                        {"attention": {"num_heads": attn_heads, "dropout": dropout}},
                        {"linear": {"in_features": embed_depth, "out_features": embed_depth}} | init_proj_w | init_b,
                        {"dropout": {"p": dropout}}]
                    },
                    {"sequential": [
                        {"layernorm": {"normalized_shape": embed_depth}},
                        {"linear": {"in_features": embed_depth, "out_features": 4 * embed_depth}} | init_w | init_b,
                        {"gelu": {}},
                        {"linear": {"in_features": 4 * embed_depth, "out_features": embed_depth}} | init_proj_w | init_b,
                        {"dropout": {"p": dropout}}]
                    }
                ]} for _ in range(tran_layers)] +
                [{"layernorm": {"normalized_shape": embed_depth}},
                 {"linear": {"in_features": embed_depth, "out_features": vocab_size, "bias": False}},
                 {"softmaxlast": {"dim": -1}}],
            "optimizer": {
                "adamw": {"lr": learning_rate}
            },
            "device": device_selection,
        }
        create_model_resp = requests.post(f"{prediction_server_url}/model/", json=create_model_request)
        log.info(f"{create_model_resp.status_code} - {create_model_resp.json()}")
    elif model_resp.status_code != 200:
        raise RuntimeError(f"Prediction Service error: {model_resp.status_code} - {model_resp.json()}")

    # Perform according to user selection
    if user_selection == 'T':
        # Encode example
        encoded_example: list[int] = [s2i[c] for c in example]
        # Build data splits
        num_items = len(encoded_example)
        print(f"{num_items=}")
        num_split_train_items = int(0.9 * num_items)
        print(f"{num_split_train_items=}")
        num_split_val_items = int(0.1 * num_items)
        print(f"{num_split_val_items=}")
        split_train_data = encoded_example[:num_split_train_items] * int(scale_factor * 1.25)
        print(f"{len(split_train_data)=}")
        split_val_data = encoded_example[num_split_train_items:]
        # Build training data
        log.info("Making train data split...")
        input_train, target_train = make_training_data(split_train_data)
        log.info("Making value data split...")
        input_val, target_val = make_training_data(split_val_data)
        log.info("Done: making training data")
        preview_training_data = bool((input('Preview training data? (default: N)') or 'N').upper() == 'Y')
        print(f"{preview_training_data=}")
        if preview_training_data: # Preview training data
            for x, y in zip(input_train[:1], target_train[:1]):
                print(f"{''.join(vocabulary[ix] for ix in x)} --> {''.join(vocabulary[iy] for iy in y)}")

        # Ask for training options
        num_training_epochs = int(input('How many epochs shall we perform training? (default: 1000)') or 1000)
        print(f"{num_training_epochs=}")
        batch_size = int(input('Set batch size=(default: 64)') or 64)
        print(f"{batch_size=}")
        compress_request = (input('Compress request payload?(default: Y)') or 'Y').upper() == 'Y'
        print(f"{compress_request=}")

        # Run training on split
        run_training(num_training_epochs, batch_size, input_train, target_train)

        # Calculate cost on splits
        num_eval_epochs = max(1, num_training_epochs // 10)
        split_train_cost = calculate_cost(num_eval_epochs, batch_size, input_train, target_train)
        log.info(f"{split_train_cost=}")
        split_val_cost = calculate_cost(num_eval_epochs, batch_size, input_val, target_val)
        log.info(f"{split_val_cost=}")

    else: # Generate sample
        # Ask for number of maximum tokens
        num_new_tokens_requested = int(input('How many new tokens at most would you like? (default: 500)') or 500)
        print(f"{num_new_tokens_requested=}")
        # Ask for prompt
        prompt = input('Prompt?') or 'Therapist: What would you like to work on today?'
        print(f"{prompt=}")
        encoded_prompt = [[s2i[s] for s in prompt]]
        print(f"{encoded_prompt=}")
        # Generate tokens
        encoded_sample = generate(encoded_prompt, num_new_tokens_requested)
        # Present generated decoded sample
        decoded_sample = ''.join([vocabulary[i] for i in encoded_sample])
        print(f"\n{decoded_sample}\n")
