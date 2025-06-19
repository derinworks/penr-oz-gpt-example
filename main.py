import os
import time
from typing import Tuple
from requests import Response
import requests
from dotenv import load_dotenv

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
    return ([source_data[i:i + block_size] for i in range(end)],
            [source_data[i + 1:i + block_size + 1] for i in range(end)])

def request_prediction_progress(delay_secs = 0, timeout_secs = 1, log_prefix = "") -> Response:
    # keep requesting until condition met or times out
    for _ in range(timeout_secs // max(1, delay_secs)):
        if delay_secs > 0: # wait for progress to build up
            time.sleep(delay_secs)
        # check progress
        progress_resp = requests.get(f"{prediction_server_url}/progress/", params=model_request)
        progress_status, progress_body = progress_resp.status_code, progress_resp.json()
        print(f"{log_prefix}{progress_status=}")
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
            print(f"{progress_body=}")
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

def calculate_cost(eval_epochs: int, eval_batch_size: int, input_data: list[list[int]], target: list[list[int]]) -> float:
    num_eval_items = len(input_data)
    print(f"Evaluate cost for data of size {num_eval_items} to average over {eval_epochs} epochs "
          f"with batch size {eval_batch_size}")

    cost_request = model_request | {
        "input": input_data,
        "target": target,
        "epochs": eval_epochs,
        "batch_size": eval_batch_size,
    }
    resp = requests.post(f"{prediction_server_url}/evaluate/", json=cost_request)

    if resp.status_code == 200:
        cost = resp.json()['cost']
        return cost
    else:
        raise RuntimeError(f"Failed to calculate cost: {resp.status_code} - {resp.json()}")

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
    print(f"Prepared training data of size {num_train_items} to run for {training_epochs} epochs "
          f"with batch size {train_batch_size}")

    # Submit training request to prediction service
    training_resp = requests.put(f"{prediction_server_url}/train/", json=training_request)
    print(f"Submitted: {training_resp.status_code} - {training_resp.json()}")
    # check progress
    delay_secs = 60 if device_selection == 'cuda' else 5
    print(f"{delay_secs=}")
    timeout_secs = max(training_epochs, 10) * (5 if device_selection == 'cuda' else 1)
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
    # Device selection
    device_selection = input('Choose device: (default: cpu)') or 'cpu'
    print(f"{device_selection=}")

    # Configure block context size
    block_size = 128 if device_selection == 'cuda' else 64
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
        # embedding depth number of dimensions
        embed_depth = 192 if device_selection == 'cuda' else 96
        print(f"{embed_depth=}")
        # number of attention heads
        attn_heads = 3 if device_selection == 'cuda' else 2
        print(f"{attn_heads=}")
        # number of transformer layers
        tran_layers = 3 if device_selection == 'cuda' else 2
        print(f"{tran_layers=}")
        # drop out ratio
        dropout = 0.2 if device_selection == 'cuda' else 0.05
        print(f"{dropout=}")
        # learning rate
        learning_rate = 3e-4
        print(f"{learning_rate=}")
        # init parameters config
        init_w = {"normal": {"mean": 0.0, "std": 0.02}}
        print(f"{init_w=}")
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
                        {"attention": {"embedding_dim": embed_depth, "num_heads": attn_heads, "block_size": block_size,
                                       "bias": False, "dropout": dropout}} | init_w]
                    },
                    {"sequential": [
                        {"layernorm": {"normalized_shape": embed_depth}},
                        {"linear": {"in_features": embed_depth, "out_features": 4 * embed_depth}} | init_w | init_b,
                        {"gelu": {}},
                        {"linear": {"in_features": 4 * embed_depth, "out_features": embed_depth}} | init_w | init_b,
                        {"dropout": {"p": dropout}}]
                    }]}
                ] * tran_layers +
                [{"layernorm": {"normalized_shape": embed_depth}},
                 {"linear": {"in_features": embed_depth, "out_features": vocab_size}},
                 {"softmaxlast": {"dim": -1}}],
            "optimizer": {
                "adamw": {"lr": learning_rate}
            },
            "device": device_selection,
        }
        create_model_resp = requests.post(f"{prediction_server_url}/model/", json=create_model_request)
        print(f"{create_model_resp.status_code} - {create_model_resp.json()}")
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
        split_train_data = encoded_example[:num_split_train_items] * (5 if device_selection == 'cuda' else 1)
        print(f"{len(split_train_data)=}")
        split_val_data = encoded_example[num_split_train_items:]
        # Build training data
        input_train, target_train = make_training_data(split_train_data)
        input_val, target_val = make_training_data(split_val_data)
        # Preview training data
        for x, y in zip(input_train[:5], target_train[:5]):
            print(f"{''.join(vocabulary[ix] for ix in x)} --> {''.join(vocabulary[iy] for iy in y)}")

        # Ask for training options
        num_training_epochs = int(input('How many epochs shall we perform training? (default: 1)') or 1)
        print(f"{num_training_epochs=}")
        batch_size = int(input('Set batch size=(default: 64)') or 64)
        print(f"{batch_size=}")

        # Run training on split
        run_training(num_training_epochs, batch_size, input_train, target_train)

        # Calculate cost on splits
        num_eval_epochs = max(1, num_training_epochs // 10)
        split_train_cost = calculate_cost(num_eval_epochs, batch_size, input_train, target_train)
        print(f"{split_train_cost=}")
        split_val_cost = calculate_cost(num_eval_epochs, batch_size, input_val, target_val)
        print(f"{split_val_cost=}")

    else: # Generate sample
        # Ask for number of maximum tokens
        num_new_tokens_requested = int(input('How many new tokens at most would you like? (default: 10)') or 10)
        print(f"{num_new_tokens_requested=}")
        # Generate tokens
        encoded_sample = generate([[0]], num_new_tokens_requested)
        # Present generated decoded sample
        decoded_sample = ''.join([vocabulary[i] for i in encoded_sample])
        print(decoded_sample)
