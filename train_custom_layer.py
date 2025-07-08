import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Assuming qwen_with_custom_feature.py and custom_feature_layer.py are accessible
from qwen_with_custom_feature import QwenWithCustomFeature

# --- Configuration ---
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # Replace with actual model if different or local path
                                                # Note: For a real run, you need the model weights downloaded or accessible.
                                                # For a dry run without actual model weights, this might fail at from_pretrained.
                                                # You might need a dummy config/model for pure script testing if weights are unavailable.

CUSTOM_FEATURE_DIM = 128  # Example: Dimension of your raw custom feature vector
IS_SEQUENCE_LEVEL_FEATURE = True  # True if one feature vector per sequence, False if per token
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2 # Keep small for example
SEQ_LEN = 32 # Example sequence length
VOCAB_SIZE = 151936 # From Qwen2.5-VL config.json, adjust if using a different tokenizer/model's vocab
TEXT_HIDDEN_SIZE = 2048 # From Qwen2.5-VL config.json (text_config.hidden_size)

# A very basic dummy dataset for demonstration
class DummyCustomDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, custom_dim, is_sequence_level):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.custom_dim = custom_dim
        self.is_sequence_level = is_sequence_level

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones_like(input_ids)
        if self.is_sequence_level:
            custom_feature = torch.randn(self.custom_dim)
        else:
            custom_feature = torch.randn(self.seq_len, self.custom_dim)

        # Create labels (e.g., shifted input_ids for language modeling)
        # For Qwen, the loss is typically calculated on non-padded tokens.
        # -100 is the ignore_index for CrossEntropyLoss by default.
        labels = input_ids.clone() # In a real LM task, labels are often shifted.
        # For simplicity, let's assume direct prediction for this dummy, or shift:
        # labels[:-1] = input_ids[1:]
        # labels[-1] = -100 # Ignore the last token's prediction if shifted

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "custom_feature": custom_feature,
            "labels": labels
        }

def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Instantiation ---
    print("Instantiating the model...")
    # For this script to run without actual model download, you might need to use a local dummy config
    # or ensure you have the model. If from_pretrained fails, the script stops.
    try:
        model = QwenWithCustomFeature(
            qwen_model_name_or_path=QWEN_MODEL_NAME,
            custom_feature_dim=CUSTOM_FEATURE_DIM,
            is_custom_feature_sequence_level=IS_SEQUENCE_LEVEL_FEATURE
        )
        model.to(device)
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        print("This script expects the Qwen model to be available.")
        print("For a dry run, you might need to mock from_pretrained or use a local dummy model/config.")
        return

    # --- Freeze Qwen Parameters & Set Projector Trainable ---
    print("Freezing Qwen model parameters...")
    for name, param in model.qwen_model.named_parameters():
        param.requires_grad = False

    print("Ensuring custom feature projector parameters are trainable...")
    for name, param in model.feature_projector.named_parameters():
        param.requires_grad = True
        # print(f"Trainable: {name}")


    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    # print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")


    if not trainable_params:
        print("No trainable parameters found. Check requires_grad settings.")
        return

    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # --- DataLoader ---
    print("Setting up DataLoader...")
    # Use VOCAB_SIZE from the actual model config if possible
    # For Qwen2VLForConditionalGeneration, config.text_config.vocab_size
    actual_vocab_size = model.config.text_config.vocab_size if hasattr(model.config, 'text_config') else VOCAB_SIZE

    dummy_dataset = DummyCustomDataset(
        num_samples=20, # Small number of samples for demo
        seq_len=SEQ_LEN,
        vocab_size=actual_vocab_size,
        custom_dim=CUSTOM_FEATURE_DIM,
        is_sequence_level=IS_SEQUENCE_LEVEL_FEATURE
    )
    train_dataloader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE)

    # --- Training Loop ---
    print("Starting training loop...")
    model.train()  # Set the model to training mode (affects dropout, batchnorm, etc.)
                # Although only projector is trained, good practice.

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            custom_feature = batch["custom_feature"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                custom_feature=custom_feature,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True # Ensure output is a dict-like object with 'loss'
            )

            loss = outputs.loss

            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (batch_idx + 1) % 5 == 0: # Print every 5 batches
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            else:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1} - No loss returned from model.")

        avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    print("Training finished.")
    print(f"Final parameters of the feature projector (first few values of weight):")
    if hasattr(model.feature_projector, 'projection_layer'):
        print(model.feature_projector.projection_layer.weight.data.flatten()[:5])


if __name__ == "__main__":
    # This is a placeholder to prevent actual execution in some environments
    # For a real run, you'd call main() directly.
    # To run this: python train_custom_layer.py
    # Ensure you have transformers, torch, and your custom Python files.
    # Also, ensure the Qwen model is accessible (e.g., internet or local cache).
    print("To run the training script, execute: python train_custom_layer.py")
    # For demonstration, we are not calling main() here to prevent actual model download
    # and training in a non-interactive environment.
    # If you want to run it, uncomment the next line:
    # main()
