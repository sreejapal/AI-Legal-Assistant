import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
import evaluate
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# ========== 1️⃣ Load Dataset ==========
df = pd.read_csv(r"C:\minor_1\dataset\cleaned_translation_data.csv").dropna().reset_index(drop=True)
print(f"✅ Loaded {len(df)} sentence pairs")

# ========== 2️⃣ Split Dataset ==========
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# reset indices to avoid KeyError
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"📊 Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

# ========== 3️⃣ Dataset Class ==========
class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, source_col="hi", target_col="en", max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.source_col = source_col
        self.target_col = target_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # use iloc to avoid KeyError
        source = self.df.iloc[idx][self.source_col]
        target = self.df.iloc[idx][self.target_col]

        source_enc = self.tokenizer(
            source, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }

# ========== 4️⃣ Load Model & Tokenizer ==========
model_name = "Helsinki-NLP/opus-mt-hi-en"
save_path = r"C:\minor_1\finetuned_hi_en_lora"

tokenizer = MarianTokenizer.from_pretrained(model_name)

if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "adapter_config.json")):
    print("🔄 Found previous checkpoint. Loading fine-tuned model...")
    base_model = MarianMTModel.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, save_path)
else:
    print("🆕 No checkpoint found. Starting fresh fine-tuning...")
    model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"🧠 Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.2%})")

# ========== 5️⃣ Dataloaders ==========
train_loader = DataLoader(TranslationDataset(train_df, tokenizer), batch_size=8, shuffle=True)
val_loader = DataLoader(TranslationDataset(val_df, tokenizer), batch_size=8)
test_loader = DataLoader(TranslationDataset(test_df, tokenizer), batch_size=8)

# ========== 6️⃣ Optimizer ==========
optimizer = AdamW(model.parameters(), lr=1e-4)

# ========== 7️⃣ Training Loop ==========
epochs = 10
patience = 2
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    start_time = time.time()
    print(f"\n🚀 Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=True)
    for step, batch in enumerate(progress):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"batch_loss": loss.item(), "avg_loss": total_loss/(step+1)})

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Average Training Loss: {avg_train_loss:.4f}")

    # ----- Validation -----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=True)
        for batch in val_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    val_loss /= len(val_loader)
    print(f"📉 Validation Loss: {val_loss:.4f} (epoch time: {(time.time()-start_time)/60:.2f} min)")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        print(f"💾 Model improved & saved to {save_path}")
    else:
        patience_counter += 1
        print(f"⏸ No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("⏹ Early stopping triggered!")
            break

# ========== 8️⃣ Evaluation ==========
metric = evaluate.load("sacrebleu")
model.eval()
translations, references = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        preds = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        targets = [tokenizer.decode(l[l != -100], skip_special_tokens=True) for l in batch["labels"]]
        translations.extend(preds)
        references.extend([[t] for t in targets])

bleu = metric.compute(predictions=translations, references=references)
print(f"\n🎯 Test BLEU Score: {bleu['score']:.2f}")
