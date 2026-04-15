import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output dict maps from verdict string to integer class
VERDICT_MAP = {
    "poor": 0,
    "acceptable": 1,
    "good": 2,
    "excellent": 3
}

class CriticDataset(Dataset):
    def __init__(self, data_path, tokenizer_name="microsoft/MiniLM-L12-H384-uncased", max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        answer = item['answer']
        
        # Combine query and answer with SEP token
        text = f"Query: {query} [SEP] Answer: {answer}"
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        score = item['feedback']['score']
        verdict = item['feedback']['verdict']
        verdict_idx = VERDICT_MAP.get(verdict, 0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'score': torch.tensor(score, dtype=torch.float32),
            'verdict': torch.tensor(verdict_idx, dtype=torch.long)
        }

class LearnedCriticModel(nn.Module):
    def __init__(self, model_name="microsoft/MiniLM-L12-H384-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head for Verdict (4 classes)
        self.verdict_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4)
        )
        
        # Regression head for Score (0 to 10)
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() 
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        verdict_logits = self.verdict_head(cls_embedding)
        
        # Sigmoid gives 0-1, multiply by 10 for 0-10 score range
        raw_score = self.score_head(cls_embedding).squeeze(-1) * 10.0
        
        return verdict_logits, raw_score

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Init Data
    dataset = CriticDataset(args.data_path, args.model_name)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Init Model
    model = LearnedCriticModel(args.model_name).to(device)
    
    # Loss functions & Optimizer
    criterion_verdict = nn.CrossEntropyLoss()
    criterion_score = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Alpha controls weight of regression vs classification (0.5 is equal weight conceptually, but scales differ)
    alpha = args.alpha
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_ce = 0
        total_mse = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_score = batch['score'].to(device)
            target_verdict = batch['verdict'].to(device)
            
            verdict_logits, pred_score = model(input_ids, attention_mask)
            
            loss_verdict = criterion_verdict(verdict_logits, target_verdict)
            loss_score = criterion_score(pred_score, target_score)
            
            loss = loss_verdict + alpha * loss_score
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_verdict.item()
            total_mse += loss_score.item()
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} (CE: {total_ce/len(train_loader):.4f}, MSE: {total_mse/len(train_loader):.4f})")
        
        # Simple Validation Logging
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_score = batch['score'].to(device)
                target_verdict = batch['verdict'].to(device)
                
                verdict_logits, pred_score = model(input_ids, attention_mask)
                loss_verdict = criterion_verdict(verdict_logits, target_verdict)
                loss_score = criterion_score(pred_score, target_score)
                val_loss += (loss_verdict + alpha * loss_score).item()
                
        logger.info(f"==> Val Loss: {val_loss / len(val_loader):.4f}")
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {args.output_dir}")
    torch.save(model.state_dict(), out_dir / "learned_critic.pt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/critic_train.json")
    parser.add_argument("--output_dir", type=str, default="trained_models/learned_critic")
    parser.add_argument("--model_name", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for MSE loss")
    args = parser.parse_args()
    
    train(args)
