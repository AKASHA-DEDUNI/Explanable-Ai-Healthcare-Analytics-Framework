import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Config ---
DATA_CSV = r"C:\Users\ASUS\Desktop\pneumonia\pneumonia_images\balanced_labels.csv"
IMAGE_ROOT = r"C:\Users\ASUS\Desktop\pneumonia\pneumonia_images"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, image_root, text_model):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.text_model = text_model

        self.label_encoder = LabelEncoder()
        self.df['label_enc'] = self.label_encoder.fit_transform(self.df['Label'])

        self.df['Patient Age'] = self.df['Patient Age'].apply(self._parse_age)
        self.df['Patient Gender'] = self.df['Patient Gender'].map({'M': 0, 'F': 1}).fillna(-1)

        self.scaler = StandardScaler()
        self.df[['Patient Age', 'Patient Gender']] = self.scaler.fit_transform(
            self.df[['Patient Age', 'Patient Gender']])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Precompute text embeddings
        self.text_embeddings = self.text_model.encode(self.df['Report'].tolist(), show_progress_bar=True)
        self.text_embeddings = torch.tensor(self.text_embeddings, dtype=torch.float32)

    def _parse_age(self, age_str):
        try:
            return float(age_str)
        except:
            digits = ''.join(filter(str.isdigit, str(age_str)))
            return float(digits) if digits else 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Label'], row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        meta_vals = row[['Patient Age', 'Patient Gender']].values.astype(float)
        metadata = torch.tensor(meta_vals, dtype=torch.float32)

        text_emb = self.text_embeddings[idx]
        label = torch.tensor(row['label_enc'], dtype=torch.long)

        return image, metadata, text_emb, label

# --- Model ---
class MultiModalPneumoniaModel(nn.Module):
    def __init__(self, metadata_input_dim=2, text_embedding_dim=384):
        super().__init__()
        # ResNet50 backbone
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_output_dim = 2048
        self.cnn.fc = nn.Identity()

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Text branch
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Combined classifier
        combined_dim = self.cnn_output_dim + 32 + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, image, metadata, text_emb):
        img_feat = self.cnn(image)
        meta_feat = self.meta_fc(metadata)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([img_feat, meta_feat, text_feat], dim=1)
        logits = self.classifier(combined)
        return logits

# --- Training & Validation ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    losses = []
    all_preds, all_labels, all_probs = [], [], []

    for image, meta, text_emb, label in tqdm(dataloader):
        image, meta, text_emb, label = image.to(DEVICE), meta.to(DEVICE), text_emb.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(image, meta, text_emb)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return sum(losses)/len(losses), acc, precision, recall, f1, auc

def eval_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for image, meta, text_emb, label in tqdm(dataloader):
            image, meta, text_emb, label = image.to(DEVICE), meta.to(DEVICE), text_emb.to(DEVICE), label.to(DEVICE)
            outputs = model(image, meta, text_emb)
            loss = criterion(outputs, label)
            losses.append(loss.item())

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return sum(losses)/len(losses), acc, precision, recall, f1, auc

# --- Main ---
def main():
    print("Loading sentence-transformers model for text embeddings...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')

    dataset = PneumoniaDataset(DATA_CSV, IMAGE_ROOT, text_model)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiModalPneumoniaModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = eval_epoch(model, val_loader, criterion)
        print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_model_resnet50.pth')
            print("Model saved!")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
