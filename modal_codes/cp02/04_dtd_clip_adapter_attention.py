import torch
import torch.utils.data
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class AttentionLinearProbe(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.fc = torch.nn.Linear(512, 47)
        self.ratio = ratio
        self.n_sqrt = np.sqrt(512)

    def forward(self, img, text):
        x = self.bn1(img) # (N, C)
        # adapter path
        y = self.bn2(text)
        adapter = x @ y.T # (N, K)
        adapter = torch.softmax(adapter / self.n_sqrt, dim=-1) @ y # (N, C)
        x = x * (1-self.ratio) + adapter * self.ratio
        x = self.bn3(x)
        x = self.fc(x)
        return x
    
def load_dataset(split, n_sample_per_class=None):
    text_data = torch.load("output/class_embedding.pt")
    all_data = torch.load(f"output/image_{split}.pt")

    pickup_indices = []
    y_all = np.array(all_data["class_idx"]).astype(np.int64)
    max_class_idx = y_all.max()
    np.random.seed(1234)
    for i in range(max_class_idx+1):
        indices = np.where(y_all == i)[0]
        np.random.shuffle(indices)
        indices = np.sort(indices[:n_sample_per_class])
        pickup_indices.append(indices)
    pickup_indices = np.concatenate(pickup_indices)

    all_data = {
        "class_names": text_data["class_names"],
        "text": text_data["embeddings"],
        "image": all_data["embeddings"][pickup_indices],
        "class_idx": y_all[pickup_indices]
    }

    dataset = torch.utils.data.TensorDataset(all_data["image"], torch.from_numpy(all_data["class_idx"]))
    return dataset, all_data["text"]

def main(ratio):
    print("--- ratio : ", ratio, "---")
    result = {
        1: 0,
        5: 0,
        10: 0,
        "all": 0
    }

    for key in result.keys():
        n_sample_per_class = key if key != "all" else None

        trainset, text_embedding = load_dataset("train", n_sample_per_class)
        testset, _ = load_dataset("test", None)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

        model = AttentionLinearProbe(ratio)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metric = torchmetrics.Accuracy("multiclass", num_classes=47)
        max_val_acc = 0.0

        for epoch in range(400):
            metric.reset()
            model.train()
            for X_img, y in tqdm(train_loader):
                optimizer.zero_grad()
                y_pred = model(X_img, text_embedding)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                metric(y_pred.argmax(dim=-1).cpu(), y.cpu())
            
            metric.reset()
            model.eval()
            for X_img, y in tqdm(test_loader):
                with torch.no_grad():
                    y_pred = model(X_img, text_embedding)
                    metric(y_pred.argmax(dim=-1).cpu(), y.cpu())
            val_acc = metric.compute()
            max_val_acc = max(val_acc, max_val_acc)
            print(f"Epoch {epoch:03} | Val accuracy : {val_acc}") 

        result[key] = max_val_acc
        print(key, max_val_acc)

    print(result)

# ratio = 0.2
# all tensor(0.7505)
# {1: tensor(0.3761), 5: tensor(0.5856), 10: tensor(0.6580), 'all': tensor(0.7505)}

if __name__ == "__main__":
    main(0.2)