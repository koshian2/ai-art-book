import torch
import torch.utils.data
import torchmetrics
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class CLIPAdapter(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(512)
        self.adapter1 = torch.nn.Linear(512, 128, bias=False)
        self.adapter2 = torch.nn.Linear(128, 512, bias=False)
        self.logit_scale = 100
        self.ratio = ratio

    def forward(self, img, text):
        x = self.bn(img)
        # adapter path
        adapter = F.relu(self.adapter1(x))
        adapter = F.relu(self.adapter2(adapter))
        x = x * (1-self.ratio) + adapter * self.ratio
        x = x / x.norm(dim=-1, keepdim=True)

        logit = self.logit_scale * x @ text.T
        return logit

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

        model = CLIPAdapter(ratio)
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
# all tensor(0.7394)
# {1: tensor(0.4995), 5: tensor(0.6122), 10: tensor(0.6622), 'all': tensor(0.7394)}

if __name__ == "__main__":
    main(0.2)