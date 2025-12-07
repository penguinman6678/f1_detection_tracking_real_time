import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np



def vit_create_loaders(data_dir=None, batch_size=32, weighted=False, seed=21):

    vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit_model_transforms = vit_weights.transforms()

    f1_dataset_complete = datasets.ImageFolder(root=data_dir, transform=vit_model_transforms)

    # Sanity check to ensure we got all the teams from the data dir
    f1_class_names = f1_dataset_complete.classes
    f1_num_classes = len(f1_class_names)
    assert f1_num_classes == 10 # as of 2024/2025 race years
    print(f"List of F1 Teams based: {f1_class_names}")

    # Split to train/val/test
    dataset_len = len(f1_dataset_complete)
    train_len = int(dataset_len * 0.8)
    val_len = int(dataset_len * 0.1)
    test_len = dataset_len - train_len - val_len # int(dataset_len * 0.15)

    train_set, val_set, test_set = random_split(f1_dataset_complete, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    # Want to consider class imbalance
    weighted_sampler = None
    if weighted == True:

        list_train_team_labels = [f1_dataset_complete.samples[idx][1] for idx in train_set.indices]
        team_counts = torch.zeros(f1_num_classes, dtype=torch.long)
        for c in list_train_team_labels:
            team_counts[c] += 1
        
        team_counts[team_counts == 0] = 1
        team_weights = 1.0 / team_counts.float()

        list_team_labels_weights = [team_weights[c] for c in list_train_team_labels]

        weighted_sampler = WeightedRandomSampler(list_team_labels_weights, num_samples=len(list_team_labels_weights),
                                                 replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, sampler=weighted_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, val_loader, test_loader

def modify_vit():

    vit_model = vit_b_16(weights=None)

    for param in vit_model.parameters():
        param.requires_grad = False
    
    for param in vit_model.heads.parameters():
        param.requires_grad = True
    
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, 10)

    return vit_model


def train_single_epoch(model, train_loader, optimizer, criterion, device):

    model.train()
    total_loss = 0
    total_corr = 0
    total = 0


    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = predictions.max(1)
        total_corr += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, total_corr / total

def eval_val_epoch(model, val_loader, optimizer, criterion, device):

    model.eval()
    total_loss = 0
    total_corr = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            _, preds = predictions.max(1)
            total_corr += (preds == labels).sum().item()
            total += images.size(0)
    
    return total_loss / total, total_corr / total

def load_vit_model(model_dir, device):

    vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit_transforms = vit_weights.transforms()

    checkpoint_state_dict = torch.load(model_dir)
    # team_names = checkpoint_state_dict["classes"]

    vit_model = vit_b_16(weights=None)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, 10)
    vit_model.load_state_dict(checkpoint_state_dict)
    vit_model.to(device)
    
    return vit_model

def eval_test_show_examples(model, test_loader, save_path, device, K=10):

    team_names = ["Alpine", "AshtonMartin", "Ferrari", "Haas", "Kick", "McLaren", "Mercedes", "RacingBull", "RedBull", "Williams"]

    model.eval()
    total = 0
    total_corr = 0
    

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            _, preds = predictions.max(1)

            # print(f"Test Accuracy: {total_corr / total}")

            K = min(K, images.size(0))
            for idx in range(K):
                
                img = images[idx].cpu().permute(1,2,0).numpy()
                team_label = team_names[labels[idx]]
                predicted = team_names[preds[idx]]

                plt.figure(figsize=(3,3))
                plt.imshow(img)
                plt.title(f"Ground Truth: {team_label} | Predicted: {predicted}")
                plt.axis("off")
                plt.savefig(f"{save_path}/test_{idx}")
                plt.close()
            
            break

def eval_test_examples(model, test_loader, device):

    team_names = ["Alpine", "AshtonMartin", "Ferrari", "Haas", "Kick", "McLaren", "Mercedes", "RacingBull", "RedBull", "Williams"]

    model.eval()
    total = 0
    total_corr = 0
    

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            _, preds = predictions.max(1)

            total_corr += (preds == labels).sum().item()
            total += images.size(0)
    
    test_acc = total_corr / total
    print(f"Test Accuracy: {test_acc}")
    return test_acc


def plot_figures(output_dir, num_epochs, arr, split, label):

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, arr, label=f"{split} {label}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{label}")
    plt.title(f"{label} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}.png")
    plt.close

    print(f"Saved at {output_dir}.png")

def generate_confusion_matrix(model, test_loader, device, output_dir, normalize=True):

    # Based on file order in f1_teams/classification
    team_names = ["Alpine", "AshtonMartin", "Ferrari", "Haas", "Kick", "McLaren", "Mercedes", "RacingBull", "RedBull", "Williams"]
    predictions = []
    gt_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            _, preds = preds.max(1)

            predictions.append(preds.cpu())
            gt_labels.append(labels.cpu())
    
    predictions = torch.cat(predictions).numpy()
    gt_labels = torch.cat(gt_labels).numpy()

    conf_matrix = confusion_matrix(gt_labels, predictions, labels=np.arange(len(team_names)))

    if normalize == True:

        conf_matrix_sum = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_sum[conf_matrix_sum == 0] = 1
        conf_matrix = conf_matrix.astype("float") / conf_matrix_sum

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")

    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(team_names)),
           yticks=np.arange(len(team_names)),
           xticklabels=team_names,
           yticklabels=team_names,
           ylabel="Ground Truth Team Names",
           xlabel="Predicted Team Names",
           title="Confusion Matrix" 
        )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    style = ".2f" if normalize else "d"
    threshold = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(conf_matrix[i, j], style),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > threshold else "black",
                fontsize=7
            )
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix to {output_dir}.png")



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 200
    best_val_acc = 0.0 # for early stopping 
    output_dir = f"./vit_scratch_runs/vit_scratch_{num_epochs}.pth"

    # # prob smart to do some weighted loss to handle imbalance
    # team_counts = [56, 70, 109, 63, 94, 123, 112, 69, 124, 96]
    # team_weights = 1 / torch.tensor(team_counts, dtype=torch.float)
    # team_weights = team_weights / team_weights.sum()
    
    data_dir = "../data/f1_teams/classification"
    train_loader, val_loader, test_loader = vit_create_loaders(data_dir=data_dir, batch_size=32, weighted=True, seed=21)
    # vit_model = modify_vit()
    # vit_model.to(device)

    # criterion = nn.CrossEntropyLoss(weight=team_weights.to(device))
    # optimizer = torch.optim.Adam(vit_model.heads.parameters(), lr=0.0001, weight_decay=0.001)

    # train_losses = []
    # val_losses = []
    # train_accs = []
    # val_accs = []

    # print("Begin Training")
    # for epoch in range(num_epochs):

    #     train_loss, train_acc = train_single_epoch(vit_model, train_loader, optimizer, criterion, device)

    #     # Metrics for training
    #     train_losses.append(train_loss)
    #     train_accs.append(train_acc)
    #     print(f"Epoch {epoch + 1} / {num_epochs} -- Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
        
    #     if (epoch + 1) % 10 == 0:
    #         val_loss, val_acc = eval_val_epoch(vit_model, val_loader, optimizer, criterion, device)
            
    #         # Metrics for validation
    #         val_losses.append(val_loss)
    #         val_accs.append(val_acc)
    #         print(f"Epoch {epoch + 1} / {num_epochs} -- Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")
            
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             torch.save(vit_model.state_dict(), output_dir)
    #             print(f"New model saved with val accuracy of {val_acc:.3f} at {output_dir}")

    # print("End Training")

    # train_metrics_df = pd.DataFrame({
    #     "num_epochs": num_epochs,
    #     "train_losses": train_losses,
    #     "train_accs": train_accs
    # })

    # val_metrics_df = pd.DataFrame({
    #     "num_epochs": num_epochs,
    #     "val_losses": val_losses,
    #     "val_accs": val_accs
    # })

    # train_metrics_df.to_csv(f"vit_scratch_{num_epochs}_train_metrics.csv", index=False)
    # val_metrics_df.to_csv(f"vit_scratch_{num_epochs}_val_metrics.csv", index=False)

    # # Plot the metrics and save
    # train_loss_dir = f"./vit_scratch_runs/figures/train_loss_vit_{num_epochs}"
    # plot_figures(train_loss_dir, num_epochs, train_losses, "train", "loss")
    # val_loss_dir  = f"./vit_scratch_runs/figures/val_loss_vit_{num_epochs}"
    # plot_figures(val_loss_dir, num_epochs // 10, val_losses, "val", "loss")

    # train_accs_dir = f"./vit_scratch_runs/figures/train_accs_vit_{num_epochs}"
    # plot_figures(train_accs_dir, num_epochs, train_accs, "train", "acc")
    # val_accs_dir = f"./vit_scratch_runs/figures/val_accs_vit_{num_epochs}"
    # plot_figures(val_accs_dir, num_epochs // 10, val_accs, "val", "acc")

    
    model_dir = "./vit_scratch_runs/vit_scratch_200.pth"
    save_path = "./vit_scratch_tracked_examples_on_test"
    vit_model = load_vit_model(model_dir, device)
    # eval_test_show_examples(vit_model, test_loader, save_path, device, K=10)
    # test_acc_vit_100 = eval_test_examples(vit_model, test_loader, device)
    cm_save_path = "./vit_scratch_cm"
    generate_confusion_matrix(vit_model, test_loader, device, cm_save_path)
