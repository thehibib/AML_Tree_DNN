# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install ucimlrepo

#IMPORTING DATA
from ucimlrepo import fetch_ucirepo

# fetch dataset
covertype = fetch_ucirepo(id=31)

# data (as pandas dataframes)
X = covertype.data.features
y = covertype.data.targets

df = pd.concat([X,y],axis=1)
#randomly sampling
#df = df.sample(n=20000, random_state=42)
df = df.groupby('Cover_Type', group_keys=False).apply(lambda x: x.sample(min(len(x), 500)))

print(df['Cover_Type'].value_counts())  # Check if it's balanced
y=df['Cover_Type']
X=df.drop(columns=['Cover_Type'])
names = covertype.variables.name
y = y-1

# df.to_pickle('/content/even_trees.pkl')
# from google.colab import files

# # # # Download the file to your local machine
# # files.download("/content/even_trees.pkl")
# #print(y)
# print(soils)
# X_scaled

#normalization and splitting data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
soils = [f"Soil_Type{i}" for i in range(1, 41)]
X_soil = X[soils]

X_nosoil = X.drop(columns=soils)

X_soil = scaler.fit_transform(X_soil)
X_nosoil = scaler.fit_transform(X_nosoil)

X_soiltrain, X_soiltest, X_nosoiltrain, X_nosoiltest, y_train, y_test = train_test_split(
    X_soil, X_nosoil, y, test_size=0.2, random_state=42
)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#X_train = X_train.to_numpy()
#X_test = X_test.no_numpy()

print(pd.DataFrame(X_nosoil).head())

#Testing and visualization
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.violinplot(x='Cover_Type', y='Soil_Type30', hue='Cover_Type', data=df)

# df_melted = df.melt(id_vars=["Cover_Type"], var_name="Feature", value_name="Value")

# # Plot violin plot with flipped axes
# plt.figure(figsize=(20, 20))
# sns.violinplot(y="Feature", x="Value", hue="Cover_Type", data=df_melted, palette="tab10", split=True, orient="h")

# plt.title("Feature Distributions by Cover_Type")
# plt.legend(title="Cover_Type", bbox_to_anchor=(1, 1))  # Move legend outside
# plt.show()
features = df.drop(columns=["Cover_Type"]).columns  # Get feature names
num_features = len(features)

# Create subplots
fig, axes = plt.subplots(nrows=(num_features // 3) + 1, ncols=3, figsize=(15, 4 * (num_features // 3)))

# Flatten axes to make it iterable
axes = axes.flatten()

# Loop through features and plot each one
for i, feature in enumerate(features):
    sns.violinplot(x=df[feature], y=df["Cover_Type"], hue=df["Cover_Type"], palette="tab10", ax=axes[i], orient="h", legend=False)
    axes[i].set_title(feature)

# Hide any extra empty subplots
for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

#CREATING MODEL CLASS
class PyTorch_NN(nn.Module):
    def __init__(self):
        super(PyTorch_NN, self).__init__()
        # TODO: Build Layers and Activation Functions
        #Soil
        self.m10 = nn.Linear(40,64)
        self.m20 = nn.Linear(64,32)
        self.m30 = nn.Linear(32,1)


        self.m11 = nn.Linear(15,32)
        self.m21 = nn.Linear(32,64)
        self.m31 = nn.Linear(64,128)
        self.m41 = nn.Linear(128,64)
        self.m51 = nn.Linear(64,32)
        self.m61 = nn.Linear(32,16)
        self.m71 = nn.Linear(16,7)

        self.f1 = nn.ReLU()
    def forward(self, x1,x2):
        #TODO: Run Layers
        x1=self.m10(x1)
        x3=self.f1(x1)
        x1=self.m20(x1)
        x3=self.f1(x1)
        x1=self.m30(x1)
        x1=self.f1(x1)
        #combining soil node
        x3 = torch.cat((x1, x2), dim=1)

        x3=self.m11(x3)
        x3=self.f1(x3)
        x3=self.m21(x3)
        x3=self.m31(x3)
        x3=self.f1(x3)
        x3=self.m41(x3)
        x3=self.f1(x3)
        x3=self.m51(x3)
        x3=self.f1(x3)
        x3=self.m61(x3)
        x3=self.f1(x3)
        x3=self.m71(x3)
        x3=self.f1(x3)
        return x3

#MODEL CRITERIA
learning_rate = 0.05
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 7500
model = PyTorch_NN()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100,gamma=0.5)

def get_accuracy(pred_arr,original_arr):
    pred_arr = pred_arr.detach().numpy()
    original_arr = original_arr.numpy()
    final_pred= []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)*100

def train_network(model, optimizer, scheduler, criterion, X_soiltrain, X_soiltest, X_nosoiltrain, X_nosoiltest, y_train, y_test, num_epochs):
    train_loss=[]

    for epoch in range(num_epochs):

        #forward feed
        output_train = model(X_soiltrain,X_nosoiltrain)

        #calculate the loss
        loss = criterion(output_train, y_train)
        train_loss.append(loss.item())

        #backward propagation: calculate gradients
        loss.backward()

        #update the weights
        optimizer.step()
        #scheduler.step()

        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()


        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}")

    return train_loss

X_soiltrain = torch.FloatTensor(X_soiltrain)
X_soiltest = torch.FloatTensor(X_soiltest)
X_nosoiltrain = torch.FloatTensor(X_nosoiltrain)
X_nosoiltest = torch.FloatTensor(X_nosoiltest)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


train_loss= train_network(model=model,
                optimizer=optimizer, criterion=criterion,
                scheduler=scheduler,
                X_soiltrain=X_soiltrain,
                X_soiltest=X_soiltest,
                X_nosoiltrain=X_nosoiltrain,
                X_nosoiltest=X_nosoiltest,
                y_train=y_train,
                y_test=y_test, num_epochs=num_epochs)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score


# Function to evaluate model with precision, recall, and per-class accuracy
def evaluate_model(model, X_soiltest, X_nosoiltest, y_test):
    with torch.no_grad():  # No gradients needed for evaluation
        output_test = model(X_soiltest, X_nosoiltest)  # Get raw logits
        predictions = torch.argmax(output_test, dim=1)  # Convert logits to class predictions

    # Convert tensors to numpy
    y_true = y_test.numpy()
    y_pred = predictions.numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute per-class accuracy (handle zero division)
    row_sums = cm.sum(axis=1)
    class_accuracy = np.where(row_sums != 0, cm.diagonal() / row_sums, 0)

    # Compute precision and recall per class with zero_division=0
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Print per-class accuracy, precision, and recall
    print("\nClass-wise Metrics:")
    for i in range(len(precision)):
        print(f"Class {i}: Accuracy = {class_accuracy[i]:.2%}, Precision = {precision[i]:.2%}, Recall = {recall[i]:.2%}")

    # Normalize confusion matrices with zero-handling
    col_sums = cm.sum(axis=0, keepdims=True)
    row_sums = cm.sum(axis=1, keepdims=True)

    precision_matrix = np.where(col_sums != 0, cm / col_sums, 0)  # Normalize by columns
    recall_matrix = np.where(row_sums != 0, cm / row_sums, 0)  # Normalize by rows

    # Plot all three confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Standard Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(7), yticklabels=range(7), ax=axes[0])
    axes[0].set_title("Confusion Matrix (Raw)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Precision Matrix (Normalized by Predicted Labels)
    sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap="Greens", xticklabels=range(7), yticklabels=range(7), ax=axes[1])
    axes[1].set_title("Precision Matrix (Column-Normalized)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # Recall Matrix (Normalized by True Labels)
    sns.heatmap(recall_matrix, annot=True, fmt=".2f", cmap="Oranges", xticklabels=range(7), yticklabels=range(7), ax=axes[2])
    axes[2].set_title("Recall Matrix (Row-Normalized)")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.tight_layout()
    plt.show()

# Call the function after training
evaluate_model(model, X_soiltest, X_nosoiltest, y_test)

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6), sharex=True)

ax1.plot(train_loss)
ax1.set_ylabel("training loss")

ax3.set_xlabel("epochs")

#
output_test = model(X_soiltest, X_nosoiltest)
print(get_accuracy(output_test, y_test))
