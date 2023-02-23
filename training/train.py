import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.resnet18 import ResidualBlock, ResNet18
from models.resnet34 import ResidualBlock, ResNet34
from models.resnet50 import Bottleneck, ResNet50
from models.resnet101 import Bottleneck, ResNet101
from training.training_config import TrainingConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

config = TrainingConfig()

classes = config.classes
model_checkpoint_path = config.model_checkpoint_path

# model = ResNet18(ResidualBlock, [2, 2, 2, 2]).to(device)
# model = ResNet34(ResidualBlock, [3, 4, 6, 3]).to(device)
# model = ResNet50(Bottleneck, [3, 4, 6, 3]).to(device)
model = ResNet101(Bottleneck, [3, 4, 23, 3]).to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.49139968, 0.48215841, 0.44653091),
            std=(0.24703223, 0.24348513, 0.26158784),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)


def load_data():
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root_dir, train=True, download=True, transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root_dir, train=False, download=True, transform=transform
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_dataloader, test_dataloader


def train(train_dataloader):
    model.train()
    best_validation_accuracy = 0

    for epoch in range(1, config.num_epochs):
        loop = tqdm(train_dataloader)
        for data in loop:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch}/{config.num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Training Loss: {loss.item():.3f}")

        accuracy = validate(test_dataloader)
        if accuracy > best_validation_accuracy:
            epochs_no_improvement = 0
            torch.save(model.state_dict(), config.model_checkpoint_path)
            best_validation_accuracy = accuracy
        else:
            epochs_no_improvement += 1
            print(f"Epochs without improvement: {epochs_no_improvement}")
            if epochs_no_improvement > config.patience:
                print(f"Early Stopping: Patience of {config.patience} reached.")
                break
    print("Finished Training")
    print(f"Highest Validation Accuracy: {best_validation_accuracy}")


def validate(test_dataloader):
    model.eval()
    correct_predictions = 0
    total_images_predicted = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, dim=1)
            total_images_predicted += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        accuracy = 100 * correct_predictions / total_images_predicted
    print(f"Validation Loss: {loss.item():.3f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy


train_dataloader, test_dataloader = load_data()
train(train_dataloader)
