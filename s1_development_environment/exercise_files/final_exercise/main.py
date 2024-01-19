import click
import torch
from model import myawesomemodel

from data import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=64, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    training_model = myawesomemodel.to(device)
    train_set, _ = mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss {loss}")

    torch.save(training_model, "trained_model.pt")   

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1).cpu()
            test_preds.append(predicted)
            test_labels.append(labels.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    accuracy = (test_preds == test_labels).float().mean()*100
    print(f'Accuracy: {accuracy:.2f}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
