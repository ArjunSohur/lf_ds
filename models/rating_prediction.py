# Library imports
from torch import nn
import torch


class rating_pred(nn.Module):
    def __init__(self, transformer_model):
        super(rating_pred, self).__init__()

        self.transformer = transformer_model
        self.fc = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attn_mask):
        input_ids = input_ids.squeeze(1)

        input_ids = input_ids
        attn_mask = attn_mask

        output = self.transformer(
            input_ids=input_ids, attention_mask=attn_mask
        ).pooler_output

        rating = self.fc(output)

        # rating = torch.clamp(rating, min=0, max=5) # makes val loss constant on low data

        return rating


def train_model(
    model, train_dl, val_dl, loss_function, optimizer, scheduler, num_epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        num_train_batches = 0

        for data in train_dl:
            input_ids = data["input_ids"]
            attention_masks = data["attention_mask"]
            targets = data["target"].float()

            targets = targets.view(-1, 1)

            outputs = model(input_ids=input_ids, attn_mask=attention_masks)

            loss = loss_function(outputs, targets)

            train_loss_total += loss.item()
            num_train_batches += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        avg_train_loss = train_loss_total / num_train_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for data in val_dl:
                input_ids = data["input_ids"]
                attention_masks = data["attention_mask"]
                targets = data["target"].float()

                targets = targets.view(-1, 1)

                outputs = model(input_ids=input_ids, attn_mask=attention_masks)

                loss = loss_function(outputs, targets)

                val_loss_total += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss_total / num_val_batches
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses
