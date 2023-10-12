import torch
from torch import nn
import scipy
from datetime import datetime

class Decsum(nn.Module):
    def __init__(self, f, K):
        super(Decsum, self).__init__()
        self.f = f
        self.K = K

    def forward(self, input_ids, attn_mask):
        self.f.train()
        X_tilda = []
        X_tilda_attention_mask = []
        k = 1

        while k <= self.K:
            if k == 1:
                temp_array = []
                for in_id, mask, in zip(input_ids, attn_mask):

                    wass_distance = scipy.stats.wasserstein_distance(
                        self.f(in_id, mask), self.f(input_ids, attn_mask)
                    )

                    log_wass_distance = torch.log(wass_distance)

                    temp_array.append(log_wass_distance)

                index_of_log_wass_distance = torch.argmin(temp_array)

                print(index_of_log_wass_distance)

                X_t = input_ids[index_of_log_wass_distance]
                X_t_attn_mask = attn_mask[index_of_log_wass_distance]
            else:
                temp_array = []

                for in_id, mask, in input_ids, attn_mask:
                    log_diff = torch.log(self.f(in_id, mask) - self.f(input_ids, attn_mask))

                    temp_array.append(log_diff)

                index_of_log_diff = torch.argmin(temp_array)

                print(index_of_log_diff)

                X_t = input_ids[index_of_log_diff]
                X_t_attn_mask = attn_mask[index_of_log_diff]

            X_tilda.append(X_t)
            X_tilda_attention_mask.append(X_t_attn_mask)
            X_tilda = torch.stack(X_tilda, dim=0)
            X_tilda_attention_mask = torch.stack(X_tilda_attention_mask, dim=0)

            input_ids.remove(X_t)
            attn_mask.remove(X_t_attn_mask)
        
            k += 1
        
        f_score = self.f(X_tilda, X_tilda_attention_mask)
        return f_score
    
def train_model_ds(
    f, ds_model, train_dl, val_dl, loss_function, optimizer, scheduler, num_epochs
):
    train_losses = []
    validation_losses = []
    print("------------------------------------ STARTING TRANING FOR DS")
    print(f"Starting training at: {datetime.now()}")

    for epoch in range(num_epochs):
        f.train()
        ds_model.train()
        train_loss_total = 0.0
        num_train_batches = 0

        for data in train_dl:
            input_ids = data["input_ids"]
            attention_masks = data["attention_mask"]
            targets = data["target"].float()

            targets = targets.view(-1, 1)

            outputs = ds_model(input_ids=input_ids, attn_mask=attention_masks)

            loss = loss_function(outputs, targets)

            train_loss_total += loss.item()
            num_train_batches += 1

            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        avg_train_loss = train_loss_total / num_train_batches
        train_losses.append(avg_train_loss)

        f.eval()
        ds_model.eval()
        val_loss_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for data in val_dl:
                input_ids = data["input_ids"]
                attention_masks = data["attention_mask"]
                targets = data["target"].float()

                targets = targets.view(-1, 1)

                outputs = ds_model(input_ids=input_ids, attn_mask=attention_masks)

                loss = loss_function(outputs, targets)

                val_loss_total += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss_total / num_val_batches
        validation_losses.append(avg_val_loss)

        time = datetime.now()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Finished at: {time}"
        )

    print(f"Finished training at: {datetime.now()}")
    print("------------------------------------ ENDING TRANING FOR DS")

    return train_losses, validation_losses