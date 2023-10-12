# Realative imports
from manage_dataset.manage_data import extract, get_data
from manage_dataset.dataset import YelpData, sentence_YelpData
from models.rating_prediction import rating_pred, train_model
from models.decsum import Decsum, train_model_ds

# Library imports
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from transformers import AutoModel, AutoConfig
import torch


# variables
datapoints_to_extract = 100
train_size = int(datapoints_to_extract * 0.7)
val_size = int(datapoints_to_extract * 0.2)
test_size = datapoints_to_extract - train_size - val_size

batch_size = 10 # int(datapoints_to_extract / 20)
num_epochs = 3

train_f = False
save_path = "/Users/arjunsohur/Desktop/lf_ds_/models"

transformer_model = AutoModel.from_pretrained("allenai/longformer-base-4096")


if __name__ == "__main__":

    if train_f:
        extract(datapoints=datapoints_to_extract)

        list_data = get_data()

        dataset = YelpData(list_data)

        train_data, val_data, test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        print("Loaded t/v/t split")

        f = rating_pred(transformer_model=transformer_model)

        print("Initialized f")

        loss_function = nn.MSELoss()
        optimizer = optim.AdamW(f.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=30
        )

        train_loss_f, val_loss_f = train_model(
            model=f,
            train_dl=train_loader,
            val_dl=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
        )

        torch.save(f.state_dict(), "{}/f.pt".format(save_path))

    if not train_f:
        f = rating_pred(transformer_model=transformer_model)
        state_dict = torch.load("{}/f.pt".format(save_path))

        f.load_state_dict(state_dict)

    extract(datapoints=datapoints_to_extract)

    list_data = get_data()

    dataset = sentence_YelpData(list_data)

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print("Loaded t/v/t split")

    ds = Decsum(f=f, K=5)

    optimizer = optim.AdamW(ds.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    loss_function = nn.MSELoss()

    train_loss_ds, val_loss_ds = train_model_ds(
        f=ds,
        ds_model=ds,
        train_dl=train_loader,
        val_dl=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
    )


    torch.save(ds.state_dict(), "{}/ds.pt".format(save_path))

    
 
