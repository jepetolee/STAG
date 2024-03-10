from Model import *
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from sklearn.metrics import f1_score

CFG = {
    'EPOCHS': 2,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':32,
    'SEED': 39
}
class CustomDataset(Dataset):
    def __init__(self, dataPath, label_list):

        self.numpy_path_list = dataPath
        self.label_list = label_list


    def __getitem__(self, index):
        img_path = self.numpy_path_list[index]
        data = np.load(img_path)
        label = torch.tensor(self.label_list[index], dtype=torch.float32)

        return (torch.from_numpy(data).float().reshape(1,270,240), label)
    def __len__(self):
        return len(self.numpy_path_list)

def extractDF(URL,ratio=0.5):
    df = pd.read_csv(URL)
    class_dfs = [df[df['POSITION'] == 2] ,df[df['POSITION'] == 1] ,df[df['POSITION'] == 0] ]

    target_ratios_goal = ratio  # 예시로 0.5로 설정, 원하는 비율에 따라 조절
    iters =int( len(class_dfs[0].sample(frac=target_ratios_goal, random_state=42))/ len(class_dfs[1].sample(frac=1, random_state=42)))

    sampled_dfs = [class_dfs[1].sample(frac=1, random_state=42) for _ in range(iters)]
    sampled_dfs.append(class_dfs[0].sample(frac=target_ratios_goal, random_state=42))
    sampled_dfs.append(class_dfs[2].sample(frac=1, random_state=42))

    return pd.concat(sampled_dfs)
def Dataset(df):
    df = df.sample(frac=1)
    train_len = int(len(df) * 0.9)
    train_df = df[:train_len]
    val_df = df[train_len:]

    print(train_df['POSITION'].value_counts(),val_df['POSITION'].value_counts())
    train_dataset = CustomDataset(train_df['url'].values, train_df['POSITION'].values)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_dataset = CustomDataset(val_df['url'].values, val_df['POSITION'].values)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    return train_loader, val_loader


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.float().reshape(-1,1).to(device)

            probs = model(imgs)


            loss = criterion(probs, labels)

            val_loss.append(loss.item())
            del imgs, labels, loss,probs
            torch.cuda.empty_cache()

        _val_loss = np.mean(val_loss)

    return _val_loss

def train(model, optimizer, train_loader, val_loader, device,epoch,link,saved=False):
    if saved:
        model.load_state_dict(torch.load(link))
    model.to(device)
    criterion = nn.MSELoss()
    model.train()
    best_train_loss = float('inf')
    best_map = 999999
    for epoch in range(1, epoch+1):
        model.train()
        train_loss = []
        targets = []
        with tqdm(iter(train_loader)) as pbar:

            for imgs, labels in pbar:
                imgs = imgs.float().to(device)
                labels = labels.float().reshape(-1,1).to(device)
                model.zero_grad()
                output = model(imgs)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                targets.append(labels.cpu().detach())
                train_loss.append(loss.item())
                del imgs, labels, loss, output

        torch.cuda.empty_cache()
        _train_loss = np.mean(train_loss)
        if best_train_loss > _train_loss:
            best_train_loss = _train_loss
            torch.save(model.state_dict(), './BestTrainValue.pt')
        _val_loss = validation(model, criterion, val_loader, device)


        lr =optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}]  Valid Loss : [{_val_loss:.5f}] LR :[{ lr:.5e}]')

        if _val_loss < best_map:
            best_map = _val_loss
            torch.save(model.state_dict(), link)

    return

model = TradingSupervisedValueModel()

optimizer = torch.optim.AdamW(params=model.parameters() , lr=CFG["LEARNING_RATE"])

BTC = extractDF('./DatasetBuilder/BTC_FUTURUE150_20_PROFIT.csv',0.4945)

ETH = extractDF('./DatasetBuilder/ETH_FUTURUE150_20_PROFIT.csv',0.5824)

SOL = extractDF('./DatasetBuilder/SOL_FUTURUE150_20_PROFIT.csv',0.6511)

XRP = extractDF('./DatasetBuilder/XRP_FUTURUE150_20_PROFIT.csv',0.5787)

LTC = extractDF('./DatasetBuilder/LTC_FUTURUE150_20_PROFIT.csv',0.6502)

BCH = extractDF('./DatasetBuilder/XRP_FUTURUE150_20_PROFIT.csv',0.5880)

df = pd.concat([BTC,ETH,SOL,XRP,LTC,BCH])

train_loader, val_loader =Dataset(df)

train(model, optimizer, train_loader, val_loader, 'cuda', CFG['EPOCHS'], './pretrainedValue.pt',False)