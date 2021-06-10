import torch
from tqdm import tqdm 
from model import FineTuner
from dataset import Caltech256 




if __name__ == '__main__':
    print('Fine Tune Backbone')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FineTuner(layer_bb=2).to(device)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_data = Caltech256(split='train')
    trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_data = Caltech256(split='test')
    testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data))
    epochs = 30
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for index, (img, label) in tqdm(enumerate(trainloader)):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for index, (img, label) in enumerate(testloader):
                img, label = img.to(device), label.to(device)
                pred = model(img)
                _, pred = torch.max(pred.data, 1)
                total = len(test_data)
                correct = (pred == label).sum().item()
                acc = correct / total
                print(f'Accuracy: {acc}')
                if acc > best_acc:
                    best_acc = acc

    print('Best acc:', best_acc)
