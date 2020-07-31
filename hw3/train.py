from util import *
from model import *

if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = sys.argv[1]
    label_path = sys.argv[2]

    learning_rate = 1e-3
    num_epochs = 300
    batch_size = 2000

    train_d = face_dataset(path=path, label_path=label_path, is_train=True, transform=img_transform)
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)


    model = model_4().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print('Number of params: %d' % (params_num(model)))

    loss_list, acc_list = train(train_loader, model, optimizer, criterion, num_epochs)
    torch.save(model.state_dict(), './model/model_state_dict')