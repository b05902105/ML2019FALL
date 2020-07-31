from util import *
from model import *

if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = sys.argv[1]
    output_path = sys.argv[2]

    batch_size = 4000

    test_d = face_dataset(path=path, is_train=False, transform=img_transform)
    test_loader = DataLoader(test_d, batch_size=batch_size, shuffle=False)

    model = model_4().cuda()
    model.load_state_dict(torch.load('./model/model_state_dict'))
    print('Number of params: %d' % (params_num(model)))

    pred = predict(model, test_loader)
    print(pred)
    output(pred, output_path)