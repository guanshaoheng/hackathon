import torch
from src.config_plot import *


class Net(torch.nn.Module):
    def __init__(self, hidden_num=128, hidden_layer_num=8):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(NUM_FEATURES+ 1, hidden_num)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_num, hidden_num) for _ in range(hidden_layer_num)
        ])
        self.fc_out = torch.nn.Linear(hidden_num, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        for i in range(len(self.layers)):
            x = self.relu(self.layers[i](x))
        x = self.fc_out(x)
        return x


NUM_FEATURES = 2

epoch_max = 2000
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Define a loss function (for example, Mean Squared Error)
criterion = torch.nn.MSELoss()


x_real = torch.linspace(0, 2.*torch.pi, 100).reshape(-1, 1)
x_noisy = torch.concat([x_real + torch.randn_like(x_real)*0.2] + [ ( torch.randn_like(x_real)*0.2) for _ in range(NUM_FEATURES)], dim=-1)
y_real = torch.sin(x_real)

x_vali = torch.linspace(0, 2.*torch.pi, 23).reshape(-1, 1) 
x_vali_noisy = torch.concat([x_vali +torch.randn_like(x_vali)*0.2 ] + [ (torch.randn_like(x_vali)*0.2) for _ in range(NUM_FEATURES)], dim=-1)
y_vali = torch.sin(x_vali)

loss_train_list =[]
loss_vali_list = []
epoch_list = []

for i in range(epoch_max):
    optimizer.zero_grad()
    y_pred = net(x_noisy)
    loss = criterion(y_pred, y_real)
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        with torch.no_grad():
            y_vali_pre = net(x_vali_noisy)
            loss_vali = criterion(y_vali, y_vali_pre)
        print(f"epoch: {i} \t loss: {loss.item():.4e} loss_vali: {loss_vali.item():.4e}")
        epoch_list.append(i)
        loss_train_list.append(loss.item())
        loss_vali_list.append(loss_vali.item())

# x_test = torch.linspace(0, 2.*torch.pi, 1000).reshape(-1, 1)
# with torch.no_grad():   
#     y_test = net(x_test)

# plt.plot(x_noisy.detach().numpy().reshape(-1), y_real.detach().numpy().reshape(-1), label='real')
# plt.plot(x_test.detach().numpy().reshape(-1), y_test.detach().numpy().reshape(-1), label='test')
# plt.legend()
# plt.savefig("test.png")
# plt.close("all")

plt.plot(epoch_list, loss_train_list, label='train')
plt.plot(epoch_list, loss_vali_list, label='vali')
plt.yscale('log')
plt.legend()
plt.savefig("test.png")
plt.close("all")
