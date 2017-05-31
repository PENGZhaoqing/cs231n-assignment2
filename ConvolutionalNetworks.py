import matplotlib.pyplot as plt
from cs231n.classifiers.cnn_custom import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

model = ThreeLayerConvNet(weight_scale=0.01, reg=0.0005, input_dim=(3, 32, 32),
                          dtype=np.float64, dropout=0.8)
solver = Solver(model, data,
                num_epochs=10, batch_size=100,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
try:
    solver.train()
except:
    print

from cs231n.vis_utils import visualize_grid

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
