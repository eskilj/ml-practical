import os
import numpy as np
import matplotlib.pyplot as plt

best_acc = 0
best_run = ''

best_acc2 = 0
best_run2 = ''


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

for root, dirs, files in os.walk("../notebooks/tf-log/norm"):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            print(path)
            data = np.load(path)
            final_acc = data['valid_error'][-1]
            if 'epochs=5' in path:
                plot_data = data['train_error']
                plot_data2 = np.subtract(plot_data, data['valid_error'])
                ax1.plot(np.arange(1, plot_data.shape[0] + 1), plot_data, label=str(path.split('wd=', 1)[1].split(',',1)[0] + ', bn=' + path.split('bn=', 1)[1].split(',',1)[0]) + ', do=' + path.split('do=', 1)[1].split(',',1)[0])
                ax2.plot(np.arange(1, plot_data2.shape[0] + 1), plot_data2)

            if final_acc > best_acc:
                print(data['valid_error'])
                if best_acc2 != 0:
                    best_acc2 = best_acc
                    best_run2 = best_run

                best_acc = final_acc
                best_run = path
            elif final_acc > best_acc2:
                best_acc2 = final_acc
                best_run2 = path

print('Best acc: {}, by run {}'.format(best_acc, best_run))
print('2nd Best acc: {}, by run {}'.format(best_acc2, best_run2))

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set Accuracy')
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set Accuracy')

fig.savefig('new_fig_error.pdf')
