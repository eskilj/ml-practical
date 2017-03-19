import os
import numpy as np

best_acc = 0
best_run = ''

best_acc2 = 0
best_run2 = ''

for root, dirs, files in os.walk("/home/s1245946/tf-log/stage2"):
    for file in files:
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            print(path)
            data = np.load(path)
            final_acc = data['valid_accuracy'][-1]
            if final_acc > best_acc:
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
