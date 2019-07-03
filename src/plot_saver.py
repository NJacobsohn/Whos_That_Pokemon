import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import itertools
import matplotlib as mpl
from keras.preprocessing.image import load_img, img_to_array


sns.set()

mpl.rcParams.update({
    'font.size'           : 18.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})


with open('../models/metrics/model_2700_cm.txt', 'rb') as f:
    cm_my_model = pickle.load(f)
with open('../pickles/class_names.p', 'rb') as f:
    class_names = pickle.load(f)

col_names = ["loss", "acc", "val_loss", "val_acc"]
my_cnn_history = pd.read_csv("../models/model_accuracy_2367_history.txt", header=None, names=col_names)


xception_acc = np.array([0.6402, 0.7998, 0.8407, 0.8619, 0.8759, 0.8859, 0.8926, 0.8969, 0.9027, 0.9075])
xception_loss = np.array([1.4825, 0.7241, 0.5611, 0.4803, 0.4321, 0.3990, 0.3732, 0.3626, 0.3466, 0.3307])
xception_val_acc = np.array([0.3166, 0.3114, 0.2991, 0.3035, 0.3114, 0.3053, 0.2951, 0.3169, 0.2993, 0.3145])
xception_val_loss = np.array([4.4622, 5.3454, 5.8250, 6.0309, 6.2075, 6.4548, 6.7880, 6.7285, 6.7837, 6.7788])


xception_df = np.empty(shape=(4, 10))
xception_df[0] = xception_loss
xception_df[1] = xception_acc
xception_df[2] = xception_val_loss
xception_df[3] = xception_val_acc
xception_df = xception_df.T


epochs = 10
x_vals = np.arange(0, epochs)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_vals, xception_df[:epochs, 0], "ob-", label="Training Loss (Xception)")
ax.plot(x_vals, xception_df[:epochs, 2], "og-", label="Testing Loss (Xception)")
ax.plot(x_vals, my_cnn_history.iloc[:epochs, 0], "or-", label="Training Loss (My CNN)")
ax.plot(x_vals, my_cnn_history.iloc[:epochs, 2], "ok-", label="Testing Loss (My CNN)")
ax.set_xticks(x_vals)
ax.set_xticklabels(x_vals+1)
#ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
ax.set_title("Loss Comparison")
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(fontsize="x-small")
plt.tight_layout()
plt.savefig("../img/loss_comparison.png", dpi=200)