import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.stdin)

df = pd.DataFrame([
    [df['tn'][0], df['fp'][0]],
    [df['fn'][0], df['tp'][0]]
    ], columns=['F','T'], index=['F', 'T'])

cf = sns.heatmap(df, annot=True, cmap='Blues', fmt='g')
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.title(sys.argv[1])
if len(sys.argv) < 3:
    plt.show()
else:
    plt.savefig(sys.argv[2])
