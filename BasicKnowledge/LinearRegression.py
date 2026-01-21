from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
impoer pandas as pd

args = Namespace(
    seed=1234,
    data_file="sample_data.csv",
    num_samples=100,
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)

np.random.seed(args.seed)

def generate_data(num_samples):
    X = np.array(range(num_samples))
    y = 3.65*X + 10
    return X, y

X, y = generate_data(args.num_samples)
data = np.vstack((X, y)).T
df = pd.DataFrame(data, columns=["X", "y"])
df.head()

plt.title("Generated Data")
plt.scatter(X=df["X"], y=df["y"])
plt.show()

