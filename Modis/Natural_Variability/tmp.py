import pandas as pd
import numpy as np

a = np.random.rand(1000)
b = np.random.rand(1000) * 10
c = np.random.rand(1000) * 100
# groups = np.array([1, 1, 2, 2, 2, 2, 3, 3, 4, 4])
df = pd.DataFrame({"a": a, "b": b, "c": c})

luc_bins = np.linspace(0.01, 1, 100)

df["bins"] = pd.cut(df["a"], bins=luc_bins)

df.groupby("bins").sum() 




def my_fun(x, y):
    tmp = np.sum((x * y)) / np.sum(y)
    return tmp


df.groupby("groups").apply(lambda d: my_fun(d["a"],d["b"]))

df.groupby("groups").apply(my_fun, ("a", "b"))

x = df[(df["luc"] > 0.01) & (df["luc"] <= 0.02)]["lst"]
y = df[(df["luc"] > 0.01) & (df["luc"] <= 0.02)]["dist"]
