#! /bin/python
import pandas as pd
from sys import argv
from matplotlib import pyplot as plt

df = pd.read_csv(argv[1], sep="  ", names=["t", "y"])
print(df.head())
df.plot(x="t", y="y")
plt.show()
