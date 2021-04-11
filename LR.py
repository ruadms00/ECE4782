from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # import data
    df = pd.read_csv("./sleepData.csv")

    X = df[["Pittsburgh", "Daily Stress", "Total Minutes in Bed", "Total Sleep Time", "Number of Awakenings", "Average Awakening Length", "Total Screen Time", "Smoking", "Alcohol"]]
    Y = df["Efficiency"]

    model = linear_model.LinearRegression()
    model.fit(X, Y)
    output =[]
    #print(X[0:1])
    for i in X.index:
        output.append(model.predict(X[0+i:1+i]))
    #    print(X[0+i:1+i])
    print(model.predict([["5", "89.95", "378", "340", "18", "4.22", "300", "300", "300"]]))
    plt.scatter(df.index, df.Efficiency)
    plt.plot(df.index, output)
    plt.xlabel("index")
    plt.ylabel("Efficiency")
    plt.show()

if __name__ == "__main__":
    main()