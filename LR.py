from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # import data
    df = pd.read_csv("./sleepData.csv")

    X = df[["Pittsburgh", "Daily Stress", "Total Minutes in Bed", "Total Sleep Time", "Number of Awakenings", "Average Awakening Length", "Total Screen Time", "Smoking", "Alcohol"]]
    Y = df["Efficiency"]

    LRr2 = 0
    LRmse = 0
    LRmae = 0
    RRr2 = 0
    RRmse = 0
    RRmae = 0
    Lar2 = 0
    Lamse = 0
    Lamae = 0
    BRRr2 = 0
    BRRmse = 0
    BRRmae = 0
    SVMr2 = 0
    SVMmse = 0
    SVMmae = 0
    KNNr2 = 0
    KNNmse = 0
    KNNmae = 0
    iterations = 10000

    for i in range(1,iterations):
        # Train / Test Split
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

        # Linear Regression
        linearReg = linear_model.LinearRegression()
        linearReg.fit(x_train, y_train)
        linearRegOutput = linearReg.predict(x_test)
        LRr2 = LRr2 + r2_score(y_test,linearRegOutput)
        LRmse = LRmse + mean_squared_error(y_test,linearRegOutput)
        LRmae = LRmae + mean_absolute_error(y_test,linearRegOutput)

        # Ridge Regression
        ridgeReg = linear_model.Ridge(alpha=0.5)
        ridgeReg.fit(x_train, y_train)
        ridgeRegOutput = ridgeReg.predict(x_test)
        RRr2 = RRr2 + r2_score(y_test,ridgeRegOutput)
        RRmse = RRmse + mean_squared_error(y_test,ridgeRegOutput)
        RRmae = RRmae + mean_absolute_error(y_test,ridgeRegOutput)

        # Lasso Regression
        lassoReg = linear_model.Lasso(alpha=0.1)
        lassoReg.fit(x_train,y_train)
        lassoRegOutput = lassoReg.predict(x_test)
        Lar2 = Lar2 + r2_score(y_test,lassoRegOutput)
        Lamse = Lamse + mean_squared_error(y_test,lassoRegOutput)
        Lamae = Lamae + mean_absolute_error(y_test,lassoRegOutput)

        # Bayesian Ridge Regression
        bayRidgeReg = linear_model.BayesianRidge()
        bayRidgeReg.fit(x_train, y_train)
        bayRidgeRegOutput = bayRidgeReg.predict(x_test)
        BRRr2 = BRRr2 + r2_score(y_test,bayRidgeRegOutput)
        BRRmse = BRRmse + mean_squared_error(y_test,bayRidgeRegOutput)
        BRRmae = BRRmae + mean_absolute_error(y_test,bayRidgeRegOutput)

        # SVM
        SVMReg = svm.SVR()
        SVMReg.fit(x_train,y_train)
        SVMRegOutput = SVMReg.predict(x_test)
        SVMr2 = SVMr2 + r2_score(y_test,SVMRegOutput)
        SVMmse = SVMmse + mean_squared_error(y_test,SVMRegOutput)
        SVMmae = SVMmae + mean_absolute_error(y_test,SVMRegOutput)

        # KNN
        KNNReg = KNeighborsRegressor(n_neighbors=3)
        KNNReg.fit(x_train,y_train)
        KNNRegOutput = KNNReg.predict(x_test)
        KNNr2 = KNNr2 + r2_score(y_test,KNNRegOutput)
        KNNmse = KNNmse + mean_squared_error(y_test,KNNRegOutput)
        KNNmae = KNNmae + mean_absolute_error(y_test,KNNRegOutput)


    LRr2 = LRr2/iterations
    LRmse = LRmse/iterations
    LRmae = LRmae/iterations
    RRr2 = RRr2/iterations
    RRmse = RRmse/iterations
    RRmae = RRmae/iterations
    Lar2 = Lar2/iterations
    Lamse = Lamse/iterations
    Lamae = Lamae/iterations
    BRRr2 = BRRr2/iterations
    BRRmse = BRRmse/iterations
    BRRmae = BRRmae/iterations
    SVMr2 = SVMr2/iterations
    SVMmse = SVMmse/iterations
    SVMmae = SVMmae/iterations
    KNNr2 = KNNr2/iterations
    KNNmse = KNNmse/iterations
    KNNmae = KNNmae/iterations


    print("Linear Regression:")
    print("R2: " + str(LRr2))
    print("MSE: " + str(LRmse))
    print("MAE: " + str(LRmae))

    print("Ridge Regression:")
    print("R2: " + str(RRr2))
    print("MSE: " + str(RRmse))
    print("MAE: " + str(RRmae))

    print("Lasso Regression:")
    print("R2: " + str(Lar2))
    print("MSE: " + str(Lamse))
    print("MAE: " + str(Lamae))

    print("Bayesian Ridge Regression:")
    print("R2: " + str(BRRr2))
    print("MSE: " + str(BRRmse))
    print("MAE: " + str(BRRmae))

    print("SVM Regression:")
    print("R2: " + str(SVMr2))
    print("MSE: " + str(SVMmse))
    print("MAE: " + str(SVMmae))

    print("KNN Regression:")
    print("R2: " + str(KNNr2))
    print("MSE: " + str(KNNmse))
    print("MAE: " + str(KNNmae))

if __name__ == "__main__":
    main()