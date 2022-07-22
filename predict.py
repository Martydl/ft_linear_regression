#!/usr/bin/python3

import pandas as pd
import numpy as np


def predict(theta0, theta1, km):
    return theta0 + (theta1 * km)


def read_thetafile():
    try:
        data = pd.read_csv("./theta.csv")
    except pd.errors.EmptyDataError:
        exit("Empty theta file.")
    except Exception as error:
        exit("Error, please check your theta file")
    data = np.array(data)
    if data.size == 0:
        exit("Error, please check your theta file")
    else:
        data = data[0, :]
        if data[0] == 0 and data[1] == 0:
            exit("Thetas are not set.")
    return data

def get_mileage():
    km = input("Enter your car mileage: ")
    if km.isdigit():
        km = int(km)
    else:
        exit("Invalid mileage, please enter a positive integer.")
    return km

if __name__ == "__main__":
    theta = read_thetafile()
    km = get_mileage()
    price = predict(theta[0], theta[1], km)
    if price > 0:
        print("Your car is predicted to be at %d euros." % price)
    else:
        print("The price of your car cannot be estimated.")
