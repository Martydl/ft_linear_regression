# ft_linear_regression

This project is a basic introduction to machine learning. The goal here is to predict the price of a car by using a linear function train with a gradient descent algorithm.

## How to use

```shell
$ pip install -r requirements
[...]
$ python train.py datasets/data.csv
Training ████████████████████████████████ 100%
The obtained thetas are :
th0 : 8499.599649933214
th1 : -0.02144896359170228

The thetafile is written, you need to use predict.py now.
$ python predict.py
Enter your car mileage: 42000
Your car is predicted to be at 7598 euros.
```

`data.csv` is the original dataset, but other are provided to further test the program.

`train.py` can take parameters to change the learning behavior or show graphs, use `--help` to learn more.
