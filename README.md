## Neural Network from NumPy

This is an example of a neural network implemented from scratch (just using numpy).

### Run a fully connected [CNN] model on the exp [mnist] dataset

1. Generate the training data (stored in a csv file)
```
python -m data.exp[mnist].generate
```
2. Check if the csv file loads
```
python -m data.exp[mnist].load
```
3. Train the Vec2Class[ConvNet] model using the dataset
```
python -m data.exp[mnist].train
```
4. Evaluate the performance of the model by plotting the results (loss, accuracy)
```
python -m data.exp[mnist].show
```

### Test script

The test script is available in the nn\tests directory. Tests can be run from the top level directory by:
```
python nn\run_tests.py
```