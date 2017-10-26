# Distributional Bellman "C51" Algorithm implemented in Keras 

This repo includes implementation of C51 Algorithm describe in [this paper](https://arxiv.org/abs/1707.06887). The implementation is tested on the [VizDoom](http://vizdoom.cs.put.edu.pl/) **Defend the Center** scenario, which is a 3D partially observable environment.

For tutorial on Distributional Bellman and step-by-step walkthrough of the implementation, please check out my blog post at https://flyyufelix.github.io/2017/10/24/distributional-bellman.html.

<img src="/resources/c51.gif" width="300">

## Results

Below is the performance chart of 15,000 episodes of **C51 DDQN** and **DDQN** running on Defend the Center. Y-axis is the average number of kills (moving average over 50 episodes).

![C51 Performance Chart](/resources/c51_chart.png)

## Usage

First follow [this](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) instruction to install VizDoom. If you use python, you can simply do pip install:

```
$ pip install vizdoom
```

Second, clone [ViZDoom](https://github.com/mwydmuch/ViZDoom) to your machine, copy the python files provided in this repo over to `examples/python`.

To test if the environment is working, run

```
$ cd examples/python
$ python c51_ddqn.py
```

You should see some printouts indicating that the C51 DDQN is running successfully. Errors would be thrown otherwise.

## Dependencies

* Keras 1.2.2 / 2.0.5
* Tensorflow 0.12.0 / 1.2.1
* [VizDoom Environment](http://vizdoom.cs.put.edu.pl/)

