# Double DQN - A Keras-based implementation

[![dqn](https://img.youtube.com/vi/yiKu66FOX6I/0.jpg)](https://www.youtube.com/watch?v=yiKu66FOX6I)

This is an implementation of the Double DQN algorithm. Apart from some smaller differences the implementation is in line with the following two article:

[Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.htm)
[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf).

The outline of this overview is:
* Brief description of the whole algorithm
* Requirements for using the current algoithm
* Results, conclusions

## Details of the algorithm


## Requirements

In order to run the algorithm you should create a [python environment](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/) and then install the required packages by typing **pip install -r requirements.txt**. You can find the txt file among the source files. This will use Tensorflow with GPU.

If you have already had all the requierements just run the run.py file. The start.txt shows examples how to set the parameters. In order to *record videos* at the end, the recorder.py script can do it. It uses the saved file from the files folder with the extension *hdf5*. It imports the environemnt.py, agent.py and tf.py scripts as well. *Learning curves* can be plotted by statistics.py scrtipt. The easiest way to run it if you put it inside the files folder and start it there. It takes three arguments:(1) evaluation frequency during training, (2) the total number of evaluations during training and (3) the number of episodes per evaluation. Example usage:

```bash
python statistics.py --eval-freq 100000 --eval-num 100 --episode-num 30.
```

## Results

![dqnavg](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwRzgtM3Vab3ZVTm8  "DQN average returns")

![dqnmax](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwT2llTlNqTFRuMnM  "DQN maximum returns")

![dqnmin](https://drive.google.com/uc?export=download&id=0B97L9zqg-lnwa0VPWFA2eUlPTXc  "DQN minimum returns")
