# DL-based-Emotion-recognition-from-EEG

Emotion recognition is a complex as it is data heavy and requires learning of all relevant details from a large pool of noisy inputs. A single layer as well as a double hidden layer was implemented in this project.

- Due to an EULA, dataset is not included
- The accuracy for single layer was found to be ~75% and for double layer is ~83%

### Dataset Description

The [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) dataset consists of two parts:

* The participant ratings, physiological recordings and face video of an experiment where 32 volunteers watched 40 music videos. 
* EEG and physiological signals were recorded and each participant also rated the videos as above. 
* For a single participant the data is subdivided into label array and eeg_data array

| data | dim | contents |
| ------ | ------ |------|
| eeg_data | 40 x 40 x 8064	| video/trial x channel x data |
| labels | 40 x 4	 | video/trial x label (valence(1-9), arousal(1-9), dominance(1-9), liking(1-9)|



# Steps:
This describes the major steps performed in both of the approaches-

- Extracting the data in such a manner thus reducing the eeg_data dimension to (320,32).
- Various preprocessing techniques like calculation standard deviation and mean for reducing feature space.
- For labels, a seperate preprocessing technique is applied which reduces its dim to (32,50).
- Then it passes through both neural network architecture and final output is used in cost calculation.
- The final output is flatened for obtaining single cost value.
- Both then outputs the most probable class of emotion.


### Keypoints

- EEG data of 10 participants is extracted then converted into feature vectors which is used as training data.
- Forward and backward propogation are created from sratch using tanh and sigmoid as activation functions
- Number of iteration were 50,000 with learning rate of 0.3
- I have used valence-arousal model for classification.
- Based on that model, 5 class of emotions can be detected using my approach.

### Procedure
Install the dependencies and devDependencies and start running knn_predict.py.

```sh
$ cd DL-based-Emotion-recognition-from-EEG
$ python nn_eeg_1l.py
$ python nn_eeg_2l.py
```

### Todos

 - Add more hidden layers
 - Applying similar approach in performing neural reconstruction


### Development

Want to contribute? Great!
You can [contact](mailto:shubhpachchigar@gmail.com) me for any suggestion or feedback!


License
----

MIT


