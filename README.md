### BinaryBot
The goal of this project was to create a bot that was capable of fully autonomous play of the game StarCraft 2. During the project I learned about utilizing Numpy for saving and loading data, constructing neural nets with TensorFlow and Keras, utilizing both local and cloud GPU's, analyzing data in real-time and working with Linux.

In the game of StarCraft 2 there is no one set way to victory and since my end goal was to create a flexible bot that developed its own playstyle and strategy, I decided to try to accomplish this by having the bot's actions be controlled by a neural net. I went with a neural net implementation utilizing LSTM, long short-term memory, which is a type of neural network that has a type of built-in memory function. Since a neural net needs to study large amounts of data to train itself, I first created a bot that would generate game data and used it to train the net. After training and saving out a model, I implemented it into the bot to read real-time data and chose in-game actions. 

Unfortunately, the neural net does not play the game all that well. The bot does analyze real-time data and sometimes makes varying decisions, but often choses the same action repeatedly until its destroyed. To improve its play, I implemented a system to counteract this that if the bot chooses the same action 5 times in a row, instead of carrying out that action, a random action is used instead. This adds significant improvements to the bot’s performance, but defeats the overall purpose of the neural net being the key decision maker.

## Files
binary_bot.py is the main bot file that uses the neural net model, which is called CuDNNLSTM-model.h5, to control its actions. If you wish to alter the models layers or configuration, use the CuDNNLSTM.py file which contains all of the code to import the training data, construct the model, then train and save it. To create training data use Binary_bot_random.py or Binary_bot_static.py. The random bot just chooses random actions and overall performs pretty poorly. The static bot iterates through sets of actions at a rapid pace and is much better at playing the game, but has little variety.

You will also need a full install of StarCraft 2 which can be downloaded for free from https://starcraft2.com

## Packages
PySC2 is a collaboration between DeepMind and Blizzard to develop StarCraft II into a rich environment for RL research. PySC2 provides an interface for RL agents to interact with StarCraft 2, getting observations and sending actions. The code can be found here: https://github.com/deepmind/pysc2

SC2 is an easy-to-use library for writing AI Bots for StarCraft II in Python 3. Documentation and details can be found here: https://github.com/Blizzard/s2client-proto#downloads

Official map packs, the Linux distribution and additional information can be found on Blizzards Git here: https://github.com/Blizzard/s2client-proto

## Papers
DeepMind’s goal for this project was to see if machine learning was capable of mastering the game of StarCraft 2. The published their findings in a research paper which can be read here: https://arxiv.org/abs/1708.04782

TenCent also developed a pair of bot’s using the environment and published a paper with their findings here: https://arxiv.org/abs/1809.07193

Unlike DeepMind, TenCent actually released their modified version of the PySC2 environment and code for their bots, which can be found on their Git here: https://github.com/Tencent/TStarBots

## Author
**Eric Born**

[**PORTFOLIO**](https://ericborn.github.io)

[**GITHUB**](https://github.com/ericborn)

[**BLOG**](https://medium.com/@eric.born85)
