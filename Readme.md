# TODO:

# Filip

- [ ] train agent on 35x35
    - [x] memorization
    - [x] replay

- [x] requirements.txt *(used pipreqs)*
- [ ] design CNN architectures on other room sizes
- [ ] explore tensorboard

## Wojtek

- [ ] fix roomgeneration module to return values in a proper colormap (see `room_mechanics.py`)
- [ ] encapsulate roomgeneration in a class
- [ ] document code (comments, docstrings)
- [ ] return type of rooms generated should be `np.array`
- [ ] visualisations of `35x35` cleaning model architecture with two variants: 
    - broom endpoints as input to first layer 
    - broom endpoints as input to a layer after flatten (900 + 4)

# Structure
 
The `config.ini` file defines some universal constants that you rather shouldn't change. If you really want to keep it mind that it can majorly disrupt environment's behaviour.

# Training

## Training the mounds planning model

If the folder `trained cleaning models` is empty there is no model on which basis we can train the mounds planning model - first train the cleaning model as explained in the next subsection.

Otherwise, you can train the model yourself with the following command:

```
    python train_planning.py --cleaning_model_weights "trained cleaning models/name_of_saved_weghts.h5" 
```

## Training the cleaning model

# Project description 

How do you plan the logistics of moving items around in a large usable space? Initially, it might seem appropriate to partition the space into smaller sections in which movement would be implicitly manageable. In our strategy, however, we make quite considerable use of the assumption that there is no easier method of alternative space allocation or movement organisation. The main objective of this work is to validate this assumption, i.e. to investigate whether there is a better method than the default approach in general. 

The deeper application of this project is to see if during a process of collective displacement of objects (randomly arranged in the beginning) it is optimal to centralise piles or rather diversify it into making smaller heaps in seperate sub-rooms. Another theoretical application is to compare the results returned by the algorithm - which describe how to plan moving things around - to what existing clustering algorithms return. 

Name conventions in this project describe actions related to cleaning a room with a broom. It's simpler to visualize the objective this way. Interpreting collective movement through the perspective of using a broom also allows us to further generalize the results returned by the model to tasks involving discontinuous values such as value of 1 standing for a large box, value 0.6 for a medium one and 0.2 for a small package.

The model atttempts to find an optimal sequence of broom movements to clean a room with a given shape. The sequence is found by sequentially applying the NEAT algorithm (```CleaningModel.py```), which has been trained on random rooms (```RoomGeneration.py```) first with naively chosen mounds of dirt (by applying clustering algorithms - intuitvely we should sweep the broom to where dirt is mostly clustered.) After the NEAT algorithm has been developed, a module (```PlanMoundsModel.py```) for planning mounds (mounds in the room mechanics act like holes in the floor consuming the dirt) is trained on random rooms and based on evaluations of the NEAT Cleaning Model. 

# Implementation details 

## Environment mechanics

## Deep Q-Learning agent

Following the [flappy bird DQL architecture](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) we started implementing the DQL agent. Unfortunately however it became apparent that the DQL algorithm is not suitable for environments with continuous action spaces. In an attempt to solve this problem we redesigned the agent to select 9 different actions: move the start/end of the broom up/down/left/right as well as a special action to perform the actual movement of transporting the dirt. 

The process of training the model became significantly slower since instead of quickly switching between different broom positions we were incentivising the agent to slowly, incrementally update the `x, y` positions of broom endpoints before the actual movement of the broom.

We proceeded to explore reinforcement learning algorithms designed specifically for continuous action spaces.

## Agent-critic model 

We began by building upon the minimal examples provided by [this reinforcement learning repository](https://github.com/rlcode/reinforcement-learning). Most of the code for their Agent-critic model in `gym-cartpole` envivornment was natural to implement in our problem. What had to be changed, however, is the existence of a `done` parameter - our agent learns to clean the room in a limited time space, without any terminating conditions throughout. Another significant aspect was to change `A2CAgent`'s method `get_action()` to return actions chosen from a continuous space - which was the primary issue in the previous DQL learning approach. 

Choosing actions from a discrete action space - let's say pressing one of the buttons `[u d l r]` - follows from the agent calculating a probability distribution dependent on the current state of the environment `S_t`. It could be for instance `[0.05 0.65 0.15 0.25]`. 
Transitioning to a continuous action space requires us to reimplement the agent so that he is calculating probability distributions for each of the individual random variables - in our case it was a list of variables `[X_1 Y_1 X_2 Y_2]`. The agent then calculates probability distributions based on current environment state `S_t` for each of them: e.g. his policy might yield that `X_2` follows a gaussian distribution `N(14.2, 3.5)`. Based on these distributions we finally arrive with coordinates describing the broom movement.

Sampling an action from a normal distribution, however, presents us with some problems, as described in [this article](https://kae1506.medium.com/actor-critic-methods-with-continous-action-spaces-having-too-many-things-to-do-e4ff69cd537d). The problem is that some values of standard deviation might deprive the agent of (or provide him with excessive) ability of exploration.  There are 3 options for how the standard deviation should be chosen for each of the distributions when calculating action's distribution:
    - let the agent calculate std for each of the random variables in the action 
    - gradually decrease the std during training until it reaches a value like `0.001`
    - keep the std constant.
