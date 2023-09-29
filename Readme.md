# Overview

This is a project conducted by Filip Morawiec and Wojciech Siostrzonek under the supervision of Politechnika Śląska.

[Here](posters/poster%20polish.pdf) is the original 22.09.2023 conference poster (polish) and [here](posters/poster%20english.pdf) is the english version.

<!-- ## Results discussion -->


# Using this repository 

## Training the cleaning model

You can run the following command to train the model (either based on previously trained checkpoints or not):

```
    python train_cleaning_ddpg.py --load_previous_models True 
                                  --load_models_path experiments/id15 
                                  --num_moves 15 
                                  --num_episodes 10000 
                                  --replay_memory_size 1024 
                                  --batch_size 32 
                                  --noise 0.3 
                                  --gpu 0 
                                  --punish_clipping 2 
                                  --actor_learning_rate 0.0001 
```

to see more details run

```
    python train_cleaning_ddpg.py --help

```

## Visualizing agent's brain 

To run the tSNE embedding on the last layer (of size 200) run the command:

```
    python 'agent tsne embedding.py' --num_data_points_expected 10000 
                                     --model_path experiments/randomweights 
                                     --random_weights
```

for more info run 

```
    python 'agent tsne embedding.py' --help
```

# Implementation details

## Room generation

Co-author of this project, Wojciech Siostrzonek has implemented the following methods of generating realistic dirt oatterns:
- **simplex noise method** - 
- **perlin noise method** - 
- **white noise method** - 

these patterns of dirt are then overlayed with a wall structure. I
During experiments, to simplify the training, only the simplex noise method was used (so that environments wouldn't differ so much between episodes).

## Cleaning room mechanics

After inputting a broom movement (of form (`x1, y1, x2, y2`)) the environment follows the steps:

1. Create a rectangle P whose midpoint is `(x1, y1)` and the midpoint opposite to it is `(x2, y2)`
2. Extend P by a rectangle F on the side of midpoint `(x2, y2)` and scale down both rectangles towards `(x1, y1)` until they do not collide with any wall
3. Calculate the total amount of dirt D inside P
4. If D does not exceed the capacity of broom:
    1. Transport the dirt to F 
    2. If F's capacity is exceeded, delete the residuals 
5. Otherwise:
    1. Construct rectnagles L, R on the remaining sides of P (apart from the side opposite to F)
    2. If they collide with any wall, crop them to the largest scale that does not produce any collision
    3. Transport the dirt to R, L and F.
    4. If R, L or F's capacity is exceeded, delete the residual dirt

## DDPG 

The whole implementation was created based on [these tutorials](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG).