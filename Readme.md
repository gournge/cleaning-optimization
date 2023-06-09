# TODO:

# Filip

- [ ] train agent on 35x35
    - [ ] memorization
    - [ ] replay

- [ ] requirements.txt
- [ ] design CNN architectures on other room sizes
- [ ] explore tensorboard

## Wojtek

- [ ] encapsulate roomgeneration in a class
- [ ] document code (comments, docstrings)
- [ ] return type of rooms generated should be `np.array`


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

Name conventions in this project describe actions related to cleaning a room with a broom. It's simpler to visualize the objective this way. Interpreting collective movement through the perspective of using a broom also allows us to further generalize the results returned by the model to tasks involving discontinous values such as value of 1 standing for a large box, value 0.6 for a medium one and 0.2 for a small package.

The model atttempts to find an optimal sequence of broom movements to clean a room with a given shape. The sequence is found by sequentially applying the NEAT algorithm (```CleaningModel.py```), which has been trained on random rooms (```RoomGeneration.py```) first with naively chosen mounds of dirt (by applying clustering algorithms - intuitvely we should sweep the broom to where dirt is mostly clustered.) After the NEAT algorithm has been developed, a module (```PlanMoundsModel.py```) for planning mounds (mounds in the room mechanics act like holes in the floor consuming the dirt) is trained on random rooms and based on evaluations of the NEAT Cleaning Model. 
