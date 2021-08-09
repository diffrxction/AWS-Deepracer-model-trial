### Evaluation Data
AWS DeepRacer leverages Amazon SageMaker to train the model behind the scenes and uses AWS RoboMaker to simulate the agent's interaction with the environment.

#### Reward function code used:
```python
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track
    
    return float(reward)
```

The graph shows how the agent (car model) behaves in the chosen environment, as prescribed by the reward function.

![Reward graph for the training of the model(Reward vs Iterations)](https://user-images.githubusercontent.com/65293175/128729280-3dce4e25-7d96-4956-a4c7-168c32b4603c.png)

For my first AWS DeepRacer model, I used a very complex track (Environment simulation Circuit de Barcelona-Catalunya) and set the training data at 60 minutes.

The hyperparameters were set as follows:
|Hyperparameter                                                       |	Value|
|---------------------------------------------------------------------|-----------------------|
|Gradient                                                             | descent batch size	64
|Entropy	                                                            | 0.01|
|Discount factor	                                                    | 0.999|
|Loss type	                                                          | Huber|
|Learning rate	                                                      |0.0003|
|Number of experience episodes between each policy-updating iteration | 20|
|Number of epochs	                                                    | 10|
