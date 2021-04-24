AI Automation Engineer Challenge
================================

## Problem Statement

You are given a production model `mnist_model` in tensorflow `SavedModel` format. Your task is to come up with a
solution that serves the model in our production pipeline.

Please note that the incoming requests are images of different shapes. Your server should return a prediction result
with an indicative score that shows how uncertain/certain the model is. Note that this is different from the raw score returned by the model - we would like to know whether the model views the raw score it has returned as being reliable or not. There are a few ways you could do this, each with different impacts on the performance of the system you are building.

Factors to consider might include,

- target incoming request (100 QPS, or a reasonable number you could think of)
- scalability (based on what?)
- required unit resource (RAM/CPU/GPU)
- ways to decouple model prediction and result serving (why are we/are we not doing this)
- CI/CD (what should we focus on?)

You should at least,

- explain why you choose what you choose
- follow good dev practice
- use docker to containerise your solution
- test & document

It would be nice to have,

- a diagram demonstrating your solution


## Evaluation

You will evaluated based on

- clarity of thinking
- dev skills
- ML knowledge

## Instruction

Fork this repository and start making commits/branches.

You should spend 4-6 hours max on this task. Though you could spend more, it is not encouraged.

Anything unclear, please feel free to contact us.
