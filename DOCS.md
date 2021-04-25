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

## Design Decisions

This section describes the solution and outlines any design decisions. While not all of the solution is optimal,
the trade-offs have been described where possible.

### Nomenclature

- Within the codebase, I use the terminology of `predictor` to refer to a ML model so as not to overload the use of the
term `model` which refers to data models in `pydantic`.

### Pre-processing
The service has been designed to accept multi scale / channel / format images and will transform it to match the
deployed model requirements. Deployed models can operate either on grayscale or RGB images.

### Modelling uncertainty

In order to quantify model uncertainty, we can't naievely use the softmax probabilities. Intuitively, this is because
the softmax function "forces" the model to make a choice (by normalising the output logits to 1). This means that
examples that lie outside the training distribution can be interpreted as high confidence (if we make the assumption
that the softmax output is a probability). The paper [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf) has a more detailed explaination of this.

This paper describes a solution where we can run our model multiple times over the same input while applying dropout
to approximate a Bayesian approach. The trade-off with this approach is that we must run many forward passes over the
same input to produce a single prediction.

Another popular approach is to use a Variational Auto-Encoder (VAE) to model the training dataset distribution and
measure uncertainty as the ability to effectively re-generate unseen data.

In general, I think I could do more research into generating an uncertainty measure, the current approach
(running multiple samples with dropout) requires some assumptions to be encoded into the architecture at training time
and also impacts serving performance quite heavily.

### Load test

The load test was run locally on my machine with 12 cores and 32G memory. These are the results of the load test with
50 users. It is worth noting that the current solution is not exactly optimised for serving. I think a further
investigation into generating uncertainty measures within the model is required.

```
 Name                                                          # reqs      # fails  |     Avg     Min     Max  Median  |   req/s failures/s
--------------------------------------------------------------------------------------------------------------------------------------------
 POST /v1/predict                                                1373    35(2.55%)  |    2096     143    6315    2000  |    9.46    0.24
--------------------------------------------------------------------------------------------------------------------------------------------
 Aggregated                                                      1373    35(2.55%)  |    2096     143    6315    2000  |    9.46    0.24

Response time percentiles (approximated)
 Type     Name                                                              50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 POST     /v1/predict                                                      2000   2500   2800   3000   3500   4100   4700   5200   6200   6300   6300   1373
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 None     Aggregated                                                       2000   2500   2800   3000   3500   4100   4700   5200   6200   6300   6300   1373

Error report
 # occurrences      Error
--------------------------------------------------------------------------------------------------------------------------------------------
 35                 POST /v1/predict: ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
--------------------------------------------------------------------------------------------------------------------------------------------
```

A production system would be deployed using an orchestration system like Kubernetes. This would handle load balancing
and horizontal autoscaling to provide another layer of scalability.

## Future work

See the project [issues](https://github.com/martinabeleda/ai-auto-challenge/issues) for a description of any future
enhancements.
