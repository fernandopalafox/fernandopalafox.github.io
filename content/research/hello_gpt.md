---
title: "GPT + Tiny Shakespeare Dataset"
tags: [wh, ml]
publishDate: 2023-11-20
draft: false
---

Last weekend's hack was spent working through [Andrej Karpathy's intro to generative pre-trained transformers](https://www.youtube.com/watch?v=kCc8FmEb1nY) (GPTs). 

Here's the repo to the code: [link](https://github.com/fernandopalafox/hello_gpt) and a sample generative output given a newline character after 30' of training (on my poor laptop's single GTX 1070):

```
Oppon the time of our mistress! Bolingbroke
To the seal'd shepherd of honour-flutted
Return to me,ignity o'er throat up on ours.

GREY:
Go, good Margaret, thou artius on, do not forth.

STANLEY:
Sir, at thou dost not stand in this world,
In hopest ourselves and smiles, for wholen
Then I be grave died, though with salt long Henry,
The a quarrel of Juliet shall war knock.

RICHAMOP:
Art thou, palace.

KING RICHARD II:
At Henry, farewell! Prince Edward.

DUKE VINCENTIO:
Come, come hither, my lord.
```

# Thoughts
It seems like instrumental to the success of the transformer was a combination of two things: 
1. Encoding of domain knowledge into the model architecture. In the case of language, the transformer architecture allows the model to understand the sequential nature of language and the connection between context and semantics through the masked attention mechanism.
2. Scalability. The architecture is amenable to scaling. See [The Bitter Lesson.](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

As to how this relates to my research, it gets me thinking about how I can bake domain knowledge about robotics and multi-agent interactions into a scalable neural network architecture. Mega broad question, and many people are probably asking themselves the same thing, I'm sure. But hey, at least I now have a better idea of where to go next. 

It also gets me thinking about what kind of scaling architectures will be most useful in robotics. Incidentally, [here's a nice blog post by Nishanth Kumar about scaling in robotics](https://nishanthjkumar.com/Will-Scaling-Solve-Robotics-Perspectives-from-CoRL-2023/). 

# What's next
- More coding! I thought of doing more LeetCode, but I think I'll just go straight into something a bit more aligned to what I'm interested in. So, for now I'm going to work through all the practical examples of [François Fleuret's deep-learning course](https://fleuret.org/dlc/)
- At some point might be a good idea to revisit the transformer architecture and maybe make a cool diagram/flowchart out of it. I wasn't satisfied by the ones I found online. 
- Continue to get better at control theory and game theory. To do this, I think I'll join my lab-mate [Kushagra Gupta](https://clearoboticslab.github.io/people/kushagra_gupta/index.html) in reading and working through Bertsekas' [*Dynamic Programming and Optimal Control*](https://www.mit.edu/~dimitrib/dpbook.html) or perhaps his new book [*Reinforcement Learning and Optimal Control*](https://www.mit.edu/~dimitrib/RLbook.html).