---
title: AI Rules of Discourse 
tags: research, note
---

> Disclaimer: This note is a stream of consciousness where I try to crystallize a fleeting idea into a concrete research question or finding. Keep that in mind when reading it!

Should there be rules of discourse in world of widespread, heterogenous, personal AI assistants? To answer this question, we probably need to understand how these AI assistants will interact with each other. I can think of a couple options: 
1. They speak to each other, as humans would. 
2. (vague) As LLM's begin to a form a completely new computing paradigm as suggested by [Karpathy](https://twitter.com/karpathy/status/1707437820045062561), they may directly interface with each other, but not through human language we're used to. Perhaps through some sort of latent space representation? 

If they speak to each other directly, my initial reaction is that some kind of moderation framework would definitely be useful. However, upon further examination, I'm probably looking at the problem too much through the lense of a human interaction. Why do we impose rules on how people should interact? In real-world interactions, we have rules of discourse because without them many interactions would likely end in gross misunderstandings (not good for either party) and perhaps even violence. In online interactions, we impose rules to avoid offending people (that's basically the only reason why, no?). In both cases, rules give users, on average, a more pleasant experience. So if we assume that it's LLM's that are talking to each other for the purpose of serving their respective human operators (is that the right word?), do we really need to worry about them getting offended? Perhaps, we should. Depending on the data they were trained on, a "rude" response from one of the LLMs might provoke an undesireable response in the other LLM, snowballing the interaction into something that's useless or undesireable for both operators. But the responsibility of how to deal with this would probably end up in the hands of the invidual LLM owners: if your LLM can't hang, get a better one. 

The question is then: does it ever make sense to introduce a third party into an interaction between two AI agents with the goal of mediating the interaction?

## Related research ideas
- How to robusitfy LLMs. Probably closely related to the red-teaming LLms to find vulnerabilities. Probably a busy field.
- AI to AI communication. Is written language the most efficient way of having LLMs communicate with each other? Perhaps we can leverage a lower-dimensional latent space. Have them communicate with each other in latent space, and then convert everything to human language when necessary. I clearly don't know what I'm talking about lol. 
- What about having an LLM simulate an interaction with another LLM so that it knows how to act best.
- AI negotiator. In the context of my research on inverse game theory, we observe an interaction between players, assume it's a Nash equilbrium for a game, infer the cost/constraint paramters of the game, and then solve for a strategy that will result in a Nash equiilibrium. In the context of interacting LLMs: 
  1. LLM A observes an interaction between Players B and C 
  2. Based on the interaction, it infers preferences and constraints.
  3. LLM A uses inferred preferences/constraints to acts in a way that will maximize its payoff, while accounting for the fact that Players B and C will react to LLM A's actions.

The last step naively be done using some kind of self-prompting chain. A more interesting approach would be to model the interaction mathematically. The Red Teaming paper looks relevant in this. 

“Everything is vague to a degree you do not realize till you have tried to make it precise.” - Bertrand Russell

## Random references
- [Nash equilibria for an evolutionary language game](https://www.researchgate.net/publication/12284156_Nash_equilibria_for_an_evolutionary_language_game)
- [Multi-Task Deep Learning Games: Investigating Nash Equilibria and Convergence Properties](https://www.mdpi.com/2075-1680/12/6/569)
- This one looks super relevant: [RED TEAMING GAME: A GAME-THEORETIC FRAMEWORK FOR RED TEAMING LANGUAGE MODELS](https://browse.arxiv.org/pdf/2310.00322.pdf)
- Reference in that paper [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://browse.arxiv.org/pdf/1711.00832.pdf). I might have some work in common in my guy Julien Perolate. I gotta dig into this. 