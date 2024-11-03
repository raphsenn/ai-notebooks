# Environments

| Environment Type | |
| --------------------------- | ---------------------------------------------------------------------- |
| Accessible vs. inaccessible | Are the relevant aspects of the environment accessible to the sensors? |
| Deterministic vs. stochastic | Is the next state of the environment completely determined by the current state and the selected action? If only actions of other agents are nondeterministic, the environment is called strategic. |
| Episodic vs. sequential | Can the quality of an action be evaluated within an episode (perception + action), or are future developments decisive for the evaluation of quality? |
| Static vs. dynamic | Can the environment change while the agent is deliberating? If the environment does not change but if the agent’s performance score changes as time passes by the environment is denoted as semi-dynamic. |
| Discrete vs. continuous | Is the environment discrete (chess) or continuous (a robot moving in a room)? |
| Single agent vs. multi-agent | Which entities have to be regarded as agents? There are competitive and cooperative scenarios |

#### Examples of Environments

| Task | Observable | Deterministic | Episodic | Static | Discrete | Agents |
| ----------------- | -------------- | ------ | ------ | --------- | -------| --------- | 
| Crossword puzzle | Observable | Deterministic | Episodic | Static | Discrete | Agents |
| Medical diagnosis | Observable | Deterministic | Episodic | Static | Discrete | Agents |