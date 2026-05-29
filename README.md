# pokepy-engine

This is a fork of a project begun by sethkarten/pokepy-engine, maintained and heavily extended by the metamon team.

`pokepy-engine` is an experimental project that aims to serve an important niche for Pokémon RL projects like `metamon`, but probably should not be used for any other purpose.

<div align="center">
    <img src="pokepy/psyduck_is_pleased.png" alt="Psyduck is Pleased" width="200">
</div>

Turning Showdown into an RL environment is very hard, and there are countless state ambiguities. Showdown is not intended for RL training and is far slower than you would expect. Leading RL projects navigate around these problems by spending years accumulating Showdown battle datasets to train agents that intentionally assume the state of the battle is difficult to know for certain. Pokémon is so complicated that starting over and making a fast sim from scratch that is 1:1 with the ever-changing Showdown sim is a massive time commitment. The value in overcoming these challenges has been obvious for years, but the short list of projects with the Pokémon expertise and time to make it happen tend to overvalue sim accuracy *in the specific context of model-free RL*. At this point, baselines and datasets have grown strong enough to compensate for sim2sim gaps. The tolerance for inaccuracies has never been higher and coding agents have never been better. The result is this repo: an almost entirely vibe-coded, inaccurate Python simulator that is faster and easier to vectorize than Showdown. It is used to generate mass-scale online RL data and climb the diminishing rate of return towards high-level gameplay, where niche sim mechanics are not a serious bottleneck, before using ground-truth Showdown data to plug the leaks.
