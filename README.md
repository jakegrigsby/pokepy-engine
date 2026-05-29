# pokepy-engine

This is a fork of `sethkarten/pokepy-engine` that has been extended to support `metamon`. It is an experimental project that aims to serve an important niche for Pokémon RL efforts... but probably should not be used for any other purpose.

<div align="center">
    <img src="pokepy/psyduck_is_pleased.png" alt="Psyduck is Pleased" width="200">
</div>

Turning Showdown into an RL environment is very hard, and there are countless state ambiguities. Showdown is not intended for RL training and is far slower than you would expect. Leading RL projects navigate around these problems by spending years accumulating Showdown battle datasets to train agents that assume the state of the battle is difficult to know for certain. A faster sim would be very useful, but making a fast and *accurate* sim (1:1 with Showdown) is a massive project. Fortunately, baselines and datasets have grown strong enough to compensate for sim2sim gaps. The tolerance for inaccuracies has never been higher and coding agents have never been better. The result is this repo: an almost entirely vibe-coded, inaccurate Python simulator that is faster and easier to vectorize than Showdown. It is used to generate mass-scale online RL data and climb the diminishing rate of return towards high-level gameplay, where niche sim mechanics are not a serious bottleneck, before using ground-truth Showdown data to plug the leaks.
