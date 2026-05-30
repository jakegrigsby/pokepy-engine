# pokepy-engine

This is a fork of `sethkarten/pokepy-engine` that has been extended to support [`metamon`](https://github.com/UT-Austin-RPL/metamon).

<div align="center">
    <img src="pokepy/psyduck_is_pleased.png" alt="Psyduck is Pleased" width="200">
</div>

Pokémon Showdown is not intended for RL training and is too slow for online RL to be viable. Leading RL projects navigate around these problems by spending years accumulating Showdown battle datasets to train agents that assume the state of the battle is difficult to know for certain. A faster sim would be very useful, but making a fast and *accurate* sim (1:1 with Showdown) is a massive project. Fortunately, baselines and datasets have grown strong enough to compensate for sim2sim gaps. The tolerance for inaccuracies has never been higher and coding agents have never been better. The result is this repo: an almost entirely vibe-coded, inaccurate Python simulator that is faster and easier to vectorize than Showdown. It is used to generate mass-scale RL data and propel training runs to high-level gameplay, where sim accuracy starts to matter, and ground-truth Showdown data can be used to plug the leaks.
