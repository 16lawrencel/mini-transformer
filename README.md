# Mini-transformer
This repo is my attempt at reproducing the [transformer model](https://arxiv.org/pdf/1706.03762.pdf)
on the tiny-shakespeare dataset.

I loosely used Andrej Karpathy's [GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) as
a reference, but I implemented everything from scratch.

## Setup instructions
This (decoder-only) transformer model was trained on the tiny-shakespeare
dataset (found [here](https://huggingface.co/datasets/karpathy/tiny_shakespeare)).

Download the tiny-shakespeare dataset and extract to "./datasets/tiny_shakespeare/data.txt".

The IPython notebook can then be run directly. It might take a while to generate the
byte-pair encodings initially (~30 mins for me).

It took me around 9-10 hours to train 3000 epochs on a Macbook M3 Max.

## Example generated text
Here's some sample generated text after training for 3000 epochs:

>>>
    KING RICHARD III:
    The queen, my love the south the all that shines upon that consorted with
    Shall I had spoke of those sinks hath offended the bending twigoves,
    And sit my true, my sovereign and jump is warm;
    Ere he be he do; and all, and by the shore of importance on the sun
    and admiration, rages and wakes.
    Ha, not.

>>>
    He dies, the vicoes; but gentle Claudio's sleep than the lips,
    That now on Thursday next way above his household harmony and supply?
    Wasly,
    Bound sadly upon;
    And his person perish,
    His neck under it is, never seen now begins the sharpest foes are displace;
    Since you no brother, and go we may she doth not that there's use a forward
    And bid God send to protect his idle the lord.
