# AtlasTune

AtlasTune is a multi-agent database optimizer, designed to improve performance by ensuring cooperation between otherwise distinct components of the database tuning process, namely, the index selector and the knob tuner. Both the index selector and knob tuner use reinforcement learning to converge on optimal table index configurations and database system knob parameters. In particular, the index selector utilizes a Deep Q-Network (DQN) while the knob tuner uses a Deep Deterministic Policy Gradient (DDPG). Both components achieve tuning cooperation by sharing states and interleaved training schedules.


A more in-depth overview of AtlasTune's underlying theory and implemention can be found in [this](https://github.com/Ajax12345/atlastune/blob/main/supporting_files/atlastune_overview.pdf) presention.