# Cell-free AP deployment

Saved experiment parameters, model checkpoints, training records, and deployment visualizations.

## Included artifacts

Twenty distinct runs cover A2C, DDPG, PPO, SAC, and TD3, with the `1t1r` and `2t2r` antenna settings and available seeds 1, 128, and 256. The original run names encode a target-circle, max-mean setting, a 50 m region, a 10.0 m parameter, one target, and three users. The 10.0 m parameter is retained without assigning a meaning not established by the supplied files.

| File | Contents |
| --- | --- |
| `hyperparameter.json` | Learning rate, batch size, network dimensions, discount factor, and related settings |
| `act*.pth`, `actor.pt`, `cri*.pth` | Actor/critic, target-network, and optimizer checkpoints |
| `recorder.npy`, `recorder_my.csv` | Recorded training history |
| `best_location_origin.npy`, `recorder_best_location.csv` | Deployment-location artifacts |
| `Deployment.png`, `LearningCurve.jpg` | Deployment and learning-curve visualizations |

## Organization

```text
experiments/<algorithm>/<antenna-setting>/seed-<seed>/
```

The supplied archive contained 32 run directories. Twelve were byte-identical copies; this collection retains all 20 distinct runs. [source-directory-map.csv](source-directory-map.csv) records the mapping. File names and parameter values inside each run are unchanged.

## Reproduction scope

These are saved experiment artifacts, not the training implementation. No Python/MATLAB training scripts, environment definitions, model class definitions, or dependency file were included in the supplied archive. The original implementation is required to instantiate checkpoint architectures and reproduce training. No checkpoint was executed or deserialized during preparation of this release.

The CSV and JSON files can be inspected directly. NumPy numeric arrays can be inspected with `numpy.load(path, allow_pickle=False)`. Checkpoints should only be loaded in a trusted environment with the corresponding model definitions.
