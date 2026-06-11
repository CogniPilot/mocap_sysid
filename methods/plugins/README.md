# Method Plugins

New benchmark methods should live in `methods/plugins/<method_name>/`.

Each plugin must include:

```text
method.json
method.py
README.md
```

`method.json` declares the method metadata:

```json
{
  "name": "MyMethod",
  "entry_point": "method:MyMethod",
  "description": "Short method description.",
  "model_families": ["aircraft6dof"],
  "observation_types": ["mocap"],
  "training_scenarios": ["aircraft_6dof_aggressive"],
  "requires_gpu": false
}
```

The class named by `entry_point` must implement:

```python
def fit(self, train_data, config):
    ...

def rollout(self, fitted, validation_data, config):
    ...
```

Validation rollouts must use only the validation initial condition and pilot-command input history. They must not assimilate validation measurements unless the method is explicitly entered in a measurement-assimilating category.

Run the plugin smoke check:

```bash
python3 -m benchmark.smoke_plugin methods/plugins/example_linear
```

The current paper-scale suite still dispatches existing methods through `comparison_suite.py`. The plugin contract is the public interface that new methods should target as the benchmark runner is refactored.

## Result Generation Policy

Method PRs should contain method code and documentation only. Do not include regenerated benchmark CSV files, figures, LaTeX generated files, or website JSON in the method PR. After review and merge, a maintainer will run the full benchmark on trusted hardware and commit result artifacts separately.
