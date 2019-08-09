from trainer import Trainer
import yaml

if __name__ == "__main__":
    config = {
        "experiment_path": "experiments/",
        "model": "",
        "optimizer": "",
        "data": {
            "root_dir": "data/",
            "filters": {
                "arteries": ["LAD"],
                "viewpoint_index_step": 3
            },
            "groups": {
                1: ["NORMAL"],
                0: ["<25%"]
            }
        }
    }
    trainer = Trainer(config)
    trainer.run()
    # with open('config.yaml', 'r') as f:
    #     config = yaml.load(f)
    # print(config)

    # with open('config.yml', 'w') as outfile:
    #     yaml.dump(config, outfile, default_flow_style=False)