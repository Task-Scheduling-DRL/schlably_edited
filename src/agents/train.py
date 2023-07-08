"""
This file provides functions to train an agent on a scheduling-problem environment.
By default, the trained model will be evaluated on the test data after training,
by running the test_model_and_heuristic function from test.py.

Using this file requires a training config. For example, you have to specify the algorithm used for the training.

There are several constants, which you can change to adapt the training process:
"""
# OS imports
import argparse
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

# Functional imports
from src.agents import intermediate_test
from src.agents import test
from src.environments.environment_loader import EnvironmentLoader
from src.utils.file_handler.data_handler import DATA_DIRECTORY
from src.utils.file_handler.model_handler import ModelHandler
from src.data_generator.task import Task
from src.utils.logger import Logger
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data


def final_evaluation(config: dict, data_test: List[List[Task]], logger: Logger):
    """
    Evaluates the trained model and logs the results

    :param config: Training config
    :param data_test: Dataset with instances to be used for the test
    :param logger: Logger object

    :return: None

    """
    # Create wandb artifact from local file to store best model with wandb
    best_model_path = ModelHandler.get_best_model_path(config)
    logger.log_wandb_artifact({'name': 'agent_model', 'type': 'model'}, file_path=best_model_path.with_suffix('.pkl'))

    # test model
    # create anv and agent
    agent = get_agent_class_from_config(config)
    best_model = agent.load(file=best_model_path, config=config, logger=logger)
    evaluation_results = test.test_model_and_heuristic(config=config, model=best_model, data_test=data_test,
                                                       logger=logger, plot_ganttchart=False, log_episode=True)

    # log the metric which you find most relevant (this should be used to optimize a hyperparameter sweep)
    success_metric = evaluation_results['agent'][config.get('success_metric')]
    logger.record({'success_metric': success_metric})
    logger.dump()

    # log evaluation to wandb
    logger.write_to_wandb_summary(evaluation_results)


def training(config: dict, data_train: List[List[Task]], data_val: List[List[Task]], logger: Logger) -> None:
    """
    Handles the actual training process.
    Including creating the environment, agent and intermediate_test object. Then the agent learning process is started

    :param config: Training config
    :param data_train: Dataset with instances to be used for the training
    :param data_val: Dataset with instances to be used for the evaluation
    :param logger: Logger object used for the whole training process, including evaluation and testing

    :return: None

    """
    # create Environment
    env, _ = EnvironmentLoader.load(config, data=data_train)

    # create Agent model
    # python -m src.agents.train -fp training/dqn/config_job3_task4_tools0.yaml 입력 시 dqn.py의 DQN class 가져오는 거 확인 완료
    agent = get_agent_class_from_config(config)(env=env, config=config, logger=logger)

    # create IntermediateTest class to save new optimum model every <n_test_steps> steps
    inter_test = intermediate_test.IntermediateTest(env_config=config,
                                                    n_test_steps=config.get('intermediate_test_interval'),
                                                    data=data_val, logger=logger)

    # Actual "learning" or "training" phase
    agent.learn(total_instances=config['total_instances'], total_timesteps=config['total_timesteps'],
                intermediate_test=inter_test)

# 참고용 예시 코드
# 실제론 hyperparameter 중 lambda만 최적화하면 됨
"""
def objective(trial):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 최적화할 하이퍼파라미터들의 구간 설정
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    gamma = trial.suggest_float("gamma", 0.9, 1.0)
    epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.9, 0.99)
    memory_size = trial.suggest_int("memory_size", 10000, 100000)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    target_update = trial.suggest_int("target_update", 1, 10)
    episodes = trial.suggest_int("episodes", 100, 500)

    agent = DQNAgent(state_dim, action_dim, memory_size, batch_size, lr, gamma, epsilon_start, epsilon_end, epsilon_decay)

    # 밑의 study.optimize()에서 각 하이퍼파라미터 조합에 대하여 이 return 값이 가장 큰(성능이 가장 좋은) 하이퍼파라미터를 선택
    # 여기서는 return 값이 전체 episode 중 마지막 10개의 episode들의 reward의 평균으로 설정했으나 나중에 주제/목적에 맞게 커스터마이징
    return agent.train(episodes, target_update)
"""

def main(config_file_name: dict = None, external_config: dict = None) -> None:
    """
    Main function to train an agent in a scheduling-problem environment.

    :param config_file_name: path to the training config you want to use for training
        (relative path from config/ folder)
    :param external_config: dictionary that can be passed to overwrite the config file elements

    :return: None
    """

    # get config and data
    config = load_config(config_file_name, external_config)
    data = load_data(config)

    # create logger and update config
    logger = Logger(config=config)
    config = logger.config

    # Random seed for numpy as given by config
    np.random.seed(config['seed'])

    # train/test/validation data split
    split_random_seed = config['seed'] if not config.get('overwrite_split_seed', False) else 1111
    train_data, test_data = train_test_split(
        data, train_size=config.get('train_test_split'), random_state=split_random_seed)
    test_data, val_data = train_test_split(
        test_data, train_size=config.get('test_validation_split'), random_state=split_random_seed)

    # log data
    logger.log_wandb_artifact({'name': 'dataset', 'type': 'dataset',
                               'description': 'job_config dataset, split into test, train and validation',
                               'metadata': {'train_test_split': config.get('train_test_split'),
                                            'test_validation_split': config.get('test_validation_split')}},
                              file_path=DATA_DIRECTORY / config['instances_file']
                              )
    """
        이 부분을 hyperparamter opmization에 맞게끔 수정해야함

        lambda
    """
    # training
    training(config=config, data_train=train_data, data_val=val_data, logger=logger)

    # evaluate results
    final_evaluation(config=config, data_test=test_data, logger=logger)


def get_perser_args():
    """Get arguments from command line"""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Train Agent in Production Scheduling Environment')

    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # get config_file from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path

    main(config_file_name=config_file_path)
