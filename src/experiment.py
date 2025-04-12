import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

import os
import sys
import csv

# Get the current working directory
base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

print(base_dir)

from utils.utils import display_experiment, display_metrics, online_evaluation
from policies.models.contextual import ContextualModel
from utils.loader import (
    get_data_by_name,
    get_estimator_by_name,
    get_policy_from_type,
    verify_settings,
)

import argparse


def experiment(args):

    settings = {
        "policy_type": args.policy_type,
        "contextual_modelling": args.contextual_modelling,
        "estimator": args.estimator,
        "n_0": args.n_0,
        "validation": args.validation,
        "pdf_type": args.pdf_type,
        "epsilon": args.epsilon,
        "clipping_parameter": args.clipping_parameter,
        "lambda_": args.lambda_,
        "variance_penalty": args.variance_penalty,
        "seed": args.seed,
        "data_name": args.data_name,
        "display": args.display,
    }
    verify_settings(settings)

    # %%
    lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    rng = np.random.RandomState(settings["seed"])
    data = get_data_by_name(settings["data_name"], settings["seed"])
    display_experiment(settings["seed"], data, settings["estimator"])

    # %%
    # Model setting
    contextual_model = ContextualModel(
        settings["contextual_modelling"], settings["seed"]
    )
    start_param = contextual_model.create_start_parameter(data)
    # create the policy that instantiate the estimator
    policy = get_policy_from_type(settings["policy_type"])(
        parameter=start_param,
        contextual_model=contextual_model,
        random_seed=settings["seed"],
        type=settings["pdf_type"],
        epsilon=settings["epsilon"],
        sigma=data.logging_scale,
        log_sigma=data.start_sigma,
    )

    # define the estimator
    estimator = get_estimator_by_name(settings["estimator"])(
        policy,
        settings["clipping_parameter"],
        settings["lambda_"],
        settings["variance_penalty"],
    )

    actions, contexts, losses, propensities, targets = data.generate_data(
        settings["n_0"]
    )

    print("Optimizing policy...")
    optimized_theta, offline_loss = estimator.optimize(
        actions, contexts, losses, propensities, seed=42
    )
    print("***Policy is optimized!***")
    optimized_theta_val = optimized_theta._value

    online_loss = online_evaluation(
        optimized_theta_val, contextual_model, data, settings["seed"]
    )

    # compute the optimal theta
    # if settings['policy_type'] == "continuous":
    optimal_theta, pistar_determinist = data.get_optimal_parameter(
        settings["contextual_modelling"]
    )
    optimal_loss = online_evaluation(
        optimal_theta, contextual_model, data, settings["seed"]
    )
    # else:
    #     pi0, pistar = make_baselines_skylines(data.X_train, data.y_train)
    #     optimal_loss = data.online_evaluation_star(pistar)
    offline_loss = offline_loss._value
    regret = online_loss - optimal_loss
    display_metrics(offline_loss, online_loss, regret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scripts for the evaluation of methods"
    )
    parser.add_argument(
        "--n_0",
        nargs="?",
        type=int,
        default=100,
        help="initial number of samples",
    )
    parser.add_argument(
        "--policy_type",
        nargs="?",
        default="continuous",
        choices=["discrete", "continuous"],
        help="policy type",
    )
    parser.add_argument(
        "--estimator",
        nargs="?",
        default="poem",
        choices=["poem", "cpme"],
        help="estimator type",
    )
    parser.add_argument(
        "--contextual_modelling",
        nargs="?",
        default="linear",
        choices=["linear", "polynomial"],
        help="contextual modelling type",
    )
    parser.add_argument(
        "--validation",
        nargs="?",
        type=bool,
        default=False,
        help="validation flag",
    )
    parser.add_argument("--pdf_type", nargs="?", default=None, help="PDF type")
    parser.add_argument(
        "--epsilon", nargs="?", type=float, default=0, help="epsilon value"
    )
    parser.add_argument(
        "--clipping_parameter",
        nargs="?",
        type=float,
        default=1e-4,
        help="clipping parameter",
    )
    parser.add_argument(
        "--lambda_", nargs="?", type=float, default=1e-4, help="lambda value"
    )
    parser.add_argument(
        "--variance_penalty",
        nargs="?",
        type=bool,
        default=False,
        help="variance penalty flag",
    )

    parser.add_argument("--seed", nargs="?", type=int, default=42, help="seed value")
    parser.add_argument(
        "--data_name",
        nargs="?",
        default="advertising",
        choices=["yeast", "pricing", "tmc2007", "advertising"],
        help="environment name",
    )
    parser.add_argument(
        "--display", nargs="?", type=bool, default=False, help="display flag"
    )

    args = parser.parse_args()
    experiment(args)
