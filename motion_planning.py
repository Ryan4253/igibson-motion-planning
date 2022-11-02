import argparse
import logging
import os
import sys
from sys import platform

import numpy as np
import yaml

import igibson
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.constants import ViewerMode
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

import random
import time
import igibson.external.motion.motion_planners.rrt_star

import matplotlib as plt

### CURRENT GOAL
# finsih data collection, 15 scenes, 100 ea, 4 algs (avg, stddev)
# save some of the results / process, compile into slide / presentation
#minimal viable product - get some results, then think about how to refine

# Robot Placement: [x, y, z], [roll, pitch yaw] (degree)

# Bounds of different scenes - [minX, maxX, minY, maxY]
bounds = {
    "Rs_int" : [-3.5, 1.5, -3.8, 3],
    "Beechwood_0_int" : [-11, 2.5, -7, 6.5],
    "Beechwood_1_int" : [-11, 2.5, -7.5, 10.5],
    "Benevolence_0_int" : [-4, 1.5, -7.5, 0],
    "Benevolence_1_int" : [-4, 2, -8.5, 2],
    "Benevolence_2_int" : [-4, 2, -8.5, 2],
    "Ihlen_0_int" : [-5, 5, -2, 11],
    "Ihlen_1_int" : [-5, 5, -2, 11],
    "Merom_0_int" : [-3, 5, -2, 10],
    "Merom_1_int" : [-3, 5, -2, 10],
    "Pomaria_0_int" : [-13, 1, -3.5, 4],
    "Pomaria_1_int" : [-14, 1, -4, 4],
    "Pomaria_2_int" : [-7.5, 1, -3, 5.5],
    "Wainscott_0_int" : [-5, 8, -6.5, 14],
    "Wainscott_1_int" : [-5, 8, -6.5, 14]
}

def calculateIterationCount(config, scene = "Rs_int", algorithm = "rrt_star", headless = False):
    # Initialize Configs
    config_data = yaml.load(open(config, "r"), Loader = yaml.FullLoader)
    config_data["load_object_categories"] = ["bottom_cabinet"]
    config_data["load_room_types"] = ["living_room"]
    config_data["hide_robot"] = False

    # Initialize Environment
    env = iGibsonEnv(
        config_file = config_data,
        scene_id = scene,
        mode = "gui_interactive" if not headless else "headless",
        action_timestep = 1.0 / 120.0,
        physics_timestep = 1.0 / 120.0,
    )   
    
    for obj in env.scene.get_objects():
        if obj.category == "bottom_cabinet":
            obj.states[object_states.Open].set_value(True)

    # Iniitalize Motion Planner
    motion_planner = MotionPlanningWrapper(
        env,
        algorithm,
        optimize_iter = 10,
        full_observability_2d_planning = True,
        collision_with_pb_2d_planning = False,
        visualize_2d_planning = not headless,
        visualize_2d_result = not headless,
    )

    # Initialize simulation viewer if not headless
    if not headless:
        env.simulator.viewer.initial_pos = [-0.8, 0.7, 1.7]
        env.simulator.viewer.initial_view_direction = [0.1, -0.9, -0.5]
        env.simulator.viewer.reset_viewer()

    # Find random legal target pose
    minX, maxX = bounds[scene][0], bounds[scene][1]
    minY, maxY = bounds[scene][2], bounds[scene][3]
    while(True):
        x = random.random() * (maxX - minX) + minX
        y = random.random() * (maxY - minY) + minY
        theta = random.random() * 360
        targetPose = [x, y, theta]
        print("Candidate Target Pose Found: ", targetPose)
        if(env.test_valid_position(env.robots[0], [targetPose[0], targetPose[1], 0], [0, 0, targetPose[2]])):
            print("Valid Target Pose Found: ", targetPose)
            break
    
    # Initialize Robot
    env.land(env.robots[0], [0, 0, 0], [0, 0, 0])
    env.robots[0].tuck()

    # Generate path
    attempt, max_attempts = 0, 15
    while(True):
        if attempt == max_attempts:
            logging.error("MP failed after {} attempts. Exiting".format(max_attempts))
            sys.exit()
        
        attempt+=1
        plan, it = motion_planner.plan_base_motion(targetPose)

        if plan is not None and len(plan) > 0:
            break
        else:
            logging.error(
                "MP couldn't find path to the base location. Attempt {} of {}".format(attempt, max_attempts)
            )

    # Follow Path
    motion_planner.dry_run_base_plan(plan)

    # Give time to look at the plan if headless
    if headless:
        time.sleep(5)

    # Close Environment
    env.close()

### Generates a random pose within the bounds of the scene
def getRandomPose(scene):
    # Find random legal target pose
    minX, maxX = bounds[scene][0], bounds[scene][1]
    minY, maxY = bounds[scene][2], bounds[scene][3]

    x = random.random() * (maxX - minX) + minX
    y = random.random() * (maxY - minY) + minY
    theta = random.random() * 360
    
    return [x, y, theta]

def generatePath(env, algorithm, targetPose, visualize):
    motion_planner = MotionPlanningWrapper(
        env,
        algorithm,
        optimize_iter = 10,
        full_observability_2d_planning = True,
        collision_with_pb_2d_planning = False,
        visualize_2d_planning = visualize,
        visualize_2d_result = True
    )

    plan, it = motion_planner.plan_base_motion(targetPose)

    if(plan is None and it == 10000):
        print("FAILED TO FIND PATH!")

    #if(not visualize and plan is not None):
        #time.sleep(30)
    return plan, it

### 
# config - environment configuration
# scene - name of the scene
# algorithm - algorithm to test
# visualize - whether to turn on visualization
# trials - number of trials to sample
#
# Return - a list containing the iteration count for each trials
def samplePaths(config, scene = "Rs_int", algorithm = "rrt_star", visualize = True, trials = 1):
    # Initialize Configs
    config_data = yaml.load(open(config, "r"), Loader = yaml.FullLoader)
    config_data["load_object_categories"] = ["bottom_cabinet"]
    config_data["load_room_types"] = ["living_room"]
    config_data["hide_robot"] = False

    # Initialize Environment
    env = iGibsonEnv(
        config_file = config_data,
        scene_id = scene,
        mode = "headless",
        action_timestep = 1.0 / 120.0,
        physics_timestep = 1.0 / 120.0,
    )   

    result = []
    while(len(result) < trials):
        while(True):
            robotPose = getRandomPose(scene)
            if(env.test_valid_position(env.robots[0], [robotPose[0], robotPose[1], 0], [0, 0, robotPose[2]])):
                env.land(env.robots[0], [robotPose[0], robotPose[1], 0], [0, 0, robotPose[2]])
                env.robots[0].tuck()
                break
        
        targetPose = getRandomPose(scene)
        path, it = generatePath(env, algorithm, targetPose, visualize)
        if(path == None):
            continue

        result.append(it)
    
    print(result)


def main(selection = "user", headless = False, short_exec = False):
    if not (selection != "user" and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            "-c",
            default = os.path.join(igibson.configs_path, "fetch_motion_planning.yaml"),
            help = "which config file to use [default: use yaml files in examples/configs]",
        )
        parser.add_argument(
            "--programmatic",
            "-p",
            dest = "programmatic_actions",
            action = "store_true",
            help = "if the motion planner should be used with the GUI or programmatically",
        )
        args = parser.parse_args()
        config = args.config
        programmatic_actions = args.programmatic_actions
    else:
        config = os.path.join(igibson.configs_path, "fetch_motion_planning.yaml")
        programmatic_actions = True

    samplePaths(config, "Benevolence_1_int", "rrt_star", False, 100)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
