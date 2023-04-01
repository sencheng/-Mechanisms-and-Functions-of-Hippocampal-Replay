## **Modeling mechanisms and functions of hippocampal replay** ##

Simulation code for the computational study by Diekmann and Cheng (2023) published in eLife:\
A model of hippocampal replay driven by experience and environmental structure facilitates spatial learning\
https://doi.org/10.7554/eLife.82301

----------------------------

**Install Requirements**

* First, download and install CoBeL-RL:

    `https://github.com/sencheng/CoBeL-RL`
    
* Install python packages required for plotting:

    `pip3 install -r requirements.txt`

------------------------------

**Demo Guide**  

Demos can be found in the 'demo' directory.
Run the SFMA demo to ensure that you installed CoBeL-RL properly.

*  Navigate to `demo/`.

*  Run the script `sfma_demo.py` to run the SFMA demo.

*  Run the script `deep_sfma_demo.py` to run the Deep SFMA demo.

**Note**: The Deep SFMA demo requires Tensorflow 2 to be installed.

------------------------------

**Simulations Guide**  

Simulations can be found in the 'simulations' directory.
The following sections will guide you through how to run specific simulations.

------------------------------

<details>
<summary>
Learning Simulation
</summary>

*  Navigate to `simulations/learning/`.

*  Run the script `make_environments.py` to generate the gridworld environments.

*  Run the script `simulation_learning.py` to run the learning simulation.

*  Run the script `analyze_learning.py` to analyze and plot the learning simulation data.

</details>

------------------------------

<details>
<summary>
Optimality Simulation
</summary>

This simulation is identical to the learning simulation but only runs
simulations for the SFMA agent and records the optimality of updates.

*  Navigate to `simulations/optimality/`.

*  Run the script `make_environments.py` to generate the gridworld environments.

*  Run the script `simulation_optimality.py` to run the optimality simulation.

*  Run the script `analyze_optimality.py` to analyze and plot the optimality simulation data.

</details>

------------------------------

<details>
<summary>
Sequences Simulation
</summary>

This simulation is identical to the learning simulation but only runs
simulations for the SFMA agent and records the replays.

*  Navigate to `simulations/sequences/`.

*  Run the script `simulation_sequences.py` to run the sequences simulation.

*  Run the script `analyze_sequences.py` to analyze and plot the sequences simulation data.

*  Run the script `plot_replays.py` to plot the replays.

</details>

------------------------------

<details>
<summary>
Random Walk Simulation
</summary>

Simulation of the experiments by Stella et al. (2019).

*  Navigate to `simulations/random_walk/`.

*  Run the script `precompute_occupancies.py` to generate occupancy data.

*  Run the script `simulation_random_walk.py` to run the random walk simulation.

*  Run the script `simulation_distribution.py` to generate data for the replay starting point and displacement distribution analysis.

*  Run the script `analyze_random_walk.py` to analyze and plot the random walk simulation data.

*  Run the script `analyze_distribution.py` to analyze and plot the replay starting points and displacement distribution.

</details>

------------------------------

<details>
<summary>
Shortcuts Simulation
</summary>

Simulation of the experiments by Gupta et al. (2010).

*  Navigate to `simulations/shortcuts/`.

*  Run the script `simulation_shortcuts_static.py` to run the shortcuts simulation.

*  Run the script `simulation_shortcuts_learning.py` to run the shortcuts learning simulation.

*  Run the script `simulation_shortcuts_strengths.py` to record experience strengths.

*  Run the script `analyze_shortcuts_static.py` to analyze the shortcuts simulation data.

*  Run the script `analyze_shortcuts_learning.py` to analyze and plot the shortcuts learning simulation data.

*  Run the script `plot_shortcuts_static.py` to plot the analyzed shortcuts learning simulation data.

*  Run the script `plot_shortcuts_static_replays.py` to plot detected shortcut replays.

*  Run the script `plot_shortcuts_strengths.py` to plot the experience strengths.

</details>

------------------------------   

<details>
<summary>
Dynamic Simulation
</summary>

Simulation of the experiments by Widloski and Foster (2022).

*  Navigate to `simulations/dynamic/`.

*  Run the script `simulation_dynamic.py` run the dynamic simulation.

*  Run the script `analyze_dynamic.py` to analyze and plot the dynamic simulation data.

*  Run the script `compare_dynamic.py` to compare results for different experience similarity metrics.

</details>

------------------------------  

<details>
<summary>
Preplay Simulation
</summary>

Simulation of the experiments by Ólfasdóttir et al. (2015).

*  Navigate to `simulations/preplay/`.

*  Run the script `simulation_preplay.py` run the preplay simulation.

*  Run the script `analyze_preplay.py` to analyze the preplay simulation data.

*  Run the script `analyze_arms.py` to plot the analyzed preplay simulation data.

*  Run the script `simulation_learning.py` run the preplay learning simulation.

*  Run the script `analyze_learning.py` to analyze and plot the preplay learning simulation data.

</details>

------------------------------  

<details>
<summary>
Reward Simulation
</summary>

Simulation for demonstrating the effect of reward on replay statistics.

*  Navigate to `simulations/reward/`.

*  Run the script `simulation_reward.py` run the reward simulation.

*  Run the script `analyze_reward.py` to analyze and plot the reward simulation data.

</details>

------------------------------ 

<details>
<summary>
Aversive Simulation
</summary>

Simulation of the experiments by Wu et al. (2017).

*  Navigate to `simulations/aversive/`.

*  Run the script `simulation_aversive.py` run the aversive simulation.

*  Run the script `analyze_aversive.py` to analyze and plot the aversive simulation data.

</details>

------------------------------ 

<details>
<summary>
Reward Change Simulations
</summary>

Simulation of the experiments by Ambrose et al. (2016).

*  Navigate to `simulations/reward_change/`.

*  Run the script `simulation_effect_of_reward.py` run the effect of reward simulation.

*  Run the script `analyze_effect_of_reward.py` to analyze and plot the effect of reward simulation data.

</details>

------------------------------ 

<details>
<summary>
Goal Change Simulation
</summary>

Simulations for the learning performance following reward changes in the different replay modes.

*  Navigate to `simulations/goal_change/`.

*  Run the script `simulation_changingGoal.py` run the goal change simulation.

*  Run the script `analyze_changingGoal.py` to analyze and plot the goal change simulation data.

</details>

------------------------------ 

<details>
<summary>
Detailed Replay Simulation
</summary>

Simulation recording replay step-by-step (for demonstration purposes).

*  Navigate to `simulations/detailed_replay/`.

*  Run the script `simulation_detailed_replay.py` run the detailed replay simulation.

*  Run the script `plot_detailed_replay.py` to plot step-by-step replay.

</details>

------------------------------

<details>
<summary>
Modes Simulation
</summary>

Simulation demonstrating the possible replay modes.

*  Navigate to `simulations/modes/`.

*  Run the script `simulation_modes.py` run the modes simulation.

*  Run the script `plot_modes.py` to plot examples for experience similarity and generated replays.

</details>

------------------------------

<details>
<summary>
Non-local Simulations
</summary>

Simulation demonstrating non-local online (awake) replay.

*  Navigate to `simulations/nonlocal/`.

*  Run the script `simulation_gupta.py` run the non-local simulation in the figure-8 maze.

*  Run the script `simulation_open_field.py` run the non-local simulation in an open field.

*  Run the script `analyze_gupta.py` to analyze and plot the replays in the figure-8 maze.

*  Run the script `analyze_open_field.py` to analyze and plot the replays in an open field.

</details>

------------------------------

<details>
<summary>
PMA Gain Simulation
</summary>

Simulation demonstrating how adjusting the gain computation in the model by Mattar and Daw (2018) affects replay.

*  Navigate to `simulations/pma_gain/`.

*  Run the script `simulation_distribution.py` run the PMA Gain simulation.

*  Run the script `analyze_replay.py` to analyze and plot the replays.

</details>

------------------------------
