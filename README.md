## **Modeling mechanisms and functions of hippocampal replays** ##

----------------------------

**Install requirements**

    
* Clone the project

* Install the CoBeL-RL framework by moving the `cobel` folder into the site-packages directory of your Python installation

* Install the Python modules listed in `.../cobel/requirements.txt`


**Getting started**


* Go to the demo folder and run the demo script (preferably use an IDE like Spyder):

    `.../demo/sfma_demo.py`

------------------------------

**Run Simulations**

This guide details the specific order in which the scripts for different simulations have to be run.
Avoid import errors by using an IDE like Spyder. Some of the simulations will require long running times and disk space!

<details></summary>
<summary>

**Simulation `learning` - Spatial Navigation (Learning Performance)**
</summary>
This simulation will generate the results/plots shown in Figure 3.

* Run script `.../simulations/learning/make_environments.py`

* Run script `.../simulations/learning/simulation_learning.py`

* Run script `.../simulations/learning/analyze_learning.py`

</details>

<details></summary>
<summary>

**Simulation `optimality` - Spatial Navigation (Replay Optimality)**
</summary>
This simulation will generate the results/plots shown in Figure 3.

* Run script `.../simulations/optimality/make_environments.py`

* Run script `.../simulations/optimality/simulation_optimality.py`

* Run script `.../simulations/optimality/analyze_optimality.py`

</details>

<details></summary>
<summary>

**Simulation `sequences` - Spatial Navigation (Replay Directionality)**
</summary>
This simulation will generate the results/plots shown in Figure 3.

* Run script `.../simulations/sequences/simulation_sequences.py`

* Run script `.../simulations/sequences/analyze_sequences.py`

</details>

<details></summary>
<summary>

**Simulation `randomWalk` - Random Walk**
</summary>
This simulation will generate the results/plots shown in Figure 4.

* Run script `.../simulations/randomWalk/precompuate_occupancies.py`

* Run script `.../simulations/randomWalk/simulation_randomWalk.py`

* Run script `.../simulations/randomWalk/simulation_distribution.py`

* Run script `.../simulations/randomWalk/analyze_randomWalk.py`

* Run script `.../simulations/randomWalk/analyze_distribution.py`

</details>

<details></summary>
<summary>

**Simulation `shortCuts` - Shortcuts**
</summary>
This simulation will generate the results/plots shown in Figure 5.

* Run script `.../simulations/shortCuts/simulation_shortCuts_static.py`

* Run script `.../simulations/shortCuts/simulation_shortCuts_learning.py`

* Run script `.../simulations/shortCuts/analyze_shortCuts_static.py`

* Run script `.../simulations/shortCuts/analyze_shortCuts_learning.py`

* Run script `.../simulations/shortCuts/plot_shortCuts_static.py`

* Run script `.../simulations/shortCuts/plot_shortCuts_static_replays.py`

</details>

<details></summary>
<summary>

**Simulation `dynamic` - Adaptive Replay**
</summary>
This simulation will generate the results/plots shown in Figure 6.

* Run script `.../simulations/dynamic/simulation_dynamic.py`

* Run script `.../simulations/dynamic/analyze_dynamic.py`

* Run script `.../simulations/dynamic/compare_dynamic.py`

</details>

<details></summary>
<summary>

**Simulation `preplay` - Preplay**
</summary>
This simulation will generate the results/plots shown in Figure 7.

* Run script `.../simulations/preplay/simulation_preplay.py`

* Run script `.../simulations/preplay/analyze_preplay.py`

* Run script `.../simulations/preplay/analyze_arms.py`

* Run script `.../simulations/preplay/simulation_learning.py`

* Run script `.../simulations/preplay/analyze_learning.py`

</details>

<details></summary>
<summary>

**Simulation `unorderedReplay` - Changing Goal**
</summary>
This simulation will generate the results/plots shown in Supplementary Figure 7.

* Run script `.../simulations/unorderedReplay/simulation_changingGoal.py`

* Run script `.../simulations/unorderedReplay/analyze_changingGoal.py`

</details>
