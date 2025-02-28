# PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks

Matthew Chang, Gunjan Chhablani, Alexander Clegg, Mikael Dallaire Cote, Ruta Desai, Michal Hlavac, Vladimir Karashchuk, Jacob Krantz, Roozbeh Mottaghi, Priyam Parashar, Siddharth Patki, Ishita Prasad, Xavier Puig, Akshara Rai, Ram Ramrakhya, Daniel Tran, Joanne Truong, John M. Turner, Eric Undersander, Tsung-Yen Yang

Work done at FAIR Meta, Alphabetical author order

We present a benchmark for Planning And Reasoning Tasks in humaN-Robot collaboration (PARTNR) designed to study human-robot coordination in household activities. PARTNR tasks exhibit characteristics of everyday tasks, such as spatial, temporal, and heterogeneous agent capability constraints. We employ a semi-automated task generation pipeline using Large Language Models (LLMs), incorporating simulation in the loop for grounding and verification. PARTNR stands as the largest benchmark of its kind, comprising 100,000 natural language tasks, spanning 60 houses and 5,819 unique objects. We analyze state-of-the-art LLMs on PARTNR tasks, across the axes of planning, perception and skill execution. The analysis reveals significant limitations in SoTA models, such as poor coordination and failures in task tracking and recovery from errors. When LLMs are paired with real humans, they require 1.5x as many steps as two humans collaborating and 1.1x more steps than a single human, underscoring the potential for improvement in these models. We further show that fine-tuning smaller LLMs with planning data can achieve performance on par with models 9 times larger, while being 8.6x faster at inference. Overall, PARTNR highlights significant challenges facing collaborative embodied agents and aims to drive research in this direction.

Date: November 4, 2024 Code: <https://github.com/facebookresearch/partnr-planner> Website: <https://aihabitat.org/partnr>

# 1 Introduction

Imagine a domestic robot that collaborates with humans in daily activities using natural language, akin to human-to-human interactions. This scenario requires two key features: the dynamic collaboration between the robot and the human, and the use of natural language for interaction. Current benchmarks in embodied AI typically satisfy one or the other condition; either robots operate in isolation [(Shridhar et al.,](#page-13-0) [2020;](#page-13-0) [Zhu et al.,](#page-14-0) [2023;](#page-14-0) [Krantz et al.,](#page-12-0) [2020;](#page-12-0) [Majumdar et al.,](#page-12-1) [2024)](#page-12-1), or tasks are not specified in natural language [(Yenamandra](#page-13-1) [et al.,](#page-13-1) [2023;](#page-13-1) [Puig et al.,](#page-13-2) [2024;](#page-13-2) [Szot et al.,](#page-13-3) [2023;](#page-13-3) [Jain et al.,](#page-12-2) [2020)](#page-12-2). Despite significant progress in the field of embodied AI, there remains a gap in realistic benchmarks that evaluate robots in collaborative settings. To bridge this gap, we introduce Planning And Reasoning Tasks in humaN-Robot collaboration (PARTNR), a novel benchmark that evaluates the ability of embodied AI agents to collaborate with humans across a range of household activities in simulated indoor environments (Figure [1)](#page-1-0).

PARTNR consists of 100,000 natural language instructions paired with tailored evaluation functions, focusing on four task types: (1) constraint-free, where sub-tasks can be completed in any manner by either agent, (2) spatial tasks that contain spatial constraints, (3) temporal tasks that require ordered execution, and (4) heterogeneous tasks that include actions that cannot be completed by one of the agents. Beyond the conventional challenges of long-horizon planning, novel partially observed environments, and large state and action spaces, PARTNR emphasizes the need for effective collaboration dynamics, such as task division and tracking partner's progress.

Curating such a benchmark of large-scale, natural language tasks with tailored evaluation functions presents significant challenges. Current benchmarks typically rely on either templated tasks [(Shridhar et al.,](#page-13-0) [2020;](#page-13-0) <span id="page-1-0"></span>**PARTNR:** A collaboration dataset for planning and reasoning in multi-agent tasks

Figure 1 We present PARTNR, a benchmark for planning and reasoning in embodied multi-agent tasks, featuring 100,000 everyday tasks and evaluation functions generated semi-automatically, spanning 60 houses and 5,819 unique objects. We analyze LLM-based planning agents and also provide a human-in-the-loop tool to evaluate how agents collaborate with real humans.

[Zhang et al.,](#page-14-1) [2024a)](#page-14-1) or tasks and evaluations crafted by humans [(Mandi et al.,](#page-12-3) [2024;](#page-12-3) [Li et al.,](#page-12-4) [2023a)](#page-12-4), which can restrict the diversity or the scale of the datasets. To overcome this, we propose a semi-automated generation method using Large Language Models (LLMs) with simulation-in-the-loop grounding. First, a Large Language Model (LLM) generates task and evaluation functions, which are grounded in the objects and furniture of a simulated house. Next, we employ simulation-in-the-loop to filter out hallucinations and infeasible instructions, complemented by human annotation to enhance diversity and accuracy. Subsequently, a set of 1,000 verified instructions and evaluation functions, along with diverse simulation houses, are utilized to guide an LLM through in-context prompting to create 100,000 tasks.

As PARTNR consists of natural language tasks and LLMs have shown strong results in planning [(Yao et al.,](#page-13-4) [2023;](#page-13-4) [Ahn et al.,](#page-11-0) [2022;](#page-11-0) [Huang et al.,](#page-11-1) [2022)](#page-11-1), we explore prompting and fine-tuning LLMs, to assess their effectiveness in collaborative scenarios. We study the effect of observability of the environment (i.e., full or partial), centralized vs. decentralized multi-agent control, learned or privileged oracle robot skills, and different ways of grounding the 3D world information for LLM-based planning. Beyond these automated evaluations with synthetic human partners, we also perform evaluations with real humans-in-the-loop, with people performing the task alone, with a human partner, or with an LLM-guided robot partner. Overall, we find that LLMs struggle at coordination, task tracking and dealing with perception and skill errors. While humans are able to solve 93% of PARTNR tasks, SoTA LLMs can only successfully complete 30% under non-privileged conditions. Moreover, in decentralized multi-agent settings, task completion takes 1.3x more steps than single-agent, due to poor tracking of partner actions, resulting in extraneous actions. In contrast, human pairs outperform single humans, in our human-in-the-loop experiments, highlighting potential for improving LLM collaboration strategies. LLMs also struggle to recover from skill failures and perception grounding errors, resulting in lower performance when privileged skills and privileged perception are removed. When comparing model sizes, we observe that a smaller fine-tuned Llama3.1-8B achieves a similar performance to a Llama3.1-70B without finetuning, while being 8.6x faster. This faster inference plays an important role when evaluated with real humans-in-the-loop, where the finetuned model takes fewer steps and offloads more tasks from the human.

In summary, PARTNR enables reproducible, large-scale, and systematic evaluations of embodied agents in a

<span id="page-2-0"></span>

|                                   | Environment | Multi-Agent | Language | Action Space | Task Types | Num tasks |
|-----------------------------------|-------------|-------------|----------|--------------|------------|-----------|
| Overcooked (Carroll et al., 2019) | 2D          | ✓           |          | HL           | C          | 4         |
| RoboGen (Wang et al., 2024)       | 3D          |             | ✓        | LL+HL        | CST        | 106       |
| GenSim (Katara et al., 2023)      | 3D          |             | ✓        | LL           | CS         | 100       |
| RoCo (Mandi et al., 2024)         | 3D          | ✓           | ✓        | LL           | CS         | 6         |
| FurnMove (Jain et al., 2020)      | 3D-S        | ✓           |          | LL           | C          | 30        |
| RoboCasa (Nasiriany et al., 2024) | 3D-S        | ✓           | ✓        | LL           | CST        | 100       |
| ALFRED (Shridhar et al., 2020)    | 3D-S        |             | ✓        | HL           | CST        | 25,743    |
| BEHAVIOR-1K (Li et al., 2023a)    | 3D-M        |             |          | LL+HL        | CST        | 1,000     |
| WAH (Puig et al., 2021)           | 3D-M        | ✓           |          | HL           | C          | 1,211     |
| Co-ELA (Zhang et al., 2024a)      | 3D-M        | ✓           | ✓        | HL           | C          | 44        |
| PARTNR                            | 3D-M        | ✓           | ✓        | LL+HL        | CSTH       | 100,000   |

Table 1 Comparison to similar embodied benchmarks. We compare PARTNR to embodied AI benchmarks, focusing on natural language and multi-agent collaboration tasks. Comparison axes are – Environment: Household single room (S), multi-room (M). Action Space: High-Level Actions (HL), Low-level Actions (LL). Task Types: Constraint-free (C), Spatial (S), Temporal (T), Heterogeneous (H) Num tasks: We measure tasks as the number of unique scene-goal pairs.

wide variety of collaborative scenarios. Through systematic evaluation, we reveal critical insights into the current limitations of LLM-based planners, opening interesting future research directions.

# 2 Related Work

Language-based benchmarks in Embodied AI. A large body of work on language benchmarks in Embodied AI has focused on navigation [(Anderson et al.,](#page-11-3) [2018;](#page-11-3) [Krantz et al.,](#page-12-0) [2020;](#page-12-0) [Chen et al.,](#page-11-4) [2019)](#page-11-4) or Embodied Question Answering [(Das et al.,](#page-11-5) [2018;](#page-11-5) [Majumdar et al.,](#page-12-1) [2024)](#page-12-1) which involve navigation and information gathering but do not require agents to modify their environments. Closer to our work are instruction-following benchmarks [(Shridhar et al.,](#page-13-0) [2020,](#page-13-0) [2021;](#page-13-6) [Puig et al.,](#page-12-8) [2018;](#page-12-8) [Wang et al.,](#page-13-5) [2024;](#page-13-5) [James et al.,](#page-12-9) [2020;](#page-12-9) [Gong](#page-11-6) [et al.,](#page-11-6) [2023)](#page-11-6), where agents interact with environments to complete tasks described via language, though the diversity of tasks is limited. In contrast, we leverage LLMs to generate diverse task definitions and scene initializations, and extend them to multi-agent settings. The idea of scaling up task generation using LLMs has been explored in a few recent works [(Katara et al.,](#page-12-5) [2023;](#page-12-5) [Wang et al.,](#page-13-5) [2024;](#page-13-5) [Xian et al.,](#page-13-7) [2023;](#page-13-7) [Nasiriany](#page-12-6) [et al.,](#page-12-6) [2024)](#page-12-6). However, these works tend to focus on single-agent tasks that span relatively short horizons, while we consider long-horizon, multi-agent problems. Table [1](#page-2-0) compares relevant benchmarks with PARTNR.

Embodied multi-agent benchmarks. Multiple works have proposed embodied multi-agent benchmarks [(Puig](#page-12-10) [et al.,](#page-12-10) [2023;](#page-12-10) [Agashe et al.,](#page-11-7) [2023;](#page-11-7) [Zhang et al.,](#page-14-1) [2024a;](#page-14-1) [Jain et al.,](#page-12-11) [2019;](#page-12-11) [Suarez et al.,](#page-13-8) [2019)](#page-13-8). Many of these benchmarks focus on coordination in simple 2D environments, limiting their applicability to real world settings [(Agashe et al.,](#page-11-7) [2023;](#page-11-7) [Carroll et al.,](#page-11-2) [2019)](#page-11-2). Recent works have developed benchmarks studying collaboration in more realistic environments and activities [(Puig et al.,](#page-12-7) [2021;](#page-12-7) [Zhang et al.,](#page-14-1) [2024a;](#page-14-1) [Jain et al.,](#page-12-11) [2019;](#page-12-11) [Puig et al.,](#page-13-2) [2024;](#page-13-2) [Szot et al.,](#page-13-3) [2023)](#page-13-3), focusing on rearranging objects or furniture in large, partially observable 3D environments [(Puig et al.,](#page-12-7) [2021,](#page-12-7) [2024;](#page-13-2) [Jain et al.,](#page-12-11) [2019;](#page-12-11) [Szot et al.,](#page-13-3) [2023)](#page-13-3), or manipulating objects in a counter-top space [(Mandi et al.,](#page-12-3) [2024)](#page-12-3). However, these benchmarks are typically limited to a predefined and reduced set of tasks, often not described in natural language and primarily involving object rearrangement. In contrast, PARTNR covers an open set of tasks, each described in natural language, requiring agents to rearrange objects with spatial and temporal constraints, as well as requiring heterogeneous actions that can only be done by the human agent, (e.g., washing dishes or turning on the oven).

LLMs for decisionmaking. Several works use LLMs as interactive policies, demonstrating their strong capabilities but also highlighting challenges in grounding them with observations and actions [(Huang et al.,](#page-11-1) [2022;](#page-11-1) [Yao](#page-13-4) [et al.,](#page-13-4) [2023;](#page-13-4) [Ahn et al.,](#page-11-0) [2022;](#page-11-0) [Huang et al.,](#page-11-8) [2023;](#page-11-8) [Zeng et al.,](#page-14-2) [2022)](#page-14-2). Some approaches improve grounding by prompting LLMs with demonstrations and task-specific constraints [(Huang et al.,](#page-11-1) [2022;](#page-11-1) [Yao et al.,](#page-13-4) [2023)](#page-13-4), or by integrating LLMs with external modules for multi-modal reasoning [(Ahn et al.,](#page-11-0) [2022;](#page-11-0) [Huang et al.,](#page-11-8) [2023;](#page-11-8) [Zeng et al.,](#page-14-2) [2022)](#page-14-2). Toolformer [(Schick et al.,](#page-13-9) [2023)](#page-13-9) allows LLMs to call APIs for information retrieval or environmental interaction. For instance, APIs can be used to call low-level policies [(Driess et al.,](#page-11-9) [2023)](#page-11-9), to

<span id="page-3-0"></span>Figure 2 The PARTNR generation pipeline. Task and evaluation generators produce episodes, which are filtered and annotated for correctness. These episodes are then treated as seeds to achieve 100k-scale. Finally, episodes are vetted during human-in-the-loop collection.

leverage VLMs for obtaining the state of the world [(Huang et al.,](#page-11-8) [2023;](#page-11-8) [Zhang et al.,](#page-14-3) [2024b)](#page-14-3), or to another LLM serving as a world model [(Zhao et al.,](#page-14-4) [2024)](#page-14-4). SayPlan [(Rana et al.,](#page-13-10) [2023)](#page-13-10) maintains a persistent graph of the current world-state [(Gu et al.,](#page-11-10) [2024;](#page-11-10) [Werby et al.,](#page-13-11) [2024)](#page-13-11), enabling detailed semantic and geometric queries. Our work synthesizes these ideas by encoding the environment into a graph, using tools to extract relevant information, and executing tasks through motor skills. Another line of work fine-tunes LLMs with data from the target environments, by learning input and output adaptors [(Li et al.,](#page-12-12) [2022;](#page-12-12) [Szot et al.,](#page-13-12) [2024;](#page-13-12) [Xiang et al.,](#page-13-13) [2023)](#page-13-13). We explore low-rank adaptation of LLMs with multi-agent data to enhance coordination and efficiency. While fewer studies focus on LLMs in multi-agent collaboration [(Zhang et al.,](#page-14-1) [2024a;](#page-14-1) [Park](#page-12-13) [et al.,](#page-12-13) [2023;](#page-12-13) [Li et al.,](#page-12-14) [2023b;](#page-12-14) [Zhou et al.,](#page-14-5) [2024)](#page-14-5), one notable example is CoELA [(Zhang et al.,](#page-14-1) [2024a)](#page-14-1) that design collaborative agents, though limited in task diversity. Our work addresses a broader range of tasks and considers agents with varying capabilities, pushing the boundaries of multi-agent system collaboration in more complex scenarios.

# <span id="page-3-1"></span>3 Benchmark Generation

We introduce PARTNR, a benchmark aimed at training and evaluating robots at solving natural language tasks in collaboration with humans. PARTNR covers four types of tasks: (1) Constraint-free tasks, where sub-tasks can be completed in any manner by either agent. For example, "Let's move all dirty plates to the sink." (2) Spatial tasks that require reasoning about the spatial positioning of objects. For instance, "Let's place the books on the shelf next to each other." (3) Temporal tasks, where the sequence in which sub-tasks are executed is important. For example, "Let's remove the candles from the dining table before bringing the plates to the table." (4) Heterogeneous tasks, involving actions that are beyond the robot's capabilities. For example, "Let's wash the dishes before putting them in shelves." In scenarios where the robot's skills do not support washing, completing this task requires reasoning about agent capabilities. Our benchmark consists of natural language instructions and corresponding evaluation functions, both of which are generated at-scale using LLMs. Specifically, we generate 1,000 human-verified instructions and corresponding evaluation functions and use them as in-prompt examples to scale to 100,000 tasks in other scenes with different layouts and objects. A unique aspect of our automatic generation is the integration of an embodied simulator within the generation loop, which significantly reduces LLM errors such as hallucinations and infeasible actions.

<span id="page-4-0"></span>Figure 3 Task and evaluation example. Language tasks have inherent complexity and ambiguity; both of which are supported by the structures of our evaluation functions.

# 3.1 Simulation-in-the-loop task instruction generation

While LLM-based task generation has been studied in literature before [(Katara](#page-12-5) [et al.,](#page-12-5) [2023;](#page-12-5) [Wang et al.,](#page-13-5) [2024;](#page-13-5) [Xian](#page-13-7) [et al.,](#page-13-7) [2023;](#page-13-7) [Nasiriany et al.,](#page-12-6) [2024)](#page-12-6), these generations are not grounded beyond user-created in-context prompts. In PARTNR, we use a simulation-in-theloop generation technique to ground the LLM in the environment, agents and available actions. Specifically, we instantiate a simulation environment in the Habitat 3.0 simulator [(Puig](#page-13-2) [et al.,](#page-13-2) [2024)](#page-13-2), populated with the HSSD dataset [(Khanna et al.,](#page-12-15) [2024)](#page-12-15), consisting of 60 unique houses and 5,819 OVMM objects [(Yenamandra et al.,](#page-13-1) [2023)](#page-13-1). The

<span id="page-4-1"></span>Figure 4 Distribution of task types in PARTNR. The left plot displays the percentage of tasks with each characteristic. Constraint-free tasks by definition exclude the other types. The top right bars correspond to the dot combination below.

simulated house is parsed into a list of rooms and available furniture, and passed to an LLM, along with all available objects. Using this information, the LLM is asked to generate free-form, viable tasks in the scene, along with an initial scene state description. For example, if the generated task is "Clear dishes from the living room", the LLM should generate an initial scene with multiple dishes in the living room. At this stage, additional objects are also added to the scene to create clutter in the environment. Once generated, the tasks, initial states, and clutter are instantiated in the simulator, and infeasible instructions are filtered. For example, if the house does not have a living room, "Clear dishes from the living room" is invalid. Similarly, if the generated task requires actions not supported by the simulator, such as folding, the task is filtered. Generally, the rate of hallucinations is high, leading to a significant number of episodes being discarded. We observe that after filtering for infeasible instructions, the diversity in generated instructions is typically limited. For example, most of the instructions use the same objects (e.g., dishes) or similar rooms (e.g., kitchen or dining room). To increase diversity of the generated tasks, we manually annotate them to ensure task and object diversity, such as maintaining a balanced distribution of constraint-free, spatial, temporal, and heterogeneous tasks by modifying the instructions to elicit specific characteristics. This process results in 1,000 human annotated and simulation-verified tasks.

Such manual annotation is not practical for large-scale generation. Instead, we leverage the human-annotated 1,000 instructions to scale generation by using them as in-prompt examples. We prompt the LLM with both a house description and an example task, and instruct it to modify the task to fit the new house. For example a task like "Clear all dishes from the living room" is modified to "Clear all toys from the bedroom." This allows us to maintain the diversity of the original annotated instruction set, while ensuring a high likelihood of successful instantiation in the simulator. Qualitatively, we filtered or edited ∼90% of free-form generated instructions and only ∼10% of scaled instructions. We use LLama3-70B-Instruct [(Dubey et al.,](#page-11-11) [2024)](#page-11-11) for all instruction generation. Finally, all tasks go through a human-in-the-loop filtering. In this step, humans attempt to solve the tasks using our human-in-the-loop tool (Appendix [A.13)](#page-36-0) and eliminate physically infeasible instructions that are difficult to detect, such as requiring an object to be at two locations. Figure [2](#page-3-0) provides an overview of our pipeline. Details on the generation process can be found in Appendix [A.4](#page-17-0) and prompts in Appendix [A.14.](#page-40-0)

# 3.2 Evaluation function generation

To determine if an agent successfully completed the instruction "Clear all dishes from the living room", we need an evaluation function that can validate the removal of all spoons, forks, and other dishes from any of the living rooms. However, manually annotating all necessary rearrangements and state changes of a task is time intensive and since all tasks are unique, impractical at scale. Similar to instruction generation, we employ an LLM to create an evaluation function that assesses task completion without requiring any manual annotations. Specifically, we leverage the ability of LLMs to generate predicate-based Python programs using three types of APIs: a list of propositions indicating what relations between entities must be satisfied, a set of dependencies indicating when propositions should be queried, and a set of constraints indicating how propositions must be satisfied. We define an expressive vocabulary of each of these components to afford evaluation of all tasks in the benchmark (e.g., Figure [3)](#page-4-0). Closely related evaluation systems include defining tasks in PDDL [(Ghallab et al.,](#page-11-12) [1998)](#page-11-12) or BDDL [(Srivastava et al.,](#page-13-14) [2022)](#page-13-14). We choose to build a new Python-based evaluation system since neither have the expressivity to evaluate PARTNR tasks while maintaining human and LLM interpretability; for instance, BDDL does not support time-varying evaluation. Since PARTNR tasks have temporal dependencies (e.g. multi-step rearrangement), the input to the evaluation function is the complete sequence of simulator states during task execution. The evaluation function returns 3 metrics: (1) Percent Complete (PC ∈ [0, 1]), the % of propositions that are satisfied w.r.t. constraints, (2) Success (S ∈ {True,False}), measuring if a task was successfully completed, defined as S := (PC = 1), and (3) Failure Explanation (FE), a human and LLM interpretable language description of why the agents failed to accomplish the task. See Appendix [A.5](#page-19-0) for details.

We use CodeLLama-70B-instruct [(Roziere et al.,](#page-13-15) [2023)](#page-13-15) for evaluation function generation. Exemplified in Figure [3,](#page-4-0) producing perfect evaluation functions is non-trivial. The LLM must correctly classify the entire space of possible actions against natural language instructions and the specific simulation environment, which can be quite complex. For example, in Figure [3,](#page-4-0) the instruction "set the plants on the shelf" refers to "the shelf", but two shelves exist in the room. The evaluation function must allow either shelf while requiring placement of all plants, and finally account for a next-to relation. Any error or missing value in either a proposition or constraint invalidates the evaluation function. Consequently, we observe a large error rate in LLM generation, particularly pertaining to incorrect propositions and temporal sequencing constraints.

<span id="page-5-0"></span>To alleviate these inaccuracies, we follow a similar semi-automated procedure to instruction generation. We first generate evaluation functions for the 1,000 human-annotated instructions and perform manual annotation to correct them (Appendix [A.6.3)](#page-25-0). This results in a dataset of 1,000 human-verified instruction and evaluation pairs. Next, we generate evaluations for the scaled 100,000 instruction set. Recall that the scaled instructions are generated by prompting the LLM with an example instruction from the annotated set. We retrieve the corresponding annotated evaluation function and prompt the LLM with it. This is similar to approaches such as retrieval-augmented generation [(Lewis et al.,](#page-12-16) [2020)](#page-12-16) and improves the accuracy of evaluation function generation from 50% to 92% as measured through manual inspection (Appendix [A.6.2)](#page-25-1). As a final step, we ask human users to solve all PARTNR tasks using our human-in-the-loop evaluation tool (Appendix [A.13)](#page-36-0). All tasks that cannot be solved by humans over 6 retries (3 single-user, 3 multi-user tries) are deemed infeasible, and removed from the dataset. We find that about 90% of instructions, and 92% of evaluation functions from automated generation are accurate, resulting in a combined generation accuracy of 90 × 92 = 83%.

### 3.3 The PARTNR Dataset

The PARTNR dataset comprises of 100,000 episodes in 37 train scenes, 1,000 episodes in 13 validation scenes, and 1,000 episodes in 10 test scenes from the HSSD dataset [(Khanna et al.,](#page-12-15) [2024)](#page-12-15). After scaled generation, all validation and test set episodes are human annotated for correctness, as well as a 2,000-episode subset of train. See Appendix [A.6.1](#page-23-0) for correctness analysis of scale-generated episodes. Below, we analyze the characteristics and diversity of this dataset.

Characteristics: As described earlier, PARTNR focuses on four task types: constraint-free, spatial, temporal, and heterogeneous. We show the distribution of these task types in the test split in Figure [4;](#page-4-1) validation split is similar. PARTNR evaluates collaboration along these axes both independently and jointly. Secondary characteristics of interest include dependent rearranges (e.g., "Place them on the same table") and multi-step rearrangement of the same object (e.g. "Move the cup to the sink, wash it, then place it in the cabinet"). 7% of tasks include dependent rearranges and 6% include multi-step rearrangement. Tasks average 4.7 propositions to be satisfied (indicative of number of steps required to complete tasks). For analysis of linguistic phenomena and more characteristics, see Appendix [A.3.](#page-15-0)

Diversity: The diversity of tasks in PARTNR is largely enabled by simulation-in-the-loop generation, which utilizes rich HSSD scenes, and the OVMM object set. Consequently, PARTNR tasks reference and require reasoning about 155 unique object types, 20 furniture classes and 13 room types. Note that each instruction, instantiated in each house, brings its own diversity. For example, "move the laptop to the office table", grounds office and table uniquely in each house, as well as different instances of laptop in different instructions. Further discussion can be found in Appendix [A.3.](#page-15-0)

# 4 Experiments and Analysis

We investigate how state-of-the-art planning and perception methods handle natural language tasks in new environments and coordinate with unseen partners using PARTNR. Since PARTNR consists of diverse spatio-temporal tasks specified in language, we primarily use LLMs in our baselines for planning, and study variants in (1) zero-shot prompting, retrieval-augmented generation or fine-tuning, (2) centralized versus decentralized planning, (3) partially versus fully observed environment, (4) learned versus oracle low-level robot skills, and (5) privileged versus non-privileged perception.

Our experiments are conducted in the Habitat 3.0 simulator [(Puig et al.,](#page-13-2) [2024)](#page-13-2) with a simulated Spot robot [(Boston-](#page-11-13)[Dynamics;](#page-11-13) [Yokoyama et al.,](#page-14-6) [2023)](#page-14-6). We adopt a two-layer hierarchical control architecture, similar to [(Puig et al.,](#page-13-2) [2024;](#page-13-2) [Szot et al.,](#page-13-16) [2021)](#page-13-16), as illustrated in Figure [5,](#page-6-0) for the robot and simulated human. At the high level, a planner selects skills from a predefined skill library (e.g., navigate, pick, place, open, close). We also use a textual world graph with a three-layer hierarchy rep-

<span id="page-6-0"></span>Figure 5 Decentralized architecture. The human and robot agents use a 2-layer hierarchical architecture, with high-level LLM planners that call low-level skills. Both agents build a world graph, updated using observations and actions.

resenting rooms, furniture, and movable objects. Each node in the graph stores a semantic category (e.g., kitchen, table or cup), 3D information (e.g., position or bounding box), and states (e.g., clean, powered on). See Appendix [A.7](#page-27-0) and Figure [10](#page-28-0) for details.

# <span id="page-7-0"></span>4.1 Baselines

We evaluate baselines along the following axes:

# 1. Variations of high-level planner:

- Heuristic expert: This approach utilizes expert-designed heuristics and privileged information about the task, environment and evaluation function to pre-plan all steps for human and robot based on their capabilities. For instance, both agents might rearrange objects, but only humans perform cleaning, filling, and toggling on/off tasks.
- Zero-shot ReAct (ReAct): We use ReAct [(Yao et al.,](#page-13-4) [2023)](#page-13-4) with an API library of functions or tools that enable the LLM to take actions. As observation, we provide the LLM with a concise, current world graph description plus a history of actions. The LLM uses this information to choose an action from [ExploreRoom, Navigate, OpenFurniture, CloseFurniture, PickObject, PlaceObject, Wait, Done] for the robot. See Appendix [A.16](#page-56-0) for prompts and Appendix [A.9](#page-31-0) for API details (human and robot).
- ReAct with Retrieval-Augmented Generation (ReAct-RAG): We also evaluate ReAct with RAG [(Lewis](#page-12-16) [et al.,](#page-12-16) [2020)](#page-12-16) to investigate whether examples of planning on similar tasks improves the performance of ReAct. We construct a database of planning examples by collecting the successful traces from ReAct from a the 2,000 task training subset (see [3.3)](#page-5-0). During test time, the most relevant planning trace from the train dataset is selected based on sentence similarity and added to the LLM's prompt [(Pang](#page-12-17) [et al.,](#page-12-17) [2024;](#page-12-17) [Madaan et al.,](#page-12-18) [2024)](#page-12-18).
- Finetuned LLMs (Finetuned): We also investigate finetuning a smaller LLM (Llama3.1-8B) as our high-level planner, using successful traces from the ReAct baselines [(Hsieh et al.,](#page-11-14) [2023)](#page-11-14) that use Llama3.1-70B. Using the React-RAG dataset, we split every episode into a sequence of high-level planning actions, filtering for only actions that were executed successfully. For every action, we build an input containing the world-graph and history of actions, similar to ReAct (see Appendix [A.10](#page-33-0) for more details). We then finetune an LLM to predict the action from the ReAct episode given this input, using a low-rank adapter [(Hu et al.,](#page-11-15) [2021)](#page-11-15). This model has reduced latency and computational demands, suitable for real world deployment.

All model generations are constrained to only output valid actions on observed objects using constrained generation [(Geng et al.,](#page-11-16) [2023)](#page-11-16). The constrained generation greatly reduces the hallucinations and 'grammatical' errors typical of LLMs. An episode is finished when both agents call Done or reach maximum simulation steps or LLM calls. Refer to Appendix [A.9](#page-31-0) for details.

- 2. Centralized versus decentralized planning: To study the overhead of coordination in multi-agent PARTNR tasks, we compare centralized and decentralized planners. In centralized, a single LLM decides actions for both agents, with complete information about both agent's states, effectively removing any need for coordination between the agents. In decentralized, each agent is controlled by a different LLM, and each LLM needs to reason about the other agent's actions.
- 3. Partial versus full observability: To evaluate if SoTA language models can explore new environments and identify task-relevant objects, we consider a partially observed setting where the planner knows the house's layout but not the object locations, requiring exploration. This is in contrast to a fully observed setting, where all object locations are known in advance.
- 4. Learned versus oracle low-level robot skills: We examine the impact of learned neural-network skills versus oracle skills (with privileged simulation information) on overall performance in PARTNR tasks. We create a library of learned skills for pick, place, navigate, open and close actions (Appendix [A.8](#page-30-0) provides more details) and compare performance with oracle skills.
- 5. Privileged versus non-privileged perception: To study perception challenges such as inaccurate detection and approximate localization, we used a non-privileged world graph with modified ConceptGraphs [(Gu](#page-11-10) [et al.,](#page-11-10) [2024)](#page-11-10), built from agents' RGBD observations only. As agents explore and take actions, this world graph is updated using onboard sensing (details in Appendix [A.7)](#page-27-0). In contrast, with privileged perception, this information is available from the simulation.

# 4.2 Results and Analysis

Metrics. We evaluate performance across different settings using four key metrics. First, the simulation steps metric measures the number of steps required for agents to complete the task within the simulation environment, serving as an indicator of efficiency. Second, the success rate reflects the completion of the task i.e. whether 'all' task constraints are satisfied. Given the complexity and long-horizon nature of PARTNR tasks, agents often partially complete the task. To account for this, we also report percent complete, which quantifies the ratio of completed task 'propositions' (percent complete = 1 for successful tasks). Lastly, we assess the reasoning efficiency of the planners through the planning cycles metric, which counts the number of high-level LLM calls each planner makes in the course of an episode. We cap the maximum planner calls at 50 in all experiments.

# 4.2.1 Task Performance Analysis

Table [2](#page-9-0) presents a comprehensive evaluation of the planning approaches defined in Section [4.1](#page-7-0) using the Llama3.1-70B-Instruct model [(Dubey et al.,](#page-11-11) [2024)](#page-11-11) for ReAct baselines, and a finetuned Llama3.1-8B base model for the Finetuned baseline. Since PARTNR tasks are multi-agent, we also need a simulated human partner, which we control with a ReAct approach, using Llama3.1-70B-Instruct. Our main findings are detailed below.

PARTNR tasks are challenging for LLM-based planners. LLM-based baselines across all observability and controllability conditions perform worse than the privileged heuristic expert, due to errors in task tracking (not completing all steps, performing them in the wrong order, or undoing completed steps), and semantic errors (placing objects on the wrong furniture, or moving the wrong object), indicating a gap in LLM task planning.

LLMs struggle with coordination in decentralized settings. Decentralized ReAct baselines which do not have privileged access to partner's intent are significantly slower at task completion than centralized ReAct (3295 steps with decentralized-partial in row(e) versus 2298 with centralized-partial in row(d)). This shows that reasoning about the partner e.g., knowing or inferring partner's intent can improve task efficiency in PARTNR tasks, and current SoTA LLMs perform poorly at this. Moreover, decentralized ReAct with two agents is even slower than ReAct with a single-agent (3295 steps with multi-agent in row(e) versus 2519 with single-agent in row(a)), indicating that LLMs suffer from a significant coordination "burden". This co-ordination burden is further highlighted in our analysis on extraneous effort in Section [4.2.2,](#page-9-1) where we find that agents end up repeating parts of the task or performing irrelevant actions with much higher frequency in decentralized settings.

LLMs are unable to recover from learned skill failures. When replacing oracle skills with learned skills, the success rate decreases from 0.73 to 0.57 (row(e) vs. row (h)). This decline can be attributed to the higher failure rate and increased number of simulation steps required by learned skills compared to privileged oracle skills. The LLMs struggle to recover from skill errors like failing to pick up an object or performing incomplete exploration, resulting in a lower success rate. Future research could investigate training large models with low-level skills in the loop, enabling them to learn recovery and replanning strategies in the face of such failures.

LLMs exhibit a high degree of sensitivity to errors in perception. When we replace privileged perception with off-the-shelf perception modules, success rate significantly declines (from 0.57 with a privileged, partial world graph in row(h) to 0.30 with Concept-Graphs [(Gu et al.,](#page-11-10) [2024)](#page-11-10) in row(i)). LLMs rely heavily on accurate world descriptions provided by the world graph and struggle to correct errors such as misclassification (e.g., shelves misidentified as tables) or incorrect room assignments (e.g., a table in the living room mislabeled as being in the bedroom). Multi-modal models like VLMs might be stronger at recovering from such failures, which we leave for future work.

Finetuned 8B model performs on par with a ReAct with a 70B model, while being 8.6x faster. We find that the finetuned planner with a small 8B model performs on par with ReAct, which uses a much larger 70B model (a 0.73 success rate with the 70B model in row(e), versus 0.70 with the finetuned 8B model in row(g)). At the same time, we find that the finetuned model is 8.6 times faster at inference. This indicates that the finetuning effectively distills task-relevant information from the training set and generalizes to new test tasks. When

<span id="page-9-0"></span>

| Method               | Controllability | Skills  | Observability | Sim<br>Steps ↓    | Success<br>Rate ↑ | Percent<br>Complete ↑ | Planning<br>Cycles ↓ |
|----------------------|-----------------|---------|---------------|-------------------|-------------------|-----------------------|----------------------|
| (a) ReAct-Single     | Single Agent    | Oracle  | Partial       | 2519.02 ± 57.48   | 0.73 ± 0.01       | 0.85 ± 0.01           | 18.68 ± 0.33         |
| (b) Heuristic-Expert | Centralized     | Oracle  | Full          | 1260.88 ± 26.97   | 0.84 ± 0.01       | 0.94 ± 0.01           | N/A                  |
| (c) ReAct            | Centralized     | Oracle  | Full          | 1347.43 ± 33.80   | 0.74 ± 0.01       | 0.88 ± 0.01           | 17.49 ± 0.34         |
| (d) ReAct            | Centralized     | Oracle  | Partial       | 2298.13 ± 61.39   | 0.74 ± 0.01       | 0.85 ± 0.01           | 20.73 ± 0.51         |
| (e) ReAct            | Decentralized   | Oracle  | Partial       | 3295.20 ± 76.27   | 0.73 ± 0.01       | 0.86 ± 0.01           | 15.24 ± 0.31         |
| (f) ReAct + RAG      | Decentralized   | Oracle  | Partial       | 3467.47 ± 82.39   | 0.71 ± 0.01       | 0.84 ± 0.01           | 14.75 ± 0.31         |
| (g) Finetuned        | Decentralized   | Oracle  | Partial       | 3228.96 ± 75.14   | 0.70 ± 0.01       | 0.84 ± 0.01           | 12.85 ± 0.24         |
| (h) ReAct            | Decentralized   | Learned | Partial       | 6494.88 ± 181.52  | 0.57 ± 0.02       | 0.76 ± 0.01           | 22.72 ± 0.58         |
| (i) ReAct            | Decentralized   | Learned | ConceptGraph  | 12490.80 ± 208.90 | 0.30 ± 0.01       | 0.56 ± 0.01           | 23.84 ± 0.45         |

Table 2 Analysis of planner baselines in various settings. We compare performance using simulation steps, success rate and percent complete on the tasks, and the average planning cycles used by the baselines (described in Section [3)](#page-3-1).

deployed with humans-in-the-loop, the finetuned model takes fewer steps and offloads more sub-tasks than the 70B model (see Table [3)](#page-10-0).

#### <span id="page-9-1"></span>4.2.2 Analysis of Collaborative Behavior and Efficiency

Our analysis in Table [2](#page-9-0) revealed challenges in LLM collaboration, prompting a deeper investigation into specific collaborative behaviors, explained below and detailed in Appendix [A.12](#page-34-0) and Table [12.](#page-36-1)

Robots offload up to 60% of tasks. We evaluate the robot's ability to offload tasks from the human, measuring the ratio of sub-tasks performed by the robot to the total sub-tasks in successful PARTNR tasks. Despite similar success rates between single- and multi-agent (0.73 vs. 0.74), the robot offloads about 60% of sub-tasks in decentralized multi-agent, reducing human effort (Table [12)](#page-36-1).

Decentralized agents are prone to performing extraneous tasks. The agents sometimes end up performing sub-tasks that are not useful for the task such as rearranging an object that is not required by the task or repeating a sub-task already performed by the other agent. To capture such extraneous agent effort, we measure the portion of agent actions that did not increase the percent complete metric i.e., did not contribute to task progress, over the total number of successful agent actions in an episode. We find a 300% increase in extraneous effort in decentralized multi-agent settings compared to single-agent (Table [12)](#page-36-1), indicating a significant coordination burden.

Temporal and heterogeneous tasks are challenging for LLMs. LLMs struggle in temporal and heterogeneous tasks. Task success drops by 27% for temporal tasks and 20% for heterogeneous tasks compared to constraint-free tasks for ReAct (Table [13)](#page-36-2). This highlights the limitations of LLMs in reasoning about agent capabilities and following strict ordering constraints.

### 4.3 Human-in-the-loop Evaluation

We build on the human-in-the-loop infrastructure from Habitat 3.0 [(Puig et al.,](#page-13-2) [2024)](#page-13-2) and adapt it to a serverclient architecture, with the server hosted on AWS capable of supporting multiple clients (see Appendix [A.13)](#page-36-0). This allows us to run at-scale evaluation of our tasks with 129 non-expert human participants. We collect single-user and multi-user data on 1000 tasks from the validation and test set using this tool. In the single-user setting, a single participant completes the whole task, by driving the human agent in the simulator via keyboard/mouse controls Figure [14](#page-38-0) in appendix shows our HITL interface. In multi-user, two participants complete the task together by controlling a human and a robot agent. The goal of these experiments is to study multi-user dynamics at PARTNR tasks, and see if multiple humans collaborating are more efficient than single human. Finally, we run a human-AI experiment where a human participant collaborates with a robot controlled by an LLM (using the ReAct and Finetuned models from Section [4.1)](#page-7-0). This experiment aims to evaluate LLM-controlled agents at collaborating with unseen, real humans. Table [3](#page-10-0) shows the success rate (SR) and percent complete (PC) of tasks from the validation set in a single-user, multi-user, human-ReAct and human-Finetuned setting. Additionally, we measure the number of steps taken by each approach to complete

<span id="page-10-0"></span>

| Method          | Success     | Percent     | Sim             | Task         | Exploration     | Extraneous  |
|-----------------|-------------|-------------|-----------------|--------------|-----------------|-------------|
|                 | Rate ↑      | Complete ↑  | Steps ↓         | Offloading ↑ | Efficiency ↓    | Effort ↓    |
| Single-user     | 0.93 ± 0.01 | 0.96 ± 0.00 | 3046.99 ± 80.79 | N/A          | 2459.22 ± 26.75 | 0.09 ± 0.01 |
| Multi-user      | 0.93 ± 0.01 | 0.96 ± 0.00 | 2369.55 ± 49.33 | 0.59 ± 0.01  | 1762.47 ± 13.99 | 0.15 ± 0.01 |
| Human-ReAct     | 0.91 ± 0.01 | 0.96 ± 0.02 | 4267.71 ± 83.40 | 0.16 ± 0.01  | 2624.39 ± 26.05 | 0.12 ± 0.01 |
| Human-Finetuned | 0.92 ± 0.01 | 0.96 ± 0.00 | 3443.33 ± 61.46 | 0.26 ± 0.01  | 2164.94 ± 21.31 | 0.13 ± 0.01 |

Table 3 Human-in-the-Loop Evaluation. We evaluate the performance of a 2-person human team and human-LLM teams, comparing them to solo human performance on PARTNR tasks using metrics described in Section [4.1.](#page-7-0) Additional results and analysis in Appendix [A.13.](#page-36-0)

the task, and the ratio of work completed by the robot i.e., task offloading. We also measure exploration efficiency in human-in-the-loop, by measuring the steps taken to pick the first object, and extraneous effort, indicating actions that were not useful for task completion. These results are summarized in Table [3.](#page-10-0) Some key findings are below (more results and analysis in [A.13.4)](#page-39-0):

Humans are significantly better than LLMs at PARTNR tasks. In both single and multiple human environments, the success rate achieved is 0.93 on the PARTNR benchmark. In contrast, the ReAct model without any privileged information, achieves a significantly lower success rate of 0.30 (row (i) of Table [2)](#page-9-0). This highlights a significant gap in the performance of LLMs in planning tasks. Note that the LLM baselines like ReAct and Finetuned achieve a success rate of 0.92 and 0.91 when evaluated with real humans (Table [3)](#page-10-0), because humans are able to adapt to LLM mistakes. On the other hand, the simulated human in Table [2](#page-9-0) is an LLM, which is unable to recover from partner mistakes.

Finetuned LLMs perform better than ReAct when coordinating with real humans. When deployed with real humans-in-the-loop, the finetuned model is faster than ReAct at task completion (3443 steps with finetuned versus 4267 with ReAct). It is also able to offload more tasks from humans than ReAct (26% with finetuned as compared to 16% with ReAct). This reflects that smaller models with faster inference can improve human experience in real-world deployment.

LLMs struggle at coordination, hampering human performance. Despite the Finetuned being faster than ReAct when collaborating with humans, both approaches are slower than the human doing the task alone. In contrast, two humans working together complete the task faster than a single human (2369 steps vs. 3046 with multi- and single-user respectively). This result is in line with the automated evaluation we observed in Table [1,](#page-2-0) where multi-agent LLMs are also slower than a single-agent LLM. This result further reinforces that LLMs suffer at coordination; while humans are able to coordinate and divide tasks between each other, decentralized LLMs are unable to do so.

LLMs are able to offload tasks from humans. Despite the aforementioned increase in the number of steps for task completion, robots guided by the finetuned model successfully offload 26% of tasks from humans. This indicates that LLMs can still offer assistance when collaborating with real human partners. Nonetheless, there remains significant potential for improvement.

# 5 Conclusion

We present PARTNR, a benchmark for reasoning and planning in multi-agent embodied tasks, featuring 100,000 natural language tasks instantiated in 60 simulated, multi-room houses with 5,819 unique objects. We use a semi-automated LLM-powered pipeline for large-scale instruction and evaluation function generation that uses simulation-in-the-loop grounding. PARTNR exhibits characteristics of everyday tasks, such as temporal and spatial constraints, and allows systematic evaluation of planning approaches. We find a significant gap between SoTA LLMs and human-level performance at PARTNR tasks. While our best LLM baseline only succeeds at 30% of tasks with no privileged information, humans are able to solve 93% of the tasks. Moreover, LLMs face challenges in coordinating with both LLM-based agents and real human partners. Human-in-the-loop evaluations, involving real humans collaborating with LLM-guided robots, reveal that LLM-guided partners decrease human efficiency compared to working solo. This suggests that LLM-based agents require significant improvements to become effective collaborative partners in embodied tasks. PARTNR serves as a challenging benchmark that highlights the substantial limitations of current models.

# References

- <span id="page-11-7"></span>Saaket Agashe, Yue Fan, Anthony Reyna, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models. arXiv, 2023.
- <span id="page-11-17"></span>Christopher Agia, Krishna Murthy Jatavallabhula, Mohamed Khodeir, Ondrej Miksik, Vibhav Vineet, Mustafa Mukadam, Liam Paull, and Florian Shkurti. Taskography: Evaluating robot task planning over large 3d scene graphs. In CoRL, 2022.
- <span id="page-11-0"></span>Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can and not as i say: Grounding language in robotic affordances. In CoRL, 2022.
- <span id="page-11-3"></span>Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In CVPR, 2018.
- <span id="page-11-13"></span>BostonDynamics. Spot robot. <https://www.bostondynamics.com/products/spot>. Accessed: 2024-10-01.
- <span id="page-11-2"></span>Micah Carroll, Rohin Shah, Mark K Ho, Tom Griffiths, Sanjit Seshia, Pieter Abbeel, and Anca Dragan. On the utility of learning about humans for human-ai coordination. In NeurIPS, 2019.
- <span id="page-11-4"></span>Howard Chen, Alane Suhr, Dipendra Misra, Noah Snavely, and Yoav Artzi. Touchdown: Natural language navigation and spatial reasoning in visual street environments. In CVPR, 2019.
- <span id="page-11-5"></span>Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In CVPR, 2018.
- <span id="page-11-9"></span>Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In ICML, 2023.
- <span id="page-11-11"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv, 2024.
- <span id="page-11-16"></span>Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. Grammar-constrained decoding for structured NLP tasks without finetuning. In EMNLP, 2023.
- <span id="page-11-12"></span>Malik Ghallab, Craig Knoblock, David Wilkins, Anthony Barrett, Dave Christianson, Marc Friedman, Chung Kwok, Keith Golden, Scott Penberthy, David Smith, Ying Sun, and Daniel Weld. Pddl - the planning domain definition language. Technical Report, 1998.
- <span id="page-11-6"></span>Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, et al. Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes. In ICCV, 2023.
- <span id="page-11-10"></span>Qiao Gu, Alihusein Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, Joshua B. Tenenbaum, Antonio Torralba, Florian Shkurti, and Liam Paull. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In ICRA, 2024.
- <span id="page-11-14"></span>Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv, 2023.
- <span id="page-11-15"></span>Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2021.
- <span id="page-11-1"></span>Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In ICML, 2022.
- <span id="page-11-8"></span>Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol

Hausman, and brian ichter. Inner monologue: Embodied reasoning through planning with language models. In CoRL, 2023.

- <span id="page-12-11"></span>Unnat Jain, Luca Weihs, Eric Kolve, Mohammad Rastegari, Svetlana Lazebnik, Ali Farhadi, Alexander G. Schwing, and Aniruddha Kembhavi. Two body problem: Collaborative visual task completion. In CVPR, 2019. first two authors contributed equally.
- <span id="page-12-2"></span>Unnat Jain, Luca Weihs, Eric Kolve, Ali Farhadi, Svetlana Lazebnik, Aniruddha Kembhavi, and Alexander G. Schwing. A cordial sync: Going beyond marginal policies for multi-agent embodied tasks. In ECCV, 2020.
- <span id="page-12-9"></span>Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 2020.
- <span id="page-12-5"></span>Pushkal Katara, Zhou Xian, and Katerina Fragkiadaki. Gen2sim: Scaling up robot learning in simulation with generative models. arXiv, 2023.
- <span id="page-12-15"></span>Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In CVPR, 2024.
- <span id="page-12-0"></span>Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-andlanguage navigation in continuous environments. In ECCV, 2020.
- <span id="page-12-19"></span>Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In EMNLP, 2020.
- <span id="page-12-16"></span>Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In NeurIPS, 2020.
- <span id="page-12-4"></span>Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In CoRL, 2023a.
- <span id="page-12-14"></span>Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. In NeurIPS, 2023b.
- <span id="page-12-12"></span>Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, et al. Pre-trained language models for interactive decision-making. In NeurIPS, 2022.
- <span id="page-12-18"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In NeurIPS, 2024.
- <span id="page-12-1"></span>Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In CVPR, 2024.
- <span id="page-12-3"></span>Zhao Mandi, Shreeya Jain, and Shuran Song. Roco: Dialectic multi-robot collaboration with large language models. In ICRA, 2024.
- <span id="page-12-6"></span>Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In RSS, 2024.
- <span id="page-12-17"></span>Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv, 2024.
- <span id="page-12-13"></span>Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In ACM Symposium on User Interface Software and Technology, 2023.
- <span id="page-12-8"></span>Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In CVPR, 2018.
- <span id="page-12-7"></span>Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Yuan-Hong Liao, Joshua B. Tenenbaum, Sanja Fidler, and Antonio Torralba. Watch-and-help: A challenge for social perception and human-{ai} collaboration. In ICLR, 2021.
- <span id="page-12-10"></span>Xavier Puig, Tianmin Shu, Joshua B Tenenbaum, and Antonio Torralba. Nopa: Neurally-guided online probabilistic assistance for building socially intelligent home assistants. In ICRA, 2023.
- <span id="page-13-2"></span>Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, Vladimír Vondruš, Theophile Gervet, Vincent-Pierre Berges, John M. Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A co-habitat for humans, avatars and robots. In ICLR, 2024.
- <span id="page-13-17"></span>Team PyTorch. Accelerating generative ai with pytorch ii: Gpt, fast. [https://pytorch.org/blog/](https://pytorch.org/blog/accelerating-generative-ai-2/) [accelerating-generative-ai-2/](https://pytorch.org/blog/accelerating-generative-ai-2/), 2023.
- <span id="page-13-10"></span>Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning. In CoRL, 2023.
- <span id="page-13-18"></span>Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In EMNLP, 2019.
- <span id="page-13-15"></span>Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv, 2023.
- <span id="page-13-9"></span>Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In NeurIPS, 2023.
- <span id="page-13-0"></span>Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. In CVPR, 2020.
- <span id="page-13-6"></span>Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In ICLR, 2021.
- <span id="page-13-14"></span>Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In CoRL, 2022.
- <span id="page-13-8"></span>Joseph Suarez, Yilun Du, Phillip Isola, and Igor Mordatch. Neural mmo: A massively multiagent game environment for training and evaluating intelligent agents. arXiv, 2019.
- <span id="page-13-16"></span>Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. In NeurIPS, 2021.
- <span id="page-13-3"></span>Andrew Szot, Unnat Jain, Dhruv Batra, Zsolt Kira, Ruta Desai, and Akshara Rai. Adaptive coordination in social embodied rearrangement. In ICML, 2023.
- <span id="page-13-12"></span>Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, and Alexander Toshev. Large language models as generalizable policies for embodied tasks. In ICLR, 2024.
- <span id="page-13-5"></span>Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. In ICML, 2024.
- <span id="page-13-11"></span>Abdelrhman Werby, Chenguang Huang, Martin Büchner, Abhinav Valada, and Wolfram Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In RSS, 2024.
- <span id="page-13-7"></span>Zhou Xian, Theophile Gervet, Zhenjia Xu, Yi-Ling Qiao, and Tsun-Hsuan Wang. Towards a foundation model for generalist robots: Diverse skill learning at scale via automated task and scene generation. arXiv, 2023.
- <span id="page-13-13"></span>Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and Zhiting Hu. Language models meet world models: Embodied experiences enhance language models. In NeurIPS, 2023.
- <span id="page-13-4"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2023.
- <span id="page-13-1"></span>Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, et al. Homerobot: Open-vocabulary mobile manipulation. In CoRL, 2023.
- <span id="page-14-6"></span>Naoki Yokoyama, Alex Clegg, Joanne Truong, Eric Undersander, Tsung-Yen Yang, Sergio Arnaud, Sehoon Ha, Dhruv Batra, and Akshara Rai. Asc: Adaptive skill coordination for robotic mobile manipulation. IEEE Robotics and Automation Letters, 2023.
- <span id="page-14-2"></span>Andy Zeng, Maria Attarian, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, et al. Socratic models: Composing zero-shot multimodal reasoning with language. In ICLR, 2022.
- <span id="page-14-1"></span>Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. ICLR, 2024a.
- <span id="page-14-3"></span>Hongxin Zhang, Zeyuan Wang, Qiushi Lyu, Zheyuan Zhang, Sunli Chen, Tianmin Shu, Yilun Du, and Chuang Gan. Combo: Compositional world models for embodied multi-agent cooperation. arXiv, 2024b.
- <span id="page-14-4"></span>Zirui Zhao, Wee Sun Lee, and David Hsu. Large language models as commonsense knowledge for large-scale task planning. In NeurIPS, 2024.
- <span id="page-14-5"></span>Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. In ICLR, 2024.
- <span id="page-14-0"></span>Hao Zhu, Raghav Kapoor, So Yeon Min, Winson Han, Jiatai Li, Kaiwen Geng, Graham Neubig, Yonatan Bisk, Aniruddha Kembhavi, and Luca Weihs. Excalibur: Encouraging and evaluating embodied exploration. In CVPR, 2023.

# A Appendix

We present PARTNR, a benchmark for reasoning and planning in multi-agent embodied tasks, featuring 100,000 natural language, everyday tasks. We show at-scale generation of these tasks using LLMs with simulation in the loop for grounding and human in the loop for filtering. We also evaluate several LLM-based planning models on these tasks and highlight avenues for future work. This appendix provides additional details on these contributions and is organized as follows:

- [A.1](#page-15-1) Code and PARTNR benchmark open-sourcing
- [A.2](#page-15-2) HSSD Scene Annotations
- [A.3](#page-15-0) Details and additional analysis on the PARTNR dataset
- [A.4](#page-17-0) Simulation features and prompts for the PARTNR task generation
- [A.5](#page-19-0) The PARTNR evaluation system
- [A.6](#page-23-1) Human annotation and quality assessment for the PARTNR tasks and evaluation functions
- [A.7](#page-27-0) World graph for perception in LLM agents
- [A.8](#page-30-0) Learned low-level robot skills
- [A.9](#page-31-0) Implementation details for ReAct agents
- [A.10](#page-33-0) Details on finetuning LLM agents for the PARTNR tasks
- [A.11](#page-34-1) Additional results
- [A.12](#page-34-0) Analysis of collaborative behavior and efficiency of LLM agents
- [A.13](#page-36-0) Human-in-the-loop (HITL) system and evaluation for the PARTNR tasks and LLM agents
- [A.14](#page-40-0) Prompts for task and evaluation generation
- <span id="page-15-1"></span>[A.16](#page-56-0) Prompts for planner baselines

# A.1 Open-sourcing PARTNR Dataset and Codebase

Accompanying this paper, we have released the code and data necessary to reproduce our experiments. Released code includes our PARTNR benchmark tasks, metrics, baseline oracle skills, large planning model framework, and dataset generation utilities. Released data includes extensions of the Habitat Synthetic Scenes Dataset (HSSD) [(Khanna et al.,](#page-12-15) [2024)](#page-12-15), generated benchmark task episodes, and model weights for our trained neural network skills and fine-tuned large planning model.

The publicly released codebase accompanying PARTNR depends on the most recent version of the AI Habitat platform (habitat-lab and habitat-sim (v0.3.2)) [(Puig et al.,](#page-13-2) [2024)](#page-13-2) which it extends to define collaboration tasks and skills.

# <span id="page-15-2"></span>A.2 HSSD Scene Annotations

<span id="page-15-0"></span>In order to model the space of rich indoor collaboration tasks we propose with PARTNR, we extended HSSD with additional asset authoring and annotation. To enable more realistic indoor object manipulation, we added articulated 3D furniture models such as drawers, cabinets, and appliances. These models were converted from rigid source assets in HSSD and swapped into the scenes. We prepared 60 scenes divided into train, val, and test splits to support our experiments. Each scene is manually adjusted by a human to ensure simulation robustness and minimize potential issues. Furniture is annotated with a set of Receptacles (surfaces which support small object placement such as shelves and drawers) and can be opened and closed by the agents. Receptacles are further filtered contextually in each scene to ensure that the active set is accessible to the agents. Additional annotations include point or marker sets for each furniture, region annotations, and semantic classification of objects. The marker sets indicate either a spread of surface points (for distance/occlusion checking) or the location of key points of interest such as faucets (for cleaning/filling) and handles (for opening/closing) necessary for low-level skill training and oracle skill execution. Region annotations included per-scene region volumes (e.g., kitchen, living-room, bedroom, etc.) for checking and specifying the location of objects and furniture. Semantic annotations indicate the object category or class (e.g. table, chair, cup, toy) to support open language prompt grounding and semantically guided task generation.

<span id="page-16-0"></span>

| Task Instruction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Task Instruction                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Let's tidy up the family room. The toys go in the toy box.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | "We're ready for dinner. Can you bring two cups to the                                                                                                                                                                                                                                                                                                                                                                                      |
| After that, set the plants on the shelf next to each other."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | kitchen? Let's fill them and take them to the dining table."                                                                                                                                                                                                                                                                                                                                                                                |
| Evaluation Function<br>Propositions:<br>0 is_inside(["toy_fire_truck_0"], ["toy_box_0"])<br>1 is_inside(["toy_food_0"], ["toy_box_0"])<br>2 is_on_top(["plant_0"], ["shelf_0", "shelf_1"])<br>3 is_on_top(["plant_1"], ["shelf_0", "shelf_1"])<br>4 is_next_to(["plant_0"], ["plant_1"])<br>Verify "is_next_to" when<br>Dependencies:<br>the plants are placed<br>0 WhileSatisfied([4], depends_on=[2,3])<br>The toys must be rearranged first<br>Constraints:<br>0 TemporalConstraint(<br>[(0,2), (1,2), (0,3),<br>All placements are terminal<br>(1,3), (0,4), (1,4)]<br>)<br>1 TerminalSatisfactionConstraint([0,1,2,3,4]) | Evaluation Function<br>Of the n cups in the scene…<br>Bring two<br>Propositions:<br>0 is_in_room(["cup_0", "cup_1",…, "cup_n"], ["kitchen"], n=2)<br>1 is_filled(["cup_0", "cup_1",…, "cup_n"], n=2)<br>2 is_on_top(["cup_0", "cup_1",…, "cup_n"], ["table_2"], n=2)<br>Use the same cups throughout<br>Constraints:<br>0 SameArgConstraint([0,1,2], arg="object")<br>1 TerminalSatisfactionConstraint([1,2])<br>Proposition 0 is temporary |

Figure 6 Additional task and evaluation examples. Expands on Figure [3](#page-4-0) to show a task and evaluation containing subset counts and dependent rearranges (right).

# A.3 Dataset Details and Additional Analysis

We expand on the details and analysis of the PARTNR dataset of Section [3.3,](#page-5-0) including analysis of linguistic phenomena, secondary task characteristics, and the distribution of sampled entities in generated tasks. See Figure [6](#page-16-0) for additional task and evaluation examples.

Linguistic Phenomena. In Table [4,](#page-17-1) we present an analysis of linguistic phenomena manually annotated over 50 random episodes (similar to [Ku et al.](#page-12-19) [(2020)](#page-12-19) and [Chen et al.](#page-11-4) [(2019)](#page-11-4)). We analyze the following phenomena:

- Class Reference: refers to a semantic class of objects, furniture, or rooms in the scene. These references typically, but not necessarily, follow from the classes defined in OVMM.
- Instance Reference: refers to a unique object, furniture, or room; this disambiguation between entities of a class is typically achieved by visual description.
- Co-Reference: an expression that refers to an entity defined elsewhere in the instruction.
- Passive Voice: the instruction is phrased such that the object is receiving the rearrangement or state change; the request is typically asked instead of commanded.
- Active Voice: the instruction is phrased such that the rearrangement or state change is to be completed with an object; the request is typically commanded instead of asked.
- High-Level Goal Spec: this sets the operating context for the task before the particulars of rearrangement or state change are specified.
- Agentic Reference: a reference to one of the agents performing the task. Typically used to incite a suggested task division between the human and robot.

We observe that PARTNR tasks have a high rate of entity class references, such as the table, (6.38/episode), and common occurrences of instance references and co-references. This signals a need for capable natural language understanding, scene grounding, and co-reference resolution. Tasks and sub-tasks are predominantly issued using active voice (92%), but 14% of tasks include at least one occurrence of passive voice. Half of tasks involve a high-level goal specification, which commonly serves to reduce the search space. For example, a task starting with Let's clean up all the toys in the playroom constrains object search to that room and softly constrains the placement of those objects to locations in which toys would commonly be stored. Finally, agentic references are present in 14% of episodes.

Secondary Task Characteristics. In Table [5,](#page-17-2) We present an analysis of secondary task characteristics present in PARTNR as derived automatically from evaluation functions of all episodes in our dataset. We find rare but present occurrences of subset counts, where agents must reason about manipulating a subset of a set of

<span id="page-17-1"></span>

| Linguistic Phenomenon | p   | µ    | Example              |
|-----------------------|-----|------|----------------------|
| Class Reference       | 100 | 6.38 | The table            |
| Instance Reference    | 12  | 0.14 | The coffee table     |
| Co-Reference          | 50  | 0.64 | That, Those, it,     |
| Passive Voice         | 14  | 0.18 | Can you bring me?    |
| Active Voice          | 92  | 2.40 | Set it on the stool. |
| High-Level Goal Spec  | 50  | 0.50 | Let's tidy up. Move  |
| Agentic Reference     | 14  | 0.16 | Do While I           |

<span id="page-17-2"></span>Table 4 Analysis of linguistic phenomena in the PARTNR dataset. p is the % of instructions that contain the phenomenon while µ is the average number of times the phenomenon occurs within each instruction. A random sample of 50 Test episodes were included in manual annotation.

| Secondary Characteristics | p  | µ    | Example                       |
|---------------------------|----|------|-------------------------------|
| Subset Count              | 1  | 0.01 | Bring two cups                |
| Resolvable Ambiguity      | 68 | 1.68 | Move the pants to any chair.  |
| Dependent Rearrange       | 7  | 0.09 | Place them on the same table. |
| Multi-Step Rearrange      | 6  | 0.11 | Cup to sink. Cup to cabinet.  |

Table 5 Analysis of secondary task characteristics in the PARTNR dataset. p is the % of instructions that contain the phenomenon while µ is the average number of times the phenomenon occurs within each instruction. All episodes in Test were included using automatic annotation.

objects (e.g. bring two cups... when more than two cups exist in the scene). We find that every episode contains at least one occurrence of resolvable ambiguity, where an instruction makes an object/furniture/room reference that may be resolved by more than one entity instance. Dependent rearrangements build on this task characteristic; when multiple objects/furniture/rooms can satisfy a sub-task, a task may require that same entity to solve another sub-task. An example of this is the same table being used to solve the placement of both a spoon and a bowl ("bring the spoon and bowl to the same table in the living room."). This occurs in 31% of episodes and the resulting dynamic dependency is a challenging aspect of collaboration. Finally, 6% of tasks include multi-step rearrangement of the same entity. For example, moving a cup to the sink and then to the cabinet after it is washed. Tasks with multi-step rearrangement have two such rearrangements on average.

Distribution of Entities in PARTNR Tasks. In Figure [7,](#page-18-0) we examine the distribution of task-relevant objects, furniture, and rooms. We define task-relevant objects to be objects requiring rearrangement or a state change, task-relevant furniture to be target furniture for rearrangement, and task-relevant rooms to be target rooms for rearrangement. Object categories are shown in Figure [7a,](#page-18-0) furniture categories are shown in Figure [7b,](#page-18-0) and room categories are shown in Figure [7c.](#page-18-0) The PARTNR dataset contains a long tail of semantic object categories, and within those categories, a wide variety of objects. Thus, for agents to perform well in PARTNR tasks, they must display visual reasoning and collaboration behaviors that generalize across the semantic particulars of a task. The skew of distributions in Figure [7b](#page-18-0) and Figure [7c](#page-18-0) can be understood by the relative occurrences of rooms and objects in HSSD scenes, e.g. there are more tables than couches on average.

# <span id="page-17-0"></span>A.4 Simulation-in-the-loop Large-Scale Task Generation

In this section, we describe in detail the simulation-in-the-loop task generation pipeline. We follow a 4-step generation pipeline:

#### 1. Simulation-in-the-loop small-scale free-form LLM generation:

In PARTNR, we initiate the process by setting up a simulation environment using the Habitat 3.0 simulator, populated with the HSSD dataset which includes 60 unique houses and 5,819 OVMM objects. This simulated environment is parsed to identify a list of rooms and available furniture. This information, along with a list of all available objects, is then passed to a Language Model (LLM). The LLM utilizes

<span id="page-18-0"></span>Figure 7 The distribution of task-relevant object categories, furniture categories, and room categories in the PARTNR dataset. The object distribution is truncated to the 50 most common entities, down from 155 total object categories. The count of entities is derived by the total number of occurrences of that entity in the Test split evaluation data.

this data to generate free-form, viable tasks within the scene, accompanied by an initial scene state description. For instance, if the task is "Clear dishes from the living room," the LLM would generate an initial scene depicting multiple dishes in the living room. To add complexity, additional objects are introduced to create clutter.

#### 2. Simulation-in-the-loop filtering and annotation.

The tasks and initial states generated by the LLM are instantiated within the simulator. At this stage, tasks that are infeasible due to the layout of the house or the capabilities of the simulator are filtered out. For example, a task like "Clear dishes from the living room" would be discarded if the simulated house lacks a living room. Similarly, tasks requiring unsupported actions, such as "folding," are also filtered out. This filtering process is crucial as it significantly reduces the number of unrealistic or impossible tasks, although it also tends to limit the diversity of the generated instructions. The full list of allowed actions in the simulator are: [open, close, power on, power off, move, clean, fill, pour, pick, place] and related actions. Next, we use manual annotation to diversity of the resulting tasks. To counter the limited diversity resulting from automated filtering, we manually annotate the tasks to ensure a balanced distribution of various types of tasks, such as constraint-free, spatial, temporal, and heterogeneous tasks. This manual intervention involves modifying the instructions to incorporate different objects and settings, resulting in a curated set of 1,000 human-annotated and simulation-verified tasks. This step is essential for maintaining quality and diversity but is impractical for scaling up due to its labor-intensive nature.

### 3. Large-scale generation.

To scale the generation of diverse tasks without extensive manual effort, we leverage the 1,000 humanannotated instructions as examples in the LLM prompts. By providing the LLM with both a house description and an example task, we instruct it to adapt the task to fit the new setting, such as changing "Clear all dishes from the living room" to "Clear all toys from the bedroom." This approach helps preserve the diversity of the tasks while enhancing the likelihood of successful instantiation in the simulator.

#### 4. Human-in-the-loop filtering.

The scaled tasks undergo a final human-in-the-loop filtering where physically infeasible instructions are eliminated, ensuring the practicality and realism of the tasks. This ensures that tasks such as "Move 4 cups to the dining table", where there are only 2 cups in the scene are removed. Or "First move the cup from the kitchen to the living room, then place a jug in the kitchen, next to the cup", which consists of a physically infeasible instruction.

### A.4.1 Habitat 3.0 and HSSD Extension

We generate PARTNR using modified HSSD scenes [(Khanna et al.,](#page-12-15) [2024)](#page-12-15) and the Habitat 3.0 [(Puig et al.,](#page-13-2) [2024)](#page-13-2) simulator due to its humanoid simulation capabilities and availability of features which support modeling of collaborative tasks as discussed in Appendix [A.1.](#page-15-1)

Our extensions to the Habitat platform include a set of features targeting: object state manipulation (e.g., clean/dirty, powered on/off and filled/un-filled), evaluation of object relative relationships (e.g., next-to, above, within, on-top, on-floor, in-region, etc), and procedural clutter generation utilities enabling generation of valid initial scene contents pre-conditioned on the output of LLM-generated requirements from open-language prompts (see Section [3)](#page-3-1). For example, in order to evaluate the task of rearranging a tea set from furniture A to furniture B, we must first generate a scene with both types of furniture in accessible locations and a tea set already sitting on or inside of furniture A.

# <span id="page-19-0"></span>A.5 The PARTNR Evaluation System

In this section, we formalize the components of the evaluation system, define the resulting metrics, and present the details and prompts used for LLM-based evaluation generation.

### A.5.1 Evaluation Predicates

We use logical predicates to query the state of objects, furniture, and rooms at the current timestep in the simulator. The evaluation system operates on the resulting binary state values. The details of all logical predicates are in Table [6.](#page-20-0)

### A.5.2 Propositions

The primary component of a task evaluation function is a list of propositions. We define a proposition as an evaluation predicate instantiated with argument values. Propositions additionally enable the evaluation of instructions with ambiguous references ("on a table" — which table?) and subset counts ("two spoons" any two of the n total spoons). Ambiguity is enabled by extending the predicate arguments to lists. Subset counts are enabled by optional arguments number, which defines the subset size, and arg_match, which is a boolean indicating whether all entities in the subset must be satisfied with the same second argument. Suppose we want to evaluate the task "Bring a spoon to the table.". If we have a single spoon and a single table, the proposition is straightforward:

```
is_on_top([spoon_1], [table_1]).
```
If we have multiple spoons and need just one (an ambiguous instruction), the following proposition is used:

```
is_on_top([spoon_1, spoon_2, spoon_3], [table_1]).
```
In the above case, a list is treated as a OR of entities. The same holds for multiple tables:

is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2]),

in which any spoon may be placed on any table. If the instruction specifies bringing two spoons to the table, the number=2 argument is added:

is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2).

<span id="page-20-0"></span>

| Predicate Name         | Category  | Description                                                                                                                                                                     |
|------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| is_on_top(o1, o2)      | Rearrange | o1<br>is considered on top of o2<br>if a downward raycast<br>from any of the o1<br>bounding box corners intersects<br>with o2.                                                  |
| is_inside(o1, o2)      | Rearrange | o1<br>is considered inside o2<br>if a threshold number of<br>opposing raycasts hit the same object. Ray casts<br>are 1.00m and the threshold number is 2.                       |
| is_in_room(o1, r1)     | Rearrange | o1<br>is considered to be in room r1<br>if at least 25%<br>of keypoints (bounding box corners and center)<br>are contained in the 3D region.                                    |
| is_on_floor(o1)        | Rearrange | o1<br>must be within 0.04m vertically of the naviga<br>tion mesh.                                                                                                               |
| is_next_to(o1, o2)     | Spatial   | The bounding box of o1<br>must overlap vertically<br>with the bounding box of o2<br>and the horizontal<br>L2 distance between bounding boxes is less than<br>or equal to 0.50m. |
| is_clustered(o1, , on) | Spatial   | ) for some j ̸=<br>Each oi<br>satisfies is_next_to(oi<br>, oj<br>i.                                                                                                             |
| is_clean(o1)           | State     | o1<br>has affordance cleanable and the state machine<br>indicates that o1<br>is clean.                                                                                          |
| is_dirty(o1)           | State     | o1<br>has affordance cleanable and the state machine<br>indicates that o1<br>is not clean.                                                                                      |
| is_filled(o1)          | State     | has affordance fillable and the state machine<br>o1<br>indicates that o1<br>is filled.                                                                                          |
| is_empty(o1)           | State     | o1<br>has affordance fillable and the state machine<br>indicates that o1<br>is not filled.                                                                                      |
| is_powered_on(o1)      | State     | o1<br>has affordance powerable and the state ma<br>chine indicates that o1<br>is powered on.                                                                                    |
| is_powered_off(o1)     | State     | o1<br>has affordance powerable and the state ma<br>chine indicates that o1<br>is powered off.                                                                                   |

Table 6 Logical predicates to evaluate the relations of objects, furniture, and rooms in the PARTNR. Predicates exist for measuring rearrangement of objects (category=Rearrange), spatial placements relative to objects or furniture (category=Spatial), and states of objects or furniture (Category=State). For all predicates, oi ∈ {object, furniture} and ri is a room.

Finally, the instruction may specify that two spoons should be brought to the same table. arg_match=True enables this:

```
is_on_top([spoon_1, spoon_2, spoon_3], [table_1, table_2], number=2, arg_match=True).
```
Propositions are represented as JSON within dataset files and as Python function calls (as shown above) during human annotation for interpretability.

# A.5.3 Dependencies

Evaluation functions operate over a sequence of simulation states. By default, all propositions in the list of evaluation propositions will be evaluated at every time step. However, many tasks in the PARTNR benchmark involve a required temporal order during execution. Take for example the instruction "Move the cup from the table to the sink. Then, return the cup to the table." This consists of a multi-step rearrangement where the proposition checking that the cup is on the table is dependent on a different proposition first being satisfied. We define a proposition dependency as the following:

PropositionDependency(proposition_indices, depends_on, relation_type),

where the argument proposition_indices is a list of integers indicating the dependent propositions, the argument depends_on is a list of integers indicating the head propositions, and the argument relation_type indicates the condition that the depends_on propositions must take in order for proposition_indices to be evaluated for satisfaction. The following relation types are supported:

- after_satisfied: evaluate propositions in proposition_indices after the propositions in depends_on have been satisfied.
- after_unsatisfied: evaluate propositions in proposition_indices after propositions in depends_on have been satisfied at some point in the past but are no longer satisfied.
- while_satisfied: evaluate propositions in proposition_indices when propositions in depends_on are currently satisfied.

As a concrete example, suppose the instruction is "Move the ball and bat to the kitchen and set them next to each other. Then, move them to the closet." The propositions for this task would be

> is_in_room([ball], [kitchen]) is_in_room([bat], [kitchen]) is_next_to([ball], [bat]) is_in_room([ball], [closet]) is_in_room([bat], [closet]).

and the dependencies would be

0 PropositionDependency([2], [0, 1], "while_satisfied") 1 PropositionDependency([3, 4], [0, 1], "after_satisfied").

The first ensures that the is_next_to predicate is only queried during kitchen placements and the second ensures that placing the ball and bat in the closet is not checked at the start of the task execution, which would be inadvertently satisfied upon scene initialization. Each evaluation function has a (possibly empty) list of such proposition dependencies.

### A.5.4 Constraints

PARTNR tasks often require constraining how tasks are completed. Examples of this include completing evaluation propositions in temporal order ("Do [x] before doing [y]" and enforcing links between ambiguities ("Fill one of the cups, then put that cup on the table"). We define a set of constraint types that enables evaluating such task complexities:

- TemporalConstraint: A directed acyclic graph (DAG) over the indices of propositions that defines the temporal requirement of when propositions should be satisfied. If propositions are satisfied out of order, then the task is marked unsuccessful. The temporal constraint is defined as
### TemporalConstraint(graph_edges),

where graph_edges is a set of pairwise temporal constraints. For example, an edge "0 → 1" indicates that the proposition at index 0 must be satisfied at an earlier time step than the proposition at index 1.

- SameArgConstraint: Requires the argument used to satisfy a proposition to be the same within a pre-determined set of propositions. The same argument constraint is defined as
# SameArgConstraint(proposition_indices, arg_names),

where proposition_indices is a list of proposition indices to link and arg_names is a list specifying which component of each proposition to link.

- DifferentArgConstraint: Requires the argument used to satisfy a proposition to be unique within a pre-determined set of propositions. The different argument constraint is defined as
# DifferentArgConstraint(proposition_indices, arg_names),

where proposition_indices is a list of proposition indices to link and arg_names is a list specifying which component of each proposition to link.

- TerminalSatisfactionConstraint: Some propositions, once satisfied, should remain satisfied at the end of the episode. Others are expected to become unsatisfied, such as in multi-step rearrangements or object state changes of a single object. The terminal satisfaction constraint is defined as
# TerminalSatisfactionConstraint(proposition_indices)

where proposition_indices is a list of proposition indices that should be satisfied at time t = T for an episode rollout of duration T.

### A.5.5 Evaluation Metrics

A task evaluation function is constituted of propositions, dependencies, and constraints as defined above. From this data, the task evaluation function serves to determine what percentage of the task is complete for a given human-robot collaboration rollout. From a sequence of simulation states, we evaluate the truth values of each proposition with respect to both the dependencies (when the propositions must be evaluated) and the constraints (how the propositions must be satisfied). The Percent Completion (P C) is defined as the ratio of satisfied propositions to the total number of propositions. Success is defined as S := (P C = 100). Both Percent Completion and Success metrics solely evaluate task completion and thus are agent-agnostic. Given that PARTNR is designed to evaluate multi-agent collaboration, we note that these metrics can be combined with duration-based metrics (e.g. simulation steps or time) to measure efficiency; multi-agent aspects like task division and partnered exploration serve to optimize task efficiency. The benefit of an agent-agnostic metric is flexibility to evaluate any number of agents performing the task with respect to the high-level goal.

### A.5.6 Evaluation Function Generation

This section provides detail on evaluation function generation beyond the overview provided in Section [3.](#page-3-1) In particular, Figure [8](#page-23-2) shows the first two steps of Figure [2](#page-3-0) (sim-in-the-loop generation and filtering + annotation) in greater detail as they pertain to evaluation generation. Notably, we take a three step process to generating evaluation functions; first, an LLM generates a list of evaluation propositions which are parsed into a usable format. Then, an LLM infers the temporal constraint over these propositions by predicting topological generations of a temporal graph. For example, the prediction [[0, 1], [2, 3]] implies that propositions at indices 0, 1 can be completed in either order, and both must be completed before propositions at indices 2, 3. This prediction is stored and evaluated more generally as edges of a directed acyclic graph (DAG). We found that the assumption of topological generations sufficiently expresses the tasks in PARTNR while being simpler for an

<span id="page-23-2"></span><span id="page-23-3"></span>Figure 8 Evaluation function generation pipeline overview. A three-step LLM-generation process produces propositions and constraints during Generate Evaluations. The evaluation function then is saved to a file for optional human annotation during Manual Annotation. Finally, the evaluation function is packed and verified in simulation during Pack & Verify in Sim to ensure that all entities and affordances exist.

|                 |      | Correctness (%) |          |
|-----------------|------|-----------------|----------|
|                 | Task | Evaluation      | Combined |
| Constraint-Free | 89   | 97              | 86       |
| Spatial         | 87   | 85              | 74       |
| Temporal        | 91   | 93              | 85       |
| Heterogeneous   | 93   | 91              | 85       |
| Average         | 90   | 92              | 83       |

Table 7 Manually-annotated generation accuracy of 100k-scale PARTNR tasks and evaluation functions. Altogether, we find that 83% of episodes are generated without any task or evaluation function errors. Analysis performed on 100 sampled episodes of each task type via PrediViz.

LLM to generate than a DAG. The third step is predicting argument constraints, in which the LLM is provided the instruction and propositions list and must predict a list of constraints, either SameArgConstraint or DifferentArgConstraint. All LLM queries are performed against CodeLlama70b-Instruct [(Roziere et al.,](#page-13-15) [2023)](#page-13-15). All prompts used for evaluation function generation are included in Appendix [A.15.](#page-48-0)

# <span id="page-23-1"></span>A.6 Human annotation and accuracy assessment for PARTNR data

### <span id="page-23-0"></span>A.6.1 Generation Accuracy for Tasks and Evaluations

It is important for tasks in PARTNR to be solvable by collaboration agents and for the associated evaluation functions to accurately reflect the task being performed. In this section, we analyze the accuracy of 100k-scale PARTNR generation with respect to both of these criteria using the PrediViz tool (Appendix [A.6.2)](#page-25-1). In Table [7,](#page-23-3) we demonstrate that the accuracy of task generation ranges from 87-93% depending on task type and averages 90%. The accuracy of evaluation generation ranges from 85-93% depending on task type and averages 92%. Combining these numbers yields an overall joint accuracy of 83% for our 100k-scale dataset. In Table [8,](#page-24-0) we annotate the failure modes that lead to unsolvable tasks and incorrect evaluation functions. Common task-related failure modes include:

- Hallucination. The produced instruction references objects, furniture, or rooms that do not exist in the scene the instruction was generated for. Example: "Move the clothes to the washing machine." produced for an environment that does not contain a washing machine.
<span id="page-24-0"></span>

|                 | Task Failures          |   | Evaluation Failures                     |   |
|-----------------|------------------------|---|-----------------------------------------|---|
|                 | Mode                   | % | Mode                                    | % |
|                 | Hallucination          | 7 | Incorrect Ambiguity                     | 2 |
| Constraint-Free | Already Satisfied      | 2 | Incorrect Furniture                     | 1 |
|                 | Contradiction          | 2 | -                                       | - |
|                 | Hallucination          | 6 | Incorrect Temporal Grouping             | 7 |
| Spatial         | Unresolvable Ambiguity | 5 | Incorrect Predicate (Other)             | 5 |
|                 | Already Satisfied      | 2 | Incorrect Predicate (Room vs Furniture) | 3 |
|                 | Hallucination          | 4 | Incorrect Temporal Grouping             | 3 |
| Temporal        | Contradiction          | 3 | Incorrect Predicate (Other)             | 3 |
|                 | Already Satisfied      | 2 | Incorrect Predicate (Room vs Furniture) | 1 |
|                 | Hallucination          | 3 | Incorrect Predicate (Room vs Furniture) | 5 |
| Heterogeneous   | Unresolvable Ambiguity | 2 | Incorrect Object/Furniture/Room         | 2 |
|                 | Contradiction          | 1 | Incorrect Predicate (Other)             | 2 |

Table 8 Top three failure modes of 100k-scale task and evaluation generation reported for each task type. Failures of task generation are led by the hallucination of non-existent entities, while failures of evaluation generation are led by incorrect temporal predictions and incorrect predicate functions. Analysis performed on 100 sampled episodes of each task type via PrediViz.

- Unresolvable Ambiguity. The produced instruction contains ambiguous directives that cannot be reasonably resolved without further communication or a detailed understanding of the task-issuer's preferences. Example: "Set the table for dinner"; how many place settings are necessary? Should we set the formal dining table or nook table? What cutlery is needed?
- Contradiction. The produced instruction involves two or more sub-tasks that cannot simultaneously be satisfied. Example: "Set the scissors on the coffee table. Set the bowl on the counter next to them."
- Already Satisfied. The produced instruction dictates sub-tasks that are all already satisfied at the start of the episode. Example: "Move the laptop to the living room and turn it on", when the laptop is already powered on and in the living room.

According to Table [8,](#page-24-0) hallucinations are the most common failure mode for task generation. While simulationin-the-loop filtering avoids this issue for evaluation generation, such filtering is inconsistent for tasks; language has a looser grounding to scene entities than statements of propositional logic. For example, a home might not have a formal dining room, but a table in the living room may serve the purpose of a dining table. Moving on to evaluation generation, common failures are as follows:

- Incorrect Temporal Grouping. The instruction implies a temporal order among sub-tasks (either explicitly via sequencing words, or implicitly via multi-step manipulations) and the predicted temporal constraint fails to reflect this order over the propositions. Example: allowing propositions to be satisfied in any order for the task "First, return the plates to the kitchen. Then, tidy up the living room."
- Incorrect Predicate (Other). The evaluation function uses the wrong predicate function to evaluate the task. Example: using is_powered_on instead of is_filled when the instruction asks to fill the kettle.
- Incorrect Predicate (Room vs Furniture). A task specifies that an object should be rearranged to another room, but the propositions overly-constrain the rearrangement to a target furniture for placement. This failure is separate from the one above because it the most common, and it indicates the tendency for the LLM to produce single solution instances rather than reflect the full space of ambiguity. Example: producing a proposition like is_on_top([electronic_cable], [bed]) for the task "Move the electronic cable to the bedroom."
- Incorrect Object/Furniture/Room. incorrect entities are selected for satisfying proposition. Example: the instruction calls for rearranging the cushion to a living room table, but table_4 in the proposition is_on_top([cushion_0], [table_4]) exist in bedrooms.

- Incorrect Ambiguity. For an instruction that can be satisfied n different ways, the evaluation affords m options for solution, where m ̸= n. Example: "Move a toy to the kid's room", where two or more toys exist in the toy bin but the evaluation function does not list out all possible toys for rearrangement.
According to Table [8,](#page-24-0) the primary failures modes of evaluation generation are incorrect temporal grouping and incorrect predicates. Regarding temporal grouping, we observe that many sub-task require multiple propositions to evaluate. Grouping these propositions consistently within the temporal constraint is a source of error. Take for example the instruction "Set the shirt and pants next to each other on the counter. Then, move them to the dresser." In this case, five propositions will exist; four for placements and one for the spatial relation. The temporal prediction may erroneously link the spatial relation with the dresser placements rather than the counter placements.

# <span id="page-25-1"></span>A.6.2 Visualization of PARTNR Tasks and Evaluation Functions

LLM-generated PARTNR episodes may contain errors so we evaluate their correctness with human annotators. This is a verification problem in which the generated instruction, evaluation function, and contents of the scene must be compared against each other. See Figure [6](#page-16-0) for an example of this in code. To make this process faster, easier, and more accurate, we designed a visualization and annotation system (PrediViz) that illustrates the state of the world and the evaluation function relative to a task instruction (Figure [9)](#page-26-0). We chose a 2D illustration style to capture the relevant structure in the data:

- Rooms. Each room is drawn as a box with a name provided underneath. Names repeat if there are multiple instances of that room category in the scene. We model the rooms as a fully-connected graph of accessibility. We treat rooms as spatially independent and reorder or wrap them as needed.
- Objects. We visualize objects as boxes with category names underneath. Each proposition of the evaluation function is assigned a unique color to separate them from other propositions. For consistency, objects are colored by the color of the first proposition they appear in.
- Receptacles. Receptacles are furniture. We designed a bespoke set of icons for 25 categories of furniture that can be easily subitized – you can tell a chair is a chair and a table is a table just by glancing at it.
- States. Both objects and receptacles have time-varying state (empty/filled, dirty/clean, powered on/off). These states are displayed using motifs for objects and textual labels for receptacles, when necessary.
- Affordances. Different furniture afford different object placements. For example, a cup can be placed on top of the fridge or inside it, but only on top of a table. These relationships are annotated by hand at the category-level. Objects are visualized at their initial receptacle/room placements. An associated dark box indicates the target placement of a particular object.
- Placements. To represent rearrangement of an object, we draw an arrow from the initial position to the target position. A simple placement is represented by a single arrow. Multiple allowable placement targets (e.g. multiple tables in the room) are represented by multiple arrows. We use solid arrows for AND placements (e.g. "place the doll and the toy truck on the couch") and dotted arrows for OR (e.g. "place the doll or the toy truck on the couch"). We also support for choosing k out of n objects for placement (e.g. "place two out of three dolls on the couch") using a numerical label on the dotted lines.
- Temporal Constraints. To visualize temporal constraints, we split the instruction into multiple frames. For example, if the instruction requires us to "place the doll on the couch and the toy truck in the chair, then put the stuffed toy inside a chest of drawers," we create one frame to represent the first half of the statement (i.e. "place the doll on the couch and the toy truck in the chair") and another for the second half (i.e. "put the stuffed toy inside a chest of drawers").
- Special Relations. We also illustrate special relations like next_to, same_as, and different_from in the style of an informational legend shown on the side.

<span id="page-25-0"></span>Resulting visualizations are wrapped in a web-based annotation tool that affords binary verification and failure model labeling. See Figure [8](#page-24-0) for results derived from this tool. We ran a small-scale experimental study (n=22) comparing human annotation using PrediViz to a text-based representation. Using PrediViz, verification was 2.6 times faster, 8% more accurate, and perceived as 24% easier.

<span id="page-26-0"></span>(a) Example task #1

(b) Example task #2: first temporal frame

(c) Example task #2: second temporal frame

Figure 9 PARTNR tasks visualized in PrediViz. The design distills the task and scene to only the components necessary for verification. In example task #2, the split frames signify that the agents must first rearrange the file sorter and folder (Figure [9b)](#page-26-0), then rearrange the backpack (Figure [9c)](#page-26-0).

#### A.6.3 Human-assisted Correction Annotation

Below is an example task and evaluation function saved in plain text for human annotation. The instruction, propositions, and constraints can be modified as necessary to ensure the task is feasible and that the evaluation function reflects it. Annotators also have access to a file containing the objects, furniture, rooms, and relations thereof to reference during this process.

```
Dataset Correction Annotation Trial
# type : i g n o r e
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# INSTRUCTION
# modi fy a s n e c e s s a r y , but keep i n mind the s c e n e i s f i x e d .
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
i n s t r u c t i o n = """
Help me o r g a ni z e the entryway . Fi r s t , pl a c e the phone , watch , and ke ych ain on the t a b l e next t o
each o t h e r .
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PROPOSITIONS
# is_on_top ( o b j e c t s , r e c e p t a c l e s , number=1, i s_ s ame_ recep t acle=F al s e )
# i s _ i n s i d e ( o b j e c t s , r e c e p t a c l e s , number=1, i s_ s ame_ recep t acle=F al s e )
# is_in_room ( o b j e c t s , rooms , number=1, is_same_room=F al s e )
# is_next_to ( e n ti ti e s _ a , e n ti ti e s_ b , number=1, is_same_b=F al se , l 2_ t h r e s h ol d =0.5)
# i s_ o n_ fl o o r ( o b j e c t s , number=1)
# Args :
# o b j e c t s / r e c e p t a c l e s / e n t i t i e s _ * : OR o f a l i s t
# number : n o b j e c t s / e n t i t i e s _ a must s a t i s f y
# is_same : the same e n t i t y must s a t i s f y a l l n o b j e c t s
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
p r o p o s i t i o n s = [
      is_on_top ( [ ' cellphone_0 ' ] , [ ' table_4 ' ] ) ,
      is_on_top ( [ ' watch_0 ' ] , [ ' table_4 ' ] ) ,
      is_on_top ( [ ' keychain_0 ' ] , [ ' table_4 ' ] ) ,
      is_next_to ( [ ' cellphone_0 ' ] , [ ' watch_0 ' ] ) ,
      is_next_to ( [ ' watch_0 ' ] , [ ' keychain_0 ' ] ) ,
]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TEMPORAL GROUPS
# Pl a c e p r o p o s i t i o n s i n g roup s s . t . one group must be s a t i s f i e d b e f o r e the next .
# Example :
# [ [ 0 , 1 ] , [ 2 , 3 ] ] means p r op s 0 & 1 must be s a t i s f i e d b e f o r e p r op s 2 & 3 .
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
temporal_groups = [
      [ 0 , 1 , 2 , 3 , 4 ] ,
]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TIE CONSTRAINTS
# o p ti o n s : SameArgConstraint , Di f f e r e n t A r g C o n s t r ai n t
# Args :
# p r o p o s i ti o n _ i n d i c e s : L i s t [ i n t ]
# a r g_i n di c e s : L i s t [ i n t ]
# Example :
# SameArgConstraint ( [ 0 , 2 ] , [ 1 , 1 ] ) . Means : P r o p o si ti o n s 0 & 2 must
# match v al u e s on the argument a t argument inde x 1 and 1 , r e s p e c t i v e l y .
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
t i e _ c o n s t r a i n t s = [
]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TERMINAL SATISFACTION CONSTRAINT:
# We assume a l l p r o p o s i t i o n s must remain s a t i s f i e d t o the end o f the e pi s o d e .
# i f a p r o p o s i t i o n * sh ould * become u n s a t i s f i e d , add i t s inde x h e r e .
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
e x cl u d e _ fi n al _ c o n s t r ai n t = [ ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# mark True i f the t a s k has a f a t a l i s s u e
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
s ki p_ e pi s o d e = F al s e
r e a s o n = ""
```
Figure 10 World graph. The high-level policies of the human and robot agents leverage a hierarchical world graph containing information about rooms, furniture, and objects in the environment. This graph is updated based on the observations and actions of both agents.

# A.7 World graph: the perception framework for LLM agents

Scene-graph style hierarchical graphs have been shown to be effective for planning problems [Gu et al.](#page-11-10) [(2024)](#page-11-10); [Agia et al.](#page-11-17) [(2022)](#page-11-17); [Rana et al.](#page-13-10) [(2023)](#page-13-10). Inspired by such prior work, as illustrated in Figure [10,](#page-28-0) our world graph is a hierarchical multi-edge directed graph with K = 3 levels for representing the entities in the world. The nodes at first level correspond to rooms in the environment, followed by furniture at second, and objects and agents at third level. The root of this graph is an abstract house-node denoting the environment where tasks are taking place. Apart from the semantic information, each node also stores the 3D location of the entity and its affordance states e.g., clean/dirty, on/off, open/close, etc. The graph can then be serialized or accessed via specialized tools by ReAct policies. Prompts in Section [A.16](#page-56-0) provide an example of how the world graph is used by the LLMs in our baselines.

### A.7.1 Building and Updating the World Graph in Simulation

Low-Level Skill

Control Signal

High-Level Policy (LLM)

Low-Level Skill

Task Description

World Graph Perception

Robot Obs

Human Obs

<span id="page-28-0"></span>Sim Env Habitat 3.0

Control Signal

Low-Level Skill

Low-Level Skill

> High-Level Policy (LLM)

> > The initial graph is built by reading the the ground-truth room region annotations and furniture placements associated with a particular scene. The location of furniture and the region boundaries are used to associate each furniture to a specific room. This creates the initial two tiers, Room and Furniture, of the privileged world-graph. In partial-observability setting this is all the planner gets to start planning. However, in the full-observability setting, object-to-furniture assignments are read from each episode's initialization information and a third Object tier is added to the world-graph.

> > Under partial-observability setting during execution, we use the panoptic sensors attached to both agents to detect all visible objects in the current-frame. Then ground-truth simulation information is used to extract the housing furniture or agent for each of these objects, as well as the location, and this new information is added to the maintained graph. For full-observability setting, each frame latest graph is mined from simulator using current information for objects and overwrites the previous world-graph.

> > It is not guaranteed in our setup that the images will pick up an object that was placed, filled or powered on/off in the previous step. Therefore, in partial-observability setting, we also add deterministic updates to the graph based on previous action and its result, e.g. successful placement triggers deletion of edge between agent who was holding the object and an addition of edge between placed object and the furniture it was placed on per action arguments.

### A.7.2 Non-privileged Perception using ConceptGraphs

In order to study the dependence of the planner on the underlying world-graph, we also follow a modified version of ConceptGraphs (CG) pipeline [Gu et al.](#page-11-10) [(2024)](#page-11-10) to create the initial layout that is used to initialize our world-graph representation.

Modified Pipeline for Creating ConceptGraphs. We adapt the original CG pipeline to only use Meta-CLIP models for getting object name and category; using YOLO and SAM for object segmentation; and LLaMA3.1 for annotating room-label and inter-object spatiosemantic relations. Instead of using LLaVA and GPT for getting open-vocab descriptive object names, we use multi-perspective averaged CLIP embeddings of each object to classify its category given our closed object-vocabulary. For the required room-to-furniture relations, we extend CG pipeline by adding another prompt and query Llama3.1 model to assign room-labels to each entity given the categories of 10 closest entities to it.

Updating ConceptGraphs. Like the privileged world-graphs there are two modes of updating a ConcepGraphsinitialized non-privileged world-graph:

- 1. Observations. Using the same panoptic sensors we extract all the visible objects in current frame for both the agents. We use depth-sensors to extract the point-cloud associated with this object. We use this 3D location to first check if this is a redetection of a previously detected object. We use location, category and whether object-is-being-held-by-agent features to assess if this is a degenerate detection. If this is a new object detection then we use its location along-with bounding-boxes of existing furniture to check if this object is contained-within or on-top-of any of the recognized furniture-pieces. If it is then the world-graph is updated with this node and the edge.
- 2. Actions. Just like privileged world-graph, we can not guarantee sensors will pick up changed state of an object that is placed, powered, etc. Thus we add similar action based updates to this version as well. A special case is when non-privileged graph is upated by human agent's action arguments which are grounded in privieleged world-graph and may refer to same physical entity with different given names, e.g. backpack_0 is backpack_153 in non-privileged graph. We use a simple proximity and category matchig heuristic to match human's arguments to known entities in non-privileged graph, falling back on proximity based matching when no entity of same category are found.

Using ConceptGraphs with Simulated Skills Simulated skills require sim-handles of the placement furniture to snap objects onto them. The furniture in non-privileged graph do not have these sim-handles by design. Thus we come up with a simple proximity and category-name based matching to match a ground-truth furniture entity to a detected furniture-entity. If we can not find any ground-truth entity of the same category close to the detected furniture, we fallback on matching to the closest entity.

Prompts and Models used in ConceptGraphs Creation. CLIP model and pretrained backbone checkpoints: ViT-H-14-quickgelu, metaclip_fcc. Object detector: YOLO checkpoint yolov8x-worldv2. LLM: Llama3.1- 70B-Instruct

Room Annotation Prompt for Modified CG <| begin_o f_text|><| s t a r t_he ade r_id |> system <| end_header_id|>You a r e an e x p e r t on house l a y o u t s . You

Figure 11 ConceptGraph pipeline. Left to right: Point-cloud of the scene built from a trajectory; All objects detected by ConceptGraphs pipeline after assigning a category-name based on CLIP; Semantic visualization of the apartment with color-coded class categories per furniture

```
w i l l be gi v e n an i n p u t which w i l l d e s c r i b e QUERY_OBJECT. This o b j e c t w i l l be d e s c ri b e d by i t s name
and the 10 p i e c e s o f f u r n i t u r e c l o s e s t t o i t . You w i l l a s s i g n a ROOM_NAME t o t h i s o b j e c t . The i n p u t
w i l l be a JSON with f i e l d s "QUERY_OBJECT_NAME" and "CLOSEST_OBJECTS" . Your output sh ould a l s o be a
JSON c o n s i s t i n g o f key "ROOM_LABEL" . You sh ould onl y output the JSON and n o thin g e l s e .
You sh ould onl y a s s i g n one o f the f o l l o w i n g l a b e l s :
1 . bedroom
2 . l i v i n g room
3 . ki t c h e n
4 . di ni n g room
5 . h allw a y
6 . bathroom
7 . unknown : onl y when none o f the above s t r i n g s d e s c r i b e the o b j e c t <| eot_id|>
```
#### <span id="page-30-0"></span>A.8 Learned low-level robot skills

We provide the implementation details for training the learned low-level robot skills: [Explore, Navigate, OpenFurniture, CloseFurniture, PickObject, PlaceObject]. These skills are based on two types of basic skills: Navigate skill, which outputs robot base control velocity command given robot depth and sensor observations, and Manipulation skill, which outputs joint angles and base control velocity commands to reach the target location given robot depth and sensor observations. We then use these two skills to get the above robot skills.

#### A.8.1 Navigate Skill

We follow the learned low-level robot skill from [Puig et al.](#page-13-2) [(2024)](#page-13-2) to get the skill. We briefly describe the details for completeness. The goal of Navigate skill is to navigate to the target object given the target object location. The observation space includes (1) an arm depth camera (224 × 171 with hFOV of 55), (2) the relative pose of the target object location in 2-dim polar coordinate system. The action space includes (1) the linear and (2) angular robot base velocities with the range of −10 and +10 m/s. The reward function is to encourage the robot to move forward while facing the target location with a correct orientation. An additional navigation success reward is given if the robot can navigate close enough to the target object, and a collision penalty is given if the robot collides with obstacles in the scene. Moreover, a slack reward is given to let the robot navigate to the target in as few steps as possible. Finally, the skill is trained with DD-PPO distributing training.

Below is the list of skills based on navigate skill.

- Navigate: As described above.
- Explore: It is a composed skill that involves calling Navigate skills sequentially given a sequence of navigation waypoints.

#### A.8.2 Manipulation Skill

We follow the work from [Puig et al.](#page-13-2) [(2024)](#page-13-2) to train the skill. We briefly describe the details for completeness. The goal of Manipulation skill is to drive the robot's arm and base to reach the target object's location. The observation space includes (1) an arm depth camera (224 × 171 with hFOV of 55), (2) the relative pose of the target object in a 3-dim Cartesian coordinate system, (3) the 7-dim arm joint angles, (4) a binary holding detector, and (5) the relative pose of arm end-effector to the target resting location in a 3-dim Cartesian coordinate system. The action space includes (1) the linear and (2) angular base velocities with the range of −10 and +10 m/s, (3) the delta arm joint angles applied to the arm with the range of −5 × 10−2 and +5 × 10−2 (7-dim), and (4) a binary command to grasp or release the object. The reward function is to encourage the arm to move toward the object. In addition, a success reward is given if the robot interacts with the right target object. Moreover, a slack reward is given to let the robot complete the task in as few steps as possible. Finally, the skill is trained with DD-PPO distributing training.

Below is the list of skills based on manipulate skill.

- OpenFurniture: Given the drawer handle location, manipulate skill drives the arm and the base to the target. The articulated furniture is opened if the gripper location is close enough to the handle location.
- CloseFurniture: Given the drawer handle location, manipulate skill drives the arm and the base to the target. The articulated furniture is closed if the gripper location is close enough to the handle location.
- PickObject: Given the target object location, manipulate skill drives the arm and the base to the target. The object is snapped to the gripper if the gripper location is close enough to the target object location.
- PlaceObject: Given the target place location, manipulate skill drives the arm and the base to the target. The object is desnapped from the gripper to the target place location if the gripper location is close enough to the target place location.

# <span id="page-31-0"></span>A.9 Implementation Details for ReAct Agents

For all experiments, LLM inferrence is performed on two Nvidia A100 GPUs using the gpt-fast inference engine [PyTorch](#page-13-17) [(2023)](#page-13-17). Inference on LLama-3.1-70B (using tensor parallelism over two A100s), resulted in an average generation speed of 11.43 tokens/s. Each planning step required an average of 52 tokens resulting in a latency of 4.55 seconds per planning step. The average wall time to complete and entire episode (planning steps for both agents and simulation time) was 36.0 minutes. The finetuned model based on Llama-3.1-8B required an average of 0.53s per planning step. For those experiments, simulation time and human agent inference time remained unchanged, giving a final wall time of 25.3 minutes per episode.

All decentralized baselines had a maximum timeout of 50 replanning calls, while centralized baselines had a maximum timeout of 100 replanning calls (to account for the fact that one planner would need to plan for both agents). Additionally all baselines had a maximum timeout of 20000 simulation steps.

# A.9.1 Skill API Library

Below is a list of the skills available across all baselines. Agents acting in the robot role do not have access to state-modifying actions (Clean, Fill, Pour, PowerOff, PowerOn). The ReAct agents considered in the main paper do not have access to perception tools (FindAgentActionTool, FindObjectTool, FindReceptacleTool, FindRoomTool). We additionally study ReAct agents that query the environment via those tools, which we name ReAct-Tools.

- Clean : Used for cleaning an object.
- Close : Used for closing an articulated entity.
- Explore : Search a specific room by visiting various receptacles or furnitures in that room.
- Fill : Used for filling an object.
- Navigate : Used for navigating to an entity.
- Open : Used for opening an articulated entity.
- Pick : Used for picking up an object. The agent cannot hold more than one object at a time.
- Place : Used for placing an object on a target location. This requires the name of the object to be placed, the name of the furniture where it should be placed, spatial relation ("on" or "within") describing the relation between the object and furniture. The object to be placed must already be held by the agent (i.e. picked previously). Additionally, you can request to place the object near another object. For that you can optionally provide a spatial constraints ("next_to") and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API tempate: Place[<object_to_ be_placed>, <spatial_relation>, <furniture to be placed on>, <spatial_constraint>, <reference_ object>]. spatial_constraint and reference_object should be set to "None" when necessary.
- Pour : Used for pouring from one container to another. This skill will pour into the specified container from whichever container is currently held by the agent.
- PowerOff : Used for turning off a powered object
- PowerOn : Used for turning on a powered object
- Rearrange : Used for moving an object from its current location to the target location. This requires the name of the object to be rearranged, the name of the furniture where it should be placed, spatial relation ("on" or "within") describing the relation between the object and furniture. This skill will automatically pick the specified object and move to the target furniture and attempt to place it. Additionally, you can request to place the object near another object. For that you can optionally provide a spatial constraints ("next_to") and the name of the reference object. To place next to an object, the reference object must already be on the target furniture. API tempate: Rearrange[<object_to_be_placed>, <spatial_relation>, <furniture to be placed on>, <spatial_constraint>, <reference_object>]. spatial_constraint and reference_object should be set to "None" when necessary.
- Wait : Used to make agent stay idle for some time
- FindPartnerAgentActionTool : This tool will return a summary of the other agent's actions.
- FindObjectTool : Used to find the exact name/names of the object/objects of interest. An LLM will be used to distill relevant objects from the user query. Example (FindObjectTool[toys on the floor] or FindObjectTool[apples])
- FindReceptacleTool : Used to know the exact name of a receptacle. An LLM will be used to distill relevant receptacles from the user query. Example (FindReceptacleTool[a kitchen counter])
- FindRoomTool : Used to know the exact name of a room in the house. An LLM will be used to distill relevant rooms from the user query. Example (FindRoomTool[a room which might have something to eat])
- Done : Used to indicate that the agent has finished the task.

#### A.9.2 Constrained Generation

We follow the procedure described in [Geng et al.](#page-11-16) [(2023)](#page-11-16), constraining token sampling to only select tokens that consistent with at least one accepting string in the specified grammar. For each call to the LLM we build a grammar which will only accept valid tool calls on observed entities. Below is the base grammar used tool calls for all experiments. For experiments utilizing a summary of the world representation (i.e. ReAct, Finetuned see Section [4.1)](#page-7-0) the perception tools (FindObjectTool, FindReceptacleTool, etc.) are omitted. The rules for object, furniture, and room are set dynamically based on the agent's current world graph. This ensures skills are called only for entities that the agent knows exist.

# Tool Call Grammar

```
r o o t ::= N a vi g a te | Pick | Pl a c e | Open | Cl o se | Re ar r ange | PowerOn | PowerO f f | Clean | F i l l |
Pour | Expl o re | Wait | Fi n dR e c e p t a cl eT o ol | FindOb jectTool | FindAgentActionTool | FindRoomTool |
Done
N a vi g a te ::= " N a vi g a te [ " nav_target " ] "
Pick ::= " Pick [ " o b j e c t " ] "
Pl a c e ::= " Pl a c e [ " o b j e c t " , " WS s p a t i a l _ r e l a t i o n " , " WS f u r n i t u r e " , " WS ( ( s p a t i a l _ c o n s t r a i n t " , "
WS o b j_ o r_ f u r ni t u r e ) | ( ( " none " | "None " ) WS " , " WS ( " none " | "None " ) ) ) " ] "
Open ::= "Open [ " f u r n i t u r e " ] "
Cl o se ::= " Cl o se [ " f u r n i t u r e " ] "
Rea r r ange ::= " Re ar r ange [ " o b j e c t " , " WS s p a t i a l _ r e l a t i o n " , " WS f u r n i t u r e " , " WS
( ( s p a t i a l _ c o n s t r a i n t " , " WS o b j_ o r_ f u r ni t u r e ) | ( ( " none " | "None " ) WS " , " WS ( " none " | "None " ) ) ) " ] "
PowerOn ::= "PowerOn [ " o b j_ o r_ f u r ni t u r e " ] "
PowerO ff ::= " PowerOf f [ " o b j_ o r_ f u r ni t u r e " ] "
Clean ::= " Clean [ " o b j_ o r_ f u r ni t u r e " ] "
F i l l ::= " F i l l [ " o b j e c t " ] "
Pour ::= "Pour [ " o b j e c t " ] "
Expl o re ::= " Expl o re [ " room " ] "
Wait ::= "Wait [ " " ] "
Fi n dR e c e p t a cl eT o ol ::= " Fi n dR e c e p t a cl eT o ol [ " f r e e_ t e x t " ] "
FindOb jectTool ::= " FindOb jectTool [ " f r e e_ t e x t " ] "
FindAgentActionTool ::= " FindAgentActionTool [ " " ] "
FindRoomTool ::= "FindRoomTool [ " f r e e_ t e x t " ] "
Done ::= "Done [ ] "
nav_target ::= ( f u r n i t u r e | room | o b j e c t )
o b j e c t ::= " ob jec t_1 " | " ob jec t_2 " | . . .
o b j_ o r_ f u r ni t u r e ::= ( f u r n i t u r e | o b j e c t )
f u r n i t u r e ::= " f u r ni t u r e_ 1 " | " f u r ni t u r e_ 2 " | . . .
room ::= "room_1" | "room_2" | . . .
s p a t i a l _ c o n s t r a i n t ::= " next_to "
s p a t i a l _ r e l a t i o n ::= "on" | " wi t hi n "
```
<span id="page-33-1"></span>Figure 12 Overview of ReAct-RAG. RAG consists of two steps: (1) We first use a training dataset to obtain traces from LLMs and log traces with task instructions. (2) During runtime, we retrieve the traces by doing dot product on the instructions and obtaining the trace with the highest score, and finally put the trace in the prompt. We ensure that the dataset used for the first step is different from the one in the second step.

f r e e_ t e x t ::= [ " ' . : , ! a - zA -Z_0 - 9 ] * WS ::= [ ] *

#### A.9.3 Retrieval-Augmented Generation for ReAct agents

Retrieval-Augmented Generation (RAG) is the method of optimizing LLM text generation by querying an external database. However, there are several challenges to applying RAG in our setup. First, when implementing RAG, it is necessary to provide an external database from which LLMs can retrieve information. However, it is unclear how to effectively generate such a dataset and determine which content is most beneficial for solving the task. Second, once LLMs retrieve the information, it is unknown where to ingest this information into the generation process. To solve the above challenges, we develop an approach inspired by the recent literature that uses an LLM to generate its training dataset to iteratively optimize performance [(Pang](#page-12-17) [et al.,](#page-12-17) [2024;](#page-12-17) [Madaan et al.,](#page-12-18) [2024)](#page-12-18). As shown in Fig. [12,](#page-33-1) we first construct the dataset by collecting the successful traces generated by LLMs for solving the training tasks. Then, during test time, we select the most relevant trace by comparing the sentence similarity between a test instruction and the ones in the dataset. The selected trace is passed back to the LLM's promppt as a successful example trace. This represents a refinement process that uses its own past success experience to increase the chance of solving downstream tasks. Figure [12](#page-33-1) illustrates the high-level idea.

Specifically, we use the PARTNR train dataset to generate ReAct traces (where the human agent is ReAct-Tools and the robot agent is ReAct). In total, we generate 925 traces that successfully solve the tasks to form the RAG dataset. During evaluation time, we use sentence similarity computed by all-mpnet-base-v2 [Reimers](#page-13-18) [and Gurevych](#page-13-18) [(2019)](#page-13-18) to select the most similar instruction to the instruction at hand in the dataset, followed by adding the trace into the ReAct prompt.

# <span id="page-33-0"></span>A.10 Finetuning LLM agents

In this section, we describe how we finetune an LLM to build our Finetuned baseline. We detail the process to generate the data for finetuning our model and the training details below.

#### A.10.1 Data Generation for Finetuning

We train the model using successful traces from the ReAct baseline, which obtains the best decentralized results. In particular, we run this baseline together with a ReAct-Tools human on the training set, and keep the episodes with 1.0 success rate. If an episode reaches a 100% success rate some time during the task, but fails at the end, we keep the actions up until the success step and replace the last action with a Done[] action to finish the episode. This process results in 1,226 episodes. We then split each episode into the sequence of robot actions, and filter out those that resulted in failure. Our training set is constructed by building a prompt for each of the successful actions, as shown in Sec. [A.16.](#page-56-0) This process results in 15,889 training samples. Note that each action prompt contains the current state of the environment and the previous robot and human actions, but filters out the thought produced by the human or robot.

We also explored using the Heuristic-Expert baseline as a data source for finetuning, but we did not observe improvements in the resulting model. Given that this baseline plans using the ground truth evaluation function, we hypothesized it would help distill the natural language task into the right predicates. We followed the same process described above, obtaining 1250 episodes and 13939 training samples. We trained a model with ReAct, Heuristic-Expert and both data sources. We show the evaluation results for each model in Table [9,](#page-34-2) with the model trained with only React data performing the best.

<span id="page-34-2"></span>

| Data source              | Sim Steps ↓     | Success Rate ↑ | Percent Complete ↑ | Planning Cycles ↓ |
|--------------------------|-----------------|----------------|--------------------|-------------------|
| Heuristic-Expert         | 3477.82 ± 78.19 | 0.63 ± 0.02    | 0.79 ± 0.01        | 15.50 ± 0.24      |
| ReAct                    | 3228.96 ± 75.14 | 0.70 ± 0.01    | 0.84 ± 0.01        | 12.85 ± 0.24      |
| ReAct + Heuristic-Expert | 3552.96 ± 61.95 | 0.69 ± 0.01    | 0.83 ± 0.01        | 14.47 ± 0.30      |

Table 9 Performance of Finetuned model when using different data sources for finetuning. We measure performance using simulation steps required to finish the episode, success rate and percent complete on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the validation set. Heuristic centralized expert has access to the task evaluation function.

### A.10.2 Implementation Details

We train the model to predict, for every example, the action taken by the agent, which corresponds to the text after the <|reserved_special_token_0>| token.

We use a low rank adapter [Hu et al.](#page-11-15) [(2021)](#page-11-15) with r = 132, α = 128, dropout=0.01, on top of the value and query projection layers WV , WQ. We train all models on 4 A100 GPUs, with a batch size of 2 per GPU. The models are trained for 40,000 steps, which takes around 24 hours.

# <span id="page-34-1"></span>A.11 Additional Results

To supplement the results in Table [2,](#page-9-0) we have included Table [10,](#page-35-0) containing results from additional baselines on the validation set, and Table [11](#page-35-1) containing results on the PARTNR test set. ReAct rows use the same summary prompting format as the baselines in [2.](#page-9-0) ReAct-Tools requires the agents to use perception tools to observe the environment instead. ReAct-8B uses Llama-3.1-8B-Instruct as the robot model for an equal capacity comparison with Finetuned.

# <span id="page-34-0"></span>A.12 Analysis of collaborative behavior and efficiency of LLM agents

Since task state success and percent complete metrics look at overall team performance for the human and robot agents in our tasks, we also evaluate metrics that allow us to look at different aspects of collaborative behavior of the agents. We measure percentage of sub-tasks done by the robot (task offloading), ratio of unnecessary rearrangements over total successful rearrangements done by both agents (extraneous effort), and number of exploration steps needed before first object is picked (exploration efficiency) to analyze agent behaviors (Table [12)](#page-36-1). Our main findings are below:

Agents are able to find objects faster when collaborating as compared to solo, but only when they successfully co-ordinate. The exploration efficiency increases i.e., agents are able to find task-relevant objects in fewer

<span id="page-35-0"></span>

| Method           | Controllability | Skills  | Observability | Sim<br>Steps ↓    | Success<br>Rate ↑ | Completion<br>Rate ↑ | Planning<br>Cycles ↓ |
|------------------|-----------------|---------|---------------|-------------------|-------------------|----------------------|----------------------|
| Heuristic-Expert | Centralized     | Oracle  | Full          | 1260.88 ± 26.97   | 0.84 ± 0.01       | 0.94 ± 0.01          | N/A                  |
| ReAct            | Centralized     | Oracle  | Full          | 1347.43 ± 33.80   | 0.74 ± 0.01       | 0.88 ± 0.01          | 17.49 ± 0.34         |
| ReAct            | Decentralized   | Oracle  | Full          | 1915.63 ± 56.84   | 0.74 ± 0.01       | 0.86 ± 0.01          | 14.20 ± 0.34         |
| ReAct            | Centralized     | Oracle  | Partial       | 2298.13 ± 61.39   | 0.74 ± 0.01       | 0.85 ± 0.01          | 20.73 ± 0.51         |
| ReAct            | Decentralized   | Oracle  | Partial       | 3295.20 ± 76.27   | 0.73 ± 0.01       | 0.86 ± 0.01          | 15.24 ± 0.31         |
| ReAct-Tools      | Decentralized   | Oracle  | Partial       | 3622.52 ± 79.09   | 0.71 ± 0.01       | 0.85 ± 0.01          | 21.41 ± 0.34         |
| ReAct + RAG      | Decentralized   | Oracle  | Partial       | 3467.47 ± 82.39   | 0.71 ± 0.01       | 0.84 ± 0.01          | 14.75 ± 0.31         |
| Finetuned        | Decentralized   | Oracle  | Partial       | 3228.96 ± 75.14   | 0.70 ± 0.01       | 0.84 ± 0.01          | 12.85 ± 0.24         |
| ReAct            | Decentralized   | Learned | Partial       | 6494.88 ± 181.52  | 0.57 ± 0.02       | 0.76 ± 0.01          | 22.72 ± 0.58         |
| ReAct            | Decentralized   | Learned | ConceptGraph  | 12274.27 ± 212.65 | 0.25 ± 0.01       | 0.53 ± 0.01          | 26.74 ± 0.45         |
| ReAct-Single     | Single Agent    | Oracle  | Partial       | 2519.02 ± 57.48   | 0.73 ± 0.01       | 0.85 ± 0.01          | 18.68 ± 0.33         |
| ReAct-Single     | Single Agent    | Oracle  | Full          | 1590.20 ± 42.73   | 0.73 ± 0.01       | 0.85 ± 0.01          | 18.60 ± 0.38         |
| ReAct-8B         | Decentralized   | Oracle  | Partial       | 3699.45 ± 87.40   | 0.64 ± 0.02       | 0.80 ± 0.01          | 23.15 ± 0.47         |

Table 10 Baseline results on PARTNR validation set. We measure performance using simulation steps required to finish the episode, success rate and completion rate on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the validation set. Heuristic centralized expert has access to the task evaluation function. Collaboration enables higher task completion and success as compared to single agent task execution (shown in gray), at the expense of more simulation steps highlighting the coordination "burden".

<span id="page-35-1"></span>

| Method           | Controllability | Skills  | Observability | Sim<br>Steps ↓   | Success<br>Rate ↑ | Completion<br>Rate ↑ | Planning<br>Cycles ↓ |
|------------------|-----------------|---------|---------------|------------------|-------------------|----------------------|----------------------|
| Heuristic-Expert | Centralized     | Oracle  | Full          | 1184.74 ± 22.88  | 0.69 ± 0.02       | 0.89 ± 0.01          | N/A                  |
| ReAct            | Centralized     | Oracle  | Full          | 1348.71 ± 53.34  | 0.67 ± 0.02       | 0.86 ± 0.01          | 21.14 ± 0.59         |
| ReAct            | Centralized     | Oracle  | Partial       | 2590.82 ± 90.71  | 0.56 ± 0.02       | 0.80 ± 0.01          | 25.57 ± 0.64         |
| ReAct            | Decentralized   | Oracle  | Partial       | 3353.33 ± 70.03  | 0.63 ± 0.02       | 0.84 ± 0.01          | 17.38 ± 0.33         |
| ReAct-Tools      | Decentralized   | Oracle  | Partial       | 3810.15 ± 86.52  | 0.61 ± 0.02       | 0.83 ± 0.01          | 25.79 ± 0.41         |
| ReAct + RAG      | Decentralized   | Oracle  | Partial       | 3489.18 ± 79.54  | 0.62 ± 0.02       | 0.83 ± 0.01          | 17.55 ± 0.38         |
| Finetuned        | Decentralized   | Oracle  | Partial       | 3460.60 ± 78.33  | 0.51 ± 0.02       | 0.78 ± 0.01          | 14.73 ± 0.25         |
| ReAct            | Decentralized   | Learned | Partial       | 5905.88 ± 162.35 | 0.50 ± 0.02       | 0.76 ± 0.01          | 24.30 ± 0.60         |
| ReAct-Single     | Single Agent    | Oracle  | Partial       | 2632.30 ± 60.04  | 0.68 ± 0.01       | 0.85 ± 0.01          | 21.28 ± 0.37         |
| ReAct-Single     | Single Agent    | Oracle  | Full          | 1559.73 ± 36.02  | 0.73 ± 0.01       | 0.88 ± 0.01          | 21.06 ± 0.38         |
| ReAct-8B         | Decentralized   | Oracle  | Partial       | 4100.21 ± 98.97  | 0.51 ± 0.02       | 0.77 ± 0.01          | 27.65 ± 0.52         |

Table 11 Baseline results on PARTNR test set. We measure performance using simulation steps required to finish the episode, success rate and completion rate on the tasks, and the average number of planning cycles performed by the planner. Mean and standard error are reported over the test set. Heuristic centralized expert has access to the task evaluation function.

<span id="page-36-1"></span>

| Method        | Sim Steps ↓    | Task Offloading↑ | Extraneous Effort↓ | Exploration Efficiency↓ |
|---------------|----------------|------------------|--------------------|-------------------------|
| Decentralized | 3295.20 ±76.27 | 0.596 ±0.01      | 0.21 ±0.01         | 994.88 ±24.890          |
| Centralized   | 2298.13 ±61.39 | 0.49±0.01        | 0.04 ±0.004        | 684.06 ±21.71           |
| Single agent  | 2519.02 ±57.48 | -                | 0.047 ±0.01        | 1121.65 ±31.256         |

<span id="page-36-2"></span>Table 12 Analysis of collaboration characteristics for LLM agents. Average and standard errors for task offloading, extraneous effort, and exploration efficiency are reported over the successful episodes from the validation set for LLM agents using ReAct approach in partially observable setting.

| Method        | Success Rate per Task-type |           |           |               |  |  |  |  |
|---------------|----------------------------|-----------|-----------|---------------|--|--|--|--|
|               | Constraint-free            | Spatial   | Temporal  | Object states |  |  |  |  |
| Decentralized | 0.82 ±0.02                 | 0.82±0.02 | 0.60±0.03 | 0.66±0.03     |  |  |  |  |
| Centralized   | 0.84 ±0.02                 | 0.85±0.02 | 0.59±0.03 | 0.66±0.03     |  |  |  |  |
| Single agent  | 0.85±0.02                  | 0.81±0.02 | 0.58±0.03 | 0.68±0.03     |  |  |  |  |

Table 13 Task performance per task type. Average and standard errors of task success rate for episodes from the validation set categorized by task type. Performance is shown for LLM agents that use ReAct approach in partially observable setting.

steps, in centralized and decentralized settings. By computing the average number of exploration steps taken before the first object is picked up for a task, we find that single agents require on average 127 steps more to locate objects compared to multi-agent. However, in centralized setting, where the co-ordination between agents is better owning to a single LLM co-ordinating the actions of both agents, shows higher gains in such efficiency as compared to decentralized settings. The challenge LLMs face in coordinating exploration in multi-agent settings also negatively impacts human-LLM team performance in our HITL experiments when paired with humans (Table [3)](#page-10-0).

Poor co-ordination also leads to wasted effort and more steps to complete the tasks than solo execution. Despite multiple agents working together, agents take longer to complete the tasks in decentralized settings as compared to solo execution owing to poor co-ordination. Poor co-ordination is further highlighted by extraneous effort, which increases by 300% in decentralized settings as compared to solo execution. The agents often repeat parts of the task – unsure of whether the other agent really executed that part, and sometimes even undo successfully completed tasks.

The robot is able to offload more than half of the tasks from the human partner. The human-robot team takes longer to complete the task in decentralized setting, however, the robot offloads 60% tasks from the human partner, reducing their load of task execution. This highlights the potential of robots assisting humans more effectively as LLMs continue to advance in reasoning, coordination, and planning capabilities.

LLMs struggle to reason about temporal constraints and agent capabilitieswhile planning PARTNR tasks. Constraintfree and spatial tasks in PARTNR require the LLMs to reason about only the final configuration and states of objects. Instead, the temporal tasks in PARTNR require tracking states of one or more objects over the entire episode, making them challenging (Table [13)](#page-36-2). Likewise, heterogeneous tasks necessitate reasoning about task distribution conditioned on each agent's capabilities, which make them challenging.

# <span id="page-36-0"></span>A.13 Human-in-the-loop Evaluation for PARTNR tasks

# A.13.1 HITL Interface and Web Hosting

We adapt the existing human-in-the-loop (HITL) infrastructure from Habitat 3.0 [(Puig et al.,](#page-13-2) [2024)](#page-13-2) to support a more robust server-client architecture, with the server component hosted on AWS. Habitat3.0 HITL includes the ability to extend functionality to resource-constrained environments such as web browsers and VR devices, making the platform versatile for different user needs and experimental setups. Figure [13](#page-37-0) shows our HITL system running on a web browser. Detailed interface is shown in Figure [14.](#page-38-0) Our adaptation to AWS hosting is crucial for handling multiple clients simultaneously, especially non-experts without access to

<span id="page-37-0"></span>powerful machines or large Habitat datasets. The server-client architecture not only enhances scalability but also ensures flexibility, allowing the system to accommodate a variety of operating systems and hardware platforms. Furthermore, the system includes a matchmaking service that enables pairing participants for multi-user sessions. When a participant requests a task, they are redirected to a "lobby screen" where they are instructed to wait until the next participant arrives.

# A.13.2 Participant Recruitment and Quality Control

The study was performed through a 3rd party company specializing in large-scale annotations. The participants were recruited and compensated for their time by this company. The participants were English speakers, 18 years or older from the United States. For training them, we created a project and task overview video and guidelines. The participants were instructed to complete the tasks correctly and efficiently by themselves or with a partner. The participants also went through a tutorial where they performed some tasks to get acquainted with the interface before performing the main tasks. Each task took on average 3-5 minutes to complete. We recruited 129 non-expert participants in total.

### Filtering data:

Each task was completed up to 3 times, in all settings, until deemed successful through task evaluation (conducted online as the participants complete the tasks). With the Failure explanation output from the evaluation function, users were also given a natural language feedback at the end of an episode, describing what went wrong in an episode, if the task was not successful. For example, if actions were completed in the wrong order, or a spatial constraint like next-to was not respected appropriately. Users can use this information to update their actions in the next episode, improving their overall performance over time. By giving 4 tutorial episodes before the start of each experiment, and also a tutorial episode in the same house, we ensure that the users are deeply familiar with the tool and tasks before starting the actual experiment.

Once each task was completed successfully at least once or retried 3 times, we assimilate data by selecting one of the tries per episode: we select the successful try for successfully completed episodes, and the highest percent complete try for unsuccessful episodes. This collection of 1000 episodes for test and val is used for collecting performance statistics described in the next section.

### A.13.3 HITL experimental details

### Task Evaluation and Data Collection:

Using the enhanced server-client setup, we conduct comprehensive evaluations of 1000 tasks from the validation set, and 1000 tasks from the test set. These evaluations are designed to capture data in both single-user and multi-user scenarios. In the single-user setting, participants individually control a human agent within the simulator using traditional keyboard and mouse inputs, completing tasks without any external assistance.

<span id="page-38-0"></span>Figure 14 HITL Interface. Participants control human and robot agents using keyboard/mouse controls to complete the PARTNR tasks. Each participant has access to their partner's viewpoint and thereby current actions via a small viewport on the bottom right.

Conversely, the multi-user setting involves collaborative efforts where two participants work together, each controlling either a human or a robot agent. This collaborative approach is specifically designed to study the dynamics between multiple users and to evaluate whether such collaborations lead to more efficient task completion compared to single-user efforts in the PARTNR tasks.

### Human-AI Collaboration Experiment:

In addition to human-only interactions, we conduct experiments where a human collaborates with a robot controlled by a Language Model (LLM), specifically using the ReAct and Finetuned models as described in Section [4.1.](#page-7-0) The primary goal of these experiments is to evaluate the effectiveness of LLM-controlled agents in real-time collaboration with non-expert humans who have not previously interacted with these AI models. We track and compare the success rates (SR) and the percentage of tasks completed (PC) across various settings including single-user, multi-user, and human-AI collaborations. The results of these experiments are systematically documented and analyzed in Table [14,](#page-39-1) providing insights into the collaborative capabilities of human-AI pairs.

To enable this setting, we host LLMs on AWS nodes, and query them intermittently based on robot observations and actions. The HITL server now queries two clients - a human and a LLM. The human client sends commands to control the human agent, and the LLM client uses the hosted LLM to control the robot. Different baselines need different numbers of GPUs to keep the inference time reasonable. For hosting and 70B models, as used by the ReAct baselines, we use 4 A100 GPUs per model. For hosting a smaller 8B model used by the Finetuned baselines, we use 1 A100 GPU. This makes deploying smaller Finetuned models much scalable than larger 70B models.

### Efficiency and Task Offloading Metrics:

To further understand the efficiency of task completion across different experimental setups, we measure several key performance metrics beyond success rate and percent complete. These include the number of steps taken to complete tasks and the exploration efficiency, which is assessed by the number of steps participants take to pick the first object. Additionally, we evaluate the extraneous effort by noting actions that do not

<span id="page-39-1"></span>

| Method                    | Success     | Percent     | Sim              | Task         | Exploration       | Extraneous  |  |  |  |
|---------------------------|-------------|-------------|------------------|--------------|-------------------|-------------|--|--|--|
|                           | Rate ↑      | Complete ↑  | Steps ↓          | Offloading ↑ | Efficiency ↓      | Effort ↓    |  |  |  |
| Validation 1,000 episodes |             |             |                  |              |                   |             |  |  |  |
| Single-user               | 0.93 ± 0.01 | 0.96 ± 0.00 | 3046.99 ± 80.79  | N/A          | 2459.22 ± 26.75   | 0.09 ± 0.01 |  |  |  |
| Multi-user                | 0.93 ± 0.01 | 0.96 ± 0.00 | 2369.55 ± 49.33  | 0.59 ± 0.01  | 1762.47 ± 13.99   | 0.15 ± 0.01 |  |  |  |
| Human-ReAct               | 0.91 ± 0.01 | 0.96 ± 0.02 | 4267.71 ± 83.40  | 0.16 ± 0.01  | 2624.39 ± 26.05   | 0.12 ± 0.01 |  |  |  |
| Human-Finetuned           | 0.92 ± 0.01 | 0.96 ± 0.00 | 3443.33 ± 61.46  | 0.26 ± 0.01  | 2164.94 ± 21.31   | 0.13 ± 0.01 |  |  |  |
| Test 1,000 episodes       |             |             |                  |              |                   |             |  |  |  |
| Single-user               | 0.89 ± 0.01 | 0.95 ± 0.00 | 3937.87 ± 110.53 | N/A          | 2737.44 ± 25.27   | 0.13 ± 0.00 |  |  |  |
| Multi-user                | 0.85 ± 0.01 | 0.95 ± 0.00 | 2667.86 ± 58.07  | 0.60 ± 0.01  | 1889.56 ± 14.9=69 | 0.20 ± 0.00 |  |  |  |
| Human-ReAct               | 0.87 ± 0.01 | 0.95 ± 0.00 | 4080.10 ± 72.24  | 0.12 ± 0.00  | 2449.50 ± 19.98   | 0.18 ± 0.01 |  |  |  |
| Human-Finetuned           | 0.87 ± 0.01 | 0.95 ± 0.00 | 3403.03 ± 62.08  | 0.26 ± 0.01  | 2162.20 ± 19.40   | 0.20 ± 0.01 |  |  |  |

Table 14 Human-in-the-Loop evaluation. We evaluate the performance of 2-person human teams and human-LLM teams, comparing them to solo human performance on PARTNR tasks using metrics described in Section [4.1.](#page-7-0) The human-LLM teams with SoTA LLMs is slower than solo human.

contribute directly to task completion. Another critical metric we analyze is the ratio of work completed by the robot, referred to as task offloading. Ideally, in a well-coordinated human-AI team, the task offloading ratio should approach 0.5, indicating an efficient division of labor between the human and the robot.

# <span id="page-39-0"></span>A.13.4 HITL analysis

Humans are significantly better than LLMs at PARTNR tasks. Both single and multi-human settings achieve a success rate of 0.93 on PARTNR validation tasks, while ReAct without any privileged information only achieves 0.30 (Table [2,](#page-9-0) row (i)). This indicates a huge gap in LLM planning performance. We observe a slightly lower human performance on the test set (0.89), also in line with a lower LLM performance in Table [11](#page-35-1) on this dataset. This indicates that the tasks in our test set are more challenging than the validation set for both humans and LLMs alike, potentially due to human annotations aimed at making them more complex and diverse.

Finetuned LLMs perform better than ReAct when coordinating with real humans. When deployed with real humans-in-the-loop on the validation set, the finetuned model is faster than ReAct at task completion (3275 steps with finetuned versus 4484 with ReAct on the validation set). It is also able to offload more tasks from humans than ReAct (26% with finetuned as compared to 16% with ReAct). This reflects that smaller models with faster inference can improve human experience in real-world deployment. This result is also reflected in the test set where finetuned model outperforms ReAct. Interestingly, the automated eval performance of finetuned is worse than ReAct on the test set, but the HITL performance is better, indicating that faster inference is more critical than task success when working with real humans.

LLMs struggle at coordination, hampering human performance. Despite the Finetuned being faster than ReAct when collaborating with humans, both approaches are slower than the human doing the task alone. In contrast, two humans working together complete the task faster than a single human (2369 steps vs. 3046 with multi- and single-user respectively). This result is in line with the automated evaluation we observed in Table [1,](#page-2-0) where multi-agent LLMs are also slower than a single-agent LLM. This result further reinforces that LLMs suffer at coordination; while humans are able to coordinate and divide tasks between each other, decentralized LLMs are unable to do so. We observe the same effect in the test set, further reinforcing this finding.

LLMs are able to offload tasks from humans. Despite the aforementioned increase in the number of steps for task completion, robots guided by the finetuned model successfully offload 26% of tasks from humans. This indicates that LLMs can still offer assistance when collaborating with real human partners. Nonetheless, there remains significant potential for improvement.

LLM's inefficiency to explore reduces the team performance when paired with humans. In multi-user condition, the two humans start in different parts of a house, and explore efficiently to locate task-relevant objects more quickly than a single user – as evidenced by the reduced number of steps before first pick (1762 steps with multi-user vs. 2459 steps for a single user). However, this efficiency is reduced when humans are paired with

LLMs (2120 steps with finetuned and 2791 steps with ReAct), indicating that LLMs struggle to coordinate at both task completion and exploration.

### <span id="page-40-0"></span>A.14 Prompts for benchmark task and evaluation

#### A.14.1 Task generation prompts

Prompts are similar for the different task types, with the primary difference being the in-context examples.

#### Constraint-free task generation prompt

You a r e a system t h a t g e n e r a t e s t a s k s f o r r o b o t s t o per fo rm with humans . Do not be v e r b o s e . Answer the q u e s ti o n with no added q u a l i f i c a t i o n s o r c a v e a t s . J u s t d i r e c t l y p r o vi d e the answer i n JSON. You w i l l be gi v e n a d e s c r i p t i o n o f a house with o b j e c t s and f u r n i t u r e and your t a s k i s t o p r o vi d e 5 i n s t r u c t i o n s f o r t a s k s t h a t a r ob o t and a human c o ul d be d oin g t o g e t h e r i n t h a t house , u si n g the o b j e c t s and f u r n i t u r e . For each t a sk , p r o vi d e the i n i t i a l s t a t e o f o b j e c t s i n the house , the i n s t r u c t i o n t h a t sh ould be per formed , and f i n a l s t a t e o f the o b j e c t s a f t e r the i n s t r u c t i o n i s pe r fo rmed . The i n i t i a l and f i n a l s t a t e w i l l c o n t ai n a l i s t o f d i c t i o n a r i e s , each with an o b j e c t type , the number o f o b j e c t s o f t h a t type , t h e i r l o c a t i o n on a f u r n i t u r e o r f l o o r , and the r e gi o n o f the house where they a r e i n e . g . , bedroom . Follow the next p r i n c i p l e s : 1 . The i n s t r u c t i o n sh ould be gi v e n a s i f the human d oin g the t a s k wanted the r o b o t t o per form p a r t o f i t . In some c a s e s the t a s k w i l l be done t o g e t h e r , i n o t h e r c a s e s , the human and r ob o t w i l l be d oin g d i f f e r e n t t a s k s . 2 . The i n i t i a l and f i n a l s t a t e sh ould c o n t ai n o b j e c t s o f d i f f e r e n t type s , and sometimes m ul ti pl e o b j e c t s o f a type . 3 . Some o f the i n s t r u c t i o n s sh ould be s e m a n t i c al l y ri c h , i n p a r t i c u l a r they sh ould r e f e r t o c l a s s e s o r g roup s o f o b j e c t s . 4 . The i n s t r u c t i o n s shouldn ' t be d e t a i l e d and e x pl ai n a l l the s t e p s , but the high - l e v e l . 5 . The r o b o t can onl y r e a r r a n g e o b j e c t s and open c o n t ai n e r s , the human can do more t a s k s e . g . , tu rn on lamp , cl e a n pl a t e s , f i l l up p i t c h e r . 6 . The i n s t r u c t i o n sh ould c o n t ai n a c l e a r g o al and a t l e a s t two s t e p s a s s o c i a t e d with the g o al . 7 . Ensure t h a t i n s t r u c t i o n s a r e d i v e r s e from each o t h e r . Some i n s t r u c t i o n s sh ould c o n t ai n s p a t i a l s p e c i f i e r s such a s " next t o " , " l e f t " , " r i g h t " , " b e si d e " , " ne a r " , " f r o n t " , " s i d e " . While some o t h e r i n s t r u c t i o n s sh ould c o n t ai n temp o r al o rde r , which can be s p e c i f i e d u si n g words such a s " a f t e r " , " then " , " b e f o r e " e t c . For i n s t a n c e : " F i l l up the k e t t l e and then tu rn i t on . A f t e r that , b ri n g two cups t o the di n ni n g t a b l e . " 8 . The i n s t r u c t i o n s sh ould c o n t ai n d i v e r s e a c t i o n s such a s " tu rn on/ o f f " , " f i l l " , " c l e a r " , " s e t " e t c . and o b j e c t s t a t e s such a s " cl e a n " , " d i r t y " , " open " , " c l o s e " e t c . w hil e r e f e r r i n g t o o b j e c t s . You w i l l be gi v e n a l l the p i e c e s o f f u r n i t u r e i n the house . You w i l l a l s o be gi v e n a l l the d i f f e r e n t t y p e s o f o b j e c t s t h a t you can u se . You can s p e c i f y m ul ti pl e i n s t a n c e s o f an o b j e c t type . Make s u r e you i n s t r u c t i o n i n c l u d e s the o b j e c t t y p e s and f u r n i t u r e p r e s e n t i n the l i s t below . The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { h o u s e_ f u r ni t u r e } You can u se the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s an example with two i n s t r u c t i o n s : JSON_OUTPUT: [ {{ " i n i t i a l s t a t e " : [ {{ " ob jec t_ t ype " : "lamp " , "how_many " : 1 , " furniture_name " : " table_10 " , " r e gi o n " : "living_room_1 " } } , {{ " ob jec t_ t ype " : " book " , "how_many " : 3 , " furniture_name " : " table_11 " , " r e gi o n " : "living_room_1 "

```
} } ,
           {{
                 " ob jec t_ t ype " : " t o y_ v e hi cl e " ,
                 "how_many " : 2 ,
                 " furniture_name " : " f l o o r " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " toy_c ac tu s " ,
                 "how_many " : 1 ,
                 " furniture_name " : " table_1 " ,
                 " r e gi o n " : "living_room_1 "
           }}
      ] ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : "lamp " ,
                 "how_many " : 1 ,
                 " furniture_name " : " table_10 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " book " ,
                 "how_many " : 3 ,
                 " furniture_name " : " shel ve s_ 2 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " t o y_ v e hi cl e " ,
                 "how_many " : 2 ,
                 " furniture_name " : "bed_2 " ,
                 " r e gi o n " : "bedroom_1"
           } } ,
           {{
                 " ob jec t_ t ype " : " toy_c ac tu s " ,
                 "how_many " : 1 ,
                 " furniture_name " : "bed_2 " ,
                 " r e gi o n " : "bedroom_1"
           } } ,
      ] ,
     " i n s t r u c t i o n " : "We need t o cl e a n up the l i v i n g room . Move a l l t o y s and books t o the s h e l f i n
     the l i v i n g room . " ,
     " r e a s o n " : "The t a s k i n v o l v e s moving m ul ti pl e o b j e c t s t o the s h e l f i n the l i v i n g room . "
} } ,
{{
     " i n i t i a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 3 ,
                 " furniture_name " : " cabinet_2 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " counter_1 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 5 ,
                 " furniture_name " : " cabinet_5 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
```
{{

```
" ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
            } } ,
      ] ,
     " i n s t r u c t i o n " : " Help me s e t up a t a b l e f o r di n n e r i n the li vi n g r o om f o r 2 p e o pl e . Pl a c e 2
      p l a t e s and 2 g l a s s e s on the t a b l e . " ,
     " r e a s o n " : "The t a s k i n c l u d e s s e m a n t i c a ll y r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) . "
}}
]
Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above .
```

```
Your output sh ould onl y be :
JSON_OUTPUT: r e s u l t _ l i s t
where r e s u l t _ l i s t sh ould be a JSON l i s t with the i n s t r u c t i o n s .
Let ' s t hi n k through t h i s c a r e f u l l y , s t e p by s t e p .
```
#### Spatial task generation prompt

```
You a r e a system t h a t g e n e r a t e s t a s k s f o r r o b o t s t o per fo rm with humans .
Do not be v e r b o s e . Answer the q u e s ti o n with no added q u a l i f i c a t i o n s o r c a v e a t s . J u s t d i r e c t l y
p r o vi d e the answer i n JSON.
You w i l l be gi v e n a d e s c r i p t i o n o f a house with o b j e c t s and f u r n i t u r e and your t a s k i s t o p r o vi d e 5
i n s t r u c t i o n s f o r t a s k s t h a t a r ob o t and a human c o ul d be d oin g t o g e t h e r i n t h a t house , u si n g the
o b j e c t s and f u r n i t u r e .
For each t a sk , p r o vi d e the i n i t i a l s t a t e o f o b j e c t s i n the house , the i n s t r u c t i o n t h a t sh ould be
per formed , and f i n a l s t a t e o f the o b j e c t s a f t e r the i n s t r u c t i o n i s pe r fo rmed .
The i n i t a l and f i n a l s t a t e w i l l c o n t ai n a l i s t o f d i c t i o n a r i e s , each with an o b j e c t type , the
number o f o b j e c t s o f t h a t type , t h e i r l o c a t i o n on a f u r n i t u r e o r f l o o r , and the r e gi o n o f the house
where they a r e i n e . g . , bedroom .
Follow the next p r i n c i p l e s :
1 . The i n s t r u c t i o n sh ould be gi v e n a s i f the human d oin g the t a s k wanted the r o b o t t o per form p a r t
o f i t . In some c a s e s the t a s k w i l l be done t o g e t h e r , i n o t h e r c a s e s , the human and r ob o t w i l l be
d oin g d i f f e r e n t t a s k s .
2 . The i n i t i a l and f i n a l s t a t e sh ould c o n t ai n o b j e c t s o f d i f f e r e n t type s , and sometimes m ul ti pl e
o b j e c t s o f a type .
3 . Some o f the i n s t r u c t i o n s sh ould be s e m a n t i c al l y ri c h , i n p a r t i c u l a r they sh ould r e f e r t o c l a s s e s
o r g roup s o f o b j e c t s .
4 . The i n s t r u c t i o n s shouldn ' t be d e t a i l e d and e x pl ai n a l l the s t e p s , but the high - l e v e l .
5 . The r o b o t can onl y r e a r r a n g e o b j e c t s and open c o n t ai n e r s , the human can do more t a s k s e . g . , tu rn
on lamp , cl e a n pl a t e s , f i l l up p i t c h e r .
6 . The i n s t r u c t i o n sh ould c o n t ai n a c l e a r g o al and a t l e a s t two s t e p s a s s o c i a t e d with the g o al .
7 . Ensure t h a t i n s t r u c t i o n s a r e d i v e r s e from each o t h e r .
8 . All the i n s t r u c t i o n s sh ould c o n t ai n a t l e a s t one o f the s p a t i a l s p e c i f i e r s from t h i s l i s t : " next
t o " , " l e f t " , " r i g h t " , " b e si d e " , " ne a r " , " f r o n t " , " s i d e " .
9 . The i n s t r u c t i o n s sh ould c o n t ai n d i v e r s e a c t i o n s such a s " tu rn on/ o f f " , " f i l l " , " c l e a r " , " s e t "
e t c . and o b j e c t s t a t e s such a s " cl e a n " , " d i r t y " , " open " , " c l o s e " e t c .
You w i l l be gi v e n a l l the p i e c e s o f f u r n i t u r e i n the house .
You w i l l a l s o be gi v e n a l l the d i f f e r e n t t y p e s o f o b j e c t s t h a t you can u se . You can s p e c i f y
m ul ti pl e i n s t a n c e s o f an o b j e c t type .
Make s u r e you i n s t r u c t i o n i n c l u d e s the o b j e c t t y p e s and f u r n i t u r e p r e s e n t i n the l i s t below .
The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e :
{ h o u s e_ f u r ni t u r e }
You can u se the f o l l o w i n g o b j e c t s :
{ o b j e c t s _ l i s t }
Here i s an example with two i n s t r u c t i o n s :
JSON_OUTPUT: [
{{
      " i n i t i a l s t a t e " : [
            {{
                  " ob jec t_ t ype " : " v a se " ,
                  "how_many " : 1 ,
                  " furniture_name " : " table_10 " ,
```

```
" r e gi o n " : "living_room_1 "
```

```
} } ,
           {{
                 " ob jec t_ t ype " : " s t u f f e d_ t o y " ,
                 "how_many " : 2 ,
                 " furniture_name " : " f l o o r " ,
                 " r e gi o n " : "bedroom_2"
           } } ,
           {{
                 " ob jec t_ t ype " : " c a n dl e " ,
                 "how_many " : 1 ,
                 " furniture_name " : " chest_o f_drawers_2 " ,
                 " r e gi o n " : "bedroom_1"
           }}
      ] ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " v a se " ,
                 "how_many " : 1 ,
                 " furniture_name " : " shel ve s_ 2 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " s t u f f e d_ t o y " ,
                 "how_many " : 2 ,
                 " furniture_name " : " shel ve s_ 2 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " c a n dl e " ,
                 "how_many " : 1 ,
                 " furniture_name " : " shel ve s_ 2 " ,
                 " r e gi o n " : "living_room_1 "
           }}
      ] ,
     " i n s t r u c t i o n " : "Let ' s d e c o r a t e ! Put the v a se on the s h e l f . Then , s e t a c a n dl e and a s t u f f e d_ t o y
     on each s i d e o f the v a se . " ,
     " r e a s o n " : "The t a s k i n c l u d e s s p a t i a l c o n s t r a i n t s p e c i f i e d by ' si d e ' . "
} } ,
{{
     " i n i t i a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 3 ,
                 " furniture_name " : " cabinet_2 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " counter_1 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 5 ,
                 " furniture_name " : " cabinet_5 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
      ] ,
```

```
44
```

```
" i n s t r u c t i o n " : " Help me s e t up a t a b l e f o r di n n e r i n the li vi n g r o om f o r 2 p e o pl e . Pl a c e 2
      p l a t e s and 2 g l a s s e s on the t a b l e . There sh ould be a f o r k next t o each p l a t e " ,
      " r e a s o n " : "The t a s k i n c l u d e s s e m a n t i c a ll y r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) and s p a t i a l
      c o n s t r a i n t s s p e c i f i e d by the word ' next ' . "
}}
]
Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above .
Your output sh ould onl y be :
JSON_OUTPUT: r e s u l t _ l i s t
where r e s u l t _ l i s t sh ould be a JSON l i s t with the i n s t r u c t i o n s .
Let ' s t hi n k through t h i s c a r e f u l l y , s t e p by s t e p .
```
#### Temporal task generation prompt

You a r e a system t h a t g e n e r a t e s t a s k s f o r r o b o t s t o per fo rm with humans . Do not be v e r b o s e . Answer the q u e s ti o n with no added q u a l i f i c a t i o n s o r c a v e a t s . J u s t d i r e c t l y p r o vi d e the answer i n JSON. You w i l l be gi v e n a d e s c r i p t i o n o f a house with o b j e c t s and f u r n i t u r e and your t a s k i s t o p r o vi d e 5 i n s t r u c t i o n s f o r t a s k s t h a t a r ob o t and a human c o ul d be d oin g t o g e t h e r i n t h a t house , u si n g the o b j e c t s and f u r n i t u r e . For each t a sk , p r o vi d e the i n i t i a l s t a t e o f o b j e c t s i n the house , the i n s t r u c t i o n t h a t sh ould be per formed , and f i n a l s t a t e o f the o b j e c t s a f t e r the i n s t r u c t i o n i s pe r fo rmed . The i n i t a l and f i n a l s t a t e w i l l c o n t ai n a l i s t o f d i c t i o n a r i e s , each with an o b j e c t type , the number o f o b j e c t s o f t h a t type , t h e i r l o c a t i o n on a f u r n i t u r e o r f l o o r , and the r e gi o n o f the house where they a r e i n e . g . , bedroom . Follow the next p r i n c i p l e s : 1 . The i n s t r u c t i o n sh ould be gi v e n a s i f the human d oin g the t a s k wanted the r o b o t t o per form p a r t o f i t . In some c a s e s the t a s k w i l l be done t o g e t h e r , i n o t h e r c a s e s , the human and r ob o t w i l l be d oin g d i f f e r e n t t a s k s . 2 . The i n i t i a l and f i n a l s t a t e sh ould c o n t ai n o b j e c t s o f d i f f e r e n t type s , and sometimes m ul ti pl e o b j e c t s o f a type . 3 . Some o f the i n s t r u c t i o n s sh ould be s e m a n t i c al l y ri c h , i n p a r t i c u l a r they sh ould r e f e r t o c l a s s e s o r g roup s o f o b j e c t s . 4 . The i n s t r u c t i o n s shouldn ' t be d e t a i l e d and e x pl ai n a l l the s t e p s , but the high - l e v e l . 5 . The r o b o t can onl y r e a r r a n g e o b j e c t s and open c o n t ai n e r s , the human can do more t a s k s e . g . , tu rn on lamp , cl e a n pl a t e s , f i l l up p i t c h e r . 6 . The i n s t r u c t i o n sh ould c o n t ai n a c l e a r g o al and a t l e a s t two s t e p s a s s o c i a t e d with the g o al . 7 . Ensure t h a t i n s t r u c t i o n s a r e d i v e r s e from each o t h e r . 8 . All the i n s t r u c t i o n s sh ould c o n t ai n temp o r al o rde r , s p e c i f i e d u si n g one o f the words from t h i s l i s t : " a f t e r " , " then " , " b e f o r e " , " f i n a l l y " , " f i r s t " . For i n s t a n c e : " F i l l up the k e t t l e and then tu rn i t on . A f t e r that , b ri n g two cups t o the di n ni n g t a b l e . " 9 . The i n s t r u c t i o n s sh ould c o n t ai n d i v e r s e a c t i o n s such a s " tu rn on/ o f f " , " f i l l " , " c l e a r " , " s e t " e t c . and o b j e c t s t a t e s such a s " cl e a n " , " d i r t y " , " open " , " c l o s e " e t c . You w i l l be gi v e n a l l the p i e c e s o f f u r n i t u r e i n the house . You w i l l a l s o be gi v e n a l l the d i f f e r e n t t y p e s o f o b j e c t s t h a t you can u se . You can s p e c i f y m ul ti pl e i n s t a n c e s o f an o b j e c t type . Make s u r e you i n s t r u c t i o n i n c l u d e s the o b j e c t t y p e s and f u r n i t u r e p r e s e n t i n the l i s t below . The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e : { h o u s e_ f u r ni t u r e } You can u se the f o l l o w i n g o b j e c t s : { o b j e c t s _ l i s t } Here i s an example with two i n s t r u c t i o n s : JSON_OUTPUT: [ {{ " i n i t i a l s t a t e " : [ {{ " ob jec t_ t ype " : "lamp " , "how_many " : 1 , " furniture_name " : " table_10 " , " r e gi o n " : "living_room_1 " } } ,

{{

" ob jec t_ t ype " : " book " ,

```
"how_many " : 3 ,
                 " furniture_name " : " table_11 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " t o y_ v e hi cl e " ,
                 "how_many " : 2 ,
                 " furniture_name " : " f l o o r " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " toy_c ac tu s " ,
                 "how_many " : 1 ,
                 " furniture_name " : " table_1 " ,
                 " r e gi o n " : "living_room_1 "
           }}
      ] ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : "lamp " ,
                 "how_many " : 1 ,
                 " furniture_name " : " table_10 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " book " ,
                 "how_many " : 3 ,
                 " furniture_name " : " shel ve s_ 2 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " t o y_ v e hi cl e " ,
                 "how_many " : 2 ,
                 " furniture_name " : "bed_2 " ,
                 " r e gi o n " : "bedroom_1"
           } } ,
           {{
                 " ob jec t_ t ype " : " toy_c ac tu s " ,
                 "how_many " : 1 ,
                 " furniture_name " : "bed_2 " ,
                 " r e gi o n " : "bedroom_1"
           } } ,
      ] ,
     " i n s t r u c t i o n " : "We need t o cl e a n up the l i v i n g room . Move a l l t o y s t o the bedroom and the books
     t o the s h e l f . A f t e r that , tu rn on the lamp i n the l i v i n g room . " ,
     " r e a s o n " : "The t a s k i n c l u d e s temp o r al c o n s t r a i n t s s p e c i f i e d by ' a f t e r ' . "
} } ,
{{
     " i n i t i a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 3 ,
                 " furniture_name " : " cabinet_2 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " counter_1 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 5 ,
                 " furniture_name " : " cabinet_5 " ,
                 " r e gi o n " : " kitchen_1 "
           } } ,
     " f i n a l s t a t e " : [
           {{
                 " ob jec t_ t ype " : " p l a t e " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
           } } ,
           {{
                 " ob jec t_ t ype " : " g l a s s " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
```

```
" r e gi o n " : "living_room_1 "
            } } ,
            {{
                 " ob jec t_ t ype " : " f o r k " ,
                 "how_many " : 2 ,
                 " furniture_name " : " table_8 " ,
                 " r e gi o n " : "living_room_1 "
            } } ,
      ] ,
      " i n s t r u c t i o n " : " Help me s e t up a t a b l e f o r di n n e r i n the li vi n g r o om f o r 2 p e o pl e . F i r s t pl a c e 2
      p l a t e s on the t a b l e . Then , pl a c e g l a s s e s and f o r k s next t o each p l a t e . " ,
      " r e a s o n " : "The t a s k i n c l u d e s s e m a n t i c a ll y r i c h d e s c r i p t i o n s ( s e t up a t a b l e ) and temp o r al
      c o n s t r a i n t s s p e c i f i e d by the words ' f i r s t ' and ' then ' . "
}}
]
Generate a JSON l i s t with {k} i n s t r u c t i o n s l i k e the examples above .
Your output sh ould onl y be :
JSON_OUTPUT: r e s u l t _ l i s t
where r e s u l t _ l i s t sh ould be a JSON l i s t with the i n s t r u c t i o n s .
Let ' s t hi n k through t h i s c a r e f u l l y , s t e p by s t e p .
                                            Large-scale task generation prompt
You a r e a system t h a t g e n e r a t e s t a s k s f o r r o b o t s t o per fo rm with humans .
Do not be v e r b o s e . Answer the q u e s ti o n with no added q u a l i f i c a t i o n s o r c a v e a t s . D i r e c t l y p r o vi d e
the answer i n JSON.
You w i l l be gi v e n a d e s c r i p t i o n o f a house with o b j e c t s and f u r n i t u r e . You w i l l a l s o be gi v e n a
sample t a s k . Your j ob i s t o modi fy t h i s sample t a s k s o t h a t i t i s a p p l i c a b l e t o t h i s house .
Here i s an example :
Task : [
{{
      " i n s t r u c t i o n " : "Move the k e t t l e and the ju g from the l i v i n g room t o the ki t c h e n and f i l l the
      k e t t l e with water , then tu rn on the k e t t l e . F i n a l l y cl e a n the l i v i n g room t a b l e . " ,
      " i n i t i a l _ s t a t e " : [
            {{
                 "number " : 1 ,
                 " o b j e c t _ c l a s s e s " : [
                       " k e t t l e "
                  ] ,
                 " fu rni tu re_n ame s " : [
                       " table_0 "
                  ] ,
                 " all ow e d_ r e gi o n s " : [
                       "living_room_0 "
                  ]
            } } ,
            {{
                 "number " : 1 ,
                 " o b j e c t _ c l a s s e s " : [
                       " ju g "
                  ] ,
                 " fu rni tu re_n ame s " : [
                       " table_1 "
                  ] ,
                 " all ow e d_ r e gi o n s " : [
                       "living_room_0 "
                  ]
            }}
      ]
}}
]
JSON_OUTPUT: [
{{
      " i n s t r u c t i o n " : "Move the jug , k e t t l e , te ap o t , and cup from the di ni n g t a b l e t o the ki t c h e n and
      f i l l a l l with water . Turn on the lamp , and cl e a n the di ni n g t a b l e . " ,
      " i n i t i a l _ s t a t e " : [
            {{
                 "number " : 1 ,
                 " o b j e c t _ c l a s s e s " : [
                       " ju g "
                  ] ,
                 " fu rni tu re_n ame s " : [
```

```
" table_1 "
                  ] ,
                  " all ow e d_ r e gi o n s " : [
                        "living_room_0 "
                  ]
            } } ,
            {{
                  "number " : 1 ,
                  " o b j e c t _ c l a s s e s " : [
                        " k e t t l e "
                  ] ,
                  " fu rni tu re_n ame s " : [
                        " table_1 "
                  ] ,
                  " all ow e d_ r e gi o n s " : [
                        "living_room_0 "
                  ]
            } } ,
            {{
                  "number " : 1 ,
                  " o b j e c t _ c l a s s e s " : [
                        " t e a p o t "
                  ] ,
                  " fu rni tu re_n ame s " : [
                        " table_1 "
                  ] ,
                  " all ow e d_ r e gi o n s " : [
                        "living_room_0 "
                  ]
            } } ,
            {{
                  "number " : 1 ,
                  " o b j e c t _ c l a s s e s " : [
                        " cup "
                  ] ,
                  " fu rni tu re_n ame s " : [
                        " table_0 "
                  ] ,
                  " all ow e d_ r e gi o n s " : [
                        "living_room_0 "
                  ]
            } } ,
            {{
                  "number " : 1 ,
                  " o b j e c t _ c l a s s e s " : [
                        "lamp"
                  ] ,
                  " fu rni tu re_n ame s " : [
                        " table_1 "
                  ] ,
                  " all ow e d_ r e gi o n s " : [
                        "living_room_0 "
                  ]
            }}
      ]
}}
]
The house has the f o l l o w i n g rooms , each with the f o l l o w i n g f u r n i t u r e :
{ h o u s e_ f u r ni t u r e }
You can u se the f o l l o w i n g o b j e c t s :
{ o b j e c t s _ l i s t }
Here i s the t a s k :
Task : [
{ t a s k }
]
Modi fy t h i s t a s k t o g e n e r a t e a JSON l i s t o f t a s k s , u si n g the rooms and f u r n i t u r e from t h i s house .
J u s t change the o b j e c t s and f u r n i t u r e .
Make s u r e i n i t i a l and f i n a l l o c a t i o n s o f o b j e c t s a r e d i f f e r e n t .
I n cl u d e a c t i o n s such a s tu rn on/ o f f , f i l l and cl e a n .
Make t a s k s multi - s tep , c o n s i s t i n g o f more than one o b j e c t and a c ti o n .
Your output sh ould onl y be :
```

```
JSON_OUTPUT: r e s u l t _ l i s t
where r e s u l t _ l i s t sh ould be a JSON l i s t with the t a s k s .
```
# <span id="page-48-0"></span>A.15 Evaluation Generation Prompts

Here we share the LLM prompts used for proposition generation, temporal constraint prediction, and argument constraint prediction. In each, the task to accomplish is described in the system prompt and between 6-13 few-shot examples follow.

```
Evaluation Generation: Propositions
```

```
S ou rce : system
You w i l l be gi v e n an i n s t r u c t i o n d e s c r i b i n g a h ou seh old t a s k and a d e s c r i p t i o n o f the i n i t i a l s t a t e
o f the house . You w i l l d e f i n e a l i s t o f python f u n c t i o n s t h a t must be s a t i s f i e d f o r the t a s k t o be
marked c omple te .
You can c a l l the f o l l o w i n g f u n c t i o n s :
- is_on_top ( ob ject_names : s t r , furniture_name : s t r ) # any o b j e c t i n ob ject_names i s on top o f a
f u r n i t u r e
- i s _ i n s i d e ( ob ject_names : s t r , furniture_name : s t r ) # any o b j e c t i n ob ject_names i s i n s i d e o f a
f u r n i t u r e
- is_in_room ( ob ject_names : s t r , room_name : s t r ) # any o b j e c t i n ob ject_names i s i n a room
- i s_ o n_ fl o o r ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s on the f l o o r
- is_next_to ( ob j ec t s_ a : s t r , ob jec t s_b : s t r ) # any o b j e c t i n o b jec t s_ a i s next t o any
o b j e c t i n ob jec t s_b
- i s_ cl e a n ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s cl e a n
- i s _ di r t y ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s d i r t y
- i s _ f i l l e d ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s f i l l e d , l i k e
with a l i q u i d
- is_empty ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s empty
- is_powered_on ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s powered on
- is_powered_o f f ( ob ject_names : s t r ) # any o b j e c t i n ob ject_names i s powered o f f
O b j e c t s i n ob ject_names can be e x p r e s s e d a s i t ap pe a r s i n the o b j e c t s l i s t ( " s tu f fed_ toy_1 " ) o r a s
an OR o f o b j e c t names ( " s tu f fed_ toy_1 o r s tu f fed_ toy_2 " ) .
A furniture_name can be e x p r e s s e d a s i t a pp e a r s i n the f u r n i t u r e l i s t ( e . g . " t a b l e " ) o r a s i t
a pp e a r s i n the f u r n i t u r e - room r e l a t i o n ( " t a b l e i n living_ room " ) .
Only u se the f u n c t i o n s l i s t e d above .
Each f u n c ti o n sh ould t e s t a s i n g l e o b j e c t s / f u r n i t u r e /room r e l a t i o n .
I f the i n s t r u c t i o n i s ambiguous such t h a t m ul ti pl e o b j e c t s c o ul d be used t o s a t i s f y a f u n c ti o n ( an
OR r e l a t i o n s h i p ) , then i n cl u d e a l l p o s s i b l e o b j e c t s .
D e fi n e a s many f u n c t i o n s a s n e c e s s a r y .
Write each f u n c ti o n on i t s own l i n e .
I t i s e s s e n t i a l t o wrap each f u n c ti o n i n d e l i m i t e r s [FN ] and [ /FN ] .
End your f u n c t i o n s with the token [END ] .
Let ' s s e e some examples . Suppose the i n i t i a l s t a t e i s :
O b j e c t s :
     * pants_1
     * s hi r t_ 1
     * s hi r t_ 2
     * s hi r t_ 3
F u r ni t u r e :
     * washer_dryer
     * t a b l e
Rooms :
     * laundryroom
Ob ject - Fu rni tu re -Room R el a ti o n s :
     * pants_1 on t a b l e i n laundryroom
     * s hi r t_ 1 on t a b l e i n laundryroom
     * s hi r t_ 2 on f l o o r i n laundryroom
Fu rni tu re -Room R el a ti o n s :
     * washer_dryer i n laundryroom
     * t a b l e i n laundryroom
I n s t r u c t i o n "Put the p an t s on the t a b l e " means
     [FN ] is_on_top ( " pants_1 " , " t a b l e i n laundryroom " ) [ /FN ]
I n s t r u c t i o n "Put the p an t s i n the washer " means
     [FN ] i s _ i n s i d e ( " pants_1 " , " washer_dryer i n laundryroom " ) [ /FN ]
```

```
I n s t r u c t i o n "Put a s h i r t i n the washer " means
      [FN ] i s _ i n s i d e ( " s hi r t_ 1 o r s hi r t_ 2 o r s hi r t_ 3 " , " washer_dryer i n laundryroom " ) [ /FN ]
I n s t r u c t i o n "Put a l l the s h i r t s i n the washer " means
      [FN ] i s _ i n s i d e ( " s hi r t_ 1 " , " washer_dryer i n laundryroom " ) [ /FN ]
      [FN ] i s _ i n s i d e ( " s hi r t_ 2 " , " washer_dryer i n laundryroom " ) [ /FN ]
      [FN ] i s _ i n s i d e ( " s hi r t_ 3 " , " washer_dryer i n laundryroom " ) [ /FN ]
I n s t r u c t i o n "Put a s h i r t on the f l o o r next t o the p an t s " means
      [FN ] i s_ o n_ fl o o r ( " s hi r t_ 1 o r s hi r t_ 2 o r s hi r t_ 3 " ) [ /FN ]
      [FN ] is_next_to ( " s hi r t_ 1 o r s hi r t_ 2 o r s hi r t_ 3 " , " pants_1 " ) [ /FN ]
I n s t r u c t i o n " Bring a s h i r t t o the laundryroom " means
      [FN ] is_in_room ( " s hi r t_ 1 o r s hi r t_ 2 o r s hi r t_ 3 " , "laundryroom " ) [ /FN ]
<s tep> S ou rce : u s e r
O b j e c t s :
      * c a s s e r ol e_ 0
      * squeeze r_0
F u r ni t u r e :
      * c a bi n e t
      * t a b l e
Rooms :
      * bathroom
      * h allw a y
      * ki t c h e n
      * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
      * c a s s e r ol e_ 0 on c a bi n e t i n ki t c h e n
      * squeeze r_0 on c a bi n e t i n ki t c h e n
Fu rni tu re -Room R el a ti o n s :
      * t a b l e i n bathroom
      * t a b l e i n h allw a y
      * t a b l e i n living_ room
I n s t r u c t i o n : "Get the c a s s e r o l e and s q u e e z e r from the ki t c h e n and put them on the t a b l e i n the
l i v i n g room . "
<s tep> S ou rce : a s s i s t a n t
[FN ] is_on_top ( " c a s s e r ol e_ 0 " , " t a b l e i n living_ room " ) [ /FN ]
[FN ] is_on_top ( " squeeze r_0 " , " t a b l e i n living_ room " ) [ /FN ]
[END ]
<s tep> S ou rce : u s e r
The i n i t i a l s t a t e i s :
O b j e c t s :
      * toy_truck_1
      * s tu f fed_ toy_1
      * s tu f fed_ toy_2
F u r ni t u r e :
      * t a b l e
      * bed
      * c a bi n e t
Rooms :
      * bedroom
      * other_room
      * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
      * toy_truck_1 on bed i n bedroom
      * s tu f fed_ toy_1 on f l o o r i n bedroom
      * s tu f fed_ toy_2 on f l o o r i n other_room
Fu rni tu re -Room R el a ti o n s :
      * t a b l e i n living_ room
I n s t r u c t i o n : " Bring a toy from the bedroom and put i t i n the l i v i n g room . "
<s tep> S ou rce : a s s i s t a n t
[FN ] is_in_room ( " toy_truck_1 o r s tu f fed_ toy_1 " , " living_ room " ) [ /FN ]
[END ]
<s tep> S ou rce : u s e r
```

```
The i n i t i a l s t a t e i s :
O b j e c t s :
     * cushion_1
     * fork_1
     * plate_1
F u r ni t u r e :
     * bed
     * couch
     * c o u n t e r
Rooms :
     * bedroom
     * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
     * cushion_1 on couch i n living_ room
     * fork_1 on couch i n living_ room
     * plate_1 on couch i n living_ room
Fu rni tu re -Room R el a ti o n s :
     * bed i n bedroom
     * couch i n living_ room
     * c o u n t e r i n ki t c h e n
I n s t r u c t i o n : " Bring the d i s h e s back t o the ki t c h e n c o u n t e r . A f t e r that , put the c u s hi o n on the bed . "
<s tep> S ou rce : a s s i s t a n t
[FN ] is_on_top ( " plate_1 " , " c o u n t e r i n ki t c h e n " ) [ /FN ]
[FN ] is_on_top ( " fork_1 " , " c o u n t e r i n ki t c h e n " ) [ /FN ]
[FN ] is_on_top ( " cushion_1 " , " bed i n bedroom " ) [ /FN ]
[END ]
<s tep> S ou rce : u s e r
The i n i t i a l s t a t e i s :
O b j e c t s :
     * candle_0
     * fork_0
     * toy_food_0
F u r ni t u r e :
     * c o u n t e r
     * t a b l e
Rooms :
     * bedroom
     * ki t c h e n
     * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
     * candle_0 on t a b l e i n living_ room
     * fork_0 on c o u n t e r i n ki t c h e n
     * toy_food_0 on c o u n t e r i n ki t c h e n
Fu rni tu re -Room R el a ti o n s :
     * t a b l e i n bedroom
     * c o u n t e r i n ki t c h e n
     * t a b l e i n living_ room
I n s t r u c t i o n : "Put the toy f o od on the t a b l e where the c a n dl e i s , and pl a c e the f o r k on the t a b l e
b e si d e the toy f o od . "
<s tep> S ou rce : a s s i s t a n t
[FN ] is_on_top ( " toy_food_0 " , " t a b l e i n living_ room " ) [ /FN ]
[FN ] is_on_top ( " fork_0 " , " t a b l e i n living_ room " ) [ /FN ]
[FN ] is_next_to ( " fork_0 " , " toy_food_0 " ) [ /FN ]
[END ]
<s tep> S ou rce : u s e r
O b j e c t s :
     * toy_airplane_0
     * toy_airplane_1
F u r ni t u r e :
     * t a b l e
Rooms :
     * h allw a y
     * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
     * toy_airplane_0 on t a b l e i n h allw a y
     * toy_airplane_1 on t a b l e i n h allw a y
Fu rni tu re -Room R el a ti o n s :
     * t a b l e i n bathroom
     * t a b l e i n h allw a y
     * t a b l e i n living_ room
```

```
I n s t r u c t i o n : " Bring a toy a i r p l a n e t o the l i v i n g room and put i t i n on the t a b l e . "
<s tep> S ou rce : a s s i s t a n t
[FN ] is_on_top ( " toy_airplane_0 o r toy_airplane_1 " , " t a b l e i n living_ room " ) [ /FN ]
[END ]
<s tep> S ou rce : u s e r
O b j e c t s :
      * k e t tl e_ 0 ( empty , d i r t y )
F u r ni t u r e :
      * t a b l e
Rooms :
      * living_ room
Ob ject - Fu rni tu re -Room R el a ti o n s :
      * k e t tl e_ 0 on t a b l e i n living_ room
      * toy_airplane_0 on t a b l e i n h allw a y
      * toy_airplane_1 on t a b l e i n h allw a y
Fu rni tu re -Room R el a ti o n s :
      * t a b l e i n living_ room
I n s t r u c t i o n : " Fi r s t , make s u r e the k e t t l e i s cl e a n . Then , f i l l the k e t t l e with water and tu rn i t
on . "
<s tep> S ou rce : a s s i s t a n t
[FN ] i s_ cl e a n ( " k e t tl e_ 0 " ) [ /FN ]
[FN ] i s _ f i l l e d ( " k e t tl e_ 0 " ) [ /FN ]
[FN ] is_powered_on ( " k e t tl e_ 0 " ) [ /FN ]
[FN ] is_on_top ( " toy_airplane_0 o r toy_airplane_1 " , " t a b l e i n living_ room " ) [ /FN ]
[END ]
{TEMPLATE_EXAMPLE}
<s tep> S ou rce : u s e r
The i n i t i a l s t a t e i s :
{INIT_STATE}
I n s t r u c t i o n : "{INSTRUCTION}"
<s tep> S ou rce : a s s i s t a n t
D e s ti n a ti o n : u s e r
[FN ]
                                            Evaluation Generation: Temporal
S ou rce : system
```
You w i l l be gi v e n an i n s t r u c t i o n d e s c r i b i n g a t a s k t o per form i n a house and a s e t o f p r o p o s i t i o n s t h a t d e f i n e whether the t a s k was done s u c c e s s f u l l y . The t a s k i n s t r u c t i o n may s ay t h a t c e r t a i n p r o p o s i t i o n s sh ould be completed b e f o r e o t h e r s ( " then " , " a f t e r " , " f i n a l l y " ) . Your j ob i s t o w ri t e python code t h a t g roup s the p r o p o s i t i o n s i n the o r d e r i n which they must be completed . The p r o p o s i t i o n s a r e w ell - d e fi n e d python f u n c t i o n s t h a t r e t u r n a b o ole an v al u e .

You w i l l be gi v e n a l i s t o f p r o p o s i t i o n s where inde x i c o r r e s p o n d s t o the i t h p r o p o s i t i o n . To s o l v e the t a sk , d e f i n e the v a r i a b l e p r o p o si ti o n_ o r d e r , which g r oup s p r o p o s i t i o n s t o g e t h e r t h a t can be completed i n any o r d e r . Each p r o p o s i t i o n group a p p e a ri n g i n p r o p o si ti o n_ o r d e r must be s a t i s f i e d b e f o r e the group t h a t comes a f t e r i t . For example ,

```
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 ]
]
means t h a t p r o p o s i t i o n s 0 and 1 can be completed i n any o r d e r . This example
p r o p o si ti o n_ o r d e r = [
      [ 0 ] ,
      [ 1 ]
]
means t h a t the p r o p o s i t i o n 0 must be completed b e f o r e p r o p o s i t i o n 1 . This example
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 ] ,
```

```
]
means t h a t p r o p o s i t i o n s 0 and 1 can be completed i n e i t h e r o rde r , but p r o p o s i t i o n 2 must be
completed a f t e r .
S t a r t by assuming t h a t p r o p o s i t i o n s can be completed i n any o r d e r . Order m a t te r s i f the i n s t r u c t i o n
i n c l u d e s time o r d e ri n g words such a s " then " , " f i n a l l y " , o r " a f t e r " . In t h i s c a se , p r o p o s i t i o n s
sh ould be i n m ul ti pl e g roup s .
Double check t h a t the inde x f o r each p r o p o s i t i o n i s i n cl u d e d i n p r o p o si ti o n_ o r d e r .
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Bring an a p pl e t o the ki t c h e n t a bl e , then b ri n g an o r an ge t o the ki t c h e n c o u n t e r . "
p r o p o s i t i o n s = [
      is_on_top ( [ " apple_1 " ] , [ " table_4 " ] ) ,
      is_on_top ( [ " orange_1 " ] , [ " counter_0 " ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 ] ,
      [ 1 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "Put the toy v e h i c l e and the water b o t t l e i n the l i v i n g room . Next , r e t u r n the di s h t o
the ki t c h e n . "
p r o p o s i t i o n s = [
      is_in_room ( [ " toy_truck_1 " ] , [ " living_room_1 " ] ) ,
      is_in_room ( [ " cup_0 " ] , [ " living_room_1 " ] ) ,
      is_in_room ( [ " bowl_2 " ] , [ " ki t c h e n " ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 ] ,
      [ 2 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "Put an a p pl e on the bench i n the entryway . Al s o move the broom t o the c l o s e t . "
p r o p o s i t i o n s = [
      is_on_top ( [ " apple_0 " , " apple_1 " ] , [ " bench_2 " ] ) ,
      i s _ i n s i d e ( [ " broom_0 " ] , [ " cl o s e t_ 0 " ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Bring me the toy t r u c k from the bedroom and put i t i n the l i v i n g room . Then put two
a p pl e s on the ki t c h e n t a b l e . "
p r o p o s i t i o n s = [
      is_in_room ( [ " toy_truck_1 " ] , [ " living_room_1 " ] ) ,
      is_on_top ( [ " apple_1 " ] , [ " table_1 " ] ) ,
      is_on_top ( [ " apple_2 " ] , [ " table_1 " ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 ] ,
      [ 1 , 2 ]
]
<s tep> S ou rce : u s e r
```
[ 2 ]

```
I n s t r u c t i o n : " Bring the d i s h e s back t o the ki t c h e n c o u n t e r . A f t e r that , put the c u s hi o n s on the
bed . "
p r o p o s i t i o n s = [
      is_on_top ( [ " plate_1 " ] , [ " counter_1 " , " counter_2 " , " counter_3 " ] ) ,
      is_on_top ( [ " fork_1 " ] , [ " counter_1 " , " counter_2 " , " counter_3 " ] ) ,
      is_on_top ( [ " spoon_0 " ] , [ " counter_1 " , " counter_2 " , " counter_3 " ] ) ,
      is_on_top ( [ " cushion_0 " ] , [ " bed_1 " , "bed_2 " ] ) ,
      is_on_top ( [ " cushion_1 " ] , [ " bed_1 " , "bed_2 " ] ) ,
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 ] ,
      [ 3 , 4 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Bring the d i s h e s back t o the ki t c h e n c o u n t e r . Put the c u s hi o n on the bed . Then , move
the c u s hi o n s t o the ki t c h e n . "
p r o p o s i t i o n s = [
      is_on_top ( [ " plate_1 " ] , [ " counter_1 " , " counter_2 " , " counter_3 " ] ) ,
      is_on_top ( [ " fork_1 " ] , [ " counter_1 " , " counter_2 " , " counter_3 " ] ) ,
      is_on_top ( [ " cushion_0 " ] , [ " bed_1 " , "bed_2 " ] ) ,
      is_on_top ( [ " cushion_1 " ] , [ " bed_1 " , "bed_2 " ] ) ,
      is_in_room ( [ " cushion_0 " ] , [ " bedroom_0 " ] ) ,
      is_in_room ( [ " cushion_1 " ] , [ " bedroom_0 " ] ) ,
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 , 3 ] ,
      [ 4 , 5 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "Move the c l o t h e s from the bedroom t o the washer . A f t e r that , Put the c u s hi o n on the
bed . Fi n all y , put the book i n the l i v i n g room . "
p r o p o s i t i o n s = [
      is_on_top ( [ " s hi r t_ 1 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " s hi r t_ 2 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " pants_1 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " cushion_1 " ] , [ " bed_1 " , "bed_2 " ] ) ,
      is_in_room ( [ " book_1 " ] , [ " living_room_1 " ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 ] ,
      [ 3 ] ,
      [ 4 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Fi r s t , move the spoon and k e t t l e from the ki t c h e n t o the l i v i n g room and pl a c e them
next t o each o t h e r . Then , pl a c e the toy f o od i n the ki t c h e n c a bi n e t . "
p r o p o s i t i o n s = [
      is_on_top ( [ ' spoon_0 ' ] , [ ' table_1 ' , ' table_2 ' , ' table_3 ' , ' table_4 ' , ' table_5 ' ] ) ,
      is_on_top ( [ ' ke t tle_ 0 ' ] , [ ' table_1 ' , ' table_2 ' , ' table_3 ' , ' table_4 ' , ' table_5 ' ] ) ,
      is_next_to ( [ ' spoon_0 ' ] , [ ' ke t tle_ 0 ' ] ) ,
      i s _ i n s i d e ( [ ' toy_food_0 ' ] , [ ' cabinet_0 ' ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 ] ,
      [ 3 ]
]
```

```
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Fi r s t , move the phone s t and from the bedroom t o the l i v i n g room and pl a c e i t on the
t a b l e next t o the lamp . Then , move the f i l e s o r t e r from the l i v i n g room t o the bedroom and pl a c e i t
on the t a b l e next t o the phone s t and . "
p r o p o s i t i o n s = [
      is_on_top ( [ ' phone_stand_0 ' ] , [ ' table_1 ' , ' table_2 ' , ' table_3 ' ] ) ,
      is_next_to ( [ ' phone_stand_0 ' ] , [ ' lamp_0 ' ] ) ,
      is_on_top ( [ ' fil e_ s o r t e r_ 0 ' ] , [ ' table_6 ' ] ) ,
      is_next_to ( [ ' fil e_ s o r t e r_ 0 ' ] , [ ' phone_stand_0 ' ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 ] ,
      [ 2 , 3 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Help me move the c a n dl e and hand t ow el t o the ki t c h e n c o u n t e r . Pl a c e them next t o
each o t h e r . Then , pl a c e
the s p a t ul a and c - clamp on the b e d si d e t a b l e next t o each o t h e r . "
p r o p o s i t i o n s = [
      is_on_top ( [ ' candle_0 ' ] , [ ' counter_0 ' ] ) ,
      is_on_top ( [ ' hand_towel_0 ' ] , [ ' counter_0 ' ] ) ,
      is_next_to ( [ ' candle_0 ' ] , [ ' hand_towel_0 ' ] ) ,
      is_on_top ( [ ' spatula_0 ' ] , [ ' table_6 ' ] ) ,
      is_on_top ( [ ' c - clamp_0 ' ] , [ ' table_6 ' ] ) ,
      is_next_to ( [ ' spatula_0 ' ] , [ ' c - clamp_0 ' ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 ] ,
      [ 3 , 4 , 5 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Fi r s t , move the dog bowl , then the placemat , and f i n a l l y a plu sh toy from the l i v i n g
room t o the bench i n the h allw a y . Pl a c e them next t o each o t h e r . "
p r o p o s i t i o n s = [
      is_on_top ( [ ' dog_bowl_0 ' ] , [ ' bench_0 ' ] ) ,
      is_on_top ( [ ' placemat_0 ' ] , [ ' bench_0 ' ] ) ,
      is_next_to ( [ ' placemat_0 ' ] , [ ' dog_bowl_0 ' ] ) ,
      is_on_top ( [ ' plush_toy_0 ' ] , [ ' bench_0 ' ] ) ,
      is_next_to ( [ ' plush_toy_0 ' ] , [ ' placemat_0 ' ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 ] ,
      [ 1 , 2 ] ,
      [ 3 , 4 ]
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "Move the t o a s t e r and the bread from the p an t ry t o the ki t c h e n and tu rn the t o a s t e r
on , then f i l l the k e t t l e . "
p r o p o s i t i o n s = [
      is_in_room ( [ ' t o a s te r_ 0 ' ] , [ ' kitchen_0 ' ] ) ,
      is_in_room ( [ ' bread_0 ' ] , [ ' kitchen_0 ' ] ) ,
      is_powered_on ( [ ' t o a s te r_ 0 ' ] ) ,
      i s _ f i l l e d ( [ ' ke t tle_ 0 ' ] )
]
<s tep> S ou rce : a s s i s t a n t
p r o p o si ti o n_ o r d e r = [
      [ 0 , 1 , 2 ] ,
      [ 3 ]
```
]

```
<s tep> S ou rce : u s e r
```
I n s t r u c t i o n : "{INSTRUCTION}"

{PROPOSITIONS}

<s tep> S ou rce : a s s i s t a n t D e s ti n a ti o n : u s e r

p r o p o si ti o n_ o r d e r = [

#### Evaluation Generation: Argument Constraints

```
S ou rce : system
```
You w i l l be gi v e n a t a s k t o per form i n a house , and a s e t o f p r o p o s i t i o n s t h a t d e f i n e whether the t a s k was done s u c c e s s f u l l y . The t a s k i s pe r fo rmed by a human and r ob o t . The t a s k i n s t r u c t i o n may imply c o n s t r a i n t s such t h a t c e r t a i n g r oup s o f p r o p o s i t i o n s sh ould be s a t i s f i e d by the same argument o r unique arguments . Your j ob i s t o w ri t e python code t h a t d e f i n e s t h e s e c o n s t r a i n t s . The p r o p o s i t i o n s a r e w ell - d e fi n e d python f u n c t i o n s t h a t r e t u r n a b o ole an v al u e . You w i l l be gi v e n a l i s t o f p r o p o s i t i o n s where inde x i c o r r e s p o n d s t o the i t h p r o p o s i t i o n . To s o l v e the t a sk , d e f i n e the v a r i a b l e ti e _ c o n s t r ai n t s , which i s a l i s t o f c o n s t r a i n t s which can be empty . The c o n s t r a i n t s you can u se a r e : SameArgConstraint ( p r o p o si ti o n _i n d i c e s : L i s t [ i n t ] , # i n d i c e s o f p r o p o s i t i o n s t h a t t h i s c o n s t r a i n t a p p l i e s t o arg_index : L i s t [ i n t ] , # i n d i c e s o f arguments t h a t sh ould be matched on ) Di f f e r e n t A r g C o n s t r ai n t ( p r o p o si ti o n _i n d i c e s : L i s t [ i n t ] , # i n d i c e s o f p r o p o s i t i o n s t h a t t h i s c o n s t r a i n t a p p l i e s t o arg_index : L i s t [ i n t ] , # i n d i c e s o f arguments t h a t sh ould be matched on ) Here a r e some examples : SameArgConstraint ( [ 0 , 1 ] , [ 0 , 0 ] ) # means t h a t p r o p o s i t i o n s a t inde x 0 and 1 must have a matching v al u e i n the f i r s t argument . Di f f e r e n t A r g C o n s t r ai n t ( [ 0 , 1 ] , [ 0 , 0 ] ) # means t h a t p r o p o s i t i o n s a t inde x 0 and 1 must have d i f f e r e n t v al u e s i n the f i r s t argument . SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] ) # means t h a t p r o p o s i t i o n s a t inde x 0 and 1 must have a matching v al u e i n the sec ond argument . I f no c o n s t r a i n t s apply t o the gi v e n i n s t r u c t i o n , j u s t w ri t e an empty l i s t . <s tep> S ou rce : u s e r I n s t r u c t i o n : " Bring an a p pl e and an o r an ge t o a t a b l e i n the ki t c h e n . " p r o p o s i t i o n s = [ is_on_top ( [ " apple_1 " ] , [ " table_3 " , " table_4 " ] ) , is_on_top ( [ " orange_1 " ] , [ " table_3 " , " table_4 " ] ) ] <s tep> S ou rce : a s s i s t a n t t i e _ c o n s t r a i n t s = [ SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] ) ] <s tep> S ou rce : u s e r I n s t r u c t i o n : "Put the toy v e h i c l e i n the l i v i n g room and r e t u r n the di s h t o the ki t c h e n . " p r o p o s i t i o n s = [ is_in_room ( [ " toy_truck_1 " ] , [ " living_room_1 " ] ) , is_in_room ( [ " bowl_2 " ] , [ " ki t c h e n " ] ) ] <s tep> S ou rce : a s s i s t a n t t i e _ c o n s t r a i n t s = [ ] <s tep> S ou rce : u s e r

```
I n s t r u c t i o n : " Pl a c e the book on the s h e l f i n the bedroom . Pl a c e the p i c t u r e frame next t o i t . "
p r o p o s i t i o n s = [
      is_on_top ( [ " book_1 " ] , [ " shel ve s_ 0 " , " shel ve s_ 1 " ] ) ,
      is_on_top ( [ " picture_ frame_0 " ] , [ " shel ve s_ 0 " , " shel ve s_ 1 " ] ) ,
      is_next_to ( [ " picture_ frame_0 " ] , [ " book_1 " ] )
]
<s tep> S ou rce : a s s i s t a n t
t i e _ c o n s t r a i n t s = [
      SameArgConstraint ( [ 0 , 1 ] , [ 1 , 1 ] )
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : " Pl a c e each c a n dl e on i t s own t a b l e i n the l i v i n g room . "
p r o p o s i t i o n s = [
      is_on_top ( [ " candle_0 " ] , [ " table_0 " , " table_2 " , " table_6 " ] ) ,
      is_on_top ( [ " candle_1 " ] , [ " table_0 " , " table_2 " , " table_6 " ] ) ,
      is_on_top ( [ " candle_2 " ] , [ " table_0 " , " table_2 " , " table_6 " ] )
]
<s tep> S ou rce : a s s i s t a n t
t i e _ c o n s t r a i n t s = [
      Di f f e r e n t A r g C o n s t r ai n t ( [ 0 , 1 , 2 ] , [ 1 , 1 ] )
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "Move the c l o t h e s from the bedroom t o the washer . A f t e r that , Put the c u s hi o n on the
bed . Fi n all y , put the book i n the l i v i n g room . "
p r o p o s i t i o n s = [
      is_on_top ( [ " s hi r t_ 1 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " s hi r t_ 2 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " pants_1 " ] , [ " washer_dryer_1 " ] ) ,
      is_on_top ( [ " cushion_1 " ] , [ " bed_1 " , "bed_2 " ] ) ,
      is_in_room ( [ " book_1 " ] , [ " living_room_1 " ] )
]
<s tep> S ou rce : a s s i s t a n t
t i e _ c o n s t r a i n t s = [
]
<s tep> S ou rce : u s e r
I n s t r u c t i o n : "{INSTRUCTION}"
{PROPOSITIONS}
<s tep> S ou rce : a s s i s t a n t
D e s ti n a ti o n : u s e r
t i e _ c o n s t r a i n t s = [
```
# <span id="page-56-0"></span>A.16 Prompts for planning baselines

Following prompts were used for various planning baselines.

#### Decentralized Single/Multi Agent | ReAct

<| s t a r t_he ade r_id |> system <| end_header_id|>You a r e an a gen t t h a t s o l v e s multi - a gen t pl a n ni n g problems . The t a s k a s si g n e d t o you w i l l be s i t u a t e d i n a house and w i l l g e n e r a l l y i n v ol v e n a vi g a ti n g t o o b j e c t s , pi c ki n g and pl a ci n g them on d i f f e r e n t r e c e p t a c l e s t o a c hi e v e re a r r an gemen t . You s t r i c t l y f o l l o w any fo rm a t s p e c i f i c a t i o n s and pay a t t e n t i o n t o the p r e vi o u s a c t i o n s t aken i n o r d e r t o a v oid r e p e a ti n g mi s t a k e s . Rooms do not need t o be e x pl o r e d more than once . There w i l l be an o the r a gen t t r yi n g t o s o l v e the same t a s k t h a t you a r e a t the same time . You may f i n d t h a t t h a t a gen t has pi c k e d up r e l e v a n t o b j e c t s o r i s i n the p r o c e s s o f c om pl e ti n g p a r t s o f the t a s k . I f t h a t i s the c a s e you may want t o move on t o a d i f f e r e n t p a r t o f the t a s k . Rooms do not need t o be e x pl o r e d more than once .

This means i f you have e x pl o r e d the l i v i n g room and have not found the o b j e c t , then you sh ould e x pl o r e the ki t c h e n , i f a r e l e v a n t o b j e c t i s s t i l l not found , you sh ould e x pl o r e the h allw a y e t c . . .

{ a g e n t_ r ol e_ d e s c ri p ti o n }

Many c a l l s t o the same a c ti o n i n a row a r e a si g n t h a t something has gone wrong and you sh ould t r y a d i f f e r e n t a c ti o n . <| eot_id |>{ op tion al_ r ag_ex ample s } <| s t a r t_he ade r_id |> u se r <| end_header_id|>Task : { t a s k _ d e s c ri p ti o n }

{ w o rl d_ d e s c ri p ti o n }

P o s s i b l e A c ti o n s : { t o o l _ d e s c r i p t i o n s }

What i s the next a c ti o n t o make p r o g r e s s towards c om pl e ti n g the t a s k ? Return your r e s p o n s e i n the f o l l o w i n g fo rm a t

Thought : <r e a s o ni n g f o r why you a r e t a ki n g the next a c ti o n > <next a c ti o n c a l l > A s si gned !

Here i s an example : Thought : Si n c e t h e r e a r e no o b j e c t s found I sh ould e x pl o r e a room I have not e x pl o r e d ye t Expl o re [<room name>] A s si gned ! <| eot_id|><| s t a r t_he ade r_id |> a s s i s t a n t <| end_header_id|>

#### Centralized | ReAct

<| s t a r t_he ade r_id |> system <| end_header_id|>You a r e a system t h a t s o l v e s multi - a gen t pl a n ni n g t a s k s . The t a s k a s si g n e d t o you w i l l be s i t u a t e d i n a house and w i l l g e n e r a l l y i n v ol v e n a vi g a ti n g t o o b j e c t s , pi c ki n g and pl a ci n g them on d i f f e r e n t r e c e p t a c l e s t o a c hi e v e re a r r an gemen t . There w i l l be a r ob o t a gen t ( Agent 0 ) and a human a gen t ( Agent 1 ) a v a i l a b l e f o r s o l v i n g the t a s k . Your g o al i s t o a s s i g n a c t i o n s f o r both o f t h e s e a g e n t s and s o l v e the t a s k . You s t r i c t l y f o l l o w any fo rm a t s p e c i f i c a t i o n s and pay a t t e n t i o n t o the p r e vi o u s a c t i o n s t aken i n o r d e r t o a v oid r e p e a ti n g mi s t a k e s .

You sh ould t r y and di vi d e the t a s k between the two a g e n t s f o r e f f i c i e n t t a s k c om pl e ti o n . Note t h a t the human a gen t can wash , cl e a n , f i l l , pour and tu rn on/ o f f d e v i c e s al o n g with d oin g o b j e c t re a r r an gemen t . However , the r o b o t can onl y do o b j e c t re a r r an gemen t i . e . , n a vi g a ti n g t o o b j e c t s , pi c ki n g and pl a ci n g them .

In the be ginnin g , you w i l l be p r o vided with the t a s k d e s c r i p t i o n and i n f o rm a ti o n about the rooms pl u s f u r n i t u r e i n each room f o r the house . Ob jec t i n f o rm a ti o n may o r may not be a v a i l a b l e . Rooms onl y need t o be e x pl o r e d i f t h e r e i s no i n f o rm a ti o n a v a i l a b l e about t a sk - r e l e v a n t o b j e c t s . Rooms do not need t o be e x pl o r e d f o r i d e n t i f y i n g which f u r n i t u r e t o t o go t o . Also , rooms do not need t o be e x pl o r e d more than once . This means i f one o f your a g e n t s has e x pl o r e d the l i v i n g room and have not found the o b j e c t , then you sh ould e x pl o r e the ki t c h e n , i f a r e l e v a n t o b j e c t i s s t i l l not found , you sh ould e x pl o r e the h allw a y e t c .

Many c a l l s t o the same a c ti o n i n a row a r e a si g n t h a t something has gone wrong and you sh ould t r y a d i f f e r e n t a c ti o n .

You sh ould t r y t o c omple te the t a s k i n the l e a s t amount o f a c t i o n s p o s s i b l e . This means i f t h e r e a r e two o b j e c t s t o be moved you sh ould have one a gen t n a vi g a t e t o each o b j e c t and then move them t o the t a r g e t l o c a t i o n a the same time .

I f a p r e vi o u s n a vi g a ti o n a c ti o n i s s t i l l i n p r o g r e s s f o r an agent , you sh ould r e a s s i g n t h a t a c ti o n t o the a gen t t i l l a s u c c e s s f u l e x e c u ti o n i s ob se r ved i n the agent ' s o b s e r v a ti o n s .

You sh ould c o n ti n u e t o e v al u a t e the t a s k p r o g r e s s and d e ci d e the a c t i o n s f o r both the a g e n t s . Once both a g e n t s a r e done , you can output "Done [ ] " t o i n d i c a t e t h a t the a g e n t s have f i n i s h e d the t a s k . Output your r e s p o n s e about t a s k c om pl e ti o n i n the f o l l o w i n g fo rm a t .

Thought : <r e a s o ni n g about why both a g e n t s have completed the e n t i r e t a s k s u c c e s s f u l l y > Done [ ]

DO NOT output "Done [ ] " u n l e s s you a r e c o n fi d e n t t h a t the whole t a s k i s s u c c e s s f u l l y completed . I f one o f the a gen t i s done with i t s p a r t o f the t a sk , w hil e the o t h e r a gen t i s s t i l l e x e c u ti n g , you can a s s i g n a "Wait [ ] " a c ti o n t o the a gen t who d oe sn t need t o a c t anymore . Pl e a s e re - s t a t e the t a s k d e s c r i p t i o n and v e r i f y i t ' s c om pl e ti o n b e f o r e o u t p u t ti n g "Done [ ] " . { eot_tag }{ user_tag }Task : { i n p u t }

{ w o rl d_ d e s c ri p ti o n }

P o s s i b l e a c t i o n s f o r each a gen t : { a g e n t _ d e s c ri p ti o n s }

What i s the next a c ti o n f o r each a gen t t o make p r o g r e s s towards c om pl e ti n g the t a s k ? Return your r e s p o n s e i n the f o l l o w i n g fo rm a t

```
Thought : <r e a s o ni n g f o r why you a r e t a ki n g the next a c ti o n >
Agent_0_Action : <next a c ti o n c a l l f o r r ob o t agent>
Agent_1_Action : <next a c ti o n c a l l f o r human agent>
A s si gned !
Here i s an example :
Thought : Si n c e t h e r e a r e m ul ti pl e t a sk - r e l e v a n t o b j e c t s t o be r e a r r a n g e d , I w i l l ask each a gen t t o
go t o one o f them
Agent_0_Action : N a vi g a te [< ob j name1>]
Agent_1_Action : N a vi g a te [< ob j name2>]
A s si gned !
<| eot_id|><| s t a r t_he ade r_id |> a s s i s t a n t <| end_header_id|>
```
The agent role description would be one of the following, depending on if the agent played the role of the human or robot.

#### Agent Role Descriptions

#### Human Description

You a r e pl a yi n g the r o l e o f the t a s k g i v e r . This means i f the i n s t r u c t i o n s a y s something l i k e "You sh ould move the o b j e c t and I w i l l wash i t " , then the o t h e r a gen t sh ould be moving the o b j e c t , and you sh ould washing the i t .

#### Robot Description

You a r e pl a yi n g the r o l e o f the t a s k r e c e i v e r . This means i f the i n s t r u c t i o n s a y s something l i k e "You sh ould move the o b j e c t and I w i l l wash i t " , then you sh ould move the o b j e c t and the o t h e r a gen t sh ould wash i t

Below are the tool descriptions. Perceptions tools are only included for the React-Tools agents.

#### Tool Descriptions

- Cl o se : Used f o r c l o s i n g an a r t i c u l a t e d e n t i t y . You must p r o vi d e the name o f the f u r n i t u r e you want t o c l o s e . Example ( Cl o se [ chest_o f_drawers_1 ] ) - Expl o re : Se a rch a s p e c i f i c room by v i s i t i n g v a ri o u s r e c e p t a c l e s o r f u r n i t u r e s i n t h a t room . The i n p u t t o the s k i l l i s the EXACT name o f the room t o be v i s i t e d . Use the room names p r o vided i n the house d e s c r i p t i o n . This t o o l e x h a u s ti vl y e x p l o r e s the s p e c i f i e d room . Example ( Expl o re [ kitchen_1 ] ) - FindAgentActionTool : Should be used t o f i n d c u r r e n t and p a s t s t a t e h i s t o r y o f o t h e r a gen t . - FindOb jectTool : Used t o f i n d the e x a c t name/names o f the o b j e c t / o b j e c t s o f i n t e r e s t . I f you want t o f i n d names o f o b j e c t s on s p e c i f i c r e c e p t a c l e s o r f u r n i t u r e s , p l e a s e i n cl u d e t h a t i n the query . Example ( FindOb jectTool [ t o y s on the f l o o r ] o r FindOb jectTool [ a p pl e s ] ) - Fi n dR e c e p t a cl eT o ol : Used t o know the e x a c t name o f a r e c e p t a c l e . A r e c e p t a c l e i s a f u r n i t u r e o r e n t i t y ( l i k e a c h ai r , t a bl e , bed e t c . ) where you can pl a c e an o b j e c t . Example ( Fi n dR e c e p t a cl eT o ol [ a ki t c h e n c o u n t e r ] ) - FindRoomTool : Used t o know the e x a c t name o f a room i n the house . A room i s a r e gi o n i n the house where f u r n i t u r e i s pl a c e d . Example ( FindRoomTool [ a room which might have something t o e a t ] ) - N a vi g a te : Used f o r n a vi g a ti n g t o an e n t i t y . You must p r o vi d e the name o f the e n t i t y you want t o n a vi g a t e t o . Example ( N a vi g a te [ counter_22 ] ) - Open : Used f o r openin g an a r t i c u l a t e d e n t i t y . You must p r o vi d e the name o f the f u r n i t u r e you want t o open . Example (Open [ chest_o f_drawers_1 ] ) - Pick : Used f o r pi c ki n g up an o b j e c t . You must p r o vi d e the name o f the o b j e c t t o be pi c k e d . The a gen t cannot h old more than one o b j e c t a t a time . Example ( Pick [ cup_1 ] ) - Pl a c e : Used f o r pl a ci n g an o b j e c t on a t a r g e t l o c a t i o n . You need t o p r o vi d e the name o f the o b j e c t t o be pl aced , the name o f the f u r n i t u r e where i t sh ould be pl aced , s p a t i a l r e l a t i o n ( " on" o r " wi t hi n " ) d e s c r i b i n g the r e l a t i o n between the o b j e c t and f u r n i t u r e . The o b j e c t t o be pl a c e d must al r e a d y be h el d by the a gen t ( i . e . pi c k e d p r e v i o u s l y ) . In a d di ti o n t o the se , you can r e q u e s t t o pl a c e the o b j e c t ne a r an o the r o b j e c t . For t h a t you can o p t i o n a l l y p r o vi d e a s p a t i a l c o n s t r a i n t s ( " next_to " ) and the name o f the r e f e r e n c e o b j e c t . To pl a c e next t o an o b j e c t , the r e f e r e n c e o b j e c t must al r e a d y be on the t a r g e t f u r n i t u r e . API tempate - Pl a c e [< ob ject_to_be_placed >, <s p a t i a l _ r e l a t i o n >, < f u r n i t u r e t o be pl a c e d on>, <s p a t i al _ c o n s t r ai n t >, <r e f e r e n c e_ o b j e c t > ]. s p a t i a l _ c o n s t r a i n t and r e f e r e n c e _ o b j e c t sh ould be s e t t o "None" when n e c e s s a r y . - Re a r range : Used f o r moving an o b j e c t from i t s c u r r e n t l o c a t i o n t o the t a r g e t l o c a t i o n . You need t o p r o vi d e the name o f the o b j e c t t o be moved , the name o f the f u r n i t u r e where i s sh ould be moved , s p a t i a l r e l a t i o n ( " on" o r " wi t hi n " ) d e s c r i b i n g the r e l a t i o n between the o b j e c t and f u r n i t u r e . This w i l l a u t om a ti c all y pi c k the s p e c i f i e d o b j e c t and move t o the t a r g e t f u r n i t u r e and attempt t o pl a c e i t . In a d di ti o n t o th e se , you can r e q u e s t t o pl a c e the o b j e c t ne a r an o the r o b j e c t . For t h a t you can o p t i o n a l l y p r o vi d e a s p a t i a l c o n s t r a i n t s ( " next_to " ) and the name o f the r e f e r e n c e o b j e c t . To pl a c e next t o an o b j e c t , the r e f e r e n c e o b j e c t must al r e a d y be on the t a r g e t f u r n i t u r e . API tempate Rea r r ange [<object_to_be_moved >, <s p a t i a l _ r e l a t i o n >, < f u r n i t u r e t o be pl a c e d on>, <s p a ti a l _ c o n s t r ai n t >, <r e f e r e n c e_ o b j e c t > ]. s p a t i a l _ c o n s t r a i n t and r e f e r e n c e _ o b j e c t sh ould be s e t t o "None" when n e c e s s a r y . - Wait : Used t o make a gen t s t a y i d l e f o r some time . Example (Wait [ ] ) - Done : Used t o i n d i c a t e t h a t the a gen t has f i n i s h e d the t a s k . Example ( Done [ ] )

For LLM agents simulating a human, additionally actions which modify the state of objects are available to be called. For centralized baselines two lists of available actions are provided in the agent description. One for the robot (without state-modifying actions) and one for the human (with state-modifying actions).

#### Human Only Tool Descriptions

- Clean : Used f o r c l e a n i n g an o b j e c t . You need t o p r o vi d e the name o f the o b j e c t t o cl e a n . - Cl o se : Used f o r c l o s i n g an a r t i c u l a t e d e n t i t y . You must p r o vi d e the name o f the f u r n i t u r e you want t o c l o s e . Example ( Cl o se [ chest_o f_drawers_1 ] ) - F i l l : Used f o r f i l l i n g an o b j e c t . You need t o p r o vi d e the name o f the o b j e c t t o f i l l . - Pour : Used f o r p ou rin g from one c o n t ai n e r t o an o the r . This s k i l l w i l l pour i n t o the s p e c i f i e d c o n t ai n e r from whicheve r c o n t ai n e r i s c u r r e n t l y h el d by the a gen t . - PowerO f f : Used f o r t u r ni n g o f f a powered o b j e c t . You need t o p r o vi d e the name o f the o b j e c t t o be tu rned o f f . - PowerOn : Used f o r t u r ni n g on a powered o b j e c t . You need t o p r o vi d e the name o f the o b j e c t t o be tu rned on .

The world description contains a text description all rooms and their contained furniture, along with currently observed objects. Below is an example for one scene:

#### World Description Example

F u r ni t u r e : bedroom_1 : floor_bedroom_1 , chair_41 , chair_42 , bed_49 , table_54 , chest_o f_drawers_72 , chest_o f_drawers_73 , chest_o f_drawers_75 , chest_o f_drawers_87 cl o s e t_ 1 : fl o o r_ cl o s e t_ 1 , wardrobe_91 living_room_1 : floor_living_room_1 , chair_13 , chair_14 , chair_15 , chair_16 , chair_17 , chair_18 , chair_19 , chair_20 , chair_21 , chair_22 , couch_26 , couch_28 , couch_29 , chair_30 , stool_31 , stool_32 , table_38 , table_39 , table_48 , table_50 , stand_52 , counter_78 t oil e t _ 1 : fl o o r _ t oil e t _ 1 , t oil e t _ 4 3 bedroom_2 : floor_bedroom_2 , bed_23 , chair_46 , chair_47 , table_53 , chest_o f_drawers_55 , chest_o f_drawers_58 , chest_o f_drawers_59 , chest_o f_drawers_60 , chest_o f_drawers_61 bedroom_3 : floor_bedroom_3 , bed_37 , chair_40 , chest_o f_drawers_74 , table_77 , chest_o f_drawers_79 , chest_o f_drawers_80 , chest_o f_drawers_89 , wardrobe_92 bedroom_4 : floor_bedroom_4 , chair_27 , bed_45 , table_51 , wardrobe_56 , wardrobe_57 , chest_o f_drawers_82 , chest_o f_drawers_83 , table_84 , table_86 kitchen_1 : fl o o r_ki tchen_ 1 , shelve s_24 , shelve s_25 , chair_33 , chair_34 , chair_35 , chair_36 , cabinet_62 , cabinet_63 , cabinet_64 , cabinet_65 , cabinet_66 , counter_67 , counter_68 , counter_69 , cabinet_70 , cabinet_71 , cabinet_76 , cabinet_81 , unknown_88 , f ri d g e_ 9 0 other_room_1 : floor_other_room_1 other_room_2 : floor_other_room_2 bathroom_1 : floor_bathroom_1 , t oil e t _ 4 4 porch_1 : floor_porch_1 entryway_1 : floor_entryway_1 , table_85 The f o l l o w i n g f u r n i t u r e s have a f a u c e t : c abinet_70 O b j e c t s : No o b j e c t s found ye t

For baselines utilizing a summary of the scene, after each action, we insert a text description of objects in the scene (as perceived by the agent so far).

#### Scene Summary Example

```
Thought : Si n c e t h e r e a r e no o b j e c t s found , I sh ould e x pl o r e a room I have not e x pl o r e d ye t . I w i l l
s t a r t with the l i v i n g room , a s i t i s one o f the rooms mentioned i n the t a sk , and i t might c o n t ai n
the v a se o r o t h e r r e l e v a n t o b j e c t s .
Expl o re [ living_room_1 ]
A s si gned ! <| eot_id|><| s t a r t_he ade r_id |> u se r <| end_header_id|>
R e s ul t : S u c c e s s f u l e x e c u ti o n !
O b j e c t s : t o y_ f rui t s_ 1 : chest_o f_drawers_49 i n bedroom_4
laptop_cover_2 : cabine t_59 i n laundryroom_1
cup_3 : cabinet_65 i n kitchen_1
vase_0 : c abinet_65 i n kitchen_1 <| eot_id|><| s t a r t_he ade r_id |> a s s i s t a n t <| end_header_id|>
Thought : Now t h a t we have found the l o c a t i o n s o f the o b j e c t s , I sh ould n a vi g a t e t o the ki t c h e n t o
pi c k up the vase , a s i t i s one o f the o b j e c t s mentioned i n the t a sk , and i t i s l o c a t e d i n the
ki t c h e n .
N a vi g a te [ cabinet_65 ]
A s si gned ! <| eot_id|><| s t a r t_he ade r_id |> u se r <| end_header_id|>
R e s ul t : S u c c e s s f u l e x e c u ti o n !
O b j e c t s : t o y_ f rui t s_ 1 : chest_o f_drawers_49 i n bedroom_4
laptop_cover_2 : cabine t_59 i n laundryroom_1
```
cup_3 : cabinet_65 i n kitchen_1 vase_0 : c abinet_65 i n kitchen_1 <| eot_id|>

For finetuning, we build the prompt by combining the task description, world description and the previous actions for both agents.

#### Finetuning Example

Task : I f i n i s h e d r e a di n g my book . Pl e a s e move the book t o the bedroom t a b l e . You can move the v a se t h e r e t o o ?

Current Environment : { w o rl d_ d e s c ri p ti o n }

P r e vi o u s a c t i o n s : Agent_Action : N a vi g a te [ table_25 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : N a vi g a te [ vase_0 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : Pick [ vase_0 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : N a vi g a te [ table_35 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Other_Agent_Action : Pick [ book_1 ] Agent_Action : Open [ table_35 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : N a vi g a te [ table_35 ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : Pl a c e [ vase_0 , on , table_35 , none , none ] Ac ti on R e s ul t : S u c c e s s f u l e x e c u ti o n ! Agent_Action : Wait [ ] Ac ti on R e s ul t : S u c c e s s f u l Execu ti on ! Other_Agent_Action : Open [ table_35 ] Other_Agent_Action : Pl a c e [ book_1 , on , table_35 , none , none ]

Next Agent_Action : <| re se r ved_ speci al_ t o ken_ 0 |>Done [ ] < end_act>

# Contributions

Matthew Chang: Implemented the state-modifying skills and built out the object state management system. Implemented constrained generation for planning baselines. Implemented ReAct based baselines and optimized their prompting and performance. Contributed to finetuning code and LLM inference infrastructure.

Gunjan Chhablani: Implemented PrediViz and helped analyze PARTNR tasks and evaluation functions.

Alexander Clegg: Designed and implemented foundation simulation features and APIs including: geometric relation utils, object states, kinematic relationship management, articulation logic, procedural sampling features, metadata handling, and some oracle skills. Contributed to design, implementation, and better engineering efforts for evaluation, HITL, simulated world graph, perception, and robot skills subsystems. Led the 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration to enable large scale experimentation on collaboration tasks.

Mikael Dallaire Cote: Led the development of a cloud platform capable of collecting collaboration data, at a large scale, by extending the Habitat 3.0 human-in-the-loop (HITL) system to (1) Run in the browser; (2) Support multiplayer sessions; (3) Evaluate LLM agents; (4) Support the full articulated HSSD dataset. Created a GUI application within that system that allows crowdsource workers to complete collaboration tasks. Deployed the HITL platform to the cloud and rolled out the data collection campaign. Created paper figures.

Ruta Desai: Defined benchmark characteristics and developed the simulation-in-the-loop LLM-based task instruction generation pipeline. Proposed and developed ReAct baselines and its analysis. Wrote experiments and analysis section in the paper.

Michal Hlaváč: Designed PrediViz, the visualization and annotation system for the PARTNR task evaluation. Wrote the section on PrediViz in the paper.

Vladimir Karashchuk: Contributed to 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration to enable large scale experimentation on collaboration tasks. Authored annotations such as regions, articulations, and marker sets as well as authoring 3D assets including render, collision, and receptacle meshes. Worked on quality assurance for annotations and assets originating from vendor workstreams.

Jacob Krantz: Designed and implemented the PARTNR evaluation system and large-scale generation of evaluation functions. Led the dataset annotation effort and maintained, analyzed, and balanced all dataset splits. Guided the development of PrediViz.

Roozbeh Mottaghi: Managed the team and coordinated different workstreams including benchmark generation, planning models, analysis, and human-in-the-loop infrastructure.

Priyam Parashar: Contributed to the world-graph representation and logic for integration into the planning stack, specifically owning the concept-graph pipeline to create non-privileged scene descriptions and nonprivileged graph updates. Built out the integration of LLM agents and planner into the HITL tool enabling single-learn evaluations.

Siddharth Patki: Designed and implemented centralized and decentralized task planning frameworks for baseline experiments in PARTNR. Proposed and developed tool based ReAct planner baselines using instruct-tuned Llama3 models, and analyzed planner's performance using various performance metrics and failure modes. Designed and supported implementation of hierarchical scene graphs to encode information about agent's environment which is consumed by all baseline planners. Paper writing and figures.

Ishita Prasad: PXFN alignment and approval for data collection/annotation for model training. Secured alignment with other orgs, XFN to enable open-sourcing/release of dataset and code for multi-agent collaboration benchmarks.

Xavi Puig: Contributed to the planning codebase, building infra to scale up planning baselines and LLM inference. Led agent finetuning experiments and oracle-based agents, and debugged simulation and oracle skill failures. Initial exploration of automated benchmark generation. Built tools for HITL data analysis and visualization. Paper writing and figures.

Akshara Rai: Implemented benchmark generation for guided scaling of rearrange, spatial, temporal and heterogenous tasks. HITL analysis, paper writing. Supported all workstreams of the project.

Ram Ramrakhya: Implemented and helped design LLM fine-tuning pipeline, auto regressive evaluation and metrics for fine-tuning LLM planners. Ran experiments to train oracle-planner traces based agents in full and partial observation settings. Implemented initial version of HITL data parsing to convert episodes into traces for LLM finetuning. Implemented and deployed HITL campaign/experiment management service end-to-end using PsiTurk to: (1) Automatically launch and manage large campaigns and metadata. (2) Matchmaking for multi-user tasks, single user task assignment and automatic task recycling. Integrated habitat-llm and initial version of task in HITL to support collaboration tasks.

Daniel Tran: Annotated PARTNR datasets and provided feedback leading to improved generation.

Joanne Truoung: Contributed to the dataset effort by refining scene compositions and ensuring validity of the scenes for HITL data collection.

John Turner: Contributed implementation of foundation simulation features and APIs including: region queries, articulations, metadata management, marker sets, and panoptic sensor. Contributed to 3D asset dataset preparation effort extending HSSD with articulations, annotations, and quality iteration by designing scenes, building editor interfaces, and implementing automated quality checks.

Eric Undersander: Contributed to HITL architecture. Implemented HITL networking capabilities for single-user evaluation. Contributed to the HITL cloud infrastructure and tooling.

Tsung-Yen Yang: Led the effort of integrating neural network skills and designing a RAG system for running planner evaluation on the benchmark. Performed planner and skill results analysis, and contributed to better engineering efforts for designing skills CIs. Improved the planner's success rate by identifying edge cases, and better script for finding bugs.