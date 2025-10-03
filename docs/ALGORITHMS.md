# Algorithms Documentation

This document provides comprehensive documentation for the reinforcement learning algorithms implemented in this project, including their theoretical foundations, implementation details, and usage patterns.

## Project Structure

- **Core algorithms**: `algorithms/dp.py`, `algorithms/mc.py`
- **Environment**: `envs/drone_delivery.py`


## Environment Model: DroneDeliveryEnv (`envs/drone_delivery.py`)

`DroneDeliveryEnv` is a finite, tabular MDP with discrete state and action spaces designed for evaluating dynamic programming (DP) and model-free methods.

- Actions (`Discrete(6)`): `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`, `STAY=4`, `CHARGE=5` (see top-level constants `A_UP` … `A_CHARGE`).
- States (`Discrete(nS)`): Encodes `(x, y, battery, has_package)` as a single integer.
  - `nS = width * height * (max_battery + 1) * 2`.
  - Encoding is handled by `_encode(x, y, b, p)` and `_decode(idx)`.
- Stochasticity: Wind slip on movement actions. With parameter `wind_slip = p`, the intended move has probability `(1 - 2p)`, and the left/right 90° rotations each have probability `p`. See `_slip_distribution(intended)`.
- Rewards and terminations (summarized):
  - Base step penalty: `-1.0` per step.
  - Delivery at `dropoff` when carrying package: `+50.0`, then `has_package -> 0`, and `done=True`.
  - Battery depletion (if battery reaches 0 and not already terminated): `-10.0` and `done=True`.
  - Collision (attempted movement into wall/out-of-bounds/obstacle): `-20.0` and position remains unchanged. Collision alone does not terminate.
- Model enumeration: `enumerate_transitions(s, a)` analytically returns a list of `(prob, next_state, reward, done)` for DP methods. It matches the transition dynamics used by `step()`.
- Observation/Action spaces (Gymnasium): `observation_space = Discrete(nS)`, `action_space = Discrete(6)`.

This exact, analytical model is what enables tabular Value Iteration and Policy Iteration in `algorithms/dp.py`.


## Dynamic Programming Algorithms (`algorithms/dp.py`)

All DP methods assume a tabular environment exposing `env.nS`, `env.nA`, and `env.enumerate_transitions(s, a)`.

### Value Iteration: 

`value_iteration(env, gamma=0.99, theta=1e-4, max_iterations=10_000, *, store_v_history=False)`

- Goal: Compute the optimal value function `V*` and derive a greedy policy `pi*(s) = argmax_a Q(s,a)`.
- Update rule per sweep:
  - For each state `s`, compute `q(a) = Σ_{s'} P(s'|s,a) [ r(s,a,s') + γ V[s'] ]` and set `V_new[s] = max_a q(a)`.
  - Track `delta = max_s |V_new[s] - V[s]|`.
- Stopping criterion: Stop when `delta < theta` or `max_iterations` is reached.
- Return values:
  - `V`: `np.ndarray [nS]` optimal value estimate.
  - `policy`: `np.ndarray [nS]` greedy policy w.r.t. final `V`.
  - `stats`: dict with keys:
    - `iterations`: number of sweeps performed.
    - `deltas`: list of per-sweep max deltas.
    - `Vs` (optional): per-sweep snapshots of `V` when `store_v_history=True`.


Pseudocode (high-level):

```
V <- zeros(nS)
repeat until convergence or max_iterations:
  delta <- 0
  for s in 0..nS-1:
    q_values <- [ sum_{s'} P(s'|s,a) * (r + gamma * V[s']) for a in 0..nA-1 ]
    V_new[s] <- max(q_values)
    delta <- max(delta, |V_new[s] - V[s]|)
  V <- V_new
```
After convergence, a greedy policy is derived by recomputing `q_values` and taking `argmax` per state.


### Policy Evaluation: 

`policy_evaluation(env, policy, gamma=0.99, theta=1e-4, max_iterations=10_000)`

- Goal: Compute `V^pi` for a fixed deterministic policy `policy[s]`.
- Update rule: For each state `s`, set
  `V[s] <- Σ_{s'} P(s'|s, policy[s]) * ( r + γ V[s'] )` iteratively until the max update is below `theta`.
- Returns: `V` as `np.ndarray [nS]`.
- Used by `policy_iteration` for iterative policy evaluation.


### Policy Iteration: 

`policy_iteration(env, gamma=0.99, theta=1e-4, max_iterations=10_000, init_policy=None)`

- Goal: Find an optimal policy by alternating evaluation and greedy improvement.
- Inputs:
  - `init_policy`: optional array `[nS]` of initial actions. If `None`, starts with all zeros.
- Loop:
  1. Evaluate current `policy` with `policy_evaluation` until threshold `theta`.
  2. Improve policy greedily: for each `s`, set `policy[s] <- argmax_a Σ_{s'} P(s'|s,a) (r + γ V[s'])`.
  3. If no action changes in the improvement step, the policy is stable (converged).
- Return values:
  - `V`: value function of the final policy.
  - `policy`: the improved (and typically optimal) policy.
  - `stats`: dict with counts:
    - `policy_eval_iters`: number of evaluation sweeps.
    - `policy_improve_iters`: number of improvement iterations.


## Monte Carlo Control (On-Policy, ε-soft) (`algorithms/mc.py`)

Function: `mc_control_epsilon_soft(env, episodes=10_000, gamma=0.99, epsilon=0.1, first_visit=True, seed=42, verbose=False, verbose_every=100, interrupt_check=None)`

- Goal: Learn action-value function `Q(s,a)` and its greedy policy by generating episodes and applying Monte Carlo returns.
- Assumptions: Tabular discrete `env.nS`, `env.nA`, and standard Gymnasium `reset()`/`step()` APIs. No model is required.
- Policy behavior: ε-greedy w.r.t. the current `Q` during data collection.
- Episode generation: Rollout using ε-greedy, store `(s_t, a_t, r_t)` until `done` or `truncated`.
- Return computation: For `t` from `T-1` to `0`, accumulate `G <- γ G + r_t`.
- First-visit handling: If `first_visit=True`, update `(s_t,a_t)` only on its first visit within the episode; otherwise, update on every visit.
- Update rule (incremental mean):
  - Maintain `returns_count[s,a]`.
  - `α = 1 / returns_count[s,a]`.
  - `Q[s,a] <- Q[s,a] + α * (G - Q[s,a])`.
- **Progress monitoring** (new):
  - `verbose=True`: Enable progress reporting with tqdm (if available) or periodic prints
  - `verbose_every`: Frequency of progress updates (default: 100 episodes)
  - Shows average return over last 100 episodes and current episode return
- **Interruption handling** (new):
  - `interrupt_check`: Optional callable that returns `True` to stop training gracefully
  - Allows Ctrl+C handling without losing partial results
- Return values:
  - `policy`: greedy w.r.t. learned `Q` (`np.argmax(Q, axis=1)`).
  - `Q`: `np.ndarray [nS, nA]`.
  - `returns_avg`: list of episode returns (sum of rewards) for monitoring and plotting.

Pseudocode (high-level):

```
Initialize Q[s,a] <- 0
for each episode:
  generate episode using epsilon-greedy(Q)
  G <- 0; visited <- {}
  for t in reversed(range(T)):
    (s,a,r) <- episode[t]
    G <- gamma * G + r
    if first_visit and (s,a) in visited: continue
    visited.add((s,a))
    returns_count[s,a] += 1
    alpha <- 1 / returns_count[s,a]
    Q[s,a] <- Q[s,a] + alpha * (G - Q[s,a])
policy <- argmax_a Q[s,a]
```

## Monte Carlo Control (Off-Policy, Importance Sampling) (`algorithms/mc.py`)

Function: `mc_control_off_policy_is(env, episodes=10_000, gamma=0.99, behavior="epsilon", behavior_epsilon=0.2, weighted=True, seed=42, verbose=False, verbose_every=100, interrupt_check=None, debug_behavior=False, debug_behavior_episodes=100)`

- Goal: Learn a greedy target policy `π(s) = argmax_a Q(s,a)` while generating data with a different behavior policy `b(a|s)`.
- Supported behavior policies:
  - `uniform`: choose actions uniformly at random.
  - `epsilon`: ε-greedy w.r.t. current `Q` with `epsilon=behavior_epsilon`.
- Update: Backward pass with importance sampling weights. Uses Weighted IS by default for better stability (maintains cumulative weights `C[s,a]` and updates `Q[s,a] += (W/C[s,a]) * (G - Q[s,a])`).
- Ordinary IS option: set `weighted=False` to use ordinary IS (higher variance).
- Early termination: if at time `t` the action is not greedy under the current target policy, the backward loop breaks (since `π(a_t|s_t)=0`).
- **Progress monitoring** (new): Same as on-policy MC (verbose, verbose_every, interrupt_check)
- **Debug utilities** (new):
  - `debug_behavior=True`: Print empirical behavior policy statistics
  - `debug_behavior_episodes`: Number of episodes to sample for behavior stats
- Returns:
  - `policy`: greedy w.r.t. learned `Q`.
  - `Q`: `np.ndarray [nS, nA]`.
  - `returns_avg`: list of episode returns.

Pseudocode (high-level):

```
Initialize Q[s,a] <- 0, C[s,a] <- 0
for each episode generated by behavior b:
  store (s_t, a_t, r_t, b_prob_t) for t = 0..T-1
  G <- 0; W <- 1
  for t from T-1 downto 0:
    G <- gamma * G + r_t
    C[s_t,a_t] <- C[s_t,a_t] + W
    Q[s_t,a_t] <- Q[s_t,a_t] + (W / C[s_t,a_t]) * (G - Q[s_t,a_t])
      break
    W <- W / b_prob_t
policy <- argmax_a Q[s,a]
```

**Notes:**

 - Off-policy enables learning from replay or previously logged data and decouples exploration from the target policy.
 - Ensure the behavior policy has support over greedy actions (e.g., ε > 0 or uniform) to avoid zero weights.
 - Weighted IS typically has lower variance than Ordinary IS (though it can be biased in finite samples). It only requires keeping a cumulative weight table C[s,a] (same shape as Q) and does not require storing full episode returns.
 - In practice, off-policy MC with importance sampling generally needs more episodes than on-policy MC to stabilize, especially in stochastic, long-horizon tasks like DroneDelivery. Consider running 5k–10k+ episodes and multiple seeds for reliable trends.


## Design Notes and Limitations

**Model-based vs Model-free:**

- **DP methods** (VI, PI) require access to the full transition model via `env.enumerate_transitions(s, a)`, which provides exact `P(s'|s,a)` and `r(s,a,s')` tuples
- **MC methods** only require sample trajectories via `env.step()`, making them applicable to black-box environments

**Stochasticity considerations:**
- The `wind_slip` parameter controls movement stochasticity
- With probability `2p`, the drone slips 90° left or right
- Higher stochasticity impacts:
  - Policy conservativeness (staying near charging stations)
  - Exploration requirements for MC methods
  - Convergence speed and stability

**State space complexity:**
- State space size: `nS = width × height × (max_battery + 1) × 2`
- Grows quickly with grid dimensions and battery capacity
- Tabular methods become impractical for large state spaces

## Implementation Notes

The algorithms implemented in this project follow the standard formulations from Sutton & Barto (2018) with the following implementation details:

### Value Iteration
- **Standard implementation**: Follows Algorithm 4.3 (Value Iteration) from Sutton & Barto exactly
- **Update strategy**: Uses two-array approach (`V` and `V_new`) for synchronous updates
- **No optimizations**: Pure tabular implementation without in-place updates or prioritized sweeping

### Policy Evaluation
- **Standard implementation**: Follows Algorithm 4.1 (Iterative Policy Evaluation) from Sutton & Barto
- **Update strategy**: In-place updates (`V[s] = v` immediately after computing)
- **Minor optimization**: In-place updates converge slightly faster than two-array approach

### Policy Iteration
- **Standard implementation**: Follows Algorithm 4.3 (Policy Iteration) from Sutton & Barto exactly
- **No optimizations**: Full policy evaluation to convergence at each iteration (not truncated/modified policy iteration)
- **Extension**: Supports custom initialization (`init_policy` parameter) not in the textbook algorithm

### Monte Carlo Control (On-Policy)
- **Standard implementation**: Follows Algorithm 5.4 (On-policy first-visit MC control for ε-soft policies) from Sutton & Barto
- **Update rule**: Incremental mean with `α = 1/N`, equivalent to sample averaging
- **No optimizations**: Standard first-visit or every-visit MC without eligibility traces or function approximation

### Monte Carlo Control (Off-Policy)
- **Standard implementation**: Follows Algorithm 5.7 (Off-policy MC control with weighted importance sampling) from Sutton & Barto
- **Weighted IS**: Default implementation uses incremental update with cumulative weights `C[s,a]`
- **Ordinary IS variant**: Available via `weighted=False`, uses diminishing step-size approximation
- **Debug extension**: Optional `debug_behavior` parameter to monitor behavior policy statistics
- **Note**: The ordinary IS implementation uses `α = 1/C[s,a]` as an online approximation rather than storing all returns

### Key Implementation Choices

**Consistency with textbook:**

- All algorithms use the exact same mathematical updates as Sutton & Barto
- Convergence criteria match the textbook (delta < theta)
- No algorithmic modifications or heuristics

**Practical differences:**

- **Error handling**: Added guards for tabular/model requirements (not in textbook pseudocode)
- **Statistics tracking**: Returns convergence metrics (`deltas`, iteration counts) for analysis
- **Deterministic policies**: All methods return deterministic policies (argmax), not stochastic
- **Episode handling**: MC methods handle both `done` and `truncated` flags (Gymnasium API)
- **Debug utilities**: Off-policy MC includes optional behavior policy monitoring


## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
  - Algorithm 4.1: Iterative Policy Evaluation
  - Algorithm 4.3: Policy Iteration (p. 80) and Value Iteration
  - Algorithm 5.4: On-policy first-visit MC control
  - Algorithm 5.7: Off-policy MC control with weighted importance sampling

