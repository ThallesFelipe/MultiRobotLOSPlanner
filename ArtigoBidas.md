# Multi-Robot Coordinated Motion Planner Ensuring Line-of-Sight Connectivity

**Authors:** Andre Cid, Arthur Vangasse, Hector Azpúrua, Luciano Pimenta, Gustavo Freitas  
**Institution:** UFMG – Belo Horizonte, MG, Brazil

---

## Abstract

Centralized multi-robot motion planner that maintains uninterrupted Line-of-Sight (LOS) connectivity via a visibility graph and custom path-finding algorithm. Two movement strategies: **deterministic** (open-loop) and **adaptive re-planning** (reactive). Validated in ROS 2 + Gazebo + Open-RMF. Adaptive method achieved gains of **12.21% less total distance**, **13.25% less lead robot distance**, and **22.73% fewer movements** vs. deterministic.

---

## Introduction

Robots in GPS-denied or infrastructure-limited environments (disaster zones, warehouses, mines) need self-organized communication. Strategy: use robots as **mobile relay nodes** forming an ad-hoc mesh network. Key challenge: maintaining LOS between robots in cluttered environments to minimize signal degradation.

This work introduces a planner that explicitly couples **task objectives with connectivity constraints**, using a **heterogeneous fleet**: one high-capability lead robot + cheaper relay-only support robots.

---

## Methodology

**Goal:** Deploy a connected LOS communication chain from a base station to a target, maintaining connectivity at every step, including during obstacle avoidance.

**Core policy — Ordered Progression:** Robots advance sequentially from base toward target; no robot skips ahead of the formation front.

**Trajectory validation:** Each candidate move is discretized into sample points. A move is only approved if LOS to at least one static chain member is confirmed at every sample point.

---

### A. Visibility Graph

Built from a 2D occupancy grid map `M`:

1. Obstacle boundaries extracted via **Canny edge detector** + morphological ops → configuration space `C_free`.
2. Corners detected → clustered with **DBSCAN** (threshold ε) → cluster centroids = vertices `V`.
3. Edges `E`: connect vertices `(vi, vj)` if segment `vi vj` lies within `C_free` (checked by discretization).

**Dynamic updates:** On obstacle detection at `p_obs`:

- Use Euclidean Distance Transform `D(p)` and its gradient `∇D(p)` to find the blocked corridor segment `S_block`.
- Remove edges intersecting `S_block` that point in the robot's direction of travel (dot product filter, threshold τ):

$$\vec{d}_{path} \cdot \vec{d}_{ij} > \tau$$

- Rebuild graph: `V' = V ∪ {p_start, p_goal, p_obs}`, re-evaluate visibility excluding removed edges.

---

### B. Path Planning

Find optimal path `P* = (p0, p1, ..., p_{k-1})` minimizing:

$$C(P) = \sum_{i=1}^{k-1} w(p_{i-1}, p_i) + (k-1) \cdot \lambda$$

- `w(·)` = Euclidean edge length
- `λ` = penalty per node (biases toward fewer relay robots)
- Solved with **Dijkstra** over the visibility graph.

A path with `k` nodes requires `k-1` relay agents. Lead robot `r1` reaches the goal; others act as relays.

---

### C. Deterministic Deployment

Open-loop pre-calculated movement sequence. All robots start co-located at `p0`.

Total movements required (triangular number):

$$M = \frac{(n-1)n}{2}$$

Fixed combinatorial sequence (e.g., r1, r2, r1, r3, r2, ...). Time complexity: **O(n²)**.

---

### D. Adaptive Re-planning (Algorithm 1)

Triggered when a path is blocked. Computes new path `P_new` on updated graph `G'`, then generates movement sequence heuristically:

**For each robot at each step:**

- If on `P_new` → advance to next vertex.
- If not on `P_new` → move to neighbor that maximizes: (i) progress toward target, (ii) minimizes travel distance, (iii) preserves LOS to existing chain.

Each move validated by **Algorithm 2**:

1. **Sequential formation check:** reject if robot skips unoccupied nodes.
2. **Lead robot priority:** reject if move blocks `r1`'s immediate advance.
3. **Connectivity check:** simulate robot at midpoint → build temp visibility graph → run BFS from base → reject if any robot becomes unreachable.
4. **Deadlock avoidance:** robot at `P_new[j]` cannot advance to `P_new[j+1]` if all forward vertices are occupied.

Complexity: high-degree polynomial (sensitive to fleet size and graph density). For large-scale problems, fallback: retract all robots to base along initial path, then re-deploy deterministically on new path.

---

## Implementation

- **Middleware:** ROS 2 Jazzy
- **Simulation:** Gazebo Harmonic, robot: Scout Mini (Agilex Robotics)
- **Fleet management:** Open-RMF (dispatches `go_to_place` and `cancel_task` commands)
- **Map processing:** OpenCV (Canny + corner detection), DBSCAN clustering
- **Graph updates:** Incremental — only edges intersecting the changed ROI are re-evaluated; full rebuild only if map metadata changes.

**Rerouting triggers:**

- Map update → ROI revalidation → re-optimize path.
- Local obstacle/interference → corridor pruning (no full rebuild) → edges removed per Section III-A.

**Centralized design rationale:** Since LOS is enforced at every step, the central node always maintains reliable uplink to all agents, enabling safe dissemination of routes and execution schedules as discrete advance permissions.
