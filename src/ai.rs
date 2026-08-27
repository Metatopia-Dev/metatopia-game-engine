//! Modular AI Behavior Trees and Portal-Aware 3D A* Pathfinding
//!
//! Provides a flexible Behavior Tree engine (Selectors, Sequences, Inverters, Actions, Conditions, Blackboard)
//! and a 3D waypoint navigation graph with non-Euclidean portal-aware A* pathfinding.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use cgmath::{Point3, InnerSpace};
use crate::manifold::ChartId;

// ─── Behavior Tree System ──────────────────────────────────────────────────

/// Execution status returned by behavior tree nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Success,
    Failure,
    Running,
}

/// Generic Blackboard value for AI state sharing
#[derive(Debug, Clone, PartialEq)]
pub enum BlackboardValue {
    Bool(bool),
    Int(i64),
    Float(f32),
    Text(String),
    Vec3([f32; 3]),
}

/// Shared data storage for AI behavior trees
#[derive(Debug, Clone, Default)]
pub struct Blackboard {
    data: HashMap<String, BlackboardValue>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn set(&mut self, key: impl Into<String>, val: BlackboardValue) {
        self.data.insert(key.into(), val);
    }

    pub fn get(&self, key: &str) -> Option<&BlackboardValue> {
        self.data.get(key)
    }

    pub fn get_bool(&self, key: &str) -> bool {
        match self.data.get(key) {
            Some(BlackboardValue::Bool(b)) => *b,
            _ => false,
        }
    }

    pub fn get_float(&self, key: &str) -> Option<f32> {
        match self.data.get(key) {
            Some(BlackboardValue::Float(f)) => Some(*f),
            _ => None,
        }
    }
}

/// Behavior Tree Node trait
pub trait BehaviorNode: Send + Sync {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus;
}

/// Sequence node: executes children in order until one fails or runs
pub struct SequenceNode {
    children: Vec<Box<dyn BehaviorNode>>,
    current_idx: usize,
}

impl SequenceNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self { children, current_idx: 0 }
    }
}

impl BehaviorNode for SequenceNode {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus {
        while self.current_idx < self.children.len() {
            let status = self.children[self.current_idx].tick(bb);
            match status {
                NodeStatus::Success => {
                    self.current_idx += 1;
                }
                NodeStatus::Running => {
                    return NodeStatus::Running;
                }
                NodeStatus::Failure => {
                    self.current_idx = 0;
                    return NodeStatus::Failure;
                }
            }
        }
        self.current_idx = 0;
        NodeStatus::Success
    }
}

/// Selector node: executes children until one succeeds or runs
pub struct SelectorNode {
    children: Vec<Box<dyn BehaviorNode>>,
    current_idx: usize,
}

impl SelectorNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>) -> Self {
        Self { children, current_idx: 0 }
    }
}

impl BehaviorNode for SelectorNode {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus {
        while self.current_idx < self.children.len() {
            let status = self.children[self.current_idx].tick(bb);
            match status {
                NodeStatus::Success => {
                    self.current_idx = 0;
                    return NodeStatus::Success;
                }
                NodeStatus::Running => {
                    return NodeStatus::Running;
                }
                NodeStatus::Failure => {
                    self.current_idx += 1;
                }
            }
        }
        self.current_idx = 0;
        NodeStatus::Failure
    }
}

/// Inverter node: inverts Success to Failure and vice versa
pub struct InverterNode {
    child: Box<dyn BehaviorNode>,
}

impl InverterNode {
    pub fn new(child: Box<dyn BehaviorNode>) -> Self {
        Self { child }
    }
}

impl BehaviorNode for InverterNode {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus {
        match self.child.tick(bb) {
            NodeStatus::Success => NodeStatus::Failure,
            NodeStatus::Failure => NodeStatus::Success,
            NodeStatus::Running => NodeStatus::Running,
        }
    }
}

/// Action node wrapping a closure
pub struct ActionNode<F: FnMut(&mut Blackboard) -> NodeStatus + Send + Sync> {
    action: F,
}

impl<F: FnMut(&mut Blackboard) -> NodeStatus + Send + Sync> ActionNode<F> {
    pub fn new(action: F) -> Self {
        Self { action }
    }
}

impl<F: FnMut(&mut Blackboard) -> NodeStatus + Send + Sync> BehaviorNode for ActionNode<F> {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus {
        (self.action)(bb)
    }
}

/// Condition node checking a predicate
pub struct ConditionNode<F: Fn(&Blackboard) -> bool + Send + Sync> {
    predicate: F,
}

impl<F: Fn(&Blackboard) -> bool + Send + Sync> ConditionNode<F> {
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F: Fn(&Blackboard) -> bool + Send + Sync> BehaviorNode for ConditionNode<F> {
    fn tick(&mut self, bb: &mut Blackboard) -> NodeStatus {
        if (self.predicate)(bb) {
            NodeStatus::Success
        } else {
            NodeStatus::Failure
        }
    }
}

// ─── 3D & Portal A* Navigation Graph ──────────────────────────────────────

/// A node in the 3D Navigation Graph
#[derive(Debug, Clone)]
pub struct NavNode {
    pub id: usize,
    pub pos: Point3<f32>,
    pub chart: ChartId,
    pub neighbors: Vec<(usize, f32)>, // (neighbor_node_id, cost)
}

/// 3D Navigation Graph with non-Euclidean portal link support
#[derive(Debug, Clone, Default)]
pub struct NavGraph {
    pub nodes: Vec<NavNode>,
}

#[derive(Copy, Clone, PartialEq)]
struct State {
    f_score: f32,
    g_score: f32,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl NavGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a waypoint node in a specific chart
    pub fn add_node(&mut self, pos: Point3<f32>, chart: ChartId) -> usize {
        let id = self.nodes.len();
        self.nodes.push(NavNode {
            id,
            pos,
            chart,
            neighbors: Vec::new(),
        });
        id
    }

    /// Connect two nodes bidirectionally with spatial distance cost
    pub fn connect(&mut self, a: usize, b: usize) {
        if a >= self.nodes.len() || b >= self.nodes.len() { return; }
        let cost = (self.nodes[a].pos - self.nodes[b].pos).magnitude();
        self.nodes[a].neighbors.push((b, cost));
        self.nodes[b].neighbors.push((a, cost));
    }

    /// Connect two nodes through a portal (even if in different charts)
    pub fn connect_portal(&mut self, node_in_chart1: usize, node_in_chart2: usize, traversal_cost: f32) {
        if node_in_chart1 >= self.nodes.len() || node_in_chart2 >= self.nodes.len() { return; }
        self.nodes[node_in_chart1].neighbors.push((node_in_chart2, traversal_cost));
        self.nodes[node_in_chart2].neighbors.push((node_in_chart1, traversal_cost));
    }

    /// Find nearest node to a 3D position within the specified chart
    pub fn find_nearest(&self, pos: Point3<f32>, chart: ChartId) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for node in &self.nodes {
            if node.chart == chart {
                let dist = (node.pos - pos).magnitude();
                if best.as_ref().map_or(true, |(_, d)| dist < *d) {
                    best = Some((node.id, dist));
                }
            }
        }
        best.map(|(id, _)| id)
    }

    /// Find shortest path using A* search across 3D nodes and portals
    pub fn find_path(&self, start: usize, goal: usize) -> Option<Vec<usize>> {
        if start >= self.nodes.len() || goal >= self.nodes.len() { return None; }
        if start == goal { return Some(vec![start]); }

        let mut dist: Vec<f32> = vec![f32::INFINITY; self.nodes.len()];
        let mut parent: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut heap = BinaryHeap::new();

        dist[start] = 0.0;
        heap.push(State { f_score: 0.0, g_score: 0.0, node: start });

        let goal_pos = self.nodes[goal].pos;

        while let Some(State { g_score, node, .. }) = heap.pop() {
            if node == goal {
                // Reconstruct path
                let mut path = Vec::new();
                let mut curr = Some(goal);
                while let Some(n) = curr {
                    path.push(n);
                    curr = parent[n];
                }
                path.reverse();
                return Some(path);
            }

            if g_score > dist[node] + 0.0001 {
                continue;
            }

            for &(neighbor, edge_cost) in &self.nodes[node].neighbors {
                let next_cost = dist[node] + edge_cost;
                if next_cost < dist[neighbor] - 0.0001 {
                    dist[neighbor] = next_cost;
                    parent[neighbor] = Some(node);
                    // Heuristic: Euclidean distance if same chart, 0 if different chart
                    let h = if self.nodes[neighbor].chart == self.nodes[goal].chart {
                        (self.nodes[neighbor].pos - goal_pos).magnitude()
                    } else {
                        0.0
                    };
                    heap.push(State { f_score: next_cost + h, g_score: next_cost, node: neighbor });
                }
            }
        }

        None
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_tree_sequence() {
        let mut bb = Blackboard::new();
        bb.set("energy", BlackboardValue::Float(100.0));

        let mut seq = SequenceNode::new(vec![
            Box::new(ConditionNode::new(|bb| bb.get_float("energy").unwrap_or(0.0) > 50.0)),
            Box::new(ActionNode::new(|bb| {
                bb.set("attacked", BlackboardValue::Bool(true));
                NodeStatus::Success
            })),
        ]);

        let status = seq.tick(&mut bb);
        assert_eq!(status, NodeStatus::Success);
        assert!(bb.get_bool("attacked"));
    }

    #[test]
    fn test_portal_astar_pathfinding() {
        let mut graph = NavGraph::new();

        // Chart 0 nodes: 0 -> 1
        let n0 = graph.add_node(Point3::new(0.0, 0.0, 0.0), ChartId(0));
        let n1 = graph.add_node(Point3::new(5.0, 0.0, 0.0), ChartId(0));
        graph.connect(n0, n1);

        // Chart 1 nodes: 2 -> 3
        let n2 = graph.add_node(Point3::new(0.0, 0.0, 0.0), ChartId(1));
        let n3 = graph.add_node(Point3::new(10.0, 0.0, 0.0), ChartId(1));
        graph.connect(n2, n3);

        // Portal link: n1 (chart 0) <--> n2 (chart 1)
        graph.connect_portal(n1, n2, 1.0);

        let path = graph.find_path(n0, n3);
        assert_eq!(path, Some(vec![0, 1, 2, 3]));
    }
}
