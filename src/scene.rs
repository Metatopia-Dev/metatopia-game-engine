//! Hierarchical 3D Scene Graph and Transform Tree
//!
//! Provides parent-child transform inheritance (position, rotation, scale),
//! local-to-world matrix propagation with dirty-flag caching, and node traversal.

use cgmath::{Point3, Vector3, Quaternion, Matrix4, SquareMatrix, Transform, One, Zero};

/// Unique identifier for a scene node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A node within the hierarchical scene graph
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub local_pos: Vector3<f32>,
    pub local_rot: Quaternion<f32>,
    pub local_scale: Vector3<f32>,
    pub world_matrix: Matrix4<f32>,
    pub visible: bool,
    dirty: bool,
}

impl SceneNode {
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            parent: None,
            children: Vec::new(),
            local_pos: Vector3::zero(),
            local_rot: Quaternion::one(),
            local_scale: Vector3::new(1.0, 1.0, 1.0),
            world_matrix: Matrix4::identity(),
            visible: true,
            dirty: true,
        }
    }

    /// Compute the local 4x4 affine transform matrix for this node
    pub fn local_matrix(&self) -> Matrix4<f32> {
        let t = Matrix4::from_translation(self.local_pos);
        let r: Matrix4<f32> = self.local_rot.into();
        let s = Matrix4::from_nonuniform_scale(self.local_scale.x, self.local_scale.y, self.local_scale.z);
        t * r * s
    }
}

/// Hierarchical Scene Graph
pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    root: NodeId,
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneGraph {
    /// Create a new Scene Graph with a root node
    pub fn new() -> Self {
        let root = SceneNode::new(NodeId(0), "Root");
        Self {
            nodes: vec![root],
            root: NodeId(0),
        }
    }

    /// Get root node ID
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Create and attach a new node under a parent
    pub fn create_node(&mut self, name: impl Into<String>, parent: Option<NodeId>) -> NodeId {
        let id = NodeId(self.nodes.len());
        let mut node = SceneNode::new(id, name);
        let parent_id = parent.unwrap_or(self.root);

        node.parent = Some(parent_id);
        self.nodes.push(node);
        self.nodes[parent_id.0].children.push(id);
        id
    }

    /// Attach a child node to a new parent
    pub fn set_parent(&mut self, child: NodeId, new_parent: NodeId) {
        if child.0 >= self.nodes.len() || new_parent.0 >= self.nodes.len() || child == new_parent {
            return;
        }

        // Remove from old parent
        if let Some(old_p) = self.nodes[child.0].parent {
            self.nodes[old_p.0].children.retain(|&c| c != child);
        }

        self.nodes[child.0].parent = Some(new_parent);
        self.nodes[new_parent.0].children.push(child);
        self.mark_dirty(child);
    }

    /// Mark a node and all its descendants as dirty (requiring world matrix recalculation)
    pub fn mark_dirty(&mut self, node_id: NodeId) {
        if node_id.0 >= self.nodes.len() { return; }
        self.nodes[node_id.0].dirty = true;

        let children = self.nodes[node_id.0].children.clone();
        for child in children {
            self.mark_dirty(child);
        }
    }

    /// Set local position of a node
    pub fn set_position(&mut self, node_id: NodeId, pos: Vector3<f32>) {
        if let Some(node) = self.nodes.get_mut(node_id.0) {
            node.local_pos = pos;
            self.mark_dirty(node_id);
        }
    }

    /// Set local rotation of a node
    pub fn set_rotation(&mut self, node_id: NodeId, rot: Quaternion<f32>) {
        if let Some(node) = self.nodes.get_mut(node_id.0) {
            node.local_rot = rot;
            self.mark_dirty(node_id);
        }
    }

    /// Set local scale of a node
    pub fn set_scale(&mut self, node_id: NodeId, scale: Vector3<f32>) {
        if let Some(node) = self.nodes.get_mut(node_id.0) {
            node.local_scale = scale;
            self.mark_dirty(node_id);
        }
    }

    /// Get immutable reference to a node
    pub fn get_node(&self, node_id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(node_id.0)
    }

    /// Get mutable reference to a node
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(node_id.0)
    }

    /// Update world matrices for the entire hierarchy
    pub fn update(&mut self) {
        let root_mat = Matrix4::identity();
        self.update_recursive(self.root, root_mat, false);
    }

    fn update_recursive(&mut self, node_id: NodeId, parent_world: Matrix4<f32>, parent_dirty: bool) {
        let is_dirty = self.nodes[node_id.0].dirty || parent_dirty;
        if is_dirty {
            let local = self.nodes[node_id.0].local_matrix();
            self.nodes[node_id.0].world_matrix = parent_world * local;
            self.nodes[node_id.0].dirty = false;
        }

        let current_world = self.nodes[node_id.0].world_matrix;
        let children = self.nodes[node_id.0].children.clone();

        for child in children {
            self.update_recursive(child, current_world, is_dirty);
        }
    }

    /// Compute world position of a node
    pub fn world_position(&self, node_id: NodeId) -> Point3<f32> {
        if let Some(node) = self.nodes.get(node_id.0) {
            node.world_matrix.transform_point(Point3::new(0.0, 0.0, 0.0))
        } else {
            Point3::new(0.0, 0.0, 0.0)
        }
    }

    /// Get total number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ─── Unit Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_graph_hierarchy() {
        let mut graph = SceneGraph::new();

        // Car body at (10, 0, 0)
        let car = graph.create_node("Car", None);
        graph.set_position(car, Vector3::new(10.0, 0.0, 0.0));

        // Turret on top of car at local (0, 2, 0)
        let turret = graph.create_node("Turret", Some(car));
        graph.set_position(turret, Vector3::new(0.0, 2.0, 0.0));

        // Barrel on turret at local (0, 0, 3)
        let barrel = graph.create_node("Barrel", Some(turret));
        graph.set_position(barrel, Vector3::new(0.0, 0.0, 3.0));

        graph.update();

        let car_pos = graph.world_position(car);
        let turret_pos = graph.world_position(turret);
        let barrel_pos = graph.world_position(barrel);

        assert!((car_pos.x - 10.0).abs() < 1e-4);
        assert!((turret_pos.x - 10.0).abs() < 1e-4 && (turret_pos.y - 2.0).abs() < 1e-4);
        assert!((barrel_pos.x - 10.0).abs() < 1e-4 && (barrel_pos.y - 2.0).abs() < 1e-4 && (barrel_pos.z - 3.0).abs() < 1e-4);
    }
}
