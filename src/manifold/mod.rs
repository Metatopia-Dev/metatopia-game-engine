//! Manifold-based world representation for non-Euclidean spaces

use cgmath::{Vector3, Matrix4, Point3, Quaternion};
use std::collections::HashMap;
use std::sync::Arc;

pub mod chart;
pub mod portal;
pub mod geodesic;
pub mod metric;

pub use chart::{Chart, ChartId, LocalCoordinate};
pub use portal::{Portal, PortalId, PortalConnection};
pub use geodesic::{Geodesic, GeodesicPath};
pub use metric::{Metric, MetricTensor, GeometryType};

/// A manifold representing the entire non-Euclidean world
#[derive(Clone)]
pub struct Manifold {
    charts: HashMap<ChartId, Arc<Chart>>,
    portals: HashMap<PortalId, Portal>,
    connections: Vec<PortalConnection>,
    active_chart: ChartId,
}

impl Manifold {
    /// Create a new empty manifold
    pub fn new() -> Self {
        let mut charts = HashMap::new();
        let default_chart = Chart::new(ChartId(0), GeometryType::Euclidean);
        charts.insert(ChartId(0), Arc::new(default_chart));
        
        Self {
            charts,
            portals: HashMap::new(),
            connections: Vec::new(),
            active_chart: ChartId(0),
        }
    }
    
    /// Add a new chart (local coordinate patch) to the manifold
    pub fn add_chart(&mut self, geometry: GeometryType) -> ChartId {
        let id = ChartId(self.charts.len() as u32);
        let chart = Chart::new(id, geometry);
        self.charts.insert(id, Arc::new(chart));
        id
    }
    
    /// Create a portal connection between two charts
    pub fn create_portal(
        &mut self,
        from_chart: ChartId,
        to_chart: ChartId,
        from_position: Point3<f32>,
        to_position: Point3<f32>,
        transform: Matrix4<f32>,
    ) -> Result<PortalId, String> {
        let from = self.charts.get(&from_chart)
            .ok_or_else(|| format!("Chart {:?} not found", from_chart))?;
        let to = self.charts.get(&to_chart)
            .ok_or_else(|| format!("Chart {:?} not found", to_chart))?;
        
        let id = PortalId(self.portals.len() as u32);
        let portal = Portal::new(
            id,
            from_chart,
            to_chart,
            from_position,
            to_position,
            transform,
        );
        
        let connection = PortalConnection {
            portal_id: id,
            from_chart,
            to_chart,
        };
        
        self.portals.insert(id, portal);
        self.connections.push(connection);
        
        Ok(id)
    }
    
    /// Transform a point from one chart to another through portals
    pub fn transform_between_charts(
        &self,
        point: Point3<f32>,
        from_chart: ChartId,
        to_chart: ChartId,
    ) -> Option<Point3<f32>> {
        if from_chart == to_chart {
            return Some(point);
        }
        
        // Find portal path between charts (simplified - direct portal only)
        for connection in &self.connections {
            if connection.from_chart == from_chart && connection.to_chart == to_chart {
                if let Some(portal) = self.portals.get(&connection.portal_id) {
                    return Some(portal.transform_point(point));
                }
            }
        }
        
        None
    }
    
    /// Get the metric tensor at a point in a specific chart
    pub fn metric_at(&self, chart_id: ChartId, point: Point3<f32>) -> Option<MetricTensor> {
        self.charts.get(&chart_id)
            .map(|chart| chart.metric().tensor_at(point))
    }
    
    /// Compute geodesic path between two points
    pub fn compute_geodesic(
        &self,
        start: Point3<f32>,
        end: Point3<f32>,
        chart_id: ChartId,
        steps: usize,
    ) -> Option<GeodesicPath> {
        self.charts.get(&chart_id)
            .map(|chart| Geodesic::compute(start, end, chart.metric(), steps))
    }
    
    /// Parallel transport a vector along a path
    pub fn parallel_transport(
        &self,
        vector: Vector3<f32>,
        path: &GeodesicPath,
        chart_id: ChartId,
    ) -> Option<Vector3<f32>> {
        self.charts.get(&chart_id)
            .map(|chart| chart.parallel_transport(vector, path))
    }
    
    /// Get all portals from a specific chart
    pub fn portals_from_chart(&self, chart_id: ChartId) -> Vec<&Portal> {
        self.connections
            .iter()
            .filter(|c| c.from_chart == chart_id)
            .filter_map(|c| self.portals.get(&c.portal_id))
            .collect()
    }
    
    /// Check if a ray intersects any portal
    pub fn ray_portal_intersection(
        &self,
        origin: Point3<f32>,
        direction: Vector3<f32>,
        chart_id: ChartId,
    ) -> Option<(PortalId, Point3<f32>, ChartId)> {
        for portal in self.portals_from_chart(chart_id) {
            if let Some(intersection) = portal.ray_intersection(origin, direction) {
                return Some((portal.id(), intersection, portal.target_chart()));
            }
        }
        None
    }
    
    /// Get chart by ID
    pub fn chart(&self, id: ChartId) -> Option<&Arc<Chart>> {
        self.charts.get(&id)
    }
    
    /// Get active chart
    pub fn active_chart(&self) -> &Arc<Chart> {
        self.charts.get(&self.active_chart).unwrap()
    }
    
    /// Set active chart
    pub fn set_active_chart(&mut self, id: ChartId) {
        if self.charts.contains_key(&id) {
            self.active_chart = id;
        }
    }
    
    /// Get all charts
    pub fn charts(&self) -> &HashMap<ChartId, Arc<Chart>> {
        &self.charts
    }
}

/// Position in the manifold (chart + local coordinates)
#[derive(Debug, Clone, Copy)]
pub struct ManifoldPosition {
    pub chart_id: ChartId,
    pub local: LocalCoordinate,
}

impl ManifoldPosition {
    pub fn new(chart_id: ChartId, position: Point3<f32>) -> Self {
        Self {
            chart_id,
            local: LocalCoordinate(position),
        }
    }
    
    /// Convert to world position (for rendering)
    pub fn to_world(&self, manifold: &Manifold) -> Option<Point3<f32>> {
        manifold.chart(self.chart_id)
            .map(|chart| chart.to_world(self.local))
    }
}

/// Orientation in the manifold with parallel transport
#[derive(Debug, Clone, Copy)]
pub struct ManifoldOrientation {
    pub quaternion: Quaternion<f32>,
    pub tangent_space: Matrix4<f32>,
}

impl ManifoldOrientation {
    pub fn new(quaternion: Quaternion<f32>) -> Self {
        Self {
            quaternion,
            tangent_space: Matrix4::from(quaternion),
        }
    }
    
    /// Update orientation with parallel transport along a path
    pub fn transport_along(&mut self, path: &GeodesicPath, manifold: &Manifold, chart_id: ChartId) {
        if let Some(chart) = manifold.chart(chart_id) {
            // Apply parallel transport to maintain orientation consistency
            let transport_matrix = chart.compute_transport_matrix(path);
            self.tangent_space = transport_matrix * self.tangent_space;
            // Update quaternion from the transported matrix (convert to Matrix3 first)
            use cgmath::Matrix3;
            let mat3 = Matrix3::new(
                self.tangent_space.x.x, self.tangent_space.x.y, self.tangent_space.x.z,
                self.tangent_space.y.x, self.tangent_space.y.y, self.tangent_space.y.z,
                self.tangent_space.z.x, self.tangent_space.z.y, self.tangent_space.z.z,
            );
            self.quaternion = Quaternion::from(mat3);
        }
    }
}