use super::*;
use crate::setters::{PointAlreadyExists, PointNotFound};
use std::convert::Infallible;

pub trait ApplyOperation {
    type Ok;
    type Err;
    type Undo;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err>;

    fn undo(self, dijkstra_map: &DijkstraMap) -> Self::Undo;
}

macro_rules! operations {
    ($(
        $(#[doc = $doc:literal])*
        fn $fn_name:ident -> pub struct $name:ident {
            $(
                pub $field:ident : $field_ty:ty
            ),* $(,)?
        }
    )*) => {
        $(
            $(#[doc = $doc])*
            #[derive(Clone, Copy, Debug)]
            pub struct $name {
                $(
                    pub $field : $field_ty,
                )*
            }

            impl From<$name> for Operation {
                fn from(x: $name) -> Self {
                    Operation::$name(x)
                }
            }
        )*

        #[derive(Clone, Copy, Debug)]
        pub enum Operation {
            $(
                $(#[doc = $doc])*
                $name ( $name ),
            )*
        }

        impl From<Infallible> for Operation {
            fn from(x: Infallible) -> Self {
                match x {}
            }
        }

        impl ApplyOperation for Operation {
            type Ok = Option<PointInfo>;
            type Err = Errors;
            type Undo = Vec<Self>;

            fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Option<PointInfo>, Errors> {
                // ==== START TRICK
                //
                // Since we are using a macro, we need to unify all the
                // `<$name as ApplyOperation>::Ok` types into a single type,
                // `Option<PointInfo>` here.
                //
                // Since we can't implement `From` as we want for `Option<PointInfo>`
                // (because it isn't defined in this crate!), we use a wrapper type
                // instead, and implement every conversion we need on it.
                struct OkWrapper(Option<PointInfo>);
                impl From<()> for OkWrapper {
                    fn from(_: ()) -> Self { Self(None) }
                }
                impl From<PointInfo> for OkWrapper {
                    fn from(x: PointInfo) -> Self { Self(Some(x)) }
                }
                impl From<Option<PointInfo>> for OkWrapper {
                    fn from(x: Option<PointInfo>) -> Self { Self(x) }
                }
                // The reasoning for this type is the same as `OkWrapper` defined above.
                struct ErrWrapper(Errors);
                impl From<Infallible> for ErrWrapper {
                    fn from(x: Infallible) -> Self { match x {} }
                }
                impl<T> From<T> for ErrWrapper where Errors: From<T> {
                    fn from(x: T) -> Self { Self(Errors::from(x)) }
                }
                // ==== END TRICK

                match self {
                    $(
                        Self::$name(x) => {
                            match x.apply_to_dikjstra_map(dijkstra_map) {
                                Ok(x) => Ok(OkWrapper::from(x).0),
                                Err(x) => Err(ErrWrapper::from(x).0)
                            }
                        },
                    )*
                }
            }

            fn undo(self, dijkstra_map: &DijkstraMap) -> Vec<Self> {
                // ==== START TRICK
                // The reasoning for this type is the same as `OkWrapper` defined above,
                // in the `apply_to_dikjstra_map` function.
                struct Wrapper(Vec<Operation>);
                $(
                    impl From<$name> for Wrapper {
                        fn from(x: $name) -> Self {
                            Wrapper(vec![Operation::from(x)])
                        }
                    }
                    impl From<Vec<$name>> for Wrapper {
                        fn from(v: Vec<$name>) -> Self {
                            Wrapper(v.into_iter().map(|x| x.into()).collect())
                        }
                    }
                )*
                impl From<Vec<Operation>> for Wrapper {
                    fn from(v: Vec<Operation>) -> Self {
                        Wrapper(v)
                    }
                }
                // ==== END TRICK

                match self {
                    $(
                        Self::$name(x) => {
                            Wrapper::from(x.undo(dijkstra_map)).0
                        }
                    )*
                }
            }
        }

        impl DijkstraMap {
            $(
                // TODO: maybe we could remove the `impl Into`, to limit implicit conversions ?
                pub fn $fn_name(&mut self, $($field : impl Into<$field_ty>),*)
                    -> Result<
                        <$name as ApplyOperation>::Ok,
                        <$name as ApplyOperation>::Err
                    >
                {
                    let operation = $name {
                        $($field : $field.into()),*
                    };
                    operation.apply_to_dikjstra_map(self)
                }
            )*

            pub(crate) fn apply_operation(&mut self, operation: Operation) -> Result<Option<PointInfo>, Errors> {
                operation.apply_to_dikjstra_map(self)
            }
        }
    };
}

operations! {
    ///  Adds new point with given ID and terrain type into the graph.
    ///
    ///  The new point will have no connections from or to other points.
    ///
    ///  # Errors
    ///
    ///  If a point with that ID already exists, returns [`Err`] without
    ///  modifying the map.
    fn add_point -> pub struct AddPoint {
        pub id : PointId,
        pub terrain_type: TerrainType,
    }

    /// Adds new point with given ID and terrain type into the graph.
    ///
    /// If a point was already associated with `id`, it is replaced.
    ///
    /// Returns a Result for consistency with other method but it connot fail.
    fn add_point_replace -> pub struct AddPointReplace {
        pub id: PointId,
        pub terrain_type: TerrainType,
    }

    /// Removes point from graph along with all of its connections.
    ///
    /// If the point exists in the map, removes it and returns the associated
    /// `PointInfo`. Else, returns `None`.
    fn remove_point -> pub struct RemovePoint {
        pub id: PointId,
    }

    /// Adds connection with given weight between a source point and target
    /// point.
    ///
    /// # Parameters
    ///
    /// - `source` : source point of the connection.
    /// - `target` : target point of the connection.
    /// - `weight` (default : `1.0`) : weight of the connection.
    /// - `bidirectional` (default : [`true`]) : wether or not the reciprocal
    /// connection should be made.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if one of the point does not exist.
    fn connect_points -> pub struct ConnectPoints {
        pub source: PointId,
        pub target: PointId,
        pub weight: Weight,
        pub directional: Directional,
    }

    /// Removes connection between source point and target point.
    ///
    /// # Parameters
    ///
    /// - `source` : source point of the connection.
    /// - `target` : target point of the connection.
    /// - `bidirectional` (default : [`true`]) : if [`true`], also removes the
    /// connection from target to source.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if one of the point does not exist.
    fn remove_connection -> pub struct RemoveConnection {
        pub source: PointId,
        pub target: PointId,
        pub directional: Directional,
    }

    /// Disables point from pathfinding.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if point doesn't exist.
    ///
    /// ## Note
    ///
    /// Points are enabled by default.
    fn disable_point -> pub struct DisablePoint {
        pub id: PointId,
    }

    /// Enables point for pathfinding.
    ///
    /// Useful if the point was previously deactivated by a call to
    /// [`disable_point`](struct.DijkstraMap.html#method.disable_point).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if point doesn't exist.
    ///
    /// ## Note
    ///
    /// Points are enabled by default.
    fn enable_point -> pub struct EnablePoint {
        pub id: PointId,
    }

    /// Sets terrain type for a given point.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the point does not exist.
    fn set_terrain_for_point -> pub struct SetTerrainForPoint {
        pub id: PointId,
        pub ttype: TerrainType,
    }
}

impl ApplyOperation for AddPoint {
    type Ok = ();
    type Err = PointAlreadyExists;
    type Undo = RemovePoint;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        if dijkstra_map.has_point(self.id) {
            Err(PointAlreadyExists)
        } else {
            dijkstra_map.points.insert(
                self.id,
                PointInfo {
                    connections: FnvHashMap::default(),
                    reverse_connections: FnvHashMap::default(),
                    terrain_type: self.terrain_type,
                },
            );
            Ok(())
        }
    }

    fn undo(self, _: &DijkstraMap) -> RemovePoint {
        RemovePoint { id: self.id }
    }
}

impl ApplyOperation for AddPointReplace {
    type Ok = ();
    type Err = Infallible;
    type Undo = Vec<Operation>; // FIXME: find the right type to put here !

    // if id exists :
    //    remove all id connections, replace terrain_type
    //    -> reverse is : add id connections back, and the old terrain.
    // else:
    //    add a new point
    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        dijkstra_map.points.insert(
            self.id,
            PointInfo {
                connections: FnvHashMap::default(),
                reverse_connections: FnvHashMap::default(),
                terrain_type: self.terrain_type,
            },
        );
        Ok(())
    }

    fn undo(self, _: &DijkstraMap) -> Self::Undo {
        todo!("cannot undo remove points without storing the connections before hand");
    }
}

impl ApplyOperation for RemovePoint {
    type Ok = Option<PointInfo>;
    type Err = Infallible;
    type Undo = Vec<Operation>; // FIXME: find the right type to put here !

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        dijkstra_map.disabled_points.remove(&self.id);
        // remove this point's entry from connections
        match dijkstra_map.points.remove(&self.id) {
            None => Ok(None),
            Some(point_info) => {
                // remove reverse connections to this point from neighbours
                for nbr in point_info.connections.keys() {
                    if let Some(point_info_nbr) = dijkstra_map.points.get_mut(nbr) {
                        point_info_nbr.reverse_connections.remove(&self.id);
                    }
                }
                // remove connections to this point from reverse neighbours
                for nbr in point_info.reverse_connections.keys() {
                    if let Some(point_info_nbr) = dijkstra_map.points.get_mut(nbr) {
                        point_info_nbr.connections.remove(&self.id);
                    }
                }
                Ok(Some(point_info))
            }
        }
    }

    fn undo(self, _: &DijkstraMap) -> Self::Undo {
        todo!("cannot undo remove points without storing the connections before hand");
    }
}

impl ApplyOperation for ConnectPoints {
    type Ok = ();
    type Err = PointNotFound;
    type Undo = RemoveConnection;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        let mut add_connection_one_way = |from, to| {
            let (connections, reverse_connections) = dijkstra_map
                .get_connections_and_reverse(from, to)
                .map_err(|_| PointNotFound)?;
            connections.insert(to, self.weight);
            reverse_connections.insert(from, self.weight);
            Ok(())
        };
        add_connection_one_way(self.source, self.target)?;
        if self.directional == Directional::Bidirectional {
            add_connection_one_way(self.target, self.source)?;
        }
        Ok(())
    }

    fn undo(self, _: &DijkstraMap) -> Self::Undo {
        RemoveConnection {
            source: self.source,
            target: self.target,
            directional: self.directional,
        }
    }
}

impl ApplyOperation for RemoveConnection {
    type Ok = ();
    type Err = PointNotFound;
    type Undo = ConnectPoints;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        let mut remove_connection_one_way = |from, to| {
            let (connections, reverse_connections) = dijkstra_map
                .get_connections_and_reverse(from, to)
                .map_err(|_| PointNotFound)?;
            connections.remove(&to);
            reverse_connections.remove(&from);
            Ok(())
        };
        remove_connection_one_way(self.source, self.target)?;
        if self.directional == Directional::Bidirectional {
            remove_connection_one_way(self.target, self.source)?;
        }
        Ok(())
    }

    fn undo(self, dijkstra_map: &DijkstraMap) -> Self::Undo {
        ConnectPoints {
            source: self.source,
            target: self.target,
            weight: dijkstra_map
                .get_connection(self.source, self.target)
                .unwrap(),
            directional: self.directional,
        }
    }
}

impl ApplyOperation for DisablePoint {
    type Ok = ();
    type Err = PointNotFound;
    type Undo = EnablePoint;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        if dijkstra_map.points.contains_key(&self.id) {
            dijkstra_map.disabled_points.insert(self.id);
            Ok(())
        } else {
            Err(PointNotFound)
        }
    }

    fn undo(self, _: &DijkstraMap) -> Self::Undo {
        EnablePoint { id: self.id }
    }
}

impl ApplyOperation for EnablePoint {
    type Ok = ();
    type Err = PointNotFound;
    type Undo = DisablePoint;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        if dijkstra_map.points.contains_key(&self.id) {
            dijkstra_map.disabled_points.remove(&self.id);
            Ok(())
        } else {
            Err(PointNotFound)
        }
    }

    fn undo(self, _: &DijkstraMap) -> Self::Undo {
        DisablePoint { id: self.id }
    }
}

impl ApplyOperation for SetTerrainForPoint {
    type Ok = ();
    type Err = PointNotFound;
    type Undo = SetTerrainForPoint;

    fn apply_to_dikjstra_map(self, dijkstra_map: &mut DijkstraMap) -> Result<Self::Ok, Self::Err> {
        match dijkstra_map.points.get_mut(&self.id) {
            Some(PointInfo { terrain_type, .. }) => {
                *terrain_type = self.ttype;
                Ok(())
            }
            None => Err(PointNotFound),
        }
    }

    fn undo(self, dijkstra_map: &DijkstraMap) -> Self::Undo {
        match dijkstra_map.get_terrain_for_point(self.id) {
            Some(terrain_type) => SetTerrainForPoint {
                id: self.id,
                ttype: terrain_type,
            },
            None => panic!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Errors {
    PointAlreadyExists(PointAlreadyExists),
    PointNotFound(PointNotFound),
}

impl From<PointAlreadyExists> for Errors {
    fn from(value: PointAlreadyExists) -> Self {
        Self::PointAlreadyExists(value)
    }
}

impl From<PointNotFound> for Errors {
    fn from(value: PointNotFound) -> Self {
        Self::PointNotFound(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo() {
        let mut dmap = DijkstraMap::new();
        dmap.add_point(PointId(3), TerrainType::DefaultTerrain)
            .unwrap();
        dmap.add_point(PointId(4), TerrainType::DefaultTerrain)
            .unwrap();
        dmap.add_point(PointId(5), TerrainType::DefaultTerrain)
            .unwrap();
        dmap.add_point(PointId(6), TerrainType::DefaultTerrain)
            .unwrap();
        dmap.connect_points(
            PointId(5),
            PointId(6),
            Weight::default(),
            Directional::Unidirectional,
        )
        .unwrap();
        let cloned = dmap.clone();
        assert_eq!(dmap, cloned);
        for op in [
            Operation::AddPoint(AddPoint {
                id: PointId(0),
                terrain_type: TerrainType::DefaultTerrain,
            }),
            Operation::ConnectPoints(ConnectPoints {
                source: PointId(3),
                target: PointId(4),
                weight: Weight::default(),
                directional: false.into(),
            }),
            Operation::DisablePoint(DisablePoint { id: PointId(4) }),
            Operation::SetTerrainForPoint(SetTerrainForPoint {
                id: PointId(3),
                ttype: TerrainType::Terrain(4),
            }),
            Operation::RemoveConnection(RemoveConnection {
                source: PointId(5),
                target: PointId(6),
                directional: false.into(),
            }),
        ] {
            let undo = op.undo(&dmap);
            dmap.apply_operation(op).unwrap();
            for undo in &undo {
                dmap.apply_operation(*undo).unwrap();
            }
            assert_eq!(
                dmap, cloned,
                "map wasn't restored to it's original state for op {:?} and undo {:?}",
                op, undo
            );
        }
    }
}

impl Default for Directional {
    fn default() -> Self {
        Directional::Bidirectional
    }
}

impl From<bool> for Directional {
    fn from(val: bool) -> Self {
        match val {
            true => Directional::Bidirectional,
            false => Directional::Unidirectional,
        }
    }
}

impl From<Option<bool>> for Directional {
    fn from(val: Option<bool>) -> Self {
        match val {
            Some(x) => x.into(),
            None => Directional::Bidirectional,
        }
    }
}

impl From<Option<Weight>> for Weight {
    fn from(val: Option<Weight>) -> Self {
        val.unwrap_or_default()
    }
}

impl From<Option<TerrainType>> for TerrainType {
    fn from(val: Option<TerrainType>) -> Self {
        val.unwrap_or_default()
    }
}
