use bevy::{
    asset::Asset,
    ecs::component::Component,
    math::{
        Isometry3d, Vec3,
        primitives::{Capsule3d, Cuboid, Measured3d, Sphere},
    },
    platform::collections::HashMap,
    reflect::Reflect,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Asset, Debug, Clone, Reflect, Serialize, Deserialize)]
pub struct Ragdoll {
    pub bodies: HashMap<BodyId, Body>,
    pub colliders: HashMap<ColliderId, Collider>,
    pub joints: HashMap<JointId, Joint>,

    #[serde(default)]
    pub suffixes: SymmetrySuffixes,

    /// Total mass will be divided among all colliders according to the volume of their attached
    /// colliders.
    #[serde(default)]
    pub total_mass: f32,

    /// Tuning for the "pose following" [`BodyMode`]s.
    #[serde(default)]
    pub pose_following: PoseFollowing,
}

impl Default for Ragdoll {
    fn default() -> Self {
        Self {
            bodies: Default::default(),
            colliders: Default::default(),
            joints: Default::default(),
            suffixes: Default::default(),

            total_mass: 70.,
            pose_following: Default::default(),
        }
    }
}

impl Ragdoll {
    pub fn get_body(&self, id: BodyId) -> Option<&Body> {
        self.bodies.get(&id)
    }

    pub fn get_body_mut(&mut self, id: BodyId) -> Option<&mut Body> {
        self.bodies.get_mut(&id)
    }

    ///  Adds new body to the ragdoll. This operation is idempotent; if you try to add a body with
    ///  an ID that already exists, it's ignored.
    pub fn add_body(&mut self, body: Body) {
        if !self.bodies.contains_key(&body.id) {
            self.bodies.insert(body.id, body);
        }
    }

    ///  Adds new collider to the ragdoll. This operation is idempotent; if you try to add a
    ///  collider with an ID that already exists, it's ignored.
    pub fn add_collider(&mut self, collider: Collider) {
        if !self.colliders.contains_key(&collider.id) {
            self.colliders.insert(collider.id, collider);
        }
    }

    ///  Adds new joint to the ragdoll. This operation is idempotent; if you try to add a
    ///  joint with an ID that already exists, it's ignored.
    pub fn add_joint(&mut self, joint: Joint) {
        if !self.joints.contains_key(&joint.id) {
            self.joints.insert(joint.id, joint);
        }
    }

    pub fn delete_body(&mut self, id: BodyId) {
        self.bodies.remove(&id);
    }

    pub fn delete_collider(&mut self, id: ColliderId) {
        self.colliders.remove(&id);

        for body in self.bodies.values_mut() {
            body.colliders.retain(|c_id| *c_id != id);
        }
    }

    pub fn delete_joint(&mut self, id: JointId) {
        self.joints.remove(&id);
    }

    pub fn get_collider(&self, id: ColliderId) -> Option<&Collider> {
        self.colliders.get(&id)
    }

    pub fn get_collider_mut(&mut self, id: ColliderId) -> Option<&mut Collider> {
        self.colliders.get_mut(&id)
    }

    pub fn get_joint(&self, id: JointId) -> Option<&Joint> {
        self.joints.get(&id)
    }

    pub fn get_joint_mut(&mut self, id: JointId) -> Option<&mut Joint> {
        self.joints.get_mut(&id)
    }

    pub fn iter_bodies(&self) -> impl Iterator<Item = &Body> {
        self.bodies.values()
    }

    pub fn iter_bodies_mut(&mut self) -> impl Iterator<Item = &mut Body> {
        self.bodies.values_mut()
    }

    pub fn iter_body_ids(&self) -> impl Iterator<Item = BodyId> {
        self.bodies.keys().copied()
    }

    pub fn iter_colliders(&self) -> impl Iterator<Item = &Collider> {
        self.colliders.values()
    }

    pub fn iter_joints(&self) -> impl Iterator<Item = &Joint> {
        self.joints.values()
    }

    pub fn iter_joint_ids(&self) -> impl Iterator<Item = JointId> {
        self.joints.keys().copied()
    }
}

/// Determines which suffixes to apply to labels of elements that make use of symmetry
#[derive(Debug, Clone, Reflect, Serialize, Deserialize)]
pub struct SymmetrySuffixes {
    pub original: String,
    pub mirror: String,
}

impl Default for SymmetrySuffixes {
    fn default() -> Self {
        Self {
            original: ".R".into(),
            mirror: ".L".into(),
        }
    }
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize)]
pub struct Body {
    pub id: BodyId,
    #[serde(default)]
    pub label: String,
    /// Position of this rigidbody relative to the character root transform.
    ///
    /// You cannot rotate the rigidbody. Instead, you can rotate colliders attached to it.
    pub offset: Vec3,
    pub colliders: Vec<ColliderId>,
    pub default_mode: BodyMode,

    #[serde(default)]
    pub use_symmetry: bool,
    /// If symmetry is enabled and this body was created as the image under the symmetry of another
    /// body, which body is it?
    #[serde(default)]
    pub created_from: Option<BodyId>,
}

impl Body {
    // The new function is non-deterministic (random uuid assigned)
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            id: BodyId {
                uuid: Uuid::new_v4(),
            },
            label: "New body".into(),
            offset: Default::default(),
            colliders: Default::default(),
            default_mode: BodyMode::Kinematic,

            use_symmetry: false,
            created_from: None,
        }
    }
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BodyMode {
    /// The body is kinematically driven to exactly match the animated pose (via computed velocities).
    #[default]
    Kinematic,
    /// The body is a free physics body (a classic ragdoll bone); its pose is read back from the sim.
    Dynamic,
    /// "Pose following", absolute mode: a dynamic body pulled towards the animated target pose by a
    /// world-space spring (PD). The body still collides and reacts to forces, but is biased towards
    /// the animation. See [`Ragdoll::pose_following`].
    FollowAbsolute,
    /// "Pose following", relative mode: a dynamic body whose joint motor drives its rotation towards
    /// the animated target *relative to its parent body*. Requires the body to be the child (`body2`)
    /// of a [`SphericalJoint`]. See [`Ragdoll::pose_following`].
    FollowRelative,
}

impl BodyMode {
    /// Whether the body simulates as a dynamic rigid body (everything except [`Kinematic`](Self::Kinematic)).
    pub fn is_dynamic(self) -> bool {
        !matches!(self, BodyMode::Kinematic)
    }

    /// Whether the body is driven towards the animated pose by forces/motors (a "pose following" mode).
    pub fn is_follow(self) -> bool {
        matches!(self, BodyMode::FollowAbsolute | BodyMode::FollowRelative)
    }
}

/// Tuning for the "pose following" [`BodyMode`]s (see [`BodyMode::FollowAbsolute`] /
/// [`BodyMode::FollowRelative`]). Lives on the [`Ragdoll`] asset so it is authorable in the editor.
///
/// The angular drive uses an implicit spring-damper (matching avian's
/// [`MotorModel::SpringDamper`](avian's joint motor model)); for `FollowRelative` it configures the
/// spherical joint motor, for `FollowAbsolute` it is applied as a world-space angular spring.
#[derive(Reflect, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PoseFollowing {
    /// Natural frequency (Hz) of the angular spring/motor. Higher = stiffer pose tracking.
    pub frequency: f32,
    /// Damping ratio of the angular spring/motor (1.0 = critically damped).
    pub damping_ratio: f32,
    /// Maximum torque the angular drive may apply (N·m). [`f32::MAX`] = unlimited.
    pub max_torque: f32,
    /// Natural frequency (Hz) of the **absolute**-mode linear spring (translational pose tracking).
    pub linear_frequency: f32,
    /// Damping ratio of the **absolute**-mode linear spring.
    pub linear_damping_ratio: f32,
    /// Fraction of gravity the **absolute**-mode linear drive compensates (1.0 = float in place).
    pub gravity_compensation: f32,
    /// Hard cap (m/s) on a follow body's linear speed — the anti-launch safety net. A stiff
    /// pose-following body levering against a contact can otherwise spike skyward; capping the
    /// per-body solver velocity bounds that to a stumble.
    pub max_body_speed: f32,
}

impl Default for PoseFollowing {
    fn default() -> Self {
        Self {
            frequency: 8.0,
            damping_ratio: 1.0,
            max_torque: f32::MAX,
            linear_frequency: 15.0,
            linear_damping_ratio: 1.0,
            gravity_compensation: 1.0,
            max_body_speed: 12.0,
        }
    }
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize, Default)]
pub enum ColliderMassMode {
    Override(f32),
    #[default]
    ByVolume,
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize)]
pub struct Collider {
    pub id: ColliderId,
    /// Local offset w.r.t. the rigidbody it's attached to
    pub local_offset: Isometry3d,
    pub shape: ColliderShape,
    pub layer_membership: u32,
    pub layer_filter: u32,
    pub override_layers: bool,

    #[serde(default)]
    pub mass_mode: ColliderMassMode,

    /// Label that will be attached to the created collider in a [`ColliderLabel`] component.
    pub label: String,
    /// If symmetry is enabled and this body was created as the image under the symmetry of another
    /// body, which body is it?
    #[serde(default)]
    pub created_from: Option<ColliderId>,
}

impl Collider {
    // The new function is non-deterministic (random uuid assigned)
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            id: ColliderId {
                uuid: Uuid::new_v4(),
            },
            local_offset: Default::default(),
            shape: ColliderShape::Cuboid(Cuboid::new(0.2, 0.2, 0.2)),
            layer_membership: 1,
            layer_filter: 1,
            override_layers: false,
            label: "New collider".into(),
            created_from: None,
            mass_mode: ColliderMassMode::ByVolume,
        }
    }

    pub fn volume(&self) -> f32 {
        match &self.shape {
            ColliderShape::Sphere(sphere) => sphere.volume(),
            ColliderShape::Capsule(capsule3d) => capsule3d.volume(),
            ColliderShape::Cuboid(cuboid) => cuboid.volume(),
        }
    }
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    pub id: JointId,
    #[serde(default)]
    pub label: String,

    #[serde(default)]
    pub use_symmetry: bool,
    /// If symmetry is enabled and this body was created as the image under the symmetry of another
    /// body, which body is it?
    #[serde(default)]
    pub created_from: Option<JointId>,

    pub variant: JointVariant,
}

impl Joint {
    // The new function is non-deterministic (random uuid assigned)
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            id: JointId {
                uuid: Uuid::new_v4(),
            },
            label: "New joint".into(),
            variant: JointVariant::Spherical(SphericalJoint::default()),
            use_symmetry: false,
            created_from: None,
        }
    }
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JointVariant {
    Spherical(SphericalJoint),
    Revolute(RevoluteJoint),
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SphericalJoint {
    pub body1: BodyId,
    pub body2: BodyId,
    pub position: Vec3,
    pub twist_axis: Vec3,
    pub swing_limit: Option<AngleLimit>,
    pub twist_limit: Option<AngleLimit>,
    pub point_compliance: f32,
    pub swing_compliance: f32,
    pub twist_compliance: f32,
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RevoluteJoint {
    pub body1: BodyId,
    pub body2: BodyId,
    pub position: Vec3,
    pub hinge_axis: Vec3,
    pub angle_limit: Option<AngleLimit>,
    pub point_compliance: f32,
    pub align_compliance: f32,
    pub limit_compliance: f32,
}

#[derive(Reflect, Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AngleLimit {
    pub min: f32,
    pub max: f32,
}

#[derive(
    Default, Reflect, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[reflect(Hash)]
pub struct BodyId {
    uuid: Uuid,
}

impl BodyId {
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self { uuid }
    }
}

impl std::fmt::Debug for BodyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.uuid.fmt(f)
    }
}

#[derive(Reflect, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct ColliderId {
    uuid: Uuid,
}

impl std::fmt::Debug for ColliderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.uuid.fmt(f)
    }
}

#[derive(Reflect, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct JointId {
    uuid: Uuid,
}

impl std::fmt::Debug for JointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.uuid.fmt(f)
    }
}

#[derive(Component, Reflect)]
pub struct BodyLabel(pub String);

#[derive(Component, Reflect)]
pub struct JointLabel(pub String);

#[derive(Component, Reflect)]
pub struct ColliderLabel(pub String);

#[derive(Debug, Clone, Reflect, PartialEq, Serialize, Deserialize)]
pub enum ColliderShape {
    Sphere(Sphere),
    Capsule(Capsule3d),
    Cuboid(Cuboid),
}

impl ColliderShape {
    #[cfg(feature = "physics_avian")]
    pub fn avian_collider(&self) -> avian3d::prelude::Collider {
        use avian3d::prelude::Collider;
        match self {
            ColliderShape::Sphere(sphere) => Collider::sphere(sphere.radius),
            ColliderShape::Capsule(capsule3d) => {
                Collider::capsule(capsule3d.radius, 2. * capsule3d.half_length)
            }
            ColliderShape::Cuboid(cuboid) => Collider::cuboid(
                2. * cuboid.half_size.x,
                2. * cuboid.half_size.y,
                2. * cuboid.half_size.z,
            ),
        }
    }
}
