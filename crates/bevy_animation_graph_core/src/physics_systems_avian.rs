use avian3d::prelude::{
    AngularMotor, AngularVelocity, ConstantAngularAcceleration, ConstantLinearAcceleration,
    Gravity, LinearVelocity, MaxLinearSpeed, MotorModel, Position, RigidBody, Rotation,
    SphericalJoint,
};
use bevy::{
    asset::Assets,
    ecs::{
        entity::Entity,
        hierarchy::ChildOf,
        query::{With, Without},
        system::{Commands, Query, Res},
    },
    math::{Isometry3d, Quat, Vec3},
    time::Time,
    transform::components::GlobalTransform,
};

use crate::{
    animation_graph::DEFAULT_OUTPUT_RAGDOLL_CONFIG,
    animation_graph_player::AnimationGraphPlayer,
    context::{
        pose_fallback::{PoseFallbackContext, RootOffsetResult},
        system_resources::SystemResources,
    },
    ragdoll::{
        bone_mapping::RagdollBoneMap,
        definition::{BodyMode, JointVariant, Ragdoll},
        read_pose_avian::read_pose,
        relative_kinematic_body::{RelativeKinematicBody, RelativeKinematicBodyPositionBased},
        spawning::spawn_ragdoll_avian,
        write_pose::write_pose_to_ragdoll,
    },
};

pub fn update_relative_kinematic_body_velocities(
    mut relative_kinematic_query: Query<(
        &RelativeKinematicBody,
        &RigidBody,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
    relative_to_query: Query<(&LinearVelocity, &AngularVelocity), Without<RelativeKinematicBody>>,
) {
    for (relative_kinematic_body, rigid_body, mut linvel, mut angvel) in
        &mut relative_kinematic_query
    {
        if !rigid_body.is_kinematic() {
            continue;
        }
        let (base_linvel, base_angvel) = relative_kinematic_body
            .relative_to
            .and_then(|relative_to| relative_to_query.get(relative_to).ok())
            .map(|(linvel, angvel)| (linvel.0, angvel.0))
            .unwrap_or((Vec3::ZERO, Vec3::ZERO));

        let composed_linvel = base_linvel + relative_kinematic_body.kinematic_linear_velocity;
        let composed_angvel = base_angvel + relative_kinematic_body.kinematic_angular_velocity;

        linvel.0 = composed_linvel;
        angvel.0 = composed_angvel;
    }
}

pub fn update_relative_kinematic_position_based_body_velocities(
    mut relative_kinematic_query: Query<(
        &RelativeKinematicBodyPositionBased,
        &Position,
        &Rotation,
        &mut RelativeKinematicBody,
    )>,
    relative_to_query: Query<(&Position, &Rotation), Without<RelativeKinematicBodyPositionBased>>,
    time: Res<Time>,
) {
    for (rel_kinbod_pos, pos, rot, mut rel_kinbod) in &mut relative_kinematic_query {
        // First we need to compute the current relative position
        let cur_isometry = if let Some((base_pos, base_rot)) = rel_kinbod_pos
            .relative_to
            .and_then(|e| relative_to_query.get(e).ok())
        {
            let cur_global_isometry = Isometry3d::new(pos.0, rot.0);
            let cur_base_isometry = Isometry3d::new(base_pos.0, base_rot.0);

            cur_base_isometry.inverse() * cur_global_isometry
        } else {
            Isometry3d::new(pos.0, rot.0)
        };

        let linvel = (rel_kinbod_pos.relative_target.translation - cur_isometry.translation)
            / time.delta_secs();

        let start = cur_isometry.rotation;
        let mut end = rel_kinbod_pos.relative_target.rotation;
        if start.dot(end) < 0.0 {
            end = -end;
        }
        let quat_diff = (end * start.conjugate()).normalize();
        let axis_angle_diff = quat_diff.to_scaled_axis();
        let angvel = axis_angle_diff / time.delta_secs();

        rel_kinbod.kinematic_linear_velocity = linvel.into();
        rel_kinbod.kinematic_angular_velocity = angvel;
        rel_kinbod.relative_to = rel_kinbod_pos.relative_to;
    }
}

pub fn spawn_missing_ragdolls_avian(
    mut commands: Commands,
    mut animation_players: Query<(Entity, &mut AnimationGraphPlayer, &GlobalTransform)>,
    parent_query: Query<&ChildOf>,
    velocity_check: Query<(), (With<LinearVelocity>, With<AngularVelocity>)>,
    ragdoll_assets: Res<Assets<Ragdoll>>,
) {
    for (entity, mut player, global_transform) in &mut animation_players {
        if let Some(ragdoll_asset_id) = player.ragdoll.as_ref().map(|h| h.id())
            && player.spawned_ragdoll.is_none()
            && let Some(ragdoll) = ragdoll_assets.get(ragdoll_asset_id)
        {
            // We need to find the nearest simulated "ancestor". This will be the origin for
            // relative kinematic bodies in the ragdoll
            let simulated_parent = parent_query
                .iter_ancestors(entity)
                .find(|ancestor| velocity_check.contains(*ancestor));

            let spawned_ragdoll = spawn_ragdoll_avian(
                entity,
                ragdoll,
                global_transform.to_isometry(),
                simulated_parent,
                &mut commands,
            );
            player.spawned_ragdoll = Some(spawned_ragdoll);
        }
    }
}

/// Updates the rigidbody modes of bodies in a ragdoll based on configuration
pub fn update_ragdoll_rigidbodies(
    animation_players: Query<&AnimationGraphPlayer>,
    ragdoll_assets: Res<Assets<Ragdoll>>,
    rigid_body_query: Query<&RigidBody>,
    mut commands: Commands,
) {
    for player in &animation_players {
        if let Some(ragdoll_asset_id) = player.ragdoll.as_ref().map(|h| h.id())
            && let Some(ragdoll) = ragdoll_assets.get(ragdoll_asset_id)
            && let Some(spawned_ragdoll) = &player.spawned_ragdoll
        {
            let config = player
                .get_outputs()
                .get(DEFAULT_OUTPUT_RAGDOLL_CONFIG)
                .and_then(|v| v.as_ragdoll_config().ok())
                .cloned()
                .unwrap_or_default();

            for body in ragdoll.bodies.values() {
                let Some(body_entity) = spawned_ragdoll.bodies.get(&body.id) else {
                    continue;
                };
                let Ok(rigid_body) = rigid_body_query.get(*body_entity) else {
                    continue;
                };

                let body_mode = config.body_mode(body.id).unwrap_or(body.default_mode);

                // The pose-following modes simulate as dynamic bodies (driven by forces/motors).
                let target_mode = if body_mode.is_dynamic() {
                    RigidBody::Dynamic
                } else {
                    RigidBody::Kinematic
                };

                if *rigid_body != target_mode {
                    commands.entity(*body_entity).insert(target_mode);
                }
            }
        }
    }
}

/// Updates the target positions of bodies in a ragdoll based on the animated pose
pub fn update_ragdolls_avian(
    animation_players: Query<&AnimationGraphPlayer>,
    ragdoll_assets: Res<Assets<Ragdoll>>,
    bone_map_assets: Res<Assets<RagdollBoneMap>>,
    mut relative_kinematic_body_query: Query<&mut RelativeKinematicBodyPositionBased>,
    system_resources: SystemResources,
) {
    for player in &animation_players {
        if let Some(ragdoll_asset_id) = player.ragdoll.as_ref().map(|h| h.id())
            && let Some(ragdoll) = ragdoll_assets.get(ragdoll_asset_id)
            && let Some(spawned_ragdoll) = &player.spawned_ragdoll
            && let Some(bone_map_handle) = &player.ragdoll_bone_map
            && let Some(bone_map) = bone_map_assets.get(bone_map_handle)
            && let Some(pose) = player.get_default_output_pose()
            && let Some(skeleton) = system_resources.skeleton_assets.get(&pose.skeleton)
        {
            let pose_fallback = PoseFallbackContext {
                entity_map: &player.entity_map,
                resources: &system_resources,
                fallback_to_identity: true,
            };
            let rb_targets =
                write_pose_to_ragdoll(pose, skeleton, ragdoll, bone_map, pose_fallback);
            let root_transform = pose_fallback.compute_root_global_transform_to_rigidbody(skeleton);

            for body_target in rb_targets.bodies {
                let Some(body_entity) = spawned_ragdoll.bodies.get(&body_target.body_id) else {
                    continue;
                };

                let Ok(mut relative_kinematic_body) =
                    relative_kinematic_body_query.get_mut(*body_entity)
                else {
                    continue;
                };

                let target = body_target.character_space_isometry;

                match root_transform {
                    RootOffsetResult::FromRigidbody(entity, transform) => {
                        relative_kinematic_body.relative_to = Some(entity);
                        relative_kinematic_body.relative_target = transform.to_isometry() * target;
                    }
                    RootOffsetResult::FromRoot(transform) => {
                        relative_kinematic_body.relative_to = None;
                        relative_kinematic_body.relative_target = transform.to_isometry() * target;
                    }
                    RootOffsetResult::Failed => continue,
                }
            }
        }
    }
}

pub fn read_back_poses_avian(
    mut animation_players: Query<&mut AnimationGraphPlayer>,
    bone_map_assets: Res<Assets<RagdollBoneMap>>,
    pos_query: Query<(&Position, &Rotation)>,
    system_resources: SystemResources,
) {
    for mut player in &mut animation_players {
        if let Some(spawned_ragdoll) = &player.spawned_ragdoll
            && let Some(bone_map_handle) = &player.ragdoll_bone_map
            && let Some(bone_map) = bone_map_assets.get(bone_map_handle)
            && let Some(pose) = player.get_default_output_pose()
            && let Some(skeleton) = system_resources.skeleton_assets.get(&pose.skeleton)
        {
            let config = player
                .get_outputs()
                .get(DEFAULT_OUTPUT_RAGDOLL_CONFIG)
                .and_then(|v| v.as_ragdoll_config().ok())
                .cloned()
                .unwrap_or_default();

            let pose_fallback = PoseFallbackContext {
                entity_map: &player.entity_map,
                resources: &system_resources,
                fallback_to_identity: true,
            };

            let updated_pose = read_pose(
                spawned_ragdoll,
                bone_map,
                skeleton,
                &pos_query,
                pose_fallback,
                pose,
                &config,
            );

            player.set_default_output_pose(updated_pose);
        }
    }
}

/// Implicit spring-damper acceleration (the acceleration form of avian's
/// [`MotorModel::SpringDamper`](avian3d::prelude::MotorModel::SpringDamper)): given a position error
/// and velocity error, returns the angular/linear acceleration that drives both towards zero with the
/// requested natural `frequency` (Hz) and `damping_ratio`. Unconditionally stable.
fn spring_damper_accel(
    position_error: Vec3,
    velocity_error: Vec3,
    frequency: f32,
    damping_ratio: f32,
    dt: f32,
) -> Vec3 {
    use core::f32::consts::TAU;
    let omega = TAU * frequency;
    let omega_sq = omega * omega;
    let two_zeta_omega = 2.0 * damping_ratio * omega;
    let inv_denominator = 1.0 / (1.0 + two_zeta_omega * dt + omega_sq * dt * dt);
    // velocity change = (omega_sq*pos + 2*zeta*omega*vel) * dt * inv_denom; acceleration = change / dt.
    (omega_sq * position_error + two_zeta_omega * velocity_error) * inv_denominator
}

/// The shortest-arc rotation vector (axis * angle) from `current` to `target`, in world space.
fn orientation_error(target: Quat, current: Quat) -> Vec3 {
    let mut error = target * current.conjugate();
    if error.w < 0.0 {
        error = -error;
    }
    error.to_scaled_axis()
}

/// Drives [`BodyMode::FollowAbsolute`] bodies towards the animated target pose with a world-space
/// spring-damper (PD), written into the bodies' [`ConstantAngularAcceleration`] /
/// [`ConstantLinearAcceleration`]. Non-following dynamic bodies have those inputs zeroed so a body
/// leaving a follow mode goes limp. The target transform is reconstructed from the body's
/// [`RelativeKinematicBodyPositionBased`] (set by [`update_ragdolls_avian`]).
pub fn drive_pose_following_absolute_avian(
    animation_players: Query<&AnimationGraphPlayer>,
    ragdoll_assets: Res<Assets<Ragdoll>>,
    gravity: Res<Gravity>,
    time: Res<Time>,
    parent_query: Query<(&Position, &Rotation)>,
    mut bodies: Query<(
        &RelativeKinematicBodyPositionBased,
        &Position,
        &Rotation,
        &LinearVelocity,
        &AngularVelocity,
        &mut ConstantAngularAcceleration,
        &mut ConstantLinearAcceleration,
    )>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    for player in &animation_players {
        let Some(ragdoll_asset_id) = player.ragdoll.as_ref().map(|h| h.id()) else {
            continue;
        };
        let (Some(ragdoll), Some(spawned_ragdoll)) = (
            ragdoll_assets.get(ragdoll_asset_id),
            player.spawned_ragdoll.as_ref(),
        ) else {
            continue;
        };

        let config = player
            .get_outputs()
            .get(DEFAULT_OUTPUT_RAGDOLL_CONFIG)
            .and_then(|v| v.as_ragdoll_config().ok())
            .cloned()
            .unwrap_or_default();

        let tuning = ragdoll.pose_following;

        for body in ragdoll.bodies.values() {
            let Some(&body_entity) = spawned_ragdoll.bodies.get(&body.id) else {
                continue;
            };
            let body_mode = config.body_mode(body.id).unwrap_or(body.default_mode);

            let Ok((rkb, pos, rot, lin_vel, ang_vel, mut ang_acc, mut lin_acc)) =
                bodies.get_mut(body_entity)
            else {
                continue;
            };

            if body_mode != BodyMode::FollowAbsolute {
                // Free / kinematic bodies don't use the absolute force inputs; relative-follow bodies
                // are driven by their joint motor, not these.
                if body_mode != BodyMode::FollowRelative {
                    ang_acc.0 = Vec3::ZERO;
                    lin_acc.0 = Vec3::ZERO;
                }
                continue;
            }

            // World-space target transform: root_world * (target relative to root).
            let root_iso = rkb
                .relative_to
                .and_then(|e| parent_query.get(e).ok())
                .map(|(p, r)| Isometry3d::new(p.0, r.0))
                .unwrap_or(Isometry3d::IDENTITY);
            let target = root_iso * rkb.relative_target;

            // Angular spring towards the target rotation.
            let pos_error = orientation_error(target.rotation, rot.0);
            ang_acc.0 = spring_damper_accel(
                pos_error,
                -ang_vel.0,
                tuning.frequency,
                tuning.damping_ratio,
                dt,
            );

            // Linear spring towards the target position, plus gravity compensation.
            let lin_error = Vec3::from(target.translation) - pos.0;
            lin_acc.0 = spring_damper_accel(
                lin_error,
                -lin_vel.0,
                tuning.linear_frequency,
                tuning.linear_damping_ratio,
                dt,
            ) - gravity.0 * tuning.gravity_compensation;
        }
    }
}

/// Drives [`BodyMode::FollowRelative`] bodies by enabling and steering the spherical-joint motor of
/// the joint whose child (`body2`) is the body. The motor's `target_orientation` is the animated
/// target rotation of the child relative to its parent (both targets share the ragdoll-root frame, so
/// the root transform cancels). Joints whose child is not in relative-follow mode have their motor
/// disabled.
pub fn drive_pose_following_relative_avian(
    animation_players: Query<&AnimationGraphPlayer>,
    ragdoll_assets: Res<Assets<Ragdoll>>,
    body_targets: Query<&RelativeKinematicBodyPositionBased>,
    mut joints: Query<&mut SphericalJoint>,
) {
    for player in &animation_players {
        let Some(ragdoll_asset_id) = player.ragdoll.as_ref().map(|h| h.id()) else {
            continue;
        };
        let (Some(ragdoll), Some(spawned_ragdoll)) = (
            ragdoll_assets.get(ragdoll_asset_id),
            player.spawned_ragdoll.as_ref(),
        ) else {
            continue;
        };

        let config = player
            .get_outputs()
            .get(DEFAULT_OUTPUT_RAGDOLL_CONFIG)
            .and_then(|v| v.as_ragdoll_config().ok())
            .cloned()
            .unwrap_or_default();

        let tuning = ragdoll.pose_following;

        for joint in ragdoll.joints.values() {
            let JointVariant::Spherical(spherical) = &joint.variant else {
                continue;
            };
            let Some(&joint_entity) = spawned_ragdoll.joints.get(&joint.id) else {
                continue;
            };
            let Ok(mut joint_component) = joints.get_mut(joint_entity) else {
                continue;
            };

            let child_mode = ragdoll
                .get_body(spherical.body2)
                .map(|b| config.body_mode(b.id).unwrap_or(b.default_mode));

            if child_mode != Some(BodyMode::FollowRelative) {
                joint_component.motor.enabled = false;
                continue;
            }

            // Parent- and child-relative targets (both in the ragdoll-root frame).
            let parent_target = spawned_ragdoll
                .bodies
                .get(&spherical.body1)
                .and_then(|&e| body_targets.get(e).ok())
                .map(|t| t.relative_target.rotation);
            let child_target = spawned_ragdoll
                .bodies
                .get(&spherical.body2)
                .and_then(|&e| body_targets.get(e).ok())
                .map(|t| t.relative_target.rotation);

            let (Some(parent_rot), Some(child_rot)) = (parent_target, child_target) else {
                joint_component.motor.enabled = false;
                continue;
            };

            joint_component.target_orientation = parent_rot.conjugate() * child_rot;
            joint_component.motor = AngularMotor {
                enabled: true,
                max_torque: tuning.max_torque,
                motor_model: MotorModel::SpringDamper {
                    frequency: tuning.frequency,
                    damping_ratio: tuning.damping_ratio,
                },
                ..joint_component.motor
            };
        }
    }
}

/// Anti-launch safety net for pose-following ragdolls. A stiff follow body levering against a contact
/// can have the constraint solver spike its velocity between the constraint solve and position
/// integration (avian's own `MaxLinearSpeed` clamp runs at substep *start*, before the spike). This
/// clamps each ragdoll body's solver velocity to its [`MaxLinearSpeed`] in that gap. Ported from the
/// guide-dog active-ragdoll controller.
pub fn clamp_pose_following_body_velocity_avian(
    mut bodies: Query<
        (
            &mut avian3d::dynamics::solver::solver_body::SolverBody,
            &MaxLinearSpeed,
        ),
        With<RelativeKinematicBodyPositionBased>,
    >,
) {
    for (mut solver_body, max_speed) in &mut bodies {
        let max = max_speed.0;
        let speed_sq = solver_body.linear_velocity.length_squared();
        if speed_sq > max * max {
            solver_body.linear_velocity *= max / speed_sq.sqrt();
        }
    }
}
