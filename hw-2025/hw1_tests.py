import numpy as np

init_qpos = np.array([ 0.3   , -0.2   ,  0.75  ,  1.    ,  0.    ,  0.    ,  0.    ,
        0.1   ,  0.3   ,  1.01  ,  1.    ,  0.    ,  0.    ,  0.    ,
        0.1   , -0.3   ,  1.01  ,  1.    ,  0.    ,  0.    ,  0.    ,
        0.55  ,  0.05  ,  1.01  ,  1.    ,  0.    ,  0.    ,  0.    ,
        0.3   ,  0.    ,  0.81  ,  1.    ,  0.    ,  0.    ,  0.    ,
       -1.5708, -1.5708,  1.5708, -1.5708, -1.5708,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ])

collide_robot_obstacle = np.array([ 3.00000000e-01, -2.00000000e-01,  7.39784489e-01,  1.00000000e+00,
        3.93360054e-17, -2.65255300e-19, -6.43201416e-17,  1.00000000e-01,
        3.00000000e-01,  1.00978449e+00,  1.00000000e+00,  4.25095801e-17,
       -6.71042126e-20, -7.62114306e-20,  1.00000000e-01, -3.00000000e-01,
        1.00978449e+00,  1.00000000e+00,  4.91108869e-17,  4.89149292e-19,
       -1.20651716e-18,  5.50000000e-01,  5.00000000e-02,  1.00978449e+00,
        1.00000000e+00,  3.58610778e-17, -5.35923108e-20, -6.42348033e-19,
        3.00000000e-01,  1.75826020e-16,  8.09784489e-01,  1.00000000e+00,
        4.01378444e-17, -7.17571046e-19,  1.83378906e-18, -9.24846673e-01,
       -5.77952844e-01,  4.32368197e-01, -5.58517578e-01, -9.47342379e-01,
        3.64148262e-01,  2.81070933e-03,  2.90346580e-03,  3.00362165e-03,
        2.81079122e-03,  2.89854556e-03,  2.99916570e-03])

holding_box_on_table = np.array([ 3.00156097e-01, -1.87852719e-01,  7.40876222e-01,  9.99787258e-01,
        1.50108045e-02,  2.66906938e-04,  1.41436560e-02,  1.00000000e-01,
        3.00000000e-01,  1.00978449e+00,  1.00000000e+00,  7.36894143e-18,
       -2.73140865e-18, -2.21178166e-17,  1.00000000e-01, -3.00000000e-01,
        1.00978449e+00,  1.00000000e+00,  9.51813498e-18,  1.29829651e-18,
        7.39372808e-17,  5.50000000e-01,  5.00000000e-02,  1.00978449e+00,
        1.00000000e+00,  2.12419599e-17, -4.51753361e-18,  3.27020961e-16,
        3.00000000e-01, -1.12285425e-15,  8.09784489e-01,  1.00000000e+00,
       -2.58442076e-17,  1.45894808e-17,  2.96287994e-16, -2.52025235e+00,
       -1.61860052e+00,  2.47046217e+00, -2.38142429e+00, -1.57160090e+00,
       -9.78319993e-01,  2.86844674e-01,  2.85494652e-01,  2.88551122e-01,
        2.86845961e-01,  2.85618346e-01,  2.90015081e-01])

holding_box_above_table = np.array([ 2.96032484e-01, -1.99640401e-01,  7.88264087e-01,  9.99964556e-01,
        1.63900359e-03,  4.27424712e-03,  7.06613982e-03,  1.00000000e-01,
        3.00000000e-01,  1.00978449e+00,  1.00000000e+00,  4.13465298e-17,
       -9.12153593e-19, -3.25543036e-17,  1.00000000e-01, -3.00000000e-01,
        1.00978449e+00,  1.00000000e+00,  4.33775075e-17,  1.27139499e-19,
        3.82391786e-17,  5.50000000e-01,  5.00000000e-02,  1.00978449e+00,
        1.00000000e+00,  4.85350930e-17, -1.37763549e-18, -6.50249375e-18,
        3.00000000e-01,  2.79863244e-16,  8.09784489e-01,  1.00000000e+00,
        4.19144129e-17, -1.36325345e-19,  1.38584434e-17,  1.36152446e+00,
       -1.42379532e+00, -2.43215206e+00, -8.77448987e-01,  1.56936045e+00,
       -2.23408927e-01,  2.87127830e-01,  2.85775262e-01,  2.88976334e-01,
        2.87138646e-01,  2.85873162e-01,  2.88864581e-01])

def test_collision(
        test_qpos: np.ndarray,
        controlled_bodies_prefix: list[str] ,
        exclude_collision_prefix_pairs: list[tuple[str, str]] = []
):
    data.qpos[:] = test_qpos
    mujoco.mj_forward(model, data)
    return is_collision_state(
        model,
        data,
        controlled_bodies_prefix=controlled_bodies_prefix,
        exclude_collision_prefix_pairs=exclude_collision_prefix_pairs,
    )

# TESTS FOR COLLISION STATE FN

assert test_collision(
    init_qpos,
    controlled_bodies_prefix=["robot_", "gripper_"],
) == False

assert test_collision(
    collide_robot_obstacle,
    controlled_bodies_prefix=["robot_", ],
) == True

assert test_collision(
    collide_robot_obstacle,
    controlled_bodies_prefix=["obstacle", ],
) == True

assert test_collision(
    collide_robot_obstacle,
    controlled_bodies_prefix=["robot_", "obstacle"],
    exclude_collision_prefix_pairs=[("obstacle", "table"), ("robot_", "obstacle")],
) == False

assert test_collision(
    holding_box_on_table,
    controlled_bodies_prefix=["robot_", "gripper", "red_box"],
) == True

assert test_collision(
    holding_box_on_table,
    controlled_bodies_prefix=["robot_", "gripper", "red_box"],
    exclude_collision_prefix_pairs=[("red_box", "table")],
) == True

assert test_collision(
    holding_box_on_table,
    controlled_bodies_prefix=["robot_", "gripper", "red_box"],
    exclude_collision_prefix_pairs=[("red_box", "gripper")],
) == True

assert test_collision(
    holding_box_on_table,
    controlled_bodies_prefix=["robot_", "gripper", "red_box"],
    exclude_collision_prefix_pairs=[("red_box", "gripper"), ("red_box", "table")],
) == False

assert test_collision(
    holding_box_above_table,
    controlled_bodies_prefix=["robot_", "gripper", "red_box"],
    exclude_collision_prefix_pairs=[("red_box", "gripper")],
) == False


# TESTS FOR SET STATE WITH BODY ATTACHMENT FN
holding_box_on_table_robot_qpos = holding_box_on_table[-12:-6]
holding_box_above_table_robot_qpos = holding_box_above_table[-12:-6]

mujoco.mj_resetData(model, data)
data.qpos[:] = holding_box_on_table
mujoco.mj_forward(model, data)

set_state(
    model,
    data,
    robot_qpos=holding_box_above_table_robot_qpos,
    robot_qpos_idx=get_entity_qpos_idx(model, "robot_"),
    attached_body_name="red_box"
)

assert is_collision_state(
    model,
    data,
    controlled_bodies_prefix=["robot_", "gripper_", "red_box"],
    exclude_collision_prefix_pairs=[("red_box", "gripper")],
    verbose=True
) == False