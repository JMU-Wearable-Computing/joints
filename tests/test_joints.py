from joints import JointCollection
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm


rng = np.random.default_rng(666)
joints = {"a": np.array([1,0,0]),
          "b": np.array([0,1,0]),
          "c": np.array([0,0,0])}
connections = [("a", "b"), ("b", "c")]

def test_constructor():
    jc = JointCollection(joints, connections)

    assert all([joint in jc.joints for joint in joints.keys()])
    assert all([jc.joint_parents[joint2] == joint1 for joint1, joint2 in connections])
    assert all([np.all(jc.pos[joint] == pos) for joint, pos in joints.items()])
    assert all([np.all(jc.lengths[joint1][joint2] == np.linalg.norm(joints[joint1] - joints[joint2]))
                for joint1, joint2 in connections])


def test_solve_for_pos():
    jc = JointCollection(joints, connections)

    vec1 = joints["a"] - joints["b"]
    vec2 = joints["c"] - joints["b"]

    axis = np.cross(vec1, vec2)
    axis = axis / norm(axis)
    theta = np.arccos(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

    length = np.linalg.norm(vec2)
    pos = jc.solve_for_pos(joint1="a",
                           joint2="b",
                           theta=theta,
                           axis=axis,
                           length=length) 

    assert np.allclose(pos, joints["c"])


def test_get_downstream_angles():
    jc = JointCollection(joints, connections)
    vec1 = joints["a"] - joints["b"]
    vec2 = joints["c"] - joints["b"]
    # rot = R.align_vectors(vec1[None, :], vec2[None, :])[0]
    expected = jc.find_arccos_angle(vec1, vec2)

    angles = jc.get_downstream_angles("a")

    assert np.allclose(angles[("a", "b", "c")][0], expected)

def get_rotation(a,b,c, theta, jc):
    vec1 = jc.pos[a] - jc.pos[b]
    vec2 = jc.pos[c] - jc.pos[b]

    axis = np.cross(vec1, vec2)
    axis = axis / norm(axis)
    q = R.from_quat([*(axis * np.sin(theta / 2)), np.cos(theta/2)])
    return q

def test_setitem1():
    jc = JointCollection(joints, connections)
    vec1 = joints["a"] - joints["b"]
    vec2 = joints["c"] - joints["b"]

    theta = np.arccos(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
    q = get_rotation("a", "b", "c", theta, jc)

    jc["a", "b", "c"] = theta
    angle = jc[("a", "b", "c")]

    assert np.allclose(jc["c"] - jc["b"], vec2)
    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

def test_setitem_downstream():
    joints = {"a": np.array([1,0,0]),
              "b": np.array([0,1,0]),
              "c": np.array([0,0,0]),
              "d": np.array([-1,-1,-1]),
              "e": np.array([-2,-1,-2]),
              "f": np.array([-1,0,-1])}
    connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

    jc = JointCollection(joints, connections)
    og_angles = jc.get_all_angles()
    del og_angles[("a", "b", "c")]

    vec1 = joints["a"] - joints["b"]
    vec2 = joints["c"] - joints["b"]

    theta = np.arccos(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
    q = get_rotation("a", "b", "c", theta, jc)

    jc["a", "b", "c"] = theta

    angle = jc[("a", "b", "c")]

    assert np.allclose(jc["c"] - jc["b"], vec2)
    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

    updated_angles = jc.get_all_angles()
    del updated_angles[("a", "b", "c")]
    assert all([np.allclose(v.as_quat(), updated_angles[k].as_quat()) for k, v in og_angles.items()])

def test_setitem_downstream_90d():
    joints = {"a": np.array([1,0,0]),
              "b": np.array([0,1,0]),
              "c": np.array([0,0,0]),
              "d": np.array([-1,-1,-1]),
              "e": np.array([-2,-1,-2]),
              "f": np.array([-1,0,-1])}
    connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

    jc = JointCollection(joints, connections)
    og_angles = jc.get_all_angles()
    del og_angles[("a", "b", "c")]

    theta = (np.pi * 90) / 180
    q = get_rotation("a", "b", "c", theta, jc)
    jc["a", "b", "c"] = theta

    angle = jc[("a", "b", "c")]

    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

    updated_angles = jc.get_all_angles()
    del updated_angles[("a", "b", "c")]
    assert all([np.allclose(np.arccos(v.as_quat()[-1]), np.arccos(updated_angles[k].as_quat()[-1])) for k, v in og_angles.items()])

def test_setitem_downstream_181d():
    joints = {"a": np.array([1,0,0]),
              "b": np.array([0,1,0]),
              "c": np.array([0,0,0]),
              "d": np.array([-1,-1,-1]),
              "e": np.array([-2,-1,-2]),
              "f": np.array([-1,0,-1])}
    connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

    jc = JointCollection(joints, connections)
    og_angles = jc.get_all_angles()
    del og_angles[("a", "b", "c")]

    theta = (np.pi * 181) / 180
    q = get_rotation("a", "b", "c", theta, jc)
    jc["a", "b", "c"] = theta

    angle = jc[("a", "b", "c")]

    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

    updated_angles = jc.get_all_angles()
    del updated_angles[("a", "b", "c")]
    assert all([np.allclose(np.arccos(v.as_quat()[-1]), np.arccos(updated_angles[k].as_quat()[-1])) for k, v in og_angles.items()])

def test_setitem_downstream_neg90d():
    joints = {"a": np.array([1,0,0]),
              "b": np.array([0,1,0]),
              "c": np.array([0,0,0]),
              "d": np.array([-1,-1,-1]),
              "e": np.array([-2,-1,-2]),
              "f": np.array([-1,0,-1])}
    connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

    jc = JointCollection(joints, connections)
    og_angles = jc.get_all_angles()
    del og_angles[("a", "b", "c")]

    theta = (np.pi * -90) / 180
    q = get_rotation("a", "b", "c", theta, jc)
    jc["a", "b", "c"] = theta

    angle = jc[("a", "b", "c")]

    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

    updated_angles = jc.get_all_angles()
    del updated_angles[("a", "b", "c")]
    assert all([np.allclose(np.arccos(v.as_quat()[-1]), np.arccos(updated_angles[k].as_quat()[-1])) for k, v in og_angles.items()])

def test_setitem_downstream_180d():
    joints = {"a": np.array([1,0,0]),
              "b": np.array([0,1,0]),
              "c": np.array([0,0,0]),
              "d": np.array([-1,-1,-1]),
              "e": np.array([-2,-1,-2]),
              "f": np.array([-1,0,-1])}
    connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

    jc = JointCollection(joints, connections)
    og_angles = jc.get_all_angles()
    del og_angles[("a", "b", "c")]

    theta = (np.pi * 180) / 180
    q = get_rotation("a", "b", "c", theta, jc)
    jc["a", "b", "c"] = theta

    angle = jc[("a", "b", "c")]

    assert (np.allclose(angle.as_quat(), q.as_quat())
            or np.allclose(angle.as_quat(), -1 * q.as_quat()))

    updated_angles = jc.get_all_angles()
    del updated_angles[("a", "b", "c")]
    assert all([np.allclose(np.arccos(v.as_quat()[-1]), np.arccos(updated_angles[k].as_quat()[-1])) for k, v in og_angles.items()])

def test_setitem_downstream_random():
    for i in range(1000):
        joints = {"a": rng.random(3),
                "b": rng.random(3),
                "c": rng.random(3),
                "d": rng.random(3),
                "e": rng.random(3),
                "f": rng.random(3)}
        connections = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e"), ("e", "f")]

        jc = JointCollection(joints, connections)
        og_angles = jc.get_all_angles()
        del og_angles[("a", "b", "c")]

        theta = 2 * np.pi * rng.random() 
        q = get_rotation("a", "b", "c", theta, jc)
        jc["a", "b", "c"] = theta

        angle = jc[("a", "b", "c")]

        assert (np.allclose(angle.as_quat(), q.as_quat())
                or np.allclose(angle.as_quat(), -1 * q.as_quat()))

        updated_angles = jc.get_all_angles()
        del updated_angles[("a", "b", "c")]
        assert all([np.allclose(np.arccos(v.as_quat()[-1]), np.arccos(updated_angles[k].as_quat()[-1])) for k, v in og_angles.items()])