"""Physical dumbbell welded to Arm26's combined forearm/hand body.

Units: kg, m, rad. Geometry and inertia describe a symmetric shaft and two
plates, with a fixed illustrative size and selectable total mass. This is
a rigid-grip approximation, not a finger/contact model.
"""
import argparse
import math
from pathlib import Path

DEFAULT_GRIP = (0.0, -0.33, 0.075)


def add_dumbbell(osim, model, mass, grip=DEFAULT_GRIP):
    if not math.isfinite(mass) or mass < 0:
        raise ValueError('Load mass must be finite and nonnegative.')
    if len(grip) != 3 or not all(math.isfinite(v) for v in grip):
        raise ValueError('Grip must contain three finite body-local coordinates.')
    if model.getBodySet().contains('hand_dumbbell'):
        raise ValueError('Input already has a dumbbell. Use the original arm26_exosuit.osim.')
    if mass == 0:
        return None  # True no-load baseline: no dummy mass or inertia.
    parent = model.updBodySet().get('r_ulna_radius_hand')
    # Shaft: 10% of mass; plates: 45% each; all cylinder axes along z.
    shaft_r, shaft_l = 0.012, 0.155
    plate_r, plate_l, plate_z = 0.05, 0.025, 0.09
    ms, mp = mass * 0.10, mass * 0.45
    iz = ms * shaft_r**2 / 2 + 2 * mp * plate_r**2 / 2
    ix = ms * (3 * shaft_r**2 + shaft_l**2) / 12
    ix += 2 * (mp * (3 * plate_r**2 + plate_l**2) / 12 + mp * plate_z**2)
    body = osim.Body('hand_dumbbell', mass, osim.Vec3(0),
                     osim.Inertia(ix, ix, iz, 0, 0, 0))
    joint = osim.WeldJoint('dumbbell_grip', parent, osim.Vec3(*grip),
                          osim.Vec3(0), body, osim.Vec3(0), osim.Vec3(0))
    # OpenSim cylinders have their axis along local y. Rotate each visual
    # frame by pi/2 about x so its axis aligns with the dumbbell body's z.
    for name, z, radius, length, color in (
        ('handle', 0, shaft_r, shaft_l, (0.65, 0.67, 0.70)),
        ('plate_left', -plate_z, plate_r, plate_l, (0.12, 0.13, 0.16)),
        ('plate_right', plate_z, plate_r, plate_l, (0.12, 0.13, 0.16))):
        frame = osim.PhysicalOffsetFrame()
        frame.setName(name)
        frame.setParentFrame(body)
        frame.set_translation(osim.Vec3(0, 0, z))
        frame.set_orientation(osim.Vec3(math.pi / 2, 0, 0))
        shape = osim.Cylinder(radius, length / 2)
        shape.setColor(osim.Vec3(*color))
        frame.attachGeometry(shape)
        body.addComponent(frame)
    model.addBody(body)
    model.addJoint(joint)
    model.finalizeConnections()
    return body


def save_loaded_model(osim, source, output, mass, grip=DEFAULT_GRIP):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source:
        raise ValueError('Output must differ from the original model.')
    model = osim.Model(str(source))
    add_dumbbell(osim, model, mass, grip)
    model.setName('arm26_dumbbell_' + str(mass).replace('.', 'p') + 'kg')
    model.initSystem()  # Validate joints, connections and inertial properties.
    if not model.printToXML(str(output)):
        raise RuntimeError('Could not save loaded model.')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--input', type=Path, default=root / 'arm26_exosuit.osim')
    parser.add_argument('--mass', type=float, default=0.5)
    parser.add_argument('--grip', type=float, nargs=3, default=DEFAULT_GRIP,
                        metavar=('X', 'Y', 'Z'), help='COM in forearm/hand body frame, metres.')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    import opensim as osim
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(root / 'Geometry'))
    dest = args.output or root / ('arm26_dumbbell_' + str(args.mass).replace('.', 'p') + 'kg.osim')
    save_loaded_model(osim, args.input, dest, args.mass, args.grip)
    print('Saved:', dest)
    print('Physical mass:', args.mass, 'kg; grip/COM:', args.grip, 'm in forearm frame.')
